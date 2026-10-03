"""Map-style adapters for canonical fused-token shards and packed CLIFATRON rows.

`representation="decile"` is a legacy name for the canonical shards written by
`src/data/tokenize.py`, whatever `value_binning.scheme` produced them (clinical
segments by default).

`representation="gem"` (U8, KTD10) reads gem_events.parquet: one row per window of a
full-hospitalization stream. Windows are regrouped per stay and validated (contiguous
source spans, consistent continuation flags); next-event targets are built ONCE on the
whole stay by a `TargetBuilder(mode="gem")` and sliced per window — the packed-segment
contract — so a window's last eligible event still predicts the first eligible event
of the next window. GEM samples carry no TTE labels.

With `TargetBuilder(mode="gem_tte")` (U3, KTD1) the same windows also carry in-stream
time-to-event labels: anchors are sampled and labelled on the whole stay, so a label
uses events from later windows, and each window keeps the anchors that fall inside it.

ANCHORS. Every segment that can be supervised lists its anchors under ``"anchors"``:
``{"offset": position in the packed row, "cr": competing-risk label or None,
"queries": [threshold query, ...]}`` with label times in minutes since the anchor
(`TargetBuilder._sample_anchors` / `outcome_anchor`). The 24 h representations have one
anchor per episode; ``gem_tte`` windows have several; ``gem`` windows have no such key.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import logging

from src.data.segments import RETOKENIZE, TOKENIZER_VERSION
from src.data.targets import GEM_MODES, TargetBuilder, TargetContractError
from src.data.tokenize_continuous import normalize_value

try:
    from external.clifatron.AR.qwen2.data.packed_dataset import PACKED_SCHEMA_VERSION  # type: ignore[import-untyped]
except ImportError:
    PACKED_SCHEMA_VERSION = "2.0.0"

logger = logging.getLogger(__name__)


class ModelDataset(Dataset):
    """Deterministic map-style dataset retaining packed document boundaries."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]] | str | Path,
        *,
        representation: str,
        target_builder: TargetBuilder,
        expected_hashes: Mapping[str, str],
        episode_targets: Mapping[str, Mapping[str, Any]] | None = None,
        epoch: int = 0,
        value_channel: bool = False,
    ) -> None:
        """`value_channel` (continuous-fused arm, U6): each canonical or full-hospitalization
        sample also carries `input_value` / `input_value_mask`, the CURRENT event's value
        normalized with the target builder's frozen per-token stats
        (`tokenize_continuous.normalize_value`); missing, categorical (NaN) and
        implausible values are masked to 0."""
        if representation not in {"decile", "clifatron_packed", "gem"}:
            raise ValueError("representation must be 'decile', 'clifatron_packed' or 'gem'")
        if value_channel and representation == "clifatron_packed":
            raise ValueError("the input value channel is defined for canonical and "
                             "full-hospitalization shards only")
        if (representation == "gem") != (getattr(target_builder, "mode", "icu_24h") in GEM_MODES):
            raise TargetContractError(
                "the gem representation requires TargetBuilder(mode='gem') or "
                "mode='gem_tte', and only those"
            )
        if isinstance(records, (str, Path)):
            records = pl.read_parquet(records).to_dicts()
        self.records = [dict(record) for record in records]
        self.representation = representation
        self.target_builder = target_builder
        self.expected_hashes = dict(expected_hashes)
        self.episode_targets = dict(episode_targets or {})
        self.epoch = int(epoch)
        self.value_channel = bool(value_channel)
        in_stream = getattr(target_builder, "in_stream", None)
        for record in self.records:
            self._validate_hashes(record)
            if in_stream is not None:
                # The threshold grid's bins and token map are this vocabulary's (KTD3).
                in_stream.grid.check_binding(record["artifact_hashes"], what="GEM shard row")
            is_gem = record.get("trajectory") == "hospitalization"
            if representation != "clifatron_packed" and is_gem != (representation == "gem"):
                raise TargetContractError(
                    "GEM (hospitalization) records load only through representation='gem'"
                )
        self._gem_built: dict[str, dict[str, Any]] = {}
        if representation == "gem":
            self.records, self._gem_streams = _gem_streams(self.records)
            # Only a stay split into several windows reads its targets more than once
            # per epoch, so only those are cached (a single-window stay rebuilds them).
            windows: dict[str, int] = {}
            for record in self.records:
                key = _episode_key(record)
                windows[key] = windows.get(key, 0) + 1
            self._gem_multi_window = {key for key, n in windows.items() if n > 1}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.representation == "gem":
            # `_gem_sample` never mutates the record and copies the per-token lists it
            # returns except the soft fields, which every consumer (collate) only reads.
            return self._gem_sample(self.records[index])
        record = deepcopy(self.records[index])
        if self.representation == "decile":
            return self._decile_sample(record)
        return self._packed_sample(record)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._gem_built.clear()

    def _validate_hashes(self, record: Mapping[str, Any]) -> None:
        hashes = record.get("artifact_hashes")
        if not isinstance(hashes, Mapping):
            raise TargetContractError("sample is missing artifact_hashes")
        if self.representation in ("decile", "gem") and (
            not hashes.get("numeric_edges")
            or str(hashes.get("tokenizer_version")) != str(TOKENIZER_VERSION)
        ):
            # KTD7: a canonical shard row records the tokenizer version and segments
            # hash it was encoded with. Without them it was built by the previous
            # tokenizer, whose bins no longer mean what the vocabulary says.
            raise TargetContractError(
                f"shard row is not bound to tokenizer-v{TOKENIZER_VERSION} segments "
                f"(no numeric_edges/tokenizer_version in artifact_hashes); {RETOKENIZE}"
            )
        for name, expected in self.expected_hashes.items():
            if hashes.get(name) != expected:
                raise TargetContractError(f"artifact hash mismatch: {name}")

    def _decile_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        built = self.target_builder.build(record, epoch=self.epoch)
        length = len(record["token"])
        sample = {
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "input_ids": record["token"],
            "attention_mask": [1] * length,
            "pos_min": record["pos_min"],
            "soft_token": record.get("soft_token"),
            "soft_weight": record.get("soft_weight"),
            "ntp_target": built["ntp_target"],
            "ntp_mask": built["ntp_mask"],
            "ntp_delta_min": built["ntp_delta_min"],
            "value_target": built["value_target"],
            "value_mask": built["value_mask"],
            "segments": [
                {
                    "episode_key": record["episode_key"],
                    "source_start": 0,
                    "source_end": length,
                    "packed_start": 0,
                    "packed_end": length,
                    "continuation_index": 0,
                    "continues_from_previous": False,
                    "continues_to_next": False,
                    "anchor_offset": built["anchor_idx"],
                    "outcome_labels": built["outcome_labels"],
                    "threshold_query": built["threshold_query"],
                    "anchors": [{
                        "offset": built["anchor_idx"],
                        **self.target_builder.outcome_anchor(record, built, epoch=self.epoch),
                    }],
                }
            ],
        }
        if self.value_channel:
            self._add_value_channel(sample, record)
        return sample

    def _add_value_channel(self, sample: dict[str, Any], record: Mapping[str, Any]) -> None:
        stats = self.target_builder.value_stats
        values = record.get("value") or [None] * len(record["token"])
        channel = [normalize_value(value, token, stats,
                                   max_abs_z=self.target_builder.max_abs_value_z)
                   for token, value in zip(record["token"], values)]
        sample["input_value"] = [value for value, _ in channel]
        sample["input_value_mask"] = [mask for _, mask in channel]

    def _gem_sample(self, record: Mapping[str, Any]) -> dict[str, Any]:
        key = _episode_key(record)
        built = self._gem_built.get(key)
        if built is None:
            built = self.target_builder.build(self._gem_streams[key], epoch=self.epoch)
            if key in self._gem_multi_window:
                self._gem_built[key] = built
        start, end = int(record["source_start"]), int(record["source_end"])
        length = end - start
        anchor = built["anchor_idx"]
        contains_anchor = anchor is not None and start <= anchor < end
        sample = {
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "input_ids": list(record["token"]),
            "attention_mask": [1] * length,
            "pos_min": list(record["pos_min"]),
            "soft_token": record.get("soft_token"),
            "soft_weight": record.get("soft_weight"),
            "segments": [{
                "episode_key": key,
                "source_start": start,
                "source_end": end,
                "packed_start": 0,
                "packed_end": length,
                "continuation_index": int(record["continuation_index"]),
                "continues_from_previous": bool(record["continues_from_previous"]),
                "continues_to_next": bool(record["continues_to_next"]),
                # Representation evaluation only: GEM has no TTE labels or queries.
                "anchor_offset": anchor - start if contains_anchor else None,
                "outcome_labels": [],
                "threshold_query": None,
            }],
        }
        for field in ("ntp_target", "ntp_mask", "ntp_delta_min", "value_target", "value_mask"):
            sample[field] = built[field][start:end]
        if "anchors" in built:
            # In-stream labels were computed on the whole stay; this window keeps the
            # anchors sampled inside it.
            sample["segments"][0]["anchors"] = [
                {"offset": item["anchor_idx"] - start, "cr": item["cr"],
                 "queries": item["queries"]}
                for item in built["anchors"] if start <= item["anchor_idx"] < end
            ]
        if self.value_channel:
            self._add_value_channel(sample, record)
        return sample

    def _packed_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("packed_schema_version") != PACKED_SCHEMA_VERSION:
            raise TargetContractError("unsupported packed schema version")
        input_ids = list(record["input_ids"])
        attention_mask = list(record["attention_mask"])
        if len(input_ids) != len(attention_mask):
            raise TargetContractError("packed input and attention mask lengths differ")
        if "pos_min" in record and record["pos_min"] is not None:
            pos_min = list(record["pos_min"])
        else:
            pos_min = list(range(len(input_ids)))
            logger.warning(
                "packed record is missing pos_min column; "
                "falling back to range(%d). Admission-relative position is degraded.",
                len(input_ids),
            )
        output = {
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pos_min": pos_min,
            "ntp_target": [0] * len(input_ids),
            "ntp_mask": [False] * len(input_ids),
            "ntp_delta_min": [0] * len(input_ids),
            "value_target": [0.0] * len(input_ids),
            "value_mask": [False] * len(input_ids),
            "segments": [],
        }
        occupied: set[int] = set()
        for segment in record.get("segments", []):
            segment = dict(segment)
            key = segment.get("episode_key")
            if not isinstance(key, str) or not key:
                raise TargetContractError("packed segment is missing an opaque episode key")
            required = {
                "source_start",
                "source_end",
                "packed_start",
                "packed_end",
                "continuation_index",
                "continues_from_previous",
                "continues_to_next",
            }
            if not required.issubset(segment):
                raise TargetContractError("packed segment is missing source/continuation metadata")
            source_start, source_end = int(segment["source_start"]), int(segment["source_end"])
            packed_start, packed_end = int(segment["packed_start"]), int(segment["packed_end"])
            if source_end <= source_start or packed_end <= packed_start:
                raise TargetContractError("packed segment spans must be non-empty")
            if source_end - source_start != packed_end - packed_start or packed_end > len(input_ids):
                raise TargetContractError("packed source and destination spans are inconsistent")
            positions = set(range(packed_start, packed_end))
            if occupied & positions:
                raise TargetContractError("packed segments overlap")
            occupied |= positions
            if key not in self.episode_targets:
                raise TargetContractError(f"packed segment has no target join: {key}")
            built = self.target_builder.build(self.episode_targets[key], epoch=self.epoch)
            for field in ("ntp_target", "ntp_mask", "ntp_delta_min", "value_target", "value_mask"):
                output[field][packed_start:packed_end] = built[field][source_start:source_end]
            anchor = built["anchor_idx"]
            contains_anchor = source_start <= anchor < source_end
            segment["anchor_offset"] = packed_start + anchor - source_start if contains_anchor else None
            segment["outcome_labels"] = built["outcome_labels"] if contains_anchor else []
            segment["threshold_query"] = built["threshold_query"] if contains_anchor else None
            segment["anchors"] = [{
                "offset": segment["anchor_offset"],
                **self.target_builder.outcome_anchor(self.episode_targets[key], built,
                                                     epoch=self.epoch),
            }] if contains_anchor else []
            output["segments"].append(segment)
        if not output["segments"]:
            raise TargetContractError("packed row contains no document segments")
        return output


def _episode_key(record: Mapping[str, Any]) -> str:
    key = record.get("episode_key") or record.get("hosp_id")
    if not isinstance(key, str) or not key:
        raise TargetContractError("GEM window is missing an opaque episode key")
    return key


_GEM_WINDOW_FIELDS = ("token", "pos_min", "value", "target_eligible")


def _gem_streams(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Group GEM windows per stay, validate the continuation contract, and rebuild each
    stay's full stream (the TargetBuilder input). Returns the windows ordered by
    (episode key, continuation index) and ``{key: stream}``."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_key.setdefault(_episode_key(record), []).append(record)
    ordered: list[dict[str, Any]] = []
    streams: dict[str, dict[str, Any]] = {}
    for key in sorted(by_key):
        windows = sorted(by_key[key], key=lambda r: int(r["continuation_index"]))
        n = len(windows)
        stream: dict[str, list] = {field: [] for field in _GEM_WINDOW_FIELDS}
        for index, window in enumerate(windows):
            length = len(window["token"])
            if (int(window["continuation_index"]) != index
                    or int(window.get("n_windows", n)) != n
                    or int(window["source_start"]) != len(stream["token"])
                    or int(window["source_end"]) - int(window["source_start"]) != length
                    or length == 0
                    or bool(window["continues_from_previous"]) != (index > 0)
                    or bool(window["continues_to_next"]) != (index < n - 1)
                    or any(len(window[f]) != length for f in _GEM_WINDOW_FIELDS)):
                raise TargetContractError(
                    f"GEM window {index} of a stay breaks the continuation contract "
                    "(missing, duplicated or misaligned window)"
                )
            for field in ("anchor_idx", "anchor_min", "partition"):
                if window.get(field) != windows[0].get(field):
                    raise TargetContractError(f"GEM windows of a stay disagree on {field}")
            for field in _GEM_WINDOW_FIELDS:
                stream[field].extend(window[field])
        anchor = windows[0].get("anchor_idx")
        streams[key] = {
            "episode_key": key,
            **stream,
            "anchor_idx": None if anchor is None else int(anchor),
            "anchor_min": windows[0].get("anchor_min"),
            "outcomes": [],
            "windows": [(int(w["source_start"]), int(w["source_end"])) for w in windows],
        }
        ordered.extend(windows)
    return ordered, streams


class LengthGroupedSampler(Sampler):
    """Shuffle, then group similar-length documents into each batch (HF
    group_by_length). Uniform shuffling over heavy-tailed CLIF lengths (mean 271,
    p95 471, max 6,413) pads every batch to its longest member — ~10x wasted
    compute and unbounded per-batch attention shape diversity, which drove the
    MPS caching-allocator watermark to 84 GiB and OOM'd an overnight run
    (gem-overnight-log 2026-09-26). Grouping bounds both: padded shapes recycle
    across batches and long stays only ever share a batch with other long stays.

    Single-process sampler (dev boxes). DDP ranks use
    `DistributedTokenBudgetBatchSampler` (U5).
    """

    def __init__(self, lengths, batch_size, *, seed=42, mega_batch_mult=50):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mega_batch_mult = int(mega_batch_mult)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=g).tolist()
        mega = max(self.batch_size * self.mega_batch_mult, self.batch_size)
        batches = []
        for start in range(0, len(order), mega):
            chunk = sorted(order[start:start + mega], key=lambda i: self.lengths[i])
            for k in range(0, len(chunk), self.batch_size):
                batches.append(chunk[k:k + self.batch_size])
        batch_order = torch.randperm(len(batches), generator=g).tolist()
        for b in batch_order:
            yield from batches[b]


class TokenBudgetBatchSampler(Sampler):
    """Batch sampler (yields index LISTS; use as DataLoader(batch_sampler=...)):
    length-groups like LengthGroupedSampler, then packs each batch to a token budget
    — B x max_len_in_batch <= max_batch_tokens. A 6,413-token stay batches ALONE;
    short stays pack tight. MPS math-path attention materializes T^2 buffers per
    layer for backward (~40 GiB for one 4x6413 batch), and uniform per_gpu batching
    over heavy-tailed CLIF lengths OOM'd three overnight runs (86 GiB watermark,
    updates ~205/~410/~2300 — gem-overnight-log). Token budgets bound the live
    transients AND cut padding waste to ~zero. opt-in via runtime.token_budget on a
    single process; under DDP `DistributedTokenBudgetBatchSampler` deals these batches
    to the ranks.

    A single sequence longer than the budget still forms its own batch (allowed
    oversize; nothing is dropped).
    """

    def __init__(self, lengths, *, max_batch_tokens, max_batch_size=None,
                 seed=42, mega_batch_mult=50, shuffle=True):
        self.lengths = [int(length) for length in lengths]
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_batch_size = int(max_batch_size) if max_batch_size else None
        self.seed = int(seed)
        self.mega_batch_mult = int(mega_batch_mult)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.lengths)

    def _batches(self) -> list[list[int]]:
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self.lengths), generator=g).tolist()
        else:
            order = list(range(len(self.lengths)))  # deterministic (validation)
        mega = max((self.max_batch_size or 1) * self.mega_batch_mult, 1)
        batches: list[list[int]] = []
        for start in range(0, len(order), mega):
            chunk = sorted(order[start:start + mega], key=lambda i: self.lengths[i])
            batch: list[int] = []
            longest = 0
            for idx in chunk:
                cand = max(longest, self.lengths[idx])
                too_big = (
                    batch
                    and cand * (len(batch) + 1) > self.max_batch_tokens
                )
                too_many = (
                    self.max_batch_size is not None
                    and batch
                    and len(batch) + 1 > self.max_batch_size
                )
                if too_big or too_many:
                    batches.append(batch)
                    batch, longest = [], 0
                batch.append(idx)
                longest = max(longest, self.lengths[idx])
            if batch:
                batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            batch_order = torch.randperm(len(batches), generator=g).tolist()
        else:
            batch_order = list(range(len(batches)))
        for b in batch_order:
            yield batches[b]


class DistributedTokenBudgetBatchSampler(Sampler):
    """Rank-aware batch sampler (DataLoader(batch_sampler=...)) for DDP training (U5).

    Batches are formed GLOBALLY, exactly as `TokenBudgetBatchSampler` forms them (length-
    grouped, B x max_len <= `max_batch_tokens`, at most `max_batch_size` rows, shuffled per
    `seed + epoch`), so every rank computes the same list without communicating. When the
    count is not a multiple of `num_replicas`, the first batches of the shuffled list are
    repeated — at most `num_replicas - 1` of them — and rank r takes batches r, r+W, ...
    Every rank therefore yields the same number of batches per pass, which the engine
    needs: it finds the epoch's last (synchronized) microbatch from `len(dl)` (KTD2).

    `set_epoch` works like DistributedSampler's: call it before each pass. Batch packing
    depends on the shuffle, so the count can differ by a few batches between epochs;
    `len()` is the current epoch's. `max_batch_tokens` None/0 caps rows only.
    """

    def __init__(self, lengths, *, max_batch_tokens=None, max_batch_size=None,
                 num_replicas: int | None = None, rank: int | None = None, seed: int = 42,
                 mega_batch_mult: int = 50, shuffle: bool = True):
        import torch.distributed as dist

        initialized = dist.is_available() and dist.is_initialized()
        self.num_replicas = int(num_replicas if num_replicas is not None
                                else dist.get_world_size() if initialized else 1)
        self.rank = int(rank if rank is not None else dist.get_rank() if initialized else 0)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank {self.rank} is outside [0, {self.num_replicas})")
        if not max_batch_tokens and not max_batch_size:
            raise ValueError("pass max_batch_tokens or max_batch_size")
        if not len(lengths):
            raise ValueError("cannot sample batches from an empty dataset")
        self._packer = TokenBudgetBatchSampler(
            lengths, max_batch_tokens=int(max_batch_tokens or 2 ** 62),
            max_batch_size=max_batch_size, seed=seed, mega_batch_mult=mega_batch_mult,
            shuffle=shuffle)
        self.epoch = 0
        self._cache: tuple[int, list[list[int]]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def global_batches(self) -> list[list[int]]:
        """This epoch's batches over the whole dataset, in order, before evening out."""
        if self._cache is None or self._cache[0] != self.epoch:
            self._packer.set_epoch(self.epoch)
            self._cache = (self.epoch, list(iter(self._packer)))
        return self._cache[1]

    @property
    def padding_batches(self) -> int:
        """How many batches this epoch repeats to even the per-rank count."""
        return -len(self.global_batches()) % self.num_replicas

    def _rank_batches(self) -> list[list[int]]:
        batches = self.global_batches()
        pad = self.padding_batches
        if pad:
            batches = batches + (batches * math.ceil(pad / len(batches)))[:pad]
        return batches[self.rank::self.num_replicas]

    def __len__(self) -> int:
        return len(self._rank_batches())

    def __iter__(self):
        yield from self._rank_batches()


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    collate_fn,
    sampler: Sampler | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a loader without passing mutually exclusive sampler options."""
    if sampler is not None and shuffle:
        raise ValueError("sampler and shuffle are mutually exclusive")
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    return DataLoader(**kwargs)
