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

STORAGE (U5). The gem windows are held columnar (`GemCorpus`: flat numpy arrays with
per-window and per-stay offsets, about 57 bytes per event against ~450-585 for Python
rows), optionally memory-mapped from a cache beside the shard that every DDP rank and
loader worker on a node shares. A stay's stream is assembled from the columns when its
targets are built, so whole-stay labelling (KTD1) is unchanged; `records` and
`_gem_streams` are read-only views over the columns.

With `TargetBuilder(mode="gem_tte")` (U3, KTD1) the same windows also carry in-stream
time-to-event labels: anchors are sampled and labelled on the whole stay, so a label
uses events from later windows, and each window keeps the anchors that fall inside it.

CONTINUATION HEADER (product authority, 2026-10-03). A stay longer than one context
window is cut into contiguous windows, so only the FIRST window starts with the stay's
header (`<bos>`, `ADMISSION//<type>` and the static admission tokens: age decile, sex,
race, ethnicity, admission type). With `continuation_header` (the header's token ids,
`header_token_ids(vocab)`) every later window of the stay is given the stay's header
again at its start: header positions keep their own minutes (admission, 0) and are never
targets (masked), so labels, anchors' labels and next-event targets are unchanged; anchor
offsets shift by the header length. The header counts against `max_tokens`: a window that
would exceed it is refused - the shard must be cut with room for it
(`gem_window_bounds_with_header`; the tokenizer change is to call it in place of
`gem_window_bounds`).

ANCHORS. Every segment that can be supervised lists its anchors under ``"anchors"``:
``{"offset": position in the packed row, "cr": competing-risk label or None,
"queries": [threshold query, ...]}`` with label times in minutes since the anchor
(`TargetBuilder._sample_anchors` / `outcome_anchor`). The 24 h representations have one
anchor per episode; ``gem_tte`` windows have several; ``gem`` windows have no such key.
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
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


def header_token_ids(vocab: Mapping[str, int]) -> frozenset[int]:
    """Token ids that make up a stay's header: `<bos>`, every `ADMISSION//*` and every
    static admission token (`tokenize.STATIC_TOKENS` concepts, `concept` or `concept=x`)."""
    from src.data.tokenize import STATIC_TOKENS

    static = tuple(STATIC_TOKENS)
    ids = set()
    for name, token in vocab.items():
        if token is None:
            continue
        concept = name.split("=", 1)[0]
        if name == "<bos>" or name.startswith("ADMISSION//") or concept in static:
            ids.add(int(token))
    return frozenset(ids)


def stay_header_length(tokens: Sequence[int], header: frozenset[int]) -> int:
    """Length of the leading run of header tokens of a stay stream."""
    n = 0
    for token in tokens:
        if int(token) not in header:
            break
        n += 1
    return n


def gem_window_bounds_with_header(n: int, max_tokens: int, header_len: int) -> list[tuple[int, int]]:
    """Contiguous `[start, end)` windows of a stay of `n` tokens whose header (the first
    `header_len` tokens) is re-inserted before every continuation window: the first window
    holds `max_tokens` tokens, every later one `max_tokens - header_len`, so each sample
    is at most `max_tokens` long. The final `<eos>` is never left alone (as in
    `tokenize.gem_window_bounds`)."""
    if max_tokens - header_len < 2:
        raise ValueError("max_tokens leaves no room after the header")
    bounds = [0, min(n, max_tokens)]
    while bounds[-1] < n:
        bounds.append(min(n, bounds[-1] + max_tokens - header_len))
    if len(bounds) > 2 and bounds[-1] - bounds[-2] == 1:
        bounds[-2] -= 1
    return list(zip(bounds[:-1], bounds[1:]))


class ModelDataset(Dataset):
    """Deterministic map-style dataset retaining packed document boundaries."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]] | str | Path | GemCorpus,
        *,
        representation: str,
        target_builder: TargetBuilder,
        expected_hashes: Mapping[str, str],
        episode_targets: Mapping[str, Mapping[str, Any]] | None = None,
        epoch: int = 0,
        value_channel: bool = False,
        continuation_header: frozenset[int] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """`value_channel` (continuous-fused arm, U6): each canonical or full-hospitalization
        sample also carries `input_value` / `input_value_mask`, the CURRENT event's value
        normalized with the target builder's frozen per-token stats
        (`tokenize_continuous.normalize_value`); missing, categorical (NaN) and
        implausible values are masked to 0.

        On the gem representation `records` may also be a `GemCorpus` or a shard path,
        read columnar without building Python rows; window rows are converted to one."""
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
        self.representation = representation
        self.target_builder = target_builder
        self.expected_hashes = dict(expected_hashes)
        self.episode_targets = dict(episode_targets or {})
        self.epoch = int(epoch)
        self.value_channel = bool(value_channel)
        if continuation_header is not None and representation != "gem":
            raise ValueError("continuation_header applies to the gem representation only")
        self.continuation_header = None if continuation_header is None else frozenset(continuation_header)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self._gem_built: dict[str, dict[str, Any]] = {}
        if isinstance(records, GemCorpus) and representation != "gem":
            raise TargetContractError(_GEM_ONLY)
        if representation == "gem" and isinstance(records, (str, Path, GemCorpus)):
            # A shard path or a GemCorpus: columnar from the start, no Python rows.
            corpus = records if isinstance(records, GemCorpus) else GemCorpus.from_parquet(records)
            for hashes in corpus.hashes():
                self._check_row({"artifact_hashes": hashes, "trajectory": "hospitalization"})
            self._set_corpus(corpus)
            return
        if isinstance(records, (str, Path)):
            records = pl.read_parquet(records).to_dicts()
        self.records = [dict(record) for record in records]
        for record in self.records:
            self._check_row(record)
        if representation == "gem":
            self._set_corpus(GemCorpus.from_records(self.records))

    def _check_row(self, record: Mapping[str, Any]) -> None:
        self._validate_hashes(record)
        in_stream = getattr(self.target_builder, "in_stream", None)
        if in_stream is not None:
            # The threshold grid's bins and token map are this vocabulary's (KTD3).
            in_stream.grid.check_binding(record["artifact_hashes"], what="GEM shard row")
        is_gem = record.get("trajectory") == "hospitalization"
        if self.representation != "clifatron_packed" and is_gem != (self.representation == "gem"):
            raise TargetContractError(_GEM_ONLY)

    def _set_corpus(self, corpus: GemCorpus) -> None:
        """The gem representation keeps ONLY the columnar corpus; `records` and
        `_gem_streams` are views that build a window's or a stay's lists on access."""
        self.corpus = corpus
        self.records = _GemRecords(corpus)
        self._gem_streams = _GemStreams(corpus)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.representation == "gem":
            return self._gem_sample(index)
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

    def _gem_sample(self, index: int) -> dict[str, Any]:
        part, row = self.corpus.window(index)
        stay = int(part.win_stay[row])
        key = part.keys[stay]
        first, last = part.stay_windows(stay)
        built = self._gem_built.get(key)
        if built is None:
            built = self.target_builder.build(part.stream(stay), epoch=self.epoch)
            if last - first > 1:
                # Only a stay split into several windows reads its targets more than
                # once per epoch, so only those are cached — as arrays, since the cache
                # fills with every long stay over an epoch.
                built = {**built, **{field: np.asarray(built[field], dtype=dtype)
                                     for field, dtype in _TARGET_DTYPES.items()}}
                self._gem_built[key] = built
        lo, hi = int(part.win_offset[row]), int(part.win_offset[row + 1])
        base = int(part.win_offset[first])
        start, end = lo - base, hi - base
        length = end - start
        anchor = built["anchor_idx"]
        contains_anchor = anchor is not None and start <= anchor < end
        sample = {
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "input_ids": part.token[lo:hi].tolist(),
            "attention_mask": [1] * length,
            "pos_min": part.pos_min[lo:hi].tolist(),
            "soft_token": part.soft_list("soft_token", lo, hi),
            "soft_weight": part.soft_list("soft_weight", lo, hi),
            "segments": [{
                "episode_key": key,
                "source_start": start,
                "source_end": end,
                "packed_start": 0,
                "packed_end": length,
                "continuation_index": row - first,
                "continues_from_previous": row > first,
                "continues_to_next": row < last - 1,
                # Representation evaluation only: GEM has no TTE labels or queries.
                "anchor_offset": anchor - start if contains_anchor else None,
                "outcome_labels": [],
                "threshold_query": None,
            }],
        }
        for field in _TARGET_DTYPES:
            window = built[field][start:end]
            sample[field] = window.tolist() if isinstance(window, np.ndarray) else window
        if "anchors" in built:
            # In-stream labels were computed on the whole stay; this window keeps the
            # anchors sampled inside it.
            sample["segments"][0]["anchors"] = [
                {"offset": item["anchor_idx"] - start, "cr": item["cr"],
                 "queries": item["queries"]}
                for item in built["anchors"] if start <= item["anchor_idx"] < end
            ]
        values = part.value_list(lo, hi)
        if self.continuation_header is not None and row > first:
            values = self._prepend_header(sample, part, base, values)
        elif self.max_tokens is not None and length > self.max_tokens:
            raise TargetContractError(f"GEM window of {length} tokens exceeds max_tokens {self.max_tokens}")
        if self.value_channel:
            self._add_value_channel(sample, {"token": sample["input_ids"], "value": values})
        return sample

    def _prepend_header(self, sample: dict[str, Any], part: "_GemPart", base: int,
                        values: list) -> list:
        """CONTINUATION HEADER (module docstring): the stay's header before this window."""
        head_tokens = part.token[base:base + 64].tolist()
        h = stay_header_length(head_tokens, self.continuation_header)
        if h == 0:
            return values
        total = h + len(sample["input_ids"])
        if self.max_tokens is not None and total > self.max_tokens:
            raise TargetContractError(
                f"continuation window with its {h}-token header is {total} tokens, over "
                f"max_tokens {self.max_tokens}: cut the shard with "
                "dataset.gem_window_bounds_with_header")
        sample["input_ids"] = part.token[base:base + h].tolist() + sample["input_ids"]
        sample["attention_mask"] = [1] * total
        sample["pos_min"] = part.pos_min[base:base + h].tolist() + sample["pos_min"]
        for field in ("soft_token", "soft_weight"):
            if sample.get(field) is not None:
                sample[field] = part.soft_list(field, base, base + h) + sample[field]
        for field, dtype in _TARGET_DTYPES.items():
            pad = [False] * h if dtype is np.bool_ else ([0.0] * h if dtype is np.float64 else [0] * h)
            sample[field] = pad + list(sample[field])
        segment = sample["segments"][0]
        segment["packed_end"] = total
        segment["header_tokens"] = h
        if segment.get("anchor_offset") is not None:
            segment["anchor_offset"] += h
        for anchor in segment.get("anchors", ()):
            anchor["offset"] += h
        return part.value_list(base, base + h) + values

    def sample_lengths(self) -> list[int]:
        """Tokens per training row, header included (for the token-budget sampler).

        With `continuation_header`, a row over `max_tokens` once its header is added is
        refused here, when the loader is built, rather than when the row is first drawn:
        the shard was cut with `gem_window_bounds`, not `gem_window_bounds_with_header`."""
        corpus = getattr(self, "corpus", None)
        if corpus is None:
            return [len(record["token"]) for record in self.records]
        lengths = corpus.window_lengths()
        if self.continuation_header is None:
            return lengths
        out = []
        over = 0
        for index, length in enumerate(lengths):
            part, row = corpus.window(index)
            first, _ = part.stay_windows(int(part.win_stay[row]))
            if row > first:
                base = int(part.win_offset[first])
                length += stay_header_length(part.token[base:base + 64].tolist(), self.continuation_header)
            if self.max_tokens is not None and length > self.max_tokens:
                over += 1
            out.append(length)
        if over:
            raise TargetContractError(
                f"{over} continuation window(s) exceed max_tokens {self.max_tokens} with the "
                "stay header: re-cut the shard with dataset.gem_window_bounds_with_header "
                "or set trunk.continuation_header: false")
        return out

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


# ------------------------------------------------------------------ columnar GEM corpus

_GEM_WINDOW_FIELDS = ("token", "pos_min", "value", "target_eligible")
_GEM_META_FIELDS = ("continuation_index", "n_windows", "source_start", "source_end",
                    "continues_from_previous", "continues_to_next", "anchor_idx",
                    "anchor_min", "partition", "artifact_hashes")
_GEM_CONTRACT = ("GEM window {index} of a stay breaks the continuation contract "
                 "(missing, duplicated or misaligned window)")
_GEM_ONLY = "GEM (hospitalization) records load only through representation='gem'"
# The on-disk cache sits beside the shard (git-ignored data/output tree), one directory
# per (partition, shard SHA-256, vocabulary hash); bump the version when its layout changes.
GEM_CACHE_DIR = "gem_cache"
GEM_CACHE_VERSION = 1
_TARGET_DTYPES = {"ntp_target": np.int64, "ntp_mask": np.bool_, "ntp_delta_min": np.int64,
                  "value_target": np.float64, "value_mask": np.bool_}


class _Mapped:
    """Pickle stand-in for a memory-mapped array: reopened from the cache, never copied."""


class _GemPart:
    """Whole stays of one shard (or one list of window rows), stored columnar.

    Event arrays are flat over the stays' streams, each stay's windows adjacent in
    continuation order: ``token`` int32, ``pos_min`` int64, ``value`` float64 (None when
    every value is missing; ``value_null`` marks missing values, None when there are
    none — a NaN stays a NaN), ``target_eligible`` bool, ``soft_token`` / ``soft_weight``
    (events x width, or None). ``win_offset`` [W+1] indexes events per window,
    ``win_stay`` / ``win_hash`` give a window's stay and artifact-hash entry, and
    ``stay_win`` [S+1] indexes windows per stay. Stay scalars validated equal across
    windows (anchor, partition) are kept once per stay. About 57 bytes per event with
    soft bins of width 3, against ~450-585 for the Python rows."""

    ARRAYS = ("token", "pos_min", "value", "value_null", "target_eligible", "soft_token",
              "soft_weight", "win_offset", "win_stay", "win_hash", "stay_win", "anchor_idx",
              "anchor_null", "anchor_min", "anchor_min_null", "stay_partition")

    def __init__(self, arrays: Mapping[str, np.ndarray | None], *, keys: list[str],
                 hashes: list[Any], partitions: list[Any], cache_dir: Path | None = None):
        for name in self.ARRAYS:
            setattr(self, name, arrays.get(name))
        self.keys = keys
        self.hashes = hashes
        self.partitions = partitions
        self.cache_dir = cache_dir

    def __getstate__(self) -> dict[str, Any]:
        # A spawned DataLoader worker reopens the cache instead of receiving a copy.
        return {name: _Mapped() if isinstance(value, np.memmap) else value
                for name, value in self.__dict__.items()}

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            if isinstance(value, _Mapped):
                state[name] = np.load(state["cache_dir"] / f"{name}.npy", mmap_mode="r")
        self.__dict__.update(state)

    # ---- construction

    @staticmethod
    def _index(meta: list[Mapping[str, Any]], keys: list[str],
               lengths: Mapping[str, list]) -> dict[str, Any]:
        """Validate the continuation contract on windows sorted by (key, continuation
        index) and return the window/stay index arrays (`_GemPart` docstring)."""
        win_offset, win_stay, win_hash, stay_win = [0], [], [], [0]
        anchor_idx, anchor_min, stay_partition = [], [], []
        stay_keys: list[str] = []
        hash_codes: dict[str, int] = {}
        hashes: list[Any] = []
        partition_codes: dict[Any, int] = {}
        i, total = 0, len(meta)
        while i < total:
            key = keys[i]
            j = i
            while j < total and keys[j] == key:
                j += 1
            first, n, offset = meta[i], j - i, 0
            for index in range(n):
                window, length = meta[i + index], lengths["token"][i + index]
                if (int(window["continuation_index"]) != index
                        or int(window.get("n_windows", n)) != n
                        or int(window["source_start"]) != offset
                        or int(window["source_end"]) - int(window["source_start"]) != length
                        or length == 0
                        or bool(window["continues_from_previous"]) != (index > 0)
                        or bool(window["continues_to_next"]) != (index < n - 1)
                        or any(lengths[f][i + index] != length for f in _GEM_WINDOW_FIELDS)):
                    raise TargetContractError(_GEM_CONTRACT.format(index=index))
                for field in ("anchor_idx", "anchor_min", "partition"):
                    if window.get(field) != first.get(field):
                        raise TargetContractError(f"GEM windows of a stay disagree on {field}")
                offset += length
                win_offset.append(win_offset[-1] + length)
                win_stay.append(len(stay_keys))
                hashed = json.dumps(window.get("artifact_hashes"), sort_keys=True, default=str)
                if hashed not in hash_codes:
                    hash_codes[hashed] = len(hashes)
                    hashes.append(window.get("artifact_hashes"))
                win_hash.append(hash_codes[hashed])
            stay_keys.append(key)
            stay_win.append(j)
            anchor_idx.append(first.get("anchor_idx"))
            anchor_min.append(first.get("anchor_min"))
            stay_partition.append(partition_codes.setdefault(first.get("partition"),
                                                             len(partition_codes)))
            i = j
        return {
            "arrays": {
                "win_offset": np.asarray(win_offset, dtype=np.int64),
                "win_stay": np.asarray(win_stay, dtype=np.int64),
                "win_hash": np.asarray(win_hash, dtype=np.int32),
                "stay_win": np.asarray(stay_win, dtype=np.int64),
                "anchor_idx": np.asarray([-1 if a is None else int(a) for a in anchor_idx],
                                         dtype=np.int64),
                "anchor_null": np.asarray([a is None for a in anchor_idx], dtype=np.bool_),
                "anchor_min": np.asarray([-1 if a is None else int(a) for a in anchor_min],
                                         dtype=np.int64),
                "anchor_min_null": np.asarray([a is None for a in anchor_min], dtype=np.bool_),
                "stay_partition": np.asarray(stay_partition, dtype=np.int32),
            },
            "keys": stay_keys,
            "hashes": hashes,
            "partitions": list(partition_codes),
        }

    @staticmethod
    def _tokens(flat: np.ndarray) -> np.ndarray:
        if flat.size and (flat.min() < np.iinfo(np.int32).min or flat.max() > np.iinfo(np.int32).max):
            raise TargetContractError("token id is outside the frozen vocabulary")
        return flat.astype(np.int32)

    @staticmethod
    def _values(flat: np.ndarray, null: np.ndarray) -> dict[str, np.ndarray | None]:
        if null.size and null.all():
            return {"value": None, "value_null": None}
        return {"value": np.where(null, np.nan, flat).astype(np.float64),
                "value_null": null if null.any() else None}

    @classmethod
    def from_rows(cls, records: Sequence[Mapping[str, Any]]) -> "_GemPart":
        by_key: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            by_key.setdefault(_episode_key(record), []).append(record)
        ordered, keys = [], []
        for key in sorted(by_key):
            windows = sorted(by_key[key], key=lambda r: int(r["continuation_index"]))
            ordered.extend(windows)
            keys.extend([key] * len(windows))
        index = cls._index(ordered, keys, {f: [len(r[f]) for r in ordered]
                                           for f in _GEM_WINDOW_FIELDS})
        events = int(index["arrays"]["win_offset"][-1])
        chain = itertools.chain.from_iterable
        values = list(chain(r["value"] for r in ordered))
        arrays = {
            **index["arrays"],
            "token": cls._tokens(np.fromiter(chain(r["token"] for r in ordered),
                                             dtype=np.int64, count=events)),
            "pos_min": np.fromiter(chain(r["pos_min"] for r in ordered), dtype=np.int64,
                                   count=events),
            "target_eligible": np.fromiter((bool(v) for v in chain(
                r["target_eligible"] for r in ordered)), dtype=np.bool_, count=events),
            **cls._values(
                np.fromiter((math.nan if v is None else float(v) for v in values),
                            dtype=np.float64, count=events),
                np.fromiter((v is None for v in values), dtype=np.bool_, count=events)),
        }
        for field, dtype in (("soft_token", np.int32), ("soft_weight", np.float64)):
            present = [r.get(field) is not None for r in ordered]
            if not any(present):
                arrays[field] = None
                continue
            blocks = [np.asarray(r[field], dtype=dtype) for r in ordered] if all(present) else []
            if (not blocks or any(b.ndim != 2 or b.shape[0] != len(r["token"])
                                  for b, r in zip(blocks, ordered))
                    or len({b.shape[1] for b in blocks}) != 1):
                raise TargetContractError(
                    f"GEM {field} must be given for every window, one equal-width row "
                    "per event")
            arrays[field] = np.concatenate(blocks)
        return cls(arrays, keys=index["keys"], hashes=index["hashes"],
                   partitions=index["partitions"])

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> "_GemPart":
        """From a gem_events frame already sorted by (key, continuation_index)."""
        if "trajectory" not in frame.columns or (
                frame.height and not (frame["trajectory"] == "hospitalization").all()):
            raise TargetContractError(_GEM_ONLY)
        key_column = "episode_key" if "episode_key" in frame.columns else "hosp_id"
        keys = frame[key_column].to_list()
        if any(not isinstance(key, str) or not key for key in keys):
            raise TargetContractError("GEM window is missing an opaque episode key")
        meta = frame.select([c for c in _GEM_META_FIELDS if c in frame.columns]).to_dicts()
        index = cls._index(meta, keys, {f: frame[f].list.len().to_list()
                                        for f in _GEM_WINDOW_FIELDS})

        def flat(field: str) -> pl.Series:
            return frame[field].explode(empty_as_null=False)

        token, pos_min = flat("token"), flat("pos_min")
        if token.null_count() or pos_min.null_count():
            raise TargetContractError("GEM window has a missing token or position")
        value = flat("value").cast(pl.Float64)
        arrays = {
            **index["arrays"],
            "token": cls._tokens(token.to_numpy()),
            "pos_min": pos_min.cast(pl.Int64).to_numpy(),
            "target_eligible": flat("target_eligible").fill_null(False).cast(pl.Boolean)
                                                      .to_numpy(),
            **cls._values(value.fill_null(math.nan).to_numpy(), value.is_null().to_numpy()),
        }
        for field, dtype in (("soft_token", np.int32), ("soft_weight", np.float64)):
            column = frame[field] if field in frame.columns else None
            if column is None or column.null_count() == frame.height:
                arrays[field] = None
                continue
            rows = column.explode(empty_as_null=False)
            widths = rows.list.len()
            if (column.null_count() or rows.null_count() or widths.n_unique() != 1
                    or len(rows) != len(token)):
                raise TargetContractError(
                    f"GEM {field} must be given for every window, one equal-width row "
                    "per event")
            values = rows.explode(empty_as_null=False)
            if values.null_count():
                raise TargetContractError(f"GEM {field} has a missing entry")
            arrays[field] = values.to_numpy().astype(dtype).reshape(len(token), widths[0])
        return cls(arrays, keys=index["keys"], hashes=index["hashes"],
                   partitions=index["partitions"])

    # ---- the on-disk cache

    def save(self, directory: Path, meta: Mapping[str, Any]) -> None:
        directory.mkdir(parents=True)
        stored = [name for name in self.ARRAYS if getattr(self, name) is not None]
        for name in stored:
            np.save(directory / f"{name}.npy", np.ascontiguousarray(getattr(self, name)))
        (directory / "meta.json").write_text(json.dumps({
            **meta, "arrays": stored, "events": self.events, "windows": self.windows,
            "keys": self.keys, "hashes": self.hashes, "partitions": self.partitions,
        }))

    @classmethod
    def load(cls, directory: Path, expected: Mapping[str, Any]) -> "_GemPart":
        meta = json.loads((directory / "meta.json").read_text())
        if any(meta.get(name) != value for name, value in expected.items()):
            raise TargetContractError(
                f"stale GEM cache {directory}: it was built from another shard, "
                "vocabulary or partition; delete it")
        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r")
                  for name in meta["arrays"]}
        part = cls(arrays, keys=meta["keys"], hashes=meta["hashes"],
                   partitions=meta["partitions"], cache_dir=directory)
        if part.events != meta["events"] or part.windows != meta["windows"] or len(
                part.token) != part.events:
            raise TargetContractError(f"stale GEM cache {directory}: incomplete; delete it")
        return part

    # ---- access

    @property
    def events(self) -> int:
        return int(self.win_offset[-1])

    @property
    def windows(self) -> int:
        return len(self.win_offset) - 1

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in self.ARRAYS
                   if getattr(self, name) is not None)

    def stay_windows(self, stay: int) -> tuple[int, int]:
        return int(self.stay_win[stay]), int(self.stay_win[stay + 1])

    def value_list(self, lo: int, hi: int) -> list[float | None]:
        if self.value is None:
            return [None] * (hi - lo)
        values = self.value[lo:hi].tolist()
        if self.value_null is None:
            return values
        return [None if null else value
                for value, null in zip(values, self.value_null[lo:hi].tolist())]

    def soft_list(self, field: str, lo: int, hi: int) -> list[list] | None:
        array = getattr(self, field)
        return None if array is None else array[lo:hi].tolist()

    def stream(self, stay: int) -> dict[str, Any]:
        """The stay's whole stream, the `TargetBuilder` input (assembled on access)."""
        first, last = self.stay_windows(stay)
        bounds = self.win_offset[first:last + 1].tolist()
        lo, hi = bounds[0], bounds[-1]
        return {
            "episode_key": self.keys[stay],
            "token": self.token[lo:hi].tolist(),
            "pos_min": self.pos_min[lo:hi].tolist(),
            "value": self.value_list(lo, hi),
            "target_eligible": self.target_eligible[lo:hi].tolist(),
            "anchor_idx": None if self.anchor_null[stay] else int(self.anchor_idx[stay]),
            "anchor_min": None if self.anchor_min_null[stay] else int(self.anchor_min[stay]),
            "outcomes": [],
            "windows": [(a - lo, b - lo) for a, b in zip(bounds, bounds[1:])],
        }

    def window_field(self, row: int, name: str) -> Any:
        stay = int(self.win_stay[row])
        first, last = self.stay_windows(stay)
        lo, hi = int(self.win_offset[row]), int(self.win_offset[row + 1])
        base, index = int(self.win_offset[first]), row - first
        fields = {
            "episode_key": lambda: self.keys[stay],
            "trajectory": lambda: "hospitalization",
            "token": lambda: self.token[lo:hi].tolist(),
            "soft_token": lambda: self.soft_list("soft_token", lo, hi),
            "soft_weight": lambda: self.soft_list("soft_weight", lo, hi),
            "pos_min": lambda: self.pos_min[lo:hi].tolist(),
            "value": lambda: self.value_list(lo, hi),
            "target_eligible": lambda: self.target_eligible[lo:hi].tolist(),
            "partition": lambda: self.partitions[int(self.stay_partition[stay])],
            "source_start": lambda: lo - base,
            "source_end": lambda: hi - base,
            "continuation_index": lambda: index,
            "n_windows": lambda: last - first,
            "continues_from_previous": lambda: index > 0,
            "continues_to_next": lambda: index < last - first - 1,
            "anchor_idx": lambda: None if self.anchor_null[stay] else int(self.anchor_idx[stay]),
            "anchor_min": lambda: (None if self.anchor_min_null[stay]
                                   else int(self.anchor_min[stay])),
            "artifact_hashes": lambda: self.hashes[int(self.win_hash[row])],
        }
        if name not in fields:
            raise KeyError(name)
        return fields[name]()


_GEM_RECORD_KEYS = ("episode_key", "trajectory", "token", "soft_token", "soft_weight",
                    "pos_min", "value", "target_eligible", "partition", "source_start",
                    "source_end", "continuation_index", "n_windows",
                    "continues_from_previous", "continues_to_next", "anchor_idx",
                    "anchor_min", "artifact_hashes")


class GemCorpus:
    """Full-hospitalization windows of one or more shards, columnar (`_GemPart`), in the
    dataset's order: stays sorted by episode key, each stay's windows in continuation
    order — the order the list-of-dicts path produced. Parts are never concatenated
    (a memory-mapped part stays mapped); a stay lives in exactly one part."""

    def __init__(self, parts: Sequence[_GemPart]):
        self.parts = list(parts)
        stays = sorted((key, p, s) for p, part in enumerate(self.parts)
                       for s, key in enumerate(part.keys))
        self.stays: dict[str, tuple[int, int]] = {}
        part_ids, rows = [], []
        for key, p, s in stays:
            if key in self.stays:
                raise TargetContractError(
                    "GEM windows of one stay come from two shards; "
                    + _GEM_CONTRACT.format(index=0))
            self.stays[key] = (p, s)
            first, last = self.parts[p].stay_windows(s)
            part_ids.append(np.full(last - first, p, dtype=np.int32))
            rows.append(np.arange(first, last, dtype=np.int64))
        self._part = np.concatenate(part_ids) if part_ids else np.zeros(0, dtype=np.int32)
        self._row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> "GemCorpus":
        return cls([_GemPart.from_rows(records)])

    @classmethod
    def from_parquet(cls, path: str | Path, *, site: str | None = None,
                     partition: str | None = None, cache: bool = False) -> "GemCorpus":
        """One gem_events.parquet (only `partition`'s rows when given) without building
        Python rows. Keys are ``<site>:<hosp_id>`` when `site` is given (the same raw
        identifier at two sites stays two stays). `cache=True` memory-maps a columnar
        copy kept in ``gem_cache/`` beside the shard (`_cached_part`)."""
        path = Path(path)
        part = _cached_part(path, partition) if cache else _GemPart.from_frame(
            _read_gem_frame(path, partition))
        if site is not None:
            part.keys = [f"{site}:{key}" for key in part.keys]
        return cls([part])

    @classmethod
    def concat(cls, corpora: Sequence["GemCorpus"]) -> "GemCorpus":
        return cls([part for corpus in corpora for part in corpus.parts])

    def __len__(self) -> int:
        return len(self._row)

    @property
    def events(self) -> int:
        return sum(part.events for part in self.parts)

    @property
    def nbytes(self) -> int:
        """Bytes of the columnar arrays (memory-mapped ones are shared page cache)."""
        return sum(part.nbytes for part in self.parts)

    def hashes(self) -> list[Any]:
        return [h for part in self.parts for h in part.hashes]

    def window(self, index: int) -> tuple[_GemPart, int]:
        return self.parts[int(self._part[index])], int(self._row[index])

    def window_lengths(self) -> list[int]:
        """Events per window, in dataset order (no per-window lists are built)."""
        lengths = np.zeros(len(self), dtype=np.int64)
        for p, part in enumerate(self.parts):
            mine = self._part == p
            lengths[mine] = np.diff(part.win_offset)[self._row[mine]]
        return lengths.tolist()

    def has_numeric_values(self) -> bool:
        return any(part.value is not None and bool(np.isfinite(part.value).any())
                   for part in self.parts)

    def drop_values(self) -> None:
        """Every value missing (a dry run without value stats)."""
        for part in self.parts:
            part.value = part.value_null = None

    def drop_soft(self) -> None:
        """Hard-token inputs: no soft bins on any window."""
        for part in self.parts:
            part.soft_token = part.soft_weight = None

    def soft_width(self) -> int | None:
        """The smallest soft-bin width over the parts, None if any part has none."""
        widths = [None if part.soft_token is None else part.soft_token.shape[1]
                  for part in self.parts]
        return None if not widths or None in widths else min(widths)


class _GemWindowView(Mapping):
    """Read-only record of one window, its fields built from the columns on access."""

    __slots__ = ("_part", "_row")

    def __init__(self, part: _GemPart, row: int):
        self._part, self._row = part, row

    def __getitem__(self, name: str) -> Any:
        return self._part.window_field(self._row, name)

    def __iter__(self):
        return iter(_GEM_RECORD_KEYS)

    def __len__(self) -> int:
        return len(_GEM_RECORD_KEYS)


class _GemRecords(Sequence):
    """`ModelDataset.records` on the gem representation: window views in dataset order."""

    def __init__(self, corpus: GemCorpus):
        self._corpus = corpus

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return _GemWindowView(*self._corpus.window(index))


class _GemStreams(Mapping):
    """``{episode key: whole-stay stream}``, each stream assembled on access."""

    def __init__(self, corpus: GemCorpus):
        self._corpus = corpus

    def __getitem__(self, key: str) -> dict[str, Any]:
        p, stay = self._corpus.stays[key]
        return self._corpus.parts[p].stream(stay)

    def __iter__(self):
        return iter(self._corpus.stays)

    def __len__(self) -> int:
        return len(self._corpus.stays)


_GEM_COLUMNS = ("hosp_id", "episode_key", "trajectory", *_GEM_WINDOW_FIELDS, "soft_token",
                "soft_weight", *_GEM_META_FIELDS)


def _read_gem_frame(path: Path, partition: str | None) -> pl.DataFrame:
    frame = pl.scan_parquet(path)
    names = frame.collect_schema().names()
    if partition is not None:
        frame = frame.filter(pl.col("partition") == partition)
    key = "episode_key" if "episode_key" in names else "hosp_id"
    return (frame.select([c for c in _GEM_COLUMNS if c in names])
            .sort([key, "continuation_index"], maintain_order=True).collect())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


_TMP_BUILD = re.compile(r"^\.(?P<name>.+)\.(?P<pid>\d+)\.tmp$")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


def remove_stale_cache_builds(root: Path) -> list[str]:
    """Delete every ``.<name>.<pid>.tmp`` cache build directory under `root` whose pid
    is not a running process (a crashed build); returns the names removed. Called with
    the cache lock held, so a live build (its pid running) is never touched."""
    removed = []
    for entry in Path(root).iterdir() if Path(root).is_dir() else ():
        match = _TMP_BUILD.match(entry.name)
        if match and entry.is_dir() and not _pid_alive(int(match.group("pid"))):
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
    return removed


def _cached_part(path: Path, partition: str | None) -> _GemPart:
    """The memory-mapped columnar copy of one shard partition, built on first use.

    It lives in ``gem_cache/<partition>-<shard sha256>-<vocabulary hash>/`` beside the
    shard, so a rewritten shard or a re-tokenized vocabulary gets a new directory, and a
    directory whose recorded shard, vocabulary or partition disagrees is refused. DDP
    ranks (and DataLoader workers) on one node map the same files, so the page cache
    holds ONE copy per node, not one per process. An exclusive lock on
    ``gem_cache/.lock`` makes the first rank build it while the others wait; it is
    written to a temporary directory and renamed into place, so a crashed build is never
    read. Old directories are not deleted."""
    digest = _file_sha256(path)
    vocabularies = sorted(str(v) for v in pl.scan_parquet(path).select(
        pl.col("artifact_hashes").struct.field("vocabulary")).unique().collect().to_series())
    vocabulary = vocabularies[0] if len(vocabularies) == 1 else hashlib.sha256(
        json.dumps(vocabularies).encode()).hexdigest()
    expected = {"cache_version": GEM_CACHE_VERSION, "shard_sha256": digest,
                "vocabulary": vocabulary, "partition": partition}
    root = path.parent / GEM_CACHE_DIR
    root.mkdir(exist_ok=True)
    directory = root / f"{partition or 'all'}-{digest[:16]}-{vocabulary[:16]}"
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            # A build that crashed (killed rank, OOM) leaves its temporary directory
            # behind; under the lock, any whose writer is no longer running is removed.
            remove_stale_cache_builds(root)
            if not directory.exists():
                temporary = root / f".{directory.name}.{os.getpid()}.tmp"
                shutil.rmtree(temporary, ignore_errors=True)
                _GemPart.from_frame(_read_gem_frame(path, partition)).save(temporary, expected)
                os.replace(temporary, directory)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return _GemPart.load(directory, expected)


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
