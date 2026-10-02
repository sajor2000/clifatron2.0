"""Continuous-fused value arm (McCann et al. 2026, medRxiv 2026.08.04).

Fused concept token + continuous value channel: each event gets a discrete CONCEPT
embedding plus a learned projection of its normalized value, summed before the trunk
(`src/model/encoder_continuous.py`). Numeric values are not binned into the token.
McCann: +30% numeric accuracy, worse calibration than discrete binning — this arm tests
that calibration trade-off. Our DEFAULT is physician clinical-segment bins + soft
discretization (configs/data.yaml, MEMORY.md §E1a).

The arm is DERIVED from the primary clinical-segment shard (U6, KTD8), so only the
representation varies — the events, their order, positions, eligibility and outcomes
are the primary shard's:

- **Edgeless input tokens.** Every binned fused token ``concept=<bin>`` collapses to the
  bare ``concept``; categorical ``concept=<value>`` tokens, bare tokens and specials keep
  their identity (`continuous_fused_vocab`).
- **Current-value channel.** The event's own value, normalized with frozen per-token
  stats fit on the arm's train partition (`normalize_value`, applied by
  `ModelDataset(value_channel=True)`); NaN/categorical/missing values are masked to 0.
- **Primary threshold bins.** The threshold head still queries VALUE bins: the arm's
  vocab artifact carries the primary clinical segments (``primary_segments``, hashed
  into the manifest), its ``n_value_bins`` comes from them, and the joined outcomes'
  ``threshold_bin`` are the primary ones (`segments.bin_index` via `threshold_bin`).

Usage:
    python -m src.data.tokenize_continuous \\
        --primary-vocab output/intermediate_phi/mimic/vocab.json \\
        --primary-events output/intermediate_phi/mimic/events_with_outcomes.parquet \\
        --out output/intermediate_phi/mimic_continuous
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

REPRESENTATION = "continuous_fused"
# Same plausibility bound as TargetBuilder.max_abs_value_z: a finite sentinel (e.g. a
# 999999 pH) is masked as an input exactly as it is dropped as a value-head target.
MAX_ABS_Z = 20.0


def normalize_value(value: float | None, token_id: int,
                    value_stats: Mapping[int, tuple[float, float]], *,
                    max_abs_z: float = MAX_ABS_Z) -> tuple[float, bool]:
    """``(z, mask)`` for one event's CURRENT value under frozen per-token stats.

    ``z = (value - center) / scale`` with the reference site's ``(center, scale)`` for
    `token_id` (`src/data/value_stats.py`). A missing, NaN or infinite value — a
    categorical finding, a presence token — is ``(0.0, False)``, as is an implausible
    ``|z| > max_abs_z``. A finite value whose token has no stats fails closed: the
    stats were not fit on this vocabulary's train partition."""
    if value is None:
        return 0.0, False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not math.isfinite(v):
        return 0.0, False
    stats = value_stats.get(int(token_id))
    if stats is None:
        raise ValueError(
            f"numeric input token {int(token_id)} is missing frozen normalization "
            "statistics; recompute value_stats.json from this arm's train partition"
        )
    center, scale = stats
    z = (v - float(center)) / float(scale)
    if not math.isfinite(z) or abs(z) > max_abs_z:
        return 0.0, False
    return float(z), True


def _binned_concept(token: str, segments: Mapping) -> str | None:
    """The concept of a binned fused token ``concept=<bin index>``, else None."""
    concept, sep, suffix = token.partition("=")
    if not sep or concept not in segments or not suffix.isdigit():
        return None
    return concept if int(suffix) < len(segments[concept]) else None


def continuous_fused_vocab(primary_blob: Mapping) -> tuple[dict, dict[int, int]]:
    """Derive the edgeless vocabulary artifact from the primary (clinical) one.

    Returns ``(blob, remap)`` where `remap` maps every primary token id to its
    continuous-fused id. Ids are assigned in primary id order, so specials keep theirs.
    The blob is tokenizer-v2 shaped with ``segments: {}`` (nothing is binned), and
    records the primary segments + their hash so the threshold head's value bins stay
    the primary ones (`primary_segments`)."""
    from src.data.segments import (
        TOKENIZER_VERSION,
        artifact_binding,
        json_sha256,
        segments_hash,
        vocab_segments,
    )

    segments = vocab_segments(primary_blob)  # refuses a pre-v2 vocabulary
    primary_binding = artifact_binding(primary_blob)
    vocab: dict[str, int] = {}
    remap: dict[int, int] = {}
    for token, old_id in sorted(primary_blob["vocab"].items(), key=lambda kv: kv[1]):
        key = _binned_concept(token, segments) or token
        if key not in vocab:
            vocab[key] = len(vocab)
        remap[int(old_id)] = vocab[key]

    units = primary_blob.get("reference_units") or {}
    manifest = json.loads(json.dumps(primary_blob.get("manifest") or {}))
    hashes = dict(manifest.get("hashes") or {})
    hashes.update({
        "vocabulary": json_sha256(vocab),
        "numeric_edges": json_sha256({}),
        "binning_sources": json_sha256({}),
        "primary_vocabulary": primary_binding["vocabulary"],
        "primary_segments": segments_hash(segments),
    })
    manifest.update({"tokenizer_version": TOKENIZER_VERSION, "hashes": hashes})
    provenance = dict(manifest.get("provenance") or {})
    provenance["representation"] = REPRESENTATION
    provenance["derived_from"] = "primary clinical-segment vocabulary"
    manifest["provenance"] = provenance
    blob = {
        "vocab": vocab,
        "segments": {},
        "binning_sources": {},
        "reference_units": {"concepts": {},
                            "dose_targets": dict(units.get("dose_targets") or {})},
        "concept_sources": primary_blob.get("concept_sources"),
        "precedence_policy": primary_blob.get("precedence_policy"),
        "representation": REPRESENTATION,
        "primary_segments": json.loads(json.dumps(segments)),
        "primary_binning_sources": dict(primary_blob.get("binning_sources") or {}),
        "primary_reference_units": units,
        "manifest": manifest,
    }
    hashes["reference_units"] = json_sha256(blob["reference_units"])
    return blob, remap


def primary_segments(blob: Mapping) -> Mapping:
    """The primary clinical segments a continuous-fused artifact is bound to, verified
    against the manifest's ``primary_segments`` hash (fail closed when absent/altered)."""
    from src.data.segments import ArtifactBindingError, segments_hash

    segments = blob.get("primary_segments") if isinstance(blob, Mapping) else None
    recorded = ((blob.get("manifest") or {}).get("hashes") or {}).get("primary_segments") \
        if isinstance(blob, Mapping) else None
    if blob.get("representation") != REPRESENTATION or not isinstance(segments, Mapping) \
            or not segments or not recorded:
        raise ArtifactBindingError(
            "continuous-fused vocabulary must carry the primary clinical segments and "
            "their primary_segments hash (derive it with src.data.tokenize_continuous)"
        )
    if segments_hash(segments) != recorded:
        raise ArtifactBindingError(
            "continuous-fused primary segments do not match their recorded "
            "primary_segments hash; re-derive the arm from the primary vocabulary"
        )
    return segments


def primary_n_value_bins(blob: Mapping) -> int:
    """`ThresholdHazardHead` value-bin count of a continuous-fused arm: the primary
    clinical segments' (`segments.n_value_bins`)."""
    from src.data.segments import TOKENIZER_VERSION, n_value_bins

    return n_value_bins({"segments": primary_segments(blob),
                         "manifest": {"tokenizer_version": TOKENIZER_VERSION}})


def derive_continuous_fused_shard(primary_events, primary_blob: Mapping,
                                  continuous_blob: Mapping, remap: Mapping[int, int]):
    """Re-express a primary shard in the edgeless vocabulary: token ids remapped, soft
    fields dropped (the value channel replaces them), values/positions/eligibility/
    outcomes kept, and every row re-bound to the continuous-fused artifact. Every input
    row must be bound to `primary_blob` (KTD7)."""
    import polars as pl

    from src.data.segments import artifact_binding, compare_binding

    expected = artifact_binding(primary_blob)
    for hashes in primary_events["artifact_hashes"].to_list():
        compare_binding(hashes, expected, what="primary shard row")
    binding = artifact_binding(continuous_blob)
    lookup = {int(k): int(v) for k, v in remap.items()}
    frame = primary_events.drop([c for c in ("soft_token", "soft_weight")
                                 if c in primary_events.columns])
    tokens = [[lookup[int(i)] for i in ids] for ids in frame["token"].to_list()]
    frame = frame.with_columns(
        pl.Series("token", tokens, dtype=frame.schema["token"]),
        pl.struct([pl.lit(v, pl.String).alias(k) for k, v in binding.items()])
        .alias("artifact_hashes"),
    )
    return frame


def write_continuous_fused_arm(primary_vocab: str | Path, primary_events: str | Path,
                               out_dir: str | Path, *, policy: dict | None = None
                               ) -> tuple[Path, Path]:
    """Write the continuous-fused arm's ``vocab.json`` and events shard (same file name
    as `primary_events`) under `out_dir`. The shard is patient-level PHI: its
    destination is checked against the artifact policy."""
    import polars as pl
    import yaml

    from src.data.cohort import validate_artifact_destination

    root = Path(__file__).parents[2]
    policy = policy or yaml.safe_load((root / "configs/artifact_policy.yaml").read_text())
    out = Path(out_dir)
    events_path = out / Path(primary_events).name
    validate_artifact_destination(events_path, "patient_level_phi", policy)
    primary_blob = json.loads(Path(primary_vocab).read_text())
    blob, remap = continuous_fused_vocab(primary_blob)
    shard = derive_continuous_fused_shard(pl.read_parquet(primary_events), primary_blob,
                                          blob, remap)
    out.mkdir(parents=True, exist_ok=True)
    shard.write_parquet(events_path)
    vocab_path = out / "vocab.json"
    vocab_path.write_text(json.dumps(blob))
    return vocab_path, events_path


def continuous_fused_embedding(x_concept, x_value, value_proj, embedding):
    """Produce fused concept+value embedding for a batch.

    x_concept: [B,T]  concept token IDs (vocab: concept names, no value bins)
    x_value:   [B,T]  normalized continuous values
    value_proj: nn.Linear(1, d) that maps scalar -> d-dimensional
    embedding: nn.Embedding(vocab_size, d)

    Returns: [B,T,d] summed embedding (one token per event, no bin expansion).
    """
    ce = embedding(x_concept)
    vp = value_proj(x_value.unsqueeze(-1))
    return ce + vp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--primary-vocab", required=True,
                    help="the primary clinical-segment vocab.json")
    ap.add_argument("--primary-events", required=True,
                    help="the primary shard (events.parquet or events_with_outcomes.parquet)")
    ap.add_argument("--out", required=True, help="the continuous-fused arm's directory")
    args = ap.parse_args()
    vocab_path, events_path = write_continuous_fused_arm(
        args.primary_vocab, args.primary_events, args.out)
    blob = json.loads(vocab_path.read_text())
    print(f"wrote {vocab_path} ({len(blob['vocab']):,} edgeless tokens) and {events_path}")


if __name__ == "__main__":
    main()
