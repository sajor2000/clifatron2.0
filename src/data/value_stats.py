"""Per-token value-head normalization statistics (frozen from a reference site).

The ORA value-regression head (`ValueRegressionHead`) predicts the continuous value
of the next event as a Gaussian. Raw ICU values span wildly different magnitudes per
concept (creatinine ~1, platelet count ~200,000), so an un-normalized Gaussian NLL is
dominated by high-magnitude concepts (observed val≈46000 on real MIMIC) and never
trains well.

Fix: standardize each numeric event to ~N(0,1) using per-**token** center/scale frozen
from a reference site (the same "freeze on MIMIC, apply identically everywhere" pattern
as the decile bin edges). `TargetBuilder.value_stats` consumes exactly this map
(`token_id -> (center, scale)`), applying `(value - center) / scale`, and refuses to
build a numeric target whose token lacks stats — so this generator is what unblocks
real value-head pretraining.

Center/scale are **robust** (median / IQR-derived) so physiologic outliers (severe
hyperkalemia, extreme lactate) do not inflate the scale and flatten the signal on the
dangerous tails we most care about.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

# 1.349 = IQR of a standard normal (Φ⁻¹(0.75) − Φ⁻¹(0.25)); makes robust scale
# comparable to a standard deviation for well-behaved concepts.
_IQR_TO_SIGMA = 1.349
_MIN_COUNT = 20          # need enough observations for a stable center/scale
_MIN_SCALE = 1e-6        # never emit a non-positive scale (TargetBuilder rejects it)


def compute_value_stats(
    tokens_per_stay: list[list[int]],
    values_per_stay: list[list[float | None]],
    *,
    min_count: int = _MIN_COUNT,
    robust: bool = True,
) -> dict[int, tuple[float, float]]:
    """Per-token (center, scale) from parallel token/value sequences.

    COVERAGE CONTRACT: every token that carries at least one finite value gets stats.
    `TargetBuilder` raises on any finite numeric target whose token lacks stats, so
    silently dropping a rare-but-real numeric token would deterministically abort
    pretraining. `min_count` therefore controls only the CENTER/SCALE ESTIMATOR, not
    omission: below the threshold a token still gets stats, but with a robust
    wider-fallback scale (never a missing entry). Only tokens that never carry a finite
    value (purely categorical) are absent — they have nothing to normalize.

    robust=True (default): center = median, scale = IQR / 1.349 (falls back to std,
    then to a fixed floor, when the IQR/std is degenerate). robust=False: mean / std.
    """
    if len(tokens_per_stay) != len(values_per_stay):
        raise ValueError("tokens_per_stay and values_per_stay must have equal length")

    buckets: dict[int, list[float]] = {}
    for toks, vals in zip(tokens_per_stay, values_per_stay):
        if len(toks) != len(vals):
            raise ValueError("token and value sequences must align within a stay")
        for tok, val in zip(toks, vals):
            if val is None:
                continue
            v = float(val)
            if not math.isfinite(v):
                continue
            buckets.setdefault(int(tok), []).append(v)

    stats: dict[int, tuple[float, float]] = {}
    for token_id, observed in buckets.items():
        arr = np.asarray(observed, dtype=np.float64)
        # A token seen too few times for a stable robust estimate still gets stats
        # (coverage contract) — we just widen the fallback so its NLL isn't overconfident.
        under_min = len(observed) < min_count
        if robust and not under_min:
            center = float(np.median(arr))
            q75, q25 = np.percentile(arr, [75, 25])
            scale = float((q75 - q25) / _IQR_TO_SIGMA)
            if not math.isfinite(scale) or scale <= _MIN_SCALE:
                scale = float(np.std(arr))  # degenerate IQR (e.g. spiky discrete value)
        else:
            # rare token, or mean/std mode: mean/std is the most stable on few points
            center = float(np.mean(arr))
            scale = float(np.std(arr))
            if under_min:
                # too few points to trust the spread → widen toward |center| so the
                # Gaussian isn't overconfident on an under-observed concept
                scale = max(scale, abs(center) * 0.5)
        if not math.isfinite(scale) or scale <= _MIN_SCALE:
            scale = max(abs(center) * 0.5, _MIN_SCALE)  # constant/degenerate token
        stats[token_id] = (center, scale)
    return stats


def compute_value_stats_from_events(
    events_path: str | Path,
    *,
    min_count: int = _MIN_COUNT,
    robust: bool = True,
) -> dict[int, tuple[float, float]]:
    """Compute value stats from a reference-site tokenized `events.parquet`.

    Expects the tokenizer schema: parallel `token` (list[int]) and `value`
    (list[float|null]) columns per stay (see `src/data/tokenize.py`)."""
    import polars as pl

    df = pl.read_parquet(events_path)
    for col in ("token", "value"):
        if col not in df.columns:
            raise ValueError(f"{events_path} missing required column {col!r}")
    tokens = df["token"].to_list()
    values = df["value"].to_list()
    return compute_value_stats(tokens, values, min_count=min_count, robust=robust)


def vocab_hash(vocab: dict) -> str:
    """SHA-256 of the fused vocab (matches tokenize._json_sha256 canonicalization).

    Binds a value-stats artifact to the exact vocabulary whose token ids give it
    meaning, so a stale or cross-vocabulary file is rejected instead of silently
    applying unrelated centers/scales."""
    payload = json.dumps(vocab, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_value_stats(
    stats: dict[int, tuple[float, float]],
    out_path: str | Path,
    *,
    vocab: dict | None = None,
    vocab_sha: str | None = None,
    min_count: int = _MIN_COUNT,
    robust: bool = True,
) -> Path:
    """Write frozen stats as a self-describing artifact bound to its vocabulary.

    Format: `{"schema": 2, "vocab_hash": <sha256|null>, "min_count", "robust",
    "stats": {token_id: [center, scale]}}`. Pass `vocab` (the fused vocab dict) or a
    precomputed `vocab_sha` to bind the artifact; the loader verifies it."""
    if vocab is not None and vocab_sha is None:
        vocab_sha = vocab_hash(vocab)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "schema": 2,
        "vocab_hash": vocab_sha,
        "min_count": int(min_count),
        "robust": bool(robust),
        "stats": {str(tok): [center, scale] for tok, (center, scale) in sorted(stats.items())},
    }
    out.write_text(json.dumps(blob, indent=2))
    return out


def load_value_stats(
    path: str | Path, *, expected_vocab_hash: str | None = None
) -> dict[int, tuple[float, float]]:
    """Load a value-stats artifact, verifying vocabulary identity when available.

    Accepts both the schema-2 self-describing format and the legacy bare
    `{token_id: [center, scale]}` map. If `expected_vocab_hash` is given, the
    artifact must be schema-2 and carry a matching `vocab_hash`; otherwise token
    ids could be interpreted against an unrelated vocabulary."""
    blob = json.loads(Path(path).read_text())
    if isinstance(blob, dict) and "stats" in blob:  # schema-2 self-describing
        stored = blob.get("vocab_hash")
        if expected_vocab_hash is not None:
            if stored is None:
                raise ValueError("value-stats artifact is unbound; expected a vocab_hash")
            if stored != expected_vocab_hash:
                raise ValueError(
                    f"value-stats vocabulary hash mismatch: artifact {stored[:12]}… != "
                    f"expected {expected_vocab_hash[:12]}… (stale or cross-vocabulary stats)"
                )
        raw = blob["stats"]
    else:  # legacy bare map — no identity to verify
        if expected_vocab_hash is not None:
            raise ValueError("legacy value-stats map cannot be verified against a vocabulary")
        raw = blob
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Freeze per-token value-head normalization stats from a reference site."
    )
    ap.add_argument("--events", required=True,
                    help="reference-site tokenized events.parquet (freeze stats here)")
    ap.add_argument("--out", required=True, help="output value_stats.json path")
    ap.add_argument("--vocab", default=None,
                    help="vocab.json to bind stats to (defaults to a sibling of --events)")
    ap.add_argument("--min-count", type=int, default=_MIN_COUNT,
                    help="observations below which a token uses a wider fallback scale "
                         "(NOT a drop threshold — every numeric token still gets stats)")
    ap.add_argument("--mean-std", action="store_true",
                    help="use mean/std instead of robust median/IQR")
    args = ap.parse_args()

    robust = not args.mean_std
    stats = compute_value_stats_from_events(
        args.events, min_count=args.min_count, robust=robust
    )

    # Bind to the fused vocabulary so a stale/cross-vocab artifact is detectable.
    vocab_path = Path(args.vocab) if args.vocab else Path(args.events).with_name("vocab.json")
    vsha = None
    if vocab_path.exists():
        blob = json.loads(vocab_path.read_text())
        vocab = blob.get("vocab", blob)  # tolerate {"vocab":..} or a bare map
        vsha = vocab_hash(vocab)
    else:
        print(f"warning: no vocab.json at {vocab_path} — artifact will be UNBOUND (no identity check)")

    path = write_value_stats(
        stats, args.out, vocab_sha=vsha, min_count=args.min_count, robust=robust
    )
    print(f"wrote {len(stats):,} per-token value stats -> {path}"
          + (f" (vocab {vsha[:12]}…)" if vsha else " (unbound)"))


if __name__ == "__main__":
    _main()
