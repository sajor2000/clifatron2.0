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

    Only numeric (non-null, finite) values contribute. A token that never carries a
    finite value, or has fewer than `min_count` observations, is omitted — its events
    are categorical (no value to normalize) or too rare to standardize stably.

    robust=True (default): center = median, scale = IQR / 1.349 (falls back to std
    when the IQR is degenerate). robust=False: center = mean, scale = std.
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
        if len(observed) < min_count:
            continue
        arr = np.asarray(observed, dtype=np.float64)
        if robust:
            center = float(np.median(arr))
            q75, q25 = np.percentile(arr, [75, 25])
            scale = float((q75 - q25) / _IQR_TO_SIGMA)
            if not math.isfinite(scale) or scale <= _MIN_SCALE:
                scale = float(np.std(arr))  # degenerate IQR (e.g. spiky discrete value)
        else:
            center = float(np.mean(arr))
            scale = float(np.std(arr))
        if not math.isfinite(scale) or scale <= _MIN_SCALE:
            scale = _MIN_SCALE  # constant-valued token: avoid a zero/negative scale
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


def write_value_stats(stats: dict[int, tuple[float, float]], out_path: str | Path) -> Path:
    """Write the frozen stats as JSON `{token_id: [center, scale]}` for `--value-stats`."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {str(tok): [center, scale] for tok, (center, scale) in sorted(stats.items())}
    out.write_text(json.dumps(blob, indent=2))
    return out


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Freeze per-token value-head normalization stats from a reference site."
    )
    ap.add_argument("--events", required=True,
                    help="reference-site tokenized events.parquet (freeze stats here)")
    ap.add_argument("--out", required=True, help="output value_stats.json path")
    ap.add_argument("--min-count", type=int, default=_MIN_COUNT,
                    help="min observations per token to emit stats")
    ap.add_argument("--mean-std", action="store_true",
                    help="use mean/std instead of robust median/IQR")
    args = ap.parse_args()

    stats = compute_value_stats_from_events(
        args.events, min_count=args.min_count, robust=not args.mean_std
    )
    path = write_value_stats(stats, args.out)
    print(f"wrote {len(stats):,} per-token value stats -> {path}")


if __name__ == "__main__":
    _main()
