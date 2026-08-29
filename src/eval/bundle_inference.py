"""Wire a verified bundle to real zero-shot inference (U9 — closes the D1 seam).

`evaluate_site` deliberately has no default `predict_fn`; the CLI used to supply one
that raised, naming the four inputs it lacked: the resolved data config, the
bundle-pinned vocabulary and numeric edges, a policy-checked shard directory, and the
episode frame. A loaded `Bundle` supplies the first three and the caller supplies the
rest, so this module builds the callable that was previously impossible to build:

    tokenize_site  ->  per-stay token sequences  ->  zero_shot_predictions

Two honesty rules:

1. **A stay the tokenizer produced no sequence for gets NaN, not a guess.** NaN is
   the one value `evaluate_site` already treats as "undefined prediction": it narrows
   to defined rows once and reports the dropped count. Scoring a pad-only sequence
   instead would manufacture a near-prior prediction for a patient the model never saw.

2. **Outcomes are asked exactly as the bundle declares.** Each outcome's
   target_index/tau_bin/direction comes from the manifest's `outcome_queries`; an
   outcome with no declared query is a hard failure, because improvising the query
   parameters would score a different question than the release attests to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.bundle import Bundle
from src.eval.clif_validate import ArtifactMismatch, zero_shot_predictions


def resolve_outcome_queries(bundle: Bundle, outcome_cfgs: list[dict]) -> list[dict]:
    """Order the bundle's query parameters to match the outcome list being evaluated."""
    queries = []
    for cfg in outcome_cfgs:
        name = cfg["name"]
        query = bundle.outcome_queries.get(name)
        if query is None:
            raise ArtifactMismatch(
                f"bundle declares no zero-shot query for outcome {name!r}. Refusing "
                "to improvise target_index/tau_bin/direction — that would score a "
                "question the release never attested to."
            )
        queries.append(query)
    return queries


def sequences_by_stay(events_df) -> dict[str, dict]:
    """Per-stay model inputs from the tokenizer's events shard.

    `tokenize_site` writes one row per stay with the hard-token sequence in `token`;
    the attention mask is all-ones over the real sequence (padding is applied later,
    per batch, inside `zero_shot_predictions`).
    """
    out: dict[str, dict] = {}
    for row in events_df.iter_rows(named=True):
        tokens = list(row["token"])
        if not tokens:
            continue
        out[str(row["hosp_id"])] = {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }
    return out


def bundle_predict_fn(bundle: Bundle, model, *, data_path: str | Path,
                      episode_artifact: str | Path, outcome_cfgs: list[dict],
                      shard_dir: str | Path, site_id: str):
    """Build the `predict_fn` callable `evaluate_site` requires.

    The tokenization runs inside the returned callable, not here, so the shard is
    produced for exactly the run that consumes it and the destination check
    (`validate_artifact_destination`, via `tokenize_site`) fires on every call.
    """
    queries = resolve_outcome_queries(bundle, outcome_cfgs)
    target_indices = [q["target_index"] for q in queries]
    tau_bins = [q["tau_bin"] for q in queries]
    directions = [q["direction"] for q in queries]

    def predict_fn(labels_df):
        import polars as pl

        from src.data.tokenize import tokenize_site

        episodes = pl.read_parquet(episode_artifact)
        shard = Path(shard_dir)
        shard.mkdir(parents=True, exist_ok=True)
        tokenize_site(
            bundle.data_cfg, site_id, Path(data_path), shard,
            dict(bundle.vocab), dict(bundle.edges),
            episodes=episodes,
            vocab_manifest=bundle.vocab_manifest,
            artifact_policy=bundle.policy,
        )
        events = pl.read_parquet(shard / "events.parquet")
        sequences = sequences_by_stay(events)

        stay_ids = [str(v) for v in labels_df["hospitalization_id"].to_list()]
        defined_rows, batches = [], []
        for i, stay_id in enumerate(stay_ids):
            seq = sequences.get(stay_id)
            if seq is None:
                continue  # NaN row: undefined, reported as dropped downstream
            defined_rows.append(i)
            batches.append(seq)

        probs = np.full((len(stay_ids), len(queries)), np.nan)
        if batches:
            got = zero_shot_predictions(model, batches, target_indices, tau_bins,
                                        directions)
            probs[np.asarray(defined_rows), :] = got
        return probs

    return predict_fn


__all__ = ["bundle_predict_fn", "resolve_outcome_queries", "sequences_by_stay"]
