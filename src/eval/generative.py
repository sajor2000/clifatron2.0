"""G3 generative evals (plan 2026-09-25-001, unit G3): score generated rollouts
against the real tokenized corpus.

Metrics (perplexity lives in the trainer's validation loop; these are the
generation-side scores):
  - event_rate_calibration: Jensen-Shannon divergence (base-2, [0,1]) between
    real and generated token-rate distributions, top-k overlap, generated-only
    mass (closed-world sampling should hold it ~0), and per-group rate comparison
    (vitals/labs via configs/data.yaml target_concepts; treatments = every token
    charted by an input-only table, binned or not, per the vocab artifact's
    `concept_sources`; other binned measurements; categoricals). The 2026-09-26 overnight log's qualitative finding —
    3000-step rollouts are labs-heavy while the real stream is vitals+meds-heavy
    (0/8 top-token overlap) — is what this metric quantifies.
  - key_event_recall: of the real continuation's token concepts (and its
    categorical/treatment tokens specifically), the fraction that appear in the
    generated rollout.
  - distance_to_observed: nearest-neighbor Jaccard distance (token SETS) from
    each generated rollout to the real corpus — a first-order proxy for
    "is there any real stay that looks like this rollout"; embedding/DTW
    distance is later G3 work.
  - rollout_hygiene: length stats, <eos> termination rate, <unk> rate,
    distinct-2 (repetition).
  - rollout mortality (U9, R20; KTD10): per stay, the death-before-discharge
    frequency over N generate-until-disposition rollouts — expired / rollouts that
    reached a KNOWN disposition — with a Wilson 95% CI. Censored rollouts (token cap),
    `<eos>` without a disposition and DISCHARGE//unknown are end of observation, never
    survival: counted separately and excluded from the denominator. Across stays with
    a known observed disposition: AUROC / AUPRC vs observed `expired`, calibration
    slope / intercept (src/eval/metrics.py), the terminal-type confusion matrix
    (modal rollout terminal vs observed), the nontermination (censoring) rate, and the
    time-to-terminal MAE when rollouts carry model-estimated elapsed minutes (the
    current sampler does not — see src/model/generate.py — so it reports unavailable).
    The summary is aggregate-only; per-stay rows carry only the caller's opaque keys.

CLI (site-local; reports stay under output/ — PHI-derived, never committed):
    uv run --frozen --group dev python -m src.eval.generative \
        --sims output/intermediate_phi/sims_mps.parquet \
        --events output/intermediate_phi/mimic/events.parquet \
        --vocab output/intermediate_phi/mimic/vocab.json \
        --prompt-source head:3 --prompt-len 12 \
        --out output/intermediate_phi/gem_eval_mps.json
`--prompt-source head:N` documents the sims' prompt convention: prompts were the
first N event rows' first --prompt-len tokens (the GEM sampler CLI's prompt
file). A prefix column in the sims parquet supersedes this in G4.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import polars as pl

from src.data.segments import vocab_segments
from src.eval.clinical_plausibility import split_sequence
from src.eval.metrics import calibration_slope_intercept
from src.eval.metrics import score as discrimination
from src.model.generate import (
    STOP_CENSORED,
    STOP_EOS,
    STOP_TERMINAL,
    TIME_UNAVAILABLE_REASON,
    stay_stream,
    terminal_token_ids,
)


_SPECIALS = {"<pad>", "<bos>", "<eos>", "<unk>"}


def token_concept(token: str) -> str:
    """Concept of a fused `concept=bin` token (bare tokens are their own concept)."""
    return token.split("=", 1)[0]


GROUPS = ("vitals", "labs", "measurements", "treatments", "categoricals", "specials")
# Key events for recall: the clinically decisive non-measurement events.
KEY_GROUPS = frozenset({"treatments", "categoricals"})


def concept_groups(data_config: dict, vocab_artifact: dict | None = None) -> dict[str, str]:
    """concept -> group (KTD7: by SOURCE, not by whether the token contains `=`).

    - ``treatments``: charted by an input-only table (treatments, devices, context,
      static tokens) per the vocab artifact's `concept_sources` — so a fused
      `device_category=imv` or a binned dose `norepinephrine_mcg_kg_min=3` lands here;
    - target concepts: their configs/data.yaml source (vitals / labs);
    - ``measurements``: any other binned concept (`binning_sources`);
    - ``categoricals``: every other concept the artifact knows.

    Without an artifact only the target concepts are mapped; `_group_of` then falls
    back on the token shape for the rest."""
    groups: dict[str, str] = {}
    if vocab_artifact:
        sources = vocab_artifact.get("concept_sources") or {}
        treatment = set(sources.get("treatment_sources") or ())
        binned = set(vocab_artifact.get("binning_sources") or ())
        for concept, tables in (sources.get("tables") or {}).items():
            if treatment & set(tables):
                groups[concept] = "treatments"
            else:
                groups[concept] = "measurements" if concept in binned else "categoricals"
        for concept in binned:
            groups.setdefault(concept, "measurements")
    for spec in data_config.get("target_concepts", []):
        if groups.get(spec["name"]) != "treatments":
            groups[spec["name"]] = str(spec.get("source", "measurements"))
    return groups


def js_divergence(real_counts: Counter, gen_counts: Counter) -> float:
    """Jensen-Shannon divergence (base-2, [0,1]) between the two rate distributions."""
    keys = sorted(set(real_counts) | set(gen_counts))
    n_real = sum(real_counts.values()) or 1
    n_gen = sum(gen_counts.values()) or 1
    divergence = 0.0
    for key in keys:
        p = real_counts.get(key, 0) / n_real
        q = gen_counts.get(key, 0) / n_gen
        m = (p + q) / 2.0
        for a, b in ((p, m), (q, m)):
            if a > 0:
                divergence += 0.5 * a * math.log2(a / b)
    return divergence


def top_k_overlap(real_counts: Counter, gen_counts: Counter, k: int) -> float:
    """|real top-k ∩ gen top-k| / k — the overnight log's eyeball metric, formalized."""
    real_top = {t for t, _ in real_counts.most_common(k)}
    gen_top = {t for t, _ in gen_counts.most_common(k)}
    if not real_top:
        return 0.0
    return len(real_top & gen_top) / len(real_top)


def event_rate_calibration(real_tokens: list[str], gen_tokens: list[str],
                           groups: dict[str, str], *, top_k: tuple[int, ...] = (8, 32)) -> dict:
    real_counts, gen_counts = Counter(real_tokens), Counter(gen_tokens)
    n_real, n_gen = len(real_tokens), len(gen_tokens)
    gen_only = [t for t in gen_counts if t not in real_counts]
    group_rates: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        real_rate = sum(c for t, c in real_counts.items() if _group_of(t, groups) == group) / (n_real or 1)
        gen_rate = sum(c for t, c in gen_counts.items() if _group_of(t, groups) == group) / (n_gen or 1)
        group_rates[group] = {"real": round(real_rate, 4), "gen": round(gen_rate, 4)}
    return {
        "n_real_tokens": n_real,
        "n_gen_tokens": n_gen,
        "js_divergence": round(js_divergence(real_counts, gen_counts), 4),
        "top_k_overlap": {str(k): round(top_k_overlap(real_counts, gen_counts, k), 4) for k in top_k},
        "gen_only_tokens": len(gen_only),
        "gen_only_mass": round(sum(gen_counts[t] for t in gen_only) / (n_gen or 1), 6),
        "group_rates": group_rates,
    }


def _group_of(token: str, groups: dict[str, str]) -> str:
    """A token's group: its concept's group from `concept_groups`. A concept the map
    does not know falls back on the token shape: a numeric bin suffix reads as a
    measurement, anything else as a categorical."""
    if token in _SPECIALS:
        return "specials"
    concept = token_concept(token)
    if concept in groups:
        return groups[concept]
    _, _, suffix = token.partition("=")
    return "measurements" if suffix.isdigit() else "categoricals"


def key_event_recall(real_continuation: list[str], gen_tokens: list[str],
                     groups: dict[str, str]) -> dict:
    """Recall of the real continuation's concepts in the generated rollout —
    overall and for key events specifically (`KEY_GROUPS`: treatments, devices,
    demographics and other categoricals — the clinically decisive events, whether or
    not their token carries a dose bin)."""
    real_concepts = {token_concept(t) for t in real_continuation}
    gen_concepts = {token_concept(t) for t in gen_tokens}
    real_key = {token_concept(t) for t in real_continuation if _group_of(t, groups) in KEY_GROUPS}
    gen_key = {token_concept(t) for t in gen_tokens if _group_of(t, groups) in KEY_GROUPS}
    return {
        "n_real_concepts": len(real_concepts),
        "unigram_recall": round(len(real_concepts & gen_concepts) / len(real_concepts), 4) if real_concepts else 0.0,
        "n_real_categorical": len(real_key),
        "categorical_recall": round(len(real_key & gen_key) / len(real_key), 4) if real_key else 0.0,
    }


def distance_to_observed(gen_tokens: list[str], real_sets: list[tuple[str, frozenset]]) -> dict:
    """Nearest-neighbor Jaccard DISTANCE (1 - |A∩B|/|A∪B|) from a rollout to the
    real corpus (token sets). 0 = an identical real stay exists; 1 = fully disjoint."""
    gen_set = frozenset(token_concept(t) for t in gen_tokens)
    best_id, best_distance = None, 1.0
    for hosp_id, real_set in real_sets:
        union = len(gen_set | real_set)
        if union == 0:
            continue
        distance = 1.0 - len(gen_set & real_set) / union
        if distance < best_distance:
            best_id, best_distance = hosp_id, distance
    return {"nearest_hospitalization_id": best_id, "nearest_jaccard_distance": round(best_distance, 4)}


def distinct_n(tokens: list[str], n: int = 2) -> float:
    """Distinct-n (Li et al. 2016): unique n-grams / total n-grams — repetition rate."""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def rollout_hygiene(sequences: list[list[str]]) -> dict:
    lengths = [len(s) for s in sequences] or [0]
    all_tokens = [t for s in sequences for t in s]
    return {
        "n_rollouts": len(sequences),
        "mean_len": round(sum(lengths) / len(lengths), 1),
        "max_len": max(lengths),
        "eos_rate": round(sum(s[-1] == "<eos>" for s in sequences) / len(sequences), 4) if sequences else 0.0,
        "unk_rate": round(sum(t == "<unk>" for t in all_tokens) / (len(all_tokens) or 1), 6),
        "mean_distinct_2": round(
            sum(distinct_n(s, 2) for s in sequences) / (len(sequences) or 1), 4
        ),
    }


def _load_sims(
    path: str | Path,
    *,
    vocab: set[str] | None = None,
) -> list[dict]:
    df = pl.read_parquet(path)
    if "generated_sequence" not in df.columns and "generated_tokens" not in df.columns:
        raise ValueError(f"sims parquet has no generated token column: {path}")
    has_prompt = "prompt" in df.columns
    rows = df.sort(["hospitalization_id", "simulation_number"]).to_dicts()
    out = []
    for row in rows:
        raw_tokens = row.get("generated_tokens")
        tokens = (
            [str(token) for token in raw_tokens]
            if isinstance(raw_tokens, list)
            else split_sequence(row.get("generated_sequence") or "", vocab=vocab)
        )
        out.append({
            "hospitalization_id": str(row.get("hospitalization_id")),
            "simulation_number": int(row.get("simulation_number", 0)),
            "tokens": tokens,
            "prompt": (
                split_sequence(row.get("prompt") or "", vocab=vocab)
                if has_prompt else []
            ),
        })
    return out


def _load_real(events_path: str | Path, id_to_token: dict[int, str]) -> list[dict]:
    df = pl.read_parquet(events_path)
    rows = df.to_dicts()
    out = []
    for row in rows:
        ids = row.get("token") or []
        tokens = [id_to_token.get(int(i), f"<unk:{int(i)}>") for i in ids]
        out.append({
            "hosp_id": str(row.get("hosp_id")),
            "tokens": tokens,
        })
    return out


def evaluate(sims: list[dict], real: list[dict], groups: dict[str, str], *,
             prompt_source: tuple[int, int] | None = None) -> dict:
    """Full G3 report: calibration + hygiene + per-prompt key-event recall +
    per-rollout distance-to-observed. prompt_source=(N, K) pairs sims groups
    (in sorted order) with the first N real rows' first K tokens as prompts."""
    real_tokens = [t for row in real for t in row["tokens"]]
    gen_tokens = [t for s in sims for t in s["tokens"]]
    report = {
        "event_rate_calibration": event_rate_calibration(real_tokens, gen_tokens, groups),
        "rollout_hygiene": rollout_hygiene([s["tokens"] for s in sims]),
    }

    # Prompt pairing: prefer the sims' `prompt` provenance column (exact prefix
    # match against the real corpus); fall back to the head:N convention.
    prompt_groups: dict[tuple[str, ...], list[dict]] = {}
    for s in sims:
        prefix = tuple(s.get("prompt") or ())
        if prefix:
            prompt_groups.setdefault(prefix, []).append(s)
    recalls = []
    if prompt_groups:
        for prefix, group in sorted(prompt_groups.items()):
            match = next((r for r in real if tuple(r["tokens"][:len(prefix)]) == prefix), None)
            if match is None:
                continue  # prompt not found in the real corpus (prefix mismatch)
            continuation = match["tokens"][len(prefix):]
            recall = key_event_recall(continuation, group[0]["tokens"], groups)
            recall["hospitalization_id"] = match["hosp_id"]
            recalls.append(recall)
        if recalls:
            report["key_event_recall"] = {
                "prompt_convention": "sims `prompt` column (exact prefix match)",
                "mean_unigram_recall": round(
                    sum(r["unigram_recall"] for r in recalls) / len(recalls), 4),
                "mean_categorical_recall": round(
                    sum(r["categorical_recall"] for r in recalls) / len(recalls), 4),
                "per_prompt": recalls,
            }
    elif prompt_source is not None and sims:
        n_prompts, prompt_len = prompt_source
        for prompt_idx in range(n_prompts):
            group = [s for s in sims if s["hospitalization_id"].endswith(f"-{prompt_idx + 1}")]
            if not group or prompt_idx >= len(real):
                continue
            continuation = real[prompt_idx]["tokens"][prompt_len:]
            recall = key_event_recall(continuation, group[0]["tokens"], groups)
            recall["hospitalization_id"] = real[prompt_idx]["hosp_id"]
            recalls.append(recall)
        if recalls:
            report["key_event_recall"] = {
                "prompt_convention": f"head:{n_prompts} first {prompt_len} tokens",
                "mean_unigram_recall": round(
                    sum(r["unigram_recall"] for r in recalls) / len(recalls), 4),
                "mean_categorical_recall": round(
                    sum(r["categorical_recall"] for r in recalls) / len(recalls), 4),
                "per_prompt": recalls,
            }

    real_sets = [(row["hosp_id"], frozenset(token_concept(t) for t in row["tokens"])) for row in real]
    if sims and real_sets:
        distances = [distance_to_observed(s["tokens"], real_sets) for s in sims]
        report["distance_to_observed"] = {
            "mean_nearest_jaccard_distance": round(
                sum(d["nearest_jaccard_distance"] for d in distances) / len(distances), 4),
            "best": min(distances, key=lambda d: d["nearest_jaccard_distance"]),
            "worst": max(distances, key=lambda d: d["nearest_jaccard_distance"]),
        }
    return report


# ------------------------------------------------ rollout mortality (U9, R20)

EXPIRED = "expired"
UNKNOWN = "unknown"
NO_TERMINAL = "none"  # confusion column for a stay whose rollouts never terminated


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """Wilson score interval for k successes in n trials (95% by default); (None, None)
    when n == 0. Stays inside [0, 1] and is defined at k = 0 and k = n, unlike Wald."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _modal(counts: Counter, order: tuple[str, ...]) -> str | None:
    """Most frequent label; ties break by `order` (the disposition allowlist), then
    alphabetically — deterministic."""
    if not counts:
        return None
    rank = {label: i for i, label in enumerate(order)}
    return min(counts, key=lambda label: (-counts[label], rank.get(label, len(rank)), label))


def rollout_mortality(records: list[dict], *, dispositions: tuple[str, ...] = ()) -> dict:
    """Mortality from one stay's rollouts (`rollout_to_disposition` records).

    mortality = expired / rollouts that reached a known disposition. Censored rollouts,
    `<eos>` stops without a disposition and DISCHARGE//unknown terminals are reported
    separately and never counted as survival. `mortality` and its Wilson CI are None
    when no rollout reached a known disposition."""
    reasons = Counter(r["stop_reason"] for r in records)
    unexpected = set(reasons) - {STOP_TERMINAL, STOP_EOS, STOP_CENSORED}
    if unexpected:
        raise ValueError(f"unknown stop_reason(s): {sorted(unexpected)}")
    terminals = Counter(r["terminal_type"] for r in records
                        if r["stop_reason"] == STOP_TERMINAL)
    n_expired = terminals.get(EXPIRED, 0)
    n_unknown = terminals.get(UNKNOWN, 0)
    n_terminated = sum(terminals.values()) - n_unknown
    lo, hi = wilson_interval(n_expired, n_terminated)
    return {
        "n_rollouts": len(records),
        "n_terminated": n_terminated,
        "n_expired": n_expired,
        "n_unknown": n_unknown,
        "n_eos": reasons.get(STOP_EOS, 0),
        "n_censored": reasons.get(STOP_CENSORED, 0),
        "nontermination_rate": reasons.get(STOP_CENSORED, 0) / len(records) if records else None,
        "mortality": n_expired / n_terminated if n_terminated else None,
        "ci_low": lo,
        "ci_high": hi,
        "terminal_distribution": dict(sorted(terminals.items())),
        "modal_terminal": _modal(terminals, dispositions),
    }


def _finite_or_none(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _time_to_terminal(stays: list[dict]) -> dict:
    """MAE (minutes) between each stay's median terminal-rollout elapsed minutes and
    its observed anchor-to-discharge minutes. Censored / eos rollouts never enter the
    median. Unavailable when no rollout carries a model-estimated elapsed time."""
    errors = []
    for stay in stays:
        observed = stay.get("observed_minutes")
        elapsed = sorted(float(r["elapsed_min"]) for r in stay["rollouts"]
                         if r["stop_reason"] == STOP_TERMINAL
                         and r.get("elapsed_min") is not None)
        if observed is None or not elapsed:
            continue
        mid = len(elapsed) // 2
        median = elapsed[mid] if len(elapsed) % 2 else (elapsed[mid - 1] + elapsed[mid]) / 2
        errors.append(abs(median - float(observed)))
    if not errors:
        return {"available": False, "n": 0, "mae_min": None,
                "reason": TIME_UNAVAILABLE_REASON}
    return {"available": True, "n": len(errors), "mae_min": sum(errors) / len(errors)}


def evaluate_rollout_mortality(stays: list[dict], *, dispositions: tuple[str, ...]) -> dict:
    """R20 evaluation over stays with rollouts and an observed disposition.

    Each stay: ``{"key": opaque id, "rollouts": [records], "observed": disposition,
    "observed_minutes": optional anchor-to-discharge minutes}``. Discrimination and
    calibration use stays whose observed disposition is known (not ``unknown``) and
    whose rollouts reached a known disposition at least once. Returns
    ``{"summary": aggregate-only metrics, "per_stay": rows keyed by the caller's key}``."""
    import numpy as np

    dispositions = tuple(dispositions)
    per_stay = []
    p, y = [], []
    n_unknown_observed = n_no_termination = 0
    predicted_labels = (*dispositions, NO_TERMINAL)
    confusion = {obs: {pred: 0 for pred in predicted_labels} for obs in dispositions}
    totals = Counter()
    for stay in stays:
        observed = stay.get("observed")
        if observed not in dispositions:
            raise ValueError(f"observed disposition {observed!r} is not in the allowlist")
        result = rollout_mortality(stay["rollouts"], dispositions=dispositions)
        per_stay.append({"key": stay.get("key"), "observed": observed, **result})
        totals.update({"n_rollouts": result["n_rollouts"],
                       "n_censored": result["n_censored"], "n_eos": result["n_eos"]})
        confusion[observed][result["modal_terminal"] or NO_TERMINAL] += 1
        if observed == UNKNOWN:
            n_unknown_observed += 1
        elif result["mortality"] is None:
            n_no_termination += 1
        else:
            p.append(result["mortality"])
            y.append(int(observed == EXPIRED))
    summary = {
        "n_stays": len(stays),
        "n_evaluable": len(p),
        "n_excluded_unknown_observed": n_unknown_observed,
        "n_excluded_no_termination": n_no_termination,
        "n_rollouts": totals["n_rollouts"],
        "n_censored": totals["n_censored"],
        "n_eos": totals["n_eos"],
        "nontermination_rate": (totals["n_censored"] / totals["n_rollouts"]
                                if totals["n_rollouts"] else None),
        "observed_mortality": sum(y) / len(y) if y else None,
        "auroc": None, "auprc": None,
        "calibration_slope": None, "calibration_intercept": None,
        "terminal_confusion": {"observed_labels": list(dispositions),
                               "predicted_labels": list(predicted_labels),
                               "counts": confusion},
        "time_to_terminal": _time_to_terminal(stays),
    }
    if len(set(y)) == 2:
        p_arr, y_arr = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
        disc = discrimination(p_arr, y_arr)
        slope, intercept = calibration_slope_intercept(p_arr, y_arr)
        summary.update(auroc=_finite_or_none(disc["auroc"]),
                       auprc=_finite_or_none(disc["auprc"]),
                       calibration_slope=_finite_or_none(slope),
                       calibration_intercept=_finite_or_none(intercept))
    return {"summary": summary, "per_stay": per_stay}


def observed_gem_outcome(windows: list[dict], vocab: dict[str, int]) -> dict:
    """A stay's observed outcome from its GEM windows: the disposition of its
    `DISCHARGE//*` token and the minutes from the anchor token to it."""
    terminal_ids = terminal_token_ids(vocab)
    stream = stay_stream(windows)
    hits = [i for i, t in enumerate(stream["token"]) if t in terminal_ids]
    if len(hits) != 1:
        raise ValueError(f"a GEM stay has exactly one terminal token, found {len(hits)}")
    at = hits[0]
    if at <= stream["anchor_idx"]:
        raise ValueError("the terminal token is at or before the anchor")
    return {"disposition": terminal_ids[stream["token"][at]],
            "minutes_after_anchor": stream["pos_min"][at]
            - stream["pos_min"][stream["anchor_idx"]]}


def main(argv: list[str] | None = None) -> None:
    import yaml

    ap = argparse.ArgumentParser(description="G3 generative evals: score rollouts vs the real corpus")
    ap.add_argument("--sims", required=True, help="simulations parquet (viewer schema)")
    ap.add_argument("--events", required=True, help="tokenizer events.parquet (real corpus)")
    ap.add_argument("--vocab", required=True, help="tokenizer vocab.json")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--prompt-source", default=None,
                    help="prompt convention for key-event recall, e.g. head:3 (sims prompts = first 3 real rows)")
    ap.add_argument("--prompt-len", type=int, default=12, help="prompt prefix length in tokens")
    ap.add_argument("--out", default=None, help="site-local JSON report path (output/ tree)")
    args = ap.parse_args(argv)

    vocab_artifact = json.loads(Path(args.vocab).read_text())
    vocab_segments(vocab_artifact)  # refuses a pre-v2 vocabulary (re-tokenize)
    vocab = vocab_artifact["vocab"]
    id_to_token = {int(i): t for t, i in vocab.items()}
    data_config = yaml.safe_load(Path(args.data_config).read_text())
    groups = concept_groups(data_config, vocab_artifact)

    sims = _load_sims(args.sims, vocab=set(vocab))
    real = _load_real(args.events, id_to_token)

    prompt_source = None
    if args.prompt_source:
        head, _, n = args.prompt_source.partition(":")
        if head != "head" or not n.isdigit():
            ap.error("--prompt-source must be head:N (only convention supported)")
        prompt_source = (int(n), args.prompt_len)

    report = evaluate(sims, real, groups, prompt_source=prompt_source)
    report["inputs"] = {"sims": str(args.sims), "events": str(args.events), "vocab": str(args.vocab)}

    calibration = report["event_rate_calibration"]
    print(f"calibration: JS={calibration['js_divergence']} "
          f"top8={calibration['top_k_overlap']['8']} top32={calibration['top_k_overlap']['32']} "
          f"gen_only_mass={calibration['gen_only_mass']}")
    for group, rates in calibration["group_rates"].items():
        print(f"  {group:>13}: real={rates['real']:.3f} gen={rates['gen']:.3f}")
    print(f"hygiene: {report['rollout_hygiene']}")
    if "key_event_recall" in report:
        ker = report["key_event_recall"]
        print(f"key_event_recall: unigram={ker['mean_unigram_recall']} "
              f"categorical={ker['mean_categorical_recall']}")
    if "distance_to_observed" in report:
        dto = report["distance_to_observed"]
        print(f"distance_to_observed: mean={dto['mean_nearest_jaccard_distance']} "
              f"best={dto['best']['nearest_jaccard_distance']}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
