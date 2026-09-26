"""G3 generative evals (plan 2026-09-25-001, unit G3): score generated rollouts
against the real tokenized corpus.

Metrics (perplexity lives in the trainer's validation loop; these are the
generation-side scores):
  - event_rate_calibration: Jensen-Shannon divergence (base-2, [0,1]) between
    real and generated token-rate distributions, top-k overlap, generated-only
    mass (closed-world sampling should hold it ~0), and per-group rate comparison
    (measurements vs categoricals; vitals vs labs via configs/data.yaml
    target_concepts). The 2026-09-26 overnight log's qualitative finding —
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


_SPECIALS = {"<pad>", "<bos>", "<eos>", "<unk>"}


def token_concept(token: str) -> str:
    """Concept of a fused `concept=bin` token (bare tokens are their own concept)."""
    return token.split("=", 1)[0]


def concept_groups(data_config: dict) -> dict[str, str]:
    """concept -> group from configs/data.yaml target_concepts (vitals/labs).
    Everything else is categorical (treatments, devices, demographics, specials)."""
    groups = {}
    for spec in data_config.get("target_concepts", []):
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
    for group in ("vitals", "labs", "categoricals", "specials"):
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
    if token in _SPECIALS:
        return "specials"
    if "=" in token:
        return groups.get(token_concept(token), "measurements")
    return "categoricals"


def key_event_recall(real_continuation: list[str], gen_tokens: list[str],
                     groups: dict[str, str]) -> dict:
    """Recall of the real continuation's concepts in the generated rollout —
    overall and for categoricals specifically (treatments, devices, demographics:
    the tokens with no `=bin`, i.e. the clinically decisive non-numeric events)."""
    real_concepts = {token_concept(t) for t in real_continuation}
    gen_concepts = {token_concept(t) for t in gen_tokens}
    real_key = {token_concept(t) for t in real_continuation if _group_of(t, groups) == "categoricals"}
    gen_key = {token_concept(t) for t in gen_tokens if _group_of(t, groups) == "categoricals"}
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


def _load_sims(path: str | Path) -> list[dict]:
    df = pl.read_parquet(path)
    if "generated_sequence" not in df.columns:
        raise ValueError(f"sims parquet has no generated_sequence column: {path}")
    rows = df.sort(["hospitalization_id", "simulation_number"]).to_dicts()
    out = []
    for row in rows:
        tokens = (row.get("generated_sequence") or "").split()
        out.append({
            "hospitalization_id": str(row.get("hospitalization_id")),
            "simulation_number": int(row.get("simulation_number", 0)),
            "tokens": tokens,
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

    if prompt_source is not None and sims:
        n_prompts, prompt_len = prompt_source
        recalls = []
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

    vocab = json.loads(Path(args.vocab).read_text()).get("vocab", {})
    id_to_token = {int(i): t for t, i in vocab.items()}
    data_config = yaml.safe_load(Path(args.data_config).read_text())
    groups = concept_groups(data_config)

    sims = _load_sims(args.sims)
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
