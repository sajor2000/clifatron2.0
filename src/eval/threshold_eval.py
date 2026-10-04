"""Zero-shot threshold evaluation of one finished run (plan U8; R35, R39; KTD3, KTD13).

One deterministic evaluation set of (anchor, registered threshold, horizon) pairs per
held-out stay, labelled by the in-stream rule, and three ways of scoring it:

    head     the threshold head, zero-shot: P(crossing within the horizon) =
             `ThresholdHazardHead.cumulative_failure` at the horizon's bin. Nothing is fitted
             on the evaluation task. An objective arm that never trained the head
             (`claims.yaml objective_arms.without_threshold_head`) has no head row.
    probe    a linear probe on the frozen trunk's anchor state (KTD13), one per
             (threshold, horizon): fitted on the fit partition, weight decay chosen on the
             selection partition, temperature on the calibration partition, scored on the
             evaluation partition - the U7 probe (`probe.linear_probe_fitter`). Every arm
             gets one, so the next-token arm is scored on every registered threshold.
    rollout  sampled futures from the anchor. `src/model/generate.py` advances time on an
             IMPOSED clock (`pos_step_min`, no learned time model), so a crossing within a
             horizon cannot be dated: with that sampler every rollout row reads "not
             evaluable" with the reason and stays in the table. A sampler that dates its
             events (`clock_imposed = False`) is scored as the share of rollouts with a
             generated value bin wholly beyond the threshold within the horizon; an
             off-edge threshold straddles one of the arm's bins, so its row is not
             evaluable either.

EVALUATION SET. Anchors are event MINUTES, the stream read at the last token of that minute
(the label rule's contract): per stay, up to `anchors_per_stay` minutes before the
terminal token's minute, drawn with `random.random()` from a generator seeded by
(evaluation seed, episode key). Minutes - not token positions - are drawn, so every
tokenization arm of the same stays gets the same anchors, the same pairs and, because
labels are computed at the exact threshold value, the same labels. Every registered
`decision` and `control` threshold is queried at every horizon in
`claims.yaml evaluation.horizons_hours`, each labelled by `TargetBuilder.label_anchors`
with that horizon (the label rule's lookback and ascertainment window are unchanged).
Only `positive` (label 1) and `negative` (label 0) are scored; `prevalent`, `censored`,
`competing_event` and `not_ascertainable` are counted (`status_counts`) and never scored,
in particular never as negatives.

EDGES (KTD3). `classify_edges` reads each arm's `ThresholdGrid.edge_distance()`: a
threshold is on-edge or off-edge PER ARM. A control threshold that is on an edge in any
arm is refused, with the arm named.

RUN OUTPUTS (`write_run_outputs`, beside the run's `run_spec.json`):

    <run_dir>/threshold_eval/scores.parquet   ROW-LEVEL (governed storage only): one row
        per scored (pair, scorer) on the evaluation partition - pair_id, cluster_id (opaque
        sha256 prefixes of the stay and anchor minute; they align runs, they are never
        reported), scorer, threshold, kind, concept, value, direction, horizon_hours,
        label, prob.
    <run_dir>/threshold_eval/summary.json     the run's binding (vocabulary and segments
        hashes), objective arm, evaluated partition, horizons, edge table, bin counts per
        concept, label-status counts per threshold and horizon, and one status row per
        scorer x threshold x horizon (evaluable / not evaluable + reason). No identifier.

CLI (one run; `src.eval.claims_report` reads the outputs of many):

    uv run python -m src.eval.threshold_eval --run-dir <run> --checkpoint <ckpt.pt> \
        --vocab <vocab.json> --shards <gem_events.parquet> [--final-evaluation]

A continuous-fused run also needs `--value-stats` (the run's train-partition
value_stats.json, bound to its vocabulary): its trunk reads each event's normalized value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.special import expit

from src.data.segments import vocab_segments
from src.data.targets import InStreamTargets, TargetBuilder
from src.data.threshold_grid import ThresholdGrid, load_thresholds
from src.data.tokenize_continuous import REPRESENTATION as CONTINUOUS_FUSED
from src.data.tokenize_continuous import (
    ContinuousThresholdGrid,
    normalize_value,
    primary_segments,
)
from src.eval import metrics as M
from src.eval import schema as _schema
from src.eval.clif_tasks import NOT_EVALUABLE
from src.eval.probe import DEFAULT_PROBE, freeze_trunk, linear_probe_fitter

ROOT = Path(__file__).parents[2]
CLAIMS_PATH = ROOT / "configs/claims.yaml"
SUMMARY_VERSION = 1
EVAL_DIR = "threshold_eval"
SCORES_FILE = "scores.parquet"
SUMMARY_FILE = "summary.json"

EVALUATED_KINDS = ("decision", "control")
SCORERS = ("head", "probe", "rollout")
SCORABLE = ("positive", "negative")
EVALUABLE = "evaluable"
# Orientation of each supported primary metric: +1 higher is better, -1 lower is better.
METRIC_ORIENTATION = {"auroc": 1, "auprc": 1, "ici": -1, "ece": -1}
DISCRIMINATION_METRICS = ("auroc", "auprc")
CALIBRATION_METRICS = ("ici", "ece")
ROLE_KEYS = ("fit", "selection", "calibration", "evaluation")
SCORE_COLUMNS = ("pair_id", "cluster_id", "scorer", "threshold", "kind", "concept", "value",
                 "direction", "horizon_hours", "label", "prob")


class ThresholdEvalError(ValueError):
    """A threshold evaluation cannot be run or read safely."""


# ------------------------------------------------------------------------------ config

def load_claims_config(path: str | Path = CLAIMS_PATH) -> dict[str, Any]:
    """Read and validate `configs/claims.yaml` (the registered metrics, correction,
    bootstrap, seeds, evaluation settings and decision-rule parameters)."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ThresholdEvalError(f"{path} is not a claims config")
    if any(key in raw for key in ("thresholds", "decision", "control")):
        raise ThresholdEvalError(
            "thresholds are registered in configs/thresholds.yaml only, never in the "
            "claims config (KTD3)")
    cfg = dict(raw)
    metrics = raw.get("primary_metrics") or {}
    for group, allowed in (("discrimination", DISCRIMINATION_METRICS),
                           ("calibration", CALIBRATION_METRICS)):
        names = metrics.get(group)
        if not isinstance(names, list) or not names or any(n not in allowed for n in names):
            raise ThresholdEvalError(
                f"primary_metrics.{group} must be a non-empty list of {allowed}; unknown "
                f"metric in {names!r}")
    cfg["metric_names"] = tuple(metrics["discrimination"]) + tuple(metrics["calibration"])
    mult = raw.get("multiplicity") or {}
    if mult.get("method") != "benjamini_hochberg" or mult.get("family") != "per_claim":
        raise ThresholdEvalError(
            "multiplicity must be method benjamini_hochberg, family per_claim")
    alpha = mult.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ThresholdEvalError(f"multiplicity.alpha must be in (0, 1), got {alpha!r}")
    boot = raw.get("bootstrap") or {}
    if not isinstance(boot.get("n_resamples"), int) or boot["n_resamples"] < 1 \
            or not isinstance(boot.get("seed"), int) \
            or not 0 < float(boot.get("confidence", 0)) < 1:
        raise ThresholdEvalError(
            "bootstrap needs an integer n_resamples >= 1, an integer seed and a "
            "confidence in (0, 1)")
    seeds = raw.get("min_seeds_per_arm")
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 3:
        raise ThresholdEvalError("min_seeds_per_arm must be an integer of at least 3 (R34)")
    evaluation = raw.get("evaluation") or {}
    horizons = evaluation.get("horizons_hours")
    if not isinstance(horizons, list) or not horizons \
            or any(isinstance(h, bool) or not isinstance(h, (int, float)) or h <= 0
                   for h in horizons):
        raise ThresholdEvalError("evaluation.horizons_hours must be positive numbers")
    if int(evaluation.get("anchors_per_stay", 0)) < 1:
        raise ThresholdEvalError("evaluation.anchors_per_stay must be at least 1")
    roles = raw.get("partitions") or {}
    if set(roles) != set(ROLE_KEYS):
        raise ThresholdEvalError(f"partitions must name exactly {', '.join(ROLE_KEYS)}")
    arms = raw.get("objective_arms") or {}
    if not isinstance(arms.get("combined"), str) or not isinstance(arms.get("next_token"), str) \
            or arms["next_token"] not in (arms.get("without_threshold_head") or ()):
        raise ThresholdEvalError(
            "objective_arms must name combined and next_token, and list the next-token "
            "arm under without_threshold_head")
    claim_1 = raw.get("claim_1") or {}
    if not isinstance(claim_1.get("attribution_in_rule"), bool):
        raise ThresholdEvalError("claim_1.attribution_in_rule must be true or false")
    if not claim_1.get("comparator_arms") or not isinstance(claim_1.get("primary_arm"), str):
        raise ThresholdEvalError("claim_1 must name a primary_arm and comparator_arms")
    if not isinstance((raw.get("claim_2") or {}).get("tokenization_arm"), str):
        raise ThresholdEvalError("claim_2 must name its tokenization_arm")
    return cfg


def has_threshold_head(objective_arm: str, claims: Mapping[str, Any]) -> bool:
    """Whether this objective arm trained the threshold head (KTD13)."""
    return objective_arm not in claims["objective_arms"]["without_threshold_head"]


# ------------------------------------------------------------------------------ edges

def threshold_key(kind: str, concept: str, value: float, direction: str) -> str:
    """The stable name of one registered threshold in every output."""
    return f"{kind}:{concept}:{float(value):g}:{direction}"


def edge_rows(grid: ThresholdGrid, *, arm: str | None = None) -> dict[str, dict]:
    """This arm's edge-distance rows for the evaluated kinds, keyed by `threshold_key`.
    Refuses a control threshold that sits on one of this arm's edges (KTD3)."""
    rows = {}
    for row in grid.edge_distance():
        if row["kind"] not in EVALUATED_KINDS:
            continue
        if row["kind"] == "control" and row["on_edge"]:
            raise ThresholdEvalError(
                f"control {row['concept']} {row['value']:g} is on a bin edge in arm "
                f"{arm or 'of this vocabulary'!r}: a control threshold must be off-edge in "
                "every arm (KTD3); remove it from configs/thresholds.yaml")
        key = threshold_key(row["kind"], row["concept"], row["value"], row["direction"])
        rows[key] = {field: row[field] for field in (
            "kind", "concept", "value", "direction", "on_edge", "distance", "nearest_edge",
            "threshold_bin")}
    return rows


def classify_edges(grids: Mapping[str, ThresholdGrid]) -> dict[str, dict[str, dict]]:
    """``{arm: {threshold_key: edge row}}`` for every arm; a control on an edge in any
    arm is refused with that arm named."""
    return {arm: edge_rows(grid, arm=arm) for arm, grid in grids.items()}


# ---------------------------------------------------------------------- evaluation set

def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _builder(grid: ThresholdGrid, horizon_hours: float, rule: Mapping) -> TargetBuilder:
    """A label-only `gem_tte` builder at one horizon (the label rule's other windows)."""
    return TargetBuilder(
        vocab_size=max(grid.terminal_tokens | set(grid.token_target)) + 1,
        n_time_bins=1, horizon_hours=float(horizon_hours), value_stats={}, mode="gem_tte",
        in_stream=InStreamTargets(
            grid=grid, anchors_per_window=1, queries_per_anchor=1,
            baseline_lookback_hours=rule["baseline_lookback_hours"],
            required_measurement_within_hours_of_horizon=rule[
                "required_measurement_within_hours_of_horizon"]))


def evaluation_set(streams: Sequence[Mapping], grid: ThresholdGrid,
                   thresholds: Mapping[str, Any], *, horizons_hours: Sequence[float],
                   anchors_per_stay: int, seed: int) -> dict[str, list[dict]]:
    """The (anchor, threshold, horizon) pairs of `streams` (full stays: `episode_key`,
    `partition`, `token`, `pos_min`, `value`), labelled at the exact threshold values.

    Returns ``{"anchors": [...], "pairs": [...]}``; a pair's `anchor` indexes `anchors`,
    and its `label` is 1 / 0 for `positive` / `negative` and None otherwise."""
    builders = {float(h): _builder(grid, h, thresholds["label_rule"]) for h in horizons_hours}
    queries = [(kind, query) for kind in EVALUATED_KINDS for query in grid.registered(kind)]
    anchors: list[dict] = []
    pairs: list[dict] = []
    for stay, stream in enumerate(streams):
        key = str(stream["episode_key"])
        token = [int(t) for t in stream["token"]]
        pos = [int(p) for p in stream["pos_min"]]
        end = next((i for i, t in enumerate(token) if t in grid.terminal_tokens), len(token))
        terminal_minute = pos[end] if end < len(token) else None
        last_of_minute: dict[int, int] = {}
        for i in range(end):
            last_of_minute[pos[i]] = i
        minutes = sorted(m for m in last_of_minute if m != terminal_minute)
        rng = random.Random(int.from_bytes(_digest(f"{seed}:{key}")[:8], "big"))
        take = min(int(anchors_per_stay), len(minutes))
        for slot in range(take):                             # partial Fisher-Yates
            pick = slot + int(rng.random() * (len(minutes) - slot))
            minutes[slot], minutes[pick] = minutes[pick], minutes[slot]
        cluster_id = _digest(f"stay:{key}").hex()[:16]
        chosen = sorted(minutes[:take])
        # One stream index per (stay, horizon): every (anchor, query) labelled against it.
        labels = {horizon: iter(builder.label_anchors(
                      stream, [(last_of_minute[minute], query) for minute in chosen
                               for _, query in queries]))
                  for horizon, builder in builders.items()}
        for minute in chosen:
            anchor_idx = last_of_minute[minute]
            anchor = len(anchors)
            pair_id = _digest(f"{key}:{minute}").hex()[:16]
            anchors.append({"stay": stay, "anchor_idx": anchor_idx, "anchor_min": minute,
                            "partition": stream.get("partition"), "pair_id": pair_id,
                            "cluster_id": cluster_id})
            per_query = {horizon: [next(labels[horizon]) for _ in queries]
                         for horizon in builders}
            for q, (kind, query) in enumerate(queries):
                for horizon in builders:
                    label = per_query[horizon][q]
                    status = label["status"]
                    pairs.append({
                        "anchor": anchor, "pair_id": pair_id, "cluster_id": cluster_id,
                        "partition": stream.get("partition"), "kind": kind, "query": query,
                        "threshold": threshold_key(kind, query.concept, query.value,
                                                   query.direction),
                        "concept": query.concept, "value": query.value,
                        "direction": query.direction, "horizon_hours": horizon,
                        "status": status, "minutes": label["minutes"],
                        "label": int(status == "positive") if status in SCORABLE else None,
                    })
    return {"anchors": anchors, "pairs": pairs}


def _horizon_label(horizon: float) -> str:
    return f"{float(horizon):g}"


def status_counts(pairs: Sequence[Mapping], *, partition: str) -> dict[str, dict]:
    """``{threshold: {horizon: {status: count}}}`` over one partition's pairs: every
    label state, scorable or not (exact counts - band them before any release)."""
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for pair in pairs:
        if pair["partition"] != partition:
            continue
        cell = counts.setdefault(pair["threshold"], {}).setdefault(
            _horizon_label(pair["horizon_hours"]), {})
        cell[pair["status"]] = cell.get(pair["status"], 0) + 1
    return counts


# ------------------------------------------------------------------------------ scoring

def default_device() -> str:
    """`--device` default: the GPU when there is one (the L40 node), else the CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def anchor_states(model, streams: Sequence[Mapping], anchors: Sequence[Mapping], *,
                  device: str = "cpu", context_tokens: int | None = None) -> np.ndarray:
    """The frozen trunk's hidden state at every anchor, ``[anchors, d]``.

    Reads `model.enc` only: called as the training forward calls it (soft bins when the
    stream carries `soft_token` / `soft_weight`, the value channel for a continuous-fused
    encoder). The trunk is causal, so one forward over a stay's prefix gives every
    anchor's state; a prefix longer than `context_tokens` is read per anchor over its
    last `context_tokens` tokens."""
    import torch

    enc = freeze_trunk(model.enc).to(device)
    value_channel = getattr(enc, "uses_value_channel", False)
    states = np.zeros((len(anchors), enc.d_model), dtype=np.float32)
    by_stay: dict[int, list[int]] = {}
    for index, anchor in enumerate(anchors):
        by_stay.setdefault(int(anchor["stay"]), []).append(index)

    def tensor(values, dtype):
        return torch.as_tensor(values, dtype=dtype, device=device).unsqueeze(0)

    with torch.no_grad():
        for stay, members in by_stay.items():
            stream = streams[stay]
            stop = max(int(anchors[i]["anchor_idx"]) for i in members) + 1
            if context_tokens is None or stop <= context_tokens:
                windows = [(0, stop, members)]
            else:
                windows = [(max(0, int(anchors[i]["anchor_idx"]) + 1 - context_tokens),
                            int(anchors[i]["anchor_idx"]) + 1, [i]) for i in members]
            for lo, hi, group in windows:
                pos = tensor(stream["pos_min"][lo:hi], torch.long)
                if stream.get("soft_token") and stream.get("soft_weight"):
                    H = enc(tensor(stream["soft_token"][lo:hi], torch.long), pos,
                            tensor(stream["soft_weight"][lo:hi], torch.float32))
                elif value_channel:
                    if stream.get("input_value") is None:
                        raise ThresholdEvalError(
                            "a continuous-fused trunk needs each stream's input_value "
                            "channel (and input_value_mask)")
                    H = enc(tensor(stream["token"][lo:hi], torch.long), pos,
                            continuous_value=tensor(stream["input_value"][lo:hi],
                                                    torch.float32),
                            continuous_value_mask=tensor(stream["input_value_mask"][lo:hi],
                                                         torch.bool))
                else:
                    H = enc(tensor(stream["token"][lo:hi], torch.long), pos)
                for i in group:
                    states[i] = H[0, int(anchors[i]["anchor_idx"]) - lo].float().cpu().numpy()
    return states


def horizon_bin(model, horizon_hours: float) -> int:
    """How many of the threshold head's time bins a horizon spans; refused unless the
    horizon is a bin boundary of the head's own grid (KTD4)."""
    n_bins, head_hours = int(model.th.n_bins), float(model.th_horizon_hours)
    spans = float(horizon_hours) * n_bins / head_hours
    if abs(spans - round(spans)) > 1e-9 or not 1 <= round(spans) <= n_bins:
        raise ThresholdEvalError(
            f"horizon {horizon_hours:g} h is not a bin boundary of the threshold head's "
            f"time grid ({n_bins} bins over {head_hours:g} h)")
    return int(round(spans))


def head_probabilities(model, states: np.ndarray, pairs: Sequence[Mapping], *,
                       device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    """Zero-shot P(crossing within the pair's horizon) from the threshold head:
    `cumulative_failure` at the horizon's last bin. No fitting."""
    import torch

    head = model.th.to(device).eval()
    out = np.zeros(len(pairs), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            h = torch.as_tensor(states[[p["anchor"] for p in chunk]], device=device)
            target = torch.tensor([p["query"].target_idx for p in chunk], device=device)
            tau = torch.tensor([p["query"].threshold_bin for p in chunk], device=device)
            direction = torch.tensor([p["query"].direction_id for p in chunk], device=device)
            cf = head.cumulative_failure(h, target, tau, direction)       # [N, bins]
            column = torch.tensor([horizon_bin(model, p["horizon_hours"]) - 1 for p in chunk],
                                  device=device)
            out[start:start + len(chunk)] = cf.gather(1, column[:, None])[:, 0].double().cpu().numpy()
    return out


def _status_row(scorer: str, pair: Mapping, status: str, reason: str | None) -> dict:
    return {"scorer": scorer, "threshold": pair["threshold"], "kind": pair["kind"],
            "concept": pair["concept"], "value": float(pair["value"]),
            "direction": pair["direction"], "horizon_hours": float(pair["horizon_hours"]),
            "status": status, "reason": reason}


def _score_row(scorer: str, pair: Mapping, prob: float) -> dict:
    return {"pair_id": pair["pair_id"], "cluster_id": pair["cluster_id"], "scorer": scorer,
            "threshold": pair["threshold"], "kind": pair["kind"], "concept": pair["concept"],
            "value": float(pair["value"]), "direction": pair["direction"],
            "horizon_hours": float(pair["horizon_hours"]), "label": int(pair["label"]),
            "prob": float(prob)}


def _cells(pairs: Sequence[Mapping]) -> dict[tuple[str, float], list[Mapping]]:
    cells: dict[tuple[str, float], list[Mapping]] = {}
    for pair in pairs:
        cells.setdefault((pair["threshold"], pair["horizon_hours"]), []).append(pair)
    return cells


def score_head(model, states: np.ndarray, pairs: Sequence[Mapping], *,
               evaluation: str, device: str = "cpu") -> tuple[list[dict], list[dict]]:
    """Head scores on the evaluation partition's scorable pairs, plus status rows."""
    scored = [p for p in pairs if p["partition"] == evaluation and p["label"] is not None]
    probs = head_probabilities(model, states, scored, device=device) if scored else []
    scores = [_score_row("head", pair, prob) for pair, prob in zip(scored, probs)]
    rows = []
    for (_, _), cell in _cells(pairs).items():
        n = sum(1 for p in cell if p["partition"] == evaluation and p["label"] is not None)
        rows.append(_status_row("head", cell[0], EVALUABLE if n else NOT_EVALUABLE,
                                None if n else f"{evaluation} partition: no scorable pair"))
    return scores, rows


def score_probe(states: np.ndarray, pairs: Sequence[Mapping], roles: Mapping[str, str], *,
                site: str, cfg: Mapping | None = None,
                seed: int = 0) -> tuple[list[dict], list[dict]]:
    """A frozen-trunk linear probe per (threshold, horizon) (KTD13; the U7 probe).

    Fitted on `roles["fit"]`, weight decay chosen on `roles["selection"]`, temperature
    fitted on `roles["calibration"]` (`metrics.fit_temperature`), scored on
    `roles["evaluation"]`. Only scorable pairs enter any partition."""
    scores, rows = [], []
    for (_, _), cell in _cells(pairs).items():
        scorable = [p for p in cell if p["label"] is not None]

        def role(name: str) -> tuple[np.ndarray, np.ndarray, list]:
            chosen = [p for p in scorable if p["partition"] == roles[name]]
            X = states[[p["anchor"] for p in chosen]] if chosen else np.zeros((0, states.shape[1]))
            return X, np.array([p["label"] for p in chosen], dtype=int), chosen

        X_fit, y_fit, _ = role("fit")
        X_sel, y_sel, _ = role("selection")
        X_cal, y_cal, _ = role("calibration")
        X_eval, _, evaluated = role("evaluation")
        if not evaluated:
            rows.append(_status_row("probe", cell[0], NOT_EVALUABLE,
                                    f"{roles['evaluation']} partition: no scorable pair"))
            continue
        if len(np.unique(y_fit)) < 2:
            rows.append(_status_row("probe", cell[0], NOT_EVALUABLE,
                                    f"{roles['fit']} partition: outcome has a single "
                                    "class, nothing to fit"))
            continue
        selectable = len(np.unique(y_sel)) > 1
        fit = linear_probe_fitter(dict(cfg or DEFAULT_PROBE), seed=seed)
        logits_fn, _ = fit(X_fit, y_fit, X_sel if selectable else None,
                           y_sel if selectable else None)
        temperature = 1.0
        if len(np.unique(y_cal)) > 1:
            temperature = float(M.fit_temperature(
                logits_fn(X_cal), y_cal, partition=f"{site}:{roles['calibration']}"))
        probs = expit(np.asarray(logits_fn(X_eval), dtype=float) / temperature)
        scores.extend(_score_row("probe", pair, prob) for pair, prob in zip(evaluated, probs))
        rows.append(_status_row("probe", cell[0], EVALUABLE, None))
    return scores, rows


class ImposedClockSampler:
    """The rollout sampler of `src/model/generate.py`: it advances `pos_step_min` per
    generated token, a clock it imposes rather than predicts, so it cannot date a
    crossing within a horizon. Never called: every row it would score is not evaluable."""

    clock_imposed = True

    @property
    def reason(self) -> str:
        from src.model.generate import TIME_UNAVAILABLE_REASON

        return (f"rollouts from src/model/generate.py: {TIME_UNAVAILABLE_REASON}; a "
                "crossing within a horizon cannot be dated on that clock")


def score_rollouts(sampler, streams: Sequence[Mapping], anchors: Sequence[Mapping],
                   pairs: Sequence[Mapping], edges: Mapping[str, Mapping], *,
                   vocab: Mapping[str, int] | None, terminal_tokens, evaluation: str,
                   n_rollouts: int, seed: int) -> tuple[list[dict], list[dict]]:
    """The rollout comparator. `sampler(prompt_ids, prompt_pos, *, n, seed)` returns `n`
    records ``{"token_ids": [...], "minutes": [...]}`` (minutes since the anchor per
    generated token, None when the sampler cannot date them); `sampler.clock_imposed`
    says so up front. A pair's probability is the share of its anchor's rollouts with a
    generated bin of the concept wholly on the event side of the threshold within the
    horizon (generation stops at a terminal token)."""
    cells = _cells(pairs)
    if sampler.clock_imposed:
        return [], [_status_row("rollout", cell[0], NOT_EVALUABLE, sampler.reason)
                    for cell in cells.values()]
    if vocab is None:
        raise ThresholdEvalError("a dating rollout sampler needs the arm's vocabulary")
    token_bin = {}
    for name, token in vocab.items():
        concept, sep, b = name.partition("=")
        if sep and b.isdigit():
            token_bin[int(token)] = (concept, int(b))
    rollouts: dict[int, list[dict]] = {}
    scores, rows = [], []
    for (key, horizon), cell in cells.items():
        if not edges[key]["on_edge"]:
            rows.append(_status_row(
                "rollout", cell[0], NOT_EVALUABLE,
                "off-edge in this arm: the threshold straddles one of its value bins, so a "
                "generated bin cannot say which side of the threshold the value fell"))
            continue
        scored = [p for p in cell if p["partition"] == evaluation and p["label"] is not None]
        for pair in scored:
            anchor = anchors[pair["anchor"]]
            if pair["anchor"] not in rollouts:
                stream = streams[anchor["stay"]]
                cut = int(anchor["anchor_idx"]) + 1
                rollouts[pair["anchor"]] = sampler(
                    list(stream["token"][:cut]), list(stream["pos_min"][:cut]),
                    n=int(n_rollouts),
                    seed=int.from_bytes(_digest(f"{seed}:{anchor['pair_id']}")[:8], "big"))
            query, limit = pair["query"], float(horizon) * 60
            crossed = 0
            for record in rollouts[pair["anchor"]]:
                if record.get("minutes") is None:
                    raise ThresholdEvalError("a dating sampler returned an undated rollout")
                for token, minute in zip(record["token_ids"], record["minutes"]):
                    if int(token) in terminal_tokens or minute > limit:
                        break
                    concept, b = token_bin.get(int(token), (None, None))
                    if concept == query.concept and (
                            b <= query.threshold_bin if query.direction == "below"
                            else b >= query.threshold_bin):
                        crossed += 1
                        break
            scores.append(_score_row("rollout", pair,
                                     crossed / max(1, len(rollouts[pair["anchor"]]))))
        rows.append(_status_row("rollout", cell[0], EVALUABLE if scored else NOT_EVALUABLE,
                                None if scored else f"{evaluation} partition: no scorable pair"))
    return scores, rows


# ---------------------------------------------------------------------- one run, end to end

def evaluate_run(model, streams: Sequence[Mapping], vocab_blob: Mapping,
                 thresholds: Mapping[str, Any], claims: Mapping[str, Any], *,
                 target_concepts: Sequence[Mapping], objective_arm: str, site: str,
                 roles: Mapping[str, str] | None = None, device: str = "cpu",
                 sampler=None, probe_cfg: Mapping | None = None, arm: str | None = None,
                 context_tokens: int | None = None) -> dict[str, Any]:
    """Score one finished run on the registered thresholds: ``{"scores": [row-level
    score dicts], "summary": aggregate run summary}`` (module docstring)."""
    roles = dict(roles or claims["partitions"])
    evaluation = claims["evaluation"]
    horizons = [float(h) for h in evaluation["horizons_hours"]]
    label_horizon = float(thresholds["label_rule"]["horizon_hours"])
    if label_horizon not in horizons:
        raise ThresholdEvalError(
            f"evaluation.horizons_hours must include the label rule's horizon "
            f"({label_horizon:g} h): it is the zero-shot threshold evaluation")
    for horizon in horizons:
        horizon_bin(model, horizon)
    continuous = vocab_blob.get("representation") == CONTINUOUS_FUSED
    # A continuous-fused vocabulary is edgeless: thresholds sit on its primary segments.
    grid = (ContinuousThresholdGrid if continuous else ThresholdGrid)(
        vocab_blob, target_concepts, thresholds)
    edges = edge_rows(grid, arm=arm)
    built = evaluation_set(streams, grid, thresholds, horizons_hours=horizons,
                           anchors_per_stay=int(evaluation["anchors_per_stay"]),
                           seed=int(evaluation["seed"]))
    anchors, pairs = built["anchors"], built["pairs"]
    states = anchor_states(model, streams, anchors, device=device,
                           context_tokens=context_tokens)
    scores, rows = [], []
    if has_threshold_head(objective_arm, claims):
        head_scores, head_rows = score_head(model, states, pairs,
                                            evaluation=roles["evaluation"], device=device)
        scores += head_scores
        rows += head_rows
    probe_scores, probe_rows = score_probe(states, pairs, roles, site=site, cfg=probe_cfg,
                                           seed=int(evaluation["seed"]))
    rollout_scores, rollout_rows = score_rollouts(
        sampler or ImposedClockSampler(), streams, anchors, pairs, edges,
        vocab=vocab_blob.get("vocab"), terminal_tokens=grid.terminal_tokens,
        evaluation=roles["evaluation"], n_rollouts=int(evaluation["n_rollouts"]),
        seed=int(evaluation["seed"]))
    scores += probe_scores + rollout_scores
    rows += probe_rows + rollout_rows
    segments = primary_segments(vocab_blob) if continuous else vocab_segments(vocab_blob)
    summary = {
        "version": SUMMARY_VERSION,
        "vocabulary": grid.binding["vocabulary"],
        "numeric_edges": grid.binding["numeric_edges"],
        "objective_arm": objective_arm,
        "evaluation_partition": roles["evaluation"],
        "horizons_hours": horizons,
        "label_horizon_hours": label_horizon,
        "edge_table": list(edges.values()),
        "bin_counts": {concept: len(segments[concept]) for concept in grid.binned},
        "status_counts": status_counts(pairs, partition=roles["evaluation"]),
        "rows": rows,
        "context_tokens": context_tokens,
    }
    return {"scores": scores, "summary": summary}


def cell_metrics(scores, metric_names: Sequence[str] = ("auroc", "auprc", "ici")) -> list[dict]:
    """Per (scorer, threshold, horizon): the primary metrics on the scored pairs, or a
    suppression status (`schema.suppress_cell`: total, positives and negatives must all
    clear the minimum cell size)."""
    rows = scores.to_dicts() if hasattr(scores, "to_dicts") else list(scores)
    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        cells.setdefault((row["scorer"], row["threshold"], float(row["horizon_hours"])),
                         []).append(row)
    out = []
    for (scorer, key, horizon), cell in cells.items():
        y = np.array([r["label"] for r in cell], dtype=int)
        p = np.array([r["prob"] for r in cell], dtype=float)
        status, reason = _schema.suppress_cell(len(y), int(y.sum()))
        entry = {"scorer": scorer, "threshold": key, "horizon_hours": horizon,
                 "status": status, "reason": reason}
        if status == _schema.EVALUABLE:
            entry.update({"n": int(len(y)),
                          "prevalence": _schema.round_prevalence(float(y.mean()))})
            entry.update(metric_values(metric_names, p, y))
        else:
            entry["n_band"] = _schema.n_band(len(y))
        out.append(entry)
    return out


def metric_values(names: Sequence[str], p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Primary metrics (`metrics.score` once for AUROC and AUPRC,
    `metrics.integrated_calibration_index`, `metrics.expected_calibration_error`); NaN
    when the labels have a single class."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    unknown = set(names) - set(METRIC_ORIENTATION)
    if unknown:
        raise ThresholdEvalError(f"unknown metric(s) {sorted(unknown)}")
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {name: float("nan") for name in names}
    out: dict[str, float] = {}
    if set(names) & set(DISCRIMINATION_METRICS):
        scored = M.score(p, y)
        out.update({name: float(scored[name]) for name in names
                    if name in DISCRIMINATION_METRICS})
    if "ici" in names:
        out["ici"] = float(M.integrated_calibration_index(p, y))
    if "ece" in names:
        out["ece"] = float(M.expected_calibration_error(p, y))
    return out


def metric_value(name: str, p: np.ndarray, y: np.ndarray) -> float:
    """One primary metric (`metric_values`)."""
    return metric_values((name,), p, y)[name]


# ------------------------------------------------------------------------------ outputs

def write_run_outputs(run_dir: str | Path, result: Mapping[str, Any]) -> Path:
    """Write `scores.parquet` (row-level; governed storage only) and `summary.json`."""
    import polars as pl

    out = Path(run_dir) / EVAL_DIR
    out.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        [{column: row[column] for column in SCORE_COLUMNS} for row in result["scores"]],
        schema={"pair_id": pl.String, "cluster_id": pl.String, "scorer": pl.String,
                "threshold": pl.String, "kind": pl.String, "concept": pl.String,
                "value": pl.Float64, "direction": pl.String, "horizon_hours": pl.Float64,
                "label": pl.Int8, "prob": pl.Float64})
    frame.write_parquet(out / SCORES_FILE)
    (out / SUMMARY_FILE).write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False) + "\n")
    return out


def read_run_outputs(run_dir: str | Path) -> dict[str, Any]:
    """``{"scores": polars frame, "summary": dict}`` of one run (`write_run_outputs`)."""
    import polars as pl

    out = Path(run_dir) / EVAL_DIR
    if not (out / SCORES_FILE).is_file() or not (out / SUMMARY_FILE).is_file():
        raise ThresholdEvalError(
            f"run has no threshold evaluation ({EVAL_DIR}/{SCORES_FILE} and "
            f"{EVAL_DIR}/{SUMMARY_FILE}); run `python -m src.eval.threshold_eval` first")
    return {"scores": pl.read_parquet(out / SCORES_FILE),
            "summary": json.loads((out / SUMMARY_FILE).read_text())}


def stay_streams(frame, *, value_stats: Mapping[int, tuple[float, float]] | None = None
                 ) -> list[dict]:
    """Full stays from GEM window rows (`gem_events.parquet`): windows concatenated in
    `continuation_index` order per episode key, with values, partition and soft bins.

    With `value_stats` (a continuous-fused run's frozen per-token stats), each stream
    also carries the value channel its trunk reads, normalized as in training
    (`dataset.ModelDataset(value_channel=True)`): `input_value` / `input_value_mask`."""
    key_column = "episode_key" if "episode_key" in frame.columns else "hosp_id"
    order = [key_column] + (["continuation_index"] if "continuation_index" in frame.columns
                            else [])
    streams: dict[str, dict] = {}
    for row in frame.sort(order).iter_rows(named=True):
        key = str(row[key_column])
        stream = streams.setdefault(key, {
            "episode_key": key, "partition": row.get("partition"), "token": [],
            "pos_min": [], "value": [], "soft_token": [], "soft_weight": []})
        if row.get("partition") != stream["partition"]:
            raise ThresholdEvalError("the windows of a stay disagree on its partition")
        n = len(row["token"])
        stream["token"] += list(row["token"])
        stream["pos_min"] += list(row["pos_min"])
        stream["value"] += list(row.get("value") or [None] * n)
        if row.get("soft_token") is not None and stream["soft_token"] is not None:
            stream["soft_token"] += list(row["soft_token"])
            stream["soft_weight"] += list(row["soft_weight"])
        else:
            stream["soft_token"] = stream["soft_weight"] = None
    out = list(streams.values())
    if value_stats is not None:
        for stream in out:
            channel = [normalize_value(value, token, value_stats)
                       for token, value in zip(stream["token"], stream["value"])]
            stream["input_value"] = [value for value, _ in channel]
            stream["input_value_mask"] = [mask for _, mask in channel]
    return out


# ---------------------------------------------------------------------------------- CLI

def context_tokens(max_context: int | None, mcfg: Mapping) -> int | None:
    """The history length each anchor is read over: `--max-context`, else the trunk's
    `max_tokens`. Recorded in the run summary (`context_tokens`)."""
    return int(max_context) if max_context is not None else mcfg["trunk"].get("max_tokens")


def run_value_stats(path, vocab_blob: Mapping) -> dict[int, tuple[float, float]] | None:
    """A continuous-fused run's value stats, bound as in training (`pretrain`): the
    vocabulary and segments hashes of `vocab_blob`, fit on `train`. Refused when missing
    for a continuous-fused run; not read for the other arms."""
    if vocab_blob.get("representation") != CONTINUOUS_FUSED:
        return None
    if path is None:
        raise SystemExit("a continuous-fused run needs --value-stats (its train-partition "
                         "value_stats.json): the trunk reads each event's normalized value")
    from src.data.segments import artifact_binding
    from src.data.value_stats import load_value_stats

    binding = artifact_binding(vocab_blob)
    return load_value_stats(path, expected_vocab_hash=binding["vocabulary"],
                            expected_segments_hash=binding["numeric_edges"],
                            expected_fit_partition="train")


def load_model(checkpoint, vocab_blob: Mapping, mcfg: Mapping, *, n_targets: int):
    """The run's model, sized from its checkpoint manifest (embedding rows = the
    vocabulary's max id + 1, the recorded trunk) and bound to `vocab_blob`."""
    from src.train.pretrain import load_model_from_checkpoint

    return load_model_from_checkpoint(checkpoint, mcfg, n_targets, vocab_blob)


def main(argv: list[str] | None = None) -> None:
    import polars as pl
    import torch

    from src.data.segments import load_vocab_blob
    from src.eval.claims_report import load_run_spec

    ap = argparse.ArgumentParser(
        description="Zero-shot threshold evaluation of one finished run (aggregate output)")
    ap.add_argument("--run-dir", required=True, help="the run's directory (run_spec.json)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab", required=True, help="the run's tokenizer-v2 vocab.json")
    ap.add_argument("--shards", required=True, help="the arm's gem_events.parquet")
    ap.add_argument("--value-stats", default=None,
                    help="the run's train-partition value_stats.json (required for a "
                         "continuous-fused run: its trunk reads normalized values)")
    ap.add_argument("--model-config", default=str(ROOT / "configs/model.yaml"))
    ap.add_argument("--data-config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--thresholds", default=str(ROOT / "configs/thresholds.yaml"))
    ap.add_argument("--claims", default=str(CLAIMS_PATH))
    ap.add_argument("--site", default="development")
    ap.add_argument("--eval-partition", help="partition to score (default: claims.yaml)")
    ap.add_argument("--final-evaluation", action="store_true",
                    help="allow scoring a sealed partition (internal_test)")
    ap.add_argument("--device", default=None,
                    help="torch device for the trunk and head (default: cuda when "
                         "available, else cpu)")
    ap.add_argument("--max-context", type=int, default=None,
                    help="read each anchor over at most this many tokens of history (default: "
                         "the trunk's max_tokens); evaluates a trained model at e.g. 4096 vs "
                         "8192 without retraining")
    args = ap.parse_args(argv)
    if args.max_context is not None and args.max_context < 2:
        raise SystemExit("--max-context must be at least 2 tokens")

    spec = load_run_spec(args.run_dir)
    claims = load_claims_config(args.claims)
    roles = dict(claims["partitions"])
    if args.eval_partition:
        roles["evaluation"] = args.eval_partition
    if roles["evaluation"] in claims.get("sealed_partitions", ()) and not args.final_evaluation:
        raise SystemExit(
            f"partition {roles['evaluation']!r} is sealed: pass --final-evaluation to score "
            "it, or --eval-partition validation for a development view")
    thresholds = load_thresholds(args.thresholds)
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    dcfg = yaml.safe_load(Path(args.data_config).read_text())
    vblob = load_vocab_blob(args.vocab, required_for="threshold evaluation")
    model = load_model(args.checkpoint, vblob, mcfg, n_targets=len(dcfg["target_concepts"]))
    streams = stay_streams(pl.read_parquet(args.shards),
                           value_stats=run_value_stats(args.value_stats, vblob))
    result = evaluate_run(model, streams, vblob, thresholds, claims,
                          target_concepts=dcfg["target_concepts"],
                          objective_arm=spec["objective_arm"], site=args.site, roles=roles,
                          device=torch.device(args.device or default_device()).type,
                          arm=spec["tokenization_arm"],
                          context_tokens=context_tokens(args.max_context, mcfg))
    if result["summary"]["vocabulary"] != spec["vocab_hash"]:
        raise SystemExit("the vocabulary does not match the run_spec.json vocab_hash")
    write_run_outputs(args.run_dir, result)
    # Aggregate only: how many rows are evaluable per scorer.
    for scorer in SCORERS:
        rows = [r for r in result["summary"]["rows"] if r["scorer"] == scorer]
        if rows:
            n = sum(r["status"] == EVALUABLE for r in rows)
            print(f"{scorer}: {n}/{len(rows)} (threshold, horizon) rows evaluable")


if __name__ == "__main__":
    main()
