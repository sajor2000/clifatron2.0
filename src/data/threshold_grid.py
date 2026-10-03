"""The threshold grid: registered and sampled thresholds on ONE vocabulary's own bins.

`configs/thresholds.yaml` registers thresholds once as (target concept, threshold value,
direction) (KTD3). Labels are always computed at the exact value. The model is shown a
threshold as (concept index, value-bin index under the arm's own segments, direction);
this module is that mapping, bound to the vocabulary hash it was computed against.

THE BIN OF A THRESHOLD is `segments.threshold_bin`, i.e. `segments.bin_index` of a value
just on the event side of it — the rule the 24 h outcome join already uses. For a
threshold strictly inside a segment that is the bin that contains it; for a threshold
that is itself a bin edge it is the adjacent bin on the event side (`below`: under the
edge; `above`: over it). So an off-edge threshold shares its bin with the nearest edge
on its non-event side, which is exactly what the on-edge vs off-edge test measures.

TARGET ELIGIBILITY (hard rule 1). Only a concept listed under `target_concepts` can be a
threshold target, and never one that the vocabulary says a treatment / input-only table
charts. Anything else is refused, not skipped.

TRAINING-TIME SAMPLING (`tau_sampling: empirical_bins`). Thresholds are drawn from the
arm's own interior bin edges: every finite segment bound of a binned target concept
except the outer floor and ceiling (nothing lies beyond those). When two edges name the
same bin (a gap between segments), one is kept — the largest for `below`, the smallest
for `above` — so two label definitions never share one query embedding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.data.segments import (
    DIRECTIONS,
    DISCHARGE_PREFIX,
    MERGE_REL_TOL,
    artifact_binding,
    compare_binding,
    threshold_bin,
    vocab_segments,
)

ROOT = Path(__file__).parents[2]
THRESHOLDS_PATH = ROOT / "configs/thresholds.yaml"
THRESHOLD_KINDS = ("decision", "control", "competing_risk_cause")
THRESHOLD_STATUSES = {
    "decision": (None,),
    "control": ("candidate",),
    "competing_risk_cause": ("contract", "proposed"),
}
LABEL_RULE_KEYS = ("horizon_hours", "baseline_lookback_hours",
                   "required_measurement_within_hours_of_horizon")
TAU_SAMPLING = ("empirical_bins",)
DEATH_TOKEN = f"{DISCHARGE_PREFIX}expired"


class ThresholdGridError(ValueError):
    """A threshold registry or grid request cannot be honoured safely."""


@dataclass(frozen=True)
class Threshold:
    """One registry entry of `configs/thresholds.yaml`."""

    kind: str
    concept: str
    value: float
    direction: str
    status: str | None = None
    paired_decision: float | None = None


@dataclass(frozen=True)
class GridQuery:
    """A threshold as one vocabulary shows it to the model. `value` stays the exact
    threshold (labels are computed at it); `threshold_bin` indexes this arm's segments."""

    concept: str
    target_idx: int
    value: float
    direction: str
    threshold_bin: int

    @property
    def direction_id(self) -> int:
        """`ThresholdHazardHead` direction embedding index: 0 = below, 1 = above."""
        return 0 if self.direction == "below" else 1


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ThresholdGridError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def load_thresholds(path: str | Path = THRESHOLDS_PATH) -> dict[str, Any]:
    """Read and validate the threshold registry.

    Returns ``{"decision": (Threshold, ...), "control": (...), "competing_risk_cause":
    (...), "label_rule": {horizon_hours, baseline_lookback_hours,
    required_measurement_within_hours_of_horizon}}``. Checked here: shape, finite values,
    directions, statuses, one competing-risk cause per concept, and that every control
    is paired with a decision threshold of its own concept and direction. Whether the
    concepts are target-eligible is `ThresholdGrid`'s check (it needs the target map)."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ThresholdGridError(f"{path} is not a threshold registry")
    registry: dict[str, Any] = {}
    for kind in THRESHOLD_KINDS:
        entries = raw.get(kind)
        if not isinstance(entries, list) or not entries:
            raise ThresholdGridError(f"threshold registry has no `{kind}` entries")
        parsed = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("concept"), str):
                raise ThresholdGridError(f"{kind} entry must name a concept: {entry!r}")
            direction = entry.get("direction")
            if direction not in DIRECTIONS:
                raise ThresholdGridError(
                    f"{kind} {entry['concept']}: direction must be one of {DIRECTIONS}")
            status = entry.get("status")
            if status not in THRESHOLD_STATUSES[kind]:
                raise ThresholdGridError(
                    f"{kind} {entry['concept']}: status must be one of "
                    f"{THRESHOLD_STATUSES[kind]}, got {status!r}")
            paired = entry.get("paired_decision")
            parsed.append(Threshold(
                kind, entry["concept"],
                _finite(entry.get("value"), f"{kind} {entry['concept']} value"), direction,
                status,
                None if paired is None else _finite(paired, f"{kind} paired_decision"),
            ))
        if len({(t.concept, t.value, t.direction) for t in parsed}) != len(parsed):
            raise ThresholdGridError(f"threshold registry repeats a `{kind}` entry")
        registry[kind] = tuple(parsed)
    decisions = {(t.concept, t.value, t.direction) for t in registry["decision"]}
    for control in registry["control"]:
        if (control.concept, control.paired_decision, control.direction) not in decisions:
            raise ThresholdGridError(
                f"control {control.concept} {control.value}: paired_decision must be a "
                "decision threshold of the same concept and direction")
        if (control.concept, control.value, control.direction) in decisions:
            raise ThresholdGridError(
                f"control {control.concept} {control.value} is also a decision threshold")
    causes = [t.concept for t in registry["competing_risk_cause"]]
    if len(set(causes)) != len(causes):
        raise ThresholdGridError("a concept has more than one competing-risk cause threshold")
    rule = raw.get("label_rule")
    if not isinstance(rule, Mapping) or set(rule) != set(LABEL_RULE_KEYS):
        raise ThresholdGridError(
            f"threshold registry label_rule must define exactly {', '.join(LABEL_RULE_KEYS)}")
    registry["label_rule"] = {}
    for key in LABEL_RULE_KEYS:
        registry["label_rule"][key] = _finite(rule[key], f"label_rule {key}")
        if registry["label_rule"][key] <= 0:
            raise ThresholdGridError(f"label_rule {key} must be positive")
    return registry


def segment_edges(segments: Sequence[Mapping]) -> list[float]:
    """Every finite bound of a concept's segments, sorted and distinct."""
    return sorted({float(seg[key]) for seg in segments for key in ("lo", "hi")
                   if seg[key] is not None})


def _nearest_edge(value: float, edges: Sequence[float]) -> tuple[float | None, float | None, bool]:
    """(nearest edge, distance to it, value is on an edge). An edge within the
    tokenizer's own merge tolerance of the value IS the value (precedence step 1)."""
    if not edges:
        return None, None, False
    nearest = min(edges, key=lambda edge: (abs(edge - value), edge))
    on_edge = abs(nearest - value) <= MERGE_REL_TOL * max(abs(nearest), abs(value))
    return nearest, 0.0 if on_edge else abs(nearest - value), on_edge


def _edge_rows(thresholds: Mapping[str, Any], segments: Mapping,
               binding: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = []
    for kind in THRESHOLD_KINDS:
        for threshold in thresholds[kind]:
            segs = segments.get(threshold.concept)
            nearest, distance, on_edge = (
                _nearest_edge(threshold.value, segment_edges(segs)) if segs
                else (None, None, None))
            rows.append({
                "kind": kind,
                "concept": threshold.concept,
                "value": threshold.value,
                "direction": threshold.direction,
                "status": threshold.status,
                "binned": bool(segs),
                "on_edge": on_edge,
                "nearest_edge": nearest,
                "distance": distance,
                "threshold_bin": (threshold_bin(threshold.value, segs, threshold.direction)
                                  if segs else None),
                "vocabulary": binding["vocabulary"],
                "numeric_edges": binding["numeric_edges"],
            })
    return rows


def edge_distance(thresholds: Mapping[str, Any], vocab_blob: Mapping) -> list[dict[str, Any]]:
    """The edge-distance table of one vocabulary (KTD3): per registered threshold,
    whether it sits exactly on a bin edge of THIS vocabulary's segments and how far the
    nearest edge is. A control is usable for a claim only where it is off-edge in every
    arm, so the table is computed per frozen vocabulary before any training.

    Rows carry the vocabulary and segments hashes. A concept the vocabulary does not
    bin has ``binned: False`` and no edge fields."""
    return _edge_rows(thresholds, vocab_segments(vocab_blob), artifact_binding(vocab_blob))


class ThresholdGrid:
    """(concept, threshold value, direction) -> (concept index, value bin, direction) for
    one vocabulary, plus what the in-stream target builder needs from that vocabulary.

    `concepts` is the target map in `target_concepts` order (the concept index every
    head uses); `binned` the subset this vocabulary has segments for. `token_target`
    maps a target concept's `concept=bin` token ids to its concept index — the events
    the label rules read. `death_token` / `terminal_tokens` are the `DISCHARGE//*` ids
    (None / empty for a vocabulary built without the GEM allowlist). `causes` maps a
    concept index to its competing-risk cause threshold; `death_cause` is the head's
    extra death slot."""

    def __init__(self, vocab_blob: Mapping, target_concepts: Sequence[Mapping],
                 thresholds: Mapping[str, Any], *, tau_sampling: str = "empirical_bins"):
        if tau_sampling not in TAU_SAMPLING:
            raise ThresholdGridError(
                f"unknown tau_sampling {tau_sampling!r}; expected one of {TAU_SAMPLING}")
        self.tau_sampling = tau_sampling
        self.binding = artifact_binding(vocab_blob)       # refuses a pre-v2 vocabulary
        segments = vocab_segments(vocab_blob)
        vocab = vocab_blob["vocab"]
        sources = vocab_blob.get("concept_sources") or {}
        tables = sources.get("tables") or {}
        input_only = set(sources.get("treatment_sources") or ())
        self.concepts = tuple(concept["name"] for concept in target_concepts)
        if len(set(self.concepts)) != len(self.concepts):
            raise ThresholdGridError("target_concepts repeats a concept")
        self._index = {name: index for index, name in enumerate(self.concepts)}
        self._direction: dict[str, str] = {}
        for concept in target_concepts:
            name, direction = concept["name"], concept.get("direction")
            if direction not in DIRECTIONS:
                raise ThresholdGridError(f"target concept {name} has no valid direction")
            charted_by = input_only & set(tables.get(name) or ())
            if charted_by:
                raise ThresholdGridError(
                    f"{name} is charted by input-only table(s) {sorted(charted_by)}: a "
                    "treatment is never a threshold target (hard rule 1)")
            self._direction[name] = direction
        self._segments = {name: segments[name] for name in self.concepts if segments.get(name)}
        self.binned = tuple(self._segments)
        self.token_target: dict[int, int] = {}
        for name, segs in self._segments.items():
            for b in range(len(segs)):
                token = vocab.get(f"{name}={b}")
                if token is not None:
                    self.token_target[int(token)] = self._index[name]
        self.death_token = vocab.get(DEATH_TOKEN)
        self.terminal_tokens = frozenset(
            int(token) for name, token in vocab.items()
            if name.startswith(DISCHARGE_PREFIX) and token is not None)
        self.death_cause = len(self.concepts)

        self._edges = {name: self._own_edges(name) for name in self.binned}
        self._pool = tuple(edges for edges in self._edges.values() if edges)
        self._thresholds = thresholds
        self._registered = {
            kind: tuple(self.query(t.concept, t.value, t.direction) for t in thresholds[kind])
            for kind in THRESHOLD_KINDS
        }
        self.causes = {query.target_idx: query
                       for query in self._registered["competing_risk_cause"]}

    def _own_edges(self, concept: str) -> tuple[GridQuery, ...]:
        segs = self._segments[concept]
        direction = self._direction[concept]
        outer = {segs[0]["lo"], segs[-1]["hi"]}
        by_bin: dict[int, float] = {}
        keep = max if direction == "below" else min
        for edge in segment_edges(segs):
            if edge in outer:
                continue
            b = threshold_bin(edge, segs, direction)
            by_bin[b] = keep(edge, by_bin.get(b, edge))
        return tuple(GridQuery(concept, self._index[concept], edge, direction, b)
                     for b, edge in sorted(by_bin.items(), key=lambda item: item[1]))

    def check_binding(self, recorded: Mapping | None, *, what: str) -> None:
        """Refuse `what` (a shard row, a checkpoint) bound to a different vocabulary."""
        compare_binding(recorded, self.binding, what=what)

    def query(self, concept: str, value: float, direction: str | None = None) -> GridQuery:
        """Map one threshold to this vocabulary's bins; refuses anything that is not a
        binned target concept queried in its registered direction."""
        if concept not in self._index:
            raise ThresholdGridError(
                f"{concept!r} is not a target concept: only target-eligible measurements "
                "can be threshold targets (hard rule 1)")
        expected = self._direction[concept]
        if direction is not None and direction != expected:
            raise ThresholdGridError(
                f"{concept} thresholds are queried {expected!r}; got direction {direction!r}")
        segs = self._segments.get(concept)
        if segs is None:
            raise ThresholdGridError(
                f"this vocabulary has no segments for target concept {concept!r}")
        value = _finite(value, f"{concept} threshold")
        return GridQuery(concept, self._index[concept], value, expected,
                         threshold_bin(value, segs, expected))

    def registered(self, kind: str) -> tuple[GridQuery, ...]:
        """The registry's `kind` thresholds on this vocabulary, in registry order."""
        return self._registered[kind]

    def sampling_edges(self, concept: str) -> tuple[GridQuery, ...]:
        """The thresholds training samples for `concept`: its own interior edges."""
        return self._edges.get(concept, ())

    def sample(self, rng) -> GridQuery:
        """One training threshold (`tau_sampling: empirical_bins`): a binned target
        concept, then one of its own edges, both uniform. Uses only `rng.random()`,
        the one `random.Random` method whose sequence is guaranteed across versions."""
        if not self._pool:
            raise ThresholdGridError("this vocabulary has no target-concept edge to sample")
        edges = self._pool[int(rng.random() * len(self._pool))]
        return edges[int(rng.random() * len(edges))]

    def edge_distance(self) -> list[dict[str, Any]]:
        """`edge_distance` of this grid's registry on its vocabulary."""
        return _edge_rows(self._thresholds, self._segments, self.binding)
