"""Deterministic token and time-to-event targets for canonical ICU episodes.

Three modes (U8, KTD10; `gem_tte`: KTD1):

- ``icu_24h`` (default): the 24 h prediction artifact. Every feature must be at or
  before the anchor (a later position is leakage and is refused), and TTE outcome
  labels / threshold queries are built at the anchor.
- ``gem``: the full-hospitalization GEM artifact. Next-event targets run over the whole
  stay (pre-ICU to discharge), so the post-anchor check is disabled for this mode ONLY;
  there are no TTE labels (an episode carrying outcomes is refused). Target eligibility
  is still the tokenizer's: the `DISCHARGE//*` terminal token is a target, while
  `<bos>`, `ADMISSION//*`, static and treatment tokens are never targets (Rule 1).
- ``gem_tte``: ``gem`` plus in-stream time-to-event labels. Anchors are sampled along the
  stay, deterministically per (run seed, epoch, episode), and every (anchor, threshold
  query) is labelled from the stream's own future, at the exact threshold value.

IN-STREAM LABEL RULE (``gem_tte``; `TargetBuilder.label_anchor`). An anchor is a MINUTE
of the stream, read at the last token of that minute; ``t`` is minutes since it. Only
finite values of the queried target concept before the terminal `DISCHARGE//*` token
count as measurements. Exactly one status, checked in this order:

1. ``prevalent``: the last measurement in ``[anchor - lookback, anchor]`` is already
   beyond the threshold. Not supervised.
2. ``positive`` at the first measurement beyond the threshold with ``0 < t <= horizon``.
3. ``competing_event`` at ``DISCHARGE//expired`` when ``t <= horizon``.
4. ``censored`` at the stream end (any other discharge, or no terminal token) when
   ``t < horizon``.
5. ``not_ascertainable``: the horizon elapsed with no crossing and no measurement in
   ``[horizon - required_measurement_within_hours_of_horizon, horizon]``. Never
   supervised as a negative.
6. ``negative`` at the horizon otherwise.

"Beyond" is strict (``below``: value < threshold; ``above``: value > threshold), as in
the 24 h outcome contract: a value equal to the threshold is not an event. Every token
of the anchor's minute is context — availability is minute-resolution, so a same-minute
value counts toward the lookback and never as a future event.

RULE CAUSES (`threshold_grid.RuleQuery`). A competing-risk cause may be a label rule; the
only one is ``kdigo_aki`` (creatinine; product authority, 2026-10-03; KDIGO 2012). A
creatinine measurement MEETS the rule when it is at least ``absolute_rise`` above the
lowest creatinine in the preceding ``absolute_window_hours``, or at least
``relative_rise`` times the lowest creatinine in the preceding ``relative_window_hours``
("preceding" = an earlier minute of the same stay, in availability order; both bounds
inclusive, as KDIGO's ">="). The baseline is in-stay only (the lowest inpatient value;
an outpatient baseline is not in the token stream). A measurement with no earlier
creatinine inside the relative window is NOT ASSESSABLE. The statuses are the ones above
with "beyond the threshold" read as "meets the rule", and the ascertainment check needs an
ASSESSABLE measurement in the window before the horizon - a stay with a single creatinine
is `not_ascertainable`, never negative. Labels are still computed on the whole stay.

SUSTAINED CROSSING (optional per competing-risk cause, off by default; product authority,
2026-10-03). With ``sustained = (k, w)`` on a fixed-value query, a crossing counts only when
it is sustained: ``k`` CONSECUTIVE measurements of the concept beyond the threshold (no
measurement that is not beyond between them), all after the anchor minute, the last at
most ``w`` minutes after the first. The event time is the reading that completes the run
(the k-th), and it must lie inside the horizon. An isolated crossing does not fire.
Prevalence is unchanged (the last lookback value beyond the threshold). An UNCONFIRMED run
- the last measurement inside the horizon is beyond the threshold but no run completed and
none could still complete inside the horizon from the readings seen - is treated as unknown,
not as event-free: the stay then reads competing_event / censored as usual when it ends
first, and otherwise ``not_ascertainable`` (never ``negative``). With ``sustained`` absent
the rule above is used unchanged.

Label times are MINUTES since the anchor. The bin is not computed here: each head bins
the minutes on its own grid (`src.model.heads.time_bin`, KTD4).
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TARGET_SCHEMA_VERSION = "1.0.0"
OUTCOME_STATUSES = {
    "positive",
    "negative",
    "censored",
    "competing_event",
    "prevalent",
    "not_ascertainable",
    "unsupported_at_site",
}


TARGET_MODES = ("icu_24h", "gem", "gem_tte")
# Modes that read the full-hospitalization stream (the `gem` representation).
GEM_MODES = ("gem", "gem_tte")
SUPERVISED_STATUSES = ("positive", "negative", "censored", "competing_event")
# Plausibility bound on a normalized value: a finite sentinel (e.g. a 999999 pH) beyond
# it is dropped as a value-head target (and masked as a continuous-fused input).
MAX_ABS_Z = 20.0


class TargetContractError(ValueError):
    """An episode cannot safely produce the declared target contract."""


@dataclass(frozen=True)
class InStreamTargets:
    """What mode ``gem_tte`` needs beyond the builder's horizon (KTD1, KTD3).

    `grid` is the vocabulary's `threshold_grid.ThresholdGrid`: the target concepts'
    tokens, the terminal tokens, the competing-risk cause thresholds and the sampler of
    training thresholds. Each window of a stay gets up to `anchors_per_window` anchors,
    each with one competing-risk label and `queries_per_anchor` threshold queries. The
    two windows are the label rule's (module docstring), in hours."""

    grid: Any
    anchors_per_window: int
    queries_per_anchor: int
    baseline_lookback_hours: float
    required_measurement_within_hours_of_horizon: float


def _whole_minutes(hours: float, what: str) -> int:
    minutes = float(hours) * 60
    if not math.isfinite(minutes) or minutes <= 0 or abs(minutes - round(minutes)) > 1e-9:
        raise TargetContractError(f"{what} must be a positive whole number of minutes")
    return round(minutes)


@dataclass(frozen=True)
class TargetBuilder:
    vocab_size: int
    n_time_bins: int
    horizon_hours: float
    value_stats: Mapping[int, tuple[float, float]]
    run_seed: int = 0
    max_abs_value_z: float = MAX_ABS_Z
    mode: str = "icu_24h"
    in_stream: InStreamTargets | None = None

    def __post_init__(self) -> None:
        if self.mode not in TARGET_MODES:
            raise TargetContractError(
                f"target mode must be one of {', '.join(TARGET_MODES)}, got {self.mode!r}"
            )
        if self.vocab_size <= 0 or self.n_time_bins <= 0 or self.horizon_hours <= 0:
            raise TargetContractError("vocabulary, time-bin count, and horizon must be positive")
        if not math.isfinite(self.max_abs_value_z) or self.max_abs_value_z <= 0:
            raise TargetContractError("maximum absolute normalized value must be positive and finite")
        for token_id, (_, scale) in self.value_stats.items():
            if not 0 <= int(token_id) < self.vocab_size or not math.isfinite(scale) or scale <= 0:
                raise TargetContractError("value statistics contain an invalid token or scale")
        if (self.mode == "gem_tte") != (self.in_stream is not None):
            raise TargetContractError(
                "mode 'gem_tte' requires an in_stream spec (InStreamTargets), and only it")
        if self.in_stream is not None:
            spec = self.in_stream
            if int(spec.anchors_per_window) < 1 or int(spec.queries_per_anchor) < 1:
                raise TargetContractError(
                    "anchors_per_window and queries_per_anchor must be at least 1")
            _whole_minutes(self.horizon_hours, "the label horizon")
            _whole_minutes(spec.baseline_lookback_hours, "baseline_lookback_hours")
            _whole_minutes(spec.required_measurement_within_hours_of_horizon,
                           "required_measurement_within_hours_of_horizon")
            if spec.grid.death_token is None or not spec.grid.terminal_tokens:
                raise TargetContractError(
                    "the vocabulary has no DISCHARGE//expired terminal token: in-stream "
                    "labels need the GEM allowlist (rebuild it with the `gem` block)")

    def build(self, episode: Mapping[str, Any], *, epoch: int = 0) -> dict[str, Any]:
        """Build targets using the next eligible physiologic-event subsequence."""
        episode_key = episode.get("episode_key")
        if not isinstance(episode_key, str) or not episode_key:
            raise TargetContractError("episode_key must be a non-empty opaque string")
        token = [int(value) for value in episode["token"]]
        pos_min = [int(value) for value in episode["pos_min"]]
        values = list(episode.get("value", [None] * len(token)))
        eligible = [bool(value) for value in episode["target_eligible"]]
        if not token or not (len(token) == len(pos_min) == len(values) == len(eligible)):
            raise TargetContractError("token fields must be non-empty and have equal lengths")
        if any(value < 0 or value >= self.vocab_size for value in token):
            raise TargetContractError("token id is outside the frozen vocabulary")
        gem = self.mode in GEM_MODES
        raw_anchor = episode.get("anchor_idx")
        anchor_idx = None if gem and raw_anchor is None else int(raw_anchor)
        if anchor_idx is not None and (anchor_idx < 0 or anchor_idx >= len(token)):
            raise TargetContractError("anchor_idx is outside the episode sequence")
        if gem:
            # The GEM stream deliberately runs past the anchor (whole hospitalization);
            # the anchor index is kept only for representation evaluation.
            if episode.get("outcomes"):
                raise TargetContractError("gem mode carries no TTE outcome labels")
        else:
            anchor_min = int(episode.get("anchor_min", pos_min[anchor_idx]))
            if any(value > anchor_min for value in pos_min):
                raise TargetContractError("post-anchor feature encountered")

        ntp_target = [0] * len(token)
        ntp_mask = [False] * len(token)
        ntp_delta_min = [0] * len(token)
        value_target = [0.0] * len(token)
        value_mask = [False] * len(token)
        physiology = [index for index, allowed in enumerate(eligible) if allowed]
        for source, target in zip(physiology, physiology[1:]):
            ntp_target[source] = token[target]
            ntp_delta_min[source] = pos_min[target] - pos_min[source]
            if ntp_delta_min[source] < 0:
                raise TargetContractError("episode positions must be nondecreasing")
            ntp_mask[source] = True
            value = values[target]
            if value is not None and math.isfinite(float(value)):
                if token[target] not in self.value_stats:
                    raise TargetContractError("numeric target is missing frozen normalization statistics")
                center, scale = self.value_stats[token[target]]
                normalized = (float(value) - center) / scale
                # CLIF source extracts can contain finite sentinel-like values (for
                # example, 999999 for arterial pH). Keep the event token as context,
                # but do not let an implausible mark target dominate the Gaussian NLL.
                if abs(normalized) <= self.max_abs_value_z:
                    value_target[source] = normalized
                    value_mask[source] = True

        labels = [self._outcome_label(row) for row in episode.get("outcomes", [])]
        query = self._threshold_query(episode_key, labels, epoch)
        built = {
            "target_schema_version": TARGET_SCHEMA_VERSION,
            "episode_key": episode_key,
            "anchor_idx": anchor_idx,
            "ntp_target": ntp_target,
            "ntp_mask": ntp_mask,
            "ntp_delta_min": ntp_delta_min,
            "value_target": value_target,
            "value_mask": value_mask,
            "outcome_labels": labels,
            "threshold_query": query,
        }
        if self.mode == "gem_tte":
            built["anchors"] = self._sample_anchors(
                episode_key, token, pos_min, values, episode.get("windows"), epoch)
        return built

    # ---- in-stream labels (mode gem_tte) ------------------------------------------------

    def _stream_index(self, token: Sequence[int], pos_min: Sequence[int],
                      values: Sequence[Any]) -> dict[str, Any]:
        """Per-stay lookup for the label rule: the terminal token, the stream end, and
        each target concept's finite measurements (minute, value) in stream order."""
        grid = self.in_stream.grid
        end_idx = next((i for i, tok in enumerate(token) if tok in grid.terminal_tokens),
                       len(token))
        times: dict[int, list[int]] = {}
        observed: dict[int, list[float]] = {}
        token_target = grid.token_target
        for i in range(end_idx):
            target = token_target.get(token[i])
            value = values[i]
            if target is None or value is None or not math.isfinite(float(value)):
                continue
            times.setdefault(target, []).append(pos_min[i])
            observed.setdefault(target, []).append(float(value))
        return {
            "pos_min": pos_min,
            "end_idx": end_idx,
            "end_min": pos_min[min(end_idx, len(token) - 1)],
            "expired": end_idx < len(token) and token[end_idx] == grid.death_token,
            "events": {target: (np.asarray(times[target], dtype=np.int64),
                                np.asarray(observed[target], dtype=np.float64))
                       for target in times},
            "rule_flags": {},           # (target, rule, params) -> (meets, assessable)
        }

    @staticmethod
    def _check_anchor(index: Mapping[str, Any], anchor_idx: int) -> None:
        pos_min = index["pos_min"]
        if not 0 <= anchor_idx < len(pos_min):
            raise TargetContractError("anchor_idx is outside the episode sequence")
        if anchor_idx >= index["end_idx"]:
            raise TargetContractError("an anchor cannot sit on or after the terminal token")
        if anchor_idx + 1 < len(pos_min) and pos_min[anchor_idx + 1] <= pos_min[anchor_idx]:
            raise TargetContractError(
                "an anchor must be the last token of its minute: every token of that "
                "minute is available at the anchor")

    def _label(self, index: Mapping[str, Any], anchor_idx: int,
               query: Any) -> tuple[str, int | None]:
        """One (anchor, query) status and its minutes since the anchor (module docstring)."""
        if getattr(query, "rule", None) is not None:
            return self._label_rule(index, anchor_idx, query)
        spec = self.in_stream
        horizon = round(self.horizon_hours * 60)
        anchor_min = index["pos_min"][anchor_idx]
        times, observed = index["events"].get(query.target_idx, (_NO_TIMES, _NO_VALUES))
        below = query.direction == "below"
        # Measurements at or before the anchor minute are context; later ones are future.
        first_future = int(np.searchsorted(times, anchor_min, side="right"))
        if first_future:
            last = first_future - 1
            lookback = round(spec.baseline_lookback_hours * 60)
            beyond = observed[last] < query.value if below else observed[last] > query.value
            if times[last] >= anchor_min - lookback and beyond:
                return "prevalent", None
        end_future = int(np.searchsorted(times, anchor_min + horizon, side="right"))
        future = observed[first_future:end_future]
        beyond_future = future < query.value if below else future > query.value
        sustained = getattr(query, "sustained", None)
        unconfirmed = False
        if sustained is None:
            crossings = np.flatnonzero(beyond_future)
            if crossings.size:
                return "positive", int(times[first_future + int(crossings[0])]) - anchor_min
        else:
            completed = _sustained_completion(times[first_future:end_future], beyond_future,
                                              *sustained)
            if completed is not None:
                return "positive", int(times[first_future + completed]) - anchor_min
            unconfirmed = bool(beyond_future.size and beyond_future[-1])
        to_end = index["end_min"] - anchor_min
        if index["expired"] and to_end <= horizon:
            return "competing_event", to_end
        if not index["expired"] and to_end < horizon:
            return "censored", to_end
        window = round(spec.required_measurement_within_hours_of_horizon * 60)
        measured_from = max(anchor_min + 1, anchor_min + horizon - window)
        if not unconfirmed and end_future > int(np.searchsorted(times, measured_from, side="left")):
            return "negative", horizon
        return "not_ascertainable", None

    def _rule_flags(self, index: Mapping[str, Any], query: Any) -> tuple[np.ndarray, np.ndarray]:
        """Per measurement of the rule's concept: (meets the rule, assessable). Cached per
        stay index."""
        key = (query.target_idx, query.rule, query.params)
        cached = index["rule_flags"].get(key)
        if cached is not None:
            return cached
        if query.rule != "kdigo_aki":
            raise TargetContractError(f"unknown competing-risk rule {query.rule!r}")
        times, observed = index["events"].get(query.target_idx, (_NO_TIMES, _NO_VALUES))
        flags = kdigo_aki_flags(
            times, observed,
            absolute_rise=query.param("absolute_rise"),
            absolute_window_minutes=round(query.param("absolute_window_hours") * 60),
            relative_rise=query.param("relative_rise"),
            relative_window_minutes=round(query.param("relative_window_hours") * 60))
        index["rule_flags"][key] = flags
        return flags

    def _label_rule(self, index: Mapping[str, Any], anchor_idx: int,
                    query: Any) -> tuple[str, int | None]:
        """The label rule of the module docstring for a rule cause (RULE CAUSES)."""
        spec = self.in_stream
        horizon = round(self.horizon_hours * 60)
        anchor_min = index["pos_min"][anchor_idx]
        times, _ = index["events"].get(query.target_idx, (_NO_TIMES, _NO_VALUES))
        meets, assessable = self._rule_flags(index, query)
        first_future = int(np.searchsorted(times, anchor_min, side="right"))
        if first_future:
            last = first_future - 1
            lookback = round(spec.baseline_lookback_hours * 60)
            if times[last] >= anchor_min - lookback and meets[last]:
                return "prevalent", None
        end_future = int(np.searchsorted(times, anchor_min + horizon, side="right"))
        crossings = np.flatnonzero(meets[first_future:end_future])
        if crossings.size:
            return "positive", int(times[first_future + int(crossings[0])]) - anchor_min
        to_end = index["end_min"] - anchor_min
        if index["expired"] and to_end <= horizon:
            return "competing_event", to_end
        if not index["expired"] and to_end < horizon:
            return "censored", to_end
        window = round(spec.required_measurement_within_hours_of_horizon * 60)
        measured_from = max(anchor_min + 1, anchor_min + horizon - window)
        start = int(np.searchsorted(times, measured_from, side="left"))
        if end_future > start and assessable[start:end_future].any():
            return "negative", horizon
        return "not_ascertainable", None

    def label_anchor(self, episode: Mapping[str, Any], anchor_idx: int,
                     query: Any) -> dict[str, Any]:
        """Label one (anchor, query) of a full stay: ``{"status", "minutes"}``.

        `query` is a `threshold_grid.GridQuery` (the exact threshold value decides the
        label; its bin does not). `minutes` is the time since the anchor of the event,
        the censoring or the horizon, and None for an unsupervised status."""
        if self.mode != "gem_tte":
            raise TargetContractError("in-stream labels require mode 'gem_tte'")
        index = self._stream_index([int(tok) for tok in episode["token"]],
                                   [int(pos) for pos in episode["pos_min"]],
                                   list(episode.get("value", [None] * len(episode["token"]))))
        self._check_anchor(index, int(anchor_idx))
        status, minutes = self._label(index, int(anchor_idx), query)
        return {"status": status, "minutes": minutes}

    def label_anchors(self, episode: Mapping[str, Any],
                      pairs: Iterable[tuple[int, Any]]) -> list[dict[str, Any]]:
        """`label_anchor` for many ``(anchor_idx, query)`` pairs of ONE stay, against one
        stream index built once (evaluation labels every registered threshold at several
        anchors and horizons). Same output, pair by pair; every anchor is checked."""
        if self.mode != "gem_tte":
            raise TargetContractError("in-stream labels require mode 'gem_tte'")
        index = self._stream_index([int(tok) for tok in episode["token"]],
                                   [int(pos) for pos in episode["pos_min"]],
                                   list(episode.get("value", [None] * len(episode["token"]))))
        out = []
        for anchor_idx, query in pairs:
            self._check_anchor(index, int(anchor_idx))
            status, minutes = self._label(index, int(anchor_idx), query)
            out.append({"status": status, "minutes": minutes})
        return out

    def _competing_risk(self, causes: Mapping[int, tuple[str, int | None]]) -> dict | None:
        """The anchor's competing-risk label from its per-concept cause labels, by the
        rule the 24 h path uses (`outcome_anchor`): the earliest crossing is the event;
        otherwise death; otherwise the longest event-free interval observed. Prevalent
        and not-ascertainable causes are not at risk and say nothing; with none left
        there is no label."""
        events = [(minutes, target) for target, (status, minutes) in causes.items()
                  if status == "positive"]
        if events:
            minutes, target = min(events)
            return {"status": "positive", "cause": target, "minutes": minutes}
        for status, minutes in causes.values():
            if status == "competing_event":
                return {"status": status, "cause": self.in_stream.grid.death_cause,
                        "minutes": minutes}
        free = [(minutes, status) for status, minutes in causes.values()
                if status in ("censored", "negative")]
        if not free:
            return None
        minutes, status = max(free)
        return {"status": status, "cause": -1, "minutes": minutes}

    def _sample_anchors(self, episode_key: str, token: Sequence[int],
                        pos_min: Sequence[int], values: Sequence[Any],
                        windows: Sequence[Sequence[int]] | None,
                        epoch: int) -> list[dict[str, Any]]:
        """Sample anchors per window and label them on the WHOLE stay.

        One generator per (run seed, epoch, episode) draws, window by window, up to
        `anchors_per_window` distinct anchor positions (tokens before the terminal one
        that are the last of their minute) and each anchor's threshold queries. Only
        `random()` is used: its sequence for a given seed is guaranteed stable."""
        spec = self.in_stream
        grid = spec.grid
        n = len(token)
        spans = [(0, n)] if windows is None else [(int(lo), int(hi)) for lo, hi in windows]
        previous = 0
        for lo, hi in spans:
            if not previous <= lo < hi <= n:
                raise TargetContractError(
                    "windows must be ordered, non-empty, non-overlapping spans of the stream")
            previous = hi
        index = self._stream_index(token, pos_min, values)
        digest = hashlib.sha256(
            f"{self.run_seed}:{epoch}:{episode_key}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        anchors = []
        for lo, hi in spans:
            candidates = [i for i in range(lo, min(hi, index["end_idx"]))
                          if i + 1 == n or pos_min[i + 1] > pos_min[i]]
            take = min(int(spec.anchors_per_window), len(candidates))
            for slot in range(take):                      # partial Fisher-Yates
                pick = slot + int(rng.random() * (len(candidates) - slot))
                candidates[slot], candidates[pick] = candidates[pick], candidates[slot]
            for anchor_idx in sorted(candidates[:take]):
                causes = {target: self._label(index, anchor_idx, cause)
                          for target, cause in grid.causes.items()}
                queries = []
                for _ in range(int(spec.queries_per_anchor)):
                    query = grid.sample(rng)
                    status, minutes = self._label(index, anchor_idx, query)
                    queries.append({
                        "target_idx": query.target_idx,
                        "threshold_bin": query.threshold_bin,
                        "direction": query.direction_id,
                        "threshold": query.value,
                        "status": status,
                        "minutes": minutes,
                    })
                anchors.append({
                    "anchor_idx": anchor_idx,
                    "cr": self._competing_risk(causes),
                    "cause_status": {target: status
                                     for target, (status, _) in causes.items()},
                    "queries": queries,
                })
        return anchors

    # ---- the 24 h episode in the per-anchor contract ------------------------------------

    def outcome_anchor(self, episode: Mapping[str, Any], built: Mapping[str, Any], *,
                       epoch: int = 0) -> dict[str, Any]:
        """The 24 h episode's one anchor in the shape `_sample_anchors` emits:
        ``{"cr": label | None, "queries": [query]}``, with times in MINUTES since the
        anchor taken from the joined outcome rows.

        `built["outcome_labels"]` carries bins on this builder's single grid, which is
        the wrong grid for at least one head; the minutes let each head bin on its own
        (KTD4). Selection is unchanged: the threshold query is `built`'s, and the
        competing-risk label is the earliest event among the supervised outcomes, else
        the longest observed event-free interval."""
        rows = list(episode.get("outcomes", []))
        labels = built["outcome_labels"]
        minutes = [
            math.floor(min(float(row["time_from_anchor_hours"]), self.horizon_hours) * 60)
            if label["tte_mask"] else None
            for row, label in zip(rows, labels, strict=True)
        ]
        supervised = [i for i, label in enumerate(labels) if label["tte_mask"]]
        events = [i for i in supervised if labels[i]["event_cause"] >= 0]
        cr = None
        if events:
            chosen = min(events, key=lambda i: minutes[i])
            cr = {"status": labels[chosen]["status"], "cause": labels[chosen]["event_cause"],
                  "minutes": minutes[chosen]}
        elif supervised:
            chosen = max(supervised, key=lambda i: minutes[i])
            cr = {"status": labels[chosen]["status"], "cause": -1, "minutes": minutes[chosen]}
        queries = []
        chosen = self._threshold_query_index(episode["episode_key"], labels, epoch)
        if chosen is not None:
            label = labels[chosen]
            queries.append({
                "target_idx": label["target_idx"],
                "threshold_bin": label["threshold_bin"],
                "direction": label["direction"],
                "threshold": None,
                "status": label["status"],
                "minutes": minutes[chosen],
            })
        return {"cr": cr, "queries": queries}

    def _outcome_label(self, row: Mapping[str, Any]) -> dict[str, Any]:
        status = row.get("status")
        if status not in OUTCOME_STATUSES:
            raise TargetContractError(f"unsupported outcome status: {status!r}")
        target_idx = int(row["target_idx"])
        if not 0 <= target_idx < self.vocab_size:
            raise TargetContractError("outcome target is outside the frozen target map")
        observed = row.get("time_from_anchor_hours")
        supervised = status in {"positive", "negative", "censored", "competing_event"}
        if supervised and (observed is None or not math.isfinite(float(observed)) or observed < 0):
            raise TargetContractError("supervised outcome requires a nonnegative observed time")
        observed_hours = min(float(observed), self.horizon_hours) if supervised else 0.0
        observed_bins = min(
            self.n_time_bins,
            max(0, math.ceil(observed_hours / self.horizon_hours * self.n_time_bins)),
        )
        event_bin = max(0, observed_bins - 1) if status in {"positive", "competing_event"} else -1
        cause_raw = row.get("cause_idx")
        cause = int(cause_raw if cause_raw is not None else target_idx) if status == "positive" else -1
        if status == "competing_event":
            if cause_raw is None:
                raise TargetContractError("competing event requires an explicit cause_idx")
            cause = int(cause_raw)
        direction = row.get("direction")
        if direction not in {"below", "above"}:
            raise TargetContractError("outcome direction must be 'below' or 'above'")
        threshold_bin = int(row["threshold_bin"])
        if threshold_bin < 0:
            raise TargetContractError("threshold_bin must be nonnegative")
        return {
            "target_idx": target_idx,
            "status": status,
            "tte_mask": supervised,
            "event_cause": cause,
            "event_bin": event_bin,
            "observed_bins": observed_bins,
            "censored": status == "censored",
            "threshold_bin": threshold_bin,
            "direction": 0 if direction == "below" else 1,
            "threshold_crossed_bin": event_bin if status == "positive" else -1,
            "threshold_mask": supervised,
        }

    def _threshold_query_index(
        self, episode_key: str, labels: Sequence[dict[str, Any]], epoch: int
    ) -> int | None:
        eligible = [i for i, label in enumerate(labels) if label["threshold_mask"]]
        if not eligible:
            return None
        digest = hashlib.sha256(
            f"{self.run_seed}:{epoch}:{episode_key}".encode("utf-8")
        ).digest()
        return eligible[int.from_bytes(digest[:8], "big") % len(eligible)]

    def _threshold_query(
        self, episode_key: str, labels: Sequence[dict[str, Any]], epoch: int
    ) -> dict[str, Any] | None:
        chosen = self._threshold_query_index(episode_key, labels, epoch)
        return None if chosen is None else dict(labels[chosen])


def _sustained_completion(times: np.ndarray, beyond: np.ndarray, min_readings: int,
                         within_minutes: int) -> int | None:
    """Index of the reading that completes the first run of `min_readings` consecutive
    beyond-threshold readings spanning at most `within_minutes`, or None."""
    run_start = None
    count = 0
    for i, flag in enumerate(beyond):
        if not flag:
            run_start, count = None, 0
            continue
        if run_start is None:
            run_start, count = i, 1
        else:
            count += 1
            # Slide the run start forward until the window fits.
            while int(times[i]) - int(times[run_start]) > within_minutes:
                run_start += 1
                count -= 1
        if count >= min_readings:
            return i
    return None


_NO_TIMES = np.empty(0, dtype=np.int64)
_NO_VALUES = np.empty(0, dtype=np.float64)
# Float tolerance for KDIGO's inclusive ">=": 1.2 - 0.9 is 0.2999... in binary.
_RULE_TOL = 1e-9


def kdigo_aki_flags(times: np.ndarray, values: np.ndarray, *, absolute_rise: float,
                    absolute_window_minutes: int, relative_rise: float,
                    relative_window_minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """KDIGO 2012 AKI by creatinine at each measurement of one stay.

    `times` (minutes, nondecreasing) and `values` are the stay's finite creatinine
    measurements in availability order. Measurement j MEETS the rule when
    ``values[j] - min(prior in [t_j - absolute_window, t_j)) >= absolute_rise`` or
    ``values[j] >= relative_rise * min(prior in [t_j - relative_window, t_j))``; "prior"
    is an earlier minute (a same-minute value is not a baseline). It is ASSESSABLE when
    the relative window holds at least one prior value (the absolute window is assumed no
    longer than the relative one). Returns ``(meets, assessable)`` boolean arrays."""
    times = np.asarray(times, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if absolute_window_minutes > relative_window_minutes:
        raise TargetContractError("the absolute-rise window must not exceed the relative one")
    n = times.size
    meets = np.zeros(n, dtype=bool)
    assessable = np.zeros(n, dtype=bool)
    for j in range(n):
        t = int(times[j])
        end = int(np.searchsorted(times, t, side="left"))            # strictly earlier
        rel_lo = int(np.searchsorted(times, t - relative_window_minutes, side="left"))
        if end <= rel_lo:
            continue
        assessable[j] = True
        relative_min = float(values[rel_lo:end].min())
        if relative_min > 0 and values[j] >= relative_rise * relative_min - _RULE_TOL:
            meets[j] = True
            continue
        abs_lo = int(np.searchsorted(times, t - absolute_window_minutes, side="left"))
        if end > abs_lo and values[j] - float(values[abs_lo:end].min()) >= absolute_rise - _RULE_TOL:
            meets[j] = True
    return meets, assessable


def anchor_status_shares(builds: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate label-status counts and shares over `gem_tte` builds, for reporting.

    Counts only, keyed by status — no episode key, position or time. `threshold_queries`
    counts every sampled (anchor, query); `cause_labels` every (anchor, competing-risk
    cause threshold); `competing_risk` every anchor's competing-risk label, with
    ``not_supervised`` for an anchor that has none."""
    counts: dict[str, dict[str, int]] = {
        "threshold_queries": {}, "cause_labels": {}, "competing_risk": {}}

    def add(group: str, status: str) -> None:
        counts[group][status] = counts[group].get(status, 0) + 1

    anchors = 0
    by_cause: dict[str, int] = {}
    supervised = 0
    for built in builds:
        for anchor in built.get("anchors", ()):
            anchors += 1
            add("competing_risk",
                "not_supervised" if anchor["cr"] is None else anchor["cr"]["status"])
            cr = anchor["cr"]
            if cr is not None:
                supervised += 1
                if cr["status"] in ("positive", "competing_event"):
                    key = str(cr["cause"])
                    by_cause[key] = by_cause.get(key, 0) + 1
            for status in anchor["cause_status"].values():
                add("cause_labels", status)
            for query in anchor["queries"]:
                add("threshold_queries", query["status"])
    report: dict[str, Any] = {"anchors": anchors}
    # Per-cause event rate: share of SUPERVISED anchors (a competing-risk label exists)
    # whose competing-risk event is that cause (keys: cause index as a string; the death
    # slot is the last index). For comparing causes, e.g. RR > 24 vs SpO2 < 88 vs
    # lactate > 4, with and without a sustained rule.
    report["competing_risk_by_cause"] = {
        "supervised_anchors": supervised,
        "events": dict(sorted(by_cause.items(), key=lambda kv: int(kv[0]))),
        "rates": {k: v / supervised for k, v in sorted(by_cause.items(), key=lambda kv: int(kv[0]))}
        if supervised else {},
    }
    for group, by_status in counts.items():
        n = sum(by_status.values())
        report[group] = {
            "n": n,
            "counts": dict(by_status),
            "shares": {status: count / n for status, count in by_status.items()},
        }
    return report
