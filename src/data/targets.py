"""Deterministic token and time-to-event targets for canonical ICU episodes.

Two modes (U8, KTD10):

- ``icu_24h`` (default): the 24 h prediction artifact. Every feature must be at or
  before the anchor (a later position is leakage and is refused), and TTE outcome
  labels / threshold queries are built at the anchor.
- ``gem``: the full-hospitalization GEM artifact. Next-event targets run over the whole
  stay (pre-ICU to discharge), so the post-anchor check is disabled for this mode ONLY;
  there are no TTE labels (an episode carrying outcomes is refused). Target eligibility
  is still the tokenizer's: the `DISCHARGE//*` terminal token is a target, while
  `<bos>`, `ADMISSION//*`, static and treatment tokens are never targets (Rule 1).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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


TARGET_MODES = ("icu_24h", "gem")
# Plausibility bound on a normalized value: a finite sentinel (e.g. a 999999 pH) beyond
# it is dropped as a value-head target (and masked as a continuous-fused input).
MAX_ABS_Z = 20.0


class TargetContractError(ValueError):
    """An episode cannot safely produce the declared target contract."""


@dataclass(frozen=True)
class TargetBuilder:
    vocab_size: int
    n_time_bins: int
    horizon_hours: float
    value_stats: Mapping[int, tuple[float, float]]
    run_seed: int = 0
    max_abs_value_z: float = MAX_ABS_Z
    mode: str = "icu_24h"

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
        gem = self.mode == "gem"
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
        return {
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

    def _threshold_query(
        self, episode_key: str, labels: Sequence[dict[str, Any]], epoch: int
    ) -> dict[str, Any] | None:
        eligible = [label for label in labels if label["threshold_mask"]]
        if not eligible:
            return None
        digest = hashlib.sha256(
            f"{self.run_seed}:{epoch}:{episode_key}".encode("utf-8")
        ).digest()
        return dict(eligible[int.from_bytes(digest[:8], "big") % len(eligible)])
