"""U3 (KTD1, KTD3, KTD4): in-stream time-to-event targets on the full-hospitalization stream.

`TargetBuilder(mode="gem_tte")` samples anchors along each stay and labels every
(anchor, threshold query) from the stream's own future, at the exact threshold value.
Every label rule below is checked on a hand-built stream; the last class trains three
steps on the synthetic site. All data is synthetic.

LABEL RULES (one status per (anchor, query)), in this order:
  prevalent          last value of the concept in [anchor - lookback, anchor] is beyond tau
  positive           first later value beyond tau, at most `horizon` after the anchor
  competing_event    `DISCHARGE//expired` at most `horizon` after the anchor, no crossing
  censored           the stream ends (any other discharge) less than `horizon` after it
  not_ascertainable  horizon elapsed, no crossing, concept not measured in the last
                     `required_measurement_within_hours_of_horizon` hours of the horizon
  negative           horizon elapsed, no crossing, concept measured in that window
TIE RULES: `below` is value < tau and `above` is value > tau (a value equal to the
threshold is not an event); an anchor is a MINUTE and is read at the last token of that
minute, so a same-minute event is context (lookback), never label future; a crossing
or a death exactly at the horizon counts, a non-death stream end exactly at the horizon
does not censor.
"""

import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch
import yaml

from src.data.collate import collate_model_samples
from src.data.dataset import ModelDataset
from src.data.segments import (
    SPECIAL,
    ArtifactBindingError,
    artifact_binding,
    bin_index,
    load_csv_segments,
    n_value_bins,
)
from src.data.targets import (
    InStreamTargets,
    TargetBuilder,
    TargetContractError,
    anchor_status_shares,
)
from src.data.threshold_grid import (
    DEATH_TOKEN,
    THRESHOLD_KINDS,
    ThresholdGrid,
    ThresholdGridError,
    load_thresholds,
)
from src.model.heads import time_bin
from src.train.engine import TrainConfig, _prepare_batch, _train_one_epoch
from src.train.pretrain import Model, in_stream_target_builder

ROOT = Path(__file__).resolve().parents[1]
DATA_CFG = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
MODEL_CFG = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
TARGETS = DATA_CFG["target_concepts"]
DIRECTIONS = {c["name"]: c["direction"] for c in TARGETS}
CONCEPTS = ("map", "lactate")
TREATMENT = "norepinephrine_mcg_kg_min"
H = 48 * 60                      # horizon, minutes
A = 600                          # the anchor minute used by most hand-built streams
CPU = torch.device("cpu")


def _segments() -> dict:
    binning = DATA_CFG["value_binning"]
    return load_csv_segments(ROOT / binning["segment_source"], list(CONCEPTS),
                             binning["forced_edges"], DIRECTIONS)


def _blob() -> dict:
    segments = _segments()
    vocab = dict(SPECIAL)
    for concept, segs in segments.items():
        for b in range(len(segs)):
            vocab[f"{concept}={b}"] = len(vocab)
    for token in (TREATMENT, "ADMISSION//ed", "DISCHARGE//home", DEATH_TOKEN):
        vocab[token] = len(vocab)
    return {
        "vocab": vocab,
        "segments": segments,
        "manifest": {"tokenizer_version": 2},
        "concept_sources": {"tables": {"map": ["vitals"], "lactate": ["labs"],
                                       TREATMENT: ["meds"]},
                            "treatment_sources": ["meds"]},
    }


def _registry() -> dict:
    full = load_thresholds()
    return {**full, **{kind: tuple(t for t in full[kind] if t.concept in CONCEPTS)
                       for kind in THRESHOLD_KINDS}}


BLOB = _blob()
VOCAB = BLOB["vocab"]
GRID = ThresholdGrid(BLOB, TARGETS, _registry())
MAP_65 = GRID.query("map", 65.0)
LACTATE_4 = GRID.query("lactate", 4.0)
HASHES = artifact_binding(BLOB)
VALUE_STATS = {token: (0.0, 100.0) for token in GRID.token_target}
MAP_60_TOKEN = VOCAB[f"map={bin_index(60.0, BLOB['segments']['map'])}"]


def builder(**overrides) -> TargetBuilder:
    in_stream = {"grid": GRID, "anchors_per_window": 3, "queries_per_anchor": 2,
                 "baseline_lookback_hours": 6,
                 "required_measurement_within_hours_of_horizon": 12}
    in_stream.update({k: overrides.pop(k) for k in list(overrides) if k in in_stream})
    kwargs = {"vocab_size": len(VOCAB), "n_time_bins": 16, "horizon_hours": 48,
              "value_stats": VALUE_STATS, "run_seed": 7, "mode": "gem_tte",
              "in_stream": InStreamTargets(**in_stream)}
    kwargs.update(overrides)
    return TargetBuilder(**kwargs)


def stream(events, *, end=None, key="opaque-stay", frame=True):
    """A full-hospitalization stream. `events` are (minute, concept, value) in stream
    order; `end` is (disposition, minute) for the `DISCHARGE//x <eos>` tail. A
    measurement's token is `concept=<its bin>`; `value=None` is a missing value."""
    token, pos, value, eligible = [], [], [], []
    if frame:
        token += [SPECIAL["<bos>"], VOCAB["ADMISSION//ed"]]
        pos += [0, 0]
        value += [None, None]
        eligible += [False, False]
    for minute, concept, v in events:
        if concept == TREATMENT:
            token.append(VOCAB[TREATMENT])
        else:
            binned = 0.0 if v is None or math.isnan(v) else v
            b = bin_index(binned, BLOB["segments"][concept])
            token.append(VOCAB[f"{concept}={b}"])
        pos.append(minute)
        value.append(v)
        eligible.append(concept != TREATMENT)
    if end is not None:
        disposition, minute = end
        token += [VOCAB[f"DISCHARGE//{disposition}"], SPECIAL["<eos>"]]
        pos += [minute, minute]
        value += [None, None]
        eligible += [True, False]
    return {"episode_key": key, "token": token, "pos_min": pos, "value": value,
            "target_eligible": eligible, "anchor_idx": None, "outcomes": []}


def anchored(future, *, before=((A, "map", 80.0),), end=("home", A + 100 * 60)):
    """Stream with one MAP 80 at the anchor minute `A`, then `future` events. Returns
    (stream, index of the anchor token)."""
    s = stream([*before, *future], end=end)
    index = max(i for i, minute in enumerate(s["pos_min"])
                if minute == A and s["token"][i] not in GRID.terminal_tokens)
    return s, index


def label(s, index, query=MAP_65, **overrides):
    return builder(**overrides).label_anchor(s, index, query)


class LabelRuleTest(unittest.TestCase):
    def test_map_below_65_five_hours_after_the_anchor_is_positive_in_hour_bin_5(self):
        s, a = anchored([(A + 120, "map", 72.0), (A + 300, "map", 64.0),
                         (A + 400, "map", 50.0)])
        self.assertEqual(label(s, a), {"status": "positive", "minutes": 300})
        minutes = torch.tensor([300])
        self.assertEqual(time_bin(minutes, 48, 48).item(), 5)     # threshold head, hourly
        self.assertEqual(time_bin(minutes, 16, 48).item(), 1)     # competing-risk, 3 h

    def test_no_crossing_with_a_measurement_near_the_horizon_is_negative_at_the_horizon(self):
        s, a = anchored([(A + 60, "map", 75.0), (A + 40 * 60, "map", 71.0)])
        self.assertEqual(label(s, a), {"status": "negative", "minutes": H})

    def test_non_death_discharge_ten_hours_after_the_anchor_is_censored_at_ten_hours(self):
        s, a = anchored([(A + 60, "map", 75.0)], end=("home", A + 600))
        self.assertEqual(label(s, a), {"status": "censored", "minutes": 600})
        # Credited only through the last FULLY observed bin of each head's own grid.
        self.assertEqual(time_bin(torch.tensor([600]), 48, 48).item(), 10)
        self.assertEqual(time_bin(torch.tensor([600]), 16, 48).item(), 3)

    def test_stream_end_without_a_terminal_token_is_censored(self):
        s, a = anchored([(A + 60, "map", 75.0), (A + 90, "map", 76.0)], end=None)
        self.assertEqual(label(s, a), {"status": "censored", "minutes": 90})

    def test_death_before_any_crossing_is_a_competing_event(self):
        s, a = anchored([(A + 60, "map", 75.0)], end=("expired", A + 600))
        self.assertEqual(label(s, a), {"status": "competing_event", "minutes": 600})

    def test_a_crossing_before_death_is_positive_not_competing(self):
        s, a = anchored([(A + 60, "map", 60.0)], end=("expired", A + 600))
        self.assertEqual(label(s, a), {"status": "positive", "minutes": 60})

    def test_last_lookback_value_already_below_the_threshold_is_prevalent(self):
        s, a = anchored([(A + 60, "map", 75.0)],
                        before=[(A - 30, "map", 80.0), (A, "map", 60.0)])
        self.assertEqual(label(s, a), {"status": "prevalent", "minutes": None})

    def test_prevalence_reads_only_the_last_value_inside_the_lookback(self):
        future = [(A + 60, "map", 75.0), (A + 40 * 60, "map", 72.0)]
        # A low value earlier, but the LAST lookback value is back above: at risk.
        s, a = anchored(future, before=[(A - 60, "map", 55.0), (A, "map", 80.0)])
        self.assertEqual(label(s, a)["status"], "negative")
        # The lookback is [anchor - 6 h, anchor], inclusive at both ends.
        s, a = anchored(future, before=[(A - 360, "map", 55.0), (A, TREATMENT, 0.1)])
        self.assertEqual(label(s, a)["status"], "prevalent")
        s, a = anchored(future, before=[(A - 361, "map", 55.0), (A, TREATMENT, 0.1)])
        self.assertEqual(label(s, a)["status"], "negative")
        # The window is a named parameter.
        self.assertEqual(label(s, a, baseline_lookback_hours=7)["status"], "prevalent")

    def test_no_measurement_in_the_registered_window_is_not_ascertainable(self):
        s, a = anchored([(A + 10 * 60, "map", 75.0)])
        self.assertEqual(label(s, a), {"status": "not_ascertainable", "minutes": None})
        # ... and it is never supervised as a negative (see BatchContractTest).

    def test_the_ascertainment_window_is_inclusive_and_configurable(self):
        s, a = anchored([(A + 36 * 60, "map", 75.0)])           # exactly 12 h before the end
        self.assertEqual(label(s, a)["status"], "negative")
        s, a = anchored([(A + 36 * 60 - 1, "map", 75.0)])
        self.assertEqual(label(s, a)["status"], "not_ascertainable")
        self.assertEqual(
            label(s, a, required_measurement_within_hours_of_horizon=13)["status"],
            "negative")
        # A measurement after the horizon does not ascertain it.
        s, a = anchored([(A + H + 1, "map", 75.0)])
        self.assertEqual(label(s, a)["status"], "not_ascertainable")

    def test_a_concept_never_measured_is_not_ascertainable_once_the_horizon_elapses(self):
        s, a = anchored([(A + 40 * 60, "map", 75.0)])
        self.assertEqual(label(s, a, LACTATE_4),
                         {"status": "not_ascertainable", "minutes": None})

    def test_a_missing_value_is_not_a_measurement(self):
        s, a = anchored([(A + 40 * 60, "map", None)])
        self.assertEqual(label(s, a)["status"], "not_ascertainable")
        s, a = anchored([(A + 40 * 60, "map", float("nan"))])
        self.assertEqual(label(s, a)["status"], "not_ascertainable")

    def test_above_direction_uses_values_over_the_threshold(self):
        s, a = anchored([(A + 30, "lactate", 3.9), (A + 90, "lactate", 4.5)])
        self.assertEqual(label(s, a, LACTATE_4), {"status": "positive", "minutes": 90})

    def test_labels_use_the_exact_threshold_value_not_the_bin(self):
        """MAP 63 and MAP 65 name the same bin; their labels still differ."""
        control = GRID.query("map", 63.0)
        self.assertEqual(control.threshold_bin, MAP_65.threshold_bin)
        s, a = anchored([(A + 60, "map", 64.0), (A + 40 * 60, "map", 70.0)])
        self.assertEqual(label(s, a, MAP_65)["status"], "positive")
        self.assertEqual(label(s, a, control)["status"], "negative")

    def test_treatment_events_are_never_measurements_of_a_target_concept(self):
        s, a = anchored([(A + 60, TREATMENT, 0.0), (A + 40 * 60, TREATMENT, 50.0)])
        self.assertEqual(label(s, a)["status"], "not_ascertainable")
        self.assertNotIn(VOCAB[TREATMENT], GRID.token_target)


class BoundaryRuleTest(unittest.TestCase):
    def test_a_value_equal_to_the_threshold_is_not_an_event(self):
        """`below` is strict (value < 65), as in the 24 h outcome contract and the
        forced edge's closure (65 itself sits in the bin above, the non-event side)."""
        s, a = anchored([(A + 60, "map", 65.0), (A + 40 * 60, "map", 65.0)])
        self.assertEqual(label(s, a)["status"], "negative")
        s, a = anchored([(A + 60, "map", 64.9)])
        self.assertEqual(label(s, a), {"status": "positive", "minutes": 60})
        # ... and a lookback value equal to the threshold is not prevalent.
        s, a = anchored([(A + 60, "map", 64.0)], before=[(A, "map", 65.0)])
        self.assertEqual(label(s, a)["status"], "positive")
        # `above` is strict too.
        s, a = anchored([(A + 60, "lactate", 4.0), (A + 40 * 60, "lactate", 4.0)])
        self.assertEqual(label(s, a, LACTATE_4)["status"], "negative")

    def test_a_crossing_exactly_at_the_horizon_counts_and_one_minute_later_does_not(self):
        s, a = anchored([(A + H, "map", 60.0)])
        self.assertEqual(label(s, a), {"status": "positive", "minutes": H})
        # It belongs to the last bin of each grid (the head clamps `time_bin`).
        self.assertEqual(time_bin(torch.tensor([H]), 48, 48).item(), 48)
        s, a = anchored([(A + H - 30, "map", 70.0), (A + H + 1, "map", 60.0)])
        self.assertEqual(label(s, a), {"status": "negative", "minutes": H})

    def test_death_exactly_at_the_horizon_competes_and_a_discharge_there_does_not_censor(self):
        s, a = anchored([(A + 40 * 60, "map", 75.0)], end=("expired", A + H))
        self.assertEqual(label(s, a), {"status": "competing_event", "minutes": H})
        s, a = anchored([(A + 40 * 60, "map", 75.0)], end=("home", A + H))
        self.assertEqual(label(s, a), {"status": "negative", "minutes": H})
        s, a = anchored([(A + 40 * 60, "map", 75.0)], end=("home", A + H - 1))
        self.assertEqual(label(s, a), {"status": "censored", "minutes": H - 1})
        s, a = anchored([(A + 40 * 60, "map", 75.0)], end=("expired", A + H + 1))
        self.assertEqual(label(s, a), {"status": "negative", "minutes": H})

    def test_anchor_at_the_last_event_of_the_stay(self):
        s, a = anchored([], end=("home", A + 30))
        self.assertEqual(a, len(s["token"]) - 3)                # last event before discharge
        self.assertEqual(label(s, a), {"status": "censored", "minutes": 30})
        s, a = anchored([], end=("expired", A + 30))
        self.assertEqual(label(s, a), {"status": "competing_event", "minutes": 30})
        s, a = anchored([], end=None)                             # the very last token
        self.assertEqual(label(s, a), {"status": "censored", "minutes": 0})

    def test_an_anchor_on_or_after_the_terminal_token_is_refused(self):
        s, a = anchored([], end=("expired", A + 30))
        for index in (a + 1, a + 2):
            with self.assertRaisesRegex(TargetContractError, "terminal"):
                label(s, index)
        with self.assertRaisesRegex(TargetContractError, "outside"):
            label(s, len(s["token"]))

    def test_a_same_minute_event_is_context_never_label_future(self):
        """Availability is minute-resolution: every token of the anchor's minute is
        available at the anchor, so the anchor is read at the minute's LAST token and a
        same-minute value counts toward the lookback."""
        s = stream([(A, "map", 80.0), (A, "map", 60.0), (A + 60, "map", 75.0)],
                   end=("home", A + 6000))
        first, last = 2, 3
        self.assertEqual(s["pos_min"][first], s["pos_min"][last])
        # Reading the anchor before the minute is complete would turn the same-minute
        # MAP 60 into a "future" crossing at 0 minutes: refused.
        with self.assertRaisesRegex(TargetContractError, "last token of its minute"):
            label(s, first)
        self.assertEqual(label(s, last), {"status": "prevalent", "minutes": None})
        # One minute later it IS future.
        s = stream([(A, "map", 80.0), (A + 1, "map", 60.0)], end=("home", A + 6000))
        self.assertEqual(label(s, 2), {"status": "positive", "minutes": 1})

    def test_an_anchor_in_the_terminal_minute_is_refused(self):
        s = stream([(A, "map", 80.0), (A + 30, "map", 70.0)], end=("expired", A + 30))
        with self.assertRaisesRegex(TargetContractError, "last token of its minute"):
            label(s, 3)


class AnchorSamplingTest(unittest.TestCase):
    def _stay(self, key="opaque-stay"):
        events = [(minute, "map", 80.0 - (minute // 97) % 30) for minute in range(5, 4000, 37)]
        events += [(minute, "lactate", 1.0 + (minute // 211) % 5) for minute in range(9, 4000, 211)]
        events.sort(key=lambda e: e[0])
        s = stream(events, end=("home", 4100), key=key)
        s["windows"] = [(lo, min(lo + 40, len(s["token"])))
                        for lo in range(0, len(s["token"]), 40)]
        return s

    def test_same_seed_epoch_and_episode_repeat_exactly(self):
        s = self._stay()
        first = builder().build(s, epoch=3)
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(builder().build(copy.deepcopy(s), epoch=3), sort_keys=True))

    def test_epoch_seed_and_episode_each_change_the_sample(self):
        def anchors(b, s, epoch):
            return [(a["anchor_idx"], [(q["target_idx"], q["threshold"]) for q in a["queries"]])
                    for a in b.build(s, epoch=epoch)["anchors"]]

        s = self._stay()
        base = anchors(builder(), s, 0)
        self.assertNotEqual(base, anchors(builder(), s, 1))
        self.assertNotEqual(base, anchors(builder(run_seed=8), s, 0))
        self.assertNotEqual(base, anchors(builder(), self._stay("opaque-other"), 0))
        self.assertEqual(len({json.dumps(anchors(builder(), s, e)) for e in range(6)}), 6)

    def test_every_window_gets_its_anchors_each_with_its_queries(self):
        s = self._stay()
        built = builder(anchors_per_window=3, queries_per_anchor=2).build(s)
        indexes = [a["anchor_idx"] for a in built["anchors"]]
        self.assertEqual(indexes, sorted(set(indexes)))
        for lo, hi in s["windows"][:-1]:
            self.assertEqual(sum(lo <= i < hi for i in indexes), 3)
        self.assertTrue(all(len(a["queries"]) == 2 for a in built["anchors"]))

    def test_anchors_are_last_of_minute_and_before_the_terminal_token(self):
        s = stream([(10, "map", 80.0), (10, "map", 79.0), (10, TREATMENT, 0.1),
                    (20, "map", 78.0), (20, "lactate", 1.0), (30, "map", 70.0),
                    (40, "map", 71.0)], end=("home", 40))
        built = builder(anchors_per_window=50).build(s)
        # Minute 0: the ADMISSION token; minute 10: index 4; minute 20: index 6;
        # minute 30: index 7. Minute 40 holds the terminal token: no anchor there.
        self.assertEqual([a["anchor_idx"] for a in built["anchors"]], [1, 4, 6, 7])

    def test_threshold_queries_are_target_concepts_own_edges_only(self):
        built = builder(anchors_per_window=8, queries_per_anchor=6).build(self._stay())
        seen = set()
        for anchor in built["anchors"]:
            for q in anchor["queries"]:
                concept = GRID.concepts[q["target_idx"]]
                seen.add(concept)
                pool = {(e.value, e.threshold_bin, e.direction_id)
                        for e in GRID.sampling_edges(concept)}
                self.assertIn((q["threshold"], q["threshold_bin"], q["direction"]), pool)
        self.assertEqual(seen, set(CONCEPTS))

    def test_an_input_only_concept_in_the_grid_is_refused(self):
        registry = _registry()
        threshold = registry["decision"][0].__class__("decision", TREATMENT, 0.1, "above")
        with self.assertRaisesRegex(ThresholdGridError, "not a target concept"):
            ThresholdGrid(BLOB, TARGETS, {**registry,
                                          "decision": (*registry["decision"], threshold)})
        with self.assertRaisesRegex(ThresholdGridError, "not a target concept"):
            GRID.query(TREATMENT, 0.1, "above")

    def test_next_event_and_value_targets_match_the_gem_mode(self):
        s = self._stay()
        gem = TargetBuilder(len(VOCAB), 16, 48, VALUE_STATS, run_seed=7, mode="gem").build(s)
        tte = builder().build(s)
        for field in ("ntp_target", "ntp_mask", "ntp_delta_min", "value_target", "value_mask"):
            self.assertEqual(tte[field], gem[field], field)
        self.assertNotIn("anchors", gem)
        self.assertEqual((tte["outcome_labels"], tte["threshold_query"]), ([], None))

    def test_labels_are_computed_on_the_whole_stay_before_windowing(self):
        """A stay longer than one window: the anchor is in window 0, the crossing in
        window 1, and the window-0 sample still carries the positive label."""
        events = [(60 * k, "map", 80.0) for k in range(1, 9)] + [(60 * 9, "map", 60.0)]
        s = stream(events, end=("home", 9000))
        s["windows"] = [(0, 6), (6, len(s["token"]))]
        self.assertGreaterEqual(s["token"].index(MAP_60_TOKEN), 6)     # the crossing token
        built = builder(anchors_per_window=50).build(s)
        in_first = [a for a in built["anchors"] if a["anchor_idx"] < 6]
        self.assertTrue(in_first)
        for anchor in in_first:
            self.assertEqual(anchor["cr"]["status"], "positive")
            self.assertEqual(anchor["cr"]["minutes"], 540 - s["pos_min"][anchor["anchor_idx"]])


class CompetingRiskLabelTest(unittest.TestCase):
    def _cr(self, s, index):
        # One window holding only the anchor token: that token is the sampled anchor.
        s = dict(s, windows=[(index, index + 1)])
        (anchor,) = builder(anchors_per_window=1).build(s)["anchors"]
        self.assertEqual(anchor["anchor_idx"], index)
        return anchor

    def test_the_earliest_cause_threshold_crossing_is_the_event(self):
        s, a = anchored([(A + 180, "lactate", 5.0), (A + 300, "map", 60.0)])
        anchor = self._cr(s, a)
        self.assertEqual(anchor["cr"], {"status": "positive", "cause": 1, "minutes": 180})
        self.assertEqual(anchor["cause_status"], {0: "positive", 1: "positive"})

    def test_death_is_the_dedicated_last_cause(self):
        s, a = anchored([(A + 60, "map", 75.0)], end=("expired", A + 600))
        self.assertEqual(self._cr(s, a)["cr"],
                         {"status": "competing_event", "cause": len(TARGETS), "minutes": 600})
        self.assertEqual(GRID.death_cause, len(TARGETS))

    def test_event_free_horizon_is_negative_and_a_prevalent_cause_is_not_at_risk(self):
        s, a = anchored([(A + 40 * 60, "map", 75.0), (A + 41 * 60, "lactate", 1.0)],
                        before=[(A - 5, "lactate", 6.0), (A, "map", 80.0)])
        anchor = self._cr(s, a)
        self.assertEqual(anchor["cause_status"], {0: "negative", 1: "prevalent"})
        self.assertEqual(anchor["cr"], {"status": "negative", "cause": -1, "minutes": H})

    def test_censored_stay_is_censored_at_the_stream_end(self):
        s, a = anchored([(A + 60, "map", 75.0)], end=("home", A + 600))
        self.assertEqual(self._cr(s, a)["cr"],
                         {"status": "censored", "cause": -1, "minutes": 600})

    def test_no_ascertainable_cause_means_no_competing_risk_label(self):
        s, a = anchored([(A + 60, "map", 75.0)])
        anchor = self._cr(s, a)
        self.assertEqual(anchor["cause_status"],
                         {0: "not_ascertainable", 1: "not_ascertainable"})
        self.assertIsNone(anchor["cr"])


class StatusShareTest(unittest.TestCase):
    def test_shares_are_aggregate_counts_over_anchors_and_queries(self):
        positive, a1 = anchored([(A + 300, "map", 60.0)])
        unknown, a2 = anchored([(A + 60, "map", 75.0)])
        builds = []
        for s, index in ((positive, a1), (unknown, a2)):
            b = builder(anchors_per_window=1, queries_per_anchor=4)
            builds.append(b.build(dict(s, windows=[(index, index + 1)])))
        report = anchor_status_shares(builds)
        self.assertEqual(report["anchors"], 2)
        self.assertEqual(report["competing_risk"]["counts"],
                         {"positive": 1, "not_supervised": 1})
        self.assertEqual(report["competing_risk"]["shares"],
                         {"positive": 0.5, "not_supervised": 0.5})
        self.assertEqual(report["cause_labels"]["n"], 4)
        self.assertEqual(report["cause_labels"]["counts"],
                         {"positive": 1, "not_ascertainable": 3})
        self.assertEqual(report["threshold_queries"]["n"], 8)
        self.assertAlmostEqual(sum(report["threshold_queries"]["shares"].values()), 1.0)
        # Aggregate only: nothing but counts and shares, keyed by status.
        self.assertEqual(set(report), {"anchors", "threshold_queries", "cause_labels",
                                       "competing_risk"})
        self.assertNotIn("opaque-stay", json.dumps(report))

    def test_empty_input_reports_zero_anchors(self):
        self.assertEqual(anchor_status_shares([])["anchors"], 0)


class BuilderContractTest(unittest.TestCase):
    def test_the_mode_requires_its_in_stream_spec_and_only_it(self):
        with self.assertRaisesRegex(TargetContractError, "in_stream"):
            TargetBuilder(len(VOCAB), 16, 48, {}, mode="gem_tte")
        with self.assertRaisesRegex(TargetContractError, "in_stream"):
            TargetBuilder(len(VOCAB), 16, 48, {}, mode="gem",
                          in_stream=builder().in_stream)

    def test_invalid_sampling_and_window_parameters_are_refused(self):
        for bad in ({"anchors_per_window": 0}, {"queries_per_anchor": 0},
                    {"baseline_lookback_hours": 0},
                    {"required_measurement_within_hours_of_horizon": -1}):
            with self.subTest(**bad), self.assertRaises(TargetContractError):
                builder(**bad)
        with self.assertRaisesRegex(TargetContractError, "whole number of minutes"):
            builder(horizon_hours=0.01)

    def test_a_vocabulary_without_the_terminal_allowlist_is_refused(self):
        blob = copy.deepcopy(BLOB)
        del blob["vocab"][DEATH_TOKEN]
        with self.assertRaisesRegex(TargetContractError, "DISCHARGE//expired"):
            builder(grid=ThresholdGrid(blob, TARGETS, _registry()))

    def test_joined_outcomes_are_refused(self):
        s, _ = anchored([])
        s["outcomes"] = [{"target_idx": 0, "status": "negative", "time_from_anchor_hours": 4,
                          "threshold_bin": 2, "direction": "below"}]
        with self.assertRaisesRegex(TargetContractError, "gem"):
            builder().build(s)

    def test_windows_must_be_ordered_spans_inside_the_stream(self):
        s, _ = anchored([])
        n = len(s["token"])
        for windows in ([(0, 0)], [(0, 3), (2, n)], [(0, n + 1)], [(3, n), (0, 3)]):
            with self.subTest(windows=windows), self.assertRaisesRegex(
                    TargetContractError, "windows"):
                builder().build(dict(s, windows=windows))


def window_rows(s, size):
    """Split a hand-built stream into GEM window records (the gem_events.parquet row)."""
    n = len(s["token"])
    bounds = [(lo, min(lo + size, n)) for lo in range(0, n, size)]
    rows = []
    for index, (lo, hi) in enumerate(bounds):
        rows.append({
            "hosp_id": s["episode_key"], "trajectory": "hospitalization",
            "artifact_hashes": dict(HASHES), "partition": "train",
            **{f: s[f][lo:hi] for f in ("token", "pos_min", "value", "target_eligible")},
            "source_start": lo, "source_end": hi, "continuation_index": index,
            "n_windows": len(bounds), "continues_from_previous": index > 0,
            "continues_to_next": index < len(bounds) - 1,
            "anchor_idx": 2, "anchor_min": s["pos_min"][2],
        })
    return rows


def tiny_mcfg(**weights):
    mcfg = {
        "trunk": {"d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0,
                  "rope_base": 10000.0, "tied_embeddings": False},
        "heads": copy.deepcopy(MODEL_CFG["heads"]),
        "in_stream": {"anchors_per_window": 3, "queries_per_anchor": 2},
    }
    mcfg["heads"]["threshold_hazard"]["threshold_embed_dim"] = 4
    for name, weight in weights.items():
        mcfg["heads"][name]["weight"] = weight
    return mcfg


class BatchContractTest(unittest.TestCase):
    """Dataset -> collate -> engine batch -> `Model`: several anchors per sample."""

    @classmethod
    def setUpClass(cls):
        events = [(60 * k, "map", 80.0) for k in range(1, 9)] + [(60 * 9, "map", 60.0)]
        events += [(60 * k + 7, "lactate", 1.5) for k in range(1, 9)]
        events.sort(key=lambda e: e[0])
        cls.stream = stream(events, end=("home", 9000))
        cls.rows = window_rows(cls.stream, 8)
        cls.builder = builder(anchors_per_window=3, queries_per_anchor=2)
        cls.dataset = ModelDataset(cls.rows, representation="gem", target_builder=cls.builder,
                                   expected_hashes=HASHES)
        cls.samples = [cls.dataset[i] for i in range(len(cls.dataset))]
        cls.batch = collate_model_samples(cls.samples)

    def test_window_samples_carry_their_own_anchors_with_window_offsets(self):
        built = self.builder.build({**self.stream,
                                    "windows": [(r["source_start"], r["source_end"])
                                                for r in self.rows]})
        expected = [a["anchor_idx"] for a in built["anchors"]]
        got = []
        for row, sample in zip(self.rows, self.samples):
            (segment,) = sample["segments"]
            for anchor in segment["anchors"]:
                self.assertTrue(0 <= anchor["offset"] < len(sample["input_ids"]))
                got.append(row["source_start"] + anchor["offset"])
        self.assertEqual(got, expected)
        self.assertEqual(len(self.samples[0]["segments"][0]["anchors"]), 3)

    def test_first_window_label_uses_events_from_the_next_window(self):
        crossing = self.stream["pos_min"][self.stream["token"].index(MAP_60_TOKEN)]
        self.assertGreaterEqual(self.stream["token"].index(MAP_60_TOKEN),
                                self.rows[0]["source_end"])
        for anchor in self.samples[0]["segments"][0]["anchors"]:
            minute = self.rows[0]["pos_min"][anchor["offset"]]
            self.assertEqual(anchor["cr"], {"status": "positive", "cause": 0,
                                            "minutes": crossing - minute})

    def test_collate_emits_one_row_per_anchor_and_one_per_query(self):
        b = self.batch
        n_anchors = sum(len(s["segments"][0]["anchors"]) for s in self.samples)
        n_queries = 2 * n_anchors
        self.assertGreater(n_anchors, len(self.samples))       # several anchors per sample
        for key in ("anchor_batch_idx", "anchor_idx", "flash_anchor_idx", "cr_mask",
                    "cr_type", "cr_time_min"):
            self.assertEqual(tuple(b[key].shape), (n_anchors,), key)
        for key in ("th_anchor", "th_mask", "th_target", "th_tau", "th_dir", "th_event",
                    "th_time_min"):
            self.assertEqual(tuple(b[key].shape), (n_queries,), key)
        self.assertEqual(b["cr_mask"].dtype, torch.bool)
        self.assertEqual(b["th_event"].dtype, torch.bool)
        self.assertEqual(b["cr_time_min"].dtype, torch.long)
        # Each query row points at its anchor; anchors are grouped per sample row.
        self.assertEqual(b["th_anchor"].tolist(), [i // 2 for i in range(n_queries)])
        rows = b["anchor_batch_idx"].tolist()
        self.assertEqual(rows, sorted(rows))
        for row, position in zip(rows, b["anchor_idx"].tolist()):
            self.assertTrue(bool(b["attention_mask"][row, position]))
        # The flattened (flash) view indexes the same tokens.
        self.assertTrue(torch.equal(b["flash_input_ids"][b["flash_anchor_idx"]],
                                    b["input_ids"][b["anchor_batch_idx"], b["anchor_idx"]]))
        self.assertNotIn("document_labels", b)
        self.assertNotIn("threshold_queries", b)

    def test_only_supervised_statuses_are_supervised(self):
        statuses = [q["status"] for s in self.samples for a in s["segments"][0]["anchors"]
                    for q in a["queries"]]
        supervised = [st in ("positive", "negative", "censored", "competing_event")
                      for st in statuses]
        self.assertEqual(self.batch["th_mask"].tolist(), supervised)
        events = [st == "positive" for st in statuses]
        self.assertEqual(self.batch["th_event"].tolist(), events)

    def test_not_ascertainable_is_never_supervised_as_a_negative(self):
        s, a = anchored([(A + 10 * 60, "map", 75.0)])
        one = builder(anchors_per_window=1, queries_per_anchor=3)
        built = one.build(dict(s, windows=[(a, a + 1)]))
        (anchor,) = built["anchors"]
        self.assertIsNone(anchor["cr"])
        sample = {"packed_schema_version": "2.0.0", "input_ids": s["token"],
                  "attention_mask": [1] * len(s["token"]), "pos_min": s["pos_min"],
                  "ntp_target": built["ntp_target"], "ntp_mask": built["ntp_mask"],
                  "ntp_delta_min": built["ntp_delta_min"],
                  "value_target": built["value_target"], "value_mask": built["value_mask"],
                  "segments": [{"episode_key": "k", "packed_start": 0,
                                "packed_end": len(s["token"]), "anchor_offset": None,
                                "anchors": [{"offset": a, "cr": anchor["cr"],
                                             "queries": anchor["queries"]}]}]}
        batch = collate_model_samples([sample])
        self.assertEqual(batch["cr_mask"].tolist(), [False])
        for q, mask, event in zip(anchor["queries"], batch["th_mask"].tolist(),
                                  batch["th_event"].tolist()):
            if q["status"] in ("not_ascertainable", "prevalent"):
                self.assertFalse(mask)
            self.assertFalse(event and not mask)

    def test_an_anchor_outside_its_segment_is_refused(self):
        sample = copy.deepcopy(self.samples[0])
        sample["segments"][0]["anchors"][0]["offset"] = len(sample["input_ids"])
        with self.assertRaisesRegex(ValueError, "anchor"):
            collate_model_samples([sample])

    def test_dataset_refuses_a_shard_bound_to_another_vocabulary(self):
        other = dict(HASHES, numeric_edges="0" * 64)
        rows = [dict(r, artifact_hashes=other) for r in self.rows]
        with self.assertRaises(ArtifactBindingError):
            ModelDataset(rows, representation="gem", target_builder=self.builder,
                         expected_hashes=other)

    def test_epoch_changes_the_anchors_and_the_cache_keeps_windows_consistent(self):
        dataset = ModelDataset(self.rows, representation="gem", target_builder=self.builder,
                               expected_hashes=HASHES)

        def offsets():
            return [[a["offset"] for a in dataset[i]["segments"][0]["anchors"]]
                    for i in range(len(dataset))]

        first = offsets()
        self.assertEqual(first, offsets())
        seen = {json.dumps(first)}
        for epoch in range(1, 5):
            dataset.set_epoch(epoch)
            seen.add(json.dumps(offsets()))
        self.assertGreater(len(seen), 1)
        dataset.set_epoch(0)
        self.assertEqual(first, offsets())


class ModelLossTest(unittest.TestCase):
    """`pretrain.Model` reads the hidden state at EVERY anchor and bins time per head."""

    def _model(self, **weights):
        torch.manual_seed(0)
        return Model(len(VOCAB), len(TARGETS), tiny_mcfg(**weights),
                     n_value_bins=n_value_bins(BLOB))

    def _batch(self, s, anchors):
        """One-sample batch with explicit anchors [(index, cr, [query, ...]), ...]."""
        built = builder().build(s)
        n = len(s["token"])
        sample = {"packed_schema_version": "2.0.0", "input_ids": s["token"],
                  "attention_mask": [1] * n, "pos_min": s["pos_min"],
                  "ntp_target": built["ntp_target"], "ntp_mask": built["ntp_mask"],
                  "ntp_delta_min": built["ntp_delta_min"],
                  "value_target": built["value_target"], "value_mask": built["value_mask"],
                  "segments": [{"episode_key": "k", "packed_start": 0, "packed_end": n,
                                "anchor_offset": None,
                                "anchors": [{"offset": i, "cr": cr, "queries": queries}
                                            for i, cr, queries in anchors]}]}
        return _prepare_batch(collate_model_samples([sample]), CPU)

    @staticmethod
    def _query(status, minutes, query=MAP_65):
        return {"target_idx": query.target_idx, "threshold_bin": query.threshold_bin,
                "direction": query.direction_id, "threshold": query.value,
                "status": status, "minutes": minutes}

    def _hidden(self, model, batch):
        return model.enc(batch.get("soft_token", batch["token"]), batch["pos_min"],
                         batch.get("soft_weight"))

    def test_each_head_bins_the_same_event_on_its_own_grid(self):
        """A crossing five hours after the anchor: hour bin 5 for the threshold head,
        3-hour bin 1 for the competing-risk head."""
        s, a = anchored([(A + 300, "map", 60.0)])
        model = self._model().eval()
        batch = self._batch(s, [(a, {"status": "positive", "cause": 0, "minutes": 300},
                                 [self._query("positive", 300)])])
        with torch.no_grad():
            losses = model(batch)
            h = self._hidden(model, batch)[0, a].unsqueeze(0)
            zero = torch.zeros(1, dtype=torch.long)
            th = model.th.loss(h, zero, torch.tensor([MAP_65.threshold_bin]), zero,
                               torch.tensor([5]), torch.tensor([5]))
            cr = model.cr.loss(h, zero, torch.tensor([1]))
        self.assertAlmostEqual(float(losses["th"]), float(th), places=5)
        self.assertAlmostEqual(float(losses["cr"]), float(cr), places=5)
        self.assertEqual((model.th.n_bins, model.cr.n_bins), (48, 16))

    def test_censoring_at_ten_hours_credits_only_fully_observed_bins(self):
        s, a = anchored([(A + 60, "map", 75.0)], end=("home", A + 600))
        model = self._model().eval()
        batch = self._batch(s, [(a, {"status": "censored", "cause": -1, "minutes": 600},
                                 [self._query("censored", 600)])])
        with torch.no_grad():
            losses = model(batch)
            h = self._hidden(model, batch)[0, a].unsqueeze(0)
            zero = torch.zeros(1, dtype=torch.long)
            logits = model.th._logits(h, zero, torch.tensor([MAP_65.threshold_bin]), zero)
            th = -torch.nn.functional.logsigmoid(-logits.float()[0, :10]).sum()
            _, event_free = model.cr.cif(h)
            cr = -torch.log(event_free[0, 2])            # 3 full 3-hour bins: 0, 1, 2
        self.assertAlmostEqual(float(losses["th"]), float(th), places=5)
        self.assertAlmostEqual(float(losses["cr"]), float(cr), places=5)

    def test_an_event_exactly_at_the_horizon_lands_in_each_heads_last_bin(self):
        s, a = anchored([(A + H, "map", 60.0)])
        model = self._model().eval()
        batch = self._batch(s, [(a, {"status": "positive", "cause": 0, "minutes": H},
                                 [self._query("positive", H)])])
        with torch.no_grad():
            losses = model(batch)
            h = self._hidden(model, batch)[0, a].unsqueeze(0)
            zero = torch.zeros(1, dtype=torch.long)
            th = model.th.loss(h, zero, torch.tensor([MAP_65.threshold_bin]), zero,
                               torch.tensor([47]), torch.tensor([47]))
            cr = model.cr.loss(h, zero, torch.tensor([15]))
        self.assertAlmostEqual(float(losses["th"]), float(th), places=5)
        self.assertAlmostEqual(float(losses["cr"]), float(cr), places=5)

    def test_hidden_state_is_read_at_each_anchor_not_only_the_last_token(self):
        s, a = anchored([(A + 60, "map", 75.0), (A + 300, "map", 60.0)])
        later = a + 1
        model = self._model().eval()
        both = self._batch(s, [
            (a, {"status": "positive", "cause": 0, "minutes": 300},
             [self._query("positive", 300)]),
            (later, {"status": "positive", "cause": 0, "minutes": 240},
             [self._query("positive", 240)]),
        ])
        self.assertEqual(both["last_idx"].tolist(), [a, later])
        singles = [self._batch(s, [(a, {"status": "positive", "cause": 0, "minutes": 300},
                                    [self._query("positive", 300)])]),
                   self._batch(s, [(later, {"status": "positive", "cause": 0, "minutes": 240},
                                    [self._query("positive", 240)])])]
        with torch.no_grad():
            joint = model(both)
            parts = [model(b) for b in singles]
        self.assertNotAlmostEqual(float(parts[0]["th"]), float(parts[1]["th"]), places=4)
        for key in ("th", "cr"):
            mean = (float(parts[0][key]) + float(parts[1][key])) / 2
            self.assertAlmostEqual(float(joint[key]), mean, places=5)

    def test_unsupervised_rows_do_not_change_the_loss(self):
        s, a = anchored([(A + 300, "map", 60.0)])
        model = self._model().eval()
        cr = {"status": "positive", "cause": 0, "minutes": 300}
        plain = self._batch(s, [(a, cr, [self._query("positive", 300)])])
        padded = self._batch(s, [
            (a, cr, [self._query("positive", 300), self._query("prevalent", None),
                     self._query("not_ascertainable", None)]),
            (a + 1, None, [self._query("not_ascertainable", None)]),
            # Censored before one full bin on either grid: nothing was observed.
            (a + 1, {"status": "censored", "cause": -1, "minutes": 30},
             [self._query("censored", 30)]),
        ])
        with torch.no_grad():
            expected, got = model(plain), model(padded)
        for key in ("th", "cr", "total"):
            self.assertAlmostEqual(float(got[key]), float(expected[key]), places=5)

    def test_batch_with_no_supervised_anchor_for_one_head_keeps_every_gradient(self):
        """KTD2 survives multiple anchors: a head with nothing supervised in the batch
        still returns its zero-valued touch term."""
        s, a = anchored([(A + 300, "map", 60.0)])
        cases = {
            "no threshold row supervised": [
                (a, {"status": "positive", "cause": 0, "minutes": 300},
                 [self._query("prevalent", None)])],
            "no competing-risk label": [(a, None, [self._query("positive", 300)])],
            "no anchor at all": [],
            "only sub-bin censoring": [
                (a, {"status": "censored", "cause": -1, "minutes": 20},
                 [self._query("censored", 20)])],
        }
        skipped = {"no threshold row supervised": ("th",),
                   "no competing-risk label": ("cr",),
                   "no anchor at all": ("th", "cr"),
                   "only sub-bin censoring": ("th", "cr")}
        for name, anchors in cases.items():
            model = self._model()
            losses = model(self._batch(s, anchors))
            with self.subTest(case=name):
                for key in ("ntp", "cr", "th", "val", "total"):
                    self.assertTrue(bool(torch.isfinite(losses[key])), key)
                for key in ("th", "cr"):
                    if key in skipped[name]:
                        self.assertEqual(float(losses[key].detach()), 0.0)
                    else:
                        self.assertGreater(float(losses[key].detach()), 0.0)
                losses["total"].backward()
                for pname, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, pname)
                    if pname.startswith(tuple(f"{key}." for key in skipped[name])):
                        self.assertEqual(float(param.grad.abs().max()), 0.0, pname)

    def test_a_label_horizon_longer_than_a_heads_horizon_is_not_an_event_on_that_head(self):
        s, a = anchored([(A + 30 * 60, "map", 60.0)])
        mcfg = tiny_mcfg()
        mcfg["heads"]["competing_risk"].update(n_time_bins=8, horizon_hours=24)
        torch.manual_seed(0)
        model = Model(len(VOCAB), len(TARGETS), mcfg, n_value_bins=n_value_bins(BLOB)).eval()
        batch = self._batch(s, [(a, {"status": "positive", "cause": 0, "minutes": 1800},
                                 [self._query("positive", 1800)])])
        with torch.no_grad():
            losses = model(batch)
            h = self._hidden(model, batch)[0, a].unsqueeze(0)
            _, event_free = model.cr.cif(h)
        self.assertAlmostEqual(float(losses["cr"]), float(-torch.log(event_free[0, 7])),
                               places=5)


class TwentyFourHourPathTest(unittest.TestCase):
    """The 24 h representation rides the same per-anchor contract (one anchor)."""

    def _sample(self, outcomes):
        record = {
            "episode_key": "opaque-24h",
            "artifact_hashes": {"vocabulary": "v", "numeric_edges": "s",
                                "tokenizer_version": "2"},
            "token": [5, 6, 7], "pos_min": [0, 30, 60], "value": [None, None, None],
            "target_eligible": [True, True, True], "anchor_idx": 2, "anchor_min": 60,
            "outcomes": outcomes,
        }
        dataset = ModelDataset([record], representation="decile",
                               target_builder=TargetBuilder(32, 16, 48, {}),
                               expected_hashes={})
        return collate_model_samples([dataset[0]])

    @staticmethod
    def _row(status, hours, target=0, **extra):
        return {"target_idx": target, "status": status, "time_from_anchor_hours": hours,
                "threshold_bin": 5, "direction": "below", **extra}

    def test_threshold_time_is_minutes_from_the_anchor_not_a_competing_risk_grid_bin(self):
        """The engine bug: a crossing at 30.5 h was passed to the hourly threshold head
        as its 3-hour competing-risk bin (10). It now arrives as 1830 minutes and each
        head bins it itself: hour 30, 3-hour bin 10."""
        batch = self._sample([self._row("positive", 30.5)])
        self.assertEqual(batch["th_time_min"].tolist(), [1830])
        self.assertEqual(batch["cr_time_min"].tolist(), [1830])
        self.assertEqual(time_bin(batch["th_time_min"], 48, 48).tolist(), [30])
        self.assertEqual(time_bin(batch["cr_time_min"], 16, 48).tolist(), [10])
        self.assertEqual((batch["th_event"].tolist(), batch["cr_type"].tolist()),
                         ([True], [0]))

    def test_one_anchor_with_the_earliest_event_as_the_competing_risk_label(self):
        batch = self._sample([self._row("positive", 30.5),
                              self._row("competing_event", 12.0, target=1, cause_idx=3),
                              self._row("censored", 5.0, target=2)])
        self.assertEqual(batch["anchor_idx"].tolist(), [2])
        self.assertEqual((batch["cr_mask"].tolist(), batch["cr_type"].tolist(),
                          batch["cr_time_min"].tolist()), ([True], [3], [720]))
        self.assertEqual(batch["th_anchor"].tolist(), [0])

    def test_no_event_keeps_the_longest_observed_interval(self):
        batch = self._sample([self._row("censored", 5.25),
                              self._row("negative", 48.0, target=1)])
        self.assertEqual((batch["cr_type"].tolist(), batch["cr_time_min"].tolist()),
                         ([-1], [2880]))

    def test_unlabelled_episode_has_an_anchor_and_no_supervision(self):
        batch = self._sample([self._row("prevalent", None),
                              self._row("not_ascertainable", None, target=1)])
        self.assertEqual(batch["anchor_idx"].tolist(), [2])
        self.assertEqual(batch["cr_mask"].tolist(), [False])
        self.assertEqual(batch["th_mask"].numel(), 0)


class TauSamplingConfigTest(unittest.TestCase):
    def test_model_config_tau_sampling_is_what_builds_the_in_stream_targets(self):
        registry = _registry()
        built = in_stream_target_builder(vocab_blob=BLOB, mcfg=MODEL_CFG, dcfg=DATA_CFG,
                                         thresholds=registry, vocab_size=len(VOCAB),
                                         value_stats=VALUE_STATS, run_seed=3)
        self.assertEqual(built.mode, "gem_tte")
        self.assertEqual(built.in_stream.grid.tau_sampling, "empirical_bins")
        self.assertEqual(MODEL_CFG["heads"]["threshold_hazard"]["tau_sampling"],
                         "empirical_bins")
        self.assertEqual(built.horizon_hours, registry["label_rule"]["horizon_hours"])
        self.assertEqual(built.in_stream.required_measurement_within_hours_of_horizon, 12)
        self.assertEqual(built.in_stream.baseline_lookback_hours, 6)
        self.assertEqual(built.in_stream.anchors_per_window,
                         MODEL_CFG["in_stream"]["anchors_per_window"])
        self.assertEqual(built.in_stream.queries_per_anchor,
                         MODEL_CFG["in_stream"]["queries_per_anchor"])

    def test_an_unknown_tau_sampling_fails_closed(self):
        mcfg = copy.deepcopy(MODEL_CFG)
        mcfg["heads"]["threshold_hazard"]["tau_sampling"] = "uniform_values"
        with self.assertRaisesRegex(ThresholdGridError, "tau_sampling"):
            in_stream_target_builder(vocab_blob=BLOB, mcfg=mcfg, dcfg=DATA_CFG,
                                     thresholds=_registry(), vocab_size=len(VOCAB),
                                     value_stats=VALUE_STATS)
        del mcfg["heads"]["threshold_hazard"]["tau_sampling"]
        with self.assertRaisesRegex(ThresholdGridError, "tau_sampling"):
            in_stream_target_builder(vocab_blob=BLOB, mcfg=mcfg, dcfg=DATA_CFG,
                                     thresholds=_registry(), vocab_size=len(VOCAB),
                                     value_stats=VALUE_STATS)


class SyntheticSiteTrainingTest(unittest.TestCase):
    """Verification: three CPU optimizer updates on the synthetic site's
    full-hospitalization shard give finite, supervised losses for every head."""

    @classmethod
    def setUpClass(cls):
        try:
            from test_gem_artifact import MAX_TOKENS, _build_site
        except ImportError:
            from tests.test_gem_artifact import MAX_TOKENS, _build_site
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import (
            FIXTURE_DATA_CONFIG,
            FIXTURE_POLICY,
            SYNTHETIC_SITE,
        )

        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            site, episodes, cfg = _build_site(work)
            out = work / "output/intermediate_phi/site"
            kw = {"episodes": episodes, "artifact_policy": FIXTURE_POLICY}
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, **kw)
            cls.blob = json.loads((out / "vocab.json").read_text())
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, cls.blob,
                          trajectory="hospitalization", max_tokens=MAX_TOKENS, **kw)
            cls.rows = pl.read_parquet(out / "gem_events.parquet").to_dicts()
        finally:
            os.chdir(old_cwd)
        cls.dcfg = FIXTURE_DATA_CONFIG
        full = load_thresholds()
        cls.registry = {**full, **{kind: tuple(t for t in full[kind] if t.concept == "map")
                                   for kind in THRESHOLD_KINDS}}
        cls.vocab_size = len(cls.blob["vocab"])
        grid = ThresholdGrid(cls.blob, cls.dcfg["target_concepts"], cls.registry)
        cls.value_stats = {token: (75.0, 10.0) for token in grid.token_target}
        cls.mcfg = tiny_mcfg()
        cls.builder = in_stream_target_builder(
            vocab_blob=cls.blob, mcfg=cls.mcfg, dcfg=cls.dcfg, thresholds=cls.registry,
            vocab_size=cls.vocab_size, value_stats=cls.value_stats, run_seed=42)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _dataset(self, rows=None):
        return ModelDataset(rows if rows is not None else self.rows, representation="gem",
                            target_builder=self.builder,
                            expected_hashes=artifact_binding(self.blob))

    def test_three_cpu_updates_give_finite_supervised_losses_for_every_head(self):
        try:
            from test_ddp_multihead import tiny_tcfg
        except ImportError:
            from tests.test_ddp_multihead import tiny_tcfg

        dataset = self._dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False,
                                             collate_fn=collate_model_samples)
        torch.manual_seed(0)
        model = Model(self.vocab_size, len(self.dcfg["target_concepts"]), self.mcfg,
                      n_value_bins=n_value_bins(self.blob))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as ckpt:
            tcfg = TrainConfig({}, tiny_tcfg(ckpt), self.mcfg, 3)
            log, samples, _, _, updates = _train_one_epoch(
                model, loader, opt, scheduler, 0, tcfg, CPU, rank=0, max_updates=3)
        self.assertEqual((updates, samples), (3, 12))
        for name in ("loss_ntp", "loss_cr", "loss_th", "loss_val", "loss_total"):
            values = getattr(log, name)
            with self.subTest(head=name):
                self.assertEqual(len(values), 3)
                self.assertTrue(all(math.isfinite(v) and v > 0.0 for v in values), values)

    def test_label_statuses_occur_on_the_synthetic_site(self):
        dataset = self._dataset()
        # Every candidate anchor of every stay, so the coverage does not depend on a draw.
        every = in_stream_target_builder(
            vocab_blob=self.blob, mcfg={**self.mcfg, "in_stream": {
                "anchors_per_window": 10_000, "queries_per_anchor": 2}},
            dcfg=self.dcfg, thresholds=self.registry, vocab_size=self.vocab_size,
            value_stats=self.value_stats, run_seed=42)
        builds = [every.build(stream, epoch=0) for stream in dataset._gem_streams.values()]
        report = anchor_status_shares(builds)
        self.assertGreater(report["anchors"], 0)
        counts = report["cause_labels"]["counts"]
        # Even stays dip below 65 after ICU hour 26, odd stays never do, MAP charting
        # stops at ICU hour 69, and synth-000 expires 20 h after its last ward event.
        for status in ("positive", "prevalent", "negative", "not_ascertainable",
                       "competing_event"):
            self.assertGreater(counts.get(status, 0), 0, status)
        self.assertAlmostEqual(sum(report["threshold_queries"]["shares"].values()), 1.0)

    def test_long_stay_windows_share_one_set_of_whole_stay_labels(self):
        try:
            from test_gem_artifact import LONG_STAY
        except ImportError:
            from tests.test_gem_artifact import LONG_STAY

        rows = [r for r in self.rows if r["hosp_id"] == LONG_STAY]
        self.assertGreater(len(rows), 100)
        dataset = self._dataset(rows)
        built = self.builder.build(dataset._gem_streams[LONG_STAY], epoch=0)
        total = sum(len(dataset[i]["segments"][0]["anchors"]) for i in range(len(dataset)))
        self.assertEqual(total, len(built["anchors"]))
        self.assertGreaterEqual(total, len(rows))


if __name__ == "__main__":
    unittest.main()
