"""U8 (KTD3, KTD13): zero-shot threshold evaluation on (anchor, registered threshold) pairs.

Vocabularies are hand-built from the physician segment CSV (clinical arm) and from
synthetic quantile edges (decile arms), as in tests/test_threshold_grid.py. Streams are
synthetic; the model is a tiny randomly initialised `pretrain.Model` — nothing here is a
result, only the machinery.
"""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.segments import SPECIAL, bin_index, load_csv_segments, segments_from_edges
from src.data.threshold_grid import (
    DEATH_TOKEN,
    THRESHOLD_KINDS,
    ThresholdGrid,
    load_thresholds,
)
from src.eval import threshold_eval as te
from src.train.pretrain import Model

ROOT = Path(__file__).resolve().parents[1]
DATA_CFG = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
TARGETS = DATA_CFG["target_concepts"]
DIRECTIONS = {c["name"]: c["direction"] for c in TARGETS}
CONCEPTS = ("map", "lactate")
ROLES = {"fit": "train", "selection": "validation", "calibration": "calibration",
         "evaluation": "internal_test"}
PROBE = {"epochs": 60, "lr": 5.0e-2, "default": {"wd": 1.0e-4}, "grid": {"wd": [1.0e-4]}}


def registry() -> dict:
    full = load_thresholds()
    return {**full, **{kind: tuple(t for t in full[kind] if t.concept in CONCEPTS)
                       for kind in THRESHOLD_KINDS}}


def clinical_segments() -> dict:
    binning = DATA_CFG["value_binning"]
    return load_csv_segments(ROOT / binning["segment_source"], list(CONCEPTS),
                             binning["forced_edges"], DIRECTIONS)


def decile_segments(*, forced: bool) -> dict:
    forced_edges = DATA_CFG["value_binning"]["forced_edges"] if forced else {}
    return {
        "map": segments_from_edges([58.0, 62.0, 68.0, 74.0, 81.0, 90.0],
                                   forced_edges.get("map", ()), "below"),
        "lactate": segments_from_edges([0.8, 1.1, 1.7, 2.6, 4.4],
                                       forced_edges.get("lactate", ()), "above"),
    }


def vocab_blob(segments: dict) -> dict:
    vocab = dict(SPECIAL)
    for concept, segs in segments.items():
        for b in range(len(segs)):
            vocab[f"{concept}={b}"] = len(vocab)
    for token in ("ADMISSION//ed", "DISCHARGE//home", DEATH_TOKEN):
        vocab[token] = len(vocab)
    return {
        "vocab": vocab,
        "segments": segments,
        "manifest": {"tokenizer_version": 2},
        "concept_sources": {"tables": {"map": ["vitals"], "lactate": ["labs"]},
                            "treatment_sources": ["meds"]},
    }


def grid_of(segments: dict) -> ThresholdGrid:
    return ThresholdGrid(vocab_blob(segments), TARGETS, registry())


def synthetic_stays(n: int = 160, seed: int = 3) -> list[dict]:
    """Raw stays (minute, concept, value) with every label status represented: low-MAP
    stays (positive / prevalent), early non-death discharges (censored), deaths
    (competing), and stays whose MAP charting stops (not ascertainable)."""
    rng = np.random.default_rng(seed)
    partitions = ("train", "validation", "calibration", "internal_test")
    stays = []
    for i in range(n):
        base = rng.uniform(58.0, 85.0)
        slope = rng.normal(-0.15, 0.2)
        length_h = int(rng.choice([20, 70, 70, 70]))
        stop_map_h = length_h if rng.random() > 0.2 else 10
        events = []
        for hour in range(1, length_h):
            if hour <= stop_map_h:
                events.append((hour * 60, "map",
                               float(base + slope * hour + rng.normal(0, 3))))
            if hour % 4 == 0:
                events.append((hour * 60 + 5, "lactate",
                               float(max(0.3, 1.5 + (75 - base) / 8 + rng.normal(0, 0.6)))))
        disposition = "expired" if rng.random() < 0.15 else "home"
        stays.append({"episode_key": f"stay-{i:04d}", "partition": partitions[i % 4],
                      "events": events, "end": (disposition, length_h * 60)})
    return stays


def encode(stays: list[dict], segments: dict) -> list[dict]:
    """One arm's token streams of the same stays (same minutes, the arm's own bins)."""
    vocab = vocab_blob(segments)["vocab"]
    streams = []
    for stay in stays:
        token, pos, value = [SPECIAL["<bos>"], vocab["ADMISSION//ed"]], [0, 0], [None, None]
        for minute, concept, v in stay["events"]:
            token.append(vocab[f"{concept}={bin_index(v, segments[concept])}"])
            pos.append(minute)
            value.append(v)
        disposition, minute = stay["end"]
        token += [vocab[f"DISCHARGE//{disposition}"], SPECIAL["<eos>"]]
        pos += [minute, minute]
        value += [None, None]
        streams.append({"episode_key": stay["episode_key"], "partition": stay["partition"],
                        "token": token, "pos_min": pos, "value": value})
    return streams


def tiny_model(vocab_size: int, n_value_bins: int, seed: int = 0) -> Model:
    mcfg = {
        "trunk": {"d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0,
                  "rope_base": 10000.0, "tied_embeddings": False},
        "heads": {
            "next_event": {"enabled": True, "weight": 0.2},
            "competing_risk": {"enabled": True, "weight": 1.0, "n_time_bins": 8},
            "threshold_hazard": {"enabled": True, "weight": 1.0, "n_time_bins": 8,
                                 "horizon_hours": 48, "threshold_embed_dim": 4},
            "value_regression": {"enabled": True, "weight": 0.5},
        },
    }
    torch.manual_seed(seed)
    return Model(vocab_size, len(TARGETS), mcfg, n_value_bins=n_value_bins)


def claims_cfg(**evaluation) -> dict:
    cfg = te.load_claims_config()
    cfg["evaluation"].update({"horizons_hours": [24, 48], "anchors_per_stay": 2},
                             **evaluation)
    return cfg


class EdgeClassificationTest(unittest.TestCase):
    def test_map_65_is_on_edge_for_the_clinical_arm_and_off_edge_for_plain_deciles(self):
        edges = te.classify_edges({"clinical_soft": grid_of(clinical_segments()),
                                   "global_deciles": grid_of(decile_segments(forced=False)),
                                   "deciles_plus_soft": grid_of(decile_segments(forced=True))})
        key = te.threshold_key("decision", "map", 65.0, "below")
        self.assertTrue(edges["clinical_soft"][key]["on_edge"])
        self.assertFalse(edges["global_deciles"][key]["on_edge"])
        self.assertEqual(edges["global_deciles"][key]["distance"], 3.0)   # nearest edge 62
        self.assertTrue(edges["deciles_plus_soft"][key]["on_edge"])
        # Every registered decision and control threshold is classified in every arm.
        expected = {te.threshold_key(t.kind, t.concept, t.value, t.direction)
                    for kind in ("decision", "control") for t in registry()[kind]}
        for arm, rows in edges.items():
            self.assertEqual(set(rows), expected, arm)

    def test_a_control_on_an_edge_in_any_arm_is_refused_with_the_arm_named(self):
        segments = decile_segments(forced=False)
        segments["map"] = segments_from_edges([58.0, 63.0, 68.0], (), "below")
        with self.assertRaisesRegex(te.ThresholdEvalError,
                                    r"control map 63.*on a bin edge in arm 'odd_deciles'"):
            te.classify_edges({"clinical_soft": grid_of(clinical_segments()),
                               "odd_deciles": grid_of(segments)})


class EvaluationSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stays = synthetic_stays()
        cls.clinical = grid_of(clinical_segments())
        cls.deciles = grid_of(decile_segments(forced=False))
        cls.cfg = claims_cfg()

    def build(self, grid, segments):
        return te.evaluation_set(encode(self.stays, segments), grid, registry(),
                                 horizons_hours=[24, 48], anchors_per_stay=2, seed=11)

    def test_the_set_is_deterministic_and_identical_across_arms(self):
        a = self.build(self.clinical, clinical_segments())
        b = self.build(self.clinical, clinical_segments())
        d = self.build(self.deciles, decile_segments(forced=False))
        key = lambda s: [(p["pair_id"], p["threshold"], p["horizon_hours"], p["status"])
                         for p in s["pairs"]]
        self.assertEqual(key(a), key(b))
        # Same stays, same anchor minutes, labels at the exact threshold: the decile arm's
        # pairs and labels are the clinical arm's (its bins differ, its labels cannot).
        self.assertEqual(key(a), key(d))
        n_thresholds = len(registry()["decision"]) + len(registry()["control"])
        self.assertEqual(len(a["pairs"]), len(a["anchors"]) * n_thresholds * 2)

    def test_non_scorable_states_are_counted_and_never_scored_as_negative(self):
        built = self.build(self.clinical, clinical_segments())
        counts = te.status_counts(built["pairs"], partition="internal_test")
        totals = {}
        for by_horizon in counts.values():
            for by_status in by_horizon.values():
                for status, n in by_status.items():
                    totals[status] = totals.get(status, 0) + n
        for status in ("positive", "negative", "prevalent", "censored",
                       "competing_event", "not_ascertainable"):
            self.assertGreater(totals.get(status, 0), 0, status)
        for pair in built["pairs"]:
            if pair["status"] in te.SCORABLE:
                self.assertEqual(pair["label"], int(pair["status"] == "positive"))
            else:
                self.assertIsNone(pair["label"], pair["status"])


class RunEvaluationTest(unittest.TestCase):
    """The tiny end-to-end check: a randomly initialised Model, no training."""

    @classmethod
    def setUpClass(cls):
        cls.stays = synthetic_stays()
        cls.segments = clinical_segments()
        cls.blob = vocab_blob(cls.segments)
        cls.grid = ThresholdGrid(cls.blob, TARGETS, registry())
        cls.streams = encode(cls.stays, cls.segments)
        from src.data.segments import n_value_bins
        cls.model = tiny_model(len(cls.blob["vocab"]), n_value_bins(cls.blob))
        cls.cfg = claims_cfg()

    def evaluate(self, objective_arm, **kwargs):
        return te.evaluate_run(self.model, self.streams, self.blob, registry(), self.cfg,
                               target_concepts=TARGETS, objective_arm=objective_arm,
                               site="SYNTH", roles=ROLES, probe_cfg=PROBE, **kwargs)

    def test_combined_arm_has_head_and_probe_rows_with_finite_metrics(self):
        result = self.evaluate("full")
        thresholds = {te.threshold_key(t.kind, t.concept, t.value, t.direction)
                      for kind in ("decision", "control") for t in registry()[kind]}
        rows = result["summary"]["rows"]
        # One row per scorer x registered threshold x horizon, rollout included.
        self.assertEqual(len(rows), 3 * len(thresholds) * 2)
        self.assertEqual({r["scorer"] for r in rows}, {"head", "probe", "rollout"})
        self.assertEqual({r["threshold"] for r in rows if r["scorer"] == "head"}, thresholds)
        scores = result["scores"]
        for scorer in ("head", "probe"):
            probs = np.array([s["prob"] for s in scores if s["scorer"] == scorer])
            self.assertGreater(len(probs), 0, scorer)
            self.assertTrue(np.all(np.isfinite(probs)) and np.all((probs >= 0) & (probs <= 1)))
        panel = te.cell_metrics(scores)
        evaluable = [c for c in panel if c["status"] == "evaluable"]
        self.assertTrue(evaluable)
        for cell in evaluable:
            for metric in ("auroc", "auprc", "ici"):
                self.assertTrue(math.isfinite(cell[metric]), (cell["threshold"], metric))
        # Only evaluation-partition pairs with a scorable status are scored.
        self.assertTrue(all(s["label"] in (0, 1) for s in scores))
        counts = result["summary"]["status_counts"]
        for key, by_horizon in counts.items():
            for horizon, by_status in by_horizon.items():
                n_scored = sum(1 for s in scores if s["scorer"] == "head"
                               and s["threshold"] == key
                               and s["horizon_hours"] == float(horizon))
                self.assertEqual(n_scored, by_status.get("positive", 0)
                                 + by_status.get("negative", 0))

    def test_the_next_token_arm_is_scored_by_the_probe_on_every_registered_threshold(self):
        result = self.evaluate("next_token_only")
        rows = result["summary"]["rows"]
        self.assertNotIn("head", {r["scorer"] for r in rows})
        probe = {(r["threshold"], r["horizon_hours"]) for r in rows if r["scorer"] == "probe"}
        expected = {(te.threshold_key(t.kind, t.concept, t.value, t.direction), h)
                    for kind in ("decision", "control") for t in registry()[kind]
                    for h in (24.0, 48.0)}
        self.assertEqual(probe, expected)
        self.assertFalse(any(s["scorer"] == "head" for s in result["scores"]))

    def test_the_probe_leaves_the_trunk_without_gradients_and_unchanged(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.evaluate("full")
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)

    def test_rollouts_on_the_imposed_clock_read_not_evaluable_and_stay_in_the_table(self):
        result = self.evaluate("full")
        rollout = [r for r in result["summary"]["rows"] if r["scorer"] == "rollout"]
        self.assertTrue(rollout)
        for row in rollout:
            self.assertEqual(row["status"], te.NOT_EVALUABLE)
            self.assertIn("pos_step_min", row["reason"])
        self.assertFalse(any(s["scorer"] == "rollout" for s in result["scores"]))

    def test_a_sampler_that_dates_events_is_scored_on_edge_and_refused_off_edge(self):
        vocab = self.blob["vocab"]
        low = vocab[f"map={bin_index(50.0, self.segments['map'])}"]
        high = vocab[f"map={bin_index(80.0, self.segments['map'])}"]

        class DatedSampler:
            clock_imposed = False
            reason = None

            def __call__(self, prompt_ids, prompt_pos, *, n, seed):
                # Half the rollouts cross MAP 65 at hour 30, half stay at 80.
                return [{"token_ids": [low if r % 2 else high], "minutes": [30 * 60]}
                        for r in range(n)]

        result = self.evaluate("full", sampler=DatedSampler())
        rows = {(r["threshold"], r["horizon_hours"]): r for r in result["summary"]["rows"]
                if r["scorer"] == "rollout"}
        on = te.threshold_key("decision", "map", 65.0, "below")
        off = te.threshold_key("control", "map", 63.0, "below")
        self.assertEqual(rows[(on, 48.0)]["status"], "evaluable")
        self.assertEqual(rows[(off, 48.0)]["status"], te.NOT_EVALUABLE)
        self.assertIn("straddles", rows[(off, 48.0)]["reason"])
        probs = {(s["threshold"], s["horizon_hours"]): s["prob"]
                 for s in result["scores"] if s["scorer"] == "rollout"}
        self.assertEqual(probs[(on, 48.0)], 0.5)       # crossed at 30 h, inside 48 h
        self.assertEqual(probs[(on, 24.0)], 0.0)       # ... but not inside 24 h

    def test_run_outputs_round_trip_and_the_summary_carries_no_identifier(self):
        result = self.evaluate("full")
        with tempfile.TemporaryDirectory() as tmp:
            te.write_run_outputs(tmp, result)
            back = te.read_run_outputs(tmp)
        self.assertEqual(len(back["scores"]), len(result["scores"]))
        self.assertEqual(back["summary"], result["summary"])
        text = str(result["summary"])
        for stay in self.stays:
            self.assertNotIn(stay["episode_key"], text)
        self.assertNotIn("pair_id", text)

    def test_a_horizon_off_the_head_grid_is_refused(self):
        cfg = claims_cfg(horizons_hours=[5, 48])
        with self.assertRaisesRegex(te.ThresholdEvalError, "bin boundary"):
            te.evaluate_run(self.model, self.streams, self.blob, registry(), cfg,
                            target_concepts=TARGETS, objective_arm="full", site="SYNTH",
                            roles=ROLES, probe_cfg=PROBE)


class ClaimsConfigTest(unittest.TestCase):
    def test_the_registered_config_loads_and_bad_ones_are_refused(self):
        cfg = te.load_claims_config()
        self.assertEqual(cfg["multiplicity"]["method"], "benjamini_hochberg")
        self.assertEqual(cfg["min_seeds_per_arm"], 3)
        self.assertIn(load_thresholds()["label_rule"]["horizon_hours"],
                      cfg["evaluation"]["horizons_hours"])
        raw = yaml.safe_load(te.CLAIMS_PATH.read_text())
        for mutate, pattern in (
                (lambda r: r["primary_metrics"].update(calibration=["brier_skill"]), "metric"),
                (lambda r: r["multiplicity"].update(alpha=1.5), "alpha"),
                (lambda r: r.update(min_seeds_per_arm=1), "min_seeds"),
                (lambda r: r["claim_1"].update(attribution_in_rule="yes"), "attribution"),
                (lambda r: r.pop("thresholds", None) or r.update(thresholds=[]), "threshold")):
            broken = copy.deepcopy(raw)
            mutate(broken)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "claims.yaml"
                path.write_text(yaml.safe_dump(broken))
                with self.assertRaisesRegex(te.ThresholdEvalError, pattern):
                    te.load_claims_config(path)


if __name__ == "__main__":
    unittest.main()
