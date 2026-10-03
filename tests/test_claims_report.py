"""U8 (R31, R35, R39; KTD12, KTD13): the claims report over finished runs.

Every run output here is synthetic and written in the `threshold_eval` layout: labels are
shared by every run (the same evaluation set), and each arm's predictions are generated
with a chosen separation per threshold, so a tie or a win is built in. Nothing here is a
result.
"""

import copy
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from scipy import stats

from src.data.threshold_grid import THRESHOLD_KINDS, load_thresholds
from src.eval import claims_report as cr
from src.eval import threshold_eval as te

MAP_65 = te.threshold_key("decision", "map", 65.0, "below")
MAP_63 = te.threshold_key("control", "map", 63.0, "below")
VOCAB = {"clinical_soft": "a" * 64, "global_deciles": "b" * 64,
         "deciles_clinical_query": "c" * 64}
ON_EDGE = {"clinical_soft": {MAP_65: True, MAP_63: False},
           "global_deciles": {MAP_65: False, MAP_63: False},
           "deciles_clinical_query": {MAP_65: False, MAP_63: False}}
N_STAYS = 300


def registry() -> dict:
    full = load_thresholds()
    keep = {("decision", 65.0), ("control", 63.0)}
    return {**full, **{kind: tuple(t for t in full[kind]
                                   if t.concept == "map" and (kind, t.value) in keep)
                       for kind in THRESHOLD_KINDS}}


def config(**overrides) -> dict:
    cfg = te.load_claims_config()
    cfg["bootstrap"] = {"n_resamples": 200, "seed": 5, "confidence": 0.95}
    cfg["evaluation"] = dict(cfg["evaluation"], horizons_hours=[24, 48])
    for key, value in overrides.items():
        cfg[key] = dict(cfg[key], **value) if isinstance(value, dict) else value
    return cfg


def labels(threshold: str, horizon: float) -> np.ndarray:
    """Shared labels of one cell: 2 anchors per stay, ~30% positive."""
    rng = np.random.default_rng([zlib.crc32(f"{threshold}|{horizon}".encode()), 1])
    return (rng.random(2 * N_STAYS) < 0.3).astype(int)


def write_run(root: Path, tokenization: str, objective: str, seed: int, *,
              quality: dict, noise_key: str | None = None, budget: str = "full",
              claim_bearing: bool = True, rollout: bool = False,
              bin_counts: dict | None = None, positives: dict | None = None,
              status_extra: dict | None = None, vocab_hash: str | None = None) -> Path:
    """One run's directory: run_spec.json + threshold_eval/{scores.parquet, summary.json}.

    `quality[(scorer, threshold)]` is the separation of that scorer's predictions (0 = no
    signal); runs with the same `noise_key` and seed get identical noise, so equal
    quality gives identical predictions (an exact tie)."""
    run_id = f"{tokenization}.{objective}.s{seed}.{budget}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_spec.json").write_text(json.dumps({
        "run_id": run_id, "tokenization_arm": tokenization, "objective_arm": objective,
        "seed": seed, "budget": budget, "claim_bearing": claim_bearing,
        "vocab_hash": vocab_hash or VOCAB[tokenization],
        "checkpoint": f"checkpoints/{run_id}/ckpt.pt"}))
    scorers = (["head"] if objective not in ("next_token_only", "minus_threshold") else []) \
        + ["probe"] + (["rollout"] if rollout else [])
    scores, rows, counts = [], [], {}
    for threshold in (MAP_65, MAP_63):
        kind, concept, value, direction = threshold.split(":")
        for horizon in (24.0, 48.0):
            y = labels(threshold, horizon)
            if positives and threshold in positives:
                y = np.zeros_like(y)
                y[:positives[threshold]] = 1
            counts.setdefault(threshold, {})[f"{horizon:g}"] = {
                "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
                **(status_extra or {"censored": 40, "not_ascertainable": 3,
                                    "competing_event": 12, "prevalent": 25})}
            for scorer in ("head", "probe", "rollout"):
                if scorer not in scorers:
                    continue
                rows.append({"scorer": scorer, "threshold": threshold, "kind": kind,
                             "concept": concept, "value": float(value),
                             "direction": direction, "horizon_hours": horizon,
                             "status": "evaluable", "reason": None})
                q = quality.get((scorer, threshold), 0.0)
                noise = np.random.default_rng(
                    [zlib.crc32(f"{noise_key or tokenization}|{scorer}|{threshold}|"
                                f"{horizon}".encode()), seed]).normal(0.0, 1.0, len(y))
                p = 1 / (1 + np.exp(-(q * (2 * y - 1) + noise - 0.8)))
                for i in range(len(y)):
                    scores.append({"pair_id": f"p{i:05d}", "cluster_id": f"c{i // 2:05d}",
                                   "scorer": scorer, "threshold": threshold, "kind": kind,
                                   "concept": concept, "value": float(value),
                                   "direction": direction, "horizon_hours": horizon,
                                   "label": int(y[i]), "prob": float(p[i])})
            if not rollout:
                rows.append({"scorer": "rollout", "threshold": threshold, "kind": kind,
                             "concept": concept, "value": float(value),
                             "direction": direction, "horizon_hours": horizon,
                             "status": te.NOT_EVALUABLE,
                             "reason": "rollouts from src/model/generate.py: imposed clock"})
    edges = [{"kind": key.split(":")[0], "concept": "map",
              "value": float(key.split(":")[2]), "direction": "below",
              "on_edge": ON_EDGE[tokenization][key],
              "distance": 0.0 if ON_EDGE[tokenization][key] else 2.0,
              "nearest_edge": None, "threshold_bin": 3} for key in (MAP_65, MAP_63)]
    summary = {"version": 1, "vocabulary": vocab_hash or VOCAB[tokenization],
               "numeric_edges": "e" * 64, "objective_arm": objective,
               "evaluation_partition": "internal_test", "horizons_hours": [24.0, 48.0],
               "label_horizon_hours": 48.0, "edge_table": edges,
               "bin_counts": bin_counts or {"map": 9}, "status_counts": counts, "rows": rows}
    te.write_run_outputs(run_dir, {"scores": scores, "summary": summary})
    return run_dir


def arm(root, tokenization, objective, quality, *, seeds=(1, 2, 3), **kwargs):
    return [write_run(root, tokenization, objective, s, quality=quality, **kwargs)
            for s in seeds]


def report_of(run_dirs, **overrides):
    runs = cr.load_runs(run_dirs)
    return cr.build_report(runs, config(**overrides), registry())


STRONG, WEAK = 2.5, 0.4


class Claim1Test(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def claim1(self, primary_on, primary_off, decile_on, decile_off, *, shared_noise=True):
        noise = "shared" if shared_noise else None
        dirs = arm(self.tmp, "clinical_soft", "full",
                   {("head", MAP_65): primary_on, ("head", MAP_63): primary_off},
                   noise_key=noise)
        dirs += arm(self.tmp, "global_deciles", "full",
                    {("head", MAP_65): decile_on, ("head", MAP_63): decile_off},
                    noise_key=noise)
        return report_of(dirs)["claims"]["claim_1"]

    def test_ae6_a_tie_on_edge_and_off_edge_is_not_supported(self):
        claim = self.claim1(WEAK, WEAK, WEAK, WEAK)
        self.assertEqual(claim["status"], cr.NOT_SUPPORTED)
        self.assertTrue(any("matches" in reason for reason in claim["reasons"]),
                        claim["reasons"])
        gains = [c for c in claim["comparisons"] if c["test"] == "on_edge_gain"]
        self.assertEqual({c["metric"] for c in gains}, {"auroc", "auprc", "ici"})
        self.assertFalse(any(c["significant"] for c in gains))

    def test_a_win_on_edge_that_ties_off_edge_is_supported(self):
        claim = self.claim1(STRONG, WEAK, WEAK, WEAK)
        self.assertEqual(claim["status"], cr.SUPPORTED, claim["reasons"])
        by = {(c["test"], c["metric"]): c for c in claim["comparisons"]}
        self.assertTrue(by[("on_edge_gain", "auroc")]["significant"])
        self.assertTrue(by[("edge_specific_gain", "auroc")]["significant"])
        self.assertGreater(by[("on_edge_gain", "auroc")]["interval"][0], 0.0)

    def test_an_equal_win_on_and_off_edge_is_not_specific_to_the_edges(self):
        claim = self.claim1(STRONG, STRONG, WEAK, WEAK)
        self.assertEqual(claim["status"], cr.NOT_SUPPORTED)
        self.assertTrue(any("not specific to the edges" in r for r in claim["reasons"]),
                        claim["reasons"])
        by = {(c["test"], c["metric"]): c for c in claim["comparisons"]}
        self.assertTrue(by[("on_edge_gain", "auroc")]["significant"])
        self.assertFalse(by[("edge_specific_gain", "auroc")]["significant"])

    def test_a_win_without_matched_bin_count_is_not_supported(self):
        dirs = arm(self.tmp, "clinical_soft", "full",
                   {("head", MAP_65): STRONG, ("head", MAP_63): WEAK})
        dirs += arm(self.tmp, "global_deciles", "full",
                    {("head", MAP_65): WEAK, ("head", MAP_63): WEAK}, bin_counts={"map": 7})
        claim = report_of(dirs)["claims"]["claim_1"]
        self.assertEqual(claim["status"], cr.NOT_SUPPORTED)
        self.assertTrue(any("matched bin count" in r for r in claim["reasons"]))

    def test_fewer_than_three_seeds_is_incomplete_and_never_averaged(self):
        dirs = arm(self.tmp, "clinical_soft", "full",
                   {("head", MAP_65): STRONG, ("head", MAP_63): WEAK})
        dirs += arm(self.tmp, "global_deciles", "full",
                    {("head", MAP_65): WEAK, ("head", MAP_63): WEAK}, seeds=(1, 2))
        report = report_of(dirs)
        claim = report["claims"]["claim_1"]
        self.assertEqual(claim["status"], cr.INCOMPLETE)
        self.assertTrue(any("2 of 3 seeds" in r for r in claim["reasons"]), claim["reasons"])
        arms = {(a["tokenization_arm"], a["objective_arm"]): a for a in report["arms"]}
        self.assertEqual(arms[("global_deciles", "full")]["status"], cr.INCOMPLETE)
        decile_cells = [c for c in report["cells"] if c["tokenization_arm"] == "global_deciles"]
        self.assertTrue(decile_cells)
        for cell in decile_cells:
            self.assertEqual(cell["status"], cr.INCOMPLETE)
            self.assertNotIn("auroc", cell)

    def test_the_attribution_arm_is_reported_and_enters_the_rule_only_when_flagged(self):
        dirs = arm(self.tmp, "clinical_soft", "full",
                   {("head", MAP_65): STRONG, ("head", MAP_63): WEAK}, noise_key="n")
        dirs += arm(self.tmp, "global_deciles", "full",
                    {("head", MAP_65): WEAK, ("head", MAP_63): WEAK}, noise_key="n")
        # The query grid, not the input tokens, carries the gain: the attribution arm
        # matches the primary arm on-edge.
        dirs += arm(self.tmp, "deciles_clinical_query", "full",
                    {("head", MAP_65): STRONG, ("head", MAP_63): WEAK}, noise_key="n")
        off = report_of(dirs)["claims"]["claim_1"]
        self.assertEqual(off["status"], cr.SUPPORTED)
        attribution = [c for c in off["descriptive"] if c["comparator"] ==
                       "deciles_clinical_query"]
        self.assertTrue(attribution)
        on = report_of(dirs, claim_1={"attribution_in_rule": True})["claims"]["claim_1"]
        self.assertEqual(on["status"], cr.NOT_SUPPORTED)
        self.assertTrue(any("deciles_clinical_query" in r for r in on["reasons"]))


class Claim2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def runs(self, head, next_probe, *, rollout=False):
        quality = {("head", MAP_65): head, ("head", MAP_63): head,
                   ("probe", MAP_65): WEAK, ("probe", MAP_63): WEAK}
        dirs = arm(self.tmp, "clinical_soft", "full", quality, noise_key="n")
        dirs += arm(self.tmp, "clinical_soft", "next_token_only",
                    {("probe", MAP_65): next_probe, ("probe", MAP_63): next_probe,
                     ("rollout", MAP_65): next_probe, ("rollout", MAP_63): next_probe},
                    noise_key="n", rollout=rollout)
        dirs += arm(self.tmp, "clinical_soft", "minus_value", quality, noise_key="n")
        return dirs

    def test_combined_head_beating_the_next_token_probe_is_supported(self):
        claim = report_of(self.runs(STRONG, WEAK))["claims"]["claim_2"]
        self.assertEqual(claim["status"], cr.SUPPORTED, claim["reasons"])
        tests = {(c["evaluation"], c["comparator_scorer"]) for c in claim["comparisons"]}
        self.assertEqual(tests, {("threshold", "probe"), ("time_to_event", "probe")})
        # Ablation rows are descriptive, all scored by the probe.
        ablation = [c for c in claim["descriptive"] if c["comparator"] == "minus_value"]
        self.assertTrue(ablation)
        self.assertTrue(all(c["scorer"] == "probe" for c in ablation))

    def test_a_next_token_probe_that_matches_rejects_the_claim(self):
        quality = WEAK
        dirs = arm(self.tmp, "clinical_soft", "full",
                   {("head", MAP_65): quality, ("head", MAP_63): quality}, noise_key="n")
        # Identical predictions: the probe matches the head on every metric.
        for s in (1, 2, 3):
            src = self.tmp / f"clinical_soft.full.s{s}.full"
            dst = write_run(self.tmp, "clinical_soft", "next_token_only", s,
                            quality={}, noise_key="n")
            import polars as pl
            head = pl.read_parquet(src / "threshold_eval/scores.parquet").filter(
                pl.col("scorer") == "head").with_columns(pl.lit("probe").alias("scorer"))
            head.write_parquet(dst / "threshold_eval/scores.parquet")
            dirs.append(dst)
        claim = report_of(dirs)["claims"]["claim_2"]
        self.assertEqual(claim["status"], cr.NOT_SUPPORTED)
        self.assertTrue(any("linear probe matches" in r for r in claim["reasons"]),
                        claim["reasons"])

    def test_a_rollout_row_that_is_not_evaluable_stays_in_the_table(self):
        report = report_of(self.runs(STRONG, WEAK))
        rollout = [c for c in report["cells"] if c["scorer"] == "rollout"]
        self.assertTrue(rollout)
        for cell in rollout:
            self.assertEqual(cell["status"], te.NOT_EVALUABLE)
            self.assertIn("imposed clock", cell["reason"])
        claim = report["claims"]["claim_2"]
        self.assertTrue(any(r["comparator_scorer"] == "rollout" and
                            r["status"] == te.NOT_EVALUABLE for r in claim["not_evaluable"]))

    def test_an_evaluable_rollout_that_matches_rejects_the_claim(self):
        dirs = self.runs(STRONG, WEAK, rollout=True)
        # Rollouts as strong as the head: copy the head's predictions into the rollout.
        import polars as pl
        for s in (1, 2, 3):
            head = pl.read_parquet(self.tmp / f"clinical_soft.full.s{s}.full"
                                   / "threshold_eval/scores.parquet").filter(
                pl.col("scorer") == "head")
            path = self.tmp / f"clinical_soft.next_token_only.s{s}.full/threshold_eval/scores.parquet"
            frame = pl.read_parquet(path).filter(pl.col("scorer") != "rollout")
            pl.concat([frame, head.with_columns(pl.lit("rollout").alias("scorer"))]
                      ).write_parquet(path)
        claim = report_of(dirs)["claims"]["claim_2"]
        self.assertEqual(claim["status"], cr.NOT_SUPPORTED)
        self.assertTrue(any("rollouts match" in r for r in claim["reasons"]), claim["reasons"])


class MultiplicityTest(unittest.TestCase):
    def test_benjamini_hochberg_matches_a_hand_computed_example(self):
        # Sorted: 0.001, 0.03, 0.04, 0.8 against k * 0.05 / 4 = 0.0125, 0.025, 0.0375, 0.05:
        # only k = 1 passes, so R = 1. Adjusted: 0.004, min(0.06, 0.0533) = 0.0533, 0.0533, 0.8.
        adjusted, rejected, r = cr.benjamini_hochberg([0.04, 0.001, 0.03, 0.8], alpha=0.05)
        np.testing.assert_allclose(adjusted, [0.04 * 4 / 3, 0.004, 0.04 * 4 / 3, 0.8])
        self.assertEqual(rejected.tolist(), [False, True, False, False])
        self.assertEqual(r, 1)

    def test_the_step_up_rejects_below_the_largest_passing_rank(self):
        # 0.02 > 1 * 0.05 / 3 fails on its own, but 0.024 <= 2 * 0.05 / 3 passes, so the
        # step-up rejects both (a step-down procedure would reject neither).
        adjusted, rejected, r = cr.benjamini_hochberg([0.02, 0.024, 0.9], alpha=0.05)
        np.testing.assert_allclose(adjusted, [0.036, 0.036, 0.9])
        self.assertEqual(rejected.tolist(), [True, True, False])
        self.assertEqual(r, 2)

    def test_adjusted_values_agree_with_scipy(self):
        ps = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
              0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
        adjusted, rejected, r = cr.benjamini_hochberg(ps, alpha=0.05)
        np.testing.assert_allclose(adjusted, stats.false_discovery_control(ps))
        self.assertEqual(r, 4)

    def test_fcr_adjusted_interval_level(self):
        self.assertAlmostEqual(cr.fcr_level(0.05, n_rejected=2, m=10), 0.99)
        self.assertAlmostEqual(cr.fcr_level(0.05, n_rejected=0, m=10), 0.995)

    def test_bootstrap_p_value(self):
        draws = np.array([0.1] * 95 + [-0.1] * 5)
        self.assertAlmostEqual(cr.bootstrap_p_value(draws), 2 * 6 / 101)
        self.assertEqual(cr.bootstrap_p_value(np.zeros(10)), 1.0)


class RunAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_a_screening_budget_run_offered_as_claim_evidence_is_refused(self):
        dirs = arm(self.tmp, "clinical_soft", "full", {})
        dirs.append(write_run(self.tmp, "clinical_soft", "full", 4, quality={},
                              budget="screening"))
        with self.assertRaisesRegex(cr.ClaimsError, r"screening-budget run .*KTD12"):
            cr.load_runs(dirs)

    def test_a_run_not_listed_as_claim_bearing_is_refused(self):
        run = write_run(self.tmp, "clinical_soft", "full", 1, quality={},
                        claim_bearing=False)
        with self.assertRaisesRegex(cr.ClaimsError, "claim-bearing"):
            cr.load_runs([run])

    def test_malformed_and_duplicate_run_specs_are_refused(self):
        run = write_run(self.tmp, "clinical_soft", "full", 1, quality={})
        twin = self.tmp / "twin"
        twin.mkdir()
        (twin / "run_spec.json").write_text((run / "run_spec.json").read_text())
        with self.assertRaisesRegex(cr.ClaimsError, "twice"):
            cr.load_runs([run, twin])
        spec = json.loads((run / "run_spec.json").read_text())
        for mutate, pattern in ((lambda s: s.pop("seed"), "seed"),
                                (lambda s: s.update(budget="medium"), "budget"),
                                (lambda s: s.update(claim_bearing="yes"), "claim_bearing"),
                                (lambda s: s.update(vocab_hash="short"), "vocab_hash"),
                                (lambda s: s.update(extra=1), "unknown")):
            broken = copy.deepcopy(spec)
            mutate(broken)
            with self.assertRaisesRegex(cr.ClaimsError, pattern):
                cr.validate_run_spec(broken)

    def test_a_vocabulary_that_does_not_match_the_run_spec_is_refused(self):
        run = write_run(self.tmp, "clinical_soft", "full", 1, quality={})
        summary = run / "threshold_eval/summary.json"
        blob = json.loads(summary.read_text())
        blob["vocabulary"] = "f" * 64
        summary.write_text(json.dumps(blob))
        with self.assertRaisesRegex(cr.ClaimsError, "vocab_hash"):
            cr.load_runs([run])

    def test_runs_scored_on_different_labels_are_refused(self):
        dirs = arm(self.tmp, "clinical_soft", "full", {})
        import polars as pl
        path = dirs[0] / "threshold_eval/scores.parquet"
        frame = pl.read_parquet(path)
        frame = frame.with_columns(
            pl.when(pl.col("pair_id") == "p00000").then(1 - pl.col("label"))
            .otherwise(pl.col("label")).cast(pl.Int8).alias("label"))
        frame.write_parquet(path)
        with self.assertRaisesRegex(cr.ClaimsError, "same evaluation set"):
            report_of(dirs)

    def test_a_control_on_edge_in_any_arm_is_refused_with_the_arm_named(self):
        ON_EDGE["odd_deciles"] = {MAP_65: False, MAP_63: True}
        VOCAB["odd_deciles"] = "d" * 64
        try:
            dirs = arm(self.tmp, "odd_deciles", "full", {})
            with self.assertRaisesRegex(cr.ClaimsError, r"control map 63.*'odd_deciles'"):
                report_of(dirs)
        finally:
            ON_EDGE.pop("odd_deciles")
            VOCAB.pop("odd_deciles")


class DisclosureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_no_identifier_small_cells_suppressed_and_written_aggregate_only(self):
        dirs = arm(self.tmp, "clinical_soft", "full",
                   {("head", MAP_65): STRONG, ("head", MAP_63): WEAK},
                   positives={MAP_63: 4}, status_extra={"censored": 3, "prevalent": 40})
        runs = cr.load_runs(dirs)
        report = cr.build_report(runs, config(), registry())
        text = json.dumps(report)
        for token in ("pair_id", "cluster_id", "p00001", "c00001"):
            self.assertNotIn(token, text)
        control = [c for c in report["cells"]
                   if c["threshold"] == MAP_63 and c["scorer"] != "rollout"]
        self.assertTrue(control)
        for cell in control:
            self.assertEqual(cell["status"], "small_cell_suppressed")
            self.assertNotIn("n", cell)
            self.assertNotIn("auroc", cell)
        status = report["label_status"][MAP_63]["48"]
        self.assertEqual(status["positive"], "<10")
        self.assertEqual(status["censored"], "<10")
        self.assertNotIn("negative", {k for k, v in status.items() if isinstance(v, int)
                                      and 0 < v < 10})
        policy = {"classes": {"aggregate_no_phi": {
            "directory": str(self.tmp / "final"), "formats": ["json"],
            "export_allowed": True}}}
        out = cr.write_report(report, self.tmp / "final/claims.json",
                              identifiers=cr.run_identifiers(runs), policy=policy)
        self.assertEqual(json.loads(out.read_text())["claims"].keys(), report["claims"].keys())
        leaked = copy.deepcopy(report)
        leaked["notes"].append("p00001")
        with self.assertRaises(Exception):
            cr.write_report(leaked, self.tmp / "final/leak.json",
                            identifiers=cr.run_identifiers(runs), policy=policy)
        with self.assertRaisesRegex(ValueError, "stored under"):
            cr.write_report(report, self.tmp / "elsewhere/claims.json",
                            identifiers=cr.run_identifiers(runs), policy=policy)

    def test_the_report_marks_each_claim_and_renders(self):
        dirs = arm(self.tmp, "clinical_soft", "full",
                   {("head", MAP_65): STRONG, ("head", MAP_63): WEAK})
        report = report_of(dirs)
        for claim in ("claim_1", "claim_2"):
            self.assertIn(report["claims"][claim]["status"],
                          (cr.SUPPORTED, cr.NOT_SUPPORTED, cr.INCOMPLETE))
        self.assertEqual(report["claims"]["claim_2"]["status"], cr.INCOMPLETE)
        text = cr.render_text(report)
        self.assertIn("claim_1", text)
        self.assertIn("resampled", json.dumps(report["config"]))


if __name__ == "__main__":
    unittest.main()
