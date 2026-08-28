"""Synthetic smoke tests for the site-local validation surfaces.

Tests the auto-labeler on synthetic CLIF data, the forest-plot generator,
and validates that aggregate-only output contains no raw patient rows.
"""

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.data.cohort import build_cohort
from src.data.splits import content_manifest
from src.eval.clif_auto_labeler import auto_label


class ClifValidateSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_auto_labeler_preserves_unsupported_state(self):
        """A missing synthetic outcome table is unsupported, never negative."""
        base = self.out / "synthetic_clif"
        base.mkdir()
        start = datetime(2026, 1, 1, tzinfo=UTC)
        pl.DataFrame(
            {
                "hospitalization_id": ["synthetic-stay"],
                "patient_id": ["synthetic-patient"],
                "hospitalization_joined_id": ["synthetic-chain"],
                "admission_dttm": [start],
                "discharge_dttm": [datetime(2026, 1, 5, tzinfo=UTC)],
                "age_at_admission": [50],
                "discharge_category": ["Home"],
                "hospital_id": ["synthetic-site"],
            }
        ).write_parquet(base / "clif_hospitalization.parquet")
        pl.DataFrame(
            {
                "hospitalization_id": ["synthetic-stay"],
                "in_dttm": [start],
                "out_dttm": [datetime(2026, 1, 4, tzinfo=UTC)],
                "location_category": ["icu"],
            }
        ).write_parquet(base / "clif_adt.parquet")

        hospitalization = pl.read_parquet(base / "clif_hospitalization.parquet")
        adt = pl.read_parquet(base / "clif_adt.parquet")
        config = {
            "anchor_hours": 24,
            "prediction_horizon_hours": 48,
            "minimum_age": 18,
            "icu_location_category": "icu",
        }
        episodes = build_cohort(hospitalization, adt, config).with_columns(
            pl.lit("train").alias("partition"),
        )
        split_hash = content_manifest(
            episodes, columns=["hospitalization_id", "patient_id", "partition"]
        )["sha256"]
        episode_hash = content_manifest(
            episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
        )["sha256"]
        episodes = episodes.with_columns(
            pl.lit("1.0.0").alias("cohort_contract_version"),
            pl.lit(split_hash).alias("split_sha256"),
            pl.lit(episode_hash).alias("episode_sha256"),
            pl.lit("{}").alias("source_provenance_json"),
        )
        episode_path = base / "episodes.parquet"
        episodes.write_parquet(episode_path)
        (base / "clif_hospitalization.parquet").unlink()
        (base / "clif_adt.parquet").unlink()

        labels = auto_label(str(base), episode_path, ["map_below_65_48h"])

        self.assertIn("hospitalization_id", labels.columns)
        self.assertIn("map_below_65_48h_status", labels.columns)
        self.assertEqual(labels["partition"].item(), "train")
        self.assertEqual(labels["map_below_65_48h_status"].item(), "unsupported_at_site")
        self.assertIsNone(labels["map_below_65_48h"].item())

        lbl_path = self.out / "test_labels.parquet"
        labels.write_parquet(lbl_path)

        reloaded = pl.read_parquet(lbl_path)
        self.assertEqual(len(reloaded), len(labels))


    def _site_report(self, site_id, auroc_map, n=1000):
        """A schema-valid site artifact. Built through the real contract, not by hand."""
        from src.eval import schema as S
        outcomes = {}
        for name, auroc in auroc_map.items():
            outcomes[name] = {
                "status": S.EVALUABLE,
                "label_validity": {
                    "outcome_definition_id": name,
                    "outcome_definition_version": "1.0.0",
                    "status_counts": {s: 10 for s in S.U1_OUTCOME_STATES},
                    "evaluable_denominator_fraction": 0.9,
                },
                "metrics": {"auroc": auroc, "auprc": auroc - 0.5, "ece": 0.03,
                            "n": n, "prevalence": 0.2},
            }
        return {
            "schema_version": S.METRIC_SCHEMA_VERSION,
            "metric_version": S.METRIC_SCHEMA_VERSION,
            "model_bundle_id": "bundle-1", "model_version": "v0",
            "vocab_hash": "a" * 16, "outcome_spec_hash": "b" * 16,
            "clif_version": "2.1", "site_id": site_id,
            "site_role": "external_confirmation", "partition_role": "test",
            "disclosure_status": "reviewed", "outcomes": outcomes,
        }

    def test_02_forest_plot_generates_on_synthetic_sites(self):
        """Forest-plot data is valid JSON with expected structure."""
        from src.eval.clif_forest_plot import forest_plot_data

        results = [
            self._site_report("SITE-01", {"in_hospital_mortality": 0.82, "new_imv_24h": 0.79}),
            self._site_report("SITE-02", {"in_hospital_mortality": 0.78, "new_imv_24h": 0.76}),
            self._site_report("SITE-03", {"in_hospital_mortality": 0.85, "new_imv_24h": 0.81}),
        ]

        forest = forest_plot_data(results)
        self.assertGreater(len(forest), 0)

        for row in forest:
            self.assertIn("outcome", row)
            self.assertIn("value", row)
            self.assertIn("ci_lower", row)
            self.assertIn("ci_upper", row)
            self.assertIsNotNone(row["value"])

        plot_path = self.out / "forest_plot.json"
        plot_path.write_text(json.dumps({"forest": forest}))
        self.assertIn("forest", json.loads(plot_path.read_text()))

    def test_02b_forest_loader_rejects_an_unrecognized_field(self):
        """U5 D9: the loader was a bare json.loads that promoted every unknown key to
        an outcome. It is now the reader-side twin of the writer-side allow-list."""
        from src.eval.clif_forest_plot import load_site_results
        from src.eval.schema import DisclosureError

        bad = self._site_report("SITE-01", {"in_hospital_mortality": 0.8})
        bad["scratch_debug_field"] = {"leaked": 1}
        path = self.out / "bad_site.json"
        path.write_text(json.dumps(bad))

        with self.assertRaises(DisclosureError):
            load_site_results([str(path)])

    def test_02c_forest_loader_rejects_an_unsigned_report_when_keys_are_registered(self):
        from src.eval.clif_forest_plot import load_site_results
        from src.eval.attestation import AuthenticationError

        report = self._site_report("SITE-01", {"in_hospital_mortality": 0.8})
        path = self.out / "unsigned_site.json"
        path.write_text(json.dumps(report))

        with self.assertRaises(AuthenticationError):
            load_site_results([str(path)], signing_keys={"SITE-01": b"secret"})

    def test_02d_forest_loader_accepts_a_correctly_signed_report(self):
        from src.eval.attestation import sign_report
        from src.eval.clif_forest_plot import load_site_results

        report = sign_report(self._site_report("SITE-01", {"in_hospital_mortality": 0.8}),
                             b"secret")
        path = self.out / "signed_site.json"
        path.write_text(json.dumps(report))
        loaded = load_site_results([str(path)], signing_keys={"SITE-01": b"secret"})
        self.assertEqual(len(loaded), 1)

    def test_02e_non_evaluable_outcome_is_not_averaged_away(self):
        """A suppressed cell must not read as a missing value that quietly vanishes
        from the mean."""
        from src.eval import schema as S
        from src.eval.clif_forest_plot import build_forest_table

        good = self._site_report("SITE-01", {"aki_kdigo_48h": 0.80})
        blocked = self._site_report("SITE-02", {"aki_kdigo_48h": 0.80})
        blocked["outcomes"]["aki_kdigo_48h"] = S.non_evaluable(
            S.UNSUPPORTED_AT_SITE, "creatinine table absent",
            blocked["outcomes"]["aki_kdigo_48h"]["label_validity"])

        table = build_forest_table([good, blocked])
        row = next(r for r in table if r["outcome"] == "aki_kdigo_48h")
        self.assertEqual(row["auroc"]["n_sites"], 1)
        self.assertIn(S.UNSUPPORTED_AT_SITE, row["non_evaluable_statuses"])

    def test_03_aggregate_only_no_raw_patient_rows(self):
        """Exported artifacts carry no patient-level fields -- enforced by the schema,
        not by a string search over a hand-built dict."""
        from src.eval import schema as S

        report = self._site_report("SITE-01", {"in_hospital_mortality": 0.82})
        S.validate_export(report)

        results_json = json.dumps(report)
        for term in ["patient_id", "hosp_id", "sequence", "token", "pos_min"]:
            self.assertNotIn(term, results_json.lower())

        # And the schema actively refuses one rather than merely not containing it.
        leaky = self._site_report("SITE-01", {"in_hospital_mortality": 0.82})
        leaky["outcomes"]["in_hospital_mortality"]["metrics"]["patient_id"] = 7
        with self.assertRaises(S.DisclosureError):
            S.validate_export(leaky)



class ValidatorFailsClosedTest(unittest.TestCase):
    """U5 D1, D2, D10: the validator must not be able to emit a report from noise,
    from partial weights, or from a call that never reached inference."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # ---------------------------------------------------------------- D1
    def test_no_random_prediction_path_exists_in_the_module(self):
        """The two np.random.random call sites that supplied predictions are gone.

        Asserted against the module's AST rather than its behaviour, because the defect
        was that a *fallback* existed at all: any test that supplies predictions would
        pass whether or not the fallback was still reachable. Parsing rather than
        grepping so that prose describing the old defect does not trip the check."""
        import ast

        tree = ast.parse(Path("src/eval/clif_validate.py").read_text())
        offending = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            dotted = ".".join(reversed(parts))
            if dotted.startswith(("np.random", "numpy.random", "random.")):
                offending.append(f"line {node.lineno}: {dotted}")
        self.assertEqual(offending, [],
                         f"a random prediction path is reachable: {offending}")

    def test_evaluate_site_refuses_to_run_without_a_prediction_function(self):
        from src.eval.clif_validate import ArtifactMismatch, evaluate_site
        with self.assertRaises(ArtifactMismatch) as ctx:
            evaluate_site("ckpt", "data", "episodes.parquet", [{"name": "x"}],
                          predict_fn=None)
        self.assertIn("no default", str(ctx.exception))

    # ---------------------------------------------------------------- D2
    def test_checkpoint_without_head_weights_fails_closed(self):
        from src.eval.clif_validate import ArtifactMismatch, load_checkpoint
        empty = self.out / "bundle"
        empty.mkdir()
        with self.assertRaises(ArtifactMismatch) as ctx:
            load_checkpoint(str(empty))
        self.assertIn("head_weights.pt", str(ctx.exception))

    def test_bundle_without_a_manifest_fails_closed(self):
        from src.eval.clif_validate import ArtifactMismatch, verify_bundle_compatibility
        empty = self.out / "nomanifest"
        empty.mkdir()
        with self.assertRaises(ArtifactMismatch):
            verify_bundle_compatibility(str(empty))

    def test_vocab_hash_mismatch_fails_closed(self):
        from src.eval.clif_validate import ArtifactMismatch, verify_bundle_compatibility
        bundle = self.out / "mismatch"
        bundle.mkdir()
        (bundle / "bundle_manifest.json").write_text(json.dumps({
            "model_bundle_id": "b1", "model_version": "v0",
            "vocab_hash": "aaaa", "outcome_spec_hash": "bbbb", "clif_version": "2.1",
        }))
        with self.assertRaises(ArtifactMismatch) as ctx:
            verify_bundle_compatibility(str(bundle), vocab_hash="cccc")
        self.assertIn("vocab_hash mismatch", str(ctx.exception))

    def test_null_vocab_hash_fails_closed(self):
        from src.eval.clif_validate import ArtifactMismatch, verify_bundle_compatibility
        bundle = self.out / "nullhash"
        bundle.mkdir()
        (bundle / "bundle_manifest.json").write_text(json.dumps({
            "model_bundle_id": "b1", "model_version": "v0",
            "vocab_hash": None, "outcome_spec_hash": "bbbb", "clif_version": "2.1",
        }))
        with self.assertRaises(ArtifactMismatch):
            verify_bundle_compatibility(str(bundle))

    def test_unsupported_clif_version_fails_closed(self):
        from src.eval.clif_validate import ArtifactMismatch, verify_bundle_compatibility
        bundle = self.out / "oldclif"
        bundle.mkdir()
        (bundle / "bundle_manifest.json").write_text(json.dumps({
            "model_bundle_id": "b1", "model_version": "v0",
            "vocab_hash": "aaaa", "outcome_spec_hash": "bbbb", "clif_version": "1.0",
        }))
        with self.assertRaises(ArtifactMismatch) as ctx:
            verify_bundle_compatibility(str(bundle))
        self.assertIn("not supported", str(ctx.exception))

    def test_good_manifest_returns_the_provenance_block(self):
        from src.eval.clif_validate import verify_bundle_compatibility
        bundle = self.out / "good"
        bundle.mkdir()
        (bundle / "bundle_manifest.json").write_text(json.dumps({
            "model_bundle_id": "b1", "model_version": "v0",
            "vocab_hash": "aaaa", "outcome_spec_hash": "bbbb", "clif_version": "2.1",
        }))
        prov = verify_bundle_compatibility(str(bundle), vocab_hash="aaaa")
        self.assertEqual(prov["model_bundle_id"], "b1")
        self.assertEqual(prov["clif_version"], "2.1")

    # ---------------------------------------------------------------- log sanitization
    def test_log_redaction_strips_paths_and_identifiers(self):
        from src.eval.schema import redact
        self.assertIn("<redacted:path>", redact("Labeling outcomes from /mnt/phi/rush ..."))
        self.assertIn("<redacted>", redact("row patient_id=12345 scored"))
        self.assertEqual(redact("labeled 412 stays"), "labeled 412 stays")


class ValidatorEndToEndTest(unittest.TestCase):
    """U5 D10: evaluate_site could not reach inference at all.

    auto_label's landed signature is (data_dir, episode_artifact, outcomes=None), but
    the validator called auto_label(data_path, outcome_names) -- passing the outcome
    list into the episode_artifact slot. Nothing exercised this path, so the suite
    never caught it. This test is that missing coverage.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _fixture(self, n=40):
        base = self.out / "clif"
        base.mkdir()
        start = datetime(2026, 1, 1, tzinfo=UTC)
        ids = [f"stay-{i}" for i in range(n)]
        pl.DataFrame({
            "hospitalization_id": ids,
            "patient_id": [f"pt-{i}" for i in range(n)],
            "hospitalization_joined_id": [f"chain-{i}" for i in range(n)],
            "admission_dttm": [start] * n,
            "discharge_dttm": [datetime(2026, 1, 5, tzinfo=UTC)] * n,
            "age_at_admission": [50] * n,
            "discharge_category": ["Home"] * n,
            "hospital_id": ["synthetic-site"] * n,
        }).write_parquet(base / "clif_hospitalization.parquet")
        pl.DataFrame({
            "hospitalization_id": ids,
            "in_dttm": [start] * n,
            "out_dttm": [datetime(2026, 1, 4, tzinfo=UTC)] * n,
            "location_category": ["icu"] * n,
        }).write_parquet(base / "clif_adt.parquet")

        hosp = pl.read_parquet(base / "clif_hospitalization.parquet")
        adt = pl.read_parquet(base / "clif_adt.parquet")
        episodes = build_cohort(hosp, adt, {
            "anchor_hours": 24, "prediction_horizon_hours": 48,
            "minimum_age": 18, "icu_location_category": "icu",
        }).with_columns(pl.lit("test").alias("partition"))
        split_hash = content_manifest(
            episodes, columns=["hospitalization_id", "patient_id", "partition"])["sha256"]
        episode_hash = content_manifest(
            episodes,
            columns=["hospitalization_id", "patient_id", "eligible", "partition"])["sha256"]
        episodes = episodes.with_columns(
            pl.lit("1.0.0").alias("cohort_contract_version"),
            pl.lit(split_hash).alias("split_sha256"),
            pl.lit(episode_hash).alias("episode_sha256"),
            pl.lit("{}").alias("source_provenance_json"),
        )
        ep_path = base / "episodes.parquet"
        episodes.write_parquet(ep_path)
        (base / "clif_hospitalization.parquet").unlink()
        (base / "clif_adt.parquet").unlink()
        return base, ep_path

    def test_evaluate_site_runs_end_to_end_with_an_explicit_episode_artifact(self):
        import numpy as np

        from src.eval import schema as S
        from src.eval.clif_validate import evaluate_site

        base, ep_path = self._fixture()
        cfgs = [{"name": "map_below_65_48h", "direction": "below"}]

        def predict(labels_df):
            # Deterministic stub. The seam exists so real inference plugs in here;
            # what matters for this test is that the call reaches it at all.
            return np.linspace(0.1, 0.9, len(labels_df)).reshape(-1, 1)

        result = evaluate_site(str(base), str(base), str(ep_path), cfgs,
                               predict_fn=predict)

        block = result["outcomes"]["map_below_65_48h"]
        # The synthetic fixture has no vitals table, so the outcome is unsupported --
        # which is the correct answer, and is reported as a status rather than a score.
        self.assertIn(block["status"], S.NON_EVALUABLE_STATUSES)
        self.assertNotIn("metrics", block)
        self.assertIn("label_validity", block)
        self.assertEqual(
            set(block["label_validity"]["status_counts"]), set(S.U1_OUTCOME_STATES))

    def test_export_is_schema_valid_and_ledgered(self):
        import numpy as np

        from src.eval.attestation import read_ledger
        from src.eval.clif_validate import build_export, evaluate_site, write_export

        base, ep_path = self._fixture()
        cfgs = [{"name": "map_below_65_48h", "direction": "below"}]
        result = evaluate_site(
            str(base), str(base), str(ep_path), cfgs,
            predict_fn=lambda df: np.linspace(0.1, 0.9, len(df)).reshape(-1, 1))

        provenance = {"model_bundle_id": "b1", "model_version": "v0",
                      "vocab_hash": "aaaa", "outcome_spec_hash": "bbbb",
                      "clif_version": "2.1"}
        payload = build_export(result["outcomes"], provenance, site_id="SITE-01",
                               site_role="development", partition_role="test",
                               signing_key=b"k")
        ledger = self.out / "ledger.jsonl"
        written = write_export(payload, self.out / "report.json", ledger)

        self.assertTrue(written.exists())
        self.assertTrue(read_ledger(ledger))
        blob = json.loads(written.read_text())
        self.assertNotIn("site", blob, "the local data path must not be exported")
        self.assertEqual(blob["site_id"], "SITE-01")

    def test_export_carrying_a_local_path_is_refused(self):
        from src.eval.clif_validate import build_export
        from src.eval.schema import DisclosureError
        with self.assertRaises(DisclosureError):
            build_export({}, {"model_bundle_id": "b1", "model_version": "v0",
                              "vocab_hash": "a", "outcome_spec_hash": "b",
                              "clif_version": "2.1"},
                         site_id="/mnt/phi/rush", site_role="development",
                         partition_role="test")


if __name__ == "__main__":
    unittest.main()
