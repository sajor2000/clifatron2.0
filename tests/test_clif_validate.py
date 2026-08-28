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
            "disclosure_status": "reviewed_approved", "release_id": f"rel-{site_id}",
            "outcomes": outcomes,
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
            load_site_results([str(path)], require_signatures=False)

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

        import src.eval.clif_validate as mod

        # Resolve from __file__ rather than a hard-coded relative path, so the guard
        # works regardless of the working directory (U5 review #29).
        source = Path(mod.__file__).read_text()
        tree = ast.parse(source)
        offending = []

        for node in ast.walk(tree):
            # (a) Importing randomness at all -- `from numpy.random import random as r`
            #     defeated a receiver-prefix check entirely.
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in ("random",) or alias.name.startswith("numpy.random"):
                        offending.append(f"line {node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                base = (node.module or "").split(".")[0]
                if base == "random" or (node.module or "").startswith("numpy.random"):
                    offending.append(f"line {node.lineno}: from {node.module} import ...")
            # (b) Attribute access, matched on the TRAILING name rather than the
            #     receiver, so `rng.random()` and `foo.bar.random()` are both caught.
            elif isinstance(node, ast.Attribute):
                if node.attr in ("random", "random_sample", "rand", "randn",
                                 "default_rng", "standard_normal", "uniform"):
                    offending.append(f"line {node.lineno}: .{node.attr}")

        self.assertEqual(offending, [],
                         f"a random prediction path is reachable: {offending}")

    def test_the_randomness_guard_actually_catches_an_alias(self):
        """The guard is itself a verification mechanism, so it needs its own proof.

        A guard matching receiver prefixes passed this snippet happily; that is exactly
        the false-pass this test exists to prevent."""
        import ast as _ast
        snippet = "from numpy.random import random as r\ndef f(n):\n    return r(n)\n"
        tree = _ast.parse(snippet)
        caught = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                if (node.module or "").startswith("numpy.random"):
                    caught.append(node.module)
        self.assertTrue(caught, "the alias form must be detectable")

    def test_evaluate_site_refuses_to_run_without_a_prediction_function(self):
        from src.eval.clif_validate import ArtifactMismatch, evaluate_site
        with self.assertRaises(ArtifactMismatch) as ctx:
            evaluate_site("ckpt", "data", "episodes.parquet", [{"name": "x"}],
                          predict_fn=None)
        self.assertIn("no default", str(ctx.exception))

    # ---------------------------------------------------------------- D2
    def test_checkpoint_without_head_weights_fails_closed(self):
        """D2: head_weights.pt was optional and loaded with strict=False."""
        from src.eval.clif_validate import ArtifactMismatch, load_checkpoint
        empty = self.out / "bundle"
        empty.mkdir()
        with self.assertRaises(ArtifactMismatch) as ctx:
            load_checkpoint(str(empty))
        self.assertIn("head_weights.pt", str(ctx.exception))

    def test_bundle_without_a_manifest_fails_closed(self):
        """D2: provenance must be establishable before any inference runs."""
        from src.eval.clif_validate import ArtifactMismatch, verify_bundle_compatibility
        empty = self.out / "nomanifest"
        empty.mkdir()
        with self.assertRaises(ArtifactMismatch):
            verify_bundle_compatibility(str(empty))

    def test_vocab_hash_mismatch_fails_closed(self):
        """D2: scoring one vocabulary's model against another's tokens."""
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
        """D2: mirrors value_stats.py rejecting schema-2 artifacts with vocab_hash: null."""
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
        """D2: an unsupported CLIF version is a failure, not a warning."""
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


class LogSanitizerIntegrationTest(unittest.TestCase):
    """U5 review #7, #21, #22.

    Only `redact()` was tested, as a standalone string function. Nothing proved the
    filter was ever attached, that it fired on a real LogRecord, that it covered
    tracebacks, or that it failed closed -- so all three defects could regress invisibly.
    """

    def _capture(self, logger_name):
        import logging

        from src.eval.log_sanitizer import install_log_sanitizer
        records = []

        class Recorder(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(Recorder())
        install_log_sanitizer(logger)
        return logger, records

    def test_filter_is_attached_and_redacts_a_real_record(self):
        logger, records = self._capture("u5-sanitizer-attached")
        logger.info("labeling from %s for %s", "/mnt/phi/rush/clif", "patient_id=12345")
        self.assertEqual(len(records), 1)
        msg = records[0].getMessage()
        self.assertNotIn("/mnt/phi/rush", msg)
        self.assertNotIn("12345", msg)
        self.assertIn("<redacted", msg)

    def test_ordinary_log_content_survives(self):
        logger, records = self._capture("u5-sanitizer-passthrough")
        logger.info("labeled 412 stays across 5 outcomes")
        self.assertEqual(records[0].getMessage(), "labeled 412 stays across 5 outcomes")

    def test_tracebacks_are_redacted(self):
        """#21: record.msg is not the only channel. This module raises with the
        checkpoint path embedded, so an operator-returned traceback carried what the
        redacted message no longer did."""
        logger, records = self._capture("u5-sanitizer-traceback")
        try:
            raise RuntimeError("bundle missing at /mnt/phi/rush/checkpoints/v0")
        except RuntimeError:
            logger.exception("bundle verification failed")
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].exc_info, "raw exc_info must not reach a sink")
        self.assertNotIn("/mnt/phi/rush", records[0].exc_text or "")
        self.assertIn("<redacted:path>", records[0].exc_text or "")

    def test_unformattable_record_fails_closed(self):
        """#7: a record whose formatting raises used to pass through untouched. A
        disclosure control that lets a record escape on its own error has the failure
        direction backwards."""
        logger, records = self._capture("u5-sanitizer-failclosed")
        logger.info("path %s and %s", "/mnt/phi/rush")  # too few args -> getMessage raises
        self.assertEqual(len(records), 1)
        msg = records[0].getMessage()
        self.assertNotIn("/mnt/phi/rush", msg)
        self.assertIn("redacted", msg)


class CheckpointStrictLoadTest(unittest.TestCase):
    """U5 review #24: only the file-absence branch of D2 was covered."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mismatched_head_weights_raise_artifact_mismatch(self):
        import torch

        from src.eval import clif_validate as CV

        bundle = self.out / "bundle"
        bundle.mkdir()
        torch.save({"not_a_real_head.weight": torch.zeros(2, 2)},
                   bundle / "head_weights.pt")

        class _Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.real = torch.nn.Linear(2, 2)

        original_backbone = CV.__dict__.get("load_backbone")
        import src.model.head_adapter as HA
        saved_load, saved_heads = HA.load_backbone, HA.CLIFATRONHeads
        HA.load_backbone = lambda p: _Stub()
        HA.CLIFATRONHeads = lambda backbone, n, freeze_backbone=True: _Stub()
        try:
            with self.assertRaises(CV.ArtifactMismatch) as ctx:
                CV.load_checkpoint(str(bundle))
            self.assertIn("do not match", str(ctx.exception))
        finally:
            HA.load_backbone, HA.CLIFATRONHeads = saved_load, saved_heads
            del original_backbone

    def test_allow_partial_escape_hatch_is_gone(self):
        """#31: an unenforced exemption inside a fail-closed control is worse than none."""
        import inspect

        from src.eval.clif_validate import load_checkpoint
        self.assertNotIn("allow_partial",
                         inspect.signature(load_checkpoint).parameters)


class ReleaseBoundaryTest(unittest.TestCase):
    """U5 review #16, #17: a draft is not a release, and the ledger must not record one
    that never reached disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _payload(self, status="reviewed_approved", release="rel-1"):
        from src.eval import schema as S
        from src.eval.clif_validate import build_export
        lv = {"outcome_definition_id": "o", "outcome_definition_version": "1.0.0",
              "status_counts": {s: 40 for s in S.U1_OUTCOME_STATES},
              "evaluable_denominator_fraction": 0.9}
        outcomes = {"o": {"status": S.EVALUABLE, "label_validity": lv,
                          "metrics": {"auroc": 0.8, "n": 400, "prevalence": 0.25}}}
        prov = {"model_bundle_id": "b1", "model_version": "v0", "vocab_hash": "aa",
                "outcome_spec_hash": "bb", "clif_version": "2.1"}
        return build_export(outcomes, prov, site_id="SITE-01", site_role="development",
                            partition_role="test", release_id=release,
                            disclosure_status=status)

    def test_a_draft_cannot_be_released(self):
        from src.eval.clif_validate import write_export
        from src.eval.schema import DisclosureError
        draft = self._payload(status="pending_review")
        with self.assertRaises(DisclosureError) as ctx:
            write_export(draft, self.out / "r.json", self.out / "ledger.jsonl")
        self.assertIn("pending_review", str(ctx.exception))
        self.assertFalse((self.out / "r.json").exists())
        self.assertFalse((self.out / "ledger.jsonl").exists(),
                         "a refused release must not appear in the ledger")

    def test_build_export_defaults_to_draft(self):
        """A caller who does not think about disclosure status gets the draft, not a
        release. The safe value is the one you get for free."""
        from src.eval import schema as S
        from src.eval.clif_validate import build_export
        lv = {"outcome_definition_id": "o", "outcome_definition_version": "1.0.0",
              "status_counts": {s: 40 for s in S.U1_OUTCOME_STATES},
              "evaluable_denominator_fraction": 0.9}
        prov = {"model_bundle_id": "b1", "model_version": "v0", "vocab_hash": "aa",
                "outcome_spec_hash": "bb", "clif_version": "2.1"}
        payload = build_export({"o": {"status": S.EVALUABLE, "label_validity": lv,
                                      "metrics": {"auroc": 0.8, "n": 400, "prevalence": 0.25}}},
                               prov, site_id="SITE-01", site_role="development",
                               partition_role="test", release_id="rel-x")
        self.assertEqual(payload["disclosure_status"], S.DRAFT_DISCLOSURE_STATUS)

    def test_ledger_records_only_after_the_artifact_is_on_disk(self):
        from src.eval.attestation import read_ledger
        from src.eval.clif_validate import write_export
        ledger = self.out / "ledger.jsonl"
        written = write_export(self._payload(), self.out / "r.json", ledger)
        self.assertTrue(written.exists())
        self.assertEqual(len(read_ledger(ledger)), 1)
        self.assertFalse(list(self.out.glob("*.partial")), "temp file must be renamed away")

    def test_a_failed_ledger_append_leaves_nothing_published(self):
        """Greptile PR #4: rename-then-append could publish a report the ledger never
        recorded, so a later differencing check would miss a real prior release."""
        from unittest import mock

        from src.eval.clif_validate import write_export
        out = self.out / "r.json"
        with mock.patch("src.eval.attestation.append_to_ledger",
                        side_effect=OSError("ledger volume full")):
            with self.assertRaises(OSError):
                write_export(self._payload(), out, self.out / "ledger.jsonl")
        self.assertFalse(out.exists(), "a report the ledger never recorded must not publish")
        self.assertFalse(list(self.out.glob("*.partial")), "temp file must be cleaned up")

    def test_ledger_entry_precedes_publication(self):
        """The artifact becomes visible only after its ledger record exists."""
        from unittest import mock

        from src.eval.attestation import append_to_ledger as real_append
        from src.eval.clif_validate import write_export
        out = self.out / "r.json"
        seen = {}

        def spy(payload, ledger_path):
            seen["published_at_append_time"] = out.exists()
            return real_append(payload, ledger_path)

        with mock.patch("src.eval.attestation.append_to_ledger", side_effect=spy):
            write_export(self._payload(), out, self.out / "ledger.jsonl")
        self.assertFalse(seen["published_at_append_time"],
                         "artifact was visible before the ledger recorded it")
        self.assertTrue(out.exists())

    def test_ledger_entry_is_durable_before_publication(self):
        """Greptile PR #4, round 2: a buffered write is not a record. A crash between the
        append and the rename would leave the report published with its ledger entry
        still in the page cache."""
        import os
        from unittest import mock

        from src.eval.clif_validate import write_export
        synced = []
        real_fsync = os.fsync

        def spy(fd):
            synced.append(fd)
            return real_fsync(fd)

        with mock.patch("os.fsync", side_effect=spy):
            write_export(self._payload(), self.out / "r.json", self.out / "ledger.jsonl")
        self.assertTrue(synced, "ledger append must reach stable storage before publishing")

    def test_replayed_release_id_is_rejected(self):
        """#13: a replayed report would be counted as another site."""
        from src.eval.clif_validate import write_export
        from src.eval.schema import DisclosureError
        ledger = self.out / "ledger.jsonl"
        write_export(self._payload(release="rel-1"), self.out / "a.json", ledger)
        with self.assertRaises(DisclosureError) as ctx:
            write_export(self._payload(release="rel-1"), self.out / "b.json", ledger)
        self.assertIn("already been recorded", str(ctx.exception))


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
                               release_id="rel-001",
                               disclosure_status="reviewed_approved",
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
                         partition_role="test", release_id="rel-002")


if __name__ == "__main__":
    unittest.main()
