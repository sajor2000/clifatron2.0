"""Allow-listed export schema and disclosure controls (src/eval/schema.py).

Covers U5 defects D3 (local paths in exported JSON), D5 (silent small-cell drops with
no complementary suppression), and the writer-side half of D9 (accept-anything export).

The three controls under test are not interchangeable and each has its own failure mode:
denominator suppression, numerator suppression (n * prevalence recovers the exact positive
count), and resolution bounding (a 50-point DCA curve inverts to per-patient TP/FP).
"""

import unittest

from src.eval import schema as S


def _label_validity(**overrides):
    block = {
        "outcome_definition_id": "map_below_65_48h",
        "outcome_definition_version": "1.0.0",
        "status_counts": {state: 10 for state in S.U1_OUTCOME_STATES},
        "evaluable_denominator_fraction": 0.82,
    }
    block.update(overrides)
    return block


def _envelope(**overrides):
    payload = {
        "schema_version": S.METRIC_SCHEMA_VERSION,
        "metric_version": "1.0.0",
        "model_bundle_id": "bundle-abc",
        "model_version": "v0",
        "vocab_hash": "a" * 16,
        "outcome_spec_hash": "b" * 16,
        "clif_version": "2.1",
        "site_id": "SITE-01",
        "site_role": "development",
        "partition_role": "test",
        "disclosure_status": "reviewed_approved",
        "release_id": "rel-001",
        "outcomes": {},
    }
    payload.update(overrides)
    return payload


def _evaluable_outcome(n=100, **overrides):
    block = {
        "status": S.EVALUABLE,
        "label_validity": _label_validity(),
        "metrics": {"auroc": 0.78, "auprc": 0.41, "n": n, "prevalence": 0.2},
    }
    block.update(overrides)
    return block


class PolicyConstantsTest(unittest.TestCase):
    def test_threshold_comes_from_the_landed_artifact_policy(self):
        """The number lives in configs/artifact_policy.yaml, not in this module.

        Two thresholds in one codebase is how a suppression rule silently stops
        applying -- which is what subgroup_panel's hard-coded 30 was.
        """
        self.assertEqual(S.min_cell_size(), 10)
        self.assertEqual(S.load_min_cell_size(), S.min_cell_size())

    def test_curve_release_minimum_is_materially_larger_than_the_cell_minimum(self):
        self.assertGreater(S.curve_release_min(), S.min_cell_size())

    def test_missing_policy_key_fails_closed_rather_than_defaulting(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "policy.yaml"
            bad.write_text("classes:\n  aggregate_no_phi:\n    directory: x\n")
            with self.assertRaises(S.DisclosureError):
                S.load_min_cell_size(bad)


class SuppressionTest(unittest.TestCase):
    def test_denominator_below_threshold_is_suppressed(self):
        status, _ = S.suppress_cell(n=S.min_cell_size() - 1, n_positive=3)
        self.assertEqual(status, S.INSUFFICIENT_N)

    def test_numerator_below_threshold_is_suppressed_even_when_n_clears(self):
        """The n=12, prevalence=0.0833 case: clears every size threshold while
        identifying exactly one outcome-positive patient."""
        status, reason = S.suppress_cell(n=12, n_positive=1)
        self.assertEqual(status, S.SMALL_CELL_SUPPRESSED)
        self.assertIn("recover", reason)

    def test_negative_count_below_threshold_is_also_suppressed(self):
        status, _ = S.suppress_cell(n=100, n_positive=98)
        self.assertEqual(status, S.SMALL_CELL_SUPPRESSED)

    def test_single_class_cell_is_reported_as_single_class(self):
        self.assertEqual(S.suppress_cell(n=100, n_positive=0)[0], S.SINGLE_CLASS)
        self.assertEqual(S.suppress_cell(n=100, n_positive=100)[0], S.SINGLE_CLASS)

    def test_healthy_cell_is_evaluable(self):
        status, reason = S.suppress_cell(n=100, n_positive=30)
        self.assertEqual(status, S.EVALUABLE)
        self.assertIsNone(reason)

    def test_recovering_the_count_from_a_released_cell_identifies_no_individual(self):
        """The property the disclosure argument actually rests on.

        This replaces `assertNotEqual(rounded * 12, 1.0)` -- a tautology that held for
        any rounding and established nothing, which is exactly how a false claim about
        2-decimal rounding shipped unchallenged.

        The honest claim is NOT that prevalence is unrecoverable. Assume an attacker
        recovers the positive count exactly from n and prevalence. `suppress_cell`
        guarantees that count is at least MIN_CELL_SIZE, and so is the negative count,
        so what they learn describes a group, never a patient. Asserted exhaustively
        across every cell shape the gate will release.
        """
        floor = S.min_cell_size()
        for n in range(1, 121):
            for pos in range(0, n + 1):
                if S.suppress_cell(n, pos)[0] != S.EVALUABLE:
                    continue
                self.assertGreaterEqual(pos, floor, f"released n={n} pos={pos}")
                self.assertGreaterEqual(n - pos, floor, f"released n={n} pos={pos}")

    def test_prevalence_quantisation_reduces_precision(self):
        """Defence in depth only -- deliberately NOT asserted as ambiguity, because it
        is not ambiguity. See prevalence_step's docstring."""
        step = S.prevalence_step()
        for raw in (1 / 3, 0.1234567, 0.98765):
            exported = S.round_prevalence(raw)
            self.assertAlmostEqual(exported / step, round(exported / step), places=6)

    def test_round_prevalence_returns_none_not_nan(self):
        """NaN serializes as bare NaN, which is not valid JSON and cannot be verified
        by a non-Python consumer."""
        self.assertIsNone(S.round_prevalence(float("nan")))
        self.assertIsNone(S.round_prevalence(None))


class ComplementarySuppressionTest(unittest.TestCase):
    def test_a_lone_suppressed_cell_takes_a_sibling_with_it(self):
        """One suppressed cell among released siblings is a subtraction away from
        the attribute total. At least two unknowns must remain."""
        cells = {
            "a": {"status": S.EVALUABLE, "n": 500},
            "b": {"status": S.EVALUABLE, "n": 40},
            "c": {"status": S.INSUFFICIENT_N, "n": 4},
        }
        out = S.apply_complementary_suppression(cells)
        suppressed = [k for k, v in out.items() if v["status"] in S.NON_EVALUABLE_STATUSES]
        self.assertEqual(len(suppressed), 2, "a single suppressed cell is recoverable")
        self.assertIn("c", suppressed)
        self.assertIn("b", suppressed, "the smallest releasable sibling is suppressed too")
        for key in suppressed:
            self.assertNotIn("n", out[key],
                             "a suppressed cell must not carry the count it is hiding")
            self.assertIn("n_band", out[key])

    def test_two_already_suppressed_cells_need_no_third(self):
        cells = {
            "a": {"status": S.EVALUABLE, "n": 500},
            "b": {"status": S.INSUFFICIENT_N, "n": 4},
            "c": {"status": S.INSUFFICIENT_N, "n": 3},
        }
        out = S.apply_complementary_suppression(cells)
        self.assertEqual(out["a"]["status"], S.EVALUABLE)

    def test_no_suppression_leaves_cells_untouched(self):
        cells = {"a": {"status": S.EVALUABLE, "n": 500}, "b": {"status": S.EVALUABLE, "n": 400}}
        self.assertEqual(S.apply_complementary_suppression(cells), cells)


class ExportAllowListTest(unittest.TestCase):
    def test_a_well_formed_export_validates(self):
        payload = _envelope(outcomes={"map_below_65_48h": _evaluable_outcome()})
        self.assertIs(S.validate_export(payload), payload)

    def test_unrecognized_envelope_field_is_rejected_at_write_time(self):
        """D9's mirror on the writer side: an accept-anything export is the same
        hole as an accept-anything loader."""
        payload = _envelope(debug_scratch="whatever")
        with self.assertRaises(S.DisclosureError) as ctx:
            S.validate_export(payload)
        self.assertIn("allow-list", str(ctx.exception))

    def test_local_filesystem_path_in_a_value_is_rejected(self):
        """D3: results["site"] = str(data_path) is how site directory layout travelled."""
        payload = _envelope(site_id="/mnt/phi/clif/rush")
        with self.assertRaises(S.DisclosureError) as ctx:
            S.validate_export(payload)
        self.assertIn("path", str(ctx.exception))

    def test_identifier_shaped_field_names_are_rejected(self):
        for bad in ("patient_id", "hospitalization_id", "hosp_id", "sequence", "token", "pos_min"):
            with self.subTest(field=bad):
                payload = _envelope()
                payload["outcomes"] = {"o": {
                    "status": S.EVALUABLE,
                    "label_validity": _label_validity(),
                    "metrics": {"n": 100, bad: 1},
                }}
                with self.assertRaises(S.DisclosureError):
                    S.validate_export(payload)

    def test_missing_required_envelope_field_is_rejected(self):
        payload = _envelope()
        del payload["vocab_hash"]
        with self.assertRaises(S.DisclosureError):
            S.validate_export(payload)

    def test_unknown_site_or_partition_role_is_rejected(self):
        with self.assertRaises(S.DisclosureError):
            S.validate_export(_envelope(site_role="whatever"))
        with self.assertRaises(S.DisclosureError):
            S.validate_export(_envelope(partition_role="whatever"))


class OutcomeBlockTest(unittest.TestCase):
    def test_non_evaluable_outcome_must_not_carry_metrics(self):
        """A non-evaluable outcome carrying numbers is how a fabricated score gets read
        as a real one."""
        payload = _envelope(outcomes={"o": {
            "status": S.SINGLE_CLASS,
            "reason": "no positives at this site",
            "label_validity": _label_validity(),
            "metrics": {"auroc": 0.5, "n": 100},
        }})
        with self.assertRaises(S.DisclosureError) as ctx:
            S.validate_export(payload)
        self.assertIn("must not carry numbers", str(ctx.exception))

    def test_unknown_status_is_rejected(self):
        payload = _envelope(outcomes={"o": {
            "status": "looks_fine", "label_validity": _label_validity(),
        }})
        with self.assertRaises(S.DisclosureError):
            S.validate_export(payload)

    def test_non_evaluable_constructor_round_trips(self):
        block = S.non_evaluable(S.UNSUPPORTED_AT_SITE, "table absent", _label_validity())
        payload = _envelope(outcomes={"o": block})
        S.validate_export(payload)

    def test_non_evaluable_constructor_rejects_the_evaluable_status(self):
        with self.assertRaises(S.DisclosureError):
            S.non_evaluable(S.EVALUABLE, "n/a", _label_validity())


class LabelValidityTest(unittest.TestCase):
    def test_report_without_label_validity_is_rejected(self):
        """Without this block a site with mis-mapped units returns a plausible AUROC
        that nothing in the payload can contradict."""
        payload = _envelope(outcomes={"o": {"status": S.EVALUABLE,
                                            "metrics": {"n": 100, "auroc": 0.7}}})
        with self.assertRaises(S.DisclosureError) as ctx:
            S.validate_export(payload)
        self.assertIn("label_validity", str(ctx.exception))

    def test_status_counts_must_account_for_all_seven_u1_states(self):
        partial = _label_validity(status_counts={"positive": 10, "negative": 90})
        payload = _envelope(outcomes={"o": _evaluable_outcome(label_validity=partial)})
        with self.assertRaises(S.DisclosureError) as ctx:
            S.validate_export(payload)
        self.assertIn("seven", str(ctx.exception))

    def test_unknown_outcome_state_is_rejected(self):
        counts = {state: 1 for state in S.U1_OUTCOME_STATES}
        counts["invented_state"] = 1
        payload = _envelope(outcomes={"o": _evaluable_outcome(
            label_validity=_label_validity(status_counts=counts))})
        with self.assertRaises(S.DisclosureError):
            S.validate_export(payload)


class CurveResolutionTest(unittest.TestCase):
    def test_small_cell_may_not_release_its_dca_curve(self):
        """NB(pt) = TP/N - (FP/N)(pt/(1-pt)) over 50 thresholds, with n and prevalence
        also released, is 50 equations that invert to per-patient TP/FP counts."""
        n = S.curve_release_min() - 1
        payload = _envelope(outcomes={"o": _evaluable_outcome(
            n=n, curves={"dca_thresholds": [0.1], "dca_model": [0.02]})})
        with self.assertRaises(S.DisclosureError) as ctx:
            S.validate_export(payload)
        self.assertIn("may not release curves", str(ctx.exception))

    def test_large_cell_may_release_its_curves(self):
        payload = _envelope(outcomes={"o": _evaluable_outcome(
            n=S.curve_release_min(), curves={"dca_thresholds": [0.1], "dca_model": [0.02]})})
        S.validate_export(payload)

    def test_curves_releasable_tracks_the_threshold(self):
        self.assertFalse(S.curves_releasable(S.curve_release_min() - 1))
        self.assertTrue(S.curves_releasable(S.curve_release_min()))


if __name__ == "__main__":
    unittest.main()
