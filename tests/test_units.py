"""U4 (R6, R7; KTD5): dose-unit parsing and conversion table (`src/data/units.py`)."""

import math
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
CSV = ROOT / "external/clifatron/tokenETL/config/critical_illness_tokenization_final_with_intervals.csv"


class NormalizeUnitTest(unittest.TestCase):
    def test_data_spellings_normalize_to_one_form(self):
        from src.data.units import normalize_unit

        cases = {
            "mcg/kg/min": "mcg/kg/min",
            "mcg/hour": "mcg/hr",
            "units/hour": "u/hr",
            "mg/hour": "mg/hr",
            "mL/hour": "ml/hr",
            "mcg/kg/hour": "mcg/kg/hr",
            "ng/kg/min": "ng/kg/min",
            "grams/hour": "g/hr",
            "units/min": "u/min",
            "mEq/min": "meq/min",
            "µL/kg/min": "ul/kg/min",
            " MG ": "mg",
            "grams": "g",
            "units": "u",
            "dose": "dose",
            None: "unknown",
            "": "unknown",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_unit(raw), expected, raw)

    def test_concept_suffix_matches_csv_measurement_spelling(self):
        from src.data.units import unit_suffix

        self.assertEqual(unit_suffix("mcg/kg/hour"), "mcg_kg_hr")
        self.assertEqual(unit_suffix("units/min"), "u_min")
        self.assertEqual(unit_suffix("/hour/min"), "hr_min")
        self.assertEqual(unit_suffix(None), "unknown")


class ConvertDoseTest(unittest.TestCase):
    def test_per_kg_target_divides_by_the_weight(self):
        from src.data.units import convert_dose

        value, status = convert_dose(100.0, "mcg/hour", "mcg/kg/hr", weight_kg=80.0)
        self.assertEqual(status, "converted")
        self.assertAlmostEqual(value, 1.25)

    def test_per_kg_target_without_a_weight_is_no_weight_and_keeps_native(self):
        from src.data.units import convert_dose

        for weight in (None, float("nan"), 0.0, -3.0):
            value, status = convert_dose(100.0, "mcg/hour", "mcg/kg/hr", weight_kg=weight)
            self.assertEqual((value, status), (100.0, "no_weight"), weight)

    def test_per_kg_source_to_flat_target_multiplies_by_the_weight(self):
        from src.data.units import convert_dose

        value, status = convert_dose(0.1, "mcg/kg/min", "mcg/min", weight_kg=70.0)
        self.assertEqual(status, "converted")
        self.assertAlmostEqual(value, 7.0)

    def test_units_per_hour_to_units_per_minute(self):
        from src.data.units import convert_dose

        value, status = convert_dose(2.4, "units/hour", "u/min")
        self.assertEqual(status, "converted")
        self.assertAlmostEqual(value, 0.04)

    def test_mass_and_time_scale_together(self):
        from src.data.units import convert_dose

        cases = [
            (1.0, "mg/hour", "mcg/kg/min", 50.0, 1000.0 / 60.0 / 50.0),
            (2.0, "grams/hour", "mg/hr", None, 2000.0),
            (5.0, "ng/kg/min", "mcg/kg/min", None, 0.005),
            (1.0, "mg/kg/hour", "mcg/kg/min", None, 1000.0 / 60.0),
            (3.0, "grams/min", "mg/min", None, 3000.0),
        ]
        for value, src, dst, weight, expected in cases:
            got, status = convert_dose(value, src, dst, weight_kg=weight)
            self.assertEqual(status, "converted", (src, dst))
            self.assertTrue(math.isclose(got, expected, rel_tol=1e-12), (src, dst, got))

    def test_same_unit_is_passthrough_and_no_target_is_passthrough(self):
        from src.data.units import convert_dose

        self.assertEqual(convert_dose(0.05, "mcg/kg/min", "mcg/kg/min"), (0.05, "passthrough"))
        self.assertEqual(convert_dose(10.0, "units/hour", None), (10.0, "passthrough"))

    def test_incompatible_or_unknown_units_are_unconvertible(self):
        from src.data.units import convert_dose

        for src, dst in [
            ("mL/hour", "mcg/kg/min"),   # volume -> mass
            ("units/hour", "mg/hr"),     # units -> mass
            ("mg", "mg/hr"),             # amount -> rate
            ("dose", "mg"),
            ("/hour/min", "mg/hr"),
            (None, "mg/hr"),
        ]:
            value, status = convert_dose(4.0, src, dst, weight_kg=70.0)
            self.assertEqual((value, status), (4.0, "unconvertible"), (src, dst))

    def test_a_zero_dose_stays_zero(self):
        from src.data.units import convert_dose

        self.assertEqual(convert_dose(0.0, "mcg/hour", "mcg/kg/hr", weight_kg=80.0),
                         (0.0, "converted"))


class CanonicalIntermittentUnitTest(unittest.TestCase):
    def test_mass_goes_to_mg_and_units_to_u(self):
        from src.data.units import canonical_unit

        self.assertEqual(canonical_unit("grams"), "mg")
        self.assertEqual(canonical_unit("mcg"), "mg")
        self.assertEqual(canonical_unit("ng"), "mg")
        self.assertEqual(canonical_unit("mg"), "mg")
        self.assertEqual(canonical_unit("units"), "u")
        self.assertEqual(canonical_unit("mEq"), "meq")
        self.assertEqual(canonical_unit("mg/kg"), "mg/kg")

    def test_dose_and_volume_have_no_canonical_unit(self):
        from src.data.units import canonical_unit

        for raw in ("dose", "mL", "L", None, "tab"):
            self.assertIsNone(canonical_unit(raw), raw)

    def test_two_grams_and_two_thousand_mg_agree(self):
        from src.data.units import canonical_unit, convert_dose

        a = convert_dose(2.0, "grams", canonical_unit("grams"))
        b = convert_dose(2000.0, "mg", canonical_unit("mg"))
        self.assertEqual(a, (2000.0, "converted"))
        self.assertEqual(b, (2000.0, "passthrough"))


class DosePlanTest(unittest.TestCase):
    def test_continuous_plan_names_the_target_and_fallback_concepts(self):
        from src.data.units import dose_plan

        plan = dose_plan("fentanyl", "mcg/hour", "mcg/kg/hr")
        self.assertEqual(plan["target_concept"], "fentanyl_mcg_kg_hr")
        self.assertEqual(plan["native_concept"], "fentanyl_mcg_hr")
        self.assertEqual(plan["weight_power"], -1)
        self.assertEqual(plan["status"], "converted")
        no_target = dose_plan("heparin", "units/hour", None)
        self.assertEqual(no_target["target_concept"], "heparin_u_hr")
        self.assertEqual(no_target["status"], "passthrough")
        volume = dose_plan("propofol", "mL/hour", "mcg/kg/min")
        self.assertEqual(volume["target_concept"], "propofol_ml_hr")
        self.assertEqual(volume["status"], "unconvertible")

    def test_med_category_is_normalized_in_concept_names(self):
        from src.data.units import dose_plan

        self.assertEqual(dose_plan("Sodium Bicarbonate", "mEq/hour", None)["target_concept"],
                         "sodium_bicarbonate_meq_hr")


class CsvPreferredUnitsTest(unittest.TestCase):
    def test_measurement_names_split_into_med_and_unit(self):
        from src.data.units import split_med_measurement

        self.assertEqual(split_med_measurement("norepinephrine_mcg_kg_min"),
                         ("norepinephrine", "mcg/kg/min"))
        self.assertEqual(split_med_measurement("vasopressin_u_min"), ("vasopressin", "u/min"))
        self.assertEqual(split_med_measurement("diltiazem_mg_hr"), ("diltiazem", "mg/hr"))
        self.assertIsNone(split_med_measurement("lactate"))

    def test_physician_csv_preferred_units_cover_every_medication_row(self):
        from src.data.units import preferred_units_from_csv

        units = preferred_units_from_csv(CSV)
        self.assertEqual(len(units), 28)
        self.assertEqual(units["fentanyl"], "mcg/kg/hr")
        self.assertEqual(units["vasopressin"], "u/min")
        self.assertEqual(units["norepinephrine"], "mcg/kg/min")
        self.assertEqual(units["angiotensin"], "ng/kg/min")    # CSV typo aliased (KTD2.7)
        self.assertNotIn("angiotension", units)


if __name__ == "__main__":
    unittest.main()
