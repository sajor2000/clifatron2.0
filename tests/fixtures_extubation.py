"""Synthetic CLIF-shaped extubation fixture, shared by the extubation study's tests.

Nothing here is real. Every identifier starts with ``SYN-`` (``SYN-PT-...``,
``SYN-HOSP-...``, ``SYN-SITE``), which no MIMIC or site identifier does, and every
timestamp hangs off an arbitrary base date in 2031.

Public shape
------------
``build_extubation_fixture(n_background=300, seed=20261003, *, outcome_scenarios=False)
-> ExtubationFixture``

``ExtubationFixture`` fields:

``tables`` : dict of CLIF 2.1-shaped frames, keyed by the name
    ``src.data.extubation_cohort`` uses for each table (file = ``clif_<key>.parquet``).

    - ``hospitalization``: patient_id, hospitalization_id, hospitalization_joined_id
      (all null), admission_dttm, discharge_dttm, age_at_admission,
      admission_type_category, discharge_category
    - ``patient``: patient_id, sex_category, race_category, ethnicity_category,
      death_dttm
    - ``adt``: hospitalization_id, hospital_id, in_dttm, out_dttm, location_category,
      location_type
    - ``respiratory_support``: hospitalization_id, recorded_dttm, device_category,
      mode_category, tracheostomy, fio2_set, lpm_set
    - ``labs``: hospitalization_id, lab_order_dttm, lab_collect_dttm, lab_result_dttm,
      lab_category, lab_value_numeric, reference_unit
    - ``vitals``: hospitalization_id, recorded_dttm, vital_category, vital_value
    - ``code_status``: patient_id, start_dttm, code_status_category
    - ``hospital_diagnosis``: hospitalization_id, diagnosis_code,
      diagnosis_code_format, diagnosis_primary, poa_present (all null, as on MIMIC)

``episodes`` : the canonical episode artifact for ``tables``, built with the real
    ``src.data.cohort.build_cohort`` and ``src.data.splits.assign_grouped_splits``
    (``PARTITIONS``, ``SPLIT_SEED``) and carrying the same hash columns, so
    ``validate_episode_artifact`` accepts it. Two scenario patients are deliberately not
    partitioned there (see ``absent_from_artifact`` and ``null_partition``).

``calendar_periods`` : patient_id, anchor_year_group -- the stand-in for MIMIC-IV's
    per-patient 3-year band. Event timestamps carry NO period information: every
    admission is drawn from the same few weeks whatever the band.

``truth`` : one row per BACKGROUND patient with the planted structure (never an input
    to the cohort): patient_id, hospitalization_id, arm, n_risk_factors, hypercapnia,
    age_over_65, bmi_over_30, copd, heart_failure, period_index,
    p_event_conventional_oxygen, p_event_hfnc, p_event_niv, event, event_cause
    (``reintubation`` | ``death`` | null), event_hours (hours after time zero).

``scenarios`` : dict scenario name -> patient_id for the hand-built patients below.

``patient_id(name)`` / ``hospitalization_id(name, index=1)`` give the ids directly.
``write_extubation_fixture(fixture, directory)`` writes ``clif_<key>.parquet`` files
(and ``mimic_anchor_year_group.parquet``) for path-based builders and CLIs.

Scenario patients (hours are since that patient's admission; IMV rows every 4 h)
------------------------------------------------------------------------------
- ``ae1_rescue_niv``: IMV 2-50, Face Mask at 50, Nasal Cannula at 52, NIPPV at 55.
- ``ae2_comfort_care``: IMV 2-62, code status AND at 60, Nasal Cannula at 62, dies at 68.
- ``late_comfort_care``: IMV 2-50, Nasal Cannula at 50, code status AND at 52.
- ``trach_flag``: IMV 2-100 with the tracheostomy flag set from hour 60, Face Mask at 100.
- ``trach_collar``: IMV 2-80, Trach Collar at 80.
- ``stitched_gap``: IMV 2-30, Nasal Cannula at 30, IMV again 30.5-50, High Flow NC at 50.
- ``vent_18h``: IMV 2-20, Nasal Cannula at 20.
- ``late_lab``: IMV 2-40, PaCO2 41 resulted at 30; PaCO2 58 collected 39.5 but resulted
  40.5; Nasal Cannula at 40.
- ``only_late_lab``: IMV 2-40, only PaCO2 is resulted at 41; Nasal Cannula at 40.
- ``hypercapnic_niv``: IMV 2-48, PaCO2 52 resulted at 45, NIPPV at 48.
- ``alternating``: IMV 2-50, High Flow NC at 50, NIPPV at 51.5, High Flow NC at 52.5.
- ``absent_from_artifact``: never in an ICU location, so not in ``episodes`` at all.
- ``null_partition``: in the ICU for 20 h only, so in ``episodes`` with a null partition.
- ``missing_code_status``: no code-status row.
- ``dni``: code status DNR/DNI at 10. ``dnr_only``: code status DNR at 10.
- ``underage``: age 16.
- ``two_extubations``: IMV 2-40, Nasal Cannula at 40, IMV 55-90, High Flow NC at 90.
- ``died_on_vent``: IMV 2-60 and no other device row.
- ``single_imv_row``: one IMV row at 2, Nasal Cannula at 40.
- ``other_first``: IMV 2-40, Other at 40, Nasal Cannula at 41.
- ``obese``: weight 110 kg and height 170 cm at hour 5; weight 60 kg at hour 42 (after
  time zero 40).
- ``copd_history``: an earlier hospitalization coded J449 and I5022.
- ``index_code_only``: no earlier hospitalization; the index stay is coded I5022.

Outcome scenario patients (only with ``outcome_scenarios=True``; added after the
background, so the default fixture is unchanged). Each is ventilated 2-40 and extubated
to Nasal Cannula at 40 (time zero), full code, discharged Home at 300 unless stated.
------------------------------------------------------------------------------
- ``out_reintubated_80h``: IMV again from 120 (80 h after time zero).
- ``out_reintubated_72h``: IMV again from 112 (exactly 72 h after time zero).
- ``out_death_day3``: dies in hospital at 100 (60 h), discharge Expired.
- ``out_discharged_day2_died_day5``: discharged Home at 88 (48 h), death timestamp at
  160 (120 h).
- ``out_discharged_day2_alive``: discharged Home at 88 (48 h), no death timestamp.
- ``out_discharge_unknown``: discharge category Missing at 88 (48 h).
- ``out_niv_day1``: NIPPV at 60 and 64, High Flow NC at 68, Nasal Cannula at 72; never
  back on IMV.
- ``out_reintubated_then_died``: IMV again 70-100 (30 h), dies at 130 (90 h), Expired.
- ``out_reintubated_at_death``: one IMV row at 90, the instant of death (50 h), Expired.
- ``out_hospice_day3``: discharged to Hospice at 100 (60 h), death timestamp at 140.
- ``out_died_10h``: dies at 50 (10 h), Expired, never back on IMV.
- ``out_reintubated_died_20h``: IMV again 46-60 (6 h), dies at 60 (20 h), Expired.
- ``out_trach_collar_day3``: Trach Collar with the tracheostomy flag at 100 (60 h); no
  IMV row after time zero.
- ``out_reintubated_via_trach``: IMV again 90-120 (50 h) with the tracheostomy flag set.
- ``out_expired_no_timestamp``: discharge Expired at 100 (60 h), no death timestamp.
- ``out_death_date_floor``: discharged Home at 88 (48 h); death timestamp at 80 (a
  day-resolution timestamp floored to before the discharge).
- ``out_charted_after_discharge``: discharge recorded at 39.75, before the time-zero row.

Background patients (``n_background``) -- the planted structure
----------------------------------------------------------------
Each is ventilated 26-220 h and extubated to one of three arms. ``PLANTED`` holds the
coefficients. Device choice is confounded by indication: hypercapnia, COPD and obesity
raise the odds of NIV; age and a later calendar band raise the odds of HFNC (a planted
adoption trend). The 7-day event (reintubation, or death for a quarter of events) has
log-odds ``intercept + per_risk_factor * n_risk_factors`` under conventional oxygen;
HFNC lowers the log-odds for everyone and NIV lowers them only for patients with at
least one risk factor, so NIV's advantage over HFNC grows with baseline risk. About 15%
have no pre-extubation PaCO2 and about 40% have no code status.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from src.data.cohort import build_cohort
from src.data.splits import assign_grouped_splits, content_manifest

BASE = datetime(2031, 3, 1, tzinfo=UTC)
SYNTHETIC_SITE = "SYN-SITE"
SPLIT_SEED = 42
PARTITIONS = {"train": 0.60, "validation": 0.15, "calibration": 0.10, "internal_test": 0.15}
PERIOD_BANDS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]
EPISODE_CONFIG = {
    "anchor_hours": 24,
    "prediction_horizon_hours": 48,
    "minimum_age": 18,
    "icu_location_category": "icu",
}
PLANTED = {
    "event_intercept": -2.4,
    "event_per_risk_factor": 0.5,
    "event_effect_hfnc": -0.4,                 # log-odds, every patient
    "event_effect_niv_if_any_risk_factor": -0.9,   # log-odds, n_risk_factors >= 1 only
    "death_share_of_events": 0.25,
    "niv_logit": {"intercept": -2.5, "hypercapnia": 1.6, "copd": 0.9, "bmi_over_30": 0.5},
    "hfnc_logit": {"intercept": -2.0, "age_over_65": 0.5, "per_period": 0.6},
    "paco2_missing": 0.15,
    "code_status_missing": 0.40,
}

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "hospitalization": {
        "patient_id": pl.String, "hospitalization_id": pl.String,
        "hospitalization_joined_id": pl.String,
        "admission_dttm": pl.Datetime("us", "UTC"), "discharge_dttm": pl.Datetime("us", "UTC"),
        "age_at_admission": pl.Int64, "admission_type_category": pl.String,
        "discharge_category": pl.String,
    },
    "patient": {
        "patient_id": pl.String, "sex_category": pl.String, "race_category": pl.String,
        "ethnicity_category": pl.String, "death_dttm": pl.Datetime("us", "UTC"),
    },
    "adt": {
        "hospitalization_id": pl.String, "hospital_id": pl.String,
        "in_dttm": pl.Datetime("us", "UTC"), "out_dttm": pl.Datetime("us", "UTC"),
        "location_category": pl.String, "location_type": pl.String,
    },
    "respiratory_support": {
        "hospitalization_id": pl.String, "recorded_dttm": pl.Datetime("us", "UTC"),
        "device_category": pl.String, "mode_category": pl.String,
        "tracheostomy": pl.Boolean, "fio2_set": pl.Float64, "lpm_set": pl.Float64,
    },
    "labs": {
        "hospitalization_id": pl.String, "lab_order_dttm": pl.Datetime("us", "UTC"),
        "lab_collect_dttm": pl.Datetime("us", "UTC"),
        "lab_result_dttm": pl.Datetime("us", "UTC"), "lab_category": pl.String,
        "lab_value_numeric": pl.Float64, "reference_unit": pl.String,
    },
    "vitals": {
        "hospitalization_id": pl.String, "recorded_dttm": pl.Datetime("us", "UTC"),
        "vital_category": pl.String, "vital_value": pl.Float64,
    },
    "code_status": {
        "patient_id": pl.String, "start_dttm": pl.Datetime("us", "UTC"),
        "code_status_category": pl.String,
    },
    "hospital_diagnosis": {
        "hospitalization_id": pl.String, "diagnosis_code": pl.String,
        "diagnosis_code_format": pl.String, "diagnosis_primary": pl.Int32,
        "poa_present": pl.Int32,
    },
}


def patient_id(name: str) -> str:
    return f"SYN-PT-{name}"


def hospitalization_id(name: str, index: int = 1) -> str:
    return f"SYN-HOSP-{name}-{index}"


@dataclass(frozen=True)
class ExtubationFixture:
    tables: dict[str, pl.DataFrame]
    episodes: pl.DataFrame
    calendar_periods: pl.DataFrame
    truth: pl.DataFrame
    scenarios: dict[str, str]


class _Builder:
    """Accumulates CLIF rows; hours are relative to each patient's own admission."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {name: [] for name in SCHEMAS}
        self.origin: dict[str, datetime] = {}
        self.periods: list[dict] = []
        self._count = 0

    def at(self, name: str, hours: float) -> datetime:
        return self.origin[name] + timedelta(hours=hours)

    def patient(self, name: str, *, period: str = PERIOD_BANDS[0], sex: str = "Female") -> None:
        # Admissions all fall in the same few weeks, whatever the calendar band: the
        # shifted dates say nothing about the period.
        self.origin[name] = BASE + timedelta(hours=7 * (self._count % 97))
        self._count += 1
        self.rows["patient"].append({
            "patient_id": patient_id(name), "sex_category": sex,
            "race_category": "Unknown", "ethnicity_category": "Unknown", "death_dttm": None,
        })
        self.periods.append({"patient_id": patient_id(name), "anchor_year_group": period})

    def stay(self, name: str, *, age: int = 60, admit: float = 0.0, los: float = 300.0,
             discharge: str = "Home", index: int = 1, icu: tuple[float, float] | None = None,
             unit: str = "medical_icu") -> str:
        hosp = hospitalization_id(name, index)
        self.rows["hospitalization"].append({
            "patient_id": patient_id(name), "hospitalization_id": hosp,
            "hospitalization_joined_id": None,
            "admission_dttm": self.at(name, admit), "discharge_dttm": self.at(name, admit + los),
            "age_at_admission": age, "admission_type_category": "ed",
            "discharge_category": discharge,
        })
        icu = (admit, admit + los) if icu is None else icu
        if icu[1] > icu[0]:
            self.rows["adt"].append({
                "hospitalization_id": hosp, "hospital_id": SYNTHETIC_SITE,
                "in_dttm": self.at(name, icu[0]), "out_dttm": self.at(name, icu[1]),
                "location_category": "icu", "location_type": unit,
            })
        if icu[1] < admit + los:
            self.rows["adt"].append({
                "hospitalization_id": hosp, "hospital_id": SYNTHETIC_SITE,
                "in_dttm": self.at(name, icu[1]), "out_dttm": self.at(name, admit + los),
                "location_category": "ward", "location_type": "general_ward",
            })
        return hosp

    def device(self, name: str, hours: float, category: str | None, *, trach: bool = False,
               index: int = 1) -> None:
        self.rows["respiratory_support"].append({
            "hospitalization_id": hospitalization_id(name, index),
            "recorded_dttm": self.at(name, hours), "device_category": category,
            "mode_category": "Assist Control-Volume Control" if category == "IMV" else None,
            "tracheostomy": trach, "fio2_set": 0.4, "lpm_set": None,
        })

    def ventilate(self, name: str, start: float, end: float, *, trach_from: float | None = None,
                  index: int = 1) -> None:
        """IMV rows every 4 h from `start`, plus one half an hour before `end`."""
        hours = [start + 4 * k for k in range(int((end - start - 0.5) // 4) + 1)]
        if hours[-1] < end - 0.5:
            hours.append(end - 0.5)
        for hour in hours:
            self.device(name, hour, "IMV", index=index,
                        trach=trach_from is not None and hour >= trach_from)
            if hour == hours[len(hours) // 2]:
                # A ventilator-settings row with no device charted (ignored as a device).
                self.device(name, hour + 1, None, index=index)

    def lab(self, name: str, value: float, *, collect: float, result: float,
            category: str = "pco2_arterial", unit: str = "mmHg", index: int = 1) -> None:
        self.rows["labs"].append({
            "hospitalization_id": hospitalization_id(name, index),
            "lab_order_dttm": self.at(name, collect), "lab_collect_dttm": self.at(name, collect),
            "lab_result_dttm": self.at(name, result), "lab_category": category,
            "lab_value_numeric": value, "reference_unit": unit,
        })

    def vital(self, name: str, category: str, value: float, hours: float, index: int = 1) -> None:
        self.rows["vitals"].append({
            "hospitalization_id": hospitalization_id(name, index),
            "recorded_dttm": self.at(name, hours), "vital_category": category,
            "vital_value": value,
        })

    def code_status(self, name: str, hours: float, category: str) -> None:
        self.rows["code_status"].append({
            "patient_id": patient_id(name), "start_dttm": self.at(name, hours),
            "code_status_category": category,
        })

    def diagnosis(self, name: str, code: str, *, index: int = 1, fmt: str = "ICD10CM") -> None:
        self.rows["hospital_diagnosis"].append({
            "hospitalization_id": hospitalization_id(name, index), "diagnosis_code": code,
            "diagnosis_code_format": fmt, "diagnosis_primary": 0, "poa_present": None,
        })

    def death(self, name: str, hours: float) -> None:
        for row in self.rows["patient"]:
            if row["patient_id"] == patient_id(name):
                row["death_dttm"] = self.at(name, hours)

    def frames(self) -> dict[str, pl.DataFrame]:
        return {name: pl.DataFrame(rows, schema=SCHEMAS[name]) for name, rows in self.rows.items()}


def _scenario(b: _Builder, name: str, *, vent: tuple[float, float] = (2.0, 40.0),
              code: str | None = "Full", **stay) -> None:
    """A patient, one stay, one IMV episode and (by default) a Full code status."""
    b.patient(name)
    b.stay(name, **stay)
    b.ventilate(name, *vent)
    if code is not None:
        b.code_status(name, 1.0, code)


def _add_scenarios(b: _Builder) -> list[str]:
    _scenario(b, "ae1_rescue_niv", vent=(2, 50))
    b.device("ae1_rescue_niv", 50, "Face Mask")
    b.device("ae1_rescue_niv", 52, "Nasal Cannula")
    b.device("ae1_rescue_niv", 55, "NIPPV")
    b.lab("ae1_rescue_niv", 40.0, collect=47.5, result=48.5)

    _scenario(b, "ae2_comfort_care", vent=(2, 62), los=68, discharge="Expired")
    b.code_status("ae2_comfort_care", 60, "AND")
    b.device("ae2_comfort_care", 62, "Nasal Cannula")
    b.death("ae2_comfort_care", 68)

    _scenario(b, "late_comfort_care", vent=(2, 50))
    b.device("late_comfort_care", 50, "Nasal Cannula")
    b.code_status("late_comfort_care", 52, "AND")

    b.patient("trach_flag")
    b.stay("trach_flag")
    b.ventilate("trach_flag", 2, 100, trach_from=60)
    b.code_status("trach_flag", 1, "Full")
    b.device("trach_flag", 100, "Face Mask", trach=True)

    _scenario(b, "trach_collar", vent=(2, 80))
    b.device("trach_collar", 80, "Trach Collar")

    _scenario(b, "stitched_gap", vent=(2, 30))
    b.device("stitched_gap", 30, "Nasal Cannula")
    b.ventilate("stitched_gap", 30.5, 50)
    b.device("stitched_gap", 50, "High Flow NC")

    _scenario(b, "vent_18h", vent=(2, 20))
    b.device("vent_18h", 20, "Nasal Cannula")

    _scenario(b, "late_lab")
    b.lab("late_lab", 41.0, collect=29.5, result=30.0)
    b.lab("late_lab", 58.0, collect=39.5, result=40.5)
    b.device("late_lab", 40, "Nasal Cannula")

    _scenario(b, "only_late_lab")
    b.lab("only_late_lab", 58.0, collect=39.5, result=41.0)
    b.device("only_late_lab", 40, "Nasal Cannula")

    _scenario(b, "hypercapnic_niv", vent=(2, 48))
    b.lab("hypercapnic_niv", 52.0, collect=44.5, result=45.0)
    b.device("hypercapnic_niv", 48, "NIPPV")

    _scenario(b, "alternating", vent=(2, 50))
    b.device("alternating", 50, "High Flow NC")
    b.device("alternating", 51.5, "NIPPV")
    b.device("alternating", 52.5, "High Flow NC")

    _scenario(b, "absent_from_artifact", icu=(0, 0), unit="general_ward")
    b.device("absent_from_artifact", 40, "Face Mask")

    _scenario(b, "null_partition", icu=(0, 20))
    b.device("null_partition", 40, "Nasal Cannula")

    _scenario(b, "missing_code_status", code=None)
    b.device("missing_code_status", 40, "Nasal Cannula")

    _scenario(b, "dni", vent=(2, 50))
    b.code_status("dni", 10, "DNR/DNI")
    b.device("dni", 50, "Nasal Cannula")

    _scenario(b, "dnr_only", vent=(2, 50))
    b.code_status("dnr_only", 10, "DNR")
    b.device("dnr_only", 50, "Nasal Cannula")

    _scenario(b, "underage", age=16)
    b.device("underage", 40, "Nasal Cannula")

    _scenario(b, "two_extubations")
    b.device("two_extubations", 40, "Nasal Cannula")
    b.ventilate("two_extubations", 55, 90)
    b.device("two_extubations", 90, "High Flow NC")

    _scenario(b, "died_on_vent", vent=(2, 60), los=60, discharge="Expired")
    b.death("died_on_vent", 60)

    b.patient("single_imv_row")
    b.stay("single_imv_row")
    b.device("single_imv_row", 2, "IMV")
    b.code_status("single_imv_row", 1, "Full")
    b.device("single_imv_row", 40, "Nasal Cannula")

    _scenario(b, "other_first")
    b.device("other_first", 40, "Other")
    b.device("other_first", 41, "Nasal Cannula")

    _scenario(b, "obese")
    b.vital("obese", "weight_kg", 110.0, 5)
    b.vital("obese", "height_cm", 170.0, 5)
    b.vital("obese", "weight_kg", 60.0, 42)
    b.device("obese", 40, "Nasal Cannula")

    _scenario(b, "copd_history")
    b.stay("copd_history", admit=-2000, los=100, index=0)
    b.diagnosis("copd_history", "J449", index=0)
    b.diagnosis("copd_history", "I5022", index=0)
    b.device("copd_history", 40, "Nasal Cannula")

    _scenario(b, "index_code_only")
    b.diagnosis("index_code_only", "I5022")
    b.device("index_code_only", 40, "Nasal Cannula")
    return list(b.origin)


def _add_outcome_scenarios(b: _Builder) -> list[str]:
    """Hand-built outcome patients for the label tests; time zero is hour 40 for each."""
    before = set(b.origin)

    def extubated(name: str, **stay) -> None:
        _scenario(b, name, **stay)
        b.device(name, 40, "Nasal Cannula")

    extubated("out_reintubated_80h")
    b.ventilate("out_reintubated_80h", 120, 150)

    extubated("out_reintubated_72h")
    b.ventilate("out_reintubated_72h", 112, 140)

    extubated("out_death_day3", los=100, discharge="Expired")
    b.death("out_death_day3", 100)

    extubated("out_discharged_day2_died_day5", los=88)
    b.death("out_discharged_day2_died_day5", 160)

    extubated("out_discharged_day2_alive", los=88)

    extubated("out_discharge_unknown", los=88, discharge="Missing")

    extubated("out_niv_day1")
    b.device("out_niv_day1", 60, "NIPPV")
    b.device("out_niv_day1", 64, "NIPPV")
    b.device("out_niv_day1", 68, "High Flow NC")
    b.device("out_niv_day1", 72, "Nasal Cannula")

    extubated("out_reintubated_then_died", los=130, discharge="Expired")
    b.ventilate("out_reintubated_then_died", 70, 100)
    b.death("out_reintubated_then_died", 130)

    extubated("out_reintubated_at_death", los=90, discharge="Expired")
    b.device("out_reintubated_at_death", 90, "IMV")
    b.death("out_reintubated_at_death", 90)

    extubated("out_hospice_day3", los=100, discharge="Hospice")
    b.death("out_hospice_day3", 140)

    extubated("out_died_10h", los=50, discharge="Expired")
    b.death("out_died_10h", 50)

    extubated("out_reintubated_died_20h", los=60, discharge="Expired")
    b.ventilate("out_reintubated_died_20h", 46, 60)
    b.death("out_reintubated_died_20h", 60)

    extubated("out_trach_collar_day3")
    b.device("out_trach_collar_day3", 100, "Trach Collar", trach=True)

    extubated("out_reintubated_via_trach")
    b.ventilate("out_reintubated_via_trach", 90, 120, trach_from=90)

    extubated("out_expired_no_timestamp", los=100, discharge="Expired")

    extubated("out_death_date_floor", los=88)
    b.death("out_death_date_floor", 80)

    extubated("out_charted_after_discharge", los=39.75)
    return [name for name in b.origin if name not in before]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _add_background(b: _Builder, n: int, rng: random.Random) -> list[dict]:
    truth: list[dict] = []
    for i in range(n):
        name = f"bg{i:04d}"
        period_index = rng.randrange(len(PERIOD_BANDS))
        age = rng.randint(35, 90)
        height = rng.uniform(150.0, 190.0)
        lean, obese_bmi = rng.uniform(19.0, 29.0), rng.uniform(31.0, 45.0)
        bmi = rng.choice([lean, lean, obese_bmi])
        paco2 = round(min(max(rng.gauss(42.0, 7.0), 28.0), 75.0), 1)
        has_history = rng.random() < 0.6
        copd = has_history and rng.random() < 0.33
        heart_failure = has_history and rng.random() < 0.25
        vent_hours = float(rng.randint(26, 220))
        extubation = 2.0 + vent_hours
        hypercapnia, over_65, obese = paco2 > 45.0, age > 65, bmi > 30.0
        n_risk = sum([hypercapnia, over_65, obese, copd, heart_failure])

        niv, hfnc = PLANTED["niv_logit"], PLANTED["hfnc_logit"]
        weights = {
            "conventional_oxygen": 1.0,
            "hfnc": math.exp(hfnc["intercept"] + hfnc["age_over_65"] * over_65
                             + hfnc["per_period"] * period_index),
            "niv": math.exp(niv["intercept"] + niv["hypercapnia"] * hypercapnia
                            + niv["copd"] * copd + niv["bmi_over_30"] * obese),
        }
        arm = rng.choices(list(weights), weights=list(weights.values()))[0]
        base = PLANTED["event_intercept"] + PLANTED["event_per_risk_factor"] * n_risk
        p_event = {
            "conventional_oxygen": _sigmoid(base),
            "hfnc": _sigmoid(base + PLANTED["event_effect_hfnc"]),
            "niv": _sigmoid(base + (PLANTED["event_effect_niv_if_any_risk_factor"]
                                    if n_risk >= 1 else 0.0)),
        }
        event = rng.random() < p_event[arm]
        cause = None
        event_hours = None
        if event:
            cause = "death" if rng.random() < PLANTED["death_share_of_events"] else "reintubation"
            event_hours = round(rng.uniform(8.0, 160.0), 1)

        b.patient(name, period=PERIOD_BANDS[period_index], sex=rng.choice(["Female", "Male"]))
        died = cause == "death"
        los = extubation + (event_hours if died else 260.0)
        b.stay(name, age=age, los=los, discharge="Expired" if died else "Home",
               unit=rng.choice(["medical_icu", "surgical_icu"]))
        if died:
            b.death(name, los)
        if has_history:
            b.stay(name, admit=-3000.0, los=120.0, index=0)
            b.diagnosis(name, "E119", index=0)
            if copd:
                b.diagnosis(name, "J449", index=0)
            if heart_failure:
                b.diagnosis(name, "I5022", index=0)
        b.ventilate(name, 2.0, extubation)
        b.vital(name, "weight_kg", round(bmi * (height / 100.0) ** 2, 1), 3.0)
        b.vital(name, "height_cm", round(height, 1), 3.0)
        if rng.random() >= PLANTED["paco2_missing"]:
            resulted = extubation - rng.uniform(1.0, 12.0)
            b.lab(name, paco2, collect=resulted - 0.5, result=resulted)
        if rng.random() >= PLANTED["code_status_missing"]:
            b.code_status(name, 1.0, "Full")

        first = {"conventional_oxygen": rng.choice(["Face Mask", "Face Mask", "Nasal Cannula"]),
                 "hfnc": "High Flow NC", "niv": "NIPPV"}[arm]
        alternates = arm == "niv" and rng.random() < 0.3
        end_of_support = extubation + min(24.0, (event_hours or 24.0) - 1.0)
        hour, step = extubation, 0
        while hour < end_of_support:
            device = first
            if alternates and step % 2 == 1:
                device = "High Flow NC"
            if arm == "conventional_oxygen" and step >= 2:
                device = "Nasal Cannula"
            b.device(name, hour, device)
            hour, step = hour + 3.0, step + 1
        if event and arm != "niv" and rng.random() < 0.3 and event_hours > 7.0:
            b.device(name, extubation + event_hours - 1.0, "NIPPV")     # rescue before the event
        if cause == "reintubation":
            b.ventilate(name, extubation + event_hours, extubation + event_hours + 24.0)

        truth.append({
            "patient_id": patient_id(name), "hospitalization_id": hospitalization_id(name),
            "arm": arm, "n_risk_factors": n_risk, "hypercapnia": hypercapnia,
            "age_over_65": over_65, "bmi_over_30": obese, "copd": copd,
            "heart_failure": heart_failure, "period_index": period_index,
            "p_event_conventional_oxygen": p_event["conventional_oxygen"],
            "p_event_hfnc": p_event["hfnc"], "p_event_niv": p_event["niv"],
            "event": event, "event_cause": cause, "event_hours": event_hours,
        })
    return truth


def episode_artifact(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The canonical episode artifact for `tables`, as `build_cohort_artifact` writes it."""
    episodes, waterfall = build_cohort(
        tables["hospitalization"], tables["adt"], EPISODE_CONFIG, return_waterfall=True
    )
    eligible = assign_grouped_splits(
        episodes.filter(pl.col("eligible")), PARTITIONS, seed=SPLIT_SEED
    )
    episodes = episodes.join(
        eligible.select("hospitalization_id", "partition"), on="hospitalization_id", how="left"
    )
    split = content_manifest(eligible, columns=["hospitalization_id", "patient_id", "partition"])
    episode = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
    )
    return episodes.with_columns(
        pl.lit("1.0.0").alias("cohort_contract_version"),
        pl.lit(split["sha256"]).alias("split_sha256"),
        pl.lit(episode["sha256"]).alias("episode_sha256"),
        pl.lit(json.dumps({"source": "synthetic"})).alias("source_provenance_json"),
        pl.lit(json.dumps(waterfall, sort_keys=True)).alias("waterfall_json"),
    )


def build_extubation_fixture(n_background: int = 300, seed: int = 20261003, *,
                             outcome_scenarios: bool = False) -> ExtubationFixture:
    builder = _Builder()
    names = _add_scenarios(builder)
    truth = _add_background(builder, n_background, random.Random(seed))
    if outcome_scenarios:
        # After the background, so no existing patient's rows or admission time move.
        names += _add_outcome_scenarios(builder)
    tables = builder.frames()
    return ExtubationFixture(
        tables=tables,
        episodes=episode_artifact(tables),
        calendar_periods=pl.DataFrame(
            builder.periods, schema={"patient_id": pl.String, "anchor_year_group": pl.String}
        ),
        truth=pl.DataFrame(truth, infer_schema_length=None),
        scenarios={name: patient_id(name) for name in names},
    )


def write_extubation_fixture(fixture: ExtubationFixture, directory: str | Path) -> Path:
    """Write the CLIF tables (and the calendar-period source) as parquet files."""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    for name, frame in fixture.tables.items():
        frame.write_parquet(base / f"clif_{name}.parquet")
    fixture.calendar_periods.write_parquet(base / "mimic_anchor_year_group.parquet")
    return base
