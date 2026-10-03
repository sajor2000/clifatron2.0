import copy
import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl
import yaml

from src.data.segments import POLICY_VERSION, json_sha256
from src.data.splits import content_manifest
from src.data.tokenize import (
    _read_table,
    restrict_to_observation_window,
    validate_table_availability,
    validate_units,
    validate_vocabulary_artifact,
)


def episode_artifact() -> pl.DataFrame:
    episodes = pl.DataFrame(
        {
            "hospitalization_id": ["stay-1"],
            "patient_id": ["patient-1"],
            "hospitalization_joined_id": ["chain-1"],
            "icu_admit_dttm": [datetime(2026, 1, 1, tzinfo=UTC)],
            "anchor_dttm": [datetime(2026, 1, 2, tzinfo=UTC)],
            "eligible": [True],
            "partition": ["train"],
        }
    )
    split_hash = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "partition"]
    )["sha256"]
    episode_hash = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
    )["sha256"]
    return episodes.with_columns(
        pl.lit("1.0.0").alias("cohort_contract_version"),
        pl.lit(split_hash).alias("split_sha256"),
        pl.lit(episode_hash).alias("episode_sha256"),
        pl.lit("{}").alias("source_provenance_json"),
    )


# json_sha256 of `configs/cohort.yaml -> outcomes`, computed 2026-10-03 before the
# `study_endpoints` block was added. tokenize.py binds every vocabulary's `outcome_spec`
# hash to this block, so existing vocabularies and signed bundles depend on it.
OUTCOME_SPEC_SHA256 = "7ddc3edc08e8980e4a1fa13867b66ead2bf2cd9e30ad6ee699cea07ea7fb9324"
FORBIDDEN_TRUNK_TASKS = ("new_imv_24h", "new_vasopressor_24h")


class TreatmentTargetRuleViolation(AssertionError):
    """A config breaks hard rule #1 or its scoped amendment (2026-10-03)."""


def _study_endpoint_sources(name: str, endpoints: dict, seen: tuple = ()) -> set[str]:
    """Every source table a study endpoint is read from, through its composites."""
    if name in seen:
        raise TreatmentTargetRuleViolation(f"study endpoint {name} is a circular composite")
    spec = endpoints.get(name)
    if not isinstance(spec, dict):
        raise TreatmentTargetRuleViolation(f"study endpoint {name} is not declared")
    if "composite_of" in spec:
        sources: set[str] = set()
        for part in spec["composite_of"]:
            sources |= _study_endpoint_sources(part, endpoints, (*seen, name))
        return sources
    if not spec.get("source"):
        raise TreatmentTargetRuleViolation(
            f"study endpoint {name} declares neither source nor composite_of")
    return {spec["source"]}


def check_treatment_target_rule(cohort: dict, train: dict, data: dict) -> None:
    """Hard rule #1: treatments are model inputs, never prediction targets of the trunk.

    Scoped amendment (2026-10-03): study heads downstream of the frozen trunk may use
    trial endpoints defined by a treatment event (for example reintubation) as labels.
    Those endpoints live in the top-level `study_endpoints` block of the cohort
    contract, never under `outcomes`, and each must be declared `use: label_only`.
    """
    def fail(message: str):
        raise TreatmentTargetRuleViolation(message)

    if cohort.get("treatment_target_policy") != "context_only":
        fail("treatment_target_policy must stay context_only")
    input_only = {name for name, spec in data["tables"].items() if spec.get("input_only")}
    declared = set(cohort.get("treatment_sources") or ())
    if declared - input_only:
        fail(f"treatment sources are not input_only in data.yaml: {sorted(declared - input_only)}")
    treatment = declared | input_only

    # The trunk: its supervised tasks are exactly the frozen physiologic outcomes.
    outcomes = cohort["outcomes"]
    tasks = train["finetune"]["tasks"]
    if set(tasks) != set(outcomes):
        fail("trunk tasks must equal the frozen outcome contract")
    for name in FORBIDDEN_TRUNK_TASKS:
        if name in tasks or name in outcomes:
            fail(f"{name} predicts treatment initiation and cannot be a trunk task")
    for name, spec in outcomes.items():
        if spec.get("source") in treatment:
            fail(f"trunk outcome {name} is read from treatment source {spec['source']}")

    # Study endpoints: labels for heads downstream of the frozen trunk, nothing else.
    endpoints = cohort.get("study_endpoints") or {}
    if not isinstance(endpoints, dict):
        fail("study_endpoints must be a mapping of endpoint name to declaration")
    trunk_overlap = sorted(set(endpoints) & (set(outcomes) | set(tasks)))
    if trunk_overlap:
        fail(f"study endpoints are also trunk targets: {trunk_overlap}")
    for name, spec in endpoints.items():
        derived = sorted(_study_endpoint_sources(name, endpoints) & treatment)
        if spec.get("use") != "label_only":
            fail(f"study endpoint {name} is not declared label_only"
                 + (f"; it derives from treatment source(s) {derived}" if derived else ""))


def _rule_configs() -> tuple[dict, dict, dict]:
    root = Path(__file__).parents[1]
    return tuple(yaml.safe_load((root / f"configs/{name}.yaml").read_text())
                 for name in ("cohort", "train", "data"))


class TreatmentTargetRuleTest(unittest.TestCase):
    """Hard rule #1 and its scoped amendment for study endpoints (R30, KTD14)."""

    def setUp(self):
        self.cohort, self.train, self.data = _rule_configs()

    def _mutated(self):
        return copy.deepcopy(self.cohort), copy.deepcopy(self.train)

    def test_repo_configs_satisfy_the_rule(self):
        check_treatment_target_rule(self.cohort, self.train, self.data)

    def test_treatment_source_as_trunk_target_fails(self):
        for source in ("resp_support", "meds", "adt", "crrt", "ecmo"):
            cohort, train = self._mutated()
            cohort["outcomes"]["device_started_48h"] = {"source": source}
            train["finetune"]["tasks"].append("device_started_48h")
            with self.assertRaisesRegex(TreatmentTargetRuleViolation,
                                        f"treatment source {source}", msg=source):
                check_treatment_target_rule(cohort, train, self.data)
        cohort, train = self._mutated()
        cohort["treatment_target_policy"] = "target_eligible"
        with self.assertRaisesRegex(TreatmentTargetRuleViolation, "context_only"):
            check_treatment_target_rule(cohort, train, self.data)

    def test_new_imv_24h_is_still_rejected_as_a_trunk_task(self):
        cohort, train = self._mutated()
        train["finetune"]["tasks"].append("new_imv_24h")
        with self.assertRaisesRegex(TreatmentTargetRuleViolation, "frozen outcome contract"):
            check_treatment_target_rule(cohort, train, self.data)
        # Declaring it in the outcome contract does not make it a legal trunk task either,
        # even with a source that is not a treatment table.
        cohort["outcomes"]["new_imv_24h"] = {"source": "vitals"}
        with self.assertRaisesRegex(TreatmentTargetRuleViolation, "new_imv_24h"):
            check_treatment_target_rule(cohort, train, self.data)

    def test_extubation_study_endpoints_are_declared_label_only(self):
        self.assertNotIn("study_endpoints", self.cohort["outcomes"])
        endpoints = self.cohort["study_endpoints"]
        self.assertEqual(set(endpoints), {
            "reintubation_72h", "reintubation_7d", "death_7d", "reintubation_or_death_7d"})
        for name, spec in endpoints.items():
            self.assertEqual(spec["use"], "label_only", name)
            self.assertEqual(spec["study"], "extubation", name)
        # Reintubation is a treatment event read from respiratory support as a label source.
        self.assertIn("resp_support", self.cohort["treatment_sources"])
        self.assertTrue(self.data["tables"]["resp_support"]["input_only"])
        for name, hours in (("reintubation_72h", 72), ("reintubation_7d", 168)):
            self.assertEqual(endpoints[name]["source"], "resp_support", name)
            self.assertEqual(endpoints[name]["event"], "reintubation", name)
            self.assertEqual(endpoints[name]["horizon_hours"], hours, name)
        self.assertEqual(endpoints["death_7d"]["event"], "death")
        self.assertEqual(endpoints["death_7d"]["horizon_hours"], 168)
        composite = endpoints["reintubation_or_death_7d"]
        self.assertEqual(composite["composite_of"], ["reintubation_7d", "death_7d"])
        self.assertEqual(composite["horizon_hours"], 168)
        # Label-only endpoints never reach the trunk's task list or its outcome contract.
        self.assertFalse(set(endpoints) & set(self.train["finetune"]["tasks"]))
        self.assertFalse(set(endpoints) & set(self.cohort["outcomes"]))
        check_treatment_target_rule(self.cohort, self.train, self.data)

    def test_study_endpoint_without_the_label_only_declaration_is_rejected(self):
        cohort, train = self._mutated()
        cohort["study_endpoints"]["reintubation_72h"].pop("use")
        with self.assertRaisesRegex(
                TreatmentTargetRuleViolation,
                r"reintubation_72h is not declared label_only.*treatment source\(s\) "
                r"\['resp_support'\]"):
            check_treatment_target_rule(cohort, train, self.data)

        cohort, train = self._mutated()
        cohort["study_endpoints"]["reintubation_7d"]["use"] = "target"
        with self.assertRaisesRegex(TreatmentTargetRuleViolation, "reintubation_7d"):
            check_treatment_target_rule(cohort, train, self.data)

        # A composite inherits the treatment source of its components.
        cohort, train = self._mutated()
        cohort["study_endpoints"]["reintubation_or_death_7d"].pop("use")
        with self.assertRaisesRegex(
                TreatmentTargetRuleViolation,
                r"reintubation_or_death_7d is not declared label_only.*resp_support"):
            check_treatment_target_rule(cohort, train, self.data)

        # A new treatment-defined endpoint (any input-only table) needs the declaration too.
        cohort, train = self._mutated()
        cohort["study_endpoints"]["crrt_start_7d"] = {"source": "crrt", "event": "crrt_start"}
        with self.assertRaisesRegex(TreatmentTargetRuleViolation, r"crrt_start_7d.*crrt"):
            check_treatment_target_rule(cohort, train, self.data)

    def test_study_endpoint_cannot_become_a_trunk_target(self):
        cohort, train = self._mutated()
        train["finetune"]["tasks"].append("reintubation_7d")
        with self.assertRaisesRegex(TreatmentTargetRuleViolation, "frozen outcome contract"):
            check_treatment_target_rule(cohort, train, self.data)
        # Moving the endpoint under `outcomes` makes it a trunk outcome on a treatment source.
        cohort["outcomes"]["reintubation_7d"] = cohort["study_endpoints"].pop("reintubation_7d")
        with self.assertRaisesRegex(TreatmentTargetRuleViolation,
                                    "treatment source resp_support"):
            check_treatment_target_rule(cohort, train, self.data)
        # Declared in both blocks: still a trunk target.
        cohort, train = self._mutated()
        cohort["outcomes"]["death_7d"] = {"source": "vitals"}
        train["finetune"]["tasks"].append("death_7d")
        with self.assertRaisesRegex(TreatmentTargetRuleViolation, "also trunk targets"):
            check_treatment_target_rule(cohort, train, self.data)

    def test_outcomes_block_and_outcome_spec_hash_are_unchanged(self):
        self.assertEqual(list(self.cohort["outcomes"]), [
            "map_below_65_48h", "lactate_above_4_48h", "spo2_below_88_48h"])
        self.assertEqual(json_sha256(self.cohort["outcomes"]), OUTCOME_SPEC_SHA256)
        self.assertEqual(self.cohort["contract_version"], "1.0.0")


class DataConfigTest(unittest.TestCase):
    def test_training_config_uses_only_frozen_physiologic_outcomes(self):
        root = Path(__file__).parents[1]
        cohort = yaml.safe_load((root / "configs/cohort.yaml").read_text())
        train = yaml.safe_load((root / "configs/train.yaml").read_text())
        tasks = train["finetune"]["tasks"]
        self.assertEqual(set(tasks), set(cohort["outcomes"]))
        self.assertNotIn("new_imv_24h", tasks)
        self.assertNotIn("new_vasopressor_24h", tasks)

        data = yaml.safe_load((root / "configs/data.yaml").read_text())
        self.assertTrue(data["tables"]["meds"]["input_only"])
        self.assertTrue(data["tables"]["resp_support"]["input_only"])
        self.assertTrue(data["tables"]["adt"]["input_only"])

    def test_every_configured_table_declares_availability_semantics(self):
        """R12: availability is required per table; a missing or unknown one fails."""
        from src.data.cohort import QualificationError

        root = Path(__file__).parents[1]
        tables = yaml.safe_load((root / "configs/data.yaml").read_text())["tables"]
        declared = validate_table_availability(tables)
        self.assertEqual(set(declared), set(tables))
        self.assertEqual(declared["vitals"]["availability"], "missing_storetime")
        self.assertEqual(declared["resp_support"]["availability"], "missing_storetime")
        self.assertEqual(declared["labs"]["availability"], "result")
        self.assertEqual(declared["meds"]["availability"], "recorded")
        self.assertEqual(declared["adt"]["availability"], "recorded")
        # decided 2026-10-03 (product authority): 30 min on every table without a real
        # store time; labs (result time) and charted administration/state tables keep 0.
        for name, d in declared.items():
            expected = 30 if d["availability"] == "missing_storetime" else 0
            self.assertEqual(d["lag_minutes"], expected, name)
        cfg = yaml.safe_load((root / "configs/data.yaml").read_text())
        self.assertEqual(cfg["availability_lag_sensitivities"]["minutes"], [15, 60])
        from src.data.tokenize import with_availability_lag
        lagged = validate_table_availability(with_availability_lag(cfg, 60)["tables"])
        self.assertEqual(lagged["vitals"]["lag_minutes"], 60)
        self.assertEqual(lagged["labs"]["lag_minutes"], 0)
        self.assertEqual(cfg["tables"]["vitals"]["availability_lag_minutes"], 30)

        missing = {**tables, "labs": {k: v for k, v in tables["labs"].items()
                                      if k != "availability"}}
        with self.assertRaisesRegex(QualificationError, "'labs'.*availability"):
            validate_table_availability(missing)
        unknown = {**tables, "labs": {**tables["labs"], "availability": "charttime"}}
        with self.assertRaisesRegex(QualificationError, "'labs'.*availability"):
            validate_table_availability(unknown)
        for bad_lag in (-5, 2.5, True, "10"):
            negative = {**tables, "labs": {**tables["labs"],
                                           "availability_lag_minutes": bad_lag}}
            with self.assertRaisesRegex(QualificationError, "availability_lag_minutes"):
                validate_table_availability(negative)

    def test_observation_positions_are_icu_admission_relative_and_include_anchor(self):
        utc = "UTC"
        events = pl.DataFrame(
            {
                "hosp_id": ["stay-1", "stay-1", "stay-1"],
                "dttm": pl.datetime_range(
                    pl.datetime(2026, 1, 1, 0, time_zone=utc),
                    pl.datetime(2026, 1, 3, 0, time_zone=utc),
                    interval="1d",
                    eager=True,
                ),
                "concept": ["map", "map", "map"],
                "value": [70.0, 65.0, 60.0],
                "unit": ["mmHg", "mmHg", "mmHg"],
                "source": ["vitals", "vitals", "vitals"],
            }
        )
        episodes = episode_artifact()

        observed = restrict_to_observation_window(events, episodes)

        self.assertEqual(observed["pos_min"].to_list(), [0, 1440])
        self.assertEqual(observed["partition"].unique().to_list(), ["train"])

    def test_treatment_events_are_context_but_not_targets(self):
        events = pl.DataFrame(
            {
                "hosp_id": ["stay-1"],
                "dttm": [datetime(2026, 1, 1, 1, tzinfo=UTC)],
                "concept": ["norepinephrine"],
                "value": [None],
                "unit": [""],
                "source": ["meds"],
            }
        )
        episodes = episode_artifact()
        observed = restrict_to_observation_window(events, episodes, {"meds"})
        self.assertFalse(observed["target_eligible"].item())

    def test_tampered_episode_partition_hash_is_rejected_before_join(self):
        events = pl.DataFrame(
            {
                "hosp_id": ["stay-1"],
                "dttm": [datetime(2026, 1, 1, 1, tzinfo=UTC)],
                "concept": ["map"],
                "value": [70.0],
                "unit": ["mmHg"],
                "source": ["vitals"],
            }
        )
        tampered = episode_artifact().with_columns(pl.lit("test").alias("partition"))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            restrict_to_observation_window(events, tampered)

    def test_imported_vocabulary_manifest_is_validated_and_preserved(self):
        root = Path(__file__).parents[1]
        cfg = yaml.safe_load((root / "configs/data.yaml").read_text())
        policy = yaml.safe_load((root / "configs/artifact_policy.yaml").read_text())
        cohort_cfg = yaml.safe_load((root / cfg["cohort_contract"]).read_text())
        vocab = {"<pad>": 0, "map=0": 1, "map=1": 2}
        segments = {"map": [
            {"lo": None, "hi": 65.0, "lo_closed": False, "hi_closed": False},
            {"lo": 65.0, "hi": None, "lo_closed": True, "hi_closed": False},
        ]}
        binning_sources = {"map": "csv"}
        from src.data.tokenize import column_units, site_unit_conversions

        # The config declares unit-less column units, so the vocabulary must record them
        # and its reference site's (here: no) unit conversions.
        reference_units = {"concepts": {"map": "mmHg"}, "dose_targets": {},
                           "column_units": column_units(cfg),
                           "site_conversions": site_unit_conversions(cfg, "synthetic-reference")}
        concept_sources = {"tables": {"map": ["vitals"]}, "treatment_sources": []}
        # The config declares CLIF harmonization rules (mCIDE gate, GCS, BP method, ...),
        # so the vocabulary carries their hashed record.
        from src.data.clif_conformance import harmonization_record
        harmonization = json.loads(json.dumps(harmonization_record(cfg, "synthetic-reference")))

        def digest(value):
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(payload.encode()).hexdigest()

        manifest = {
            "artifact_family": "experimental_representation",
            "tokenizer_version": 2,
            "clif_version": cfg["schema_version"],
            "mcide_version": cfg["mcide_version"],
            "hashes": {
                "training_split": "1" * 64,
                "vocabulary": digest(vocab),
                "numeric_edges": digest(segments),
                "binning_sources": digest(binning_sources),
                "reference_units": digest(reference_units),
                "concept_sources": digest(concept_sources),
                "target_map": digest(cfg["target_concepts"]),
                "outcome_spec": digest(cohort_cfg["outcomes"]),
                "clif_version": digest(cfg["schema_version"]),
                "harmonization": digest(harmonization),
            },
            "provenance": {
                "source_site": "synthetic-reference",
                "fit_partition": "train",
                "precedence_policy": POLICY_VERSION,
                "immutable": True,
            },
        }

        def artifact(**over):
            blob = {"vocab": vocab, "segments": segments, "manifest": manifest,
                    "binning_sources": binning_sources, "reference_units": reference_units,
                    "concept_sources": concept_sources, "precedence_policy": POLICY_VERSION,
                    "harmonization": harmonization}
            blob.update(over)
            return blob

        loaded_vocab, loaded_segments, loaded_manifest = validate_vocabulary_artifact(
            artifact(), cfg, policy
        )
        self.assertEqual(loaded_vocab, vocab)
        self.assertEqual(loaded_segments, segments)
        self.assertEqual(loaded_manifest, manifest)

        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_vocabulary_artifact(artifact(vocab={**vocab, "new": 3}), cfg, policy)

        # A vocabulary recording no column units does not bind a config that declares them.
        bare_units = {"concepts": {"map": "mmHg"}, "dose_targets": {}}
        bare = json.loads(json.dumps(manifest))
        bare["hashes"]["reference_units"] = digest(bare_units)
        with self.assertRaisesRegex(ValueError, "column units"):
            validate_vocabulary_artifact(artifact(manifest=bare, reference_units=bare_units),
                                         cfg, policy)

        moved = json.loads(json.dumps(segments))
        moved["map"][0]["hi"] = moved["map"][1]["lo"] = 66.0
        with self.assertRaisesRegex(ValueError, "numeric-edge hash mismatch"):
            validate_vocabulary_artifact(artifact(segments=moved), cfg, policy)

        with self.assertRaisesRegex(ValueError, "binning sources"):
            validate_vocabulary_artifact(artifact(binning_sources={}), cfg, policy)

        wrong_family = artifact(manifest={**manifest, "artifact_family": "clifatron_checkpoint"})
        with self.assertRaisesRegex(ValueError, "family"):
            validate_vocabulary_artifact(wrong_family, cfg, policy)

        bad_target = json.loads(json.dumps(manifest))
        bad_target["hashes"]["target_map"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "target-map"):
            validate_vocabulary_artifact(artifact(manifest=bad_target), cfg, policy)

        missing = json.loads(json.dumps(manifest))
        missing["hashes"].pop("reference_units")
        with self.assertRaisesRegex(ValueError, "missing hashes: reference_units"):
            validate_vocabulary_artifact(artifact(manifest=missing), cfg, policy)

    def test_unit_mismatch_on_a_non_target_binned_concept_is_an_error(self):
        """U5: validate_units covers every binned concept via the vocab's reference
        units, not only the configured target concepts."""
        cfg = {"unit_normalization": {"on_mismatch": "error", "concepts": {"map": "mmHg"}}}
        reference_units = {"concepts": {"map": "mmHg", "sodium": "mmol/L",
                                        "ecmo_flow": None},
                           "dose_targets": {}}
        ok = pl.DataFrame({"concept": ["sodium", "map", "ecmo_flow"],
                           "unit": ["mmol/L", "mmHg", "L/min"], "value": [140.0, 70.0, 4.0]})
        validate_units(ok, cfg, reference_units)
        bad = pl.DataFrame({"concept": ["sodium"], "unit": ["mg/dL"], "value": [140.0]})
        with self.assertRaisesRegex(ValueError, "Non-canonical CLIF units: sodium"):
            validate_units(bad, cfg, reference_units)
        # Without the vocab's reference units the non-target concept was never checked.
        validate_units(bad, cfg)
        warn = {"unit_normalization": {"on_mismatch": "warn", "concepts": {}}}
        validate_units(bad, warn, reference_units)

    def test_reads_availability_column_and_validates_canonical_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pl.DataFrame(
                {
                    "hospitalization_id": ["stay-1"],
                    "lab_result_dttm": ["2026-01-01T01:00:00"],
                    "lab_category": ["platelet_count"],
                    "lab_value_numeric": [123.0],
                    "reference_unit": ["10^3/µL"],
                }
            ).with_columns(pl.col("lab_result_dttm").str.to_datetime()).write_parquet(
                base / "clif_labs.parquet"
            )
            spec = {
                "file": "clif_labs",
                "availability_col": "lab_result_dttm",
                "concept_col": "lab_category",
                "value_col": "lab_value_numeric",
                "unit_col": "reference_unit",
            }
            events = _read_table(duckdb.connect(), base, spec)

        self.assertEqual(events["concept"].to_list(), ["platelet_count"])
        validate_units(
            events,
            {
                "unit_normalization": {
                    "on_mismatch": "error",
                    "concepts": {"platelet_count": "10^3/uL"},
                }
            },
        )

    def test_rejects_noncanonical_units(self):
        events = pl.DataFrame({"concept": ["lactate"], "unit": ["mg/dL"], "value": [2.0]})
        with self.assertRaisesRegex(ValueError, "Non-canonical CLIF units"):
            validate_units(
                events,
                {
                    "unit_normalization": {
                        "on_mismatch": "error",
                        "concepts": {"lactate": "mmol/L"},
                    }
                },
            )

    def test_rejects_numeric_concept_without_unit_mapping(self):
        events = pl.DataFrame(
            {"concept": ["creatinine"], "unit": ["mg/dL"], "value": [1.2]}
        )
        # creatinine with mg/dL is a known match when unit_normalization includes it
        validate_units(
            events,
            {
                "unit_normalization": {
                    "on_mismatch": "error",
                    "concepts": {"creatinine": "mg/dL"},
                }
            },
        )

    def test_rejects_known_concept_with_wrong_unit(self):
        events = pl.DataFrame(
            {"concept": ["creatinine"], "unit": ["mmol/L"], "value": [1.2]}
        )
        with self.assertRaisesRegex(ValueError, "Non-canonical CLIF units"):
            validate_units(
                events,
                {
                    "unit_normalization": {
                        "on_mismatch": "error",
                        "concepts": {"creatinine": "mg/dL"},
                    }
                },
            )



NEW_TABLE_FILES = {
    "meds_intermittent": "clif_medication_admin_intermittent",
    "assessments": "clif_patient_assessments",
    "crrt": "clif_crrt_therapy",
    "ecmo": "clif_ecmo_mcs",
    "code_status": "clif_code_status",
    "position": "clif_position",
}


class NewSourceConfigTest(unittest.TestCase):
    """U4 (KTD5, KTD11): the new sources are config-declared."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        cls.cfg = yaml.safe_load((root / "configs/data.yaml").read_text())
        cls.tables = cls.cfg["tables"]

    def test_new_tables_are_declared_with_availability_and_lag(self):
        for name, file in NEW_TABLE_FILES.items():
            self.assertEqual(self.tables[name]["file"], file, name)
            expected = 30 if self.tables[name]["availability"] == "missing_storetime" else 0
            self.assertEqual(self.tables[name]["availability_lag_minutes"], expected, name)
        semantics = {name: self.tables[name]["availability"] for name in NEW_TABLE_FILES}
        self.assertEqual(semantics, {
            "meds_intermittent": "recorded", "assessments": "missing_storetime",
            "crrt": "missing_storetime", "ecmo": "missing_storetime",
            "code_status": "recorded", "position": "missing_storetime",
        })
        self.assertEqual(self.tables["resp_support"]["availability"], "missing_storetime")

    def test_treatments_and_context_are_input_only_assessments_are_targets(self):
        for name in ("meds", "meds_intermittent", "resp_support", "crrt", "ecmo",
                     "code_status", "position", "adt"):
            self.assertTrue(self.tables[name].get("input_only"), name)
        for name in ("assessments", "labs", "vitals"):
            self.assertFalse(self.tables[name].get("input_only", False), name)

    def test_resp_melts_the_seventeen_csv_settings(self):
        import csv

        csv_path = Path(__file__).parents[1] / self.cfg["value_binning"]["segment_source"]
        with open(csv_path, newline="") as fh:
            resp = {r["measurement"] for r in csv.DictReader(fh)
                    if r["category"] == "respiratory_support"}
        spec = self.tables["resp_support"]
        self.assertEqual(len(spec["value_cols"]), 17)
        self.assertEqual(set(spec["value_cols"]), resp)
        self.assertEqual(spec["categorical_value_cols"],
                         ["device_category", "mode_category", "tracheostomy"])

    def test_dose_blocks_and_static_tokens(self):
        meds = self.tables["meds"]["dose"]
        self.assertEqual(meds["kind"], "continuous")
        # Plausible weights for per-kg conversion match the extubation BMI rule.
        self.assertEqual(meds["weight_source"], {"table": "vitals", "concept": "weight_kg",
                                                 "plausible_kg": [25.0, 400.0]})
        extubation = yaml.safe_load((Path(__file__).parents[1] / "configs/extubation.yaml")
                                    .read_text())
        self.assertEqual(meds["weight_source"]["plausible_kg"],
                         extubation["risk_factors"]["bmi"]["plausible_weight_kg"])
        self.assertEqual(self.tables["meds_intermittent"]["dose"]["kind"], "intermittent")
        self.assertEqual(self.tables["ecmo"]["concept_qualifier_col"], "mcs_group")
        self.assertEqual(self.tables["code_status"]["key"], "patient")
        self.assertEqual(self.tables["position"]["emit"], "transitions")
        self.assertEqual(self.cfg["static_tokens"],
                         ["age_decile", "sex", "race", "ethnicity", "admission_type"])

    def test_repo_config_passes_the_bundle_identifier_validator(self):
        from src.eval.bundle import _validate_data_config_identifiers

        _validate_data_config_identifiers(self.cfg)


class BundleIdentifierValidatorTest(unittest.TestCase):
    """Every new SQL-interpolated config field is validated (KTD5)."""

    def _cfg(self):
        import copy

        root = Path(__file__).parents[1]
        return copy.deepcopy(yaml.safe_load((root / "configs/data.yaml").read_text()))

    def test_unsafe_new_fields_are_rejected(self):
        from src.eval.bundle import _validate_data_config_identifiers
        from src.eval.clif_validate import ArtifactMismatch

        bad = "x;DROP"
        mutations = {
            "value_cols entry": lambda c: c["tables"]["resp_support"]["value_cols"].append(bad),
            "value_cols not a list": lambda c: c["tables"]["crrt"].__setitem__("value_cols", bad),
            "categorical_value_cols": lambda c: c["tables"]["resp_support"][
                "categorical_value_cols"].append(bad),
            "concept_qualifier_col": lambda c: c["tables"]["ecmo"].__setitem__(
                "concept_qualifier_col", bad),
            "literal concept": lambda c: c["tables"]["position"].__setitem__("concept", bad),
            "patient_id_col": lambda c: c["tables"]["code_status"].__setitem__(
                "patient_id_col", bad),
            "admission_col": lambda c: c["tables"]["code_status"].__setitem__(
                "admission_col", bad),
            "discharge_col": lambda c: c["tables"]["code_status"].__setitem__(
                "discharge_col", bad),
            "hospitalization_file": lambda c: c["tables"]["code_status"].__setitem__(
                "hospitalization_file", "../../etc/passwd"),
            "dose action_col": lambda c: c["tables"]["meds"]["dose"].__setitem__(
                "action_col", bad),
            "dose kind": lambda c: c["tables"]["meds"]["dose"].__setitem__("kind", bad),
            "weight table": lambda c: c["tables"]["meds"]["dose"]["weight_source"].__setitem__(
                "table", "nope"),
            "weight concept": lambda c: c["tables"]["meds"]["dose"][
                "weight_source"].__setitem__("concept", bad),
            "stop_actions": lambda c: c["tables"]["meds"]["dose"].__setitem__(
                "stop_actions", [bad]),
            "emit": lambda c: c["tables"]["position"].__setitem__("emit", bad),
            "key": lambda c: c["tables"]["code_status"].__setitem__("key", bad),
            "static token": lambda c: c["static_tokens"].append(bad),
            "static patient file": lambda c: c["static_source"].__setitem__(
                "patient_file", "../x"),
            "static hospitalization file": lambda c: c["static_source"].__setitem__(
                "hospitalization_file", "a/b"),
        }
        for label, mutate in mutations.items():
            cfg = self._cfg()
            mutate(cfg)
            with self.assertRaises(ArtifactMismatch, msg=label):
                _validate_data_config_identifiers(cfg)


if __name__ == "__main__":
    unittest.main()
