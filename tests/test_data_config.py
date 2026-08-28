import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl
import yaml

from src.data.splits import content_manifest
from src.data.tokenize import (
    _read_table,
    restrict_to_observation_window,
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
        vocab = {"<pad>": 0, "map=0": 1}
        edges = {"map": [65.0]}

        def digest(value):
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(payload.encode()).hexdigest()

        manifest = {
            "artifact_family": "experimental_representation",
            "clif_version": cfg["schema_version"],
            "mcide_version": cfg["mcide_version"],
            "hashes": {
                "training_split": "1" * 64,
                "vocabulary": digest(vocab),
                "numeric_edges": digest(edges),
                "target_map": digest(cfg["target_concepts"]),
                "outcome_spec": digest(cohort_cfg["outcomes"]),
                "clif_version": digest(cfg["schema_version"]),
            },
            "provenance": {"source_site": "synthetic-reference", "immutable": True},
        }
        loaded_vocab, loaded_edges, loaded_manifest = validate_vocabulary_artifact(
            {"vocab": vocab, "edges": edges, "manifest": manifest}, cfg, policy
        )
        self.assertEqual(loaded_vocab, vocab)
        self.assertEqual(loaded_edges, edges)
        self.assertEqual(loaded_manifest, manifest)

        tampered = {"vocab": {**vocab, "new": 2}, "edges": edges, "manifest": manifest}
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_vocabulary_artifact(tampered, cfg, policy)

        wrong_family = {"vocab": vocab, "edges": edges, "manifest": {**manifest, "artifact_family": "clifatron_checkpoint"}}
        with self.assertRaisesRegex(ValueError, "family"):
            validate_vocabulary_artifact(wrong_family, cfg, policy)

        bad_target = json.loads(json.dumps(manifest))
        bad_target["hashes"]["target_map"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "target-map"):
            validate_vocabulary_artifact({"vocab": vocab, "edges": edges, "manifest": bad_target}, cfg, policy)

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


if __name__ == "__main__":
    unittest.main()
