"""U5 (R14, KTD7): the tokenizer-v2 artifact contract and the hard compatibility break.

`vocab.json` v2 carries the segments, per-concept binning sources, reference units (dose
target units included, hashed), the precedence policy and `tokenizer_version: 2`; shards,
value statistics and checkpoints are bound to its vocabulary and segments hashes. Every
artifact the previous tokenizer built is refused with a re-tokenize message.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch

from src.data.cohort import QualificationError


TINY_MCFG = {
    "trunk": {"d_model": 8, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0,
              "rope_base": 10000.0, "tied_embeddings": False, "target_vocab": 64},
    "heads": {
        "competing_risk": {"n_time_bins": 4},
        "threshold_hazard": {"n_time_bins": 4, "threshold_embed_dim": 4},
        "value_regression": {"enabled": True},
    },
}


def _shift_segments(blob: dict) -> dict:
    """A copy of `blob` whose `map` segments differ (same vocabulary, same bin count)."""
    other = copy.deepcopy(blob)
    seg = other["segments"]["map"]
    inner = next(i for i, s in enumerate(seg) if s["hi"] is not None and s["lo"] is not None)
    seg[inner]["hi"] += 1e-3
    seg[inner + 1]["lo"] += 1e-3
    return other


class ArtifactV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml

        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import (
            FIXTURE_COHORT,
            FIXTURE_DATA_CONFIG,
            FIXTURE_POLICY,
            SYNTHETIC_SITE,
            build_synthetic_site,
        )

        cls._td = tempfile.TemporaryDirectory()
        cls.work = Path(cls._td.name)
        old = os.getcwd()
        os.chdir(cls.work)
        try:
            site = cls.work / "site"
            episodes = pl.read_parquet(build_synthetic_site(site))
            (cls.work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
            (cls.work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
            cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
            cfg["cohort_contract"] = str((cls.work / "cohort.yaml").resolve())
            cfg["artifact_policy"] = str((cls.work / "artifact_policy.yaml").resolve())
            out = Path("output/intermediate_phi/v2_build")
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, episodes=episodes,
                          artifact_policy=FIXTURE_POLICY)
            cls.out = (cls.work / out).resolve()
            cls.blob = json.loads((cls.out / "vocab.json").read_text())
            cls.events = pl.read_parquet(cls.out / "events.parquet")
        finally:
            os.chdir(old)
        cls.cfg, cls.policy = cfg, FIXTURE_POLICY

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    # --- vocab.json v2 ---------------------------------------------------------------
    def test_v2_vocab_records_version_segments_sources_units_and_policy(self):
        from src.data.segments import POLICY_VERSION, TOKENIZER_VERSION, json_sha256

        blob = self.blob
        self.assertEqual(TOKENIZER_VERSION, 2)
        self.assertEqual(blob["manifest"]["tokenizer_version"], 2)
        self.assertNotIn("edges", blob)
        self.assertEqual(set(blob["binning_sources"]), set(blob["segments"]))
        self.assertEqual(blob["precedence_policy"], POLICY_VERSION)
        units = blob["reference_units"]
        self.assertEqual(units["concepts"]["map"], "mmHg")
        self.assertEqual(set(units["concepts"]), set(blob["segments"]))
        self.assertIn("dose_targets", units)
        hashes = blob["manifest"]["hashes"]
        self.assertEqual(hashes["numeric_edges"], json_sha256(blob["segments"]))
        self.assertEqual(hashes["reference_units"], json_sha256(units))
        self.assertEqual(hashes["binning_sources"], json_sha256(blob["binning_sources"]))
        self.assertEqual(hashes["concept_sources"], json_sha256(blob["concept_sources"]))
        self.assertEqual(blob["concept_sources"]["tables"]["map"], ["vitals"])

    def test_v2_vocab_round_trips_through_validation(self):
        from src.data.tokenize import validate_vocabulary_artifact

        vocab, segments, manifest = validate_vocabulary_artifact(
            copy.deepcopy(self.blob), self.cfg, self.policy)
        self.assertEqual(segments, self.blob["segments"])

    def test_a_v1_vocab_is_refused_with_a_retokenize_message(self):
        from src.data.tokenize import validate_vocabulary_artifact

        v1 = copy.deepcopy(self.blob)
        v1["edges"] = v1.pop("segments")
        v1["manifest"].pop("tokenizer_version")
        with self.assertRaisesRegex(QualificationError, "re-tokeni[sz]e"):
            validate_vocabulary_artifact(v1, self.cfg, self.policy)
        old = copy.deepcopy(self.blob)
        old["manifest"]["tokenizer_version"] = 1
        with self.assertRaisesRegex(QualificationError, "re-tokeni[sz]e"):
            validate_vocabulary_artifact(old, self.cfg, self.policy)

    def test_tampered_reference_units_or_dose_targets_fail_their_hash(self):
        from src.data.tokenize import validate_vocabulary_artifact

        tampered = copy.deepcopy(self.blob)
        tampered["reference_units"]["dose_targets"]["norepinephrine"] = "mg/kg/hr"
        with self.assertRaisesRegex(QualificationError, "reference-units hash mismatch"):
            validate_vocabulary_artifact(tampered, self.cfg, self.policy)

    def test_a_wrong_precedence_policy_is_refused(self):
        from src.data.tokenize import validate_vocabulary_artifact

        bad = copy.deepcopy(self.blob)
        bad["precedence_policy"] = 99
        with self.assertRaisesRegex(QualificationError, "precedence policy"):
            validate_vocabulary_artifact(bad, self.cfg, self.policy)

    def test_a_policy_v1_vocabulary_is_refused_with_a_retokenize_message(self):
        """Policy v2 changed dose segments (step 8), so a v1-built vocabulary's dose bins
        disagree with this build: refused, never reused."""
        from src.data.tokenize import validate_vocabulary_artifact

        old = copy.deepcopy(self.blob)
        old["precedence_policy"] = 1
        old["manifest"]["provenance"]["precedence_policy"] = 1
        with self.assertRaisesRegex(QualificationError, "precedence policy 1.*re-tokeni[sz]e"):
            validate_vocabulary_artifact(old, self.cfg, self.policy)

    def test_n_value_bins_is_max_segments_plus_one(self):
        from src.data.segments import n_value_bins

        expected = max(len(s) for s in self.blob["segments"].values()) + 1
        self.assertEqual(n_value_bins(self.blob), expected)
        self.assertGreaterEqual(len(self.blob["segments"]["map"]), 12)
        with self.assertRaisesRegex(ValueError, "re-tokeni[sz]e"):
            n_value_bins({"vocab": {}, "edges": {"map": [65.0]}})

    # --- shards ------------------------------------------------------------------------
    def test_every_shard_record_carries_the_segments_binding(self):
        from src.data.segments import artifact_binding

        binding = artifact_binding(self.blob)
        self.assertEqual(binding["tokenizer_version"], "2")
        for hashes in self.events["artifact_hashes"].to_list():
            self.assertEqual(hashes, binding)

    def _dataset(self, records, expected):
        from src.data.dataset import ModelDataset
        from src.data.targets import TargetBuilder

        return ModelDataset(records, representation="decile",
                            target_builder=TargetBuilder(64, 4, 48, {}),
                            expected_hashes=expected)

    def test_model_dataset_accepts_v2_and_rejects_v1_and_mismatched_shards(self):
        from src.data.segments import artifact_binding
        from src.data.targets import TargetContractError
        from src.train.pretrain import _load_decile_records

        binding = artifact_binding(self.blob)
        records = _load_decile_records(self.out / "events.parquet",
                                       drop_values_without_stats=True)
        self._dataset(records, binding)

        v1 = copy.deepcopy(records)
        for record in v1:
            record["artifact_hashes"].pop("numeric_edges")
        with self.assertRaisesRegex(TargetContractError, "re-tokeni[sz]e"):
            self._dataset(v1, {})

        other = artifact_binding(_shift_segments(self.blob))
        with self.assertRaisesRegex(TargetContractError, "numeric_edges"):
            self._dataset(records, other)

    # --- value statistics ---------------------------------------------------------------
    def test_value_stats_for_different_segments_with_the_same_vocab_are_rejected(self):
        from src.data.segments import segments_hash
        from src.data.value_stats import load_value_stats, vocab_hash, write_value_stats

        path = write_value_stats({4: (1.0, 2.0)}, self.work / "vs.json",
                                 vocab=self.blob["vocab"], segments=self.blob["segments"])
        same_vocab = vocab_hash(self.blob["vocab"])
        load_value_stats(path, expected_vocab_hash=same_vocab,
                         expected_segments_hash=segments_hash(self.blob["segments"]))
        other = _shift_segments(self.blob)["segments"]
        with self.assertRaisesRegex(ValueError, "segments hash mismatch"):
            load_value_stats(path, expected_vocab_hash=same_vocab,
                             expected_segments_hash=segments_hash(other))
        unbound = write_value_stats({4: (1.0, 2.0)}, self.work / "vs_v1.json",
                                    vocab=self.blob["vocab"])
        with self.assertRaisesRegex(ValueError, "segments"):
            load_value_stats(unbound, expected_vocab_hash=same_vocab,
                             expected_segments_hash=segments_hash(self.blob["segments"]))

    # --- checkpoints ----------------------------------------------------------------------
    def _checkpoint(self, blob, name):
        from src.data.segments import artifact_binding, n_value_bins
        from src.train.checkpoint import save_checkpoint
        from src.train.manifest import Manifest
        from src.train.pretrain import Model

        model = Model(64, 1, TINY_MCFG, n_value_bins=n_value_bins(blob))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        path = self.work / name
        save_checkpoint(path, model=model, optimizer=opt,
                        scheduler=torch.optim.lr_scheduler.StepLR(opt, 10), epoch=0,
                        manifest=Manifest("t", {}, seed=0, ckpt_dir=str(self.work)),
                        vocab_binding=artifact_binding(blob))
        return path

    def _configs(self):
        import yaml

        mpath, dpath = self.work / "model.yaml", self.work / "data.yaml"
        mpath.write_text(yaml.safe_dump(TINY_MCFG))
        dpath.write_text(yaml.safe_dump(self.cfg))
        return mpath, dpath

    def test_generate_loads_a_bound_checkpoint_and_refuses_other_segments(self):
        from src.model.generate import load_generation_model

        ckpt = self._checkpoint(self.blob, "gen_ok.pt")
        mpath, dpath = self._configs()
        load_generation_model(ckpt, mpath, dpath, self.blob)
        with self.assertRaisesRegex(ValueError, "numeric_edges"):
            load_generation_model(ckpt, mpath, dpath, _shift_segments(self.blob))

    def test_generate_refuses_an_unbound_legacy_checkpoint(self):
        from src.model.generate import load_generation_model
        from src.train.checkpoint import load_checkpoint

        ckpt = self._checkpoint(self.blob, "gen_legacy.pt")
        legacy = load_checkpoint(ckpt)
        legacy.pop("vocab_binding")
        torch.save(legacy, ckpt)
        mpath, dpath = self._configs()
        with self.assertRaisesRegex(ValueError, "re-tokeni[sz]e"):
            load_generation_model(ckpt, mpath, dpath, self.blob)

    def test_viewer_refuses_a_checkpoint_bound_to_other_segments(self):
        from src.viewer.sequence_viewer import load_vocab_segments

        ckpt = self._checkpoint(self.blob, "viewer.pt")
        vocab_path = self.work / "vocab_other.json"
        vocab_path.write_text(json.dumps(_shift_segments(self.blob)))
        with self.assertRaisesRegex(ValueError, "numeric_edges"):
            load_vocab_segments(vocab_path, checkpoint=ckpt)
        ok_path = self.work / "vocab_ok.json"
        ok_path.write_text(json.dumps(self.blob))
        self.assertEqual(load_vocab_segments(ok_path, checkpoint=ckpt),
                         self.blob["segments"])

    def test_training_resume_refuses_a_checkpoint_bound_to_other_segments(self):
        from src.data.segments import artifact_binding
        from src.train.engine import TrainConfig, train

        ckpt = self._checkpoint(self.blob, "resume.pt")

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return {"input_ids": torch.ones(2, dtype=torch.long)}

        from src.data.segments import n_value_bins
        from src.train.pretrain import Model

        model = Model(64, 1, TINY_MCFG, n_value_bins=n_value_bins(self.blob))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        cfg = TrainConfig({}, {
            "batch": {"per_gpu": 1, "grad_accum": 1},
            "runtime": {"ckpt_dir": str(self.work / "resume_ckpts"), "ckpt_every": 99},
            "schedule": {"warmup_steps": 1, "total_steps": 1},
            "optimizer": {"grad_clip": 1.0},
        }, {"compile": False}, total_steps=1)
        with self.assertRaisesRegex(ValueError, "numeric_edges"):
            train(model, torch.utils.data.DataLoader(_DS()), None, opt,
                  torch.optim.lr_scheduler.StepLR(opt, 10), cfg, torch.device("cpu"),
                  resume_ckpt=ckpt,
                  vocab_binding=artifact_binding(_shift_segments(self.blob)))


if __name__ == "__main__":
    unittest.main()


class UnitSpellingEquivalenceTest(unittest.TestCase):
    """Re-importing the reference site's own vocab must not fail on unit spellings
    (U8 real-data finding): 'mm Hg' == 'mmHg', and a dose concept's reference unit
    recorded in suffix form ('mg_hr') == the charted 'mg/hour'."""

    def _events(self, concept, unit):
        import polars as pl
        return pl.DataFrame({"concept": [concept], "unit": [unit]})

    def test_spacing_and_suffix_forms_are_equivalent(self):
        from src.data.tokenize import validate_units
        cfg = {"unit_normalization": {"on_mismatch": "error", "concepts": {}}}
        ref = {"concepts": {"pco2_arterial": "mm Hg", "nicardipine_mg_hr": "mg_hr"}}
        validate_units(self._events("pco2_arterial", "mmHg"), cfg, ref)
        validate_units(self._events("nicardipine_mg_hr", "mg/hour"), cfg, ref)

    def test_a_real_unit_mismatch_still_fails_closed(self):
        from src.data.tokenize import validate_units
        cfg = {"unit_normalization": {"on_mismatch": "error", "concepts": {}}}
        ref = {"concepts": {"glucose_serum": "mg/dL"}}
        with self.assertRaises(ValueError):
            validate_units(self._events("glucose_serum", "mmol/L"), cfg, ref)
