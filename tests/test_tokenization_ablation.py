"""Tokenization ablation tests (U6; R15, KTD8).

`TokenizationSmokeTest` forward-passes the encoder variants on random tensors.

`AblationArmsEndToEndTest` is the real acceptance test: a synthetic CLIF site is
tokenized with the REAL tokenizer twice (physician clinical segments and population
deciles, both with soft discretization), outcomes are auto-labelled and joined, the
continuous-fused arm is derived from the primary clinical shard, and every arm in
`configs/tokenization_ablation.yaml` (only its data paths overridden) runs 2 optimizer
steps through `build_loaders` -> `Model.forward` -> `engine.train`. The TextCode arm
uses an injected deterministic text encoder, so no network or model download is
needed. All data is synthetic. Runs on CPU in well under 60 s.
"""

import copy
import hashlib
import json
import os
import tempfile
import unittest
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

try:  # pytest puts tests/ on sys.path (rootdir-less test modules)
    from test_tokenize_alignment import _repartition
except ImportError:  # pragma: no cover - run from the repo root as a package
    from tests.test_tokenize_alignment import _repartition

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
B, T = 4, 32
VOCAB = 200
D_MODEL = 64
ROOT = Path(__file__).resolve().parents[1]


class TokenizationSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = torch.device(DEVICE)
        cls.pos = torch.randint(0, 1440, (B, T)).to(cls.dev)
        cls.token = torch.randint(1, VOCAB, (B, T)).to(cls.dev)
        cls.soft_token = torch.randint(1, VOCAB, (B, T, 3)).to(cls.dev)
        cls.soft_weight = torch.rand(B, T, 3).to(cls.dev)
        cls.soft_weight = cls.soft_weight / cls.soft_weight.sum(-1, keepdim=True)
        cls.value = torch.randn(B, T).to(cls.dev)

    def _mini_cfg(self, tied=False):
        return {
            "trunk": {
                "d_model": D_MODEL,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_mult": 2,
                "dropout": 0.0,
                "tied_embeddings": tied,
            }
        }

    def test_01_discrete_hard_tokens(self):
        """Standard CLIFEncoder with hard [B,T] tokens (clinical bins)."""
        from src.model.encoder import CLIFEncoder

        enc = CLIFEncoder(VOCAB, self._mini_cfg()).to(self.dev)
        H = enc(self.token, self.pos)
        logits = enc.lm_logits(H)
        self.assertEqual(H.shape, (B, T, D_MODEL))
        self.assertEqual(logits.shape, (B, T, VOCAB))
        print(f"  discrete hard: H={list(H.shape)} logits={list(logits.shape)}")

    def test_02_discrete_soft_tokens(self):
        """CLIFEncoder with soft [B,T,K] weighted tokens (deciles+soft)."""
        from src.model.encoder import CLIFEncoder

        enc = CLIFEncoder(VOCAB, self._mini_cfg()).to(self.dev)
        H = enc(self.soft_token, self.pos, self.soft_weight)
        self.assertEqual(H.shape, (B, T, D_MODEL))
        print(f"  discrete soft: H={list(H.shape)}")

    def test_03_continuous_fused_forward(self):
        """ContinuousFusedEncoder with concept tokens + value channel."""
        from src.model.encoder_continuous import ContinuousFusedEncoder

        enc = ContinuousFusedEncoder(VOCAB, self._mini_cfg()).to(self.dev)
        H = enc(self.token, self.pos, continuous_value=self.value)
        logits = enc.lm_logits(H)
        self.assertEqual(H.shape, (B, T, D_MODEL))
        self.assertEqual(logits.shape, (B, T, VOCAB))

        grad = torch.sum(H)
        grad.backward()
        grads = sum(p.grad is not None and p.grad.abs().sum() > 0
                    for p in enc.parameters())
        self.assertGreater(grads, 0)
        print(f"  continuous_fused: H={list(H.shape)} grads={grads}")

    def test_03b_continuous_fused_nan_value_is_masked(self):
        """A NaN (categorical) value never reaches the trunk as NaN."""
        from src.model.encoder_continuous import ContinuousFusedEncoder

        enc = ContinuousFusedEncoder(VOCAB, self._mini_cfg()).to(self.dev)
        value = self.value.clone()
        value[:, ::3] = float("nan")
        H = enc(self.token, self.pos, continuous_value=value)
        self.assertTrue(bool(torch.isfinite(H).all()))

    def test_04_textcode_embedding_shape(self):
        """TextCode returns correct [B,T,token_dim] shape."""
        from src.data.tokenize_textcode import textcode_embedding

        model_dim = 128
        projection = torch.nn.Linear(model_dim, D_MODEL, bias=False)
        cached = np.random.randn(VOCAB, model_dim).astype(np.float32)

        tokens = torch.randint(0, VOCAB, (2, 16))
        output = textcode_embedding(tokens, cached, projection)
        self.assertEqual(output.shape, (2, 16, D_MODEL))
        print(f"  textcode: {list(output.shape)}")

    def test_05_every_configured_arm_builds(self):
        """Each arm in tokenization_ablation.yaml builds its model (TextCode with an
        injected embedding table: it has no model without one)."""
        import yaml
        from src.train.run_tokenization_ablation import TokenizationAblationModel

        abl = yaml.safe_load((ROOT / "configs/tokenization_ablation.yaml").read_text())
        mcfg = {
            "trunk": {
                "d_model": 32, "n_layers": 1, "n_heads": 2,
                "ffn_mult": 2, "dropout": 0.0, "tied_embeddings": False,
            },
            "heads": {
                "next_event": {"enabled": True, "weight": 0.2},
                "competing_risk": {"enabled": True, "weight": 1.0, "n_time_bins": 8},
                "threshold_hazard": {
                    "enabled": True, "weight": 1.0,
                    "horizon_hours": 48, "n_time_bins": 24,
                    "threshold_embed_dim": 16,
                },
                "value_regression": {"enabled": True, "weight": 0.5},
            },
        }

        table = np.random.default_rng(0).normal(size=(50, 12)).astype(np.float32)
        for arm_name, arm in abl["arms"].items():
            text_table = table if arm["tokenizer"] == "textcode" else None
            model = TokenizationAblationModel(50, 5, mcfg, arm, n_value_bins=12,
                                              text_table=text_table)
            total = sum(p.numel() for p in model.parameters())
            print(f"  {arm_name}: {total:,} params, tokenizer={arm['tokenizer']}")
            self.assertGreater(total, 0)

    def test_06_textcode_arm_without_a_table_fails_closed(self):
        from src.train.run_tokenization_ablation import TokenizationAblationModel

        mcfg = {
            "trunk": {"d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2,
                      "dropout": 0.0, "tied_embeddings": False},
            "heads": {"competing_risk": {"n_time_bins": 4},
                      "threshold_hazard": {"n_time_bins": 4, "threshold_embed_dim": 4},
                      "value_regression": {"enabled": True}},
        }
        with self.assertRaisesRegex(ValueError, "embedding table"):
            TokenizationAblationModel(50, 2, mcfg, {"tokenizer": "textcode"},
                                      n_value_bins=4)


# ------------------------------------------------------------------------- fixture

VALIDATION = [f"synth-{i:03d}" for i in range(20, 24)]
CSV = ROOT / ("external/clifatron/tokenETL/config/"
              "critical_illness_tokenization_final_with_intervals.csv")
LACTATE = (1.0, 2.1, 3.0, 4.5, 6.5, 1.4)

TINY_MCFG = {
    "trunk": {"d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2,
              "dropout": 0.0, "rope_base": 10000.0, "tied_embeddings": False,
              "target_vocab": 256},
    "heads": {
        "next_event": {"enabled": True, "weight": 0.2},
        "competing_risk": {"enabled": True, "weight": 1.0, "n_time_bins": 4,
                           "horizon_hours": 48},
        "threshold_hazard": {"enabled": True, "weight": 1.0, "horizon_hours": 48,
                             "n_time_bins": 4, "threshold_embed_dim": 4},
        "value_regression": {"enabled": True, "weight": 0.5},
    },
}


def _tiny_tcfg(ckpt_dir: Path) -> dict:
    return {
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.95],
                      "grad_clip": 1.0},
        # warmup_steps null: warmup_frac scales it (1 update of 2; none for 1-update runs).
        "schedule": {"warmup_steps": None, "total_steps": 2, "cosine_decay": True},
        "batch": {"per_gpu": 4, "grad_accum": 1},
        "runtime": {"num_workers": 0, "log_every": 1000, "ckpt_every": 1000,
                    "ckpt_dir": str(ckpt_dir)},
        "eval_schedule": {"val_every": 1000},
    }


def fake_text_encoder(texts):
    """Deterministic stand-in for the frozen clinical text encoder (no network)."""
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rows.append(np.random.default_rng(seed).normal(size=12))
    return np.asarray(rows, dtype=np.float32)


def _site_config(work: Path, scheme: str) -> dict:
    from src.eval.synthetic_bundle import FIXTURE_DATA_CONFIG

    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
    cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())
    cfg["tables"]["labs"] = {
        "file": "clif_labs", "availability_col": "lab_result_dttm",
        "availability": "result", "concept_col": "lab_category",
        "value_col": "lab_value_numeric", "unit_col": "reference_unit",
    }
    cfg["tables"]["assessments"] = {
        "file": "clif_patient_assessments", "availability_col": "recorded_dttm",
        "availability": "missing_storetime", "concept_col": "assessment_category",
        "value_col": "numerical_value", "categorical_value_col": "categorical_value",
    }
    cfg["target_concepts"] = [
        {"name": "map", "source": "vitals", "direction": "below", "unit": "mmHg"},
        {"name": "lactate", "source": "labs", "direction": "above", "unit": "mmol/L"},
    ]
    cfg["value_binning"]["forced_edges"] = {"map": [65.0], "lactate": [2.0, 4.0]}
    cfg["value_binning"]["coverage"] = "all"
    if scheme == "clinical_segment":
        cfg["value_binning"].update({"scheme": "clinical_segment",
                                     "segment_source": str(CSV)})
    else:
        cfg["value_binning"].update({"scheme": "decile", "n_bins": 16})
    return cfg


def _build_fixture(work: Path) -> dict:
    """Synthetic site -> clinical + decile shards (real tokenizer, soft on), joined
    outcomes, frozen value stats, and the derived continuous-fused arm."""
    import polars as pl
    import yaml

    from src.data.outcome_join import join_outcomes
    from src.data.tokenize import tokenize_site
    from src.data.tokenize_continuous import write_continuous_fused_arm
    from src.data.value_stats import compute_value_stats_from_events, write_value_stats
    from src.eval.clif_auto_labeler import auto_label
    from src.eval.synthetic_bundle import (
        FIXTURE_COHORT,
        FIXTURE_POLICY,
        SYNTHETIC_SITE,
        build_synthetic_site,
    )
    site = work / "site"
    episodes = _repartition(pl.read_parquet(build_synthetic_site(site)), VALIDATION)
    episode_path = site / "episodes_split.parquet"
    episodes.write_parquet(episode_path)
    (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
    (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))

    labs = {"hospitalization_id": [], "lab_result_dttm": [], "lab_category": [],
            "lab_value_numeric": [], "reference_unit": []}
    assess = {"hospitalization_id": [], "recorded_dttm": [], "assessment_category": [],
              "numerical_value": [], "categorical_value": []}
    for i, ep in enumerate(episodes.iter_rows(named=True)):
        stay, admit = ep["hospitalization_id"], ep["icu_admit_dttm"]
        for k, hour in enumerate((4, 12, 20)):
            labs["hospitalization_id"].append(stay)
            labs["lab_result_dttm"].append(admit + timedelta(hours=hour))
            labs["lab_category"].append("lactate")
            labs["lab_value_numeric"].append(LACTATE[(i + k) % len(LACTATE)])
            labs["reference_unit"].append("mmol/L")
        for k, hour in enumerate((3, 9, 15, 21)):
            when = admit + timedelta(hours=hour)
            assess["hospitalization_id"] += [stay, stay]
            assess["recorded_dttm"] += [when, when]
            assess["assessment_category"] += ["braden_mobility", "cam_total"]
            assess["numerical_value"] += [float(1 + (i + k) % 4), None]
            # cam_total is a categorical finding with NO numeric value (NaN value).
            assess["categorical_value"] += [None, "Positive" if (i + k) % 2 else "Negative"]
    utc = pl.Datetime("us", "UTC")
    pl.DataFrame(labs, schema={
        "hospitalization_id": pl.String, "lab_result_dttm": utc, "lab_category": pl.String,
        "lab_value_numeric": pl.Float64, "reference_unit": pl.String,
    }).write_parquet(site / "clif_labs.parquet")
    pl.DataFrame(assess, schema={
        "hospitalization_id": pl.String, "recorded_dttm": utc,
        "assessment_category": pl.String, "numerical_value": pl.Float64,
        "categorical_value": pl.String,
    }).write_parquet(site / "clif_patient_assessments.parquet")

    clinical_cfg = _site_config(work, "clinical_segment")
    data_cfg_path = work / "data_config.yaml"
    data_cfg_path.write_text(yaml.safe_dump(clinical_cfg))
    labels = auto_label(str(site), episode_path, ["map_below_65_48h"],
                        cohort_config=work / "cohort.yaml", data_config=data_cfg_path)

    dirs = {}
    for name, scheme in (("clinical", "clinical_segment"), ("decile", "decile")):
        cfg = _site_config(work, scheme)
        out = Path(f"output/intermediate_phi/{name}")
        tokenize_site(cfg, SYNTHETIC_SITE, site, out, None,
                      episodes=episodes, artifact_policy=FIXTURE_POLICY)
        blob = json.loads((out / "vocab.json").read_text())
        joined = join_outcomes(labels, pl.read_parquet(out / "events.parquet"), blob,
                               cfg, FIXTURE_COHORT)
        joined.write_parquet(out / "events_with_outcomes.parquet")
        dirs[name] = out

    cont = Path("output/intermediate_phi/continuous")
    write_continuous_fused_arm(dirs["clinical"] / "vocab.json",
                               dirs["clinical"] / "events_with_outcomes.parquet", cont,
                               policy=FIXTURE_POLICY)
    dirs["continuous"] = cont

    for out in dirs.values():
        blob = json.loads((out / "vocab.json").read_text())
        stats = compute_value_stats_from_events(out / "events_with_outcomes.parquet")
        write_value_stats(stats, out / "value_stats.json", vocab=blob["vocab"],
                          segments=blob["segments"], fit_partition_name="train")
    return {name: out.resolve() for name, out in dirs.items()}


def _fixture_dir(dirs: dict, arm: dict) -> Path:
    if arm["tokenizer"] == "continuous_fused":
        return dirs["continuous"]
    return dirs["decile"] if arm["scheme"] == "decile_ablation" else dirs["clinical"]


class AblationArmsEndToEndTest(unittest.TestCase):
    ARMS = ("clinical_soft", "clinical_hard", "global_deciles", "deciles_plus_soft",
            "continuous_fused", "textcode")

    @classmethod
    def setUpClass(cls):
        import yaml

        from src.train.engine import _prepare_batch
        from src.train.run_tokenization_ablation import (
            masked_target_counts,
            resolve_arm,
            setup_arm,
            train_arm,
        )

        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            cls.dirs = _build_fixture(work)
            cls.abl = yaml.safe_load((ROOT / "configs/tokenization_ablation.yaml").read_text())
            cls.cpu = torch.device("cpu")
            cls.arm_cfgs, cls.runs, cls.batches, cls.counts = {}, {}, {}, {}
            cls.first_losses, cls.ledgers = {}, {}
            for name in cls.abl["arms"]:
                d = _fixture_dir(cls.dirs, cls.abl["arms"][name])
                arm = resolve_arm(cls.abl, name, events=d / "events_with_outcomes.parquet",
                                  vocab=d / "vocab.json", value_stats=d / "value_stats.json",
                                  primary_vocab=cls.dirs["clinical"] / "vocab.json")
                tcfg = _tiny_tcfg(work / "ckpt" / name)
                run = setup_arm(arm, mcfg=TINY_MCFG, tcfg=tcfg, n_targets=2,
                                device=cls.cpu, text_encoder=fake_text_encoder, seed=0)
                batch = _prepare_batch(next(iter(run.loaders.train)), cls.cpu)
                with torch.no_grad():
                    run.model.eval()
                    cls.first_losses[name] = run.model(batch)
                    run.model.train()
                _, manifest = train_arm(run, tcfg=tcfg, mcfg=TINY_MCFG, device=cls.cpu,
                                        total_steps=2, lr=float(arm["lr"]))
                cls.arm_cfgs[name], cls.runs[name] = arm, run
                cls.batches[name] = batch
                cls.counts[name] = masked_target_counts(batch)
                cls.ledgers[name] = manifest.ledger
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _sample(self, arm: str, key: str):
        ds = self.runs[arm].loaders.train_dataset
        for index, record in enumerate(ds.records):
            if record["episode_key"] == key:
                return ds[index]
        raise AssertionError(f"{key} not in {arm}")

    # ------------------------------------------------------------------ config

    def test_config_has_the_six_arms_mapped_to_data_yaml_schemes(self):
        import yaml

        self.assertEqual(tuple(self.abl["arms"]), self.ARMS)
        schemes = {"clinical_segment", "decile_ablation"}
        for name, arm in self.abl["arms"].items():
            self.assertIn(arm["scheme"], schemes, name)
            self.assertIn(arm["tokenizer"], ("fused", "continuous_fused", "textcode"))
            for key in ("events", "vocab", "value_stats"):
                self.assertTrue(arm.get(key), f"{name} needs its own {key} path")
        self.assertEqual(self.abl["arms"]["clinical_soft"]["scheme"], "clinical_segment")
        self.assertTrue(self.abl["arms"]["clinical_soft"]["soft_discretization"])
        self.assertTrue(self.abl["arms"]["clinical_soft"].get("primary"))
        self.assertEqual(self.abl["arms"]["textcode"]["textcode_encoder"],
                         "thomas-sounack/BioClinical-ModernBERT-base")
        cohort = yaml.safe_load((ROOT / "configs/cohort.yaml").read_text())
        for outcome in (self.abl["shared"]["outcomes"]
                        + self.abl["shared"].get("zero_shot_outcomes", [])):
            self.assertIn(outcome, cohort["outcomes"])
        for section in ("tail_sensitivity",):
            for outcome in self.abl["report"][section]["outcomes"]:
                self.assertIn(outcome, cohort["outcomes"])

    # --------------------------------------------------------- end-to-end arms

    def test_each_arm_runs_two_steps_with_finite_loss_and_masked_targets(self):
        self.assertEqual(set(self.ledgers), set(self.ARMS))
        for name in self.ARMS:
            with self.subTest(arm=name):
                self.assertEqual(self.ledgers[name]["optimizer_updates"], 2)
                self.assertGreater(self.ledgers[name]["ntp_eligible_tokens"], 0)
                counts = self.counts[name]
                self.assertGreater(counts["ntp"], 0)
                self.assertGreater(counts["value"], 0)
                self.assertGreater(counts["th"], 0)
                self.assertGreater(counts["cr"], 0)
                for key, loss in self.first_losses[name].items():
                    self.assertTrue(bool(torch.isfinite(loss)), f"{name} {key}")
                self.assertGreater(float(self.first_losses[name]["th"]), 0.0)
                self.assertGreater(float(self.first_losses[name]["val"]), 0.0)

    def test_arm_inputs_differ_on_the_same_fixture(self):
        key = self.runs["clinical_soft"].loaders.train_dataset.records[0]["episode_key"]
        soft = self._sample("clinical_soft", key)
        hard = self._sample("clinical_hard", key)
        # Hard vs soft tensors: same hard ids, but only the soft arm carries [T, K].
        self.assertEqual(soft["input_ids"], hard["input_ids"])
        self.assertIsNotNone(soft["soft_token"])
        self.assertEqual(len(soft["soft_token"][0]), 3)
        self.assertIsNone(hard["soft_token"])
        self.assertIn("soft_token", self.batches["clinical_soft"])
        self.assertNotIn("soft_token", self.batches["clinical_hard"])
        self.assertEqual(self.batches["clinical_soft"]["soft_token"].ndim, 3)
        self.assertIn("soft_token", self.batches["deciles_plus_soft"])
        self.assertNotIn("soft_token", self.batches["global_deciles"])

        # Deciles vs clinical: a different vocabulary, so different ids per stay.
        dec = self._sample("global_deciles", key)
        self.assertNotEqual(self.runs["global_deciles"].binding["vocabulary"],
                            self.runs["clinical_hard"].binding["vocabulary"])
        self.assertNotEqual(dec["input_ids"], hard["input_ids"])

        # Continuous-fused: edgeless concept ids plus a normalized current-value channel.
        cont = self._sample("continuous_fused", key)
        self.assertEqual(len(cont["input_ids"]), len(hard["input_ids"]))
        self.assertNotEqual(cont["input_ids"], hard["input_ids"])
        cvocab = self.runs["continuous_fused"].vocab_blob["vocab"]
        self.assertIn("map", cvocab)
        self.assertFalse(any(t.startswith("map=") for t in cvocab))
        self.assertTrue(any(cont["input_value_mask"]))
        self.assertIn("input_value", self.batches["continuous_fused"])
        for arm in ("clinical_hard", "global_deciles", "textcode"):
            self.assertNotIn("input_value", self.batches[arm])

        # TextCode: the encoder's input table is the frozen text embedding of each
        # fused id's generated description, projected by a trainable layer.
        from src.data.tokenize_textcode import vocab_descriptions

        enc = self.runs["textcode"].model.enc
        blob = self.runs["textcode"].vocab_blob
        descriptions = vocab_descriptions(blob)
        ids = sorted(i for i in descriptions if i != blob["vocab"]["<pad>"])
        expected = fake_text_encoder([descriptions[i] for i in ids])
        self.assertTrue(np.allclose(enc.text_table[ids].numpy(), expected, atol=1e-6))
        self.assertEqual(float(enc.text_table[blob["vocab"]["<pad>"]].abs().sum()), 0.0)
        self.assertNotIn("text_table", dict(enc.named_parameters()))
        self.assertTrue(enc.text_proj.weight.requires_grad)
        self.assertFalse(hasattr(enc, "tok_emb"))

    def test_continuous_fused_nan_categorical_event_and_primary_threshold_bins(self):
        from src.data.segments import n_value_bins, threshold_bin

        run = self.runs["continuous_fused"]
        cvocab = run.vocab_blob["vocab"]
        cam = [cvocab["cam_total=negative"], cvocab["cam_total=positive"]]
        saw_nan = False
        for index, record in enumerate(run.loaders.train_dataset.records):
            sample = run.loaders.train_dataset[index]
            for tok, raw, v, m in zip(record["token"], record["value"],
                                      sample["input_value"], sample["input_value_mask"]):
                if tok in cam:
                    self.assertTrue(raw is None or not np.isfinite(raw))
                    self.assertEqual((v, m), (0.0, False))
                    saw_nan = True
            for outcome in record["outcomes"]:
                self.assertGreaterEqual(outcome["threshold_bin"], 0)
        self.assertTrue(saw_nan, "fixture has no categorical (NaN) event")
        losses = self.first_losses["continuous_fused"]
        self.assertTrue(all(bool(torch.isfinite(v)) for v in losses.values()))
        batch = self.batches["continuous_fused"]
        self.assertTrue(bool((batch["th_tau"][batch["th_mask"]] >= 0).all()))

        # threshold_bin and n_value_bins come from the PRIMARY clinical segments.
        primary = json.loads((self.dirs["clinical"] / "vocab.json").read_text())
        self.assertEqual(run.n_value_bins, n_value_bins(primary))
        expected = threshold_bin(65.0, primary["segments"]["map"], "below")
        taus = set(batch["th_tau"][batch["th_mask"]].tolist())
        self.assertEqual(taus, {expected})

    def test_continuous_fused_artifact_requires_the_primary_segments_hash(self):
        from src.data.segments import ArtifactBindingError
        from src.data.tokenize_continuous import primary_segments

        blob = copy.deepcopy(self.runs["continuous_fused"].vocab_blob)
        self.assertIn("primary_segments", blob["manifest"]["hashes"])
        primary_segments(blob)  # verified
        blob["primary_segments"]["map"] = blob["primary_segments"]["map"][:-1]
        with self.assertRaisesRegex(ArtifactBindingError, "primary"):
            primary_segments(blob)
        del blob["manifest"]["hashes"]["primary_segments"]
        with self.assertRaisesRegex(ArtifactBindingError, "primary"):
            primary_segments(blob)

    def test_normalize_value_uses_frozen_stats_and_masks_missing(self):
        from src.data.tokenize_continuous import normalize_value

        stats = {7: (2.0, 0.5)}
        self.assertEqual(normalize_value(3.0, 7, stats), (2.0, True))
        self.assertEqual(normalize_value(None, 7, stats), (0.0, False))
        self.assertEqual(normalize_value(float("nan"), 7, stats), (0.0, False))
        self.assertEqual(normalize_value(float("nan"), 9, stats), (0.0, False))
        # An implausible sentinel (|z| > 20) is masked like the value-head target.
        self.assertEqual(normalize_value(999999.0, 7, stats), (0.0, False))
        with self.assertRaisesRegex(ValueError, "normalization statistics"):
            normalize_value(1.0, 9, stats)

    def test_textcode_description_for_lactate_bin_has_concept_interval_unit(self):
        from src.data.segments import interval_label
        from src.data.tokenize_textcode import code_description

        blob = self.runs["textcode"].vocab_blob
        desc = code_description("lactate=6", blob)
        self.assertIn("lactate", desc)
        self.assertIn(interval_label(blob["segments"]["lactate"][6]), desc)
        self.assertIn("(2, 2.2]", desc)
        self.assertIn("mmol/L", desc)
        self.assertIn("labs", desc)
        self.assertIn("negative", code_description("cam_total=negative", blob))
        self.assertIn("icu", code_description("icu", blob))

    def test_freeze_trunk_without_init_checkpoint_warns_and_keeps_trunk_trainable(self):
        from src.train.run_tokenization_ablation import setup_arm

        arm = dict(self.arm_cfgs["clinical_hard"], freeze_trunk=True)
        # catch_warnings, not assertWarns: assertWarns walks sys.modules and trips
        # transformers' lazy submodules when another test has imported transformers.
        with tempfile.TemporaryDirectory() as td, warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            run = setup_arm(arm, mcfg=TINY_MCFG, tcfg=_tiny_tcfg(Path(td)),
                            n_targets=2, device=self.cpu, seed=0)
        self.assertTrue(any(issubclass(w.category, UserWarning)
                            and "init checkpoint" in str(w.message) for w in seen))
        self.assertTrue(all(p.requires_grad for p in run.model.parameters()))

    def test_freeze_trunk_with_init_checkpoint_freezes_only_the_trunk(self):
        from src.train.checkpoint import save_checkpoint
        from src.train.run_tokenization_ablation import setup_arm

        base = self.runs["clinical_hard"]
        opt = torch.optim.SGD(base.model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        arm = dict(self.arm_cfgs["clinical_hard"], freeze_trunk=True)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "init.pt"
            save_checkpoint(ckpt, model=base.model, optimizer=opt, scheduler=sched,
                            epoch=0, vocab_binding=base.binding)
            run = setup_arm(arm, mcfg=TINY_MCFG, tcfg=_tiny_tcfg(Path(td)), n_targets=2,
                            device=self.cpu, seed=1, init_checkpoint=ckpt)
            for (name, p), (_, q) in zip(run.model.state_dict().items(),
                                         base.model.state_dict().items()):
                self.assertTrue(torch.equal(p, q), name)
            # A checkpoint bound to a different vocabulary is refused.
            other = self.runs["global_deciles"]
            bad = Path(td) / "bad.pt"
            save_checkpoint(bad, model=other.model, optimizer=opt, scheduler=sched,
                            epoch=0, vocab_binding=other.binding)
            with self.assertRaises(ValueError):
                setup_arm(arm, mcfg=TINY_MCFG, tcfg=_tiny_tcfg(Path(td)), n_targets=2,
                          device=self.cpu, seed=1, init_checkpoint=bad)
        self.assertFalse(any(p.requires_grad for p in run.model.enc.blocks.parameters()))
        self.assertTrue(all(p.requires_grad for p in run.model.th.parameters()))
        self.assertTrue(run.model.enc.lm_head.projection.weight.requires_grad)

    def test_seed_sets_initial_weights_batch_order_and_target_sampling(self):
        from src.train.run_tokenization_ablation import setup_arm

        arm = self.arm_cfgs["clinical_soft"]
        tcfg = _tiny_tcfg(Path(self._td.name) / "ckpt_seed")

        def state(seed):
            run = setup_arm(arm, mcfg=TINY_MCFG, tcfg=tcfg, n_targets=2, device=self.cpu,
                            seed=seed)
            weights = {k: v.clone() for k, v in run.model.state_dict().items()}
            order = list(iter(run.loaders.train.sampler))
            return weights, order, run.loaders.train_dataset.target_builder.run_seed

        (w1, o1, r1), (w1b, o1b, r1b), (w2, o2, r2) = state(1), state(1), state(2)
        self.assertTrue(all(torch.equal(w1[k], w1b[k]) for k in w1))
        self.assertEqual(o1, o1b)
        self.assertEqual((r1, r1b, r2), (1, 1, 2))
        self.assertFalse(all(torch.equal(w1[k], w2[k]) for k in w1))
        self.assertNotEqual(o1, o2)

    def test_embedding_rows_are_the_vocabulary_max_id_plus_one(self):
        run = self.runs["clinical_soft"]
        rows = max(run.vocab_blob["vocab"].values()) + 1
        self.assertEqual(run.vocab_size, rows)
        self.assertEqual(run.model.enc.tok_emb.num_embeddings, rows)

    # ------------------------------------- consumers of a fresh training checkpoint

    def _fresh_checkpoint(self):
        """A checkpoint written by `engine.train` (the pretrain loop): max id + 1
        embedding rows, `config.vocab_size` and `config.trunk` in its manifest."""
        if getattr(type(self), "_fresh", None) is None:
            from src.train.run_tokenization_ablation import setup_arm, train_arm

            work = Path(self._td.name)
            tcfg = _tiny_tcfg(work / "ckpt_fresh")
            tcfg["runtime"]["ckpt_every"] = 1
            arm = self.arm_cfgs["clinical_soft"]
            run = setup_arm(arm, mcfg=TINY_MCFG, tcfg=tcfg, n_targets=2, device=self.cpu,
                            seed=3)
            train_arm(run, tcfg=tcfg, mcfg=TINY_MCFG, device=self.cpu, total_steps=1,
                      lr=1e-3)
            path = max((work / "ckpt_fresh").glob("ckpt_*.pt"))
            type(self)._fresh = (path, run.vocab_blob)
        return type(self)._fresh

    def _other_size_mcfg(self):
        """A model config whose trunk differs from the checkpoint's: consumers must
        rebuild the trunk the manifest records, and its embedding rows."""
        mcfg = copy.deepcopy(TINY_MCFG)
        mcfg["trunk"].update({"d_model": 32, "target_vocab": 10000})
        return mcfg

    def test_fresh_checkpoint_records_its_vocabulary_size_and_trunk(self):
        from src.train.checkpoint import load_checkpoint

        path, blob = self._fresh_checkpoint()
        config = load_checkpoint(path)["manifest"]["config"]
        self.assertEqual(config["vocab_size"], max(blob["vocab"].values()) + 1)
        self.assertEqual(config["trunk"]["d_model"], TINY_MCFG["trunk"]["d_model"])

    def test_fresh_checkpoint_loads_in_threshold_eval(self):
        from src.eval.threshold_eval import load_model

        path, blob = self._fresh_checkpoint()
        model = load_model(path, blob, self._other_size_mcfg(), n_targets=2)
        self.assertEqual(model.enc.tok_emb.num_embeddings, max(blob["vocab"].values()) + 1)

    def test_fresh_checkpoint_loads_in_generate(self):
        import yaml

        from src.model.generate import load_generation_model

        path, blob = self._fresh_checkpoint()
        work = Path(self._td.name)
        mpath, dpath = work / "gen_model.yaml", work / "gen_data.yaml"
        mpath.write_text(yaml.safe_dump(self._other_size_mcfg()))
        dpath.write_text(yaml.safe_dump({"target_concepts": [{"name": "map"},
                                                             {"name": "lactate"}]}))
        enc = load_generation_model(path, mpath, dpath, blob)
        self.assertEqual(enc.tok_emb.num_embeddings, max(blob["vocab"].values()) + 1)
        self.assertEqual(enc.d_model, TINY_MCFG["trunk"]["d_model"])

    def test_fresh_checkpoint_loads_in_checkpoint_sweep(self):
        from src.eval.checkpoint_sweep import build_model

        path, blob = self._fresh_checkpoint()
        model = build_model(path, blob, self._other_size_mcfg(), n_targets=2)
        self.assertEqual(model.enc.tok_emb.num_embeddings, max(blob["vocab"].values()) + 1)

    def test_fresh_checkpoint_loads_in_run_arm(self):
        from src.train.checkpoint import load_checkpoint
        from src.train.run_arm import build_from_scratch

        path, blob = self._fresh_checkpoint()
        model = build_from_scratch(blob, TINY_MCFG, n_targets=2)
        model.load_state_dict(load_checkpoint(path)["model"])

    def test_fresh_checkpoint_initialises_a_tokenization_arm(self):
        from src.train.run_tokenization_ablation import setup_arm

        path, _ = self._fresh_checkpoint()
        tcfg = _tiny_tcfg(Path(self._td.name) / "ckpt_init")
        run = setup_arm(self.arm_cfgs["clinical_soft"], mcfg=TINY_MCFG, tcfg=tcfg,
                        n_targets=2, device=self.cpu, init_checkpoint=path)
        self.assertIsNotNone(run.model)

    def test_value_regression_weight_changes_value_loss_contribution(self):
        from src.train.run_tokenization_ablation import TokenizationAblationModel

        run = self.runs["clinical_hard"]
        batch = self.batches["clinical_hard"]
        totals, vals = {}, {}
        for weight in (0.5, 2.0):
            mcfg = copy.deepcopy(TINY_MCFG)
            mcfg["heads"]["value_regression"]["weight"] = weight
            torch.manual_seed(0)
            model = TokenizationAblationModel(256, 2, mcfg, self.arm_cfgs["clinical_hard"],
                                              n_value_bins=run.n_value_bins).eval()
            self.assertEqual(model.w["value_regression"], weight)
            with torch.no_grad():
                losses = model(batch)
            totals[weight], vals[weight] = float(losses["total"]), float(losses["val"])
        self.assertGreater(vals[0.5], 0.0)
        self.assertAlmostEqual(vals[0.5], vals[2.0], places=5)
        self.assertAlmostEqual(totals[2.0] - totals[0.5], 1.5 * vals[0.5], places=4)


class ArmResumeTest(unittest.TestCase):
    """`train_arm(resume=...)` / `--resume`: a matrix run continues from its checkpoint
    with the same curriculum weights, optimizer groups (one per head) and result as a
    run straight through (4 microbatches per pass, accumulation 1, 8 updates)."""

    TOTAL = 8

    def setUp(self):
        try:  # pytest puts tests/ on sys.path (rootdir-less test modules)
            import test_ddp_multihead as ddp
        except ImportError:  # pragma: no cover
            from tests import test_ddp_multihead as ddp
        self.ddp = ddp
        self.mcfg = ddp.tiny_mcfg(curriculum="ntp_then_tte")
        self._td = tempfile.TemporaryDirectory()
        self.work = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _run(self, *, seed: int = 0):
        from src.train.pretrain import Loaders
        from src.train.run_tokenization_ablation import ArmRun, TokenizationAblationModel

        ddp = self.ddp
        arm = {"name": "clinical_soft", "tokenizer": "fused", "total_steps": self.TOTAL,
               "lr": 1e-2}
        torch.manual_seed(seed)
        model = TokenizationAblationModel(ddp.VOCAB, ddp.N_TARGETS, self.mcfg, arm,
                                          n_value_bins=ddp.N_VALUE_BINS)
        weights = ddp.record_weights(model)
        dl = ddp.microbatch_loader([ddp.synthetic_batch(900 + i) for i in range(4)],
                                   distributed=False)
        loaders = Loaders(train=dl, validation=None, train_dataset=dl.dataset,
                          validation_dataset=None, records=[], data_path=self.work,
                          value_stats={})
        return ArmRun("clinical_soft", arm, model, loaders, {}, ddp.BINDING,
                      ddp.N_VALUE_BINS, seed=seed), weights

    def _tcfg(self):
        tcfg = self.ddp.tiny_tcfg(self.work / "unused", ckpt_every=4)
        tcfg["schedule"]["warmup_steps"] = None      # warmup_frac: scales with the run
        return tcfg

    def _train(self, out_dir, **kwargs):
        from src.train.run_tokenization_ablation import train_arm

        run, weights = self._run(seed=kwargs.pop("seed", 0))
        trained, manifest = train_arm(run, tcfg=self._tcfg(), mcfg=self.mcfg,
                                      device=torch.device("cpu"), out_dir=out_dir, **kwargs)
        return trained, manifest, weights

    @staticmethod
    def _groups(path):
        from src.train.checkpoint import load_checkpoint

        return [{k: v for k, v in g.items() if k != "params"}
                for g in load_checkpoint(path)["optimizer"]["param_groups"]]

    def test_resume_continues_with_the_same_curriculum_and_optimizer_groups(self):
        straight_dir, split_dir = self.work / "straight", self.work / "split"
        straight, _, straight_weights = self._train(straight_dir)
        ckpts = straight_dir / "checkpoints"
        self.assertEqual(sorted(p.name for p in ckpts.glob("*.pt")),
                         ["ckpt_ep1_step4.pt", "ckpt_ep2_step8.pt"])

        # The split run: the first pass, then (as after a crash) its later checkpoints
        # are gone and `--resume latest` picks the update-4 one.
        self._train(split_dir)
        (split_dir / "checkpoints" / "ckpt_ep2_step8.pt").unlink()
        resumed, manifest, resumed_weights = self._train(split_dir, resume="latest", seed=0)
        self.assertEqual(manifest.ledger["optimizer_updates"], self.TOTAL)
        self.assertTrue(manifest.lineage_parent)
        # Curriculum: the resumed updates 4..7 used exactly the straight run's weights.
        self.assertEqual(resumed_weights, straight_weights[4:])
        self.assertNotEqual(len(set(straight_weights)), 1)       # the weights do change
        # Optimizer: same groups (one per head, with their scheduled weight decay).
        final = split_dir / "checkpoints" / "ckpt_ep2_step8.pt"
        groups = self._groups(final)
        self.assertEqual(groups, self._groups(ckpts / "ckpt_ep2_step8.pt"))
        self.assertEqual({g.get("head") for g in groups} - {None},
                         {"competing_risk", "threshold_hazard", "value_regression"})
        for name, value in straight.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)

    def test_resume_refuses_another_seed_or_schedule(self):
        out = self.work / "run"
        self._train(out)
        ckpt = out / "checkpoints" / "ckpt_ep1_step4.pt"
        with self.assertRaisesRegex(ValueError, "different run: seed"):
            self._train(out, resume=ckpt, seed=1)
        with self.assertRaisesRegex(ValueError, "total_steps"):
            self._train(out, resume=ckpt, total_steps=12)
        # --fresh-schedule: a continuation with a new length is allowed.
        _, manifest, _ = self._train(out, resume=ckpt, total_steps=12, fresh_schedule=True)
        self.assertEqual(manifest.ledger["optimizer_updates"], 12)
        with self.assertRaisesRegex(SystemExit, "no checkpoint"):
            self._train(self.work / "empty", resume="latest")

    def test_cli_accepts_resume(self):
        import contextlib
        import io

        from src.train import run_tokenization_ablation as runner

        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            runner.main(["--help"])
        self.assertIn("--resume", out.getvalue())
        self.assertIn("--fresh-schedule", out.getvalue())


if __name__ == "__main__":
    unittest.main()
