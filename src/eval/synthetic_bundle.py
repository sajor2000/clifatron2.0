"""Fully synthetic site + bundle builder for qualification runs (U9).

One implementation, used three ways: the repo's tests, the wheel's tests, and — because
it is vendored into `clif-validate` with everything else — a site-side self-test that
exercises the entire pipeline (tokenize → zero-shot inference → export ceremony)
without touching a single governed row.

Everything here is generated from a seed. **No real data, no real model.** The
backbone is a randomly initialised one-layer GPT-2 small enough to run on any CPU; the
"patients" are arithmetic. The point is not clinical realism but CONTRACT realism:
the bundle this module produces passes every check `load_bundle` makes, the site data
passes every check `tokenize_site` and `auto_label` make, and the export it yields is
schema-valid — so a green synthetic run proves the machinery, and only the machinery.

The cohort contract and artifact policy are embedded as module constants rather than
copied from `configs/` — a wheel has no repository checkout to copy from (the same
dangling-path rule `bundle.py` exists to enforce). Their content mirrors the repo
policy's structure exactly; `configs/artifact_policy.yaml` remains the production
authority and nothing here is read outside synthetic builds.

CWD contract: the artifact policy classifies intermediate shards under the RELATIVE
directory `output/intermediate_phi`, so callers must chdir into a scratch workdir
before building (tests do; the CLI self-test will). That is deliberate — the
destination check firing against the caller's working directory is the production
behaviour, not a fixture quirk.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

# The one outcome the synthetic bundle evaluates, and the zero-shot query the
# fixture manifest declares for it. Per ThresholdHazardHead: `tau_bin` is the
# queried threshold's VALUE-bin index (0..n_value_bins), not a time bin, and
# `direction` is 0=below, 1=above. MAP < 65 mmHg → the forced 65.0 edge's bin,
# direction below.
SYNTHETIC_OUTCOME = "map_below_65_48h"
SYNTHETIC_OUTCOME_QUERIES = {
    SYNTHETIC_OUTCOME: {"target_index": 0, "tau_bin": 1, "direction": 0},
}
SYNTHETIC_SITE = "SYNTH-A"

FIXTURE_COHORT = {
    "contract_version": "1.0.0",
    "clif_version": "2.1.0",
    "mcide_version": "2.1.0",
    "outcomes": {
        SYNTHETIC_OUTCOME: {
            "source": "vitals",
            "concept": "map",
            "direction": "below",
            "threshold": 65.0,
            "unit": "mmHg",
            "minimum_post_anchor_measurements": 1,
            "baseline_lookback_hours": 6,
            "required_measurement_within_hours_of_horizon": 12,
        },
    },
}

# Structure mirrors configs/artifact_policy.yaml (which tests/test_artifact_policy.py
# pins); only the classes and manifests the synthetic pipeline exercises are carried.
FIXTURE_POLICY = {
    "policy_version": "1.0.0-synthetic",
    "default_deny": True,
    "classes": {
        "patient_level_phi": {
            "directory": "output/intermediate_phi",
            "formats": ["parquet"],
            "export_allowed": False,
        },
        "aggregate_no_phi": {
            "directory": "output/final_no_phi",
            "formats": ["json", "csv", "pdf", "png"],
            "export_allowed": True,
            "disclosure_review_required": True,
            "minimum_cell_size": 10,
        },
        "operational_logs": {
            "directory": "output/intermediate_phi",
            "formats": ["jsonl"],
            "export_allowed": False,
            "prohibited_content": [
                "identifiers", "patient_rows", "local_source_paths", "free_text",
            ],
        },
    },
    "compatibility_manifests": {
        "experimental_representation": {
            "required_hashes": [
                "training_split", "vocabulary", "numeric_edges",
                "target_map", "outcome_spec", "clif_version",
            ],
            "vocabulary_origin": "reference_training_partition",
        },
    },
}

FIXTURE_DATA_CONFIG = {
    "schema_version": "2.1.0",
    "mcide_version": "2.1.0",
    # Rewritten to bundled absolute paths by load_bundle; the literal values in the
    # written data_config.yaml are the bundle-relative names, for the human reader.
    "cohort_contract": "cohort.yaml",
    "artifact_policy": "artifact_policy.yaml",
    "tables": {
        "vitals": {
            "file": "clif_vitals",
            "availability_col": "recorded_dttm",
            "concept_col": "vital_category",
            "value_col": "vital_value",
            "unit_col": "vital_unit",
        },
    },
    "target_concepts": [
        {"name": "map", "source": "vitals", "direction": "below", "unit": "mmHg"},
    ],
    "value_binning": {
        "scheme": "decile",
        "n_bins": 4,
        "build_from_site": SYNTHETIC_SITE,
        "fit_partition": "train",
        "soft_discretization": True,
        "soft_kernel_bins": 1,
        "forced_edges": {"map": [65.0]},
    },
    "unit_normalization": {"on_mismatch": "error", "concepts": {}},
}


def build_synthetic_site(site_dir: str | Path, *, n_stays: int = 24,
                         seed: int = 7) -> Path:
    """Write a synthetic single-hospital CLIF site + episode artifact.

    Every stay is ICU-admitted at t0, anchored at hour 24, with MAP measured through
    hour 70 — so each satisfies the outcome's baseline-lookback, post-anchor, and
    near-horizon measurement requirements. Even-indexed stays dip below the 65 mmHg
    threshold after the anchor (positive outcome); odd-indexed stays never do. With
    the default 24 stays that yields 12/12, clearing the minimum cell size of 10 on
    both sides so the export has something evaluable to release.
    """
    import numpy as np
    import polars as pl

    from src.data.cohort import build_cohort
    from src.data.splits import content_manifest

    rng = np.random.default_rng(seed)
    base = Path(site_dir)
    base.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    ids = [f"synth-{i:03d}" for i in range(n_stays)]
    admits = [t0 + timedelta(days=i) for i in range(n_stays)]
    pl.DataFrame({
        "hospitalization_id": ids,
        "patient_id": [f"synth-p-{i:03d}" for i in range(n_stays)],
        "hospitalization_joined_id": ids,
        "admission_dttm": admits,
        "discharge_dttm": [a + timedelta(days=5) for a in admits],
        "age_at_admission": [40 + (i % 40) for i in range(n_stays)],
        "discharge_category": ["Home"] * n_stays,
        "hospital_id": [SYNTHETIC_SITE] * n_stays,
    }).write_parquet(base / "clif_hospitalization.parquet")

    pl.DataFrame({
        "hospitalization_id": ids,
        "in_dttm": admits,
        "out_dttm": [a + timedelta(days=4) for a in admits],
        "location_category": ["icu"] * n_stays,
        "hospital_id": [SYNTHETIC_SITE] * n_stays,  # single-hospital guard reads this
    }).write_parquet(base / "clif_adt.parquet")

    v_ids, v_dttm, v_cat, v_val = [], [], [], []
    for i, (stay, admit) in enumerate(zip(ids, admits)):
        positive = i % 2 == 0
        for hour in range(1, 71, 2):
            base_map = 78.0 + float(rng.normal(0.0, 4.0))
            if positive and 26 <= hour <= 40:
                value = 55.0 + float(rng.normal(0.0, 2.0))  # post-anchor crossing
            else:
                value = max(base_map, 70.0)                  # never below threshold
            v_ids.append(stay)
            v_dttm.append(admit + timedelta(hours=hour))
            v_cat.append("map")
            v_val.append(round(value, 1))
    pl.DataFrame({
        "hospitalization_id": v_ids,
        "recorded_dttm": v_dttm,
        "vital_category": v_cat,
        "vital_value": v_val,
        "vital_unit": ["mmHg"] * len(v_ids),  # outcome derivation requires canonical units
    }).write_parquet(base / "clif_vitals.parquet")

    hospitalization = pl.read_parquet(base / "clif_hospitalization.parquet")
    adt = pl.read_parquet(base / "clif_adt.parquet")
    episodes = build_cohort(hospitalization, adt, {
        "anchor_hours": 24,
        "prediction_horizon_hours": 48,
        "minimum_age": 18,
        "icu_location_category": "icu",
    }).with_columns(pl.lit("train").alias("partition"))

    split_hash = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "partition"]
    )["sha256"]
    episode_hash = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
    )["sha256"]
    episodes = episodes.with_columns(
        pl.lit(FIXTURE_COHORT["contract_version"]).alias("cohort_contract_version"),
        pl.lit(split_hash).alias("split_sha256"),
        pl.lit(episode_hash).alias("episode_sha256"),
        pl.lit("{}").alias("source_provenance_json"),
    )
    episode_path = base / "episodes.parquet"
    episodes.write_parquet(episode_path)
    return episode_path


def build_synthetic_bundle(bundle_dir: str | Path, site_dir: str | Path,
                           episode_path: str | Path, *, seed: int = 7) -> Path:
    """Assemble a complete, manifest-sealed bundle from a synthetic site.

    Build order matters and mirrors a real release: configs first, then the
    vocabulary (built by the same `tokenize_site` path a reference site would run),
    then the model, and the manifest LAST so its file map covers everything.

    Requires the CWD contract in the module docstring: the vocabulary build writes
    its intermediate shard under `output/intermediate_phi/` relative to the CWD,
    exactly as the artifact policy classifies it.
    """
    import polars as pl
    import torch
    import yaml
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.data.tokenize import tokenize_site
    from src.eval.bundle import write_bundle_manifest
    from src.model.head_adapter import CLIFATRONHeads, load_backbone

    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)

    (root / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT, sort_keys=True))
    (root / "artifact_policy.yaml").write_text(
        yaml.safe_dump(FIXTURE_POLICY, sort_keys=True)
    )
    (root / "data_config.yaml").write_text(
        yaml.safe_dump(FIXTURE_DATA_CONFIG, sort_keys=True)
    )

    # Build-time config: absolute paths, because tokenize_site resolves the cohort
    # contract as ROOT / cfg["cohort_contract"] and this build must not depend on a
    # repo checkout any more than a site run may.
    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str((root / "cohort.yaml").resolve())
    cfg["artifact_policy"] = str((root / "artifact_policy.yaml").resolve())

    episodes = pl.read_parquet(episode_path)
    vocab_out = Path("output/intermediate_phi/synthetic_vocab_build")
    tokenize_site(cfg, SYNTHETIC_SITE, Path(site_dir), vocab_out, None, None,
                  episodes=episodes, artifact_policy=FIXTURE_POLICY)
    blob = json.loads((vocab_out / "vocab.json").read_text())
    (root / "vocab.json").write_text(json.dumps(blob))

    torch.manual_seed(seed)
    vocab_size = max(blob["vocab"].values()) + 1
    backbone_cfg = GPT2Config(
        vocab_size=max(vocab_size, 32), n_positions=128,
        n_embd=16, n_layer=1, n_head=2,
        bos_token_id=0, eos_token_id=0,  # inside the tiny vocab; GPT-2's 50256 is not
    )
    GPT2LMHeadModel(backbone_cfg).save_pretrained(root)

    # head_weights.pt holds the FULL CLIFATRONHeads state dict (backbone included):
    # load_checkpoint loads it with strict=True over the whole module.
    torch.manual_seed(seed)
    model = CLIFATRONHeads(load_backbone(str(root)), 10, freeze_backbone=True)
    torch.save(model.state_dict(), root / "head_weights.pt")

    hashes = blob["manifest"]["hashes"]
    write_bundle_manifest(
        root,
        model_bundle_id="synthetic-fixture",
        model_version="0.0-synthetic",
        vocab_hash=hashes["vocabulary"],
        outcome_spec_hash=hashes["outcome_spec"],
        clif_version="2.1",
        outcome_queries=SYNTHETIC_OUTCOME_QUERIES,
    )
    return root


__all__ = [
    "FIXTURE_COHORT",
    "FIXTURE_DATA_CONFIG",
    "FIXTURE_POLICY",
    "SYNTHETIC_OUTCOME",
    "SYNTHETIC_OUTCOME_QUERIES",
    "SYNTHETIC_SITE",
    "build_synthetic_bundle",
    "build_synthetic_site",
]
