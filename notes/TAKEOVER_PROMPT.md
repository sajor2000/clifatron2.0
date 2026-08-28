# CLIFATRON 2.0 — Complete Handoff Prompt (2026-08-27)

Use the CE DataScience plugin for all work. Clone and enter the repo, then `uv sync`.

## Environment
- Repo: `sajor2000/clifatron2.0` (private, SSH: `git@github.com:sajor2000/clifatron2.0.git`)
- Branch: `main` (20 commits; Steps 1–6 merged + data-onboarding fixes)
- Dev: Mac Studio (M4 Max, 64GB, MPS) — smoke-test only
- Training: 2× L40 Linux box `rudu-hpcg004` (48GB each, no NVLink, bf16, DDP via torchrun)
- Data (as staged on L40 box, 2026-08-27): `~/Data/clif-source/CLIF_MIMIC/` — **MIMIC-IV-Ext-CLIF 2.1
  ONLY** (16 tables, **546,028 stays, ~134M events**). Rush + UChicago are dev-cohort sites but are
  **NOT staged on this box yet**. Override the data dir with `CLIF_DATA_DIR=<path>`.
- Package: `uv` only — never pip into shared interpreter
- **Known box issue:** `nvidia-smi` fails with a driver/library version mismatch (NVML 580.173 vs
  kernel 580.159) — torch CUDA still allocates, but reboot before long multi-GPU runs.

## What's Built (Steps 1-6)
- Untied next-event projection with tie_weights flag
- CLIF 2.1 availability timestamps + unit normalization
- Soft Gaussian-weighted discretization + forced clinical edges
- CR D-calibration + AJ K-cal metrics (arXiv:2602.00194)
- NTP→TTE curriculum scheduler
- 4-arm finetune-vs-scratch ablation framework
- 5-arm tokenization ablation framework
- Federation validation package (clif_validate, clif_auto_labeler, clif_forest_plot)
- Notes modality encoder (frozen BioClinical ModernBERT)
- Selective prediction with per-outcome deferral confidence
- 35 tests pass (unit + MPS + federation smoke)

## What to Run Next (on L40 box)
```
# Finetune-vs-scratch
torchrun --nproc_per_node=2 -m src.train.run_arm --arm from_scratch --data <narratives>
torchrun --nproc_per_node=2 -m src.train.run_arm --arm frozen_backbone_head_only --checkpoint <ckpt> --data <narratives>

# Tokenization ablation
for arm in clifatron_clinical_bins global_deciles deciles_plus_soft continuous_fused textcode; do
    torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation --arm $arm --data <events.parquet>
done

# Federation validation
python -m src.eval.clif_validate --checkpoint <ckpt> --data /path/to/clif_parquet --site-name "SiteName"
python -m src.eval.clif_forest_plot --results results/SiteA.json results/SiteB.json

# Compare
python -m src.eval.ablation_compare --results results/ablation
```

## Hard Rules
1. Treatments are model inputs, never prediction targets
2. Vocab = frozen CLIF mCIDE, applied identically to all sites
3. Pre-anchor notes only (leakage rule)
4. No data leaves its node — external validation returns aggregate-only

## Current Status (2026-08-27, L40 box)
- Working tree: clean, pushed to `main`.
- Tests: **42 pass** on the L40 box with real MIMIC data staged (`CLIF_DATA_DIR=~/Data/clif-source/CLIF_MIMIC`);
  4 skip cleanly when no CLIF data is present. Env set up via `uv sync` (torch 2.13.0+cu130, CUDA available).
- Data-onboarding fixes landed: tokenizer sample-limit (`limit_stays`) so smoke tests don't grind the full
  546k-stay dataset; uniform soft-bin width; metrics-panel hardened vs non-finite logits.
- **Blockers for real training:** (1) no CLIFATRON checkpoint staged yet (needed for Method-3 wedge);
  (2) Rush + UChicago data not on this box; (3) value-regression head loss is unnormalized/huge — needs
  per-concept value scaling before Step-3 joint pretraining.