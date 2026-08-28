# CLIFATRON 2.0 — Complete Handoff Prompt (2026-08-27)

Use the CE DataScience plugin for all work. Clone and enter the repo, then `uv sync`.

## Environment
- Repo: `sajor2000/clifatron2.0` (private, SSH: `git@github.com:sajor2000/clifatron2.0.git`)
- Branch: `fix/step1-config-code-reconciliation` (15 commits, Steps 1-6)
- Dev: Mac Studio (M4 Max, 64GB, MPS) — smoke-test only
- Training: 2× L40 Linux box (48GB each, no NVLink, bf16, DDP via torchrun)
- Data: `~/Data/clif-source/` (CLIF 2.1 parquet — Rush + MIMIC, 546K stays, 111M events)
- Package: `uv` only — never pip into shared interpreter

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

## Current Status
- Working tree: clean, pushed
- Tests: 35 pass on MPS (M4 Max, 64GB)
- PR pending: `gh auth login -h github.com` needed on Mac Studio (code: 444F-B135 at https://github.com/login/device)