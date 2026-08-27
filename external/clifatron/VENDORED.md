# Vendored: CLIFATRON (upstream)

This directory is a **vendored snapshot** of the CLIF consortium's CLIFATRON, the
foundation model CLIFATRON 2.0 builds on. It is committed into this repo (rather than
used as a submodule or PyPI dependency) so the whole project is self-contained on any
machine and so we can fork/modify tokenETL, the AR trainers, and the benchmark in place.

- **Upstream:** https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON
- **Snapshot commit:** `d3d281825af6d85ffc35b9461e71e8e6a5d034cd` (2025-11-03, "fix: benchmark readme")
- **License:** MIT (upstream `LICENSE` retained in this directory)
- **Vendored on:** 2026-08-27

## What was changed on vendoring
- Removed upstream `.git/` (this tree now lives in our history).
- Removed the nested `.gitignore` so our top-level `.gitignore` governs; the tokenizer
  definition files it would otherwise drop are force-added and tracked here:
  `tokenETL/config/critical_illness_tokenization_final_with_intervals.csv`,
  `AR/*/tokenizer/clinical_tokenizer/vocab.json`.
- No model checkpoints, `.parquet`/`.csv` data, or `.pt`/`.bin` artifacts were vendored
  (upstream shipped none; verified). `clif_config.json` (real wandb key) is git-ignored —
  only `clif_config_template.json` (placeholder) is tracked.

## Re-syncing with upstream
To pull upstream changes later, diff against a fresh clone of the snapshot commit and
apply by hand (this is a fork, not a live mirror):
```bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON /tmp/clifatron-upstream
diff -ru /tmp/clifatron-upstream external/clifatron --exclude=.git --exclude=VENDORED.md
```
