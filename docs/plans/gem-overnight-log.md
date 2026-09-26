# GEM overnight log — local validation track (this Mac)

Working loop state file. The overnight loop (hourly) reconstructs progress from this
log + `git log` + `output/intermediate_phi/` contents. Append a dated entry per
completed unit. Plan of record: `2026-09-25-001-feat-icu-gem-generative-model-plan.md`.

Goal: prove the whole GEM stack — ETL → packed shards → MPS smoke train → generate →
viewer — against the real staged MIMIC CLIF tables (`~/Data/clif-source`, 16 parquet
files) on THIS Mac, so the code is proven before it moves to the L40 box.

## Checklist

- [ ] U1. ETL dry-run on staged tables (schema/config mismatches fixed)
- [ ] U2. Cohort artifact (`src.data.cohort` → `output/intermediate_phi/episodes.parquet`)
- [ ] U3. Full tokenize, first site builds vocab (stats → log)
- [ ] U4. Packed shards (8192-row spec) + loader smoke batch on MPS
- [ ] U5. MPS smoke train: short pure-NTP run (fp32, small scale) → checkpoint
- [ ] U6. Smoke generate via `src.model.generate` CLI → sims parquet → viewer-verified
- [ ] Suites green after each unit; code-only commits pushed

## Entries

_No entries yet — first firing starts here. Reconstructed state at loop creation: G0
committed (ec77527 viewer commit e8b7eba + import cleanup 2060803), both suites green
(407 + 32), no artifacts under output/ yet._
