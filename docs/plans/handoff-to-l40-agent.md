# Handoff: run CLIFATRON 2.0 on the L40 node

Written 2026-10-03 for the agent (or person) who takes this branch to `rudu-hpcg004`
(2× NVIDIA L40, Linux, Rush's governed environment). Read this, then `AGENTS.md`
("GOAL RE-CONFIRMED"), then `docs/plans/l40-runbook.md`, which holds the exact commands.

## 1. What the project is

- **Paper 1:** one from-scratch, CLIF-native ICU foundation model (about 30M parameters,
  8,192-token context). Two testable claims: threshold-aligned tokenization, and a combined
  time-to-event objective beating next-token training at equal compute.
- **Paper 2:** an extubation-failure risk model built on the frozen trunk: calibrated 7-day
  reintubation-or-death risk at extubation, plus exploratory per-device risk from injecting an
  NIV, HFNC or conventional-oxygen token. Safeguards are a classical comparison and negative
  controls. No device recommendation. The agreement-margin rubric is frozen as optional.
- Data: MIMIC-IV-Ext-CLIF (staged) and Rush University Medical Center (one hospital, no ECMO,
  sparse dialysis; to be staged). UChicago is external validation later, aggregate only.

## 2. State of the branch

- Branch `t3code/extubation-counterfactual-plan`, PR #17 (draft). Everything up to the
  CLIF 2.1 conformance commit is pushed and CI is green on Python 3.11 and 3.13.
- The commit after that holds the data-pipeline and Rush-onboarding work. It passes the
  full local suite (about 1,515 tests) and the `clif-validate` suite (34), and the docs
  build. **It has not been run at full MIMIC scale** (see section 4).
- Code is built for every Milestone 1 unit: tokenizer arms and literature-grounded bins,
  in-stream time-to-event targets, curriculum, multi-site loader, experiment matrix (96 runs),
  claims panel, CLIF task suite and baselines, extubation cohort, labels, estimators, audit
  runner, pre-flight, runbook.
- Not built (Milestone 2/3, needs a trained checkpoint): time-zero prompts, injected-token
  head, doubly robust on frozen state and the known-answer test, the risk model, the UChicago
  runner.

## 3. What was verified, and where

| Check | Where it ran | Result |
|---|---|---|
| Full test suite, `clif-validate` suite, vendor sync, docs build | Mac | pass |
| mCIDE v2.1.1 conformance gate on full MIMIC | Mac | zero failing values |
| Full MIMIC tokenization (earlier build, before the latest data-pipeline work) | Mac | 1–2 min per step, 33–36 GB peak RAM |
| Real-data training smoke on all 6 tokenization arms (5,000-stay sample) | Mac | finite losses |
| Two-process training (gloo, CPU): identical parameters, checkpoint, clean stop | Mac | pass |
| CUDA, NCCL, bf16, SDPA kernel restriction, driver checks | **never run** | first run is the pre-flight |

## 4. Known risks and open problems (read before launching)

1. **Unexplained tokenizer slowdown.** The last local full-MIMIC full-hospitalization
   tokenization had run about 25 minutes (about 176 CPU-minutes, 23 GB) when it was stopped;
   the earlier build of the same step took about 96 seconds. The new work in that step (each
   extubation's index stay added to the tokenized set, site config, conformance gate, memory
   gate) may explain it, or it is a regression. Time runbook step 8 on the node with
   `/usr/bin/time -v`; if it is far slower than a few minutes, profile before going further.
2. **CUDA paths are untested.** Run the pre-flight (runbook step 15) first; it checks GPUs,
   the NCCL wheel and version, the driver against the CUDA 13 build (needs driver 580+), a
   two-GPU smoke, memory, data binding, value-stat coverage for every site, the schedule and
   the context length. Fallback if the driver is older: pin the cu126 index (documented in
   the runbook).
3. **Real-data checks not run after the final changes:** full-MIMIC rebuild peak memory,
   per-cause event rates (RR > 24 vs SpO2 < 88 vs lactate > 4, with and without the sustained
   rule), the context-length report on the rebuilt shards, the edge-distance check on the
   provisional controls (creatinine 1.35, MAP 62.5), and the pre-flight on rebuilt artifacts.
   All are in the runbook; run them on the node.
4. **Every vocabulary, shard and value-stats file must be rebuilt** from this branch. Nothing
   is trained, so nothing is lost.
5. **Extubation cohort is small at MIMIC.** With the two-non-invasive-row primary rule the NIV
   and HFNC arms are a few hundred patients across all partitions. Claim 3 depends on Rush.
6. **Not-yet-resolved items:** the arterial-only MAP outcome filter is wired as an optional
   sensitivity (off); `generate.py` ignores prompt values for the continuous-fused arm
   (rollouts are illustrations only); `run_arm.py` compile/DDP order is documented, not
   changed.

## 4a. Decisions already made by the product authority

All 47 clinical decisions and the Rush decisions are in
`docs/decisions/2026-10-03-clinical-decisions.md`. Notable: temperature cause above 39.166 °C,
creatinine as a KDIGO rule, GCS verbal "not testable" token with Brennan imputation, BP method
token, first device within a 3 h grace window, two non-invasive rows primary, discharge alive
before day 7 counted event-free, stratified held-out split (50% of NIV and HFNC patients),
vocabulary fit on MIMIC only with an mCIDE categorical allowlist, value statistics fit on MIMIC
plus Rush train, no Rush outcome labels before the split freeze, Rush-specific mappings only in
the git-ignored `configs/sites/rush.local.yaml`.

## 5. Decisions still needed from the product authority

- Physician sign-off on every `literature_proposed` bin edge and on the competing-risk
  thresholds still marked proposed.
- The held-out share, then the split freeze (MIMIC and Rush in one record, before any
  checkpoint).
- Control thresholds creatinine 1.35 and MAP 62.5 stay provisional until the production
  vocabularies' edge-distance table confirms them.
- Trial-matching margins (frozen; revive only if a reviewer asks; the approved pooled method is
  in the decision log).
- Staging Rush data under governance, and gate G5 (approval to move Rush-derived weights).
- Data-side fixes outside this repo: WBC and absolute differential mapping in the MIMIC-to-CLIF
  conversion, the source item behind `ultrafiltration_out`, and the MIMIC-IV
  `anchor_year_group` table.

## 6. First hour on the node

```bash
git clone https://github.com/sajor2000/clifatron2.0.git && cd clifatron2.0   # or git pull
git checkout t3code/extubation-counterfactual-plan
uv sync --group dev
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)"
nvidia-smi        # fails today on a driver/library mismatch: reboot first (runbook step 2)
CLIF_DATA_DIR=~/Data/clif-source uv run --with pytest --with pytest-xdist python -m pytest tests/ -q -n auto
uv run --with pytest python -m pytest clif-validate/tests -q
uv run python clif-validate/scripts/sync_vendor.py --check
```

Then follow `docs/plans/l40-runbook.md` in order: stage data (step 3), build the episode artifact
(4), extubation cohort and labels (5), audit blind stage (6), stratified split and freeze (7),
tokenize arms (8–11), refit value stats (12), pre-flight (15), edge check and run specs (16),
timing run (17), screening runs (18), evaluation (19). **Do the "Rush day 1" section before any
Rush tokenization.** Use `uv run torchrun ...` for launches. Stop a run cleanly with
`pkill -TERM -P <torchrun pid>`.

## 7. Rules that must hold

- Patient-level data and row-level artifacts stay on the node, in git-ignored
  `output/intermediate_phi/`. Only aggregates go to `output/final_no_phi/`, with cells under 10
  suppressed. Never paste tracebacks or row values into chat, issues or commits.
- No Rush-specific strings or counts in committed config. Use `configs/sites/rush.local.yaml`.
- No outcome-by-arm comparison before the protocol hash (Casey 2021 on MIMIC is the one
  exception, run deliberately by the product authority:
  `uv run python -m src.eval.extubation_audit unblinded --cohort mimic=... --labels mimic=...`).
- Treatments are inputs, never trunk prediction targets (hard rule 1). Availability ordering,
  not charttime (hard rule 4). One frozen vocabulary for all sites (hard rule 2).
- `uv` only. Never `pip install` into a shared interpreter.

## 8. Where things are

| What | Where |
|---|---|
| Plan | `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md` |
| Runbook, pre-flight, this handoff | `docs/plans/l40-runbook.md`, `src/train/preflight.py`, this file |
| Clinical decisions | `docs/decisions/2026-10-03-clinical-decisions.md` |
| Evidence | `notes/ai-novelty-audit.md`, `notes/extubation-evidence-review.md` |
| Protocol draft | `docs/protocols/extubation-emulation-protocol.md` |
| CLIF 2.1.1 mCIDE snapshot | `configs/clif_mcide_2.1.1/` |
| Literature-grounded bins | `configs/literature_segments/` |
| Experiment matrix | `configs/experiment_matrix.yaml`, `src/train/run_matrix.py` |
| Rush template | `configs/sites/rush.example.yaml` (copy to `rush.local.yaml`) |
| Pooled-margin work (frozen, local only) | `output/patches/pooled-margins-wip.patch` on the Mac, not in git |
