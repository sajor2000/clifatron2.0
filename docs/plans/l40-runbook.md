# L40 runbook: raw CLIF tables to the screening runs

Written 2026-10-03 under `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md`
(unit U19). It replaces `docs/plans/l40-g2-runbook.md`, which predates tokenizer v2 and
the full-hospitalization path and must not be used to launch anything.

Node: `rudu-hpcg004`, 2x L40 (48 GB each, no NVLink), bf16, DDP through `torchrun`.
Run every command from the repository root. Launch torchrun as `uv run torchrun`, never a
bare `torchrun` (that is whichever one is first on PATH, with whatever torch it brings).
`tests/test_runbook_commands.py` checks that every `uv run python -m ...` and
`uv run torchrun ... -m ...` line in this file names an existing module and only flags
that module accepts, and that no line starts a bare `torchrun`, so edit commands here and
in the code together.

This runbook ends at the screening runs and their evaluation. Full-budget runs, and so
every claim, wait for the decisions in "Before the first launch" and for the full budget
to be sized from the timing run (step 17).

## Storage and governance

| Path | Contents | Class |
|---|---|---|
| `~/Data/clif-source/` | MIMIC-IV-Ext-CLIF tables (PhysioNet-credentialed) | governed, never copied off the node |
| `~/Data/clif-rush/` | Rush CLIF tables, once staged under governance | governed, institutional |
| `output/intermediate_phi/` | episode artifacts, shards, vocabularies, value stats, the GEM caches, extubation cohort and labels, run directories (checkpoints, `threshold_eval/scores.parquet`) | patient-level; local, git-ignored, never exported |
| `output/final_no_phi/` | audit exports, `split_freeze.json`, claims report, baselines | aggregate only; leaves the node only after disclosure review |

Run directories under `output/intermediate_phi/runs/` hold row-level scores keyed by
hashed stay identifiers (`threshold_eval/scores.parquet`). The hashes are opaque but the
rows are patient-level: keep them in governed storage. Only the JSON written under
`output/final_no_phi/` is aggregate.

The GEM cache (`<shard dir>/gem_cache/`) is memory-mapped by both ranks and built under
an `fcntl` lock. It must sit on a local filesystem, not NFS or SMB: keep
`output/intermediate_phi/` on the node's local disk. The pre-flight checks this.

## Before the first launch: decisions for the product authority

These must be settled before step 15 (pre-flight) can pass or before the first full run.
The code runs either way; none of these is decided by an engineer. The product authority
(J.C. Rojas) decided items 1-4 on 2026-10-03; the record, with references, is
`docs/decisions/2026-10-03-clinical-decisions.md`.

1. **Held-out share (KTD6) - decided: stratified held-out split (item 46).** The split is
   baked into the vocabulary, every shard and every checkpoint. The product authority chose
   to enlarge the held-out partitions for the scarce arms: patients whose first
   post-extubation device is NIV or HFNC are over-sampled into the held-out partitions
   until each arm holds 50% of its patients there, with the overall 60/15/10/15 kept
   (`configs/train.yaml` `data_contract.held_out_stratification`; step 7 switches it on).
   Changing it after the freeze means rebuilding everything from step 4.
2. **Competing-risk thresholds - decided (items 1-7).** All ten causes are `contract` or
   `confirmed`: respiratory rate > 24, bilirubin > 2.0, platelets < 100, heart rate > 130,
   SBP < 90, temperature > 39.166 C (the physician-CSV edge at NEWS2's top temperature
   band), and creatinine by the KDIGO AKI rule (+0.3 mg/dL in 48 h or 1.5x the lowest
   value in 7 days, in-stay baseline). The pre-flight check passes.
3. **Off-edge controls - provisional (item 8).** Creatinine 1.35 and MAP 62.5 replace 1.3
   and 63. They are provisional: step 16's edge check on the production vocabularies
   decides; if it refuses one, return it to the product authority.
4. **Agreement margins - decided in method, values `proposed` (items 42-45).** Each trial's
   margin is derived by the FDA fixed-margin approach (half of the conservative benchmark
   effect preserved), with the Roehmel-Kieser second hurdle, a harmful-side stop on the
   whole interval above 1, overlap measured on the two compared arms, and registered
   negative-control outcomes. Values stay `proposed` until the protocol is registered;
   they do not gate the screening runs.
5. **Casey 2021 comparison.** The unblinded all-comer comparison against Casey 2021 is the
   one outcome-by-arm run allowed before registration. It is the product authority's
   deliberate run, not part of this runbook.
6. **Claim 1 attribution arm.** Off in `configs/experiment_matrix.yaml`; whether the claim 1
   rule uses it is open.
7. **Checkpoint cadence and warm-up against the run length.** Settled in code: the
   defaults in `configs/train.yaml` now scale with the run (`engine.resolve_schedule`).
   With `per_gpu: 4`, `grad_accum: 32` and two ranks an update is 256 windows, so one
   pass over the MIMIC sample's train stays is 13 updates and the full corpus roughly
   180-300; a fixed `ckpt_every: 2000` / `warmup_steps: 2000` wrote no checkpoint and
   never left warm-up at that length. Now:
   - warm-up is `floor(schedule.warmup_frac x updates)` (0.05; at least one update,
     never the whole run);
   - a checkpoint is written every `max(1, updates // runtime.checkpoints_per_run)`
     updates (5 per run), plus a final checkpoint at the end of every run and on a clean
     stop (SIGTERM / Ctrl-C: the run finishes the current update, saves, and exits);
   - validation runs every `max(1, updates // eval_schedule.validations_per_run)` updates
     (2 per run: mid-run and at the end); under DDP each rank validates its share of the
     validation batches and the ranks combine their sums;
   - an absolute `schedule.warmup_steps` / `runtime.ckpt_every` /
     `eval_schedule.val_every` still wins when set; the
     launcher refuses a warm-up as long as the run, or a run that would write no
     checkpoint at all, and the pre-flight's schedule check (step 15) applies the same
     rules to every matrix budget (it warns when only the final checkpoint would be
     written).

   Left to decide: `grad_accum`. It sets the effective batch (256 windows) and therefore
   the update count: a 0.05-pass screening run on full MIMIC is only about 9-15 updates,
   so its learning-rate schedule is one warm-up update and a short cosine. A smaller
   `grad_accum` gives more, noisier updates per pass and changes the recipe for every
   arm; it is a choice for the product authority, made before the first full run and
   recorded in `configs/train.yaml` (commit, then rerun step 15).

## 1. Update the code and the environment

```bash
git pull
uv sync --group dev
git log -1 --format='%h %s'
```

Produces: the locked environment. Check: `git status` is clean (the pre-flight refuses a
dirty tree), and the commit is the one you mean to train.

## 2. Driver check

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)"
```

Check: `nvidia-smi` lists two L40s and `True 2`. If `nvidia-smi` reports
`Failed to initialize NVML: Driver/library version mismatch` (the known fault on this
node), reboot before anything long runs:

```bash
sudo reboot
```

### CUDA build and driver

`uv.lock` resolves torch from PyPI, which on Linux is the CUDA 13 build
(`torch.version.cuda` prints `13.0`). CUDA 13.x runs on NVIDIA driver branch 580 or newer
(CUDA 13.0 GA needs 580.65.06; minor-version compatibility covers 13.x on any >= 580
driver), and CUDA 12.x on >= 525. The pre-flight (step 15, `gpu: driver vs CUDA build`)
fails with this message when the driver is older. Fixes, in order of preference:

1. upgrade the node's driver to the 580 branch or newer (needs the node administrator, a
   reboot, then step 2 again);
2. otherwise pin torch to the CUDA 12.6 wheels for Linux only. Not applied in the
   repository; to do it, add to `pyproject.toml`, then `uv lock`, `uv sync --group dev`,
   check that `torch.version.cuda` prints `12.6`, and rerun the pre-flight:

   ```toml
   [[tool.uv.index]]
   name = "pytorch-cu126"
   url = "https://download.pytorch.org/whl/cu126"
   explicit = true

   [tool.uv.sources]
   torch = { index = "pytorch-cu126", marker = "sys_platform == 'linux'" }
   ```

   `explicit = true` keeps every other package on PyPI; the marker keeps macOS on the
   default PyPI wheels. The cu126 build pulls its own `nvidia-nccl-cu12`, so the
   `override-dependencies` entry in `pyproject.toml` that drops `nvidia-nccl-cu12` must be
   removed in the same change, and xgboost pinned to a release that also uses the cu12
   NCCL (or the xgboost wheel's `nvidia-nccl-cu13` dropped instead); the pre-flight's
   `gpu: NCCL` check confirms one NCCL wheel.

One NCCL: older xgboost releases (3.2.x, picked under Python 3.11) pull
`nvidia-nccl-cu12` while torch's CUDA 13 build pulls `nvidia-nccl-cu13`; both install
`nvidia/nccl/lib/libnccl.so.2`, so the last one installed wins. `pyproject.toml` drops the
cu12 wheel (`[tool.uv] override-dependencies`; xgboost is used on CPU only). The
pre-flight prints `torch.cuda.nccl.version()` and fails when more than one
`nvidia-nccl-*` wheel is installed.

## 3. Stage the data

MIMIC-IV-Ext-CLIF is already at `~/Data/clif-source` (546,028 stays, about 134M events).

```bash
ls ~/Data/clif-source/*.parquet | wc -l
df -hT ~/Data/clif-source output
```

Check: the CLIF tables are present, and `output/` is on a local filesystem (`ext4`, `xfs`;
not `nfs`, `cifs`).

Rush: stage the Rush CLIF tables at `~/Data/clif-rush` only once the governance approval
is in place, then run "Rush day 1" (below) before any Rush step. Every Rush step below is
marked "(Rush)"; skip it until then. UChicago is external validation by model-to-data and
is never staged here.

## Rush day 1 (aggregate-only checks before any Rush step)

Rush University Medical Center is ONE hospital (one `hospital_id`; the cohort and the
tokenizer refuse more than one). Rush has no ECMO/MCS table, and its CRRT rows often say
only that a patient was on dialysis. Nothing has run on Rush data yet: do these steps in
order, each aggregate-only, before steps 4-13 for Rush.

**Governance (every step).**

- Rush tables and everything derived from rows (episode and extubation artifacts, shards,
  vocabularies built from them, caches) stay under `~/Data/clif-rush/` and
  `output/intermediate_phi/`, on the node.
- Never paste a traceback, a log line or an error message off the node: error messages
  are aggregate (counts under 10 print as `<10`, small-cell quantiles are withheld) but
  can echo Rush's own source strings. Copy off the node only the aggregate JSON written
  under `output/final_no_phi/`, after disclosure review.
- No Rush string (source names, unit labels, local codes) in a committed file: Rush
  mappings live only in the git-ignored `configs/sites/rush.local.yaml`. `git status`
  must not list it.

**1. Stage the files.** One parquet per CLIF table in `~/Data/clif-rush/`, named
`clif_<table>.parquet` (`clif_hospitalization.parquet`, `clif_adt.parquet`,
`clif_vitals.parquet`, ...), identifiers (`patient_id`, `hospitalization_id`,
`hospital_id`) as strings, timestamps in UTC (a naive column is read as Rush's wall clock,
America/Chicago, and converted; a time inside the spring DST gap becomes null and is
counted).

```bash
ls ~/Data/clif-rush/clif_*.parquet | wc -l
```

**2. Site-local declarations.**

```bash
cp configs/sites/rush.example.yaml configs/sites/rush.local.yaml
git status --short configs/sites/
```

Fill the profile (`extraction_dttm`, the data cut, as ISO-8601 UTC), keep
`expected_absent_tables: [ecmo_mcs]`, and fill the BP patterns from check 10 below. The
local file is merged into `sites.rush`, `site_harmonization.rush` and
`site_unit_conversions.rush` at load time and hashed into every Rush shard's binding.
`git status` must print nothing for it (it is ignored).

**3. CLIF 2.1 / mCIDE conformance (report only).**

```bash
uv run python -m src.data.clif_conformance --data ~/Data/clif-rush --site rush --report-only
```

Check: `failure_messages` is empty, or each failing value gets a `category_map` alias or a
`declared_exceptions` entry in `rush.local.yaml`; `tables.ecmo` reads `expected_absent`.

**4. Tokenizer dry run (every table read through Rush's declarations, no window).**

```bash
uv run python -m src.data.tokenize --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush_day1 --vocab output/intermediate_phi/mimic/vocab.json --episodes output/intermediate_phi/episodes_rush.parquet --dry-run
```

Check: events and concepts per table, `ecmo: absent (declared ...)`, no `NOT DECLARED`
table, and the count of timestamps nulled in a DST gap (expected near zero).

**5. Episode artifact for Rush.**

```bash
uv run python -m src.data.cohort --site rush --data ~/Data/clif-rush --out output/intermediate_phi/episodes_rush.parquet
```

Check: the waterfall; `stays_open_censored_at_extraction` is the number of stays still in
hospital at the data cut (0 if `extraction_dttm` is not yet declared, in which case those
stays are `open_stay_no_extraction_time`, never eligible).

**6. Extubation cohort for Rush.**

```bash
uv run python -m src.data.extubation_cohort --data ~/Data/clif-rush --site rush --episodes output/intermediate_phi/episodes_rush.parquet --out output/intermediate_phi/extubation_cohort_rush.parquet
```

No labels: the labeler refuses Rush (confirmatory) before the freeze.

**7. Sample tokenization with the frozen MIMIC vocabulary, and the `<unk>` gate.**

```bash
uv run python -m src.data.tokenize --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush_day1 --vocab output/intermediate_phi/mimic/vocab.json --episodes output/intermediate_phi/episodes_rush.parquet --sample-episodes 2000 --workers 8
uv run python -m src.data.tokenize --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush_day1 --vocab output/intermediate_phi/mimic/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes_rush.parquet --extubation-cohort output/intermediate_phi/extubation_cohort_rush.parquet --sample-episodes 2000 --workers 8
uv run python -m src.data.tokenization_report --report output/intermediate_phi/rush_day1/tokenization_report.json --report output/intermediate_phi/rush_day1/gem_tokenization_report.json
```

Check: `GATE PASS` (`<unk>` under 1% on Rush, which is all non-fit); the report's
`unk.by_concept` names any concept to harmonize; `data_quality.crrt_coverage` gives the
share of CRRT rows with each setting (low shares mean the derived effluent dose is a
MIMIC-mostly concept); `data_quality.site_profile` lists the expected-absent tables.

**8. Context length of Rush histories.**

```bash
uv run python -m src.eval.context_length --site rush --shard output/intermediate_phi/rush_day1/gem_events.parquet --cohort output/intermediate_phi/extubation_cohort_rush.parquet --episodes output/intermediate_phi/episodes_rush.parquet
```

Check: the share of Rush stays and prompts over 8,192 tokens (revisit 16K only if Rush is
much denser than MIMIC); `prompts.missing_from_shard` counts sampled-out index stays here
(the full step-13 build has none).

**9. Blind audit stage (no outcome read).**

```bash
uv run python -m src.eval.extubation_audit blind --cohort rush=output/intermediate_phi/extubation_cohort_rush.parquet
```

**10. On-node checks (aggregate prints only; run them, do not copy their output off the
node).**

Local time stored as UTC: if admissions cluster 5-6 hours off the expected morning and
afternoon peaks of Chicago wall-clock time, Rush wrote local times with a UTC label;
declare it with the data provider before going on (the hour-of-day histogram, counts only):

```bash
uv run python -c "import duckdb; print(duckdb.sql(\"SELECT hour(admission_dttm AT TIME ZONE 'America/Chicago') AS local_hour, count(*) AS n FROM read_parquet('$HOME/Data/clif-rush/clif_hospitalization.parquet') GROUP BY 1 ORDER BY 1\").fetchall())"
```

BP source names, before writing the `bp_method` patterns: only names charted at least 10
times are printed (the number of rarer names is printed as one count); write patterns
from these, map the rest to `unknown`, and keep the names in `rush.local.yaml` only:

```bash
uv run python -c "import duckdb; q = \"SELECT meas_site_name, count(*) AS n FROM read_parquet('$HOME/Data/clif-rush/clif_vitals.parquet') WHERE vital_category IN ('sbp','dbp','map') GROUP BY 1\"; rows = duckdb.sql(q).fetchall(); print([r for r in rows if r[1] >= 10]); print('names under 10 rows:', sum(1 for r in rows if r[1] < 10))"
```

Troponin assay (decided 2026-10-03: no code change). If Rush's `troponin_t` is a
conventional (4th-generation) assay in ng/mL rather than high-sensitivity ng/L, add the
MIMIC-style rename to `rush.local.yaml` (`concept_renames: {labs: {troponin_t:
troponin_t_conventional}}`) before step 13; the unit and median say which:

```bash
uv run python -c "import duckdb; print(duckdb.sql(\"SELECT reference_unit, count(*) AS n, median(lab_value_numeric) AS p50 FROM read_parquet('$HOME/Data/clif-rush/clif_labs.parquet') WHERE lab_category = 'troponin_t' GROUP BY 1 HAVING count(*) >= 10\").fetchall())"
```

Lab result time (hard rule 4: availability, not collection): the share of labs with no
result time, with a result before collection, and the median collection-to-result delay;
a result time equal to the collection time on most rows means Rush stores collection
time there, which must be raised before training:

```bash
uv run python -c "import duckdb; print(duckdb.sql(\"SELECT count(*) AS n, avg((lab_result_dttm IS NULL)::INT) AS no_result, avg((lab_result_dttm < lab_collect_dttm)::INT) AS result_before_collect, avg((lab_result_dttm = lab_collect_dttm)::INT) AS result_equals_collect, median(epoch(lab_result_dttm - lab_collect_dttm) / 60) AS median_minutes FROM read_parquet('$HOME/Data/clif-rush/clif_labs.parquet')\").fetchall())"
```

Calendar period (decided 2026-10-03): optional for Rush; `configs/extubation.yaml`
`sites.rush.calendar_period_source` stays null.

## 4. Build the episode artifact

```bash
uv run python -m src.data.cohort --site mimic --data ~/Data/clif-source --out output/intermediate_phi/episodes.parquet
# (Rush)
uv run python -m src.data.cohort --site rush --data ~/Data/clif-rush --out output/intermediate_phi/episodes_rush.parquet
```

Produces: the patient-grouped episode and split artifact, with partitions from
`configs/train.yaml` `data_contract.partitions` (60/15/10/15 today) and `split_seed`, bound
to its site (a `site` column; every later step refuses an artifact of another site).
Timestamps follow the site profile (`configs/data.yaml` `sites.<site>`): naive local times
are converted to UTC, and a stay with no discharge time is censored at the declared
`extraction_dttm`. Check: the command finishes without a `QualificationError` and prints
the aggregate waterfall (small cells as `<10`).

## 5. Extubation cohort and labels

```bash
uv run python -m src.data.extubation_cohort --data ~/Data/clif-source --site mimic --episodes output/intermediate_phi/episodes.parquet --out output/intermediate_phi/extubation_cohort.parquet
uv run python -m src.eval.extubation_labeler --data ~/Data/clif-source --cohort output/intermediate_phi/extubation_cohort.parquet --out output/intermediate_phi/extubation_cohort_labels.parquet
# (Rush) the cohort only; see the note on Rush labels below
uv run python -m src.data.extubation_cohort --data ~/Data/clif-rush --site rush --episodes output/intermediate_phi/episodes_rush.parquet --out output/intermediate_phi/extubation_cohort_rush.parquet
```

Produces: the extubation cohort (each patient's partition inherited from the episode
artifact) and its outcome labels. Check: the printed waterfall counts are suppressed
aggregates only. The labels are not read by the blind stage. A site other than MIMIC must
pass `--episodes` and `--out` (never the MIMIC defaults), and `--data` must be the
directory its episode artifact was built from (file hashes compared).

Rush labels: none before the freeze (decided 2026-10-03). Rush is `confirmatory` in
`configs/extubation_benchmarks.yaml`, so the labeler refuses it unless the R28 freeze
manifest verifies; after the freeze:

```bash
uv run python -m src.eval.extubation_labeler --data ~/Data/clif-rush --cohort output/intermediate_phi/extubation_cohort_rush.parquet --out output/intermediate_phi/extubation_cohort_rush_labels.parquet --freeze-manifest output/final_no_phi/freeze_manifest.json
```

Anything that reads labels (the unblinded audit) refuses labels whose recorded
`extubation_sha256` is not the cohort's: relabel after every cohort rebuild.

## 6. Audit blind stage

```bash
uv run python -m src.eval.extubation_audit blind --cohort mimic=output/intermediate_phi/extubation_cohort.parquet
```

Produces: `output/final_no_phi/extubation_audit_blind_mimic.json` and `.csv`
(`pending_review`), and a ledger entry. Check: the held-out arm sizes
(`mimic|held_out|eligible|arm=...`) for conventional oxygen, HFNC and NIV. At 60/15/10/15
the MIMIC held-out NIV and HFNC arms are small (see the blind-stage report on the node; arm
counts are not copied into this public repository).

## 7. Stratify the held-out partitions and freeze the split

The product authority chose the stratified held-out split (decision 1; item 46). Switch it
on in `configs/train.yaml` (`data_contract.held_out_stratification.enabled: true`), commit,
then rebuild the episode artifact with the step-5 extubation cohort as the strata source
(only its `patient_id`, `eligible` and `arm_first_device` columns are read; no outcome),
and rerun steps 5 and 6 so the cohort inherits the new partitions:

```bash
uv run python -m src.data.cohort --site mimic --data ~/Data/clif-source --out output/intermediate_phi/episodes.parquet --held-out-strata output/intermediate_phi/extubation_cohort.parquet
# (Rush) the same stratified rule, the same 0.5 share for NIV and HFNC
uv run python -m src.data.cohort --site rush --data ~/Data/clif-rush --out output/intermediate_phi/episodes_rush.parquet --held-out-strata output/intermediate_phi/extubation_cohort_rush.parquet
```

`--held-out-strata` must be the same site's extubation cohort (its `site` column is
checked).

Check: the printed `held_out_stratification.achieved` shows each of NIV and HFNC at about
0.5 (aggregate shares only), and the step-6 blind stage's `held_out` arm sizes grew. The
other patients only rebalance between train and the held-out partitions, so the overall
shares stay 60/15/10/15. Rebuilding with the extubation cohort of the old split is fine:
arms do not depend on the partition. With Rush staged, do the same for
`episodes_rush.parquet` with the Rush extubation cohort (the second line above), and rebuild
the Rush extubation cohort from it. Then freeze MIMIC and Rush together in ONE record:

```bash
uv run python -m src.train.preflight --write-split-freeze --episodes mimic=output/intermediate_phi/episodes.parquet --episodes rush=output/intermediate_phi/episodes_rush.parquet --audit-blind output/final_no_phi/extubation_audit_blind_mimic.json --approver "J.C. Rojas"
```

Before Rush is staged, drop the `--episodes rush=...` argument (and refreeze with it, by
`--force`, before any checkpoint exists). Each `SITE=PATH` must be that site's own
artifact (its `site` column is checked).

Produces: `output/final_no_phi/split_freeze.json` (episode artifact SHA-256, content split
hash, configured and observed partition shares, split seed, held-out arm counts from the
blind export, approver, date). Check: the printed split hash. The freeze is written once;
a different split is refused unless `--force`, which you must not use after any
checkpoint exists. Every vocabulary, shard and checkpoint from here on carries this split.

## 8. Tokenize the clinical arm (reference vocabulary)

The 24-hour build fits the ONE frozen vocabulary on the train partition; the
hospitalization build reuses it to write the full-hospitalization shard, with every
eligible extubation's index stay added (`--extubation-cohort`; partition inherited from
the patient).

Memory: each run holds one site's windowed events, their sort and the encoded shard.
Measured on full MIMIC (2026-10-03 review) at 33-36 GB peak before the window pushdown;
see the after-change figures recorded with this runbook's commit. Before each run check
`free -g` (`available` column): the tokenizer warns below 48 GB free. Run the tokenizations
one at a time, never beside a training run or a cache build.

```bash
uv run python -m src.data.tokenize --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic --build-vocab --episodes output/intermediate_phi/episodes.parquet --workers 8
uv run python -m src.data.tokenize --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic --vocab output/intermediate_phi/mimic/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes.parquet --extubation-cohort output/intermediate_phi/extubation_cohort.parquet --workers 8
```

`--workers 8`: encode processes (each single-threaded in polars); `--workers 0` means every
usable CPU, capped at 8. The vocabulary carries every permissible CLIF 2.1.1 value of the
`vocabulary_allowlist` lists even where MIMIC never charts it.

Produces in `output/intermediate_phi/mimic/`: `vocab.json`, `events.parquet` (24 h, for
the baselines), `gem_events.parquet` (full hospitalization) and the aggregate
`tokenization_report.json` / `gem_tokenization_report.json`. Check: the vocabulary has no
`provenance.sample` flag (never pass `--sample-episodes` here), and the reports show the
expected stay counts. This one shard serves `clinical_soft`, `clinical_hard` (soft bins
dropped at load) and `textcode`.

## 9. Tokenize the two decile arms

Each decile arm has its own vocabulary (KTD11: matched granularity; the forced-edge arm
pins the decision thresholds). The tokenizer takes the arm's binning through a data
config derived with `run_tokenization_ablation.arm_data_config`:

```bash
mkdir -p output/intermediate_phi/configs
uv run python -c "import yaml; from src.train.run_tokenization_ablation import arm_data_config as f; cfg = yaml.safe_load(open('configs/data.yaml')); abl = yaml.safe_load(open('configs/tokenization_ablation.yaml')); [open(f'output/intermediate_phi/configs/data.{a}.yaml', 'w').write(yaml.safe_dump(f(cfg, abl['arms'][a]), sort_keys=False)) for a in ('global_deciles', 'deciles_plus_soft')]"
```

```bash
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.global_deciles.yaml --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic_decile --build-vocab --episodes output/intermediate_phi/episodes.parquet --workers 8
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.global_deciles.yaml --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic_decile --vocab output/intermediate_phi/mimic_decile/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes.parquet --workers 8
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.deciles_plus_soft.yaml --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic_decile_forced --build-vocab --episodes output/intermediate_phi/episodes.parquet --workers 8
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.deciles_plus_soft.yaml --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic_decile_forced --vocab output/intermediate_phi/mimic_decile_forced/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes.parquet --workers 8
```

Check: `tokenization_report.json` lists the matched-granularity exceptions (concepts with
too few distinct values keep fewer bins, KTD11).

## 10. Derive the continuous-fused arm

```bash
uv run python -m src.data.tokenize_continuous --primary-vocab output/intermediate_phi/mimic/vocab.json --primary-events output/intermediate_phi/mimic/gem_events.parquet --out output/intermediate_phi/mimic_continuous
```

Produces `output/intermediate_phi/mimic_continuous/vocab.json` and `gem_events.parquet`
(edgeless concept ids; threshold bins from the primary clinical segments).

## 11. TextCode encoder

The TextCode arm reads the clinical shard and embeds every token's description with a
frozen encoder at launch. Fetch it once so launches do not depend on the network:

```bash
uv run python -c "from transformers import AutoModel, AutoTokenizer; m = 'thomas-sounack/BioClinical-ModernBERT-base'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
```

## 12. Refit value stats on the full-hospitalization train stays

The 24-hour stats lack tokens seen only outside the ICU window, so every arm directory
gets stats fit on its `gem_events.parquet` train stays. Decided 2026-10-03 (product
authority): the stats are fit on the MIMIC AND Rush train partitions together (each
shard bound to the arm's one vocabulary). Before Rush is staged, MIMIC alone:

```bash
uv run python -m src.data.value_stats --events mimic=output/intermediate_phi/mimic/gem_events.parquet --out output/intermediate_phi/mimic/gem_value_stats.json
uv run python -m src.data.value_stats --events mimic=output/intermediate_phi/mimic_decile/gem_events.parquet --out output/intermediate_phi/mimic_decile/gem_value_stats.json
uv run python -m src.data.value_stats --events mimic=output/intermediate_phi/mimic_decile_forced/gem_events.parquet --out output/intermediate_phi/mimic_decile_forced/gem_value_stats.json
uv run python -m src.data.value_stats --events mimic=output/intermediate_phi/mimic_continuous/gem_events.parquet --out output/intermediate_phi/mimic_continuous/gem_value_stats.json
```

With Rush tokenized (step 13), refit each arm with both sites, e.g. the clinical arm:

```bash
uv run python -m src.data.value_stats --events mimic=output/intermediate_phi/mimic/gem_events.parquet --events rush=output/intermediate_phi/rush/gem_events.parquet --vocab output/intermediate_phi/mimic/vocab.json --out output/intermediate_phi/mimic/gem_value_stats.json
```

Check: each JSON carries `fit_partition: train`, the vocabulary and segments hashes, and
(pooled) `sites` with the train rows read per site. The pre-flight (step 15) and the
launcher check that EVERY site's numeric tokens have stats and list the gaps per site.
Memory: the shard is streamed in row batches and only the finite values are kept (about
8 bytes per numeric event). On a synthetic 8.8M-event shard the peak was 0.55 GB
resident against 1.5 GB for the previous whole-shard version (which peaked at 1.7 GB on
the 8.85M-event MIMIC sample); the full shard (about 134M events) should need a few GB,
not 20-25. Run the four commands one at a time and watch `free -g` the first time.

## 13. (Rush) Tokenize Rush with the frozen MIMIC vocabularies

Only once Rush is staged. Rush never fits a vocabulary; it imports each arm's.

```bash
uv run python -m src.data.tokenize --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush --vocab output/intermediate_phi/mimic/vocab.json --episodes output/intermediate_phi/episodes_rush.parquet --workers 8
uv run python -m src.data.tokenize --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush --vocab output/intermediate_phi/mimic/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes_rush.parquet --extubation-cohort output/intermediate_phi/extubation_cohort_rush.parquet --workers 8
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.global_deciles.yaml --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush_decile --vocab output/intermediate_phi/mimic_decile/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes_rush.parquet --workers 8
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.deciles_plus_soft.yaml --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush_decile_forced --vocab output/intermediate_phi/mimic_decile_forced/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes_rush.parquet --workers 8
uv run python -m src.data.tokenize_continuous --primary-vocab output/intermediate_phi/mimic/vocab.json --primary-events output/intermediate_phi/rush/gem_events.parquet --out output/intermediate_phi/rush_continuous
```

Then, in a commit, add `rush` to `launch.sites` in `configs/experiment_matrix.yaml` and a
`rush:` directory to every `arm_data` entry (`rush`, `rush_decile`,
`rush_decile_forced`, `rush_continuous`). The training vocabulary stays the reference
site's (the first `--data` directory's `vocab.json`); the value stats are refit on both
sites (step 12, second form). A Rush directory without a `vocab.json` is fine; one that has
it must carry the same binding, which the pre-flight checks. Every Rush shard row records
the hash of Rush's declarations (`site_declarations`, including the site-local file); the
pre-flight fails a shard built under other declarations.

## 14. Pre-flight rehearsal off the node (optional)

On a Mac, before going to the node:

```bash
uv run python -m src.train.preflight --synthetic --skip-gpu
```

Check: `PRE-FLIGHT PASSED`; the two-rank smoke runs over gloo on CPU.

## 15. Pre-flight on the node

```bash
uv run python -m src.train.preflight --episodes mimic=output/intermediate_phi/episodes.parquet --extubation-cohort mimic=output/intermediate_phi/extubation_cohort.parquet
```

With Rush: add `--episodes rush=output/intermediate_phi/episodes_rush.parquet
--extubation-cohort rush=output/intermediate_phi/extubation_cohort_rush.parquet`.

Prints one table and exits non-zero on any FAIL. It checks:

- environment: versions, commit, a clean tree;
- GPU: CUDA, two devices, native bf16, `nvidia-smi` (the driver mismatch is named), the
  driver against torch's CUDA build (step 2, "CUDA build and driver"), one NCCL library
  and its version, free memory per device, and a two-rank nccl bf16 smoke step through
  `engine.setup_ddp` and `engine.train` that must leave both ranks with identical
  parameters;
- the split freeze matches every episode artifact, and every shard's stays carry the
  frozen partitions;
- every arm's vocabulary (not a sample), shard rows and per-site vocabularies are bound to
  each other; soft arms have soft bins; value stats are bound, fit on train and cover the
  shard's numeric tokens;
- `configs/thresholds.yaml` loads; the edge check refuses any control on a bin edge and
  names it; proposed competing-risk causes are a warning;
- memory: builds the GEM cache for a 500-stay sample under
  `output/intermediate_phi/preflight_scratch/` (deleted afterwards), measures bytes per
  event and projects page cache, per-rank and peak node memory for the full train and
  validation corpus (warn above 70 % of available RAM, fail above 90 %); the cache
  directories are on a local filesystem;
- schedule: updates per screening and full budget, computed with the launch's own
  sampler, and the warm-up and checkpoint interval each run resolves to (decision 7):
  fails on a warm-up as long as the run or a run that writes no checkpoint, warns when
  only the final checkpoint would be written;
- disk: room for the caches still to build and for checkpoints.
- context (informational, per site): share of candidate anchors and of extubation
  time-zero prompts whose history exceeds 4,096 / 8,192 / 16,384 tokens
  (`src/eval/context_length.py`, aggregate only); warns when more than 10% of the prompts
  exceed 8,192. A trained run can then be scored at 4K vs 8K with
  `src.eval.threshold_eval --max-context 4096`.

If the two-rank smoke step hangs or times out (300 s) rather than failing, rerun it with
NCCL's own log and look at the interconnect (the L40s have no NVLink; peer-to-peer goes
over PCIe):

```bash
nvidia-smi topo -m
NCCL_DEBUG=INFO uv run python -m src.train.preflight --episodes mimic=output/intermediate_phi/episodes.parquet --extubation-cohort mimic=output/intermediate_phi/extubation_cohort.parquet
```

If the log stalls at peer-to-peer setup (`P2P` / `via P2P/IPC`), retry with
`NCCL_P2P_DISABLE=1` in front of the same command (traffic goes through host memory;
slower, but the run is gradient all-reduce of a ~30M model). If that passes, launch the
training runs with the same variable set and record it in the run's notes.

Memory figures measured on the MIMIC verification sample (1.2M sampled events) on
2026-10-03: 57 bytes per event of memory-mapped cache (one copy per node), 8-19 bytes per
event private to each rank, 26 bytes per event of multi-window stays for each rank's
target cache, and at most 380-515 bytes per event while rank 0 builds a cache. For the
full MIMIC train and validation partitions (about 100M events) that projects to roughly
6 GB of page cache, 2-5 GB per rank, and a one-time build peak of up to 30-40 GB on
rank 0 for the train partition. These are projections from a sample; the node's own
pre-flight output is the number to trust. If the build peak is too high, build the
caches with a single process first (step 17's dry run does this) so the two ranks only
map them.

## 16. Edge check and run specifications

```bash
uv run python -m src.train.run_matrix --edge-check
uv run python -m src.train.run_matrix --write
```

Produces: the edge-distance table of every arm's frozen vocabulary, then
`output/intermediate_phi/runs/<run_id>/run_spec.json` and `matrix_entry.json` for all 96
runs (16 configurations x 3 seeds x screening/full). Check: `edge check passed`; if a
control is refused, settle decision 3 above, then rerun. `--write` refuses until every
arm's vocabulary exists. The 33 claim-bearing runs are fixed here, before any launch
(KTD12).

## 17. Timing screening run (sizes the full budget)

Build the primary arm's GEM caches with one process (no training), then time one
screening run of the primary configuration:

```bash
uv run python -m src.train.run_tokenization_ablation --arm clinical_soft --objective-arm full --seed 1 --trajectory hospitalization --data output/intermediate_phi/mimic --site mimic --value-stats output/intermediate_phi/mimic/gem_value_stats.json --dry-run
uv run torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation --arm clinical_soft --objective-arm full --seed 1 --trajectory hospitalization --data output/intermediate_phi/mimic --site mimic --value-stats output/intermediate_phi/mimic/gem_value_stats.json --passes 0.05 --trunk d_model=512 --trunk n_layers=8 --trunk n_heads=8 --run-dir output/intermediate_phi/runs/clinical_soft.full.30m.time.s1.screening
```

The second command is the matrix's own launch line for
`clinical_soft.full.30m.time.s1.screening` (also in its `matrix_entry.json`). Check the
log: the curriculum schedule line, `updates_per_min`, loss values that are finite and
fall, the loader memory line, a `validation ... loss=` line from each validation, and the
measured parameter count in `run.json`.

Runtime notes for every training launch:

- Process-group timeout: collectives wait up to `CLIFATRON_DDP_TIMEOUT_MIN` minutes
  (default 120; NCCL's own default is 10) before the job is aborted. Raise it only if a
  legitimate step (a cache build, a checkpoint write) takes longer.
- Reproducibility: a seed fixes initialisation, batch order and sampling, but CUDA
  kernels that reduce with atomics (attention backward, embedding backward, index_add)
  are not bit-reproducible run to run. Two runs with one seed agree statistically, not
  bitwise; claims rest on the seed spread, never on bitwise equality.
- Attention: training attention on CUDA is restricted to the flash and memory-efficient
  SDPA kernels. A shape or dtype neither accepts raises instead of falling back to the
  math kernel (which would materialize `[batch, heads, T, T]` scores and run out of
  memory at 8,192 tokens).
- `runtime.compile` stays false. If it is turned on, the launchers compile the inner
  model and then wrap it in DDP. PyTorch documents both orders (the DDP notes wrap first
  so DDPOptimizer splits graphs at gradient buckets; the torch.compile guides compile the
  inner module first): time both on the node before enabling it for a claim run.

Sizing: the full budget is a placeholder of 1.0 pass in `configs/experiment_matrix.yaml`.
A pass is `ceil(batches per rank / grad_accum)` updates; the screening run trains 0.05 of
that. Hours per full run = (screening updates / 0.05 x full passes) / `updates_per_min` /
60. Multiply by the 48 full runs (one at a time, both GPUs each) to get the calendar
cost, choose the full budget, set `budgets.full`, commit, and rerun step 16 before any
full run. Screening results never decide which arms or seeds are reported (KTD12).

## 18. Launch the screening runs

Runs go one at a time; each uses both GPUs. The launch line of each run is in its
`matrix_entry.json`:

```bash
for d in output/intermediate_phi/runs/*.screening; do
  [ -f "$d/run.json" ] && continue
  cmd=$(uv run python -c "import json, sys; print(json.load(open(sys.argv[1]))['launch']['torchrun'])" "$d/matrix_entry.json")
  echo "$cmd" > "$d/launch.txt"
  eval "$cmd" > "$d/train.log" 2>&1 || echo "FAILED $d"
done
```

Check after each run: `run.json` exists (written at the end), `train.log` has no
traceback and a `schedule:` line, and `checkpoints/` holds the periodic checkpoints and
the final one (`ckpt_ep<E>_step<S>.pt` with `S` the run's update count; decision 7). Run
in `tmux` so a dropped SSH session does not kill the loop.

## 19. Evaluation

Zero-shot threshold evaluation of each finished run, on the partition `configs/claims.yaml`
names (never the sealed `internal_test` without `--final-evaluation`). Each arm reads its
own vocabulary and shard:

```bash
declare -A ARM_DIR=([clinical_soft]=mimic [clinical_hard]=mimic [textcode]=mimic [global_deciles]=mimic_decile [deciles_plus_soft]=mimic_decile_forced [continuous_fused]=mimic_continuous)
for d in output/intermediate_phi/runs/*.screening; do
  [ -f "$d/run.json" ] || continue
  arm=$(uv run python -c "import json, sys; print(json.load(open(sys.argv[1]))['tokenization_arm'])" "$d/run_spec.json")
  dir=output/intermediate_phi/${ARM_DIR[$arm]}
  ckpt=$(ls -t "$d"/checkpoints/*.pt | head -1)
  uv run python -m src.eval.threshold_eval --run-dir "$d" --checkpoint "$ckpt" --vocab "$dir/vocab.json" --shards "$dir/gem_events.parquet" --value-stats "$dir/gem_value_stats.json" --site mimic --device cuda
done
```

`--value-stats` is read for the continuous-fused arm only (its trunk reads each event's
normalized value; refused without it) and ignored for the others.

Produces `<run>/threshold_eval/scores.parquet` (row-level, governed) and
`<run>/threshold_eval/summary.json` (aggregate). Check: every scorer row is `evaluable`
or carries a stated reason.

The claims report reads ONLY claim-bearing full-budget runs and refuses anything else, so
it waits for the full runs. When they are done and evaluated:

```bash
uv run python -m src.eval.claims_report --runs $(uv run python -c "import glob, json; print(' '.join(d for d in sorted(glob.glob('output/intermediate_phi/runs/*.full')) if json.load(open(d + '/run_spec.json'))['claim_bearing']))") --out output/final_no_phi/claims_report.json
```

Count and token baselines on the 24-hour shard (no training run needed; fits on train,
tunes on validation):

```bash
uv run python -m src.eval.baselines --site mimic --data ~/Data/clif-source --episodes output/intermediate_phi/episodes.parquet --shards output/intermediate_phi/mimic/events.parquet --vocab output/intermediate_phi/mimic/vocab.json --out output/final_no_phi/baselines_mimic.json
```

Both outputs under `output/final_no_phi/` are aggregate and go through disclosure review
before leaving the node.

## Recovery

- A run that dies, or was stopped cleanly, resumes. To stop cleanly, send SIGTERM to
  the two worker processes, not to torchrun:

  ```bash
  pgrep -f "bin/torchrun --nproc_per_node=2"   # the torchrun agent's pid (not uv's)
  pkill -TERM -P <torchrun pid>                 # SIGTERM to its two workers
  ```

  (`uv run torchrun` leaves a `uv` parent whose command line also contains
  `torchrun --nproc_per_node=2`; the `bin/torchrun` pattern matches the agent only.)

  Each worker finishes its current update, they agree, rank 0 checkpoints, and both exit
  0. SIGINT and SIGHUP are handled the same way. A repeat of the signal within 5 s of the
  first is ignored (a Ctrl-C reaches each worker twice: from the terminal and forwarded by
  torchrun); a signal after that aborts without a checkpoint. Signalling torchrun itself
  (or Ctrl-C in its terminal) forwards the signal but kills the workers 30 s later, which
  may be before a 32-microbatch update and its checkpoint finish: prefer the `pkill`
  form. Then relaunch it with `--resume latest`, the
  `launch.torchrun_resume` line of its `matrix_entry.json` (the launch line plus
  `--resume latest`). `latest` is the checkpoint with the most updates in
  `<run-dir>/checkpoints`; a path works too. For the timing run of step 17:

  ```bash
  uv run torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation --arm clinical_soft --objective-arm full --seed 1 --trajectory hospitalization --data output/intermediate_phi/mimic --site mimic --value-stats output/intermediate_phi/mimic/gem_value_stats.json --passes 0.05 --trunk d_model=512 --trunk n_layers=8 --trunk n_heads=8 --run-dir output/intermediate_phi/runs/clinical_soft.full.30m.time.s1.screening --resume latest
  ```

  Model, optimizer (every per-head group), LR schedule, RNG, the optimizer-update
  counter (so the curriculum continues where it stopped) and the epoch (the sampler's)
  are restored. A checkpoint of another arm, trunk, objective, seed or schedule length is
  refused; `--fresh-schedule` keeps the new run's LR schedule for a deliberate
  continuation. A mid-pass checkpoint resumes by replaying that pass from its start
  (the documented approximation). `--resume latest` with no checkpoint is refused: drop
  it to restart from scratch.
- `src.train.pretrain` resumes the same way (`--resume <ckpt>` or `--resume latest` in
  `runtime.ckpt_dir`, `--fresh-schedule`).
- A stale or half-built GEM cache is refused with "stale GEM cache ... delete it": remove
  that `gem_cache/<partition>-...` directory and rerun; it is rebuilt under the lock.
- A rebuilt shard or vocabulary gets a new cache directory automatically (the name holds
  the shard and vocabulary hashes); old ones are not deleted, so prune `gem_cache/` by hand
  when disk runs short.
- After a reboot, rerun step 2 and step 15 before relaunching.

## Known gaps in the entry points (as of 2026-10-03)

- `src.data.tokenize` has no `--arm` option; the decile arms' data configs are written by
  the `python -c` line in step 9.
- Checkpoints are not pruned: a run keeps all its periodic checkpoints (about 5 + 1;
  the 30M model with AdamW state is a few hundred MB each). Prune finished runs by hand
  when the pre-flight's disk check gets close.
