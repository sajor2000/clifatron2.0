# L40 runbook: raw CLIF tables to the screening runs

Written 2026-10-03 under `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md`
(unit U19). It replaces `docs/plans/l40-g2-runbook.md`, which predates tokenizer v2 and
the full-hospitalization path and must not be used to launch anything.

Node: `rudu-hpcg004`, 2x L40 (48 GB each, no NVLink), bf16, DDP through `torchrun`.
Run every command from the repository root. `tests/test_runbook_commands.py` checks that
every `uv run python -m ...` and `torchrun ... -m ...` line in this file names an existing
module and only flags that module accepts, so edit commands here and in the code together.

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
The code runs either way; none of these is decided by an engineer.

1. **Held-out share (KTD6).** The split is baked into the vocabulary, every shard and every
   checkpoint. Decide the share from the held-out arm sizes of the audit blind stage
   (step 6) and record it with `--write-split-freeze` (step 7). Changing it later means
   rebuilding everything from step 4.
2. **Seven proposed competing-risk thresholds.** `configs/thresholds.yaml` lists ten
   competing-risk causes; three repeat the cohort contract and seven carry
   `status: proposed` (respiratory rate 24, creatinine 2.0, bilirubin 2.0, platelets 100,
   heart rate 130, SBP 90, temperature 38.3). They need physician confirmation. The
   pre-flight reports them as a warning, not a failure.
3. **Creatinine 1.3 control.** On the MIMIC verification sample the control creatinine
   above 1.3 sits on a bin edge in both decile arms, so the edge check refuses it. The
   full-data decile edges may differ; if step 16 refuses it again, either drop the control
   from `configs/thresholds.yaml` or accept fewer controls for creatinine.
4. **Agreement margins.** The extubation benchmark margins in
   `configs/extubation_benchmarks.yaml` are `proposed` until the emulation protocol is
   registered. They do not gate the screening runs.
5. **Casey 2021 comparison.** The unblinded all-comer comparison against Casey 2021 is the
   one outcome-by-arm run allowed before registration. It is the product authority's
   deliberate run, not part of this runbook.
6. **Claim 1 attribution arm.** Off in `configs/experiment_matrix.yaml`; whether the claim 1
   rule uses it is open.
7. **Checkpoint cadence and warm-up against the run length.** The engine writes a
   checkpoint only every `runtime.ckpt_every` updates (2000 in `configs/train.yaml`); there
   is no unconditional final checkpoint. With `per_gpu: 4`, `grad_accum: 32` and two ranks
   an update is 256 windows, so one pass over the MIMIC sample's train stays is 13
   updates, and the full corpus is likely a few hundred. At that length a screening run
   (0.05 pass) and even a 1-pass full run write no checkpoint, so step 19 has nothing to
   evaluate and nothing can be resumed; `warmup_steps: 2000` also exceeds the run, so the
   learning rate never leaves the linear warm-up. The pre-flight's schedule check prints
   the update count per budget from the real sampler and fails on this. Set
   `runtime.ckpt_every`, `schedule.warmup_steps` (and if wanted `grad_accum`) in
   `configs/train.yaml` from the step 15 numbers, commit, and rerun step 15.

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

## 3. Stage the data

MIMIC-IV-Ext-CLIF is already at `~/Data/clif-source` (546,028 stays, about 134M events).

```bash
ls ~/Data/clif-source/*.parquet | wc -l
df -hT ~/Data/clif-source output
```

Check: the CLIF tables are present, and `output/` is on a local filesystem (`ext4`, `xfs`;
not `nfs`, `cifs`).

Rush: placeholder. Stage the Rush CLIF tables at `~/Data/clif-rush` only once the
governance approval is in place. Every Rush step below is marked "(Rush)"; skip it until
then. UChicago is external validation by model-to-data and is never staged here.

## 4. Build the episode artifact

```bash
uv run python -m src.data.cohort --data ~/Data/clif-source --out output/intermediate_phi/episodes.parquet
# (Rush)
uv run python -m src.data.cohort --data ~/Data/clif-rush --out output/intermediate_phi/episodes_rush.parquet
```

Produces: the patient-grouped episode and split artifact, with partitions from
`configs/train.yaml` `data_contract.partitions` (60/15/10/15 today) and `split_seed`.
Check: the command finishes without a `QualificationError`.

## 5. Extubation cohort and labels

```bash
uv run python -m src.data.extubation_cohort --data ~/Data/clif-source --site mimic --episodes output/intermediate_phi/episodes.parquet --out output/intermediate_phi/extubation_cohort.parquet
uv run python -m src.eval.extubation_labeler --data ~/Data/clif-source --cohort output/intermediate_phi/extubation_cohort.parquet --out output/intermediate_phi/extubation_cohort_labels.parquet
```

Produces: the extubation cohort (each patient's partition inherited from the episode
artifact) and its outcome labels. Check: the printed waterfall counts are suppressed
aggregates only. The labels are not read by the blind stage.

## 6. Audit blind stage

```bash
uv run python -m src.eval.extubation_audit blind --cohort mimic=output/intermediate_phi/extubation_cohort.parquet
```

Produces: `output/final_no_phi/extubation_audit_blind_mimic.json` and `.csv`
(`pending_review`), and a ledger entry. Check: the held-out arm sizes
(`mimic|held_out|eligible|arm=...`) for conventional oxygen, HFNC and NIV. At 60/15/10/15
the MIMIC held-out NIV and HFNC arms are small (see the blind-stage report on the node; arm
counts are not copied into this public repository).

## 7. Decide the held-out share and freeze the split

If the held-out share changes, edit `configs/train.yaml` `data_contract.partitions`,
commit, and rerun steps 4 to 6. Then freeze:

```bash
uv run python -m src.train.preflight --write-split-freeze --episodes mimic=output/intermediate_phi/episodes.parquet --audit-blind output/final_no_phi/extubation_audit_blind_mimic.json --approver "J.C. Rojas"
```

With Rush staged, add `--episodes rush=output/intermediate_phi/episodes_rush.parquet`.

Produces: `output/final_no_phi/split_freeze.json` (episode artifact SHA-256, content split
hash, configured and observed partition shares, split seed, held-out arm counts from the
blind export, approver, date). Check: the printed split hash. The freeze is written once;
a different split is refused unless `--force`, which you must not use after any
checkpoint exists. Every vocabulary, shard and checkpoint from here on carries this split.

## 8. Tokenize the clinical arm (reference vocabulary)

The 24-hour build fits the ONE frozen vocabulary on the train partition; the
hospitalization build reuses it to write the full-hospitalization shard.

```bash
uv run python -m src.data.tokenize --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic --build-vocab --episodes output/intermediate_phi/episodes.parquet --workers 0
uv run python -m src.data.tokenize --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic --vocab output/intermediate_phi/mimic/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes.parquet --workers 0
```

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
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.global_deciles.yaml --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic_decile --build-vocab --episodes output/intermediate_phi/episodes.parquet --workers 0
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.global_deciles.yaml --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic_decile --vocab output/intermediate_phi/mimic_decile/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes.parquet --workers 0
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.deciles_plus_soft.yaml --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic_decile_forced --build-vocab --episodes output/intermediate_phi/episodes.parquet --workers 0
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.deciles_plus_soft.yaml --site mimic --in ~/Data/clif-source --out output/intermediate_phi/mimic_decile_forced --vocab output/intermediate_phi/mimic_decile_forced/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes.parquet --workers 0
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
gets stats fit on its own `gem_events.parquet` train stays:

```bash
uv run python -m src.data.value_stats --events output/intermediate_phi/mimic/gem_events.parquet --out output/intermediate_phi/mimic/gem_value_stats.json
uv run python -m src.data.value_stats --events output/intermediate_phi/mimic_decile/gem_events.parquet --out output/intermediate_phi/mimic_decile/gem_value_stats.json
uv run python -m src.data.value_stats --events output/intermediate_phi/mimic_decile_forced/gem_events.parquet --out output/intermediate_phi/mimic_decile_forced/gem_value_stats.json
uv run python -m src.data.value_stats --events output/intermediate_phi/mimic_continuous/gem_events.parquet --out output/intermediate_phi/mimic_continuous/gem_value_stats.json
```

Check: each JSON carries `fit_partition: train` and the vocabulary and segments hashes.
Memory: this step reads the whole shard into Python lists. On the MIMIC verification
sample (8.85M events) it peaked at 1.7 GB resident; scaled linearly, the full shard
(about 134M events) needs roughly 20-25 GB. Run the four commands one at a time and
watch `free -g`.

## 13. (Rush) Tokenize Rush with the frozen MIMIC vocabularies

Only once Rush is staged. Rush never fits a vocabulary; it imports each arm's.

```bash
uv run python -m src.data.tokenize --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush --vocab output/intermediate_phi/mimic/vocab.json --episodes output/intermediate_phi/episodes_rush.parquet --workers 0
uv run python -m src.data.tokenize --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush --vocab output/intermediate_phi/mimic/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes_rush.parquet --workers 0
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.global_deciles.yaml --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush_decile --vocab output/intermediate_phi/mimic_decile/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes_rush.parquet --workers 0
uv run python -m src.data.tokenize --config output/intermediate_phi/configs/data.deciles_plus_soft.yaml --site rush --in ~/Data/clif-rush --out output/intermediate_phi/rush_decile_forced --vocab output/intermediate_phi/mimic_decile_forced/vocab.json --trajectory hospitalization --episodes output/intermediate_phi/episodes_rush.parquet --workers 0
uv run python -m src.data.tokenize_continuous --primary-vocab output/intermediate_phi/mimic/vocab.json --primary-events output/intermediate_phi/rush/gem_events.parquet --out output/intermediate_phi/rush_continuous
```

Then, in a commit, add `rush` to `launch.sites` in `configs/experiment_matrix.yaml` and a
`rush:` directory to every `arm_data` entry (`rush`, `rush_decile`,
`rush_decile_forced`, `rush_continuous`). Value stats stay the reference site's (MIMIC,
the first `--data` directory), and so does the training vocabulary (the first `--data`
directory's `vocab.json`). A Rush directory without a `vocab.json` is fine; one that has
it must carry the same binding, which the pre-flight checks.

## 14. Pre-flight rehearsal off the node (optional)

On a Mac, before going to the node:

```bash
uv run python -m src.train.preflight --synthetic --skip-gpu
```

Check: `PRE-FLIGHT PASSED`; the two-rank smoke runs over gloo on CPU.

## 15. Pre-flight on the node

```bash
uv run python -m src.train.preflight --episodes mimic=output/intermediate_phi/episodes.parquet
```

With Rush: add `--episodes rush=output/intermediate_phi/episodes_rush.parquet`.

Prints one table and exits non-zero on any FAIL. It checks:

- environment: versions, commit, a clean tree;
- GPU: CUDA, two devices, native bf16, `nvidia-smi` (the driver mismatch is named), free
  memory per device, and a two-rank nccl bf16 smoke step through `engine.setup_ddp` and
  `engine.train` that must leave both ranks with identical parameters;
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
  sampler, against `ckpt_every` and `warmup_steps` (decision 7);
- disk: room for the caches still to build and for checkpoints.

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
torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation --arm clinical_soft --objective-arm full --seed 1 --trajectory hospitalization --data output/intermediate_phi/mimic --site mimic --value-stats output/intermediate_phi/mimic/gem_value_stats.json --passes 0.05 --trunk d_model=512 --trunk n_layers=8 --trunk n_heads=8 --run-dir output/intermediate_phi/runs/clinical_soft.full.30m.time.s1.screening
```

The second command is the matrix's own launch line for
`clinical_soft.full.30m.time.s1.screening` (also in its `matrix_entry.json`). Check the
log: the curriculum schedule line, `updates_per_min`, loss values that are finite and
fall, the loader memory line, and the measured parameter count in `run.json`.

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
traceback, and `checkpoints/` holds a checkpoint (it will not if the run had fewer than
`ckpt_every` updates; see decision 7). Run in `tmux` so a
dropped SSH session does not kill the loop.

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
  uv run python -m src.eval.threshold_eval --run-dir "$d" --checkpoint "$ckpt" --vocab "$dir/vocab.json" --shards "$dir/gem_events.parquet" --site mimic
done
```

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

- A run that dies: delete its `run.json` if present, and relaunch the same line from its
  `matrix_entry.json`. The run is reproducible from its seed, but it restarts from step
  0: `src.train.run_tokenization_ablation` has no `--resume` yet (see "Known gaps").
- `src.train.pretrain` does resume (`--resume <ckpt>`, `--fresh-schedule` for a new
  schedule). On resume the curriculum continues from the checkpoint's optimizer-update
  counter, not a count of forward calls, so the loss weights pick up where they stopped.
  Checkpoints are written every `runtime.ckpt_every` updates (2000) and at the end.
- A stale or half-built GEM cache is refused with "stale GEM cache ... delete it": remove
  that `gem_cache/<partition>-...` directory and rerun; it is rebuilt under the lock.
- A rebuilt shard or vocabulary gets a new cache directory automatically (the name holds
  the shard and vocabulary hashes); old ones are not deleted, so prune `gem_cache/` by hand
  when disk runs short.
- After a reboot, rerun step 2 and step 15 before relaunching.

## Known gaps in the entry points (as of 2026-10-03)

- `src.train.run_tokenization_ablation` (the matrix launcher) has no `--resume`; a crashed
  matrix run restarts from scratch.
- `src.data.tokenize` has no `--arm` option; the decile arms' data configs are written by
  the `python -c` line in step 9.
- `src.data.value_stats` loads the whole shard into Python lists (step 12 memory note).
- `src.train.engine.train` checkpoints only every `ckpt_every` updates and never at the
  end of a run (decision 7).
