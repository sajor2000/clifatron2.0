# L40 G2 runbook — pure-NTP GEM base pretraining on the full MIMIC restage

Everything below was proven on the Mac (MPS) against the staged 64.9k-stay MIMIC
extract on 2026-09-26 (see `gem-overnight-log.md` for the full dated record and
the numbers to beat). This is the operational handoff for `rudu-hpcg004`
(2× L40, 48 GB, bf16, DDP via torchrun).

## 0. Prerequisites

- `git pull` (main through `f71f5a9` or later) — every fix below is committed.
- Reboot FIRST if `nvidia-smi` still fails on the driver/library mismatch
  (AGENTS.md known blocker).
- Restage the FULL 546k-stay MIMIC-IV-Ext-CLIF tables at `~/Data/clif-source`
  (the Mac extract was a 64.9k-stay subset; L40 numbers will differ from the
  overnight log's).
- `uv sync --group dev`.

## 1. ETL (order matters; all steps proven 2026-09-26)

```bash
CLIF_DATA_DIR=~/Data/clif-source uv run --frozen --group dev \
  python -m src.data.cohort --data ~/Data/clif-source \
  --out output/intermediate_phi/episodes.parquet

CLIF_DATA_DIR=~/Data/clif-source uv run --frozen --group dev \
  python -m src.eval.clif_auto_labeler --data ~/Data/clif-source \
  --episodes output/intermediate_phi/episodes.parquet \
  --out output/intermediate_phi/mimic/labels.parquet
# (expect a benign warning: vitals table has no unit_col)

CLIF_DATA_DIR=~/Data/clif-source uv run --frozen --group dev \
  python -m src.data.tokenize --site mimic --in ~/Data/clif-source \
  --out output/intermediate_phi/mimic --build-vocab \
  --episodes output/intermediate_phi/episodes.parquet

# outcome_join needs tokenize's events.parquet + vocab.json — run AFTER tokenize
CLIF_DATA_DIR=~/Data/clif-source uv run --frozen --group dev \
  python -m src.data.outcome_join \
  --labels output/intermediate_phi/mimic/labels.parquet \
  --events output/intermediate_phi/mimic/events.parquet \
  --vocab output/intermediate_phi/mimic/vocab.json \
  --out output/intermediate_phi/mimic/events_with_outcomes.parquet

CLIF_DATA_DIR=~/Data/clif-source uv run --frozen --group dev \
  python -m src.data.value_stats \
  --events output/intermediate_phi/mimic/events.parquet \
  --out output/intermediate_phi/mimic/value_stats.json
```

Verify the data dir contains `events.parquet`, `events_with_outcomes.parquet`,
`vocab.json`, `value_stats.json`. The trainer HARD-FAILS without
`events_with_outcomes.parquet` (TTE gate) and without `--value-stats`
(numeric-values gate) — both gates are correct, not bugs.

## 2. Training launch (2× L40, DDP)

```bash
CLIF_DATA_DIR=~/Data/clif-source torchrun --nproc_per_node=2 -m src.train.pretrain \
  --config configs/train.yaml \
  --model-config configs/model.gem-ntp.yaml \
  --data output/intermediate_phi/mimic --site mimic \
  --value-stats output/intermediate_phi/mimic/value_stats.json
```

- `configs/model.gem-ntp.yaml` = the pure-NTP GEM base (plan D-G1). The trunk is
  state-dict compatible with the TTE recipe, so prediction heads can attach later.
- `configs/train.yaml` (NOT train.mps.yaml): CUDA flash attention needs NONE of
  the MPS guards — `token_budget` and `cache_clear_every` stay UNSET on CUDA;
  uniform per_gpu=4 batches are correct there. Length-grouped batching applies
  to single-process runs only (DDP uses DistributedSampler); if padding waste
  matters at scale, fold a world-aware TokenBudgetBatchSampler in as a G2 L40
  item — do NOT silently change batching mid-recipe.
- bf16 + compile stay per `configs/train.yaml` (`compile: true` after the
  first production run stabilizes — flip it only deliberately).
- Checkpoints + validation are step-granular (ckpt_every 2000, val_every from
  `eval_schedule`): mid-epoch checkpoints work under DDP; mid-epoch VALIDATION
  is single-process-only (rank-0-only eval would desync ranks) — DDP validates
  at epoch boundaries. If finer val cadence is wanted on DDP, that's a
  collective-aware engine change, not a config.
- Resume: `--resume <ckpt>`. Continuation runs (new schedule) use
  `--fresh-schedule`, which also restores configured LRs (the optimizer state
  pins group LRs at save time). End-of-epoch checkpoints record
  epochs-consumed, so resume never replays a completed epoch; mid-epoch
  checkpoints replay the partial epoch.

## 3. Post-train: generate + evaluate + inspect (all proven 2026-09-26)

```bash
CLIF_DATA_DIR=~/Data/clif-source uv run --frozen --group dev \
  python -m src.model.generate \
  --checkpoint output/intermediate_phi/checkpoints/<best>.pt \
  --model-config configs/model.gem-ntp.yaml --data-config configs/data.yaml \
  --vocab output/intermediate_phi/mimic/vocab.json \
  --prompts <prefixes.txt> --n-simulations 8 --max-new-tokens 256 \
  --temperature 1.0 --top-p 0.95 --device cuda \
  --output output/intermediate_phi/sims_l40.parquet

CLIF_DATA_DIR=~/Data/clif-source uv run --frozen --group dev \
  python -m src.eval.generative \
  --sims output/intermediate_phi/sims_l40.parquet \
  --events output/intermediate_phi/mimic/events.parquet \
  --vocab output/intermediate_phi/mimic/vocab.json \
  --out output/intermediate_phi/gem_eval_l40.json
```

- Sampling is closed-world over the frozen CLIF vocab (hard rule 2): the
  trunk embeds 10k slots but the real vocab is ~292 tokens; untrained slot
  rows cannot leak into rollouts.
- The sims parquet carries a `prompt` provenance column: the eval pairs
  rollouts to their real continuations by exact prefix match (no conventions).
- Reports and sims are PHI-derived: keep them under `output/` on the node
  (git-ignored); never commit or export.

## 4. Numbers to beat (Mac, 64.9k-stay subset — expect better on full data)

| metric | 3000 steps | 6000 steps |
|---|---|---|
| val loss (perplexity) | 2.4138 (~11.2) | 2.4211 (plateaued on the subset) |
| JS divergence (gen vs real rates) | 0.425 | 0.358 |
| top-8 / top-32 overlap | 0.0 / 0.5 | 0.125 / 0.594 |
| vitals rate gen (real 0.538) | 0.141 | 0.276 |
| key-event recall (unigram) | 0.359 | 0.318 (anchoring does NOT improve with NTP — that's G4/G5) |

## 5. Gotchas proven on the Mac that transfer

- `clif-validate`'s vendor-drift guard goes red when `src/` and the vendored
  tree diverge — re-run `clif-validate/scripts/sync_vendor.py` after touching
  vendored-closure files (tokenize.py etc.), then the guard is green again.
- Engine `ckpt_every`/`val_every` are STEP-granular (verified live); don't
  re-assume epoch-boundary semantics.
- The resume-equivalence engine test is load-bearing: it caught an
  epochs-consumed off-by-one in step-granular checkpointing before it shipped.
- Prompt files are space-joined TOKEN STRINGS (not ids), mapped through the
  vocab with `<unk>` fallback; empty lines become a `<bos>`-only prompt.

## 6. After G2

G3 eval harness is ready (`src/eval/generative.py`); G4 (instruction-prefix
conditioning) and G5 (GRPO/DAPO post-training per the plan) follow the L40 base.
The Mac's G3 finding stands: NTP steps improve the marginal (calibration) but
not rollout anchoring — anchoring is G4/G5 work.
