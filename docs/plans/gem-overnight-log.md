# GEM overnight log — local validation track (this Mac)

Working loop state file. The overnight loop (hourly) reconstructs progress from this
log + `git log` + `output/intermediate_phi/` contents. Append a dated entry per
completed unit. Plan of record: `2026-09-25-001-feat-icu-gem-generative-model-plan.md`.

Goal: prove the whole GEM stack — ETL → packed shards → MPS smoke train → generate →
viewer — against the real staged MIMIC CLIF tables (`~/Data/clif-source`, 16 parquet
files) on THIS Mac, so the code is proven before it moves to the L40 box.

## Checklist

- [x] U1. ETL dry-run on staged tables (all 5 tables read clean)
- [x] U2. Cohort artifact (`src.data.cohort` → `output/intermediate_phi/episodes.parquet`)
- [x] U3. Full tokenize, first site builds vocab (stats → log)
- [x] U4. Auto-label → outcome_join → value_stats (packed-loader smoke happens inside U5 dry-run)
- [x] U5. MPS smoke train: short pure-NTP run (fp32, small scale) → checkpoint
- [x] U6. Smoke generate via `src.model.generate` CLI → sims parquet → viewer-verified
- [x] Suites green after each unit; code-only commits pushed

## Entries

### 2026-09-26 00:15 — U1–U6 complete, full GEM loop proven on this Mac (real MIMIC)

**U1 ETL dry-run** — all 5 staged tables read clean, zero schema fixes needed:
vitals 55.5M / labs 44.9M / resp_support 1.6M / meds 7.6M / adt 1.5M raw events
(111M total; 9/45/8/53/8 concepts — all mCIDE).

**U2 cohort** — 64,940 staged stays → 50,986 eligible with clean splits
(train 30,696 / validation 7,640 / calibration 5,009 / internal_test 7,641).
NOTE: staged extract is 64.9k stays, not the full 546k MIMIC — L40 numbers
will differ.

**U3 tokenize** — 50,986 stays, 13,824,973 tokens; mean 271 / median 247 /
p95 471 / max 6,413 tokens per stay (max < 8192 context ✓).
Vocab: **292 tokens**, 10 numeric concepts, clinical-segment bins + forced edges.
Manifest hashes: vocabulary `eb02faa…`, numeric_edges `6ab0d24…`
(full set in `output/intermediate_phi/mimic/vocab.json`).

**U4 labels + outcomes + value stats** — auto-labeler: 50,986 outcome rows
(vitals unit_col warning is expected/benign); outcome_join: 28,209
positive-outcome instances across 3 outcomes; value_stats: 223 per-token stats,
vocab-hash-bound (`eb02faa…`).

**U5 MPS smoke train (pure NTP, configs/model.gem-ntp.yaml + configs/train.smoke.yaml)**
— 45.2M params, 40 optimizer steps, ~186 updates/min on MPS, loss 5.1→~4.3,
cr/th/val exactly 0.0000 (fail-to-zero guards verified live on real data).
Run ID `51403cf50a85`, checkpoint `output/intermediate_phi/checkpoints_smoke/ckpt_ep1_step40.pt`.
Found + fixed en route: pretrain device selection was CUDA-or-CPU only — never
MPS, contradicting the AGENTS.md dev workflow. Now cuda → mps → cpu.

**U6 smoke generate + viewer** — 3 real-sequence prompts × 4 sims, 128 tokens each,
on MPS. Found + fixed: sampling was open-world over the 10k embedding slots while
the real frozen vocab is 292 tokens — weakly-trained slot rows leaked as `<unk:N>`
decodes. Added `allowed_token_ids` closed-world masking (hard rule 2: frozen mCIDE
vocab); regenerated: **0 unks**, all tokens in-vocab, clinically legible rollouts.
Viewer verified on 127.0.0.1:8042: `sims_smoke` (12 rows, parquet) + 200 real
stays (txt) + vocab-lock OOV panel, root 200, previews legible.

**Suites**: tests/ 408 passed 4 skipped (+1 closed-world sampler test);
clif-validate/ 32 passed.

**Ready for the L40 box (G2)**: configs/model.gem-ntp.yaml + configs/train.yaml
+ `python -m src.train.pretrain --config configs/train.yaml --model-config
configs/model.gem-ntp.yaml --data <site dir> --site mimic --value-stats
<value_stats.json>`. Same data dir must contain events_with_outcomes.parquet +
vocab.json + value_stats.json. Suggest full 546k MIMIC restage before the run.

### 2026-09-26 00:55 — MPS OOM found + fixed; viewer loads events.parquet natively

**Viewer upgrades** (src/viewer/sequence_viewer.py): ParquetSource now ingests
LIST-typed token columns directly (tokenizer events.parquet `token`: Int64 ids,
joined at read time), decodes integer ids to vocab token strings when a
vocab lock is passed (unknown ids stay numeric — never silent), and accepts
`hospitalization_id`/`hosp_id` as id columns. Real check: 50,986-row
events.parquet served on 127.0.0.1:8042 with legible previews; no txt export
needed anymore.

**MPS OOM crash — the real bug of the night.** The 3000-step overnight run
(configs/train.mps.yaml, effective batch 32) died at update ~205 (loss 2.71,
descending; no validation involved — val_every 250 not reached):
`MPS backend out of memory (allocated 83.94 GiB, tried 4.90 GiB more)`.
Root cause: uniform shuffling over heavy-tailed lengths (mean 271 / p95 471 /
max 6,413) pads every batch to its longest member — the math-path attention
for a 4×6413 batch needs ~4.9 GiB, and every distinct padded shape gets
cached by the MPS allocator, so the watermark climbed to 84 GiB over ~200
updates and the next long batch killed it.

**Fix: LengthGroupedSampler** (src/data/dataset.py, HF group_by_length style):
shuffle, sort by length within mega-batches, regroup, shuffle batch order.
Padded shapes recycle (flat watermark), long stays only meet long stays, and
padding waste collapses — the 40-step sanity smoke went 186 → 296 updates/min
(1.6x). Wired into pretrain.py for single-process runs; DDP ranks keep
DistributedSampler until G2 L40 bring-up (fold world-awareness in there —
the L40s will want grouping for the same throughput reason, bf16 or not).

Also: ckpt_every 500 → 100 in train.mps.yaml (the crash lost all 200 updates —
no checkpoint had fired). 3000-step run relaunched at 00:55, ~91 updates/min.

**Suites**: tests/ green after each change (dataset 32, viewer 6, generate 19).

**Live for the morning**: viewer on 127.0.0.1:8042 (events.parquet 50,986 rows +
sims_smoke 12 rows + vocab lock). Relaunch with /tmp/run_viewer.sh if it dies.

### 2026-09-26 01:55 — second OOM + engine step-granularity fixed

**Second OOM** (overnight run #2, length-grouped): died at update ~410, same
signature (85 GiB held, 4.9 GiB math-path attention attempt). Grouping narrows
shape diversity but the real ratchet is the math-path attention's saved-for-
backward T² buffers per layer on long batches (~40 GiB for one 4×6413 batch):
the MPS allocator caches those blocks at lengths that never recur, so the
watermark ratchets up over the first couple of long batches and the next one
dies. Fix: `runtime.cache_clear_every` (engine calls `torch.mps.empty_cache()`
every N updates on MPS only; opt-in, set to 10 in train.mps.yaml). CUDA is
unaffected (caching allocator + real flash attention) — this is an MPS-local
guard.

**Engine step-granularity (the bigger fix)**: validation and checkpointing
previously fired at EPOCH BOUNDARIES ONLY — ckpt_every/val_every promised step
granularity they never delivered (run #1's crash lost all 200 updates; the
first epoch at effective batch 32 is ~959 updates). Now `_train_one_epoch` takes
a per-update `boundary_cb` (all ranks, same update):
- mid-epoch checkpoints — DDP-safe (synchronized updates + all-gather RNG);
  end-of-epoch saves record epochs-consumed (epoch+1) so resume never replays
  a completed epoch (the resume-equivalence test caught the off-by-one);
  mid-epoch saves replay the partial epoch on resume (documented).
- mid-epoch validation — single-process only (rank-0-only eval would desync
  DDP ranks mid-epoch; DDP keeps epoch-boundary val).
Proven live: 60-step boundary smoke → validation at steps 15/30/45/60
(loss 5.93 → 4.04), checkpoints `ckpt_ep0_step20/40` mid-epoch.

Overnight run #3 relaunched 01:55 with cache guard + ckpt_every 100.

**Suites**: tests/ 410 passed 4 skipped; clif-validate/ 32.

**Run #3 status at 02:10 (all fixes live)**: update 410+ (past both prior crash
points), loss ~2.0-2.7, RSS flat ~9.5 GiB (was ratcheting to 85), checkpoints
ckpt_ep0_step100/200/300/400 on disk, first mid-epoch validation
`global_step=250 loss=2.7462`. Throughput ~68-82 updates/min (cache clears
cost ~15%). ETA to 3000 steps: ~40 min. Next loop firing: monitor to
completion, then generate rollouts from the best checkpoint (val metric
manifest) into sims_mps.parquet, inspect in the viewer, and log final numbers.

### 2026-09-26 03:15 — OOM #3 + the real fix: token-budget batches

**OOM #3** (run #3, length-grouped + cache clears every 10): died at update
~2300 mid-BACKWARD (86.4 GiB held, 4.9 GiB attempt). The clear bounds CACHED
junk between updates, but one long batch's LIVE backward transients are ~40 GiB
across 8 layers (math-path attention saves T² per layer for backward) — plus
junk since the last clear, it crossed the ceiling anyway. 2,300 updates and 23
checkpoints survived on disk.

**Fix: TokenBudgetBatchSampler** (src/data/dataset.py) — length-grouped batches
packed to B × max_len ≤ `runtime.token_budget` (8192): a 6,413-token stay
batches ALONE, short stays pack tight. Live transients bounded, padding waste
~zero. Opt-in (CUDA flash attention doesn't need it; DDP/CUDA keeps uniform
per_gpu). Wired for train (shuffled) + validation (deterministic sorted);
engine's set_epoch hook now reaches batch samplers (DataLoader.sampler is None
when batch_sampler is used). cache_clear_every 10 → 1 in train.mps.yaml.

Proven: 60-step boundary smoke with token budget 4096 — validations at
15/30/45, RSS flat ~8.8 GiB, Training complete. Both suites green
(tests/ 411, clif-validate/ 32).

**Run #4 (resume)**: resumed from ckpt_ep2_step2300 (epoch 2, loss ~2.4
continuity, cosine-tail LR) at 03:15 with token budget + clear-every-1;
~74 updates/min; ~660 steps to 3000. Mid-epoch resume replays epoch 2
(documented approximation — ~380 updates of duplicate exposure, noted).
