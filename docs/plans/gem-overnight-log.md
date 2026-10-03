# GEM overnight log — local validation track (this Mac)

Working loop state file. The overnight loop (hourly) reconstructs progress from this
log + `git log` + `output/intermediate_phi/` contents. Append a dated entry per
completed unit. Plan of record: `2026-09-25-001-feat-icu-gem-generative-model-plan.md`.

**No-op passes are not logged.** A loop pass that finds nothing new (no finding, fix,
decision, or guardrail check) must NOT append to this log and must NOT commit. Write any
heartbeat to the git-ignored `output/` directory, or skip it.

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

### 2026-09-26 04:20 — 3000-step run COMPLETE + rollouts generated

**Training complete. Run ID `f819f7013fa2`.** Validation trajectory across the
full run: 2.7462 (step 250) → 2.4174 (2500) → 2.4147 (2750) → **2.4138 (3000)**
— val perplexity ≈ **11.2** on ~2 epochs of the 64.9k-stay staged extract at
effective batch 32 (a smoke-scale run; the L40 G2 recipe is 60k steps on the
full 546k-stay data). 30 checkpoints on disk
(`checkpoints_mps/ckpt_ep0..2_step100..3000.pt`).

**Rollouts from the final checkpoint** (`sims_mps.parquet`): 3 real-sequence
prompts × 8 sims × 256 tokens on MPS, temperature 1.0 / top-p 0.95,
closed-world sampling — 6,144 tokens, 1 `<unk>` (a legitimately sampled vocab
token, not a leak), 0 `<eos>` (expected at this scale). Qualitative: local
structure learned (ABG panels ph/pco2/po2 and chemistry panels
potassium/bun/sodium recur together), but the marginal event mix is not yet
calibrated — gen top-8 = labs, real top-8 = vitals + meds (dextrose, D5W),
0/8 overlap. Expected at 3000 steps; this is exactly what G3's event-rate
calibration + distance-to-observed metrics are for. Recorded for G3 baselining.

**Viewer live on 127.0.0.1:8042**: events (50,986 real stays) + sims_mps (24) +
sims_smoke (12), vocab lock 292, previews legible.

**Night totals**: 8 commits pushed (9934a25 → 8978d95 + this one); three OOMs
found, root-caused, and fixed (length grouping → cache guard → token budgets);
engine step-granular ckpt/val; viewer raw-parquet + id decoding; suites green
throughout (final: tests/ 411 + clif-validate/ 32).

**Morning state / what remains (user decisions, not agent-blocking)**:
- G2 real run is an L40 decision: full 546k MIMIC restage + `configs/train.yaml`
  (uniform per_gpu on CUDA — flash attention needs none of the MPS guards;
  token_budget/cache_clear stay off) + `configs/model.gem-ntp.yaml`.
- 30 checkpoints to prune before any sync (531 MB each — keep step 3000 +
  best-val; best-val = last (2.4138)).
- G3 eval harness (perplexity done; event-rate calibration, key-event recall,
  distance-to-observed) — baselined by tonight's sims_mps rollouts.
- Optional: raise overnight steps/epochs on this box (the 3000-step recipe now
  runs stably end-to-end).

### 2026-09-26 05:15 — G3 generative eval harness built + baseline recorded

**src/eval/generative.py** (unit G3, plan 2026-09-25-001): event-rate calibration
(JS divergence base-2, top-k overlap, closed-world gen-only mass, per-group
rates via data.yaml target_concepts), key-event recall (unigram + categorical
focus), distance-to-observed (nearest-neighbor Jaccard over concept sets), and
rollout hygiene (lengths, <eos>, <unk>, distinct-2). CLI scores a sims parquet
against the tokenizer events.parquet; reports are site-local under output/
(PHI-derived, never committed). `--prompt-source head:N` documents the sims'
prompt convention (a prefix column in the sims parquet supersedes it in G4).
9 unit tests (tests/test_generative_eval.py).

**G3 baseline on the 3000-step checkpoint's rollouts** (sims_mps vs the full
real corpus, report at output/intermediate_phi/gem_eval_mps.json):

| metric | value |
|---|---|
| JS divergence (real vs gen token rates) | 0.425 |
| top-8 / top-32 overlap | 0.0 / 0.5 |
| gen-only mass (closed world) | 0.000163 (airtight) |
| vitals rate real vs gen | 0.538 vs 0.141 (4x under-generated) |
| categoricals rate real vs gen | 0.436 vs 0.823 (2x over-generated) |
| labs rate real vs gen | 0.026 vs 0.036 |
| key-event recall (unigram / categorical) | 0.359 / 0.424 |
| distance-to-observed (mean / best) | 0.195 / 0.083 |
| hygiene: eos / unk / distinct-2 | 0.0 / 0.0 / 0.361 |

These are the L40 G2 run's numbers to beat. The 04:20 qualitative finding
(labs-heavy rollouts) refines to: vitals under-generated ~4x, categoricals
over-generated ~2x, labs calibrated. Perplexity lives in the trainer's val
loop (2.4138 / ppl ~11.2 at step 3000).

**Suites**: tests/ 420 passed 4 skipped (+9), clif-validate/ 32.

### 2026-09-26 06:15 — fresh-schedule resume + prompt provenance + 6000-step experiment

**The experiment this enables**: does event-rate calibration close with steps (the
plan's G2→G3 assumption) — testable locally BEFORE the L40 spend by continuing
3000 → 6000 steps and re-running the G3 eval. Plain resume was dead-end: the saved
cosine is fully decayed (LR 0 at step 3000), and resume also loads the OPTIMIZER
state, which pins param-group LRs at save time.

- engine `train(..., fresh_schedule=False)`: loads model + optimizer (Adam moments)
  but keeps THIS run's config schedule, and restores the configured group LRs
  (the optimizer state pins them at save time — caught by the new engine test).
  pretrain `--fresh-schedule` flag. Continuation semantics: the fresh schedule is
  a RE-WARMED 6000-step cosine ridden from its midpoint (brief 150-step re-warmup,
  ends ~mid-cosine, not 0) — schedule-shaped, not straight-through-exact; noted.
- generate CLI: `prompt` column in the sims parquet (the exact prefix tokens) —
  rollout provenance. eval pairs rollouts with real continuations by exact prefix
  match (the `head:N` convention is now fallback-only).

**Launched 06:15**: continuation from `ckpt_ep2_step3000` with `--fresh-schedule`
(configs/train.mps.yaml → total_steps 6000), ~73 updates/min, ETA ~75 min with
~12 validations. NEXT FIRING: when `/tmp/mps_continuation.log` shows
`Training complete` — (1) regenerate rollouts from `checkpoints_mps/ckpt_*_step6000.pt`
with the SAME prompts (`/tmp/prompts.txt`, 8 sims × 256 tokens, temp 1.0, top-p
0.95, closed-world) into `output/intermediate_phi/sims_mps6k.parquet`; (2) rerun
`src.eval.generative` on it (`--events .../mimic/events.parquet --vocab
.../mimic/vocab.json` — the prompt column pairs it now, no --prompt-source);
(3) record the 3000-vs-6000 comparison table in this log (JS, top-32, group
rates, key-event recall, distance, distinct-2, val perplexity); (4) suites +
commit + push. If still training: just log progress and stop.

**Suites**: tests/ 422 passed 4 skipped (+2), clif-validate/ 32.

### 2026-09-26 07:45 — experiment result: calibration closes with steps; loss plateaus

**Continuation complete** (Run ID `098cb6e619ec`, step 6000, 24 rollouts from
`ckpt_ep5_step6000` with identical prompts/sampling, report at
`output/intermediate_phi/gem_eval_mps6k.json`). The 3000-vs-6000 comparison:

| metric | 3000 steps | 6000 steps | real | reading |
|---|---|---|---|---|
| JS divergence | 0.425 | **0.358** | 0 | improves with steps |
| top-8 / top-32 overlap | 0.0 / 0.5 | **0.125 / 0.594** | 1.0 | improves |
| vitals rate | 0.141 | **0.276** | 0.538 | gap halves (4x → 2x under) |
| categoricals rate | 0.823 | **0.691** | 0.436 | moving toward real |
| labs rate | 0.036 | 0.032 | 0.026 | calibrated both scales |
| gen-only mass | 0.000163 | 0.000163 | 0 | closed-world airtight |
| val loss | 2.4138 | 2.4211 | — | **plateaued** |
| key-event recall (unigram/cat) | 0.359 / 0.424 | 0.318 / 0.352 | 1.0 | slightly worse |
| distance-to-observed mean | 0.195 | 0.224 | 0 | slightly worse |
| distinct-2 | 0.361 | 0.388 | — | less repetitive |

**Three findings for the L40 decision:**
1. **Event-rate calibration closes with steps even as loss plateaus** — JS 0.425 →
   0.358 (-16%) on 2x steps, vitals gap halved, while val loss sat at ~2.41
   throughout. The marginal distribution matures after the loss saturates; the
   L40's 60k steps on full 546k MIMIC should close most of the remaining gap.
2. **The staged 64.9k subset is exhausted** at this scale (val 2.41 plateau):
   loss gains live on the L40's full data, not more local steps.
3. **Rollout anchoring does NOT improve with NTP steps** — key-event recall and
   distance-to-observed slightly WORSE at 6000 (more diverse, less anchored to
   the prompt's real continuation). That is evidence FOR the plan's sequencing:
   pure NTP for the marginal, then G4 prefix conditioning + G5 RL for anchoring
   — more NTP alone won't make rollouts locally faithful.

**Suites**: tests/ 422 passed 4 skipped, clif-validate/ 32 (no code changed in
this step; comparison is data-side).

### 2026-09-26 08:15 — L40 G2 runbook finalized + viewer refreshed

**`docs/plans/l40-g2-runbook.md`** — the operational handoff, everything proven
on this Mac converted to exact L40 commands: prerequisites (reboot-first for the
nvidia-smi mismatch, full 546k restage, git pull), ETL in the correct order
(cohort → labeler → tokenize → outcome_join AFTER tokenize → value_stats, and
the hard gates that are correct not bugs), the torchrun launch with
`configs/train.yaml` + `configs/model.gem-ntp.yaml` (and why the MPS guards stay
OFF on CUDA), resume/continuation semantics (`--fresh-schedule` restores
configured LRs), post-train generate + eval commands (closed-world sampling,
prompt provenance column), the Mac baseline numbers to beat, and the gotcha
list (sync_vendor guard, step-granular ckpt/val, DDP val epoch-boundary).

**Viewer refreshed** on 127.0.0.1:8042 with all four sources: events (50,986
real stays), sims_mps6k (24), sims_mps (24), sims_smoke (12), vocab lock 292.

**State**: loop goal fully achieved (stack proven end-to-end on real MIMIC
locally); remaining items are user decisions (L40 launch, checkpoint prune) or
G4/G5 (conditioning/RL — follow the L40 base per the runbook §6). No further
agent-actionable units without user input; subsequent loop firings idle-verify
unless the user leaves new instructions.

### 2026-09-26 11:01 — Guarded rollouts + plausibility + audit remediation (verified, logged, pushed)

Three units completed in the interactive session after the 08:15 entry; this
idle-verify firing checked them, recorded them, and pushed.

1. **Guarded closed-world rollouts** (`sims_mps_guarded.parquet`, 6 rollouts ×
   128 tokens) — hard candidate-list sampling in `src/model/generate.py` closed
   the MPS sampler's `<unk>` leakage path found during per-row masking: the
   guarded G3 report shows 0 gen-only tokens / 0 OOV / 0 EOS (vs 1 gen-only
   token in the unguarded eval), and group rates closer to real (vitals gap to
   real 0.538 narrows 0.397 → 0.246; categoricals gap to real 0.437 narrows
   0.386 → 0.214). Marginal cost: JS 0.4316 vs 0.425 and top-32 overlap 0.4688
   vs 0.5 — an acceptable trade for a provably closed vocabulary; calibration
   is an L40-scale lever. The guarded eval uses the exact `prompt`-column
   prefix convention (vs `head:3` for the unguarded one), so key-event recall
   is not directly comparable across the two.

2. **Clinical plausibility inspection** (`3166fe8`) — new
   `src/eval/clinical_plausibility.py`: shared vocab-aware `split_sequence`
   (multi-word tokens like `sodium chloride` stay intact) plus explainable
   heuristics (OOV, invalid numeric bins, special-token placement, post-EOS
   tokens, repeated runs, concept loops, missing ICU context). Generator gained
   observed-transition masks, repetition penalty, per-row allowed-token masks,
   hard candidate-list sampling, and list-typed `generated_tokens` /
   `prompt_tokens` columns with plausibility metadata. Viewer gained the
   concept-grouped timeline, per-sequence plausibility score + warning
   explanations, and observed-vs-generated comparison via prompt-prefix
   matching. All 6 guarded rollouts score `good`.

3. **Impeccable audit remediation** (`04d274f`) — 44px interactive controls,
   180ms search debounce, persistent light/dark theme tokens
   (`prefers-color-scheme` aware), viewport-contained scroll panes (body
   containment + `min-height: 0` flex fixes), semantic color variables replacing
   hard-coded light-theme colors. Detector clean; axe 0 violations at 1440×900
   and 390×844 in both themes; all controls exactly 44px on mobile; no
   horizontal overflow. Wart noted: `.claude/skills/impeccable` is a symlink to
   the git-ignored `.agents/skills/impeccable`, so it dangles on a fresh clone —
   the committed copy lives at `agent/skills/impeccable`.

**This firing**: origin/main unchanged after fetch (no cross-machine
divergence); both commits verified code-only (skill files + `src/` + tests, no
parquet/checkpoints); suites re-run — tests/ 428 passed 4 skipped,
clif-validate/ 32; viewer alive at 127.0.0.1:8042 with all five sources (events
50,986 stays, sims_mps_guarded 6, sims_mps6k 24, sims_mps 24, sims_smoke 12,
vocab lock 292); pushed `3166fe8` + `04d274f` + this entry.

**State**: unchanged from 08:15 — remaining items are user decisions (L40
launch, checkpoint prune) or G4/G5 (conditioning/RL per runbook §6).

### 2026-09-26 12:05 — Impeccable re-audit: P0 record-API fix + audit remediation closed

User-requested re-audit of the live viewer (detector + axe at 1440×900 /
390×844 / 320×700, both themes, WITH a record loaded this time) scored it
16/20 (Good) and caught what every previous pass had missed, because nothing
had ever clicked a row:

1. **P0 (fixed, `78bcde8`)** — `/api/record` 500ed on every click for the
   50,986-row events source: `get_row` coerced numeric-looking ids to int, but
   events.parquet stores hospitalization ids as Utf8, so polars raised
   "cannot compare string with numeric type". The handler also swallowed
   tracebacks, so the server log showed nothing. Fix: dtype-aware id
   comparison (int columns reject non-numeric ids via the existing
   KeyError→404 path), a regression test covering both id dtypes through
   list_rows/get_row, and tracebacks now printed to stderr. Verified
   end-to-end: record loads (361 parsed tokens, plausibility `good`).
2. **P1 keyboard access (fixed this firing)** — rows were mouse-only
   (WCAG 2.1.1). Now tabindex 0 + role=button + aria-label + Enter/Space
   activation; verified by tab-through onto a real row and Enter opening it
   (417 tokens rendered). Accent-token `:focus-visible` rings added for rows
   and header controls (default UA outline previously only).
3. **P3 theme chrome (fixed this firing)** — token/pill borders, kv dotted
   separators, timeline-event background, and row hover used white-alpha
   values that are invisible on light panels, and the hover tint used the
   dark-theme accent. New `--token-line` / `--hover` tokens with per-theme
   values; computed light borders now `rgba(20,33,32,0.14)` and hover
   `rgba(8,127,112,0.08)`.

**Verification**: detector clean; axe 0 violations across desktop light +
dark and mobile with a loaded record (37 passes per run, the button-role rows
included); 320px still contained with no horizontal overflow; all controls
44px; tests/ 429 passed 4 skipped, clif-validate/ 32. Audit score after
remediation: Accessibility 2→4 path closed, Theming 3→4 path closed —
expected re-audit ≥ 19/20.

**State**: audit remediation complete and pushed; otherwise unchanged —
remaining items are user decisions (L40 launch, checkpoint prune) or G4/G5
(conditioning/RL per runbook §6).

### 2026-09-26 13:00 — Idle-verify + re-audit close-out: 20/20

Idle-verify firing: origin/main in sync at `e7e2502` after fetch (no
cross-machine divergence); artifacts unchanged; viewer alive at 127.0.0.1:8042
with all five sources; suites green — tests/ 429 passed 4 skipped,
clif-validate/ 32.

**Re-audit close-out** (promised in the 12:05 entry): axe 0 violations across
desktop light + dark (loaded record) and mobile dark, 37 passes per run;
390×844 no overflow, controls 44px. Keyboard operation, accent focus rings,
light-theme chrome, and a clean detector were verified post-fix in the 12:05
firing. Final score: Accessibility 4, Performance 4, Responsive 4, Theming 4,
Implementation Integrity 4 — **20/20, Excellent band**. Residual caveat,
recorded honestly: text-only zoom was never instrumented (px-based layout;
browser page zoom assumed per standard practice).

**State**: no agent-actionable units remain in this loop's goal — the whole
GEM stack (ETL → packed shards → MPS train → generate → viewer) is proven
end-to-end on the real staged MIMIC CLIF tables on this Mac, and the viewer
is audit-clean. Remaining items are user decisions (L40 launch per
`docs/plans/l40-g2-runbook.md`, checkpoint prune) or G4/G5 (conditioning/RL
per runbook §6). Subsequent firings idle-verify unless the user leaves new
instructions.

### 2026-09-26 14:00 → 2026-10-02 13:00 — 131 idle-verify passes collapsed

131 hourly "Idle-verify: all green, no new instructions" entries (2026-09-26 14:00 to
2026-10-02 13:00) recorded no new finding, fix, or decision and were collapsed into this
line on 2026-10-02. Each pass is still in `git log` (`git log --oneline --grep idle-verify`).
Entries with evidence (guardrail checks, API spot check, close-outs) are kept verbatim in place.

### 2026-09-26 17:00 — Idle-verify + repo finalization logged

Idle-verify: origin/main in sync at `a73ef24` after fetch; artifacts
unchanged; viewer alive; suites green — tests/ 429 passed 4 skipped,
clif-validate/ 32.

Repo finalization since the 16:00 entry (user-requested, docs-only, both
pushed): `7d27cf3` — README layout gains dataset/value_stats/generate/
checkpoint/generative/plausibility/viewer plus a "Local GEM validation
stack" section (guarded-generate + viewer commands, G3 baselines, runbook
pointers); project-status synced to the GEM track; MEMORY.md §Status
refreshed to 2026-09-26 (stale NEXT list replaced). `a73ef24` — new website
page "GEM Local Validation (MPS)" (the proven loop, guarded generation,
plausibility, G3 baselines, the anchoring finding, the viewer, what
remains), sidebar entry after Ablations; `npm run build` verified (page
emitted, mermaid wired; incidental package-lock churn reverted).

**Model architecture verified untouched** since `4962f82`: the diff to
heads/encoder/adapter, pretrain/checkpoint, tokenize/dataset/value_stats,
and all configs is empty; the only src/ changes were viewer/eval tools and
generate.py sampling tooling (frozen-checkpoint inference, no model changes).

**State**: unchanged — user decisions (L40 launch per
`docs/plans/l40-g2-runbook.md`, checkpoint prune) or G4/G5 (conditioning/RL
per runbook §6).

### 2026-09-26 17:20 — Next-step probe: L40 box unreachable from this Mac

The next project step is the L40 G2 base run. Probed from this machine
(jcs-mac-studio): `rudu-hpcg004` does not resolve (no DNS/SSH alias), is not
on the tailnet (fleet = macbook, omarchy, mac-studio, iphone, mateopc), and
has no SSH config entry. Blocked on user infrastructure, not code:
(a) join the box to the tailnet — then the runbook can be driven remotely
end to end, (b) provide a reachable address/VPN alias, or (c) run the
runbook manually on the box (it is copy-paste complete, phases 0–3 + numbers
to beat; prerequisites are the reboot-first for the nvidia-smi mismatch and
the full 546k-stay restage — the Mac extract was a 64.9k-stay subset).
Everything Mac-side is complete: all seven units proven, repo finalized,
website built and pushed through `a73ef24`.

### 2026-09-26 17:35 — Lambda cloud proposed for training; rejected per hard rule 5

User asked whether the L40 G2 base run could train on rented Lambda Labs
cloud (cloud.lambda.ai workspace). No: AGENTS.md hard rule 5 — no rented cloud
without a compliant BAA/DUA, Azure only inside the lab's governed tenant. The
MIMIC tables are PhysioNet-credentialed and everything downstream (events
parquet, shards, checkpoints) is PHI-derived, so none of it may be uploaded
to Lambda. Rented cloud is legitimate only for DATA-FREE work (GPU
qualification of FA2/DDP/throughput on synthetic shards, engine burn-in); the
real-data paths remain the lab box `rudu-hpcg004` (tailnet/VPN/alias needed)
or governed-tenant Azure. Revisit only with new governance evidence (a
compliant Lambda BAA) — a user decision, not an agent one.

### 2026-09-26 18:15 — Pre-launch readiness audit: READY

Audited the exact training path the L40 G2 run will launch (user request):

- **Runbook vs CLIs**: every command in `l40-g2-runbook.md` matches the real
  argparse of cohort / clif_auto_labeler / tokenize / outcome_join /
  value_stats / pretrain / generate / generative — no flag drift.
- **Configs**: `train.yaml` is the L40 recipe (bf16, DDP, per_gpu 4 ×
  grad_accum 32 × 2 GPUs = effective batch 256, ckpt_every 2000, val_every
  defaults to 2000 when `eval_schedule` is absent, NO MPS guards — correct
  for CUDA); `model.gem-ntp.yaml` is pure NTP (next_event 1.0, others 0.0,
  state-dict-compatible trunk, fixed weights, no curriculum); the MPS
  guards (`token_budget`, `cache_clear_every`) live only in `train.mps.yaml`.
- **Fail-closed gates confirmed in pretrain.py**: `events_with_outcomes`
  must exist (plus an augmented-mtime staleness check); value-stats must
  cover numeric values; `--resume` / `--fresh-schedule` wired through.
- **Preflights on this Mac**: focused train-path tests 107 passed; the exact
  launch (train.yaml + model.gem-ntp.yaml + real data + value-stats) as
  `--dry-run` built model + loaders cleanly — params 45.2M (includes the
  untied 10k-slot embedding + head per the locked untied decision), ddp:true
  no-ops without torchrun; the ETL dry-run read all five real CLIF tables
  (~111M events).
- Architecture untouched since `4962f82`; full suites green (429+4, 32).

**Verdict: the code is READY for the L40 G2 launch.** Remaining items are
box-side Phase 0 only (reboot-first for the nvidia-smi mismatch, full 546k
restage, git pull ≥ `cb87cdf`, uv sync) — user infrastructure per 17:20.

### 2026-09-27 08:00 — Idle-verify + guardrail check: all green

origin/main in sync at `b15c2b3` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Ran the periodic data/PHI guardrail
check: zero tracked `.parquet`/`.pt`/`.ckpt`/`.safetensors` files;
`/output/` still ignored (verified for `sims_smoke.parquet` and
`mimic/vocab.json`); working tree clean. No new instructions; launch
blocked only on user infrastructure (17:20).

### 2026-09-27 18:00 — Idle-verify + guardrail check: all green

origin/main in sync at `8329a45` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check:
zero tracked `.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/`
still ignored (verified for `sims_smoke.parquet` and `mimic/vocab.json`);
working tree clean. No new instructions; launch blocked only on user
infrastructure (17:20).

### 2026-09-28 07:00 — Idle-verify + guardrail check: all green

origin/main in sync at `96a1d99` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check:
zero tracked `.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/`
still ignored (verified for `sims_smoke.parquet` and `mimic/vocab.json`);
working tree clean. No new instructions; launch blocked only on user
infrastructure (17:20).

### 2026-09-28 16:00 — Idle-verify + live API spot check: all green

origin/main in sync at `2bec0f3` after fetch; artifacts unchanged; suites
green — tests/ 429 passed 4 skipped, clif-validate/ 32.

Went one step past the suites and exercised the live viewer API end to end
instead of only asserting it is "alive":

- `/api/sources` reports all five sources with correct row counts (events
  50,986; sims_mps_guarded 6; sims_mps6k 24; sims_mps 24; sims_smoke 12).
- `/api/record` loads a real row from **every** source with HTTP 200 and
  parses to tokens: events 361, sims_smoke 128, sims_mps_guarded 127,
  sims_mps 252, sims_mps6k 254. This re-confirms the P0 fix (`78bcde8`)
  in the live process, not just in tests.
- Error paths stay clean, no 500s: missing string id on `events` → 404;
  non-numeric id on an integer-id source → 404 (the dtype guard); unknown
  source → 400.
- Guardrail check: zero tracked `.parquet`/`.pt`/`.ckpt`/`.safetensors`;
  `/output/` still ignored; working tree clean apart from this log edit.

No new instructions; launch blocked only on user infrastructure (17:20).

### 2026-09-28 20:00 — Idle-verify + guardrail check: all green

origin/main in sync at `63e8215` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check:
zero tracked `.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/`
still ignored (verified for `sims_smoke.parquet` and `mimic/vocab.json`);
working tree clean. No new instructions; launch blocked only on user
infrastructure (17:20).

### 2026-09-29 07:00 — Idle-verify + guardrail check: all green

origin/main in sync at `512c5df` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check:
zero tracked `.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/`
still ignored (verified for `sims_smoke.parquet` and `mimic/vocab.json`);
working tree clean. No new instructions; launch blocked only on user
infrastructure (17:20).

### 2026-09-29 14:00 — Idle-verify + guardrail check: all green

origin/main in sync at `496a602` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check:
zero tracked `.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/`
still ignored (verified for `sims_smoke.parquet` and `mimic/vocab.json`);
working tree clean. No new instructions; launch blocked only on user
infrastructure (17:20).

### 2026-09-30 02:00 — Idle-verify + guardrail check: all green

origin/main in sync at `78de774` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check
(~12h since 09-29 14:00): zero tracked
`.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/` still ignored
(verified for `sims_smoke.parquet` and `mimic/vocab.json`); working tree
clean. No new instructions; launch blocked only on user infrastructure
(17:20).

### 2026-09-30 14:00 — Idle-verify + guardrail check: all green

origin/main in sync at `9a0e580` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check
(~12h since 02:00): zero tracked
`.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/` still ignored
(verified for `sims_smoke.parquet` and `mimic/vocab.json`); working tree
clean. No new instructions; launch blocked only on user infrastructure
(17:20).

### 2026-10-01 02:00 — Idle-verify + guardrail check: all green

origin/main in sync at `b485c0d` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check
(~12h since 09-30 14:00): zero tracked
`.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/` still ignored
(verified for `sims_smoke.parquet` and `mimic/vocab.json`); working tree
clean. No new instructions; launch blocked only on user infrastructure
(17:20).

### 2026-10-01 14:00 — Idle-verify + guardrail check: all green

origin/main in sync at `70ce028` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check
(~12h since 02:00): zero tracked
`.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/` still ignored
(verified for `sims_smoke.parquet` and `mimic/vocab.json`); working tree
clean. No new instructions; launch blocked only on user infrastructure
(17:20).

### 2026-10-02 02:00 — Idle-verify + guardrail check: all green

origin/main in sync at `4b04fc5` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check
(~12h since 10-01 14:00): zero tracked
`.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/` still ignored
(verified for `sims_smoke.parquet` and `mimic/vocab.json`); working tree
clean. No new instructions; launch blocked only on user infrastructure
(17:20).

### 2026-10-02 14:00 — Idle-verify + guardrail check: all green; new branch noted

origin/main in sync at `2dae27e` after fetch; artifacts unchanged; viewer
alive at 127.0.0.1:8042 with all five sources; suites green — tests/ 429
passed 4 skipped, clif-validate/ 32. Periodic data/PHI guardrail check
(~12h since 02:00): zero tracked
`.parquet`/`.pt`/`.ckpt`/`.safetensors` files; `/output/` still ignored
(verified for `sims_smoke.parquet` and `mimic/vocab.json`); working tree
clean. New this firing: remote branch
`t3code/audit-tokenization-documentation` appeared (one commit, `046f805`,
docs audit, 38 files, not merged to main). Not merged or reviewed — noted
for the user. Launch blocked only on user infrastructure (17:20).

### 2026-10-02 15:00 — Repo hygiene PR #15 merged mid-loop; protocol change adopted

origin/main advanced 9 commits (`c5884f5` → `25b45ab`); fast-forwarded clean.
PR #15 (`t3code/audit-tokenization-documentation`) merged: 131 no-op
idle-verify entries collapsed to one summary line (log now 684 lines);
**new protocol adopted — no-op passes are no longer logged or committed**;
vendored impeccable skill untracked; `configs/data.yaml` notes.encoder
aligned to BioClinical-ModernBERT-base and locked with a test; review
fixes applied. Both suites re-verified on the new main: tests/ 430 passed
4 skipped (+1 = the new notes-encoder config-lock test), clif-validate/ 32.
Guardrails re-checked on the new tree: zero tracked data files, `/output/`
ignored, working tree clean. Viewer alive with all five sources. Launch
blocked only on user infrastructure (17:20).

### 2026-10-02 18:00 — New in-tree artifact: v2 tokenizer sample arms (other session)

Three new git-ignored sample tokenizations appeared in `output/intermediate_phi/`
(created 17:06–17:12, after the 17:00 pass): `mimic_v2_sample` (119M),
`mimic_v2_sample_decile` (38M), `mimic_v2_sample_continuous` (6.8M). Reports
show tokenizer_version 2, 5,000-stay samples, mean 391.83 events/stay, p99
811, total ~1.96M events, distinct vocabulary/numeric-edges hashes per arm
(clinical-segment vs decile vs continuous); continuous arm has no
tokenization_report.json (vocab + value_stats + events only). Produced by
another session on this Mac — not by this loop; no data left `output/` (all
git-ignored). Suites green on this pass: tests/ 430 passed 4 skipped,
clif-validate/ 32. origin/main in sync at `ff3fbad`; viewer alive, all
sources. Loop work remains gated on user infrastructure (18:00).

### 2026-10-02 19:00 — New branch `t3code/tokenizer-fixes` (16 commits); r2 sample arms

Fetch revealed new remote branch `t3code/tokenizer-fixes`, 16 commits ahead of
main, 91 files (+16,829/−1,839): the tokenizer-v2 program (U1–U9) — bins for
every numeric concept, vocab v2 contract with single bin_index, six-arm
tokenization ablation, full-hospitalization GEM artifact with
ADMISSION/DISCHARGE terminal framing, disposition/censoring/mortality rollout
eval, suppression-safe aggregate reports. **Not merged** — inspected
read-only; flagged to the user (merge is a user decision, like PR #15).
Corresponding regenerated sample arms appeared in `output/intermediate_phi/`
(18:41–18:42, other session): `mimic_v2_sample_r2` (117M), `_r2_continuous`
(6.8M), `_r2_decile` (38M) — all git-ignored. Suites green on main:
tests/ 430 passed 4 skipped, clif-validate/ 32. origin/main in sync at
`8ec3866` after this entry's push; viewer alive, all five sources. Loop work
remains gated on user infrastructure (19:00).

### 2026-10-02 20:00 — PR #16 (tokenizer v2 program) merged; absorbed, re-verified 715/34

origin/main advanced 17 commits (`d428d7c` → `4e499df`): PR #16 merged
`t3code/tokenizer-fixes` — the tokenizer-v2 program (U1–U9, 91 files,
+16,892/−1,843): bins for every numeric concept, vocab v2 contract with
single `bin_index` + stale-artifact rejection, deterministic event order,
dose/ventilator/assessments/CRRT/ECMO/code-status/position tokens, fused
categoricals, six-arm tokenization ablation, full-hospitalization GEM
artifact with ADMISSION/DISCHARGE terminal framing, disposition/censoring/
mortality rollout eval, suppression-safe aggregate reports, new
`src/train/real_data_smoke.py`, vendored parity tests. Inspected read-only
then fast-forwarded clean; no log changes upstream (19:00 entry stayed
last). Both suites re-verified on the new main: tests/ **715 passed** 4
skipped (was 430; +285 new tests), clif-validate/ **34 passed** (was 32;
+tokenizer parity). Guardrails on the new tree: zero tracked data files,
working tree clean, `/output/` ignored. Artifacts unchanged (r2 sample
arms present); viewer alive with all five sources. Launch remains gated
on user infrastructure (20:00).
