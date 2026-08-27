# Integration Plan — Build ON CLIFATRON (not parallel to it)

**Decision (2026-08-27):** CLIFATRON (github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON,
MIT, PyPI `clifatron`) is the consortium's working CLIF-native ICU FM — built by our lab's
data scientist, and **we control it**. We build our RESEARCH.md methods ON it rather than
rebuild a parallel model. Our contribution = the objective + multimodal + federated-eval
upgrade CLIFATRON lacks.

## What CLIFATRON already has (KEEP — do not rebuild)
- **`tokenETL/`** — 9-domain tokenizer on `clifpy`, ~1,284-token vocab, fused clinical-range
  value tokens (`lab_creatinine_0.7_to_0.9`), narrative assembly with `day_N`/`hour_N` markers.
- **`AR/{gpt2_hf,qwen2}/`** — HF GPT2 (12M–355M) + Qwen2 (0.5B, RoPE/GQA/SwiGLU/RMSNorm),
  8192 ctx, sequence packing, DeepSpeed ZeRO, Optuna, L40 guides.
- **`benchmark/`** — 4 tasks (discharged-home, LTACH, 72h outcome, hypoxic proportion),
  ~30k stays / ~85M tokens; Method 1 = frozen-emb→XGBoost, Method 2 = MC rollout.
- Models are standard HF → expose per-token hidden states via `output_hidden_states=True`
  with an `attention_mask` (see `benchmark/utils/model_loader.get_model_embeddings`).

## What we ADD (our RESEARCH.md methods — the keeper code)
1. **Survival/threshold heads on CLIFATRON's backbone** (`src/model/heads.py` — already written):
   `ThresholdHazardHead` (ICareFM zero-shot), `CompetingRiskHead` (SurvivEHR), `ValueRegressionHead`
   (ORA mark). They consume `H_t` = CLIFATRON hidden state at the anchor position (hour-24 token).
2. **Multi-objective training**: joint next-token (their loss) + our threshold/CR/value losses on
   the same backbone. Two entry points — (a) fine-tune a released CLIFATRON checkpoint with heads
   attached; (b) pretrain from scratch with the mixed objective (curriculum: NTP → +TTE heads).
3. **Multimodal branch**: frozen BioClinical-ModernBERT note embeddings injected as timestamped
   tokens in the narrative stream (pre-anchor only).
4. **Federated eval**: `src/eval/matrix.py` (already written) — task×site matrix, adaptation ladder
   (as-is/recalibrate/finetune), LPE, inference-time ensembling; metric panel AUROC/AUPRC/ECE/
   Brier/calib-slope/ICI/DCA + subgroup fairness + TRIPOD+AI.

## What we DROP / demote from our scaffold
- `src/model/encoder.py` (our flat Llama trunk) — **superseded** by CLIFATRON's Qwen2 backbone
  (same family: RoPE/SwiGLU/RMSNorm). Keep only if we run the from-scratch ablation arm.
- `src/data/tokenize.py` (our duckdb tokenizer) — **superseded** by `tokenETL`. Keep only as the
  **decile arm** of the tokenization ablation (below).

## Wedge deliverable — "Method 3: calibrated survival heads" (smallest publishable unit)
Reuse CLIFATRON's 4-task benchmark unchanged. Take a trained CLIFATRON checkpoint, extract `H_t`
at the anchor, attach our heads, and show, on their own tasks:
- **Better AUPRC + calibration than Method 1** (XGBoost-on-embeddings) — imbalanced tasks (LTACH
  3.6% positive) are where survival heads should win.
- **Cheaper + calibrated vs Method 2** (MC rollout) — threshold-query = one forward pass, no
  simulation variance (Bedi/Fries/Shah).
- **Zero-shot** for tasks expressible as a threshold-crossing (hypoxia, organ-failure composites).
This proves the objective thesis with ~none of the infra cost, on the consortium's benchmark.

## Publishable ablation we can own
CLIFATRON bins values into **clinical reference ranges**; Lee et al. 2026 (on MIMIC-IV-Ext-CLIF)
found **empirical deciles** the best default. Run both tokenizations through the same backbone +
tasks → a clean, CLIF-native tokenization ablation nobody has published.

## Immediate next code tasks
1. `src/model/head_adapter.py` — wrap a HF CLIFATRON checkpoint; forward returns `H_t` at a given
   anchor index (+ optional joint NTP logits) for our heads. Interface matches `attention_mask`.
2. `benchmark/method3-survival/` (in the CLIFATRON fork) — driver that runs the wedge above.
3. Add our metric panel to `benchmark/utils/metrics.py` (ECE/Brier/calib-slope/ICI/DCA/LPE/subgroup).
4. Anchor definition: the hour-24 token position in the narrative (their benchmark already truncates
   to first 24h) — locate it from the `day/hour` markers.

## Sites & federated design — RESOLVED (2026-08-27): develop on 3, validate on the CLIF federation
- **Development cohort (data we hold):** MIMIC + Rush + UChicago. Freeze CLIFATRON mCIDE vocab
  across all → identical token space everywhere, ensemble-able, turnkey at any CLIF site.
- **Internal 3×3 transportability matrix**: rows=train site, cols=test site; diagonal=internal,
  off-diagonal=external transport. AUROC/AUPRC + calibration slope per cell. Ensemble column =
  mean of the 3 site models' probs (Elemento). Adaptation ladder per off-diagonal cell
  (as-is → temperature recalibrate → finetune), LPE each rung.
- **External validation = model-to-data across ALL OTHER CLIF sites.** Ship a self-contained
  package: frozen checkpoint + clifpy/tokenETL eval script. Each site runs it on its LOCAL CLIF
  tables, auto-labels outcomes from standard CLIF fields, runs ZERO-SHOT threshold/CR heads
  (no local training/labels needed), returns ONLY aggregate + subgroup metrics. Nothing raw leaves.
- **Headline figure:** forest/box plot of AUROC/AUPRC/calibration across N external CLIF sites per
  outcome. This is the "one node → many hospitals" result and the axis CLIFATRON can't claim.
- Restrict outcomes to CLIF-derivable labels (mortality, disposition/home/LTACH, IMV on/off,
  hypoxia, organ-failure thresholds) so external sites need no manual annotation.

### New deliverable: `clif-validate/` — the shippable external-validation package
A standalone runner other CLIF sites execute against their `clif_config.json`:
tokenize (tokenETL) → load frozen checkpoint → zero-shot threshold/CR predictions →
auto-label from CLIF tables → compute metric panel (AUROC/AUPRC/ECE/Brier/calib/DCA/LPE + subgroup)
→ emit ONLY a metrics JSON. Must be pip-installable, GPU-optional, and depend only on public code
(clifpy, transformers, our heads) + the released checkpoint — no dev-cohort data.

## Open items to verify on the CLIFATRON side
- Benchmark leakage check: retrospective tokens (disposition) must not appear in the 24h input.
- storetime vs charttime ordering inside `tokenETL` (our hard rule 4).
- Which sites is the current CLIFATRON checkpoint trained on? (We want the 3-site split explicit.)
