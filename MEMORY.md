# CLIF Foundation Model — project memory

Global memory: ~/.claude/CLAUDE.md (rules) + ~/.claude/memory/memory.md (index).

## What this is
Compact (~30M-param) multimodal ICU foundation model on **federated CLIF 2.1**
(dev cohort = Site 1 + Site 2 + Site 3; **only Site 1 is currently staged on the L40
box** — see LOCKED DECISIONS G2), pretrained with a **threshold-conditioned
time-to-event** objective. Thesis: one small model → many outcomes → many
hospitals → one node (2× L40, no cluster). Nature-Medicine framing: efficiency +
federated-fairness + CLIF-native + deployable multimodal SLM.

## PIVOT 2026-08-27 — build ON CLIFATRON (see notes/INTEGRATION.md)
**CLIFATRON** (github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON, MIT, PyPI
`clifatron`) is the consortium's WORKING CLIF-native ICU FM, built by our lab's data
scientist (vchaudha) — **we control it**. It already occupies "compact CLIF-native ~30M AR
ICU FM" (GPT2 12–355M + Qwen2 0.5B; tokenETL on clifpy; 4-task benchmark; ~30k stays/85M
tokens; L40 training). So that axis is NO LONGER our novelty. DECISION: build our methods ON
it, not parallel. KEEP their tokenETL+packing+DeepSpeed+benchmark. ADD our objective
(threshold-TTE + competing-risk + value heads — CLIFATRON is pure next-token, the weakest
objective per ORA/MOTOR/ICareFM), notes modality, federation + calibration/LPE/fairness eval.
KEEPER code = heads.py + eval/matrix.py. SUPERSEDED = our encoder.py (use their Qwen2) +
tokenize.py (use tokenETL; keep only as decile arm of a tokenization ablation vs their clinical
bins). Wedge deliverable = "Method 3": attach our survival heads to a CLIFATRON checkpoint's
hidden states, beat their Method 1 (XGBoost-on-emb) on AUPRC/calibration + Method 2 (MC rollout)
on cost/calibration, on their own benchmark.

## Design spec — REVISED 2026-08-27 (full synthesis in notes/RESEARCH.md)
Five-thread deep research + 2 focused 2026-preprint threads (tokenization a76bb9 / architecture aeb4d2) → these changes:
- **Tokenizer:** FUSED single token `concept=bin` (settled: Lee 0.891→0.915, Guo 73/74 tasks −39.5% FLOPs);
  DECILE bins frozen from reference site (NOT clinical bins — Lee: no gain from ref-range anchoring);
  SOFT DISCRETIZATION (wins the dangerous tails); forced clinical-threshold edges as a constraint;
  admission-relative-MINUTE RoPE (drop day_N/hour_N tokens, −11% seq); storetime ordering; unit-normalize.
  Continuous-fused value channel (McCann) = ablation (worse calibration). configs/data.yaml updated.
- **Trunk:** KEEP Qwen2/Llama transformer — do NOT switch to Mamba (arch thread: within-noise at 8k/30M,
  objective>backbone per ORA). FLAT decoder (RMSNorm/**time-aware RoPE**/SwiGLU), d512 × 8 layers × 8 heads,
  context 8192 (match CLIFATRON trunk; 4096 = ablation), **UNTIED embeddings** (+4-7% AUPRC, gap widens under
  federation), target vocab ~10k (untied+large-vocab budget tension → cap vocab). configs/model.yaml updated.
  Reconcile: next_event_loss in heads.py ties LM head to input emb → needs separate output proj when untied.
  New eval to add: competing-risk D-calibration / Aalen-Johansen K-cal (arXiv:2602.00194).
- **Objectives:** primary = threshold-hazard (ICareFM) + competing-risk CIF (SurvivEHR);
  **ENABLED value-regression** (ORA "mark", arXiv:2602.00541, +33-38% on physiology tasks);
  next-event demoted to low-weight (0.2) auxiliary. Uncertainty+gradnorm loss balancing; NTP→TTE curriculum.
- **Multimodal (v2):** BioClinical ModernBERT-base frozen; inject per-note embeddings as
  timestamped event tokens (pre-anchor only). Oversized note gain = leakage flag.
- **Eval:** CLIF→MEDS ETL (exists, consortium) → MEDS-Tab XGBoost baseline (mandatory) + MEDS-DEV;
  task×site matrix + adaptation ladder (as-is/recalibrate/finetune) + LPE; panel AUROC/AUPRC/ECE/
  Brier/calib-slope/ICI/DCA; TRIPOD+AI. Comparators: ICareFM (DUA), MOTOR+CLMBR-T (CLIF→OMOP→FEMR).
- **Novelty:** 4-way intersection (CLIF-native × ICU structured+notes × ~30M × federated × threshold-TTE).
  **REVISED 2026-08-28:** the "CLIF v3.0 ships ~July 2026 → first-mover window, move now" timing argument is
  RETIRED — that window has passed. Contribution rests on CLIF-native execution, openly released validation
  tooling, and real-federation deployment; none requires being first. The empirical priority claim (first
  OPEN CLIF-native ICU FM with threshold-TTE validated by real model-to-data) is unaffected and still stands.

## Method sources (see notes/METHODS.md for line-cited detail)
- **ICareFM** (Burger/Rätsch, medRxiv 2025) — per-target hazard head, random τ, direction+threshold
  embeddings, discrete hazard 48h; treatments INPUTS not targets; composite via conj/disj (cond. indep).
- **SurvivEHR** (medRxiv 2025) — competing-risk CIF; Gaussian value head.
- **Elemento** (medRxiv 2026) — inference-time ensembling ≈ federated, lighter governance; skip FedProx.
- **Cadence** (medRxiv 2026) — small-beats-big; temperature scaling; dual-sex TRIPOD+AI reporting.

## Sites & federated design (2026-08-27) — DEVELOP on 3, VALIDATE on the whole CLIF federation
DEVELOPMENT cohort (data we hold): Site 1 + Site 2 + Site 3. EXTERNAL VALIDATION:
ALL OTHER CLIF consortium sites via model-to-data — ship the frozen model + a turnkey clifpy/
tokenETL eval script; each site runs it on its LOCAL CLIF tables and returns ONLY aggregate
metrics. No raw data, labels, or gradients ever leave any node. This IS the thesis
("one node → many hospitals") and the axis CLIFATRON (single-center) lacks.
- **Why our objective (not CLIFATRON's) enables this:** threshold-hazard/competing-risk heads are
  ZERO-SHOT → a new site needs **no local model training and no manually-annotated labels** to get
  calibrated predictions. CLIFATRON's pure-AR path needs per-task XGBoost-on-embeddings (which
  requires local labels to *fit*) or expensive MC rollout. The shippable, **training-free** zero-shot
  model is the federated-validation argument.
  **PRECISION (do not overclaim):** "label-free" refers to the MODEL — it emits predictions with no
  local fitting. **Computing the metrics (AUROC/AUPRC/ECE/calibration) still requires ground-truth
  labels**, which each site auto-derives from its own CLIF tables (below). So the pipeline is
  *no-local-training + no-manual-annotation*, NOT "no labels at all." The auto-labeler's definitions
  are themselves a validity dependency — audit them per site.
- **Labels auto-derived locally (for evaluation only):** the auto-labeler derives **incident physiologic
  threshold-crossings** from standard CLIF vitals/labs (implemented: `map_below_65_48h`,
  `lactate_above_4_48h`, `spo2_below_88_48h` — see `configs/cohort.yaml → outcomes`;
  `src/eval/clif_auto_labeler.py → derive_outcome_states`). Each is a `{concept, direction, threshold,
  horizon}` crossing, aligning outcomes with the threshold-hazard head's zero-shot query. **Treatments
  (IMV, vasopressors) are NOT outcomes — inputs only (Rule 1); `tests/test_data_config.py` asserts it.
  Mortality enters only as the competing-risk death event.** Roadmap: add more organ-failure cutpoints
  (creatinine/KDIGO, bilirubin, platelets) as coverage allows. These derived labels are noisy phenotypes
  that vary by site coding; report each outcome's definition + provenance alongside its metrics.
- **Vocab:** CLIFATRON frozen mCIDE across ALL sites → turnkey everywhere, no refitting.
- **Governance:** only aggregate + subgroup metrics return; fairness reported aggregate (ICareFM precedent).
- **Internal eval (3 held sites):** 3×3 train-A/test-B matrix + adaptation ladder
  (as-is/recalibrate/finetune) + LPE + Elemento ensemble column.
- **HEADLINE FIGURE:** forest/box plot of AUROC/AUPRC/calibration across N external CLIF sites per
  outcome — the "does it travel" result. Site 3 = CLIF origin site (Bhavani).

## STEERING PRINCIPLE (2026-08-27) — clinically derived: encode "good vs bad for doctors"
> **⚠️ BINNING DECISION BELOW REVERSED 2026-09-02 — see LOCKED DECISIONS §E1a.** This section argued
> for population deciles as the base scheme (Lee 2026: deciles ≈ ref-range). We have since made
> **physician-designed clinical segments the PRIMARY scheme** on clinical-relevance grounds, with
> deciles as the `decile_ablation` arm that measures the two head-to-head. The GOAL (sensitive where
> danger is, legible to a doctor) is unchanged; only the mechanism was reversed. §E1a is authoritative.
> Every bullet below that argues for deciles as the correct choice is superseded by the 2026-09-02 reversal.

GOAL (unchanged, correct): the model must be most sensitive where clinical DANGER is, and be legible
to a doctor. But the MECHANISM matters — the 2026 evidence (both research threads landed) says hard
clinical bins are NOT how you achieve it. CORRECTED tokenization decision (SUPERSEDED 2026-09-02):
- **Base bins = population DECILES** — superseded. We now use physician-designed clinical-segment bins as primary.
- **Soft discretization** — retained. Applied on top of clinical segments for boundary sensitivity.
- **Clinical thresholds as a CONSTRAINT** — retained. Forced edges (lactate 2/4, MAP 65, etc.) guaranteed on the clinical-segment grid.
- **Deciles transport** — academic finding, superseded by clinical-team judgment that domain-expert bins are more important than transportability.
- **Continuous-fused value channel (McCann 2026.08.04) = ABLATION, not default:** +30% numeric acc, −34%
  seq len, but WORSE calibration than discrete; calibration is our headline. Naive unfused xVal collapses
  to median (Lee) → never. This resolves the previously-open value-channel question: discrete+soft ships.
- Principle still governs the rest: outcomes = states doctors act on (not treatments); threshold heads
  DIRECTIONAL (crossing into danger); eval headline = net benefit/DCA (clinical good/bad, not just AUROC).

## Hard rules (do not violate)
1. Treatments = model inputs, NEVER prediction targets.
2. Vocab = CLIFATRON frozen mCIDE, applied identically to all 3 sites — no cross-site pooling of raw data.
3. Retrospective reports/discharge summaries = LABEL source only; only pre-anchor notes may be features.
4. Site 1 is governance-credentialed; Site 2 + Site 3 institutional — none leave their node.
   Compute = 3 tiers: MacBook (dev, MPS), 2× L40 Linux box (default training, DDP), and Azure hourly GPU
   (burst) — Azure ONLY inside a BAA/DUA-covered lab tenant (never an ad-hoc personal sub for real PHI).

## LOCKED DECISIONS (2026-08-27, grounded in 2026 literature via Paperclip) — do not re-litigate
These resolve every open design question as of this date. Change only with new evidence.

**A. Novelty & positioning (LOCKED)**
- A1. **Our novelty is CLIF-native + open-weights + real model-to-data federation — NOT a new method.**
  The 2026 literature owns every *method* piece: ORA (arXiv:2602.00541) owns marked-TTE; ICareFM
  (medRxiv 2025.07.25.25331635) owns threshold-directional dual-zero-shot BUT on **ricu, not CLIF**, and
  its **weights are DUA-gated**; SurvivEHR owns competing-risk (UK primary care, not ICU); EveryQuery/ETHOS
  own query-conditioned zero-shot; Elemento owns no-data-sharing ensembling (on Site 1 partitions, not real
  hospitals). **Unclaimed = the first OPEN, CLIF-native ICU FM with a threshold-TTE objective validated by
  model-to-data across REAL CLIF-consortium hospitals.** Novelty = integration + open tooling + deployment,
  which is the stronger axis for a Nature-Medicine clinical framing. Do NOT claim method invention.
  (2026-08-28: "first-mover" replaced by "open tooling" as the middle pillar — the timing window closed; the
  priority claim in bold above is retained.)
- A2. The `clif-validate/` open shippable package is the deliverable that distinguishes us from DUA-gated
  ICareFM — treat it as a headline artifact, not plumbing. **2026-08-28 — what "open" means:** the package,
  its source, and its bundle-compatibility contract are publicly obtainable with NO DUA and NO per-site
  approval; trained-weight bundles stay signed and governed. Without the first half the differentiator does
  not exist, because a signed, approval-gated distribution channel is operationally what ICareFM already has.

**B. Backbone & pretraining (LOCKED)**
- B1. Backbone = Qwen-family transformer; it is a **footnote, not novelty** (ORA: objective>backbone,
  within-noise at ~30M/8k). Do not spend novelty budget here.
- B2. **From-scratch path → Qwen3 architecture** (free QK-Norm training stability; `Qwen3Config`/`Qwen3ForCausalLM`
  confirmed present in transformers 5.16.1). **Attach/wedge path → Qwen2** (must match CLIFATRON's checkpoint).
  Keep a Qwen2-arch from-scratch arm too → "Qwen2 vs Qwen3" becomes one MEASURED ablation row, not an assertion.
- B3/B4. **PRIMARY PAPER = from-scratch Qwen3-arch decoder + objective D (marked-TTE), ~30M, fully ours,
  no upstream dependency.** Run it as a LADDER: (1) frozen-probe Method-3 wedge on a CLIFATRON Qwen2 ckpt
  (cheap, de-risks the objective, first result) → (2) from-scratch Qwen3 pretrain (novel headline) →
  (3) the two together ARE the finetune-vs-scratch ablation.

**C. Size coherence (LOCKED)**
- C1. Our own model genuinely targets **~30M** (d512×8L×8H ≈ 30–37M) — that is the compact/one-node thesis.
  When we attach to CLIFATRON's **Qwen2-0.5B** for the wedge, state explicitly it is a LARGER comparator,
  not our compact claim. Never imply the 0.5B is "our ~30M model." (Fixes the 16× coherence gap.)

**D. Objective (LOCKED — where the novelty lives)**
- D1. Loss weights CR 1.0 · threshold 1.0 · value 0.5 · NTP 0.2 (in code). D2. NTP→TTE curriculum
  (15% warmup / 5% transition). D3. **RESOLVED (was: value-head loss unnormalized, val≈46000 on the
  development site).** `src/data/value_stats.py` now freezes per-**token** robust (median/IQR) value stats from a
  reference site, vocab-hash-bound; `pretrain.py --value-stats` applies `(value−center)/scale` and
  fails closed on a missing/stale map. Verified: standardization collapses mean raw value² from ~1.4e10
  to ~0.95 (O(1) NLL). Landed via PR #3.

**E. Tokenization (LOCKED — binning scheme REVISED 2026-09-02)**
- E1. Fused `code=bin`, soft discretization, forced clinical-threshold edges, storetime ordering,
  untied embeddings, 8192 context.
- **E1a. PRIMARY binning = physician-designed CLINICAL SEGMENTS (revised 2026-09-02).** The CLIF
  consortium's `critical_illness_tokenization_final_with_intervals.csv` (1268 clinician-designed
  segments across labs/vitals) is the default (`configs/data.yaml → value_binning.scheme:
  clinical_segment`; `src/data/build_clinical_segment_bins`). These encode measurement-density
  granularity — tighter intervals in decision zones, extreme-value quintiles at the tails — that
  the team judges **more clinically relevant** than data-driven deciles, and it is what differentiates
  this model from a plain AR token predictor. **Population deciles are demoted to the `decile_ablation`
  arm** (Lee 2026 found deciles ≈ ref-range at matched granularity — so we MEASURE the two head-to-head
  in the tokenization ablation rather than assert either; whichever wins under the frozen protocol
  stays the default). *This supersedes the earlier "frozen deciles PRIMARY, clinical bins = ablation"
  decision (2026-08-27) — reversed on clinical-relevance grounds; the ablation keeps it honest.*
  Clinical decision thresholds (lactate 2/4, MAP 65, SpO₂ 88/90, KDIGO, P/F Berlin) are guaranteed
  bin edges under either scheme.
- E2. **TextCode / language-grounded arm ELEVATED from future-work to a real transfer-robustness arm.**
  PORTER (arXiv:2606.24102, 2026): frozen-vocab models drop ~69% of events on cross-site transfer;
  language-grounded recovers 97.1% AUROC without vocab mapping. Frozen mCIDE stays PRIMARY (turnkey, matches
  CLIFATRON); TextCode is the ablation that shows whether language-grounding buys cross-site robustness — the
  one 2026 result that helps OUR federation metric.

**F. Federation (LOCKED)**
- F1. Model-to-data, aggregate + subgroup metrics only, nothing raw leaves a node (settled).
- F2. "Label-free" refers to the MODEL only — evaluation auto-derives ground-truth labels locally; those
  derived labels are a **validity dependency** (noisy phenotypes, vary by site coding) — audit + report
  each outcome's label definition and provenance per site.
- F3. **Add a small-cell suppression rule** (e.g. suppress subgroup metrics with n<10) before shipping.

**G. Practical blockers (not design decisions — must clear before real runs)**
- G1. Locate a CLIFATRON checkpoint (needed for the Method-3 wedge). G2. Site 2 + Site 3 data not staged
  on the L40 box (only Site 1: 546,028 stays / ~134M events) — the 3-site claim needs them. G3. Verify
  `head_adapter.anchor_state` against a real checkpoint on transformers v5 (output_hidden_states API drift).

> **Site ID mapping (public docs use these IDs):** Site 1 = MIMIC-IV-Ext-CLIF (PhysioNet), Site 2 = Rush
> (institutional), Site 3 = UChicago (institutional, CLIF origin site). Website docs refer to sites by
> generic ID only. Internal notes, configs, and code use the real names.

## HANDOFF → notes/NEXT_STEPS.md (2026-08-27)
Full agent-handoff written: finalized token+arch decisions (with the 2026 evidence tables + citations
from research threads a76bb9/aeb4d2), ordered file-level next steps (config↔code reconcile → run Method-3
on real ckpt → phase-2 head pretrain → tokenization ablation → clif-validate/ → notes modality), open
items to verify, hard rules, env mechanics. A fresh agent should read notes/NEXT_STEPS.md first.

## Status (2026-08-27)
Deep research (notes/RESEARCH.md) + PIVOT to build-on-CLIFATRON (notes/INTEGRATION.md).
METHOD-3 WEDGE BUILT (keeper methods layer, all py_compile clean; panel numerically validated
via `uv run --with numpy --with scikit-learn`):
- src/model/head_adapter.py — attach our heads to a CLIFATRON HF checkpoint's hidden states
  (frozen-probe or joint fine-tune; zero-shot threshold_prob()).
- src/eval/metrics.py — full TRIPOD+AI panel: auroc/auprc/ece/brier/calib-slope+intercept/ICI/
  net-benefit(DCA)/temperature/LPE/subgroup. Validated on synthetic imbalanced data.
- src/eval/method3.py — driver: anchor states → probe(TaskHead) vs xgboost(Method 1) →
  3×3 transportability matrix (per-site local fit, no pooling) + Elemento ensemble + LPE.
- src/eval/matrix.py slimmed to re-export shim. pyproject += xgboost, transformers.
Heads.py (ThresholdHazard/CompetingRisk/ValueRegression) unchanged = the keeper.
NEXT: (1) confirm CLIFATRON benchmark parquet column names (seq/label/subgroup) + which sites
its checkpoint is trained on. (2) Run method3 on a real checkpoint across Site 1/Site 2/Site 3 (L40).
(3) Phase 2: joint-pretrain the heads on CLIFATRON backbone → enables zero-shot survival for the
external CLIF-federation validation. (4) clif-validate/ shippable package (model-to-data). (5) tokenization
ablation: CLIFATRON clinical bins vs Lee-2026 deciles. (6) notes modality (BioClinical ModernBERT).
