# Deep Research Synthesis — Compact Multimodal ICU Foundation Model on Federated CLIF

> **⚠️ PRE-PIVOT DOCUMENT — evidence base is current; the design spec in §2–§7 is SUPERSEDED.**
> This synthesis was written **before** the 2026-08-27 decision to build ON CLIFATRON
> (see `notes/INTEGRATION.md`) and before the two focused 2026-preprint threads
> (tokenization a76bb9 / architecture aeb4d2). Its **literature/evidence** remains the
> reference of record, but several **design decisions here are now overridden**. Where this
> file disagrees with `MEMORY.md`, **`MEMORY.md` wins.** Specifically superseded:
> - **Embeddings:** this file says *tied* (§2, §7) → **now UNTIED** (+4–7% AUPRC, widens under federation).
> - **Context:** this file says *4096 primary* (§2) → **now 8192 primary** (match CLIFATRON trunk); 4096 = ablation.
> - **Trunk:** this file says *from-scratch flat Llama-style decoder is primary* (§2, §6 "Track A") →
>   **now attach heads to CLIFATRON's Qwen2 backbone** as the primary path; from-scratch is an ablation arm.
> - **Sites:** this file frames *2 sites (Rush + MIMIC)* → **now 3 dev sites (MIMIC + Rush + UChicago)** + federated external validation.
>
> The finalized, current spec lives in `MEMORY.md` ("Design spec — REVISED") and `notes/NEXT_STEPS.md` §2.

Five parallel research threads (2026-08-27): landscape/novelty, tokenization/architecture,
pretraining objectives/multi-task, multimodal fusion, benchmarks/fairness/assets. This
file is the **evidence record**: what the literature says. The **adopted design spec** it
originally proposed has been partly superseded — see the banner above and defer to `MEMORY.md`.
Line-cited primary-source detail is in `notes/METHODS.md`.

---

## 0. The thesis (unchanged, now defended)

**One small (~30M) multimodal model, pretrained once on federated CLIF ICU data with a
threshold-conditioned time-to-event objective, answers many outcomes across many
hospitals from one node (2× L40, no cluster).** "One small model → many outcomes → many
hospitals → one node."

---

## 1. Novelty verdict

No single published paper occupies the 4-way intersection we target:

  CLIF-native  ×  structured+notes ICU  ×  ~30M efficient  ×  real-data federated  ×  threshold-TTE

Each axis alone is contested; the **conjunction + first-mover CLIF-native execution** is
the defensible headline.

- **CLIF-native is our only fully-open axis — and it is on a countdown.** CLIF v3.0
  ("Going Multimodal", notes+imaging) is slated **July 2026** (Bhavani 2026; Lyons,
  "Federation not centralization," Ann ATS 2026). The consortium is racing toward exactly
  this target. **Move now; this is a first-mover window, not a durable moat.**
- **Efficiency axis** validated: CoMET saturates at **11–101M params** (arXiv 2508.12104);
  EHR scaling-laws show utility saturating ~28M (arXiv 2505.22964); specialized ~10²M
  models beat fine-tuned 70B LLMs on **AUPRC** (medRxiv 2026.04.24.26351503). Our ~30M is right.
- **Threshold-TTE axis** owned by ICareFM (medRxiv 2025.07.25.25331635) — but on ricu, not
  CLIF, and weights are DUA-gated. We port the objective to CLIF.
- **Federated-real-data axis**: Elemento (inference-time ensembling ≈ federated, lighter
  governance; FedProx collapsed) and GPT-MEDIC (Rockenschaub, in progress) are close; none
  are CLIF-native with a threshold-TTE core.

**Framing for Nature Medicine:** efficiency + federated-fairness + CLIF-native +
deployable multimodal SLM, with the ICareFM zero-shot engine as the multi-outcome mechanism.

---

## 2. Tokenization & architecture — REVISED (biggest code change)

Driver: **Lee et al. 2026 (arXiv:2604.16775, github.com/bbj-lab/fms-ehrs)** — 28 matched
decoders trained on **MIMIC-IV-Ext-CLIF**, the single most direct evidence for our stack.

| Decision | Old scaffold | **New (adopted)** | Why |
|---|---|---|---|
| Code+value token | split: `C::concept` + `V::concept::bin` (2 tokens) | **FUSED single token** `concept=bin` | Biggest single win: mortality 0.891→0.915 |
| Value binning | deciles | **deciles** (keep) | Best default in ablation |
| Position | continuous-time Δt-ALiBi | **admission-relative RoPE, 1-min-resolution IDs** | ≥ inserted time tokens, ~11% shorter sequences |
| Context | 512 events | ~~4096~~ → **8192 tokens** (match CLIFATRON trunk; 4096 = ablation) | Covers >99.95% of first-24h stays; long ctx more robust to copy-forward |
| Backbone | dual-level (intra-event pool + inter-event) | ~~flat Llama-style causal decoder (from scratch)~~ → **CLIFATRON Qwen2 backbone** (from-scratch flat decoder = ablation arm) | Objective, not backbone, is the lever (ORA); attach to consortium checkpoint |
| Units | (unspecified) | **normalize units before binning** | CLIF remap is free but units are not harmonized |
| Ordering | (unspecified) | **`storetime`/availability, not `charttime`** | No lookahead on when a value was actually knowable |

**Revised trunk (SUPERSEDED — see banner):** ~~Llama-style — dim 512, ~8 layers, 8 heads,
SwiGLU, RMSNorm, RoPE, tied input/output embeddings, ~30M params, context 4096, bf16.~~
**CURRENT:** attach heads to CLIFATRON's **Qwen2** backbone (same family: RoPE/SwiGLU/RMSNorm),
**UNTIED** embeddings, context **8192**; d512×8L×8H flat from-scratch decoder retained only as
an ablation arm (4096 = its ablation context). See `MEMORY.md`.

*Note on the marked-TTE tension:* fused token is the **input** representation; the
continuous value is still predicted as a **target** by the value-regression head (§3).
Not contradictory — one is representation, the other is supervision.

---

## 3. Pretraining objectives — REVISED (enable the value head)

Driver: **ORA (arXiv:2602.00541)** — the cleanest apples-to-apples result in the field,
tokenizer+backbone fixed: **marked-TTE > TTE > next-token-prediction**. "Marked" = predict
*(type, time, value)* jointly.

**Objective mix (adopted):**
1. **PRIMARY — threshold-conditioned hazard (ICareFM) + competing-risk CIF (SurvivEHR).**
   Random τ, random horizons, learned threshold+direction embeddings, discrete hazard over
   48h; composite events by conjunction/disjunction under conditional independence. This is
   the zero-shot multi-outcome engine. ICareFM dual-zero-shot (unseen task **and** unseen
   hospital) median AUROC **0.837** — strongest ICU evidence.
2. **NEW AUXILIARY — value regression (the ORA "mark").** Predict the continuous lab/vital
   value. Lifts physiology/regression tasks **+33–38% over NTP** — exactly our threshold
   events (AKI, RRT, lactate, vasopressor need). **`value_regression` flips false→true.**
3. **LOW-WEIGHT AUXILIARY — next-event (NTP).** Weakest representation learner alone, but
   retains open-ended zero-shot for tasks not expressible as a threshold query (the
   readmission failure class of EveryQuery/ETHOS). At inference use **SCOPE/REACH**
   (arXiv:2602.03730) to cut rollout cost 2.5–3.4× (>80× for rarest outcomes), calibration preserved.

**Loss balancing:** uncertainty-weighting (Kendall 1/2σ² learned per task) + grad-norm
normalization to the primary task's gradient, so the dense mortality signal doesn't starve
sparse auxiliaries ("signal-balance problem," arXiv:2607.22264). Robust fallback: just sum
the TTE loss over many randomly-sampled τ per step (MOTOR/ORA default).

**Curriculum:** warm up on NTP to stabilize token embeddings, then phase in TTE/threshold/
value heads. (Reasoned default; no paper ablates the exact schedule.)

**Skip as primary:** masked/BERT (no zero-shot) and standalone contrastive (auxiliary only).

---

## 4. Multimodal fusion — notes as timestamped event tokens

Driver: multimodal thread + **Generative Deep Patient** (Nature 2026, s44401-026-00095-y).

- **Encoder:** **BioClinical ModernBERT-base** (150M, 8192-token, RoPE+FlashAttn,
  arXiv:2506.10896), **frozen**. Precompute one embedding per note; project into trunk dim.
- **Fusion:** inject each note embedding as a **timestamped event token in the same stream**
  at its `storetime`. Intermediate/Pre-RNN fusion beats late fusion. This makes our
  pre-anchor leakage rule *automatic* — a note token literally cannot appear after the anchor.
- **Expected lift:** modest **+2–3 AUROC / larger relative AUPRC**, concentrated in **≥24h
  outcomes**; near-zero for imminent events; text-alone ≈ chance.
- **LEAKAGE ALARM:** a paper claimed **+22% AUROC** by feeding discharge summaries +
  radiology reports — retrospective documents that encode the label. **Rule of thumb: an
  oversized note gain means the note leaked the outcome.** Only pre-anchor notes are features
  (hard rule 3).

---

## 5. Evaluation framework

**CLIF → MEDS is a supported near-drop-in path.** The CLIF consortium ships `CLIF_MEDS`
(config-driven CLIF→MEDS ETL) and `CLIF-MIMIC` (github.com/Common-Longitudinal-ICU-data-Format,
Apache-2.0). This unlocks the whole MEDS ecosystem below.

**Baselines & harnesses (adopt):**
- **MEDS-Tab** (arXiv:2411.00200) — XGBoost AutoML. **Mandatory strong baseline** every FM
  paper reports; cheap on our hardware.
- **MEDS-DEV** — portable ACES task configs so identical task defs run on Rush and MIMIC-CLIF.
- **YAIB** (arXiv:2306.05109) and **HiRID-ICU-Benchmark** (arXiv:2111.08536) — canonical ICU
  dynamic-prediction task sets (mortality, AKI, sepsis, circulatory/respiratory failure, LoS);
  both motivate AUPRC because tasks are highly imbalanced.
- **EHRSHOT** (arXiv:2307.02028) — template for label-efficiency (shots-vs-AUROC) curves,
  complementing our LPE metric.

**Metric panel (fixed, report all):**
- Discrimination: **AUROC + AUPRC** on every binary task; balanced acc (multiclass); C-index (TTE).
- Calibration: **ECE + Brier + calibration slope/intercept + ICI**, with calibration plots
  (temperature scaling is the recalibration step feeding this).
- Clinical utility: **Decision-Curve Analysis / net benefit** — run *after* recalibration
  (DCA assumes calibrated probabilities). Now expected at high-impact venues.
- Label efficiency: **LPE** (ICareFM) + EHRSHOT shots curves.
- Fairness: per-subgroup AUROC/AUPRC/**calibration** for sex, age, race/ethnicity, site
  (TRIPOD+AI subgroup item). Calibration degrades faster than discrimination off-site.

**Task × Site transportability matrix** (the money figure):
- Rows = train site (Rush, MIMIC-CLIF, pooled); cols = test site. Diagonal = internal,
  off-diagonal = external transport. Report AUROC/AUPRC **+ calibration slope** per cell.
- **Adaptation ladder** per off-diagonal cell (Guo et al., npj Digit Med 2022): (i) as-is
  zero-shot, (ii) **recalibration only** (temperature — cheap, no source data), (iii)
  fine-tune on local labels. Report LPE at each rung. Plain ERM + recalibration is a
  hard-to-beat baseline — include it.

**Reporting standard:** **TRIPOD+AI** (BMJ 2024;385:e078378, DOI 10.1136/bmj-2023-078378) —
27-item checklist; new items we must hit: subgroup performance, patient-and-public
involvement, open-science (protocol registration, data + code sharing). External validation
on ≥1 unseen site (we have Rush + MIMIC-IV-Ext-CLIF). Path to prospective/silent eval
(ICareFM's registered cluster-RCT of a predecessor, NCT07119411, is the framing precedent).

---

## 6. Reusable assets — reuse vs benchmark vs skip

**Reuse (code/pipelines):**
- **ratschlab/icarefm** — harmonization code + pipelines public (weights DUA-gated).
  Primary comparator & architectural template; replicate its dual-zero-shot + staged-
  adaptation + LPE protocol. ricu 130-concept abstraction maps cleanly onto CLIF mCIDE.
- **CLIF_MEDS → {MEDS-DEV, MEDS-Tab, ETHOS-ARES}** — lowest-friction, highest-leverage.
- **SurvivEHR** (code open, no weights — CPRD forbids), **ETHOS-ARES** (code, no weights —
  MIMIC-derivative rules) — retrain on our data.

**Benchmark against (released weights, fine-tunable on 2×L40):**
- **CLMBR-T-base** (141M) and **MOTOR** (143M) — the two fully-released coded-EHR FMs with
  weights, via the harder **CLIF→OMOP→FEMR** bridge (mapping "a crucial next step," not done).
  MOTOR also gives the marked-TTE C-index comparator.

**Skip as comparators:** CoMET (unreleased, Cosmos-only), Foresight (no weights, needs
free-text NER). GatorTron only relevant if/when we add the notes modality (it's a text
encoder). CLIF→OMOP is less mature than CLIF→MEDS — treat as secondary.

**Build tracks (SUPERSEDED ordering — see banner):** the pivot to build ON CLIFATRON
(`notes/INTEGRATION.md`) makes **attach-heads-to-CLIFATRON the primary track**, not from-scratch.
- **PRIMARY — build on CLIFATRON:** attach our survival/threshold heads to the consortium's
  released CLIFATRON Qwen2 checkpoint (frozen-probe → joint fine-tune). This is "Method 3", the
  smallest publishable unit, and the paper's spine.
- **Ablation — from scratch:** pretrain our ~30M CLIF-native trunk on the L40s; kept as the
  from-scratch arm of the finetune-vs-scratch ablation, ~~not the primary paper~~.
- **Optional — fine-tune ICareFM (DUA-gated):** request ICareFM weights; fine-tune on CLIF as a
  head-to-head comparator. Long lead time.

---

## 7. Finalized design spec (SUPERSEDED — current spec is in MEMORY.md + NEXT_STEPS.md §2)

> The spec below is the **pre-pivot** version. Corrected values inline; authoritative copy in `MEMORY.md`.

**Data:** CLIF 2.1 → fused-decile tokens (`concept=bin`), unit-normalized, `storetime`
ordering, per-site shards, vocab frozen on a reference site & applied identically across the
**3 dev sites (MIMIC + Rush + UChicago)** — no raw pooling. Context **8192** tokens
(~~4096~~; 4096 = ablation). Notes (v2) injected as frozen BioClinical-ModernBERT event tokens,
pre-anchor only.

**Trunk:** **CLIFATRON Qwen2 backbone** (primary) — RoPE/SwiGLU/RMSNorm, **UNTIED** embeddings,
~30M-neighborhood, bf16. From-scratch flat Llama-style decoder (d512×8L×8H, admission-relative
1-min RoPE) retained as the **ablation arm only** (~~tied embeddings~~ → untied there too).

**Heads / objectives:** threshold-hazard (ICareFM, primary) + competing-risk CIF (SurvivEHR)
+ **value regression (ORA mark, NEW-enabled)** + low-weight next-event. Uncertainty +
grad-norm loss weighting; NTP→TTE curriculum; random τ/horizons.

**Training:** 2× L40 (48GB, no NVLink), DDP via torchrun, bf16. MacBook (MPS, fp32, no
compile/DDP) for dev/overfit-one-batch/finetune only. Do not rent GPUs — PhysioNet
credentialed data stays on the lab server.

**Eval:** task×site matrix + adaptation ladder (as-is/recalibrate/fine-tune) + LPE; metric
panel AUROC/AUPRC/ECE/Brier/calibration-slope/ICI/DCA; subgroup fairness; MEDS-Tab XGBoost
baseline; MEDS-DEV portable tasks; TRIPOD+AI reporting.

---

## 8. Hard rules (unchanged — the non-negotiables)

1. Treatments (meds) are **inputs, NEVER prediction targets**.
2. Value-bin vocab **frozen on one site (MIMIC)**, applied to both — no cross-site raw pooling.
3. Retrospective reports/discharge summaries = **LABEL source only**; only pre-anchor notes
   may be features. Oversized note gains = leakage.
4. Use **`storetime`** (availability), not `charttime`, for event ordering.
5. MIMIC-IV-Ext-CLIF and Rush CLIF are **PhysioNet credentialed / institutional restricted** —
   never upload to rented cloud without a compliant BAA/DUA.

---

## 9. Key references

ICareFM medRxiv 2025.07.25.25331635 · SurvivEHR s41746-026-02709-z / medRxiv 2025.08.04.25332916 ·
ORA arXiv:2602.00541 · MOTOR arXiv:2301.03150 (ICLR 2024) · Lee/Beaulieu-Jones "Representation
Before Training" arXiv:2604.16775 · CoMET arXiv:2508.12104 · ETHOS s41746-024-01235-0 / ARES
arXiv:2502.06124 · EveryQuery arXiv:2603.07900 · SCOPE/REACH arXiv:2602.03730 · Generative Deep
Patient Nature 2026 s44401-026-00095-y · BioClinical ModernBERT arXiv:2506.10896 · EHR scaling-laws
arXiv:2505.22964 · CFM-vs-LLM medRxiv 2026.04.24.26351503 · MEDS-Tab arXiv:2411.00200 · YAIB
arXiv:2306.05109 · HiRID-ICU-Benchmark arXiv:2111.08536 · EHRSHOT arXiv:2307.02028 · TRIPOD+AI
BMJ 2024;385:e078378 (10.1136/bmj-2023-078378) · CLIF PMC11398431 / consortium
github.com/Common-Longitudinal-ICU-data-Format.
