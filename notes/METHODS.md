# Methods — copied recipe from 2025–26 preprints

Line-level citations from full text (Paperclip). This is the source of truth for
what each module implements; when in doubt, match the paper.

## 1. ICareFM — the core pretraining objective (COPY THIS)
*Burger et al., "A Foundation Model for Intensive Care." medRxiv 2025.*

- **Objective**: self-supervised. Conditioned on patient history, estimate the
  *time-dependent probability that a target clinical variable crosses a threshold τ
  in a specified direction* within a horizon. **K = 35** target variables.
  τ is **randomly varied during training** to learn diverse events.
- **Backbone**: a **time-step encoder + causal Transformer** producing **hourly
  patient-state representations**. Total size **≈30M params**.
- **Heads**: *per target variable, a dedicated hazard head* maps (patient state, queried τ)
  → a **discrete hazard function over a 48-hour horizon**, via **learned threshold
  and direction embeddings**.
- **Critical rule**: **treatments EXCLUDED from the target set** — prevents learning
  hospital-specific treatment patterns. (Treatments still allowed as *inputs*.)
- **Concepts**: 130 harmonized clinical concepts (demographics, vitals, labs, treatments);
  ~40 treatment concepts are inputs only.
- **Zero-shot inference**: each head yields cumulative failure prob
  `F_k(h | H_t, τ_k)` = P(variable k crosses τ_k within horizon h | state H_t).
  **Composite events = conjunction/disjunction of univariate failures under
  conditional independence.** e.g. 8h circulatory failure ≈
  `F_MAP(8h | H_t, <65 mmHg) · F_Lact(8h | H_t, >2 mmol/L)`.
  Sweeping τ across value bins approximates composite clinical scores — **no retraining**.
- **Deployment modes** (all exclude target site from pretraining):
  (i) dual zero-shot; (ii) external adaptation; (iii) local adaptation; (iv) staged adaptation.
- **Tasks evaluated**: circulatory / respiratory / kidney / liver failure, hyperglycemia,
  mortality (decompensation), sepsis; all binary early-event prediction, 2–48h horizons.

## 2. SurvivEHR — competing-risks generative head + event embedding (COPY THIS)
*Gadd et al., "SurvivEHR." medRxiv 2025.*

- **Split valued-event embedding**: each event = (category token p_c, optional value v_c).
  Two embedding networks: a category embedding `W[p_c]` + a **value-weighted projection**;
  **deviation of value from the population average** also contributes to the embedding.
- **Positional embedding**: encode **time since birth t_c** (days) with the *original
  sinusoidal positional encoding* — learned positional embeddings gave **no benefit** for
  the extra cost. (For ICU we replace "since birth" with "since ICU admission / Δt".)
- **Causal survival pretraining (competing risks)**: predict next event type `k_{c+1}`
  and time-to-event `Δt_{c+1}`. Competing-risk assumes exactly one of the possible next
  events occurs; event times ~ Poisson process ⇒ **no null-event token needed**.
  Each event type has a **cumulative incidence function (CIF)**.
- **Causal value prediction**: for valued events, a **probabilistic regression head**
  predicts a **Gaussian (mean, std)** for the value; log-likelihood added to the loss.

## 3. HealthFormer — dual-level time-aware encoder (COPY STRUCTURE)
*Kőrösi-Szabó et al., medRxiv 2026.*

- **Intra-Event Encoder**: aggregate heterogeneous domain tokens *within one typed event*
  via code-specific embeddings + **attention pooling** → one event embedding.
- **Inter-Event Encoder**: event embeddings + a **Date Encoder** + a
  **continuous-time ALiBi attention bias** (elapsed Δt biases attention, not a positional id).
- **Multi-task self-supervision**: (i) per-domain masked-token prediction (MLM),
  (ii) event-level MLM (full-event masking → predict event type),
  (iii) next-event-type prediction, (iv) **Δt time-to-next-event regression**.

## 4. Elemento — two-site federation story (COPY THE EVAL)
*Elemento, medRxiv 2026.*

- GPT-style EHR FM on MIMIC-IV, partitioned into non-IID sites by Dirichlet over ICD chapters.
- **Inference-time ensembling** (average predictions, no weight sharing) recovered **77%**
  of centralized AUPRC, **97–99%** of per-condition AUROC; helped **87%** of hospitals;
  lightest governance. **FedProx collapsed** — skip it.
- Use this to frame Rush+MIMIC: ensemble the two site-local models at inference.

## 5. Cadence — small-model + reporting rigor (COPY THE RIGOR)
*Rouhollahi & Nezami, medRxiv 2026.*

- ~5.86M-param model *matches XGBoost* on next-event prediction — small-beats-big precedent.
- **Temperature scaling** (single scalar T*) fixes calibration (ECE 0.077 → 0.028).
- **Dual-sex TRIPOD+AI reporting** — report every metric split by sex (and we add race).

---

## Our synthesis (what this repo builds)
Trunk = HealthFormer dual-level encoder (intra-event attention-pool + inter-event causal
transformer with Δt-ALiBi), ~30M params (ICareFM scale). Pretraining = **three heads**:
(A) next-event type (SurvivEHR/HealthFormer), (B) competing-risk Δt CIF (SurvivEHR),
(C) **threshold-conditioned hazard over 48h with learned τ+direction embeddings (ICareFM)** —
this last one is the zero-shot multi-prediction engine. Downstream = K frozen-trunk task heads.
Two-site = inference-time ensemble (Elemento). Report = task×site matrix + LPE + temperature
scaling + dual-sex/race slices (ICareFM + Cadence).

**Non-negotiables copied verbatim:**
1. Treatments are inputs, NEVER prediction targets (ICareFM).
2. Vocab/value-bins frozen from one site, applied to both (no leakage across sites).
3. Retrospective reports/discharge summaries = LABEL source only, never features
   (only pre-anchor notes may be features — temporal leakage guard).
4. Keep raw data per-site; exchange predictions, not data (Elemento).
