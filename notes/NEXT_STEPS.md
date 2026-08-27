# Handoff — CLIFATRON 2.0: research, decisions, and next steps

**Audience:** the next agent (or engineer) taking this over cold. Read this top-to-bottom before
touching code. It carries the *why* behind every decision plus the 2026 evidence, so you can
execute — or overrule with cause — without re-running the research.

**Last updated:** 2026-08-27. **Repo:** https://github.com/sajor2000/clifatron2.0 (private).
**Companion docs:** `MEMORY.md` (living design record), `notes/RESEARCH.md` (5-thread deep research),
`notes/INTEGRATION.md` (build-on-CLIFATRON plan), `notes/METHODS.md` (line-cited method sources).

---

## 0. One-paragraph orientation

CLIFATRON 2.0 is a **methods-upgrade layer** on [CLIFATRON](https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON)
(vendored at `external/clifatron/`, upstream commit `d3d2818`, MIT — the CLIF consortium's compact
~30M CLIF-native ICU foundation model, built by our lab's data scientist `vchaudha`, which **we
control**). CLIFATRON already owns the "compact CLIF-native AR ICU FM" axis, so that is *not* our
novelty. Our contribution is: (1) a **threshold-TTE + competing-risk + value-regression** objective
replacing pure next-token prediction; (2) **zero-shot, label-free** survival heads; (3) **federated
external validation by model-to-data** across the whole CLIF consortium; (4) a full **TRIPOD+AI**
calibration / decision-curve / fairness eval panel. Thesis: *one small model → many outcomes → many
hospitals → one node (2× L40, no cluster).*

**Guiding principle (do not lose this):** the model must be most sensitive where clinical **danger**
is and be legible to a clinician. Outcomes are states doctors *act on* (never treatments); threshold
heads are **directional** (crossing *into* danger); the headline metric is **net benefit / decision-curve
analysis**, not AUROC alone.

---

## 1. Current state — what is built and what works

**Built and verified (py_compile clean; metric panel numerically validated on synthetic data):**
- `src/model/heads.py` — `ThresholdHazardHead` (ICareFM zero-shot), `CompetingRiskHead` (SurvivEHR
  discrete-time CIF), `ValueRegressionHead` (ORA "mark", Gaussian mean/logvar), `next_event_loss`
  (tied-embedding CE), `composite_or/and`, `TaskHead`. **[KEEPER]**
- `src/model/head_adapter.py` — attaches our heads to a CLIFATRON HF checkpoint's hidden states;
  `load_backbone`, `CLIFATRONHeads`, `anchor_state`, `threshold_prob` (zero-shot), `loss`. **[KEEPER]**
- `src/eval/metrics.py` — full TRIPOD+AI panel: auroc/auprc/ece/brier/calib-slope+intercept/ICI/
  net-benefit(DCA)/temperature/LPE/subgroup. Pure numpy+sklearn; torch only in `temperature_scale`.
  **Validated**: AUROC 0.953/AUPRC 0.826 on 15%-prevalence synthetic; slope 1.94 correctly flagged
  miscalibration. **[KEEPER]**
- `src/eval/method3.py` — the wedge driver: frozen anchor states → probe(TaskHead) vs xgboost(Method 1)
  → 3×3 transportability matrix + Elemento ensemble + LPE. **[KEEPER]**
- `src/eval/matrix.py` — re-export shim (stable import surface).

**Ablation-arm code (kept, not the default path):**
- `src/model/encoder.py` — from-scratch Llama-style RoPE decoder. Default trunk is CLIFATRON's Qwen2;
  this is the from-scratch ablation arm.
- `src/data/tokenize.py` — our fused-token / decile tokenizer. Default is CLIFATRON's tokenETL; this is
  the decile-vs-clinical-bins ablation arm.

**Not yet built:** phase-2 head pretraining on the CLIFATRON backbone; the shippable `clif-validate/`
model-to-data package; the notes modality; the CR D-calibration metric; several config↔code reconciliations
(see §4).

**NOTE — nothing here has been run on a real checkpoint or real CLIF data yet.** Torch is not installed
on the Mac dev box (verify code with `python3 -m py_compile` and `uv run --with numpy --with scikit-learn`).
The 2× L40 Linux box is where real runs happen (bf16, DDP via torchrun, no NVLink).

---

## 2. FINALIZED design decisions (with rationale)

These are settled by the two 2026-preprint research threads (§3). Configs already reflect them
(`configs/model.yaml`, `configs/data.yaml`). Change only with new evidence.

### 2.1 Tokenization
| Decision | Value | Why |
|---|---|---|
| Fusion | **Fused `code=bin` single token** (keep CLIFATRON's) | Settled: Lee mortality 0.891→0.915; Guo 73/74 tasks, −39.5% FLOPs. Avoids the "local binding problem." |
| Base bins | **Population deciles, frozen from a reference site** — NOT clinical-reference-range bins | Lee tested ref-range anchoring vs deciles at matched granularity → **no consistent advantage**. Frozen decile *edges* transport (Federated GEMs: 0.025 AUROC cross-site penalty vs 0.079 LightGBM). |
| Tail sensitivity | **Soft discretization** (adjacent-bin mass) | Lee's one encoder that WINS on exactly the dangerous tails: severe hypokalemia/hypernatremia/hypotension/CRRT/K-extremes. This is how we honor "sensitive where it's dangerous." |
| Clinical anchoring | **Force ICU decision thresholds as guaranteed bin edges** (lactate 2/4, MAP 65, SpO₂ 88/90, KDIGO, P/F Berlin) | Legible at decision points + aligns token boundaries with the threshold-hazard head's queries. A constraint on top of deciles — does NOT contradict Lee (he anchored the *whole* scheme). |
| Continuous value channel | **Ablation arm only** (McCann fused-continuous + numeric head + MC) | +30% numeric accuracy, −34% seq len, but **worse calibration** than discrete; calibration is our headline. Naive *unfused* xVal collapses to the median (Lee) → never use unfused. |
| Time encoding | **Admission-relative-minute RoPE, same timestamp → same position; drop `day_N/hour_N` tokens** | Matches/exceeds inserted time tokens while cutting sequences ~11% (Lee); 71/74 tasks, −9.6% FLOPs (Guo). It is also the part that transfers across sites. |
| Vocabulary | **CLIF-native controlled (mCIDE), fused-decile, ~10k, frozen from a reference site** | Controlled vocab transfers; learned BPE-over-events does not (Guo). ~10k chosen to fit the untied-embedding budget (see 2.2). |
| Ordering | storetime (availability) not charttime; unit-normalize before binning | Avoids look-ahead leakage; makes bins comparable across sites. |

### 2.2 Architecture
| Decision | Value | Why |
|---|---|---|
| Backbone | **Keep Qwen2/Llama transformer — do NOT switch to Mamba** | At 8k context / 30M, Mamba is within noise and only cheaper; the *objective*, not the backbone, drives EHR performance (ORA, backbone-agnostic gains). We also attach to CLIFATRON's Qwen2 checkpoint. Keep a Mamba/HyMaTE hybrid as a *stretch ablation* only. |
| Highest-leverage change | **Time-aware RoPE** (rotate by admission-relative minutes, shared within an event; optional log-Δt ALiBi) | Broadest 2026 support (Context Clues, MOTOR, RAVEN, Tokenization-Tradeoffs, HealthFormer). Handles irregular sampling; transports across hospitals. |
| Shape | d_model 512, 8–10 layers, 8 heads, FFN ~2048, SwiGLU, RMSNorm, keep GQA | Mirrors the ~28M scaling-law utility optimum and the validated 31.6M federated GPT. GQA is free at this scale. |
| Embeddings | **Untied** | +4–7% AUPRC, gap **widens under federation** (validated 31.6M federated GPT). |
| Context | **8192** (match CLIFATRON trunk we attach to); 4096 = Lee-tokenizer ablation | Long context is more robust to copy-forwarding and irregular gaps. |
| Systems | **DDP + FlashAttention varlen + per-patient packing masks**; NOT FSDP | FSDP only pays off past ~2.3B and is worse on no-NVLink. |
| Multimodal (v2) | Frozen note encoder → 2-layer MLP → **in-stream soft token at the note timestamp**; NOT cross-attention | Cross-attention ~doubles params at 30M. Pre-anchor notes only (leakage rule). |

### 2.3 The one real tension you must respect
**Untied embeddings + large vocab both eat the ~30M budget through the embedding tables.** At untied,
`vocab × d × 2` lives in embeddings. Chosen resolution: **untied + ~10k vocab** (≈8–12M emb + ~25M trunk
≈ 33–37M, still "~30M neighborhood"). A ~30k vocab would force *tied* embeddings and cost the federation
AUPRC bump. If you grow the vocab, revisit this.

### 2.4 Objective/head stack (fit confirmed by the architecture thread)
- Primary pretraining loss: **ORA marked-TTE** (value regression + calibrated marks, −20% head params).
- Zero-shot heads on the frozen trunk: **MOTOR piecewise-exponential TTE** + **cause-specific
  competing-risk subnetworks** (TraCeR/SurvivEHR).
- `next_event` demoted to low-weight (0.2) auxiliary.
- Loss balancing: uncertainty (Kendall 1/2σ²) + grad-norm. Curriculum: NTP → TTE.
- **New required eval:** competing-risk **D-calibration / Aalen-Johansen K-cal** (deep CR models are
  badly miscalibrated by default).

---

## 3. The 2026 evidence base (so you can audit or extend the decisions)

### 3.1 Tokenization thread (research agent a76bb9)
| Paper | ID | Settles |
|---|---|---|
| Lee et al., "Representation Before Training" | arXiv:**2604.16775** (2026-04-21) | Head-to-head on MIMIC-IV, 28 matched transformers, 30 outcomes: value encoding, time encoding, vocab. **Fused > split; deciles ≈ ref-range (no gain from clinical anchoring); soft discretization wins the tails; admission-relative RoPE ≥ inserted time tokens (−11% seq); centile-fused vocab blows up (~84k) not worth it.** |
| McCann et al., "Continuous Value Tokenization Improves Medical Event FMs" | medRxiv **2026.08.04.26359713** | Fused continuous value channel + numeric head + MC: −34% seq, +30% numeric acc, ICD AUPRC 0.457>0.446, **but discrete kept better calibration.** |
| Guo/Sung et al., "Tokenization Tradeoffs in Structured EHR FMs" | arXiv:**2603.15644** | Pediatric factorial, 74 tasks + external adult-ICU transfer. Fused beats factorized 73/74. Positional time > interval tokens on 71/74. **Structural/vocab associations transfer; temporal/workflow effects are institution-specific → prioritize vocab alignment, don't over-engineer site-specific Δt.** |
| ORA, "One Loss to Rule Them All: Marked Time-to-Event" | arXiv:**2602.00541** | Marked-TTE objective (timing + value) beats classification/TTE/regression baselines; backbone-agnostic (helps Mamba +11.4% and Transformer +10.7%). DeepHit-style discretized marked head, factorized projection (−20% head params). |
| SCOPE / REACH estimators | arXiv:**2602.03730** | CLIF 2.1 (MIMIC-IV + UChicago), decile-fused tokens. Efficient generative inference for rare outcomes: match 100-sample MC with >80× token reduction, calibration preserved. |
| Federated generative event models | arXiv:**2608.02939** | 122,251 ICU stays, 3 CLIF sites. Vocab + numeric bins learned on MIMIC then **frozen across sites** → 0.025 AUROC / 0.027 PR-AUC cross-site penalty (vs 0.079/0.089 for LightGBM). Direct precedent for our frozen-vocab federation. |
| Multi-Hospital EHR FM | PMC**13142595** | 6,256-token vocab; RoPE on inter-visit day-delta; two-level attention mask. |
| Supporting | xVal (Golkar 2024); Scaling Laws for EHR FMs (arXiv:2505.22964, 2025); Kirchler npj Digit Med 9:177 (2026) | — |

**Single highest-leverage tokenization change:** deciles + soft discretization on the existing fused CLIF
tokens. It removes an assumption Lee showed buys nothing (ref-range anchoring), targets our rare/threshold
outcomes with the proven-best encoder, and keeps everything else CLIFATRON/federation-ready.

### 3.2 Architecture thread (research agent aeb4d2)
| Paper | ID | Settles |
|---|---|---|
| Context Clues | arXiv:**2412.16178** (ICLR 2025) | GPT vs Llama vs Mamba vs Hyena at ~120M. Mamba@16k wins EHRSHOT avg; GPT absolute-position perplexity spikes **fixed by RoPE**; long context robust to copy-forward/irregular gaps. |
| ORA | arXiv:**2602.00541** | Transformer vs Mamba head-to-head at 120M/8192: gains are **backbone-agnostic**; "objective, not backbone, is the critical design choice." |
| Scaling Laws for EHR FMs | arXiv:**2505.22964** (2025) | Clinical utility improves to **~28M then saturates** (MIMIC is data-constrained). a≈0.58 params, b≈0.44 tokens. |
| RAVEN | arXiv:**2603.24562** (2026) | Data-constrained U-shaped loss-vs-size: optimal 47M for 10–50% subsets, 144M only at full 23M-patient cohort → **expanding the cohort keeps paying** (favors our federation). |
| Comet | arXiv:**2508.12104** (2025) | Qwen2 backbone, 8192 ctx. Compute-optimal token:param ≈ **1000:1**. |
| Federated vs Inference-Time Ensembling (31.6M GPT) | medRxiv **2026.04.24.26351702** | **d512/8L/8H/FFN2048 = 31.6M**; **untied embeddings beat tied +4% central / +7% FedAvg**; multi-label next-visit → zero-shot sigmoid. Our direct shape precedent. |
| HealthFormer | medRxiv **2026.03.25.26349262** | Continuous-time (log-Δt) ALiBi bias with head-specific slopes. |
| MOTOR | arXiv:**2301.03150** | Piecewise-exponential zero-shot TTE head + low-rank task embeddings; RoPE on days-since-birth. |
| SurvivEHR | npj Digit Med 9:546 (2026) / medRxiv 2025.08.04.25332916 | Decoder-only, competing-risk time-to-next-event, zero-shot forecasting. |
| TraCeR | arXiv:**2512.18129** (2025) | K parallel cause-specific hazard subnetworks; best-in-class CR calibration (IBS 0.111 vs DynamicDeepHit 0.230 on MIMIC-IV sepsis). |
| CR calibration | arXiv:**2602.00194** (2026) | D-calibration / AJ-K-cal metrics; deep CR models badly miscalibrated by default. |
| Genomics-into-EHR fusion | arXiv:**2510.23639** (2025) | Adapter-MLP → soft token inserted mid-sequence at the modality's timestamp (our note-fusion template). |
| Others | EHRMamba 2405.14567; HyMaTE 2509.24118; MedTok 2502.04397; DDP/FSDP crossover 2505.12832 | — |

**Gaps/UNCERTAIN (flagged by the agents):** no EHR-native xLSTM/RWKV FM exists — do not adopt. Mamba-vs-
Transformer "winner" is close/task-dependent; the robust finding is objective + time-aware positions >
backbone. Some Lee/ORA numbers were read from abstracts + tables, not full text — re-verify exact figures
before quoting in a paper.

---

## 4. NEXT STEPS — ordered, concrete, file-level

Do these roughly in order. Each is scoped to be a single clean commit. **Commit + push after each.**

### Step 1 — Reconcile config↔code for the finalized spec (mechanical, do first)
1. **Untied embeddings in `heads.py`.** `next_event_loss` currently does tied-embedding CE (reuses the
   input embedding as the LM head). With `tied_embeddings: false` (now in `configs/model.yaml`), add a
   **separate output projection** `nn.Linear(d, vocab, bias=False)` and use it for the next-event logits.
   Keep a `tie_weights` flag so the ablation arm can still tie. Update `src/model/encoder.py` similarly
   (it currently ties `lm_logits`).
2. **`configs/data.yaml` — verify CLIF 2.1 field names** against the real data dictionary (`clif_vitals`,
   `clif_labs`, etc.). The file is annotated "VERIFY field names." Confirm `storetime`/availability columns
   exist for event ordering; add unit-normalization config per concept.
3. Implement **soft discretization** and **forced clinical-threshold edges** in the tokenizer
   (`src/data/tokenize.py` for the ablation arm; and as a patch/config over tokenETL for the default path —
   see Step 4). Config keys already exist in `data.yaml` (`soft_discretization`, `soft_kernel_bins`,
   `forced_edges`).

### Step 2 — Run the Method-3 wedge on a REAL checkpoint (the first real result)
- Get a trained CLIFATRON checkpoint + its tokenized-narratives parquet for MIMIC / Rush / UChicago.
- **Verify column names** in `method3.py` against CLIFATRON's `external/clifatron/benchmark/` build script
  (currently assumed `sequence`/`label`/`sex,race,age_band` — the code says VERIFY).
- Run: `python -m src.eval.method3 --checkpoint <ckpt> --site MIMIC=... --site Rush=... --site UChicago=... --method both`
- **Headline to produce:** our probe vs their XGBoost (Method 1) on external-transport AUPRC + calibration,
  per test site. This is the smallest publishable unit and validates the whole thesis cheaply.
- Runs on ANY released checkpoint TODAY (frozen-probe mode needs no retraining).

### Step 3 — Phase-2: joint-pretrain our heads on the CLIFATRON backbone
- Use `head_adapter.CLIFATRONHeads.loss` (weights `w_ntp=0.2, w_cr=1.0, w_th=1.0, w_val=0.5`) with the
  NTP→TTE curriculum. This produces **zero-shot** threshold/CR predictions — the mechanism that makes
  external federation validation work (a new site needs no local labels).
- This is where the finalized tokenizer/trunk spec (untied emb, time-aware RoPE, deciles+soft) actually
  gets trained. On the 2× L40 box: bf16, DDP via torchrun, sequence packing.
- Add the **CR D-calibration / AJ-K-cal** metric (arXiv:2602.00194) to `metrics.py` and report it.

### Step 4 — Tokenization ablation (the empirical payoff of §2.1)
- Arms: CLIFATRON clinical bins (baseline) vs global deciles vs **deciles + soft discretization** vs
  continuous-fused (McCann). Lee predicts deciles+soft wins, especially on tail/threshold AUPRC.
- The clinical-bins baseline already lives in `external/clifatron/tokenETL/config/critical_illness_tokenization_final_with_intervals.csv`
  (their scheme is clinical-segment-anchored *quantiles*, not fixed ranges — describe it accurately).

### Step 5 — Ship `clif-validate/` (the federation deliverable)
- A turnkey package: frozen model + `clifpy`/tokenETL eval script. Each external CLIF site runs it on its
  LOCAL tables, auto-labels from standard CLIF fields (mortality, disposition/home/LTACH, IMV on/off,
  hypoxia, organ-failure thresholds), runs zero-shot, and returns **only aggregate + subgroup metrics**.
- Reuse `metrics.full_panel`. No raw data / labels / gradients leave any node.
- **Headline figure:** forest/box plot of AUROC/AUPRC/calibration across N external CLIF sites per outcome
  ("does it travel"). UChicago = CLIF origin site (Bhavani).

### Step 6 — Notes modality (v2 multimodal)
- Frozen **BioClinical ModernBERT** (arXiv:2506.10896) → 2-layer MLP → in-stream soft token at the note
  timestamp. **Pre-anchor notes only** (leakage rule). An oversized note-gain = leakage red flag.

---

## 5. Open items to VERIFY before relying on them
- [ ] CLIFATRON benchmark parquet column names (`sequence`/`label`/subgroup) — check `external/clifatron/benchmark/`.
- [ ] Which sites the released CLIFATRON checkpoint was trained on (leakage risk for our "external" claim).
- [ ] `storetime` vs `charttime` availability in each site's CLIF 2.1 tables (event-ordering correctness).
- [ ] Exact Lee/ORA figures (some read from abstracts) before quoting in a manuscript.
- [ ] CLIF v3.0 multimodal timeline (~July 2026 per notes) — first-mover window; confirm status.
- [ ] Whether the ~10k vocab actually fits after fused-decile expansion at all K target concepts.

---

## 6. Hard rules (do NOT violate)
1. **Treatments are model inputs, NEVER prediction targets.**
2. **Vocab = frozen CLIF mCIDE, applied identically to all sites — no cross-site pooling of raw data.**
3. **Retrospective reports / discharge summaries = LABEL source only; only pre-anchor notes may be features.**
4. **MIMIC-IV-Ext-CLIF is PhysioNet-credentialed; Rush + UChicago institutional — no data leaves its node.**
   No rented cloud without a compliant BAA/DUA. External validation returns only aggregate metrics.

---

## 7. Environment / mechanics

**Compute tiers (three options):**
- **Dev (this Mac):** torch NOT installed. Verify with `python3 -m py_compile <file>` and run metric checks
  via `uv run --with numpy --with scikit-learn python -c "..."`. `python` is absent — use `python3`.
- **Training (2× L40, Linux):** 48GB each, no NVLink, bf16, DDP via `torchrun`. `uv sync` installs deps.
  The default training box.
- **Azure hourly GPU (burst):** rent by the hour for bigger runs / faster ablation sweeps.
  **COMPLIANCE GATE (hard rule 4):** only inside a **BAA/DUA-covered Azure tenant** — each site's PHI
  (Rush, UChicago) may not leave its governed environment, and MIMIC's PhysioNet DUA governs its cloud use.
  Use the lab's compliant tenant; do NOT spin up an ad-hoc personal subscription for real data. Fine for
  code/synthetic-data smoke tests anywhere. (The lab already runs Azure Foundry/blob/containers.)
- **Package management:** `uv` only (`uv add` for libs, `uvx` to run). Never `pip install` into a shared interpreter.
- **Multi-machine workflow:** this repo is worked across computers. **Always `git pull` at session start;
  `git commit` + `git push` after each change** so no work is stranded. Data/checkpoints/`clif_config.json`
  live locally per machine (git-ignored) — only code + configs are committed.
- **Vendored upstream:** `external/clifatron/` is a snapshot (see its `VENDORED.md`); to re-sync, diff against
  a fresh clone of the upstream commit and apply by hand (it's a fork, not a live mirror).
