# AGENTS.md — CLIFATRON 2.0

> Read this first. It states the project **goal**, the **locked decisions**, and the
> **hard rules** an agent must not violate. For depth, follow the pointers at the bottom;
> `MEMORY.md` is the single source of truth where anything here is ambiguous.

---

## GOAL RE-CONFIRMED 2026-10-03 (product authority) — read this first

**Why we train the foundation model:** one CLIF-native model, trained once on whole
hospitalizations, that predicts many ICU outcomes better than hand-built features and transfers
to other hospitals. The AI contribution is Paper 1, claims 1–2 (threshold-aligned tokenization;
the combined time-to-event objective vs next-token at equal compute), tested on the CLIF task
suite and zero-shot threshold questions, with external validation. This is the reason the model
exists; it does not depend on any RCT statistics.

**What we do with it for extubation (Paper 2, built on the frozen trunk):**
1. **Failure risk at extubation** — calibrated 7-day reintubation-or-death risk from the full
   pre-extubation trajectory, placed on the ATS guideline low/high-risk axis. It must beat the
   guideline rule, the trial risk-factor count, a published score and gradient boosting;
   externally validated at UChicago.
2. **Injected-device risk** — cut the history at extubation, inject one device token (NIV, HFNC,
   nasal cannula / conventional O2), read the 7-day risk under each: "patients like this one do
   worse on nasal cannula than on NIV". Checked against the trial pattern (NIV's advantage over
   HFNC grows with baseline risk; HFNC beats conventional O2 in low-risk; NIV in hypercapnic).
3. **Safeguards only:** a classical confounding-adjusted comparison on the same patients and a few
   negative controls. The registered agreement-margin rubric (margins, pooling, feasibility
   floor, stop rules) is **frozen as optional** — code kept, off the critical path
   (pooled-margin work saved at `output/patches/pooled-margins-wip.patch`, local only).
4. **No device recommendation.** Per-device risks stay exploratory. Rollouts are illustrations,
   never evidence.

**Context length:** stay at 8,192 tokens (97% of MIMIC hospitalizations and 98.8% of
pre-extubation histories fit; ICU literature shows no gain past 8K — METHOD peaks near 3–6K,
Wornow 2025 8K→16K within noise). Long stays use continuation windows with the admission header
re-inserted; RoPE base raised for minute positions; per-site context-length report at
pre-flight; a 4K-vs-8K inference check after training. Revisit 16K only if Rush is much denser.

Clinical decisions: `docs/decisions/2026-10-03-clinical-decisions.md`.

---

## The goal (one paragraph)

> **Three-claim framing locked 2026-10-03** (plan:
> `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md`; evidence:
> `notes/ai-novelty-audit.md`). It supersedes the 2026-08-27 headline "two paths, run as a
> ladder". The ladder is kept under the locked decisions below; it no longer leads.

CLIFATRON 2.0 is **one from-scratch, CLIF-native ICU foundation model** — a ~30M-param (design
target) Qwen2-architecture decoder with a marked time-to-event objective (threshold-hazard +
competing-risk CIF + value-regression mark) in place of pure next-token prediction — and two
papers built on it.

**Paper 1 makes three falsifiable claims about that one model**, each with a pre-specified test
that can fail:

1. **Threshold-aligned tokenization.** Bin edges at clinical decision thresholds, shared with the
   training objective, improve prediction and calibration at those thresholds.
2. **Combined time-to-event objective.** The combined objective beats next-token training at
   equal compute.
3. **Extubation application.** With the post-extubation device token injected at the real
   extubation, the model reproduces the randomized-trial pattern of who benefits from NIV and
   HFNC at least as well as a classical emulation.

**Paper 2 is an externally validated extubation-failure risk model.** UChicago validation belongs
to Paper 2. Per-device risks are exploratory, and the output makes no device recommendation.

A null result on any claim is reported as such. Everything is evaluated retrospectively; a
clinician-facing tool is not part of this work. Development data are MIMIC-IV-Ext-CLIF and Rush,
trained together on the Rush L40 node; UChicago is external validation by model-to-data;
Northwestern joins later and is not active scope.

The released CLIFATRON Qwen2 checkpoint (0.5B) stays a *larger comparator*: it is reported as a
frozen comparator row when it is available and carries none of the three claims. Qwen3-arch
(free QK-Norm) is a measured ablation row, not the primary path — see the locked design
decisions below.

From CLIFATRON we **keep**: the fused `code=bin` token format, the physician-designed
clinical-segment CSV (1267 segment rows, 92 measurements — the bin boundaries you built), the
mCIDE vocabulary, the CLIF 2.1 data format, the treatment-as-input (never a trunk target) rule,
the Qwen2 backbone for both paths, and the document-isolation sequence-packing approach. The
tokenization ETL was rewritten from a multi-file pandas pipeline to a single-file polars pipeline
supporting soft discretization, admission-relative RoPE, and `build_edges` dispatch — the binning
data (segmentation CSV) and fused-token concept are preserved.

What we **add**: a **threshold-conditioned time-to-event** objective (ICareFM)
+ **competing-risk CIF** (SurvivEHR) + a **value-regression "mark"** head (ORA), trained together
in place of pure next-token prediction; bin edges shared between the tokenizer and the threshold
queries; **zero-shot, training-free** survival heads; and **external validation by
model-to-data** (UChicago now, through `clif-validate`; other CLIF sites later), with a full
**TRIPOD+AI** calibration / decision-curve / fairness eval panel. Each component is published;
what is ours is their pairing, the controlled tests, and the CLIF-native execution.

**Thesis:** *one small model → many outcomes → many hospitals → one node (2× L40, no cluster).*

---

## What our novelty IS and IS NOT (locked 2026-08-27, grounded in 2026 literature; IS revised 2026-10-03)

**IS (revised 2026-10-03):** an **open, CLIF-native** ICU foundation model whose AI contribution
is shown by **three tests that could have failed** — what threshold-aligned tokenization adds,
what the combined time-to-event objective adds at equal compute, and whether the model, with the
device injected at extubation, reproduces what randomized trials established about NIV and HFNC.
Contribution = the pairing of published components + the controlled tests + CLIF-native execution.

> **Priority-claim headline superseded 2026-10-03.** The 2026-08-27 statement — "the first
> **open, CLIF-native** ICU foundation model with a **threshold-TTE objective** validated by
> **model-to-data across real CLIF-consortium hospitals**; novelty = integration + CLIF-native
> execution + real-federation deployment" — is kept as the record. It no longer leads: generic
> and federated CLIF GEMs and a CLIF tokenization benchmark are published (Burkhart 2026,
> arXiv 2608.02939; Lee 2026, arXiv 2604.16775), so the AI contribution has to lead, and a
> general claim that a new tokenization improves performance would not survive review. What that
> work left open is what the three claims test. Model-to-data validation and the open
> `clif-validate` package remain deliverables.

> **First-mover framing retired 2026-08-28.** The ~July 2026 CLIF v3.0 multimodal window this
> framing depended on has passed, so urgency is no longer an argument and no gate should be
> justified by it. The *priority* claim above — first **open**, CLIF-native, threshold-TTE,
> real-federation — stands on its own and is unaffected.

**IS NOT:** a new method. The 2026 literature already owns every method piece — ORA
(marked-TTE), ICareFM (threshold-directional dual-zero-shot, but on *ricu*, DUA-gated), SurvivEHR
(competing-risk), Elemento (no-data-sharing ensembling). **Do not claim method invention.** The
open, deployable, CLIF-native execution is the defensible contribution. Also not this work
(2026-10-03): a per-patient device recommendation, proof of cause (claim 3 is agreement with the
trials' pattern), reward-based post-training, or federated training methods.

**"Open" means: the package, its source, and its bundle-compatibility contract are publicly
obtainable with no DUA and no per-site approval. Trained-weight bundles remain signed and governed.**
That split is what keeps the ICareFM contrast honest — the tooling is inspectable and runnable by
anyone, which is the claim the project can actually keep.

The `clif-validate/` open shippable package is the headline artifact that distinguishes us from
DUA-gated ICareFM — treat it as a deliverable, not plumbing.

---

## Locked design decisions (do not re-litigate — change only with new evidence)

- **Framing (locked 2026-10-03):** Paper 1 = three falsifiable claims about one from-scratch
  model (threshold-aligned tokenization; the combined time-to-event objective vs next-token at
  equal compute; the extubation application). Paper 2 = the externally validated
  extubation-failure risk model; UChicago validation belongs to Paper 2; per-device risks are
  exploratory; no device recommendation. Claims are read only from the pre-specified tests.
- **Extubation study (locked 2026-10-03):** one foundation model, separate downstream study — the
  trunk is reused unchanged. The decision studied is the device at the actual extubation, not
  extubation timing. Reintubation is read from a dedicated study head; respiratory support stays
  input-only (hard rule #1 and its scoped amendment). A go/no-go design audit comes first; MIMIC
  is exploratory and Rush confirmatory. Rollouts are descriptive only. The device-choice
  (propensity) model is an estimator nuisance, never a model target. No reward-based post-training.
- **Sites (locked 2026-10-03):** development on MIMIC + Rush on the Rush L40 node; UChicago is
  external validation now, by model-to-data; Northwestern later. This supersedes "3 dev sites
  (MIMIC + Rush + UChicago)" for these two papers.
- **Objective (where the novelty lives):** threshold-hazard (primary) + competing-risk CIF +
  value-regression (ORA mark) + low-weight next-event (0.2). NTP→TTE curriculum. This is the lever;
  the backbone is a footnote.
- **Primary paper = from-scratch Qwen2-arch decoder + the objective, ~30M, fully ours.** Run as a
  ladder: (1) frozen-probe **Method-3 wedge** on a CLIFATRON Qwen2 checkpoint (cheap first result) →
  (2) from-scratch **Qwen2** pretrain (novel headline) → (3) the two = the finetune-vs-scratch ablation.
  *(2026-10-03: the three claims are tested on the from-scratch model. The wedge is the larger
  frozen comparator row, reported when a checkpoint is available; it does not gate the claims.)*
- **Backbone:** Qwen-family transformer. **From-scratch → Qwen2-arch** (standard pre-norm, no QK-Norm);
  **attach/wedge path → Qwen2** (must match CLIFATRON's checkpoint). Qwen3-arch (free QK-Norm) is
  a measured ablation row so "Qwen2 vs Qwen3" is a quantified finding, not an assertion.
- **Size:** our own model targets **~30M** (d512×8L×8H). CLIFATRON's Qwen2 checkpoint we attach to is
  **0.5B** — always state it as a *larger comparator*, never as our compact model.
  *(2026-10-03: the primary arm is also trained at about 3M and 10M; the reported size is the
  measured one, not the design target.)*
- **Tokenizer:** fused `code=bin`, **physician-designed clinical-segment bins** (primary, revised 2026-09-02;
  population deciles = `decile_ablation` arm), soft discretization, forced clinical-threshold
  edges, storetime ordering, **untied embeddings**, **8192** context.
- **TextCode / language-grounded arm is elevated to a real transfer-robustness arm** (PORTER 2026:
  frozen-vocab drops ~69% of events on cross-site transfer). Frozen mCIDE stays primary; TextCode is
  the ablation that tests cross-site robustness. *(2026-10-03: reported as an ablation, not a
  headline claim, alongside event-order vs admission-relative time positions. PORTER's 69% was on
  an unharmonized transfer; under mCIDE harmonization the question is whether language grounding
  adds anything.)*
- **Federation:** model-to-data, **aggregate + subgroup metrics only**, nothing raw leaves a node.
  "Label-free" refers to the MODEL only — evaluation still auto-derives ground-truth labels locally,
  which is a validity dependency to audit per site. Add small-cell suppression (n<10) before shipping.

---

## Hard rules (NEVER violate)

1. **Treatments are model inputs, NEVER prediction targets of the trunk.** Scoped amendment
   (2026-10-03): study heads downstream of the frozen trunk may use trial endpoints defined by a
   treatment event (for example reintubation) as labels; the trunk itself never trains on
   treatment targets, and such endpoints must be declared label-only study endpoints.
   Declared in `configs/cohort.yaml → study_endpoints` (never under `outcomes`); enforced by
   `tests/test_data_config.py`.
2. **Vocab = frozen CLIF mCIDE, applied identically to all sites — no cross-site pooling of raw data.**
3. **Retrospective reports / discharge summaries = LABEL source only; only pre-anchor notes may be features.**
   An oversized note-gain is a leakage flag.
4. **`storetime`/availability ordering, not `charttime`** (no look-ahead on when a value was knowable).
5. **No data leaves its node.** MIMIC-IV-Ext-CLIF is PhysioNet-credentialed; Rush + UChicago are
   institutional. External validation returns aggregate metrics only. No rented cloud without a
   compliant BAA/DUA (Azure only inside the lab's governed tenant).

---

## Known blockers before any real training run

- ~~Value-head loss unnormalized~~ **RESOLVED** — `src/data/value_stats.py` freezes per-token robust
  value stats (vocab-hash-bound); `pretrain.py --value-stats` applies them and fails closed if missing.
- **Training-readiness fixes from the 2026-10-03 audit — IN PROGRESS** under
  `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md`: the multi-GPU (DDP) crash when a
  head is skipped, the full-hospitalization loader, curriculum wiring, and the threshold-head time
  grid. No base checkpoint exists until they land.
- **Rush data not staged on the L40 box** (only MIMIC: 546,028 stays / ~134M events). Claim 3 is
  confirmed at Rush and development needs it for power. Override data dir with `CLIF_DATA_DIR`.
  *(Revised 2026-10-03: was "Rush + UChicago data not on the L40 box". UChicago is external
  validation by model-to-data and is not staged here.)*
- **Written approval to transfer Rush-derived weights to UChicago (gate G5) not yet obtained.**
- **`clif-validate` is not partner-ready** for the UChicago run.
- **MIMIC calendar period is not evaluable** without the MIMIC-IV `anchor_year_group` table
  (MIMIC dates are shifted per patient); the by-unit table is unaffected.
- **No CLIFATRON checkpoint staged** yet (needed for the Method-3 wedge / larger-comparator row;
  it does not gate the three claims).
- **transformers is v5** — verify `head_adapter.anchor_state` against a real checkpoint
  (`output_hidden_states` API changed in v5; final hidden state may be normalized).

---

## Environment & workflow

- **Package management:** `uv` only (`uv sync`, `uv run`). Never `pip install` into a shared interpreter.
- **Compute:** dev on Mac (MPS, smoke tests) or this **2× L40 Linux box `rudu-hpcg004`** (48GB each, no
  NVLink, bf16, DDP via `torchrun`). Note: `nvidia-smi` currently fails on a driver/library mismatch —
  torch CUDA still allocates, but reboot before long multi-GPU runs.
- **Tests:** `CLIF_DATA_DIR=~/Data/clif-source uv run --with pytest python -m pytest tests/ -q`
  (data-gated tests skip cleanly when no CLIF data is present).
- **Docs site:** `website/` (Docusaurus, Mermaid). `cd website && npm run build`. Auto-deploys to
  GitHub Pages on push to `main` (when the repo is public / Pages is enabled).
- **Git:** work is done across machines — `git pull` at session start, commit + push after each change.
  Data / checkpoints / `clif_config.json` are per-machine and git-ignored; only code + configs are committed.
- **Do NOT commit:** data (`*.parquet`), checkpoints, `bin/` binaries, `.venv/`, `node_modules/`, `.agents/`,
  vendored agent skills (`agent/`, `.claude/skills/`, `skills-lock.json`) — install those globally.

---

## Where to look for detail

| For… | Read |
|------|------|
| **Single source of truth** for the spec + locked decisions | `MEMORY.md` |
| Three-claim paper + extubation risk model plan (2026-10-03) | `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md` |
| Evidence for the three claims and the extubation study | `notes/ai-novelty-audit.md`, `notes/extubation-evidence-review.md` |
| Ordered, file-level next steps + 2026 evidence tables | `notes/NEXT_STEPS.md` |
| Build-on-CLIFATRON integration plan | `notes/INTEGRATION.md` |
| Literature/evidence (⚠ pre-pivot design spec is superseded) | `notes/RESEARCH.md`, `notes/METHODS.md` |
| The rendered scientific-workflow docs (diagrams) | `website/docs/` |
| Keeper code | `src/model/heads.py`, `src/model/head_adapter.py`, `src/eval/metrics.py`, `src/eval/method3.py` |

**Precedence:** if any two documents disagree, `MEMORY.md` wins.
