---
slug: /
id: overview
title: Overview — the full scientific pipeline
sidebar_position: 1
---

# CLIFATRON 2.0 — Scientific Workflow

CLIFATRON 2.0 is a **methods-upgrade layer** on
[CLIFATRON](https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON), the CLIF
consortium's compact (~30M-parameter) CLIF-native ICU foundation model. We keep CLIFATRON's
tokenizer, sequence packing, and trained backbone, and add the pieces that make a small ICU
model **transportable and clinically deployable**.

> **Thesis:** *one small model → many outcomes → many hospitals → one node (2× L40, no cluster).*

This site documents the **full planned scientific workflow** end to end, one stage per page,
with diagrams drawn directly from the implementation in `src/`.

:::note Single source of truth
The finalized design spec lives in `MEMORY.md` and `notes/NEXT_STEPS.md`. `notes/RESEARCH.md`
and `notes/METHODS.md` carry the evidence base but are marked **pre-pivot** — where they
disagree with `MEMORY.md`, `MEMORY.md` wins. These docs mirror the current (post-pivot) spec.
:::

:::tip What's new / where things stand
Every **data-free, unblocked** unit has landed — the codebase is a complete, CI-enforced, reproducible
methods artifact. New since the first docs: per-token **value-head normalization**, the full
**governance / trust / disclosure** machinery (Ed25519 release-trust, cumulative disclosure ledger,
artifact-classification policy), the **synthetic federation harness**, and **CI + one-command
reproduction**. What remains (real-site federation, method experiments, scaling) is gated on data /
GPU / governance, not code. See **[Governance, Trust & Reproducibility](./governance-trust.md)** and
**[Project Status & Roadmap](./project-status.md)**.
:::

---

## The end-to-end pipeline

```mermaid
flowchart TB
    subgraph DEV["DEVELOPMENT (data we hold — 3 sites)"]
        direction TB
        A["CLIF 2.1 parquet<br/>MIMIC · Rush · UChicago"] --> B["Tokenization<br/>fused code=bin · deciles<br/>soft discretization · RoPE"]
        B --> C["Backbone<br/>from-scratch Qwen3 ~30M (primary)<br/>· CLIFATRON Qwen2 0.5B wedge<br/>8192 ctx · untied emb"]
        C --> D["Our heads<br/>threshold-hazard · competing-risk<br/>value-regression · next-event"]
        D --> E["Joint pretrain<br/>NTP → TTE curriculum<br/>uncertainty + grad-norm balancing"]
    end

    E --> F["Method 3 wedge<br/>anchor state → probe vs XGBoost<br/>3×3 transportability matrix"]
    E --> G["Frozen zero-shot model<br/>threshold / competing-risk heads"]

    subgraph EXT["EXTERNAL VALIDATION (model-to-data)"]
        direction TB
        G --> H["Ship frozen model + turnkey eval<br/>to every other CLIF site"]
        H --> I["Site runs locally:<br/>tokenize → zero-shot → auto-label"]
        I --> J["Return ONLY aggregate + subgroup metrics<br/>no raw data / labels / gradients leave"]
    end

    F --> K["TRIPOD+AI panel<br/>AUROC · AUPRC · ECE · Brier<br/>calibration · DCA · fairness · LPE"]
    J --> L["Headline figure:<br/>forest plot of AUROC/AUPRC/calibration<br/>across N external CLIF sites"]

    classDef dev fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef ext fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef out fill:#fff3e0,stroke:#e65100,color:#0d1b2a;
    class A,B,C,D,E dev;
    class H,I,J ext;
    class F,G,K,L out;
```

---

## Why this is novel — the 4-way intersection

No single published model occupies the intersection this project targets. Each axis alone is
contested; the **conjunction**, executed CLIF-native, is the defensible headline.

```mermaid
mindmap
  root((CLIFATRON 2.0))
    CLIF-native
      open tooling, no DUA
      frozen mCIDE vocab
    Structured + notes ICU
      vitals / labs / meds as events
      pre-anchor notes only
    ~30M efficient
      utility saturates ~28M on MIMIC (arXiv:2505.22964)
      one node, 2× L40
    Federated real-data
      model-to-data validation
      aggregate-only return
    Threshold-TTE objective
      zero-shot multi-outcome
      calibrated survival heads
```

---

## The mechanism: one model answers many outcomes

The **threshold-hazard head** (ICareFM) is what lets a single trained model answer an
open-ended family of clinical questions with **no retraining**. At inference you query a
target concept, a threshold τ, and a direction; composite events combine univariate failure
probabilities under conditional independence.

```mermaid
flowchart LR
    Ht["Patient state H_t<br/>(hour-24 anchor)"] --> Q1["Query: MAP crosses &lt;65 within h?"]
    Ht --> Q2["Query: Lactate crosses &gt;2 within h?"]
    Q1 --> F1["F_MAP(h | H_t, &lt;65)"]
    Q2 --> F2["F_Lact(h | H_t, &gt;2)"]
    F1 --> COMP["composite_and<br/>= F_MAP · F_Lact"]
    F2 --> COMP
    COMP --> OUT["Circulatory failure risk<br/>(no retraining)"]

    classDef q fill:#f3e5f5,stroke:#6a1b9a,color:#0d1b2a;
    classDef f fill:#e1f5fe,stroke:#0277bd,color:#0d1b2a;
    class Q1,Q2 q;
    class F1,F2,COMP,OUT f;
```

*Implemented in `src/model/heads.py`:* `ThresholdHazardHead.cumulative_failure()`,
`composite_or()`, `composite_and()`.

---

## Sites — develop on 3, validate on the whole federation

```mermaid
flowchart TB
    subgraph HOLD["We hold the data (development)"]
        M["MIMIC-IV-Ext-CLIF 2.1"]
        R["Rush"]
        U["UChicago (CLIF origin site)"]
    end
    HOLD -->|"3×3 train-A / test-B matrix<br/>+ adaptation ladder + LPE + ensemble"| INT["Internal transportability"]

    subgraph FED["CLIF federation (external)"]
        S1["Site 1"]
        S2["Site 2"]
        SN["… Site N"]
    end
    FROZEN["Frozen model + turnkey eval script"] --> FED
    FED -->|"aggregate + subgroup metrics only"| HEAD["Headline: does it travel?"]

    classDef hold fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef fed fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class M,R,U hold;
    class S1,S2,SN fed;
```

---

## Non-negotiable rules

These constrain every stage of the pipeline.

| # | Rule | Where it bites |
|---|------|----------------|
| 1 | **Treatments are model inputs, never prediction targets** | Tokenization, target-concept selection |
| 2 | **Vocab = frozen CLIF mCIDE, applied identically to all sites — no cross-site raw pooling** | Tokenization, federation |
| 3 | **Retrospective reports / discharge summaries = label source only; only pre-anchor notes are features** | Notes modality, eval labeling |
| 4 | **`storetime`/availability ordering, not `charttime`** | Tokenization (no look-ahead) |
| 5 | **MIMIC is PhysioNet-credentialed; Rush + UChicago institutional — no data leaves its node** | Federation, compute |

---

## Read next

1. **[Data & Tokenization](./data-tokenization.md)** — CLIF parquet → fused decile tokens
2. **[Architecture](./architecture.md)** — CLIFATRON Qwen2 backbone + our four heads
3. **[Objectives & Training](./objectives-training.md)** — the marked-TTE loss stack + curriculum
4. **[Method 3 Wedge](./method3-wedge.md)** — the smallest publishable unit
5. **[Federated Validation](./federated-validation.md)** — model-to-data across the federation
6. **[Evaluation Panel](./evaluation-panel.md)** — TRIPOD+AI metrics
7. **[Ablations](./ablations.md)** — finetune-vs-scratch and tokenization arms
