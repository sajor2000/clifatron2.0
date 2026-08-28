---
id: evaluation-panel
title: Evaluation Panel (TRIPOD+AI)
sidebar_position: 7
---

# Evaluation Panel — TRIPOD+AI

Every (task, site) cell runs through one panel so results compose across the internal matrix
and the external federation. Implemented in `src/eval/metrics.py` (pure numpy + sklearn; torch
only inside temperature scaling). The headline metric is **net benefit**, not AUROC alone.

---

## The metric pipeline (order matters)

```mermaid
flowchart TB
    IN["predicted probs p · labels y · logits"] --> RECAL["temperature_scale(logits, y)<br/>Cadence single-scalar T*"]
    RECAL --> P2["recalibrated p = σ(logits / T*)"]

    P2 --> DISC["Discrimination<br/>AUROC + AUPRC"]
    P2 --> CAL["Calibration<br/>ECE · Brier · slope · intercept · ICI"]
    P2 --> DCA["Clinical utility<br/>net benefit / decision-curve<br/>(AFTER recalibration)"]
    P2 --> SUB["Fairness<br/>subgroup_panel(sex, race, age, site)"]
    IN --> LPE["Label efficiency<br/>local_patient_equivalence (LPE)"]

    DISC & CAL & DCA & SUB & LPE --> CELL["full_panel → one (task, site) cell"]

    classDef pre fill:#fff8e1,stroke:#f9a825,color:#0d1b2a;
    classDef metric fill:#e1f5fe,stroke:#0277bd,color:#0d1b2a;
    class RECAL,P2 pre;
    class DISC,CAL,DCA,SUB,LPE metric;
```

:::warning DCA assumes calibrated probabilities
Decision-curve analysis is computed **after** temperature scaling — net benefit is only
meaningful on calibrated probabilities. The panel enforces this ordering in `full_panel`.
:::

---

## The full metric set

```mermaid
mindmap
  root((TRIPOD+AI panel))
    Discrimination
      AUROC
      AUPRC (imbalanced ICU)
    Calibration
      ECE
      Brier
      calibration slope
      intercept-in-the-large
      ICI
    Recalibration
      temperature scaling T*
    Clinical utility
      net benefit / DCA
    Label efficiency
      LPE (ICareFM)
    Fairness
      per-subgroup AUROC/AUPRC/calibration
    Competing-risk calibration
      D-calibration
      Aalen-Johansen K-cal
```

| Group | Metrics | Function |
|-------|---------|----------|
| Discrimination | AUROC, AUPRC | `score()` |
| Calibration | ECE, Brier, slope, intercept, ICI | `full_panel()` |
| Recalibration | temperature T* | `temperature_scale()` |
| Clinical utility | net benefit / DCA | `net_benefit()` |
| Label efficiency | LPE | `local_patient_equivalence()` |
| Fairness | subgroup AUROC/AUPRC/calibration | `subgroup_panel()` |
| Competing-risk | D-calibration, AJ K-cal | `cr_d_calibration()`, `aj_k_calibration()` |

---

## Why AUPRC *and* calibration, not AUROC alone

ICU outcomes are highly imbalanced (LTACH ~3.6% positive), and calibration degrades faster
than discrimination off-site. A model can look great on AUROC and still be clinically useless.

```mermaid
flowchart LR
    AUROC["AUROC alone<br/>(insensitive to imbalance)"] -->|"misleading on rare outcomes"| GAP["Deployment gap"]
    AUPRC["+ AUPRC"] --> GOOD
    CAL["+ calibration slope"] --> GOOD
    DCA["+ net benefit / DCA"] --> GOOD["Clinically honest evaluation"]

    classDef bad fill:#ffebee,stroke:#c62828,color:#0d1b2a;
    classDef good fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class AUROC,GAP bad;
    class AUPRC,CAL,DCA,GOOD good;
```

---

## Competing-risk calibration

Deep competing-risk models are badly miscalibrated by default. Two dedicated checks
(arXiv:2602.00194) go beyond the binary panel: **D-calibration** (a chi-squared uniformity
test on the CIF at observed event times) and **Aalen-Johansen K-calibration** (per-cause
predicted-vs-observed by decile).

```mermaid
flowchart TB
    CIF["cause-specific CIF<br/>[N, K, H]"] --> DC["cr_d_calibration<br/>χ² uniformity of CIF at event times"]
    CIF --> AJ["aj_k_calibration<br/>per-cause decile: predicted vs observed"]
    DC --> P["D-calib p-value + per-bin histogram"]
    AJ --> S["per-cause calibration slope"]

    classDef m fill:#e1f5fe,stroke:#0277bd,color:#0d1b2a;
    class DC,AJ m;
```

---

## Selective prediction — defer the uncertain cases

The threshold head can abstain: `predict_with_confidence()` returns both a failure probability
and a confidence derived from the variance of the cumulative hazard across time bins. High
variance → low confidence → candidate for deferral to human review.

```mermaid
flowchart LR
    H["H_t + query (target, τ, dir)"] --> HAZ["hazard over time bins"]
    HAZ --> F["failure prob F_horizon"]
    HAZ --> VAR["cumulative hazard variance"]
    VAR --> CONF["confidence = 1 - tanh(√Σ var)"]
    CONF --> GATE{confidence ≥ threshold?}
    GATE -->|"yes"| AUTO["automated prediction"]
    GATE -->|"no"| DEFER["defer to clinician"]

    classDef d fill:#fff3e0,stroke:#e65100,color:#0d1b2a;
    class DEFER d;
```

The deferral threshold is set at calibration time and tuned per-outcome on a held-out set.

---

## Reporting standard

**TRIPOD+AI** (BMJ 2024;385:e078378). New items this project explicitly hits: subgroup
performance, open-science (protocol registration, data + code sharing), and external validation
on ≥1 unseen site. Baselines are mandatory: **MEDS-Tab XGBoost** as the strong ML floor, plus
the CLIFATRON Method-1 XGBoost-on-embeddings comparator.
