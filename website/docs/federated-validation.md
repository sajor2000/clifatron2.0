---
id: federated-validation
title: Federated Validation
sidebar_position: 6
---

# Federated Validation (model-to-data)

The thesis axis CLIFATRON (single-center) cannot claim: ship a **frozen model + turnkey eval
script** to every other CLIF consortium site; each runs it on its **local** tables and returns
**only aggregate + subgroup metrics**. No raw data, labels, or gradients ever leave a node.
Implemented in `src/eval/clif_validate.py`, `src/eval/clif_auto_labeler.py`,
`src/eval/clif_forest_plot.py`.

---

## The model-to-data flow

```mermaid
flowchart TB
    subgraph HUB["Our node"]
        FROZEN["Frozen zero-shot checkpoint<br/>+ clif-validate package"]
    end
    FROZEN -->|"ship code + weights only"| SITE

    subgraph SITE["External CLIF site (runs locally)"]
        direction TB
        T["tokenize (tokenETL / clifpy)<br/>frozen mCIDE vocab"] --> Z["zero-shot threshold / CR heads<br/>(no local training)"]
        RAW["site's CLIF tables"] --> T
        RAW --> AL["auto-label from standard CLIF fields"]
        Z --> SC["score: metrics.full_panel"]
        AL --> SC
        SC --> J["metrics.json<br/>(aggregate + subgroup ONLY)"]
    end

    J -->|"return metrics only"| HUB2["Aggregate across N sites"]
    HUB2 --> FIG["Headline forest plot"]

    classDef hub fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef site fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef leave fill:#fff3e0,stroke:#e65100,color:#0d1b2a;
    class FROZEN,HUB2,FIG hub;
    class T,Z,AL,SC,RAW site;
    class J leave;
```

:::danger What crosses the node boundary
**Only** an aggregate/subgroup metrics JSON leaves a site. Raw rows, per-patient predictions,
labels, gradients, and identifiers **never** leave. The validator output is checked to contain
no `patient_id`, `hosp_id`, `sequence`, `token`, or `pos_min` fields (see
`tests/test_clif_validate.py`).
:::

---

## "Zero-shot" precisely — model is training-free, eval still needs labels

A common overclaim. The **model** needs no local training and no manual annotation. But
**computing the metrics** still requires ground-truth labels, which each site *auto-derives*
from its own CLIF fields.

```mermaid
flowchart LR
    subgraph FREE["Training-free (the claim)"]
        A["Model emits predictions<br/>with NO local fitting"]
    end
    subgraph NEED["Not label-free (the precision)"]
        B["AUROC / AUPRC / calibration<br/>REQUIRE ground-truth labels"]
        C["Labels auto-derived from CLIF fields<br/>(noisy phenotypes — audit per site)"]
    end
    A --> SCORE["Score predictions"]
    B --> SCORE
    C --> B

    classDef ok fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef care fill:#fff3e0,stroke:#e65100,color:#0d1b2a;
    class A ok;
    class B,C care;
```

---

## Auto-labeler — outcomes derivable from standard CLIF fields

To need no manual annotation, outcomes are restricted to those computable from standard CLIF
tables. Implemented in `src/eval/clif_auto_labeler.py`.

```mermaid
flowchart TB
    subgraph CLIF["Standard CLIF 2.1 fields"]
        H["clif_hospitalization<br/>discharge_category"]
        RS["clif_respiratory_support<br/>device_category"]
        MED["clif_medication_admin_continuous<br/>vasopressors"]
        ADT["clif_adt / icu_admit"]
    end
    H --> O1["in_hospital_mortality<br/>= discharge == Expired"]
    RS --> O2["new_imv_24h<br/>= new IMV ≤ 24h of ICU admit"]
    MED --> O3["new_vasopressor_24h<br/>= new pressor ≤ 24h"]
    ADT --> O2
    ADT --> O3

    O1 & O2 & O3 --> LBL["labels.parquet (local only)"]

    classDef f fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef o fill:#f3e5f5,stroke:#6a1b9a,color:#0d1b2a;
    class H,RS,MED,ADT f;
    class O1,O2,O3 o;
```

Other CLIF-derivable outcomes on the roadmap: discharge disposition (home/LTACH), IMV on/off,
hypoxia, organ-failure thresholds.

---

## Governance boundary

```mermaid
flowchart LR
    subgraph NODE["Governed node (per site)"]
        RAW["Raw CLIF (PHI)"]
        PRED["Per-patient predictions"]
        GRAD["Gradients"]
    end
    subgraph OUT["Leaves the node"]
        AGG["Aggregate metrics"]
        SUB["Subgroup metrics<br/>(sex / race / age / site)"]
    end
    NODE -.->|"❌ never"| OUT
    RAW --> LOCAL["compute locally"] --> AGG
    LOCAL --> SUB

    classDef node fill:#ffebee,stroke:#c62828,color:#0d1b2a;
    classDef out fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class RAW,PRED,GRAD node;
    class AGG,SUB out;
```

Fairness is reported aggregate (ICareFM precedent). Small-cell suppression for subgroup metrics
is an open item to specify before shipping.

---

## The headline figure

```mermaid
flowchart LR
    S1["Site 1 metrics.json"] --> AGG["clif_forest_plot"]
    S2["Site 2 metrics.json"] --> AGG
    SN["… Site N metrics.json"] --> AGG
    AGG --> FOREST["Forest / box plot:<br/>AUROC · AUPRC · calibration<br/>across N external CLIF sites, per outcome"]
    FOREST --> ANS["'Does it travel?' — the deployability result"]

    classDef out fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class FOREST,ANS out;
```

Run:

```bash
# at each external site (returns aggregate metrics only)
python -m src.eval.clif_validate \
      --checkpoint <ckpt> \
      --data /path/to/clif_parquet \
      --episode-artifact /path/to/episodes.parquet \
      --site-id SITE-07 \
      --release-id 2026-08-28-site07-v0 \
      --signing-key-file /secure/site07.key
# at the hub (metrics JSONs only)
python -m src.eval.clif_forest_plot --results results/SiteA.json results/SiteB.json
```

:::info Open design question — vocabulary transfer
The federation assumes the frozen mCIDE vocab covers each site's concepts. PORTER (arXiv:2606.24102)
shows fixed-vocabulary models can drop a large fraction of events on cross-site transfer. The
**TextCode** tokenization arm (language-grounded event descriptions) is the mitigation to
evaluate before relying on frozen-vocab turnkey transfer.
:::
