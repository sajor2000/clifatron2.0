---
id: method3-wedge
title: Method 3 — the wedge
sidebar_position: 5
---

# Method 3 — the wedge (smallest publishable unit)

Attach our calibrated survival/probe heads to a **released CLIFATRON checkpoint's** hour-24
anchor hidden state, and beat their **Method 1** (XGBoost-on-embeddings) on AUPRC/calibration
and **Method 2** (Monte-Carlo rollout) on cost — on CLIFATRON's own 4-task benchmark, across
Site 1 / Site 2 / Site 3. Implemented in `src/eval/method3.py`.

:::tip Why this is the wedge
It runs on **any released checkpoint today** in frozen-probe mode — no retraining, minimal
infra — yet it validates the whole objective thesis. It is the cheapest publishable result.
:::

---

## The three methods compared

```mermaid
flowchart TB
    CKPT["Trained CLIFATRON checkpoint<br/>+ tokenized narratives (first 24h)"] --> ANCHOR["Extract hour-24 anchor state H_t<br/>(frozen forward pass)"]

    ANCHOR --> M1["Method 1 (theirs)<br/>XGBoost on embeddings<br/>needs local labels to fit"]
    ANCHOR --> M3["Method 3 (ours)<br/>calibrated probe / survival head<br/>+ temperature scaling"]
    CKPT --> M2["Method 2 (theirs)<br/>Monte-Carlo rollout<br/>expensive, simulation variance"]

    M1 --> CMP{Head-to-head}
    M2 --> CMP
    M3 --> CMP
    CMP --> WIN["Ours: better AUPRC + calibration<br/>than M1; cheaper + calibrated vs M2"]

    classDef theirs fill:#eceff1,stroke:#546e7a,color:#0d1b2a;
    classDef ours fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class M1,M2 theirs;
    class M3,WIN ours;
```

---

## Execution sequence

```mermaid
sequenceDiagram
    autonumber
    participant D as Site parquet (local)
    participant BB as CLIFATRON backbone
    participant AE as extract_anchor_states()
    participant P as Probe / survival head
    participant X as XGBoost (Method 1)
    participant MET as metrics.full_panel

    D->>BB: load_site() → sequences, labels, subgroups
    BB->>AE: frozen forward, output_hidden_states=True
    AE->>AE: H_t = last real token (attention_mask.sum-1)
    AE->>P: anchor states (train split)
    AE->>X: anchor states (train split)
    P->>MET: predicted probs (+ temperature scale)
    X->>MET: predicted probs
    MET-->>D: AUROC · AUPRC · ECE · Brier · calib · DCA per cell
```

Two head modes:

- **`probe`** — a binary `TaskHead` on the frozen anchor state. Runs on any checkpoint today; the
  fair head-to-head vs XGBoost.
- **`zero_shot`** — `CompetingRiskHead` / `ThresholdHazardHead`. Needs a checkpoint pretrained
  *with* our heads (the joint phase); gives label-free predictions — the mechanism that makes
  external federation validation possible.

---

## The 3×3 transportability matrix

The money figure of the development cohort. Rows = train site, columns = test site. The
diagonal is internal performance; off-diagonal is external transport. **Data never crosses
sites** — each site fits locally and the matrix is assembled from per-site metric JSONs.

```mermaid
flowchart TB
    subgraph MAT["3×3 matrix — AUROC / AUPRC / calibration-slope per cell"]
        direction TB
        R1["train Site 1 → test Site 1 (internal)"]
        R2["train Site 1 → test Site 2 (transport)"]
        R3["train Site 1 → test Site 3 (transport)"]
        R4["train Site 2 → test Site 1 (transport)"]
        R5["train Site 2 → test Site 2 (internal)"]
        R6["train Site 2 → test Site 3 (transport)"]
        R7["train Site 3 → test Site 1 (transport)"]
        R8["train Site 3 → test Site 2 (transport)"]
        R9["train Site 3 → test Site 3 (internal)"]
    end
    MAT --> ENS["Elemento ensemble column<br/>mean of the 3 site models' probs"]
    MAT --> LADDER["Adaptation ladder per off-diagonal cell"]

    classDef diag fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef off fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    class R1,R5,R9 diag;
    class R2,R3,R4,R6,R7,R8 off;
```

---

## The adaptation ladder

For every off-diagonal (transport) cell, report three rungs — from cheapest/most-portable to
most-adapted — plus the Local Patient Equivalence (LPE) at each rung.

```mermaid
flowchart LR
    Z["Rung 1 · as-is zero-shot<br/>no source data, no local labels"] --> T["Rung 2 · recalibrate only<br/>temperature scaling (cheap)"]
    T --> F["Rung 3 · fine-tune<br/>on local labels"]
    Z -. "LPE" .-> M["Local Patient Equivalence<br/>= smallest local-LightGBM<br/>training size matching the FM"]
    T -. "LPE" .-> M
    F -. "LPE" .-> M

    classDef r fill:#fff8e1,stroke:#f9a825,color:#0d1b2a;
    class Z,T,F r;
```

*Baseline to beat:* plain ERM + recalibration is hard-to-beat — it is included as a comparator,
not omitted.

---

## Running it

```bash
python -m src.eval.method3 \
  --checkpoint /path/to/clifatron_checkpoint \
  --site MIMIC=/path/mimic_narratives.parquet \
  --site Rush=/path/rush_narratives.parquet \
  --site UChicago=/path/uchicago_narratives.parquet \
  --method both
```

:::warning Verify before running
Confirm CLIFATRON's benchmark parquet column names (`sequence` / `label` / subgroup columns)
against its `build_benchmark.py`, and which sites the released checkpoint was trained on (a
leakage risk for the "external" claim).
:::
