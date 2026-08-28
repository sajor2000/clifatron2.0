---
id: objectives-training
title: Objectives & Training
sidebar_position: 4
---

# Objectives & Training

The core methodological upgrade: replace pure next-token prediction (the weakest objective per
ORA/MOTOR/ICareFM) with a **marked time-to-event** stack. Implemented in
`src/model/head_adapter.py → CLIFATRONHeads.loss` and scheduled by
`src/train/curriculum.py`.

---

## The composite loss

```mermaid
flowchart TB
    H["Hidden states H / anchor H_t"] --> CR["CompetingRisk loss<br/>discrete-time CIF NLL"]
    H --> TH["ThresholdHazard loss<br/>discrete-time hazard NLL, random τ"]
    H --> VR["ValueRegression loss<br/>Gaussian NLL (the ORA mark)"]
    H --> NE["NextEvent loss<br/>cross-entropy (aux, joint mode only)"]

    CR -->|"w_cr = 1.0"| SUM["Total loss"]
    TH -->|"w_th = 1.0"| SUM
    VR -->|"w_val = 0.5"| SUM
    NE -->|"w_ntp = 0.2"| SUM

    classDef primary fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef aux fill:#fff8e1,stroke:#f9a825,color:#0d1b2a;
    class CR,TH primary;
    class VR,NE aux;
```

| Term | Weight | Rationale |
|------|:------:|-----------|
| Competing-risk CIF | `1.0` | Calibrated time-to-next-event over competing types |
| Threshold hazard | `1.0` | The zero-shot multi-outcome engine (primary) |
| Value regression | `0.5` | The ORA "mark" — lifts physiology tasks |
| Next-event (NTP) | `0.2` | Low-weight aux; retains open-ended zero-shot; **joint mode only** |

:::note NTP only when the backbone trains
`next-event` loss is added only when `freeze_backbone=False` — a frozen probe reads the
backbone's existing representation, so re-fitting a next-token head would be pointless.
:::

---

## The NTP → TTE curriculum

Warm up on next-token prediction to stabilize token embeddings, then phase in the survival
heads. Three phases, exactly as `curriculum_weights(step, total_steps)` returns them.

```mermaid
flowchart LR
    subgraph P1["Phase 1 · Warmup (0 → 15%)"]
        direction TB
        A1["NTP only<br/>w_ntp = 1.0"]
        A2["TTE heads FROZEN<br/>train_heads = False"]
    end
    subgraph P2["Phase 2 · Transition (15% → 20%)"]
        direction TB
        B1["Linear blend<br/>w_cr, w_th, w_val ramp 0→full"]
        B2["TTE heads unfrozen<br/>w_ntp drops to 0.2"]
    end
    subgraph P3["Phase 3 · Mixed (20% → 100%)"]
        direction TB
        C1["Full ORA marked-TTE<br/>w_cr=1, w_th=1, w_val=0.5"]
        C2["NTP as low-weight aux<br/>w_ntp = 0.2"]
    end
    P1 --> P2 --> P3
```

### Weight schedule over training

```mermaid
xychart-beta
    title "Loss weights across the NTP→TTE curriculum"
    x-axis "Training progress (%)" [0, 10, 15, 17, 20, 60, 100]
    y-axis "Weight" 0 --> 1
    line "w_ntp" [1, 1, 1, 0.2, 0.2, 0.2, 0.2]
    line "w_cr / w_th" [0, 0, 0, 0.5, 1, 1, 1]
    line "w_val" [0, 0, 0, 0.25, 0.5, 0.5, 0.5]
```

*Boundaries:* `warmup_frac = 0.15`, `transition_frac = 0.05`. During transition the TTE weights
scale linearly with `progress`; `w_val` ramps to `0.5·progress`.

---

## Loss balancing — don't let mortality starve the tails

The dense mortality signal can dominate sparse auxiliaries (the "signal-balance problem").
Two mechanisms keep the objective balanced.

```mermaid
flowchart TB
    L["Per-task losses"] --> UNC["Uncertainty weighting<br/>(Kendall 1/2σ² learned per task)"]
    UNC --> GN["Grad-norm normalization<br/>to the primary task's gradient"]
    GN --> STEP["Optimizer step"]
    L -.->|"robust fallback"| SUM["Just sum TTE loss over<br/>many random τ per step<br/>(MOTOR/ORA default)"]
    SUM --> STEP

    classDef m fill:#e1f5fe,stroke:#0277bd,color:#0d1b2a;
    class UNC,GN,SUM m;
```

---

## Two training entry points

```mermaid
flowchart TB
    CKPT["Released CLIFATRON checkpoint"] --> MODE{Training mode}

    MODE -->|"freeze_backbone=True"| FROZEN["Frozen probe<br/>train only the heads on local labels<br/>→ runs on ANY checkpoint TODAY"]
    MODE -->|"freeze_backbone=False"| JOINT["Joint fine-tune<br/>unfreeze + NTP→TTE curriculum<br/>→ produces zero-shot survival heads"]

    FROZEN --> M3["Method 3 wedge (probe mode)"]
    JOINT --> FED["Frozen zero-shot model<br/>for federated validation"]

    classDef a fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef b fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class FROZEN,M3 a;
    class JOINT,FED b;
```

**Systems:** 2× L40 (48GB, no NVLink), bf16, DDP via `torchrun`, per-patient sequence packing.
FSDP is *not* used (only pays off past ~2.3B params and is worse without NVLink).
`src/train/joint_pretrain.py` drives the joint path; `src/train/run_arm.py` drives the ablation
arms.

:::warning Known tuning issue
On real MIMIC the value-regression loss is currently unnormalized and dominates (raw lab
magnitudes — creatinine, platelet counts in the thousands). Per-concept value scaling is needed
before Step-3 joint pretraining. See `notes/TAKEOVER_PROMPT.md`.
:::
