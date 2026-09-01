---
id: ablations
title: Ablations
sidebar_position: 8
---

# Ablations

Two design decisions are not asserted — they are empirically tested against the same backbone
and tasks. Configured by `configs/ablation.yaml` (finetune-vs-scratch) and
`configs/tokenization_ablation.yaml` (representation).

---

## Finetune vs train-new — the 4 arms

The central "build ON CLIFATRON or train from scratch?" question. Hypothesis
(`configs/ablation.yaml`): **frozen-backbone head-training > joint fine-tune > from-scratch >
no-pretrain** for in-domain tasks; from-scratch may only close the gap on *transfer*.

```mermaid
flowchart TB
    DATA["Same CLIF data · same tasks · same metric panel"] --> A1 & A2 & A3 & A4

    A1["1 · Frozen backbone + heads<br/>CLIFATRON Qwen2 frozen<br/>train only our heads · lr 1e-3 · 20k steps"]
    A2["2 · Joint fine-tune<br/>CLIFATRON init, UNFREEZE<br/>NTP→TTE curriculum · lr [5e-5, 1e-3] · 30k"]
    A3["3 · From scratch<br/>random-init CLIFEncoder<br/>NTP→TTE · lr 3e-4 · 60k steps"]
    A4["4 · No-pretrain baseline<br/>frozen random encoder + TaskHead<br/>negative control · 5k steps"]

    A1 & A2 & A3 & A4 --> CMP["ablation_compare<br/>outcome × arm table + headroom + transfer gap"]

    classDef win fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef test fill:#fff8e1,stroke:#f9a825,color:#0d1b2a;
    classDef ctrl fill:#eceff1,stroke:#546e7a,color:#0d1b2a;
    class A1 win;
    class A2,A3 test;
    class A4 ctrl;
```

| Arm | Backbone | Trainable | Evidence anchor |
|-----|----------|-----------|-----------------|
| **Frozen backbone + heads** | CLIFATRON Qwen2 (frozen) | heads only (~0.7M) | Al Attrach 2025 (frozen > trainable); Mataraso 2025 |
| Joint fine-tune | CLIFATRON init (unfrozen) | full (~30M) | tests catastrophic forgetting |
| From scratch | random CLIFEncoder | full | TOO-BERT (from-scratch can win specific tasks) |
| No-pretrain | random encoder (frozen) | head only | negative control (floor) |

:::tip Why frozen-probe is the expected winner
At ~30M on data-constrained MIMIC (utility saturates ~28M, arXiv:2505.22964), unfreezing risks catastrophic
forgetting, and the task-aligned survival objective *is* the supervision. Frozen-probe is also
the **only** mode that supports label-free federated validation — a new site never trains.
:::

---

## Trainable-parameter contrast

```mermaid
flowchart LR
    subgraph FROZEN["Frozen probe"]
        FB["backbone ~30M (frozen)"]
        FH["heads ~0.7M (trainable)"]
    end
    subgraph JOINT["Joint fine-tune"]
        JB["backbone ~30M (trainable)"]
        JH["heads ~0.7M (trainable)"]
    end

    classDef frozen fill:#90a4ae,stroke:#37474f,color:#fff;
    classDef train fill:#66bb6a,stroke:#2e7d32,color:#0d1b2a;
    class FB frozen;
    class FH,JB,JH train;
```

*Measured on real MIMIC:* the model builds at **34.9M params**; frozen arm = **0.7M
trainable**; joint arm = **34.9M trainable**. Gradients flow in every arm (smoke-tested).

---

## Tokenization ablation (recap)

The five representation arms from **[Data & Tokenization](./data-tokenization.md)** run through
the same trunk. Lee 2026 predicts **deciles + soft** wins, especially on tail/threshold AUPRC.

```mermaid
flowchart LR
    B["clinical bins<br/>(CLIFATRON baseline)"] --> R
    D["global deciles"] --> R
    S["deciles + soft<br/>(predicted winner)"] --> R
    C["continuous-fused<br/>(McCann)"] --> R
    T["textcode<br/>(frozen BERT / PORTER)"] --> R
    R["run_tokenization_ablation<br/>same backbone + tasks"] --> OUT["tail/threshold AUPRC + calibration"]

    classDef win fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class S win;
```

---

## How the ablations feed the paper

```mermaid
flowchart TB
    T1["Tokenization ablation<br/>→ best representation"] --> SPEC["Locked spec"]
    T2["Finetune-vs-scratch ablation<br/>→ best training mode"] --> SPEC
    SPEC --> M3["Method 3 wedge<br/>(smallest publishable unit)"]
    M3 --> FED["Federated external validation<br/>(the deployability headline)"]

    classDef out fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class M3,FED out;
```

Run:

```bash
# finetune-vs-scratch
torchrun --nproc_per_node=2 -m src.train.run_arm --arm frozen_backbone_head_only --checkpoint <ckpt> --data <narratives>

# tokenization ablation
for arm in clifatron_clinical_bins global_deciles deciles_plus_soft continuous_fused textcode; do
    torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation --arm $arm --data <events.parquet>
done

# compare
python -m src.eval.ablation_compare --results results/ablation
```
