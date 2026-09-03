---
id: architecture
title: Model Architecture
sidebar_position: 3
---

# Model Architecture

The trunk is a flat causal decoder in the Llama/Qwen family (RoPE, SwiGLU, RMSNorm, GQA), 8192
context, **untied** embeddings. Two backbone paths, run as a ladder:

- **Primary paper — from-scratch Qwen3-arch decoder, ~30M, fully ours** (`src/model/encoder.py`).
  Qwen3 adds free QK-Norm training stability; no upstream dependency. This is the headline.
- **Wedge / cheap first result — attach heads to CLIFATRON's released Qwen2 checkpoint** (`head_adapter.py`).
  CLIFATRON's Qwen2 is **0.5B** — a *larger comparator*, never our ~30M model. This is the fast first
  rung (Method 3) and half of the finetune-vs-scratch ablation.

From CLIFATRON v1 we **keep** the Qwen2 architecture for the wedge path, the fused `code=bin`
token format, the mCIDE vocabulary, and the document-isolation packing approach. The from-scratch
Qwen3 path is new and fully independent.

We attach the same four heads to either backbone's per-token hidden states. "Qwen2 vs Qwen3" is itself
a **measured ablation row**, not an assertion.

:::info Objective, not backbone, is the lever
ORA (arXiv:2602.00541) shows the gains are backbone-agnostic — so the backbone is a footnote and the
*objective* is where the novelty lives. The from-scratch Qwen3 model is the primary contribution; the
CLIFATRON-Qwen2 attach is the cheap wedge that de-risks it first. See `MEMORY.md` §B.
:::

---

## Backbone + heads (component view)

```mermaid
flowchart TB
    IN["input_ids · attention_mask<br/>(fused CLIF tokens, ≤8192)"] --> BB

    subgraph BB["Backbone (from-scratch Qwen3 ~30M · OR · CLIFATRON Qwen2 0.5B wedge)"]
        direction TB
        EMB["Untied token embeddings"] --> L1["Decoder layers<br/>RoPE · SwiGLU · RMSNorm · GQA"]
        L1 --> HS["Per-token hidden states H<br/>output_hidden_states=True"]
    end

    HS --> ANCH["anchor_state()<br/>H_t at hour-24 token<br/>(last real token via attention_mask)"]

    ANCH --> TH["ThresholdHazardHead<br/>(ICareFM, zero-shot engine)"]
    ANCH --> CR["CompetingRiskHead<br/>(SurvivEHR CIF)"]
    HS --> VR["ValueRegressionHead<br/>(ORA mark)"]
    HS --> NE["NextEventHead<br/>(low-weight aux)"]
    ANCH --> TK["TaskHead<br/>(K downstream binary probes)"]

    classDef bb fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef head fill:#f3e5f5,stroke:#6a1b9a,color:#0d1b2a;
    classDef primary fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class EMB,L1,HS bb;
    class CR,VR,NE,TK head;
    class TH primary;
```

*Wiring:* `src/model/head_adapter.py → CLIFATRONHeads`. `hidden_states()` runs the backbone;
`anchor_state()` selects H_t; each head consumes it.

---

## The four heads (class view)

```mermaid
classDiagram
    class CLIFATRONHeads {
        +backbone : HF causal LM
        +frozen : bool
        +hidden_states(ids, mask) H
        +anchor_state(H, mask, idx) H_t
        +threshold_prob(...) zero-shot
        +loss(batch, w_ntp, w_cr, w_th, w_val) dict
    }
    class ThresholdHazardHead {
        +thr_emb : Embedding
        +dir_emb : Embedding
        +target_emb : Embedding
        +hazard(H_t, target, tau, dir)
        +cumulative_failure(...) F
        +predict_with_confidence(...) selective
        +loss(..., crossed_bin)
    }
    class CompetingRiskHead {
        +n_types, n_bins
        +hazard(H) per-type,bin
        +loss(H, event_type, dt_bin)
    }
    class ValueRegressionHead {
        +next_emb : Embedding
        +mlp -> (mu, logvar)
        +loss(H, next_tok, next_val, mask) Gaussian NLL
    }
    class NextEventHead {
        +projection : Linear
        +tie_weights : bool
        +forward(H) logits
    }
    class TaskHead {
        +fc : Linear
        +forward(H_t) logits
        +loss(H_t, labels, mask) masked BCE
    }

    CLIFATRONHeads --> ThresholdHazardHead
    CLIFATRONHeads --> CompetingRiskHead
    CLIFATRONHeads --> ValueRegressionHead
    CLIFATRONHeads --> NextEventHead
    CLIFATRONHeads ..> TaskHead : downstream probe
```

---

## What each head is for

| Head | Source | Role | Zero-shot? |
|------|--------|------|:----------:|
| **ThresholdHazardHead** | ICareFM | P(concept k crosses τ in `direction` within horizon h). Learned threshold + direction + target embeddings → discrete hazard over 48h. **The multi-outcome engine.** | ✅ |
| **CompetingRiskHead** | SurvivEHR | Discrete-time cumulative incidence over competing event types; exactly one event occurs next. | ✅ |
| **ValueRegressionHead** | ORA "mark" | Predict the continuous value of the next event as a Gaussian (μ, log-var). Lifts physiology tasks +33–38% vs NTP. | — |
| **NextEventHead** | SurvivEHR / HealthFormer | Next-token projection; low-weight auxiliary, **untied** by default (explicit tied ablation). | — |
| **TaskHead** | — | K downstream binary heads on the frozen trunk (the fair head-to-head vs XGBoost). | — |

---

## The untied-embedding budget tension

Untied embeddings add +4–7% AUPRC (widening under federation), but at untied the embedding
tables cost `vocab × d × 2`. This constrains vocabulary size.

```mermaid
flowchart LR
    subgraph BUDGET["~30M parameter budget"]
        TRUNK["Trunk ≈ 25M"]
        EMB["Embeddings ≈ 8–12M<br/>(untied, vocab ~10k)"]
    end
    V10["vocab ~10k"] -->|"untied fits"| BUDGET
    V30["vocab ~30k"] -->|"would force TIED<br/>(loses federation bump)"| WARN["⚠ revisit if vocab grows"]

    classDef ok fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef warn fill:#fff3e0,stroke:#e65100,color:#0d1b2a;
    class TRUNK,EMB,V10 ok;
    class V30,WARN warn;
```

Resolution: **untied + ~10k vocab** (≈8–12M emb + ~25M trunk ≈ 33–37M, still the "~30M
neighborhood"). Documented in `notes/NEXT_STEPS.md §2.3`.

---

## The anchor: hour-24 hidden state

Every head that produces a per-stay prediction reads **H_t at the hour-24 anchor** — the last
real token of the first-24h window (CLIFATRON's benchmark truncates there). It is selected from
the `attention_mask`, or an explicit `anchor_idx`.

```mermaid
sequenceDiagram
    participant B as Backbone
    participant A as anchor_state()
    participant H as Head
    B->>A: hidden_states H [B, T, d]
    A->>A: anchor_idx = attention_mask.sum(1) - 1
    A->>H: H_t = H[i, anchor_idx]  [B, d]
    H->>H: consume H_t (+ query embeddings)
    H-->>B: prediction (hazard / CIF / value / logit)
```

:::warning Transformers v5 caveat
The installed `transformers` is v5, where `output_hidden_states` moved to a
`_can_record_outputs` mechanism and the *final* hidden state may differ from the last
hidden-states entry due to extra normalization. Verify `anchor_state` against a real CLIFATRON
checkpoint before shipping the turnkey validator.
:::
