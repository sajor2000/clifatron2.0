---
id: data-tokenization
title: Data & Tokenization
sidebar_position: 2
---

# Data & Tokenization

Turning raw CLIF 2.1 parquet into the fused event-token stream the backbone consumes.
Implemented in `src/data/tokenize.py` (decile arm), `src/data/tokenize_continuous.py`
(McCann ablation), and `src/data/tokenize_textcode.py` (language-grounded ablation);
configured by `configs/data.yaml`.

---

## From CLIF tables to an event stream

Each CLIF domain table is melted to long `(hospitalization_id, dttm, concept, value, unit)`
events, ordered by **availability time** (`storetime`, not `charttime`) so the model never
sees a value before it was knowable.

```mermaid
flowchart TB
    subgraph SRC["CLIF 2.1 source tables"]
        V["clif_vitals"]
        L["clif_labs"]
        M["clif_medication_admin_continuous"]
        RS["clif_respiratory_support"]
        ADT["clif_adt"]
    end
    V & L & M & RS & ADT --> MELT["Melt to long events<br/>(hosp_id, dttm, concept, value, unit)"]
    MELT --> UNIT["Unit normalization<br/>(error on non-canonical CLIF units)"]
    UNIT --> ORDER["Order by storetime / availability<br/>per hospitalization"]
    ORDER --> BIN["Value binning<br/>per-concept deciles"]
    BIN --> FUSE["Fuse: concept=bin → single token"]
    FUSE --> SOFT["Soft discretization<br/>(spread mass to ± neighbor bins)"]
    SOFT --> ROPE["Admission-relative-minute position<br/>(time-aware RoPE)"]
    ROPE --> OUT["events.parquet<br/>token · soft_token · soft_weight · pos_min · value"]

    classDef src fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef step fill:#fff8e1,stroke:#f9a825,color:#0d1b2a;
    class V,L,M,RS,ADT src;
    class MELT,UNIT,ORDER,BIN,FUSE,SOFT,ROPE step;
```

:::warning Look-ahead leakage guard (Rule 4)
Ordering by `storetime` (when a value became **available**) rather than `charttime` (when it
was nominally measured) prevents the model from conditioning on information a clinician could
not yet have seen at decision time.
:::

---

## Fused `code=bin` tokens vs the split alternative

A single fused token per event beat the split `concept` + `value-bin` representation in the
Lee 2026 benchmark (mortality 0.891 → 0.915). It also avoids the "local binding problem"
where a value token can attach to the wrong concept.

```mermaid
flowchart LR
    RAW["creatinine = 1.4 mg/dL<br/>(decile bin 7)"]

    subgraph SPLIT["Split (rejected)"]
        S1["token: C::creatinine"]
        S2["token: V::bin7"]
    end

    subgraph FUSED["Fused (adopted)"]
        FU["token: creatinine=7"]
    end

    RAW --> SPLIT
    RAW --> FUSED

    classDef bad fill:#ffebee,stroke:#c62828,color:#0d1b2a;
    classDef good fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class S1,S2 bad;
    class FU good;
```

---

## Soft discretization — sensitivity where danger lives

Hard binning throws away where within a bin a value fell. **Soft discretization** spreads a
Gaussian-weighted mass to adjacent bins, which Lee 2026 found is the one encoder that *wins*
on exactly the dangerous tails (severe hypokalemia, hypernatremia, hypotension, CRRT). The
hard token stays the next-token-prediction target; the soft weights feed the encoder input.

```mermaid
flowchart TB
    VAL["Observed value v<br/>(e.g. lactate = 4.1)"] --> HARD["Hard bin b = _bin_of(v)"]
    HARD --> KERN["Gaussian kernel over<br/>bins b-1, b, b+1<br/>centered on sub-bin position"]
    KERN --> W["soft_token = [b-1, b, b+1]<br/>soft_weight = [0.11, 0.79, 0.11]"]
    HARD --> TGT["token = b<br/>(hard NTP target)"]

    W --> ENC["→ encoder input<br/>(weighted embedding)"]
    TGT --> LOSS["→ next-event loss target"]

    classDef v fill:#f3e5f5,stroke:#6a1b9a,color:#0d1b2a;
    classDef o fill:#e1f5fe,stroke:#0277bd,color:#0d1b2a;
    class VAL,HARD,KERN v;
    class W,TGT,ENC,LOSS o;
```

*Config:* `value_binning.soft_discretization: true`, `soft_kernel_bins: 1`.
*Code:* `_soft_bins()` in `src/data/tokenize.py` returns a fixed width `2·kernel_bins+1` for
every event (numeric or not) so batching stays dense.

---

## Value-head normalization — per-token, frozen from a reference site

The ORA value-regression head predicts the continuous *value* of the next event. Raw ICU values span
~5 orders of magnitude per concept (creatinine ~1, platelets ~2×10⁵), so an un-normalized Gaussian NLL
is dominated by high-magnitude concepts and never trains (observed `val≈46000` on real MIMIC). The fix:
standardize each numeric event to ~N(0,1) using per-**token** center/scale frozen from a reference site
— the same freeze-on-MIMIC pattern as the decile edges.

```mermaid
flowchart TB
    REF["reference-site events.parquet<br/>(token, value pairs)"] --> STATS["compute per-token (center, scale)<br/>robust: median · IQR÷1.349"]
    STATS --> BIND["bind to vocab hash<br/>(reject stale / cross-vocab file)"]
    BIND --> JSON["value_stats.json<br/>{token_id: [center, scale]}"]
    JSON --> STD["at training: (value − center) / scale<br/>→ ~N(0,1), well-scaled NLL"]

    classDef ref fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef out fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class REF,STATS,BIND ref;
    class JSON,STD out;
```

- **Robust** center/scale (median, IQR÷1.349, std fallback) so physiologic outliers (severe
  hyperkalemia, extreme lactate) don't inflate the scale and flatten the dangerous tails.
- **Coverage contract:** every token carrying a finite value gets stats — a rare numeric token is
  *widened*, never dropped — because the target builder rejects any numeric target lacking stats.
- **Vocabulary-bound:** the artifact carries the vocab hash; a stale or cross-vocabulary stats file is
  rejected at load, not silently applied.

*Verified:* per-token standardization collapses mean raw value² from ~1.4×10¹⁰ to **0.95** (the O(1)
target). *Code:* `src/data/value_stats.py`; generate with
`python -m src.data.value_stats --events <ref_events.parquet> --out value_stats.json`.

---

## Frozen decile edges + forced clinical thresholds

Bin **edges** are frozen from a reference site (`build_from_site: mimic`) and applied
identically everywhere, so the token space transports across hospitals (Federated GEMs: only
0.025 AUROC cross-site penalty). ICU decision cutpoints are then **forced to be guaranteed bin
edges** so tokens align with the threshold-hazard head's queries.

```mermaid
flowchart LR
    REF["Reference site (MIMIC)<br/>per-concept value distribution"] --> DEC["Compute deciles<br/>(10 bins / concept)"]
    DEC --> FORCE["Force clinical edges onto grid:<br/>lactate 2/4 · MAP 65 · SpO₂ 88/90<br/>KDIGO · P/F Berlin"]
    FORCE --> FROZEN["Frozen edges (vocab.json)"]
    FROZEN -->|"applied identically, no refit"| MIMIC["MIMIC tokens"]
    FROZEN -->|"applied identically, no refit"| RUSH["Rush tokens"]
    FROZEN -->|"applied identically, no refit"| UC["UChicago tokens"]

    classDef ref fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a;
    classDef site fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class REF,DEC,FORCE,FROZEN ref;
    class MIMIC,RUSH,UC site;
```

---

## Time encoding: drop `day_N/hour_N` tokens, rotate by minutes

Instead of inserting literal `day_N` / `hour_N` marker tokens into the stream, position is
encoded as **admission-relative minutes** via time-aware RoPE. Events with the same timestamp
share a position. This matches or beats inserted time tokens while cutting sequence length
~11%.

```mermaid
flowchart LR
    subgraph OLD["Inserted time tokens (dropped)"]
        O1["day_1"] --> O2["hour_3"] --> O3["creatinine=7"] --> O4["hour_4"] --> O5["map=2"]
    end
    subgraph NEW["Admission-relative-minute RoPE (adopted)"]
        N1["creatinine=7<br/>pos_min=183"] --> N2["map=2<br/>pos_min=240"]
    end

    classDef bad fill:#ffebee,stroke:#c62828,color:#0d1b2a;
    classDef good fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    class O1,O2,O3,O4,O5 bad;
    class N1,N2 good;
```

---

## The 5-arm tokenization ablation

The tokenizer choice is not asserted — it is tested. `configs/tokenization_ablation.yaml`
runs five arms through the **same** backbone and tasks so only the representation varies.

```mermaid
flowchart TB
    DATA["Same CLIF events<br/>same backbone, same tasks"] --> A1
    DATA --> A2
    DATA --> A3
    DATA --> A4
    DATA --> A5

    A1["1 · Clinical bins<br/>(CLIFATRON baseline)"]
    A2["2 · Global deciles"]
    A3["3 · Deciles + soft<br/>(predicted winner)"]
    A4["4 · Continuous-fused<br/>(McCann)"]
    A5["5 · TextCode<br/>(frozen BERT descriptions)"]

    A1 & A2 & A3 & A4 & A5 --> CMP["Compare AUPRC + calibration,<br/>esp. tail/threshold outcomes"]

    classDef base fill:#eceff1,stroke:#546e7a,color:#0d1b2a;
    classDef win fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a;
    classDef abl fill:#fff8e1,stroke:#f9a825,color:#0d1b2a;
    class A1 base;
    class A3 win;
    class A2,A4,A5 abl;
```

| Arm | Representation | Evidence anchor |
|-----|----------------|-----------------|
| Clinical bins | CLIFATRON reference-range quantiles | baseline |
| Global deciles | population deciles, frozen | Lee 2026 |
| **Deciles + soft** | deciles + Gaussian adjacent-bin mass | Lee 2026 (wins tails) |
| Continuous-fused | fused numeric channel + MC | McCann 2026 |
| TextCode | frozen BioClinical-ModernBERT code descriptions | Al Attrach 2025 / PORTER 2026 |

:::tip Target concepts (K hazard heads)
The threshold-hazard head is trained over `configs/data.yaml → target_concepts`: MAP, lactate,
SpO₂, respiratory rate, creatinine, bilirubin, platelets, heart rate, SBP, temperature — each
with a clinical **direction** (which way is dangerous). Treatments are deliberately absent
(Rule 1). Extend toward K≈35 as coverage allows.
:::
