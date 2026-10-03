---
module: src/data/tokenize
date: "2026-09-02"
problem_type: methods_decision
component: data_cleaning
severity: high
tags: [binning, clinical-segments, deciles, tokenization, icu-data, clif]
applies_when: designing value-binning scheme for ICU EHR foundation model; choosing between data-driven and domain-expert bin boundaries
fingerprint: methods_decision::tokenize::physician-clinical-segment-binning-primary-over-deciles
---

# Clinical-segment binning as the primary tokenization scheme

## Context

CLIFATRON 2.0's tokenizer had been configured to use **population deciles** (10 data-driven
quantile bins per concept) as the default binning scheme, based on Lee (arXiv:2604.16775)
showing deciles ≈ clinical-reference-range anchoring at matched granularity. The original
CLIFATRON v1 used **physician-designed clinical segments** from a CSV of 1267 segment rows plus a header
(`critical_illness_tokenization_final_with_intervals.csv`) — tighter bins in decision zones,
progressively wider intervals above/below normal, extreme-value quintiles at tails.

The code had been changed to deciles as default with clinical segments relegated to an
"ablation arm." This was a mistake: the clinical team's 1267 segments encode measurement-
density domain expertise that data-driven deciles cannot recover. For example, lactate has 15
physician-designed CSV segments (16 bins once the forced 4.0 edge is added, vs 10 deciles), with 5
extreme-value quintiles above 5.4 mmol/L that capture the physiologically dangerous tail the model
must be most sensitive at.

## Guidance

**Physician-designed clinical segments are the primary scheme; population deciles are the
`decile_ablation` arm only.** The implementation:

1. `build_clinical_segment_bins()` reads the CSV into closure-aware segments per concept
   (`src/data/segments.py`, the precedence policy; v2 since 2026-10-02 adds only the dose
   gap rule, step 8) and pins the forced clinical thresholds
   (lactate 2.0/4.0, MAP 65, SpO₂ 88/90, creatinine 1.5/2.0/3.0) with outcome-direction
   closure. (Updated 2026-10-02: it now serves every CSV concept, not only the 10 targets.)
2. `build_segments()` dispatches on `value_binning.scheme` — `clinical_segment` is the default —
   and records each concept's binning source.
3. `build_value_bins()` (decile quantile estimator) is retained as the `decile_ablation` path.
4. Soft discretization (Gaussian-weight spread to ±1 neighbor bin) is applied on top of the
   clinical-segment edges to smooth quantization jitter at boundary crossings.

## Why This Matters

The founding claim of the CLIF consortium is that **clinical expertise makes the model better**.
Using data-driven deciles contradicts that claim. The clinical segments are what differentiate
this model from an auto-regressive token predictor on EHR sequences — they are the reason a
clinician trusts the output at a lactate of 2.1 or a MAP of 64.

Lee (2026) found deciles ≈ clinical reference ranges "at matched granularity." But the CLIF
consortium's segments are finer-grained than 10-bin deciles (lactate: 16 vs 10, MAP: 23 vs
10, temp: 24 vs 10) and invest granularity where it matters — decision zones and dangerous
tails. The decile ablation arm exists specifically to measure the head-to-head difference on
this consortium's data rather than asserting one is better.

## When to Apply

- **Default:** Always use `scheme: "clinical_segment"` in `configs/data.yaml`
- **Ablation:** Set `scheme: "decile"` or `scheme: "decile_ablation"` when measuring the
  contribution of domain-expert binning via the tokenization ablation framework
- **New concept:** A numeric concept without CSV segments is still binned (`coverage: all`):
  it gets `ordinal`, `quantile` or `single` fallback bins, recorded per concept in
  `vocab.json → binning_sources`. Queue those concepts for clinician review and promote them
  to CSV segments once reviewed; a new `target_concepts` entry should have CSV segments
  before it is used as a prediction target

## Examples

**Config (`configs/data.yaml`):**
```yaml
value_binning:
  scheme: "clinical_segment"
  segment_source: "external/clifatron/tokenETL/config/critical_illness_tokenization_final_with_intervals.csv"
  fit_partition: "train"
  soft_discretization: true
  soft_kernel_bins: 1
```

**Tokenize dispatch (`src/data/tokenize.py`, as of 2026-09-02; tokenizer v2 dispatches in `build_segments`):**
```python
def build_edges(bin_cfg, fit_events, target_concepts):
    scheme = bin_cfg.get("scheme", "clinical_segment")
    if scheme == "clinical_segment":
        source = bin_cfg["segment_source"]
        return build_clinical_segment_bins(ROOT / source, target_concepts, forced)
    if scheme in ("decile", "decile_ablation"):
        return build_value_bins(fit_events, n_bins, forced)
```

**Bins produced (10 target concepts; recomputed from code 2026-10-02 under tokenizer v2 /
precedence policy v1 — 186 numeric tokens total):**
| Concept | Bins | Notable edges |
|---------|------|---------------|
| lactate | 16 | `(1.6, 2]` / `(2, 2.2]` (CSV, threshold on the non-event side), 4.0 forced → `(3.4, 4]` / `(4, 5.4]`; 5 extreme quintiles above 5.4 |
| MAP | 23 | 65 (in CSV, direction `below`) → `(61, 65)` / `[65, 67]` |
| SpO₂ | 11 | 88, 90 (in CSV) → `[49.9, 88)` / `[88, 90)`; `[92]` exact point |
| respiratory_rate | 19 | `[0]` and `[10]` exact points (was 18 under the old edge-list rule) |
| creatinine | 18 | 1.5, 2.0, 3.0 forced (≈ KDIGO) |
| bilirubin_total | 15 | — |
| platelet_count | 19 | — |
| heart_rate | 19 | `[0]` exact point (was 18) |
| sbp | 23 | — |
| temp_c | 23 | tight febrile-range intervals; the 37.499 / 37.5 overlap now resolves to one segment (was 24 with a near-empty bin) |

**Coverage is now `all` (2026-10-02):** every numeric concept in the reference train partition
is binned — CSV segments first, then one point bin per value for ordinal scales (GCS, RASS,
Braden), then frozen quantile bins (`single` when sparse); every dose concept gets a `[0, 0]` stop
bin. The decile arm bins the same concept set. The CSV's interval flags are honored (no longer
`[a, b)`), and bin counts can differ from the v1 edge-list rule because exact rows become point
segments and overlaps resolve under the precedence policy. The full per-concept segment table,
the precedence policy, and the residuals (two suspect exact CSV rows, clinician review of the
data-driven bins) are in `website/docs/data-tokenization.md`.
