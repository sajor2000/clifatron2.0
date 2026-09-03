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
CLIFATRON v1 used **physician-designed clinical segments** from a 1268-row CSV
(`critical_illness_tokenization_final_with_intervals.csv`) — tighter bins in decision zones,
progressively wider intervals above/below normal, extreme-value quintiles at tails.

The code had been changed to deciles as default with clinical segments relegated to an
"ablation arm." This was a mistake: the clinical team's 1268 segments encode measurement-
density domain expertise that data-driven deciles cannot recover. For example, lactate has 15
physician-designed bins (vs 10 deciles), with 5 extreme-value quintiles above 5.0 mmol/L that
capture the physiologically dangerous tail the model must be most sensitive at.

## Guidance

**Physician-designed clinical segments are the primary scheme; population deciles are the
`decile_ablation` arm only.** The implementation:

1. `build_clinical_segment_bins()` reads the CSV, extracts interior bin edges per concept,
   filters to the 10 target concepts from `data.yaml`, and pins any additional forced
   clinical thresholds (lactate 2.0/4.0, MAP 65, SpO₂ 88/90) as guaranteed edges.
2. `build_edges()` dispatches on `value_binning.scheme` — `clinical_segment` is the default.
3. `build_value_bins()` (decile quantile estimator) is retained as the `decile_ablation` path.
4. Soft discretization (Gaussian-weight spread to ±1 neighbor bin) is applied on top of the
   clinical-segment edges to smooth quantization jitter at boundary crossings.

## Why This Matters

The founding claim of the CLIF consortium is that **clinical expertise makes the model better**.
Using data-driven deciles contradicts that claim. The clinical segments are what differentiate
this model from an auto-regressive token predictor on EHR sequences — they are the reason a
clinician trusts the output at a lactate of 2.1 or a MAP of 64.

Lee (2026) found deciles ≈ clinical reference ranges "at matched granularity." But the CLIF
consortium's segments are finer-grained than 10-bin deciles (lactate: 15 vs 10, MAP: 25 vs
10, temp: 26 vs 10) and invest granularity where it matters — decision zones and dangerous
tails. The decile ablation arm exists specifically to measure the head-to-head difference on
this consortium's data rather than asserting one is better.

## When to Apply

- **Default:** Always use `scheme: "clinical_segment"` in `configs/data.yaml`
- **Ablation:** Set `scheme: "decile"` or `scheme: "decile_ablation"` when measuring the
  contribution of domain-expert binning via the tokenization ablation framework
- **New concept:** When adding a new numeric concept to `target_concepts`, verify it has
  entries in the clinical-segment CSV. If not, add them with clinician input rather than
  falling back to data-driven deciles

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

**Tokenize dispatch (`src/data/tokenize.py`):**
```python
def build_edges(bin_cfg, fit_events, target_concepts):
    scheme = bin_cfg.get("scheme", "clinical_segment")
    if scheme == "clinical_segment":
        source = bin_cfg["segment_source"]
        return build_clinical_segment_bins(ROOT / source, target_concepts, forced)
    if scheme in ("decile", "decile_ablation"):
        return build_value_bins(fit_events, n_bins, forced)
```

**Bins produced (10 target concepts):**
| Concept | Bins | Notable edges |
|---------|------|---------------|
| lactate | 17 | 2.0, 4.0; 5 extreme quintiles above 5.0 |
| MAP | 25 | 65 hypotension threshold; tight 60–80 zone |
| SpO₂ | 13 | 88, 90 hypoxemia thresholds |
| creatinine | 17 | 1.5, 2.0, 3.0 KDIGO stages |
| temp_c | 26 | Tight febrile-range intervals |