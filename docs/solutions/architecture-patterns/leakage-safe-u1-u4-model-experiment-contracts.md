---
title: "Leakage-safe model experiment contracts before architecture sweeps"
date: "2026-08-28"
category: architecture-patterns
module: "clifatron2.0 evidence-ready model experiments"
problem_type: architecture_pattern
component: model_training
severity: high
applies_when:
  - "Replacing synthetic or placeholder ML runners with real evidence-ready experiment contracts"
  - "Building CLIF-native model training pipelines with anchor-time and split-leakage constraints"
  - "Adding TTE, competing-risk, value-regression, and masked NTP objectives to a shared training contract"
  - "Qualifying model experiments before expensive architecture or scaling runs"
tags:
  - u1-u4
  - leakage-safety
  - time-to-event
  - competing-risk
  - value-regression
  - document-isolation
  - artifact-policy
  - resumable-training
fingerprint: "architecture_pattern::model_training::leakage-safe-u1-u4-model-experiment-contracts"
---

# Leakage-safe model experiment contracts before architecture sweeps

## Context

CLIFATRON 2.0 needed to move from placeholder model plumbing toward evidence-ready experiments. The highest-risk gaps were not model size or architecture choice; they were invalid training contracts: post-anchor leakage risk, missing outcome/value artifacts, treatment tokens as prediction targets, packed-document leakage, invalid competing-risk probabilities, and non-resumable training.

## Guidance

Build the first implementation slice as a contract layer before running architecture or scaling experiments:

- Define a canonical cohort/anchor/split/outcome artifact before tokenization, labeling, training, or validation.
- Treat raw tokenized `events.parquet` as context-only until joined with outcome and value-normalization artifacts.
- Preserve episode/document metadata through sequence packing, including opaque episode keys, source spans, packed spans, and continuation flags.
- Make target tensors explicit: `ntp_target`, `ntp_mask`, `value_target`, `value_mask`, CR event/censor labels, and threshold query fields.
- Fail closed when value statistics or supervised TTE outcomes are absent for real training; allow only labeled dry-run plumbing to proceed without them.
- Use a normalized competing-risk distribution over `K causes + no event`, not independent sigmoid hazards.
- Drive training checkpoints and validation from explicit optimizer-step counters, not epoch counters or optimizer internals.

## Why This Matters

Foundation-model experiments can look successful while optimizing the wrong task. If token shifts are used directly, treatments become targets. If packed records are treated as one sequence, patients leak through causal attention. If outcome artifacts are missing, TTE losses can silently become no-op or false negative supervision. If competing-risk hazards are independent sigmoids, cumulative incidence can violate probability constraints.

The contract-first pattern makes invalid states fail before expensive runs and before benchmark-shaped metrics can be produced.

## When to Apply

- Before running CLIF-native TTE, competing-risk, value-regression, or label-efficiency experiments.
- When adapting a next-token EHR foundation model to supervised or self-supervised survival objectives.
- When packed sequences can contain multiple hospitalizations or split one hospitalization across rows.
- When training has to be resumable under DDP and epoch-dependent sampling.

## Examples

Use explicit masked targets rather than raw next-token shifts:

```python
ntp = next_event_loss(logits, batch["ntp_target"], batch["ntp_mask"])
val = value_head.loss_aligned(H, batch["ntp_target"], batch["value"], batch["val_mask"])
```

For competing risks, compute event mass from event-free probability and conditional cause probability:

```python
q = softmax(logits, dim=cause_plus_no_event_axis)
event_free = cumprod(q_no_event, dim=time_axis)
event_mass = shifted(event_free) * q_cause
cif = cumsum(event_mass, dim=time_axis)
```

For real pretraining, fail closed when the run lacks required artifacts:

```python
if not dry_run and not value_stats and has_numeric_values(records):
    raise SystemExit("value-head normalization is required before real training")

if not dry_run and not has_supervised_outcomes(records):
    raise SystemExit("TTE supervision is required before real pretraining")
```

## Related

- Plan: `docs/plans/2026-08-27-001-feat-evidence-ready-model-experiments-plan.md`
- Code: `src/data/cohort.py`, `src/data/dataset.py`, `src/data/collate.py`, `src/data/targets.py`
- Code: `src/model/heads.py`, `src/train/engine.py`, `src/train/pretrain.py`
