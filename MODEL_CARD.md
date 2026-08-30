# Model Card — CLIFATRON 2.0

A data-free model card for the CLIFATRON 2.0 methods-upgrade layer and its federated
external-validation infrastructure. It describes intended use, data *classes* (no PHI lives in this
repository), evaluation, disclosure controls, limitations, and — explicitly — what is **proven** in code
today versus what is **pending** hardware, real data, or governance.

## Model details

- **What it is.** A methods-upgrade layer on [CLIFATRON](https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON),
  the CLIF consortium's compact (~30M-param) CLIF-native ICU foundation model. It replaces pure
  next-token prediction with a **threshold-conditioned time-to-event** objective, a **competing-risk
  cumulative-incidence** head, and a **value-regression** head, and adds **zero-shot, training-free**
  survival/threshold heads so a new hospital needs no local model training and no manually-annotated
  labels to run the model.
- **Input.** Sequences of fused CLIF event tokens (`code=bin`) over a frozen CLIF-native mCIDE
  vocabulary, applied identically at every site.
- **Output.** Per-stay, per-outcome risk at a clinical horizon (e.g. 48h), and — at a coordinating
  center — disclosure-controlled **aggregate** metrics across sites (AUROC/AUPRC/ECE and a
  TRIPOD+AI calibration / decision-curve / fairness panel). Raw data, labels, and gradients never
  leave a node.
- **Backbone.** Qwen2/Llama-style transformer (the objective, not the backbone, is the lever).

## Intended use

- **Intended.** Research development and **federated external validation** of an ICU foundation model
  across CLIF-consortium hospitals by *model-to-data*: ship a frozen, signed model + a turnkey
  evaluator; each site runs it on its **local** CLIF 2.1 tables and returns only aggregate + subgroup
  metrics.
- **Out of intended use.** Direct clinical decision-making without prospective validation and local
  governance; any use that requires raw patient data to leave its node; treating model outputs as a
  substitute for clinician judgment. Treatments are model **inputs**, never prediction targets.

## Data

- **Class, not contents.** This repository contains **no PHI and no real patient data**. All tests and
  the one-command reproduction run on **synthetic** fixtures.
- **Development cohort (held by the project, not in this repo):** MIMIC-IV-Ext-CLIF v2.1 (PhysioNet-
  credentialed), Rush, and UChicago — each governed at its own institution.
- **External validation:** all other CLIF-consortium sites via model-to-data; the vocabulary is a
  **frozen** CLIF-native mCIDE applied identically everywhere; **raw data is never pooled.**
- **Labels** are auto-derived from each site's own standard CLIF fields (no manual annotation);
  retrospective reports/discharge summaries are a **label source only**, and only *pre-anchor* notes are
  features.

## Evaluation

- **Panel.** TRIPOD+AI: AUROC/AUPRC/ECE/Brier, calibration slope, ICI, decision-curve / net-benefit
  analysis, a task-specific ML floor (LPE), and subgroup metrics. The headline metric is **net benefit**,
  not AUROC alone — does acting on the model help the patient.
- **Federated aggregation.** A coordinating-center aggregator verifies each site's signed report and
  pools only allow-listed summary statistics into a multi-site panel; non-evaluable cells contribute
  their status, never a number.

## Disclosure controls & privacy (fail-closed by default)

- **No raw data leaves a node.** The site evaluator exports only disclosure-controlled aggregates;
  small cells are suppressed (a minimum cell size is pinned from the bundle's policy), and suppressed
  cells carry no metric value.
- **Cross-release differencing** is blocked by an append-only cumulative disclosure ledger — a cell
  suppressed in one release cannot be exposed by differencing a later one — enforced both site-side and,
  independently, at the aggregator.
- **Release trust.** Bundles are signed by the releaser (**Ed25519**) and verified at every site against
  an out-of-band trust root, with a signed revocation list and anti-rollback. A release is bound to the
  reviewed payload by content hash. Site→aggregator report authentication and the access-log chain use
  HMAC. Every control fails closed; an unverifiable, revoked, rolled-back, or unreviewed artifact is
  refused rather than degraded into a success-shaped report.

## Limitations & ethical considerations

- Retrospective, observational data; outcomes are states clinicians act on, but label ascertainment
  and cohorting inherit each site's coding practices.
- Small-cell suppression trades some analytic resolution for disclosure safety, by design.
- Multi-site transferability is the research question, not a settled result — see the pending list.
- The model is a decision *support* artifact for research, not an autonomous clinical system.

## Proven vs pending (honest scope)

**Proven in code today (synthetic, CPU, data-free; covered by the test suites and CI):**
- The site evaluator, bundle contract, and disclosure controls (small-cell suppression, ledger
  differencing).
- Release trust: Ed25519 signing/verification, revocation, anti-rollback, approval-by-content-hash.
- The coordinating-center aggregator and the end-to-end releaser→site→aggregator loop
  (`python -m src.eval.reproduce_synthetic`).
- Training-engine resume-equivalence and DDP sample-coverage (CPU), and the document-isolated
  (variable-length) attention core (CPU path).

**Pending (blocked on hardware, real data, or governance — not on code):**
- Real-data training and evaluation on the development cohort (needs governed CLIF tables + data QA).
- GPU qualification: the FA2 attention training forward, and the 2×L40 training/throughput report.
- Real-site federation: onboarding an external CLIF site, and the governance approval to distribute even
  a pre-selection bundle for a real run (see `configs/trust_roles.yaml` → `pending_governance`).

## Citation

If you build on this work, please cite CLIFATRON and the authors' foundation-model work on EHR
representation dynamics and transferability (Burkhart, Rojas, Parker et al., 2025). See the README's
References section for the full methods bibliography (ICareFM, SurvivEHR, ORA, TRIPOD+AI, etc.).

## License

MIT (see `LICENSE`), consistent with upstream CLIFATRON.
