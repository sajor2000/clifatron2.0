# CLIFATRON 2.0

[![CI](https://github.com/sajor2000/clifatron2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/sajor2000/clifatron2.0/actions/workflows/ci.yml)

**One from-scratch, CLIF-native ICU foundation model, and two papers built on it.** The model is a
Qwen2-architecture decoder (design target about 30M parameters) trained with a marked time-to-event
objective in place of pure next-token prediction: a threshold-conditioned hazard head (ICareFM), a
competing-risk incidence head (SurvivEHR), a value-regression "mark" head (ORA) and a low-weight
next-event term, with a next-token-to-time-to-event curriculum. It builds on
[CLIFATRON](https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON), the CLIF consortium's
model, whose physician-designed clinical-segment bins and fused `code=bin` token idea it keeps.

**Paper 1 makes three claims about that one model**, each with a pre-specified test that can fail:

1. **Threshold-aligned tokenization.** Bin edges at clinical decision thresholds, shared with the
   training objective, improve prediction and calibration at those thresholds.
2. **Combined time-to-event objective.** The combined objective beats next-token training at equal
   compute.
3. **Extubation application.** With the post-extubation device token injected at the real extubation,
   the model reproduces the randomized-trial pattern of who benefits from NIV and HFNC at least as well
   as a classical emulation.

**Paper 2 is an externally validated extubation-failure risk model.** Per-device risks are exploratory,
and the output makes no device recommendation. A null result on any claim is reported as such.
Everything is evaluated retrospectively; a clinician-facing tool is not part of this work.

Every method component is published. What is ours is their pairing, the controlled tests, and the open,
CLIF-native execution. The released CLIFATRON Qwen2 checkpoint (0.5B) is a larger frozen comparator and
carries none of the claims.

**Status (2026-10-03):** the training path, the claims panel and the classical extubation audit are built
and tested on synthetic data. No model has been trained under the three-claim plan, so no claim has a
result. See [`website/docs/project-status.md`](website/docs/project-status.md).

**Thesis:** one small model → many outcomes → many hospitals → one node (2× L40, no cluster).

> **Taking this over? Start with [`AGENTS.md`](AGENTS.md)** (goal, locked decisions, hard rules), then
> [`MEMORY.md`](MEMORY.md) (single source of truth) and the current plan,
> [`docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md`](docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md).
> `notes/` (incl. `NEXT_STEPS.md`) is the historical pre-2026-09 design record — evidence tables still
> useful, decisions superseded.

## Documentation

The documentation site lives in `website/` (Docusaurus; `cd website && npm run build`). Start with:

| Page | What it covers |
|------|----------------|
| [Overview](website/docs/overview.md) | The three claims, Paper 2, and the pipeline as built |
| [Paper 1 — the three claims](website/docs/paper-claims.md) | Each claim's test, what falsifies it, and the code that implements it |
| [Extubation application](website/docs/extubation-application.md) | Cohort, labels, estimators, benchmark trials, and the go/no-go audit |
| [Data & Tokenization](website/docs/data-tokenization.md) | The canonical tokenizer spec and the target modes |
| [Objectives & Training](website/docs/objectives-training.md) | The loss, in-stream targets, curriculum, objective arms, and the full-hospitalization loader |
| [Ablations](website/docs/ablations.md) | The experiment matrix and comparator rows |
| [Project Status & Roadmap](website/docs/project-status.md) | What is built, what is not, and the blockers |

The extubation protocol draft is [`docs/protocols/extubation-emulation-protocol.md`](docs/protocols/extubation-emulation-protocol.md).

## Quick start

```bash
uv sync

# tests (data-gated tests skip cleanly when no CLIF data is present)
CLIF_DATA_DIR=~/Data/clif-source uv run --with pytest python -m pytest tests/ -q

# data-free reproduction: releaser -> site -> aggregator on synthetic fixtures
uv run python -m src.eval.reproduce_synthetic

# docs site
cd website && npm run build
```

Running on the 2× L40 node, from raw CLIF tables to the screening runs, follows
[the L40 runbook](docs/plans/l40-runbook.md).

## Design principle — clinically derived
The model must be most sensitive where clinical **danger** is, and legible to a clinician. Outcomes are
states doctors *act on* (never treatments — those are inputs only), and threshold heads are
**directional** (crossing *into* danger).

## Sites
- **Development:** MIMIC-IV-Ext-CLIF and Rush, trained together on the Rush L40 node. *Only MIMIC is
  staged today; Rush is not.* MIMIC is exploratory for the extubation claim and Rush is confirmatory.
- **External validation:** UChicago, by *model-to-data*: it runs the turnkey `clif-validate` package on
  its **local** CLIF tables and returns only aggregate metrics. Northwestern joins later and is not active
  scope.
- Vocab = a **frozen** CLIF-native mCIDE, applied identically everywhere; **raw data is never pooled.**

## Tokenizer & trunk (see `MEMORY.md` §E + `AGENTS.md`)
Full tokenizer spec: [`website/docs/data-tokenization.md`](website/docs/data-tokenization.md).
- **Tokens (tokenizer v2, 2026-10-02):** fused `code=bin` for **every** numeric concept · **physician-designed
  clinical-segment bins** from the CLIF consortium CSV with its interval flags honored (primary; else ordinal
  or frozen quantile bins; zero bin for every dose; population deciles = `decile_ablation` arm) · fused
  `concept=value` categoricals · doses, vent settings, assessments, CRRT, ECMO/MCS, code status, position and
  static admission tokens · **soft discretization** · ICU decision thresholds forced as bin edges with
  outcome-direction closure (lactate 2/4, MAP 65, SpO₂ 88/90, creatinine 1.5/2/3) · deterministic order +
  declared availability · a full-hospitalization GEM artifact · an aggregate-only report + gate.
  Every v1 artifact and checkpoint is refused; a `--sample-episodes` vocabulary is smoke-only.
- **Time:** admission-relative minute **time-aware RoPE** (ICU admission on the 24 h artifact, hospital
  admission on the full-hospitalization stream; no inserted `day_N/hour_N` tokens). Event order only is an
  ablation arm.
- **Trunk:** from-scratch Qwen2-arch decoder (~30M; objective, not backbone, is the lever), d512 × 8L × 8H,
  SwiGLU/RMSNorm, no QK-Norm, **untied embeddings**, context 8192. Qwen3-arch = measured ablation row.

## Layout
```
external/clifatron/      vendored upstream CLIFATRON (tokenETL, AR trainers, benchmark) — see its VENDORED.md
configs/                data.yaml · model.yaml · train.yaml · thresholds.yaml · objective_arms.yaml · claims.yaml
                        extubation.yaml · extubation_benchmarks.yaml · extubation_audit.yaml · experiment_matrix.yaml
src/data/tokenize.py     CLIF parquet → fused event-token shards + vocab (primary tokenizer)
src/data/dataset.py       datasets + samplers: columnar full-hospitalization corpus, rank-aware token-budget batches
src/data/targets.py       next-event, value and time-to-event targets (icu_24h · gem · gem_tte in-stream labels)
src/data/threshold_grid.py  registered thresholds → each arm's bins; edge-distance table
src/data/extubation_cohort.py  extubation cohort, device arms and trial risk factors
src/data/value_stats.py   per-token robust value stats, vocab-hash-bound (training fails closed if missing)
src/model/encoder.py     from-scratch time-aware Qwen2-arch decoder (~30M primary trunk)
src/model/heads.py       threshold-hazard · competing-risk · value-regression · task heads   [KEEPER]
src/model/head_adapter.py  attach our heads to a CLIFATRON checkpoint's hidden states          [KEEPER]
src/model/generate.py     guarded rollout generation: observed-transition masks · repetition penalty · closed-world sampling
src/train/pretrain.py    torchrun DDP pretraining (NTP→TTE curriculum, objective arms, multi-site full-hospitalization path)
src/train/run_matrix.py   experiment matrix → run specs + launch commands (never trains)
src/train/checkpoint.py   step-granular checkpoint / resume (fresh-schedule restore)
src/eval/metrics.py      TRIPOD+AI panel: AUROC/AUPRC/ECE/Brier/calib-slope/ICI/DCA/LPE/subgroup  [KEEPER]
src/eval/method3.py      the wedge: anchor states → our probe vs XGBoost → 3×3 transport matrix   [KEEPER]
src/eval/clif_tasks.py   CLIF post-24-hour task suite; src/eval/baselines.py count and token baselines
src/eval/threshold_eval.py  zero-shot threshold evaluation of one run (head · probe · rollout)
src/eval/claims_report.py   claim 1 and claim 2 decisions over seeds (bootstrap, BH, FCR intervals)
src/eval/extubation_labeler.py  reintubation, death and composite labels (label-only study endpoints)
src/eval/causal/          clone-censor-weight + doubly robust estimators, benchmark registry, emulation, simulation
src/eval/extubation_audit.py  go/no-go audit: blind · simulate · unblinded, aggregate-only export
src/eval/generative.py    G3 rollout eval: event-rate calibration vs the real corpus · key-event recall · distance-to-observed
src/eval/clinical_plausibility.py  explainable rollout plausibility (OOV · invalid bins · special-token placement · loops)
src/eval/matrix.py       stable re-export surface
src/viewer/sequence_viewer.py  local token-sequence viewer: plausibility score · concept timeline · observed-vs-generated compare
```

## Validation & release-trust infrastructure (landed, data-free)

The federated model-to-data path is implemented and tested end to end on synthetic fixtures — no
real data or GPU needed to exercise it:

- **`clif-validate/`** — the turnkey site package a hospital runs on its **local** CLIF 2.1 tables.
  It loads a frozen, **signed** bundle, runs zero-shot inference, and exports only
  disclosure-controlled aggregate metrics. Fail-closed and data-free by default.
- **Release trust** (`src/eval/trust.py`, `configs/trust_roles.yaml`) — an **Ed25519** releaser→site
  signature verified at load, a signed revocation list, anti-rollback, and **approval-by-content-hash**
  so a release can only carry the reviewed payload. Report authentication + the cumulative disclosure
  ledger use HMAC (`src/eval/attestation.py`).
- **Aggregator** (`src/eval/aggregator.py`) — the coordinating-center reader: it verifies every signed
  site report, independently re-enforces the releasable-status gate, and blocks a cross-release
  differencing leak against its own cumulative ledger.
- **Model card:** [`MODEL_CARD.md`](MODEL_CARD.md) · **Architecture + reproducibility guide:**
  [`docs/architecture.md`](docs/architecture.md).

### Reproduce + test

One command (`python -m src.eval.reproduce_synthetic`) runs the whole releaser → site → aggregator loop
on synthetic fixtures; both data-free test suites run in CI. Exact commands:
[`docs/architecture.md` → Reproduce the synthetic result](docs/architecture.md#reproduce-the-synthetic-result).

## Local GEM validation stack — proven on real MIMIC

The full generative path is proven end-to-end on the dev Mac against the real staged MIMIC CLIF
tables: ETL (fused `code=bin` tokens + frozen vocab lock) → 8192-row document-isolated shards →
pure-NTP MPS smoke train (6,000 steps, fp32, step-granular checkpoints) → **guarded** rollout
generation → explainable plausibility + viewer inspection. Full evidence trail:
[`docs/plans/gem-overnight-log.md`](docs/plans/gem-overnight-log.md).

```bash
# guarded closed-world rollouts — hard candidate-list sampling + observed-transition
# masks + repetition penalty, so OOV / out-of-vocab tokens cannot leak
python -m src.model.generate --checkpoint <ckpt> \
  --vocab output/intermediate_phi/mimic/vocab.json \
  --reference-events output/intermediate_phi/mimic/events.parquet \
  --n-simulations 6 --max-new-tokens 128 --device mps \
  --output output/intermediate_phi/sims.parquet

# inspect real + generated sequences: plausibility warnings, concept timelines,
# and observed-vs-generated comparison
python -m src.viewer.sequence_viewer \
  --parquet output/intermediate_phi/mimic/events.parquet \
  --parquet output/intermediate_phi/sims.parquet \
  --vocab-lock output/intermediate_phi/mimic/vocab.json --port 8042
```

G3 baselines (6k-step MPS checkpoint, guarded): JS divergence 0.43 vs the real corpus, top-32
overlap 0.47, **0 OOV / 0 gen-only tokens**, every rollout `good` on the plausibility panel.
Key-event recall (~0.30) does **not** improve with more NTP steps. *(2026-10-03: reward-based
post-training is no longer planned; the combined time-to-event objective is the lever under test.)*
The earlier G2 handoff is in [`docs/plans/l40-g2-runbook.md`](docs/plans/l40-g2-runbook.md); the current
L40 sequence is [the L40 runbook](docs/plans/l40-runbook.md).

## Method 3 — the wedge (larger frozen comparator)
*(2026-10-03: the three claims are tested on the from-scratch model. The wedge is the larger frozen
comparator row, reported when a CLIFATRON checkpoint is staged; it carries none of the claims.)*
Attach our calibrated survival/probe heads to a CLIFATRON checkpoint's hour-24 anchor hidden state and
beat their **Method 1** (XGBoost-on-embeddings) on AUPRC/calibration and **Method 2** (MC rollout) on cost,
on their own benchmark — across MIMIC / Rush / UChicago with an Elemento inference-time ensemble.

```bash
uv sync
python -m src.eval.method3 \
  --checkpoint /path/to/clifatron_checkpoint \
  --site MIMIC=/path/mimic_narratives.parquet \
  --site Rush=/path/rush_narratives.parquet \
  --site UChicago=/path/uchicago_narratives.parquet \
  --method both
```

## Non-negotiable rules
1. Treatments are model **inputs**, never prediction targets of the trunk. Scoped amendment (2026-10-03):
   study heads downstream of the frozen trunk may use endpoints defined by a treatment event (for example
   reintubation) as labels, declared label-only in `configs/cohort.yaml → study_endpoints`.
2. Vocab = frozen CLIF mCIDE applied identically to all sites — **no cross-site raw pooling**.
3. Retrospective reports / discharge summaries are a **label source only**; only *pre-anchor* notes are features.
4. `storetime`/availability ordering, not `charttime` — no look-ahead on when a value was knowable.
5. MIMIC-IV-Ext-CLIF is PhysioNet-credentialed; Rush + UChicago are institutional — **no data leaves its node.**
   External validation returns aggregate metrics only.

## References
ICareFM · SurvivEHR (npj Digit Med 2026) · ORA (arXiv:2602.00541) · Lee "Representation Before Training"
(arXiv:2604.16775) · Context Clues (arXiv:2412.16178) · Federated GEMs (arXiv:2608.02939) · Elemento ·
Cadence · TRIPOD+AI (BMJ 2024;385:e078378). Line-cited detail in `notes/METHODS.md` and `notes/RESEARCH.md`.

## License
MIT (see `LICENSE`), consistent with upstream CLIFATRON.
