# CLIFATRON 2.0

[![CI](https://github.com/sajor2000/clifatron2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/sajor2000/clifatron2.0/actions/workflows/ci.yml)

A **methods-upgrade layer** on [CLIFATRON](https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON),
the CLIF consortium's CLIF-native ICU foundation model (released Qwen2 checkpoint: 0.5B). We keep
CLIFATRON's physician-designed clinical-segment bins + fused `code=bin` token idea (the tokenizer itself is
rewritten as `src/data/tokenize.py`), its sequence-packing approach and benchmark, and add the pieces
that make a small (~30M, from-scratch) ICU model **transportable and clinically deployable**:

- a **threshold-conditioned time-to-event** objective (ICareFM) + **competing-risk CIF** (SurvivEHR)
  + a **value-regression "mark"** head (ORA) — replacing pure next-token prediction, the weakest objective;
- **zero-shot, training-free** survival/threshold heads → a new hospital needs no local model training
  and no manually-annotated labels to run the model (evaluation still auto-derives ground-truth labels
  from that site's own CLIF fields — no data leaves the node);
- **federated external validation** by *model-to-data*: ship a frozen model + turnkey eval, sites return
  only aggregate metrics — no raw data, labels, or gradients ever leave a node;
- a full **TRIPOD+AI calibration / decision-curve / fairness** evaluation panel.

**Thesis:** one small model → many outcomes → many hospitals → one node (2× L40, no cluster).

> **Taking this over? Start with [`AGENTS.md`](AGENTS.md)** (goal, locked decisions, hard rules), then
> [`MEMORY.md`](MEMORY.md) (single source of truth). `notes/` (incl. `NEXT_STEPS.md`) is the
> historical pre-2026-09 design record — evidence tables still useful, decisions superseded.

## Design principle — clinically derived
The model must be most sensitive where clinical **danger** is, and legible to a clinician. Concretely:
outcomes are states doctors *act on* (never treatments — those are inputs only); threshold heads are
**directional** (crossing *into* danger); the headline metric is **net benefit / decision-curve analysis**
(does acting on the model help the patient), not AUROC alone.

## Sites — develop on 3, validate on the whole CLIF federation
- **Development cohort:** MIMIC-IV-Ext-CLIF v2.1 · Rush · UChicago (CLIF origin site). *Currently
  only MIMIC is staged on the training box; Rush + UChicago are planned dev sites, not yet staged.*
- **External validation:** *all other CLIF consortium sites* via model-to-data — each runs the turnkey
  `clif-validate` package on its **local** CLIF tables and returns only aggregate + subgroup metrics.
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
- **Time:** minutes-since-ICU-admission **time-aware RoPE** (drop inserted `day_N/hour_N` tokens).
- **Trunk:** from-scratch Qwen2-arch decoder (~30M; objective, not backbone, is the lever), d512 × 8L × 8H,
  SwiGLU/RMSNorm, no QK-Norm, **untied embeddings**, context 8192. Qwen3-arch = measured ablation row.

## Layout
```
external/clifatron/      vendored upstream CLIFATRON (tokenETL, AR trainers, benchmark) — see its VENDORED.md
configs/                data.yaml · model.yaml · train.yaml
src/data/tokenize.py     CLIF parquet → fused event-token shards + vocab (primary tokenizer)
src/data/dataset.py       8192-row document-isolated sequence packing → shards (MPS/CUDA loaders)
src/data/value_stats.py   per-token robust value stats, vocab-hash-bound (training fails closed if missing)
src/model/encoder.py     from-scratch time-aware Qwen2-arch decoder (~30M primary trunk)
src/model/heads.py       threshold-hazard · competing-risk · value-regression · task heads   [KEEPER]
src/model/head_adapter.py  attach our heads to a CLIFATRON checkpoint's hidden states          [KEEPER]
src/model/generate.py     guarded rollout generation: observed-transition masks · repetition penalty · closed-world sampling
src/train/pretrain.py    torchrun DDP self-supervised pretraining (NTP→TTE curriculum)
src/train/checkpoint.py   step-granular checkpoint / resume (fresh-schedule restore)
src/eval/metrics.py      TRIPOD+AI panel: AUROC/AUPRC/ECE/Brier/calib-slope/ICI/DCA/LPE/subgroup  [KEEPER]
src/eval/method3.py      the wedge: anchor states → our probe vs XGBoost → 3×3 transport matrix   [KEEPER]
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
Key-event recall (~0.30) does **not** improve with more NTP steps — G4 prefix conditioning and
G5 RL on the L40 base are the anchoring levers. The L40 handoff (exact commands, gates, gotchas)
is frozen in [`docs/plans/l40-g2-runbook.md`](docs/plans/l40-g2-runbook.md).

## Method 3 — the wedge (smallest publishable unit)
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
1. Treatments are model **inputs**, never prediction targets.
2. Vocab = frozen CLIF mCIDE applied identically to all sites — **no cross-site raw pooling**.
3. Retrospective reports / discharge summaries are a **label source only**; only *pre-anchor* notes are features.
4. MIMIC-IV-Ext-CLIF is PhysioNet-credentialed; Rush + UChicago are institutional — **no data leaves its node.**

## References
ICareFM · SurvivEHR (npj Digit Med 2026) · ORA (arXiv:2602.00541) · Lee "Representation Before Training"
(arXiv:2604.16775) · Context Clues (arXiv:2412.16178) · Federated GEMs (arXiv:2608.02939) · Elemento ·
Cadence · TRIPOD+AI (BMJ 2024;385:e078378). Line-cited detail in `notes/METHODS.md` and `notes/RESEARCH.md`.

## License
MIT (see `LICENSE`), consistent with upstream CLIFATRON.
