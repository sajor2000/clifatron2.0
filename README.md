# CLIFATRON 2.0

[![CI](https://github.com/sajor2000/clifatron2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/sajor2000/clifatron2.0/actions/workflows/ci.yml)

A **methods-upgrade layer** on [CLIFATRON](https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON),
the CLIF consortium's compact (~30M-param) CLIF-native ICU foundation model. We keep CLIFATRON's
tokenizer / sequence-packing / DeepSpeed training / benchmark and add the pieces that make a small
ICU model **transportable and clinically deployable**:

- a **threshold-conditioned time-to-event** objective (ICareFM) + **competing-risk CIF** (SurvivEHR)
  + a **value-regression "mark"** head (ORA) — replacing pure next-token prediction, the weakest objective;
- **zero-shot, training-free** survival/threshold heads → a new hospital needs no local model training
  and no manually-annotated labels to run the model (evaluation still auto-derives ground-truth labels
  from that site's own CLIF fields — no data leaves the node);
- **federated external validation** by *model-to-data*: ship a frozen model + turnkey eval, sites return
  only aggregate metrics — no raw data, labels, or gradients ever leave a node;
- a full **TRIPOD+AI calibration / decision-curve / fairness** evaluation panel.

**Thesis:** one small model → many outcomes → many hospitals → one node (2× L40, no cluster).

> **Taking this over? Start with [`notes/NEXT_STEPS.md`](notes/NEXT_STEPS.md)** — a full handoff:
> finalized token/architecture decisions, the 2026 evidence base with citations, and ordered,
> file-level next steps. Read it before touching code.

## Design principle — clinically derived
The model must be most sensitive where clinical **danger** is, and legible to a clinician. Concretely:
outcomes are states doctors *act on* (never treatments — those are inputs only); threshold heads are
**directional** (crossing *into* danger); the headline metric is **net benefit / decision-curve analysis**
(does acting on the model help the patient), not AUROC alone.

## Sites — develop on 3, validate on the whole CLIF federation
- **Development cohort (data we hold):** MIMIC-IV-Ext-CLIF v2.1 · Rush · UChicago (CLIF origin site).
- **External validation:** *all other CLIF consortium sites* via model-to-data — each runs a turnkey
  `clifpy`/tokenETL eval script on its **local** CLIF tables and returns only aggregate + subgroup metrics.
- Vocab = a **frozen** CLIF-native mCIDE, applied identically everywhere; **raw data is never pooled.**

## Tokenizer & trunk (2026-preprint spec — see `MEMORY.md` + `notes/`)
- **Tokens:** fused `code=bin` · **population deciles frozen from a reference site** (not clinical-range
  bins — no consistent gain, Lee 2026) · **soft discretization** for tail/threshold sensitivity ·
  ICU decision thresholds forced as bin edges (lactate 2/4, MAP 65, SpO₂ 88/90, KDIGO, P/F Berlin).
- **Time:** admission-relative-minute **time-aware RoPE** (drop inserted `day_N/hour_N` tokens).
- **Trunk:** Qwen2/Llama-style transformer (keep — objective, not backbone, is the lever), d512 × 8L × 8H,
  SwiGLU/RMSNorm/GQA, **untied embeddings**, context 8192.

## Layout
```
external/clifatron/      vendored upstream CLIFATRON (tokenETL, AR trainers, benchmark) — see its VENDORED.md
configs/                data.yaml · model.yaml · train.yaml
src/data/tokenize.py     CLIF parquet → fused event-token shards + vocab (decile ablation arm)
src/model/encoder.py     from-scratch time-aware decoder (ablation arm; default = CLIFATRON Qwen2)
src/model/heads.py       threshold-hazard · competing-risk · value-regression · task heads   [KEEPER]
src/model/head_adapter.py  attach our heads to a CLIFATRON checkpoint's hidden states          [KEEPER]
src/train/pretrain.py    torchrun DDP self-supervised pretraining (NTP→TTE curriculum)
src/eval/metrics.py      TRIPOD+AI panel: AUROC/AUPRC/ECE/Brier/calib-slope/ICI/DCA/LPE/subgroup  [KEEPER]
src/eval/method3.py      the wedge: anchor states → our probe vs XGBoost → 3×3 transport matrix   [KEEPER]
src/eval/matrix.py       stable re-export surface
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

### Reproduce the synthetic federated-validation result (one command)

```bash
uv sync --group dev
python -m src.eval.reproduce_synthetic
```

Runs the whole **releaser → site → aggregator** loop on synthetic fixtures (data-free, CPU) and prints
the disclosure-controlled two-site aggregate. Every trust gate is exercised on the governed path.

### Run the test suites

```bash
uv run --frozen --group dev pytest tests/ -q                 # the repo's data-free suite
uv run --frozen --group dev pytest clif-validate/tests/ -q   # the site-package suite (+ vendor-drift guard)
```

CI (`.github/workflows/ci.yml`) runs both suites on every push and pull request across Python 3.11 and
3.13, installing from the committed `uv.lock` with `--frozen` for reproducibility.

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
