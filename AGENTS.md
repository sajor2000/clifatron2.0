# AGENTS.md — CLIFATRON 2.0

> Read this first. It states the project **goal**, the **locked decisions**, and the
> **hard rules** an agent must not violate. For depth, follow the pointers at the bottom;
> `MEMORY.md` is the single source of truth where anything here is ambiguous.

---

## The goal (one paragraph)

CLIFATRON 2.0 is a **methods-upgrade layer** on
[CLIFATRON](https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON) — the CLIF
consortium's compact (~30M-param) CLIF-native ICU foundation model. We keep CLIFATRON's
tokenizer / packing / trained backbone and add the pieces that make a small ICU model
**transportable and clinically deployable**: a **threshold-conditioned time-to-event**
objective (ICareFM) + **competing-risk CIF** (SurvivEHR) + a **value-regression "mark"**
head (ORA), replacing pure next-token prediction; **zero-shot, training-free** survival heads;
and **federated external validation by model-to-data** across the whole CLIF consortium, with a
full **TRIPOD+AI** calibration / decision-curve / fairness eval panel.

**Thesis:** *one small model → many outcomes → many hospitals → one node (2× L40, no cluster).*

---

## What our novelty IS and IS NOT (locked 2026-08-27, grounded in 2026 literature)

**IS:** the first **open, CLIF-native** ICU foundation model with a **threshold-TTE objective**
validated by **model-to-data across real CLIF-consortium hospitals**. Novelty = integration +
CLIF-native first-mover + real-federation deployment.

**IS NOT:** a new method. The 2026 literature already owns every method piece — ORA
(marked-TTE), ICareFM (threshold-directional dual-zero-shot, but on *ricu*, DUA-gated), SurvivEHR
(competing-risk), Elemento (no-data-sharing ensembling). **Do not claim method invention.** The
open, deployable, CLIF-native execution is the defensible contribution.

The `clif-validate/` open shippable package is the headline artifact that distinguishes us from
DUA-gated ICareFM — treat it as a deliverable, not plumbing.

---

## Locked design decisions (do not re-litigate — change only with new evidence)

- **Objective (where the novelty lives):** threshold-hazard (primary) + competing-risk CIF +
  value-regression (ORA mark) + low-weight next-event (0.2). NTP→TTE curriculum. This is the lever;
  the backbone is a footnote.
- **Primary paper = from-scratch Qwen3-arch decoder + the objective, ~30M, fully ours.** Run as a
  ladder: (1) frozen-probe **Method-3 wedge** on a CLIFATRON Qwen2 checkpoint (cheap first result) →
  (2) from-scratch **Qwen3** pretrain (novel headline) → (3) the two = the finetune-vs-scratch ablation.
- **Backbone:** Qwen-family transformer. **From-scratch → Qwen3-arch** (free QK-Norm); **attach/wedge
  path → Qwen2** (must match CLIFATRON's checkpoint). Keep a Qwen2-arch from-scratch arm so
  "Qwen2 vs Qwen3" is a *measured* ablation row, not an assertion.
- **Size:** our own model targets **~30M** (d512×8L×8H). CLIFATRON's Qwen2 checkpoint we attach to is
  **0.5B** — always state it as a *larger comparator*, never as our compact model.
- **Tokenizer:** fused `code=bin`, frozen population deciles, soft discretization, forced clinical-threshold
  edges, storetime ordering, **untied embeddings**, **8192** context.
- **TextCode / language-grounded arm is elevated to a real transfer-robustness arm** (PORTER 2026:
  frozen-vocab drops ~69% of events on cross-site transfer). Frozen mCIDE stays primary; TextCode is
  the ablation that tests cross-site robustness.
- **Federation:** model-to-data, **aggregate + subgroup metrics only**, nothing raw leaves a node.
  "Label-free" refers to the MODEL only — evaluation still auto-derives ground-truth labels locally,
  which is a validity dependency to audit per site. Add small-cell suppression (n<10) before shipping.

---

## Hard rules (NEVER violate)

1. **Treatments are model inputs, NEVER prediction targets.**
2. **Vocab = frozen CLIF mCIDE, applied identically to all sites — no cross-site pooling of raw data.**
3. **Retrospective reports / discharge summaries = LABEL source only; only pre-anchor notes may be features.**
   An oversized note-gain is a leakage flag.
4. **`storetime`/availability ordering, not `charttime`** (no look-ahead on when a value was knowable).
5. **No data leaves its node.** MIMIC-IV-Ext-CLIF is PhysioNet-credentialed; Rush + UChicago are
   institutional. External validation returns aggregate metrics only. No rented cloud without a
   compliant BAA/DUA (Azure only inside the lab's governed tenant).

---

## Known blockers before any real training run

- **Value-head loss is unnormalized** (val≈46000 on real MIMIC) — add per-concept value scaling first.
- **No CLIFATRON checkpoint staged** yet (needed for the Method-3 wedge).
- **Rush + UChicago data not on the L40 box** (only MIMIC: 546,028 stays / ~134M events) — the 3-site
  claim needs them. Override data dir with `CLIF_DATA_DIR`.
- **transformers is v5** — verify `head_adapter.anchor_state` against a real checkpoint
  (`output_hidden_states` API changed in v5; final hidden state may be normalized).

---

## Environment & workflow

- **Package management:** `uv` only (`uv sync`, `uv run`). Never `pip install` into a shared interpreter.
- **Compute:** dev on Mac (MPS, smoke tests) or this **2× L40 Linux box `rudu-hpcg004`** (48GB each, no
  NVLink, bf16, DDP via `torchrun`). Note: `nvidia-smi` currently fails on a driver/library mismatch —
  torch CUDA still allocates, but reboot before long multi-GPU runs.
- **Tests:** `CLIF_DATA_DIR=~/Data/clif-source/CLIF_MIMIC uv run --with pytest python -m pytest tests/ -q`
  (data-gated tests skip cleanly when no CLIF data is present).
- **Docs site:** `website/` (Docusaurus, Mermaid). `cd website && npm run build`. Auto-deploys to
  GitHub Pages on push to `main` (when the repo is public / Pages is enabled).
- **Git:** work is done across machines — `git pull` at session start, commit + push after each change.
  Data / checkpoints / `clif_config.json` are per-machine and git-ignored; only code + configs are committed.
- **Do NOT commit:** data (`*.parquet`), checkpoints, `bin/` binaries, `.venv/`, `node_modules/`, `.agents/`.

---

## Where to look for detail

| For… | Read |
|------|------|
| **Single source of truth** for the spec + locked decisions | `MEMORY.md` |
| Ordered, file-level next steps + 2026 evidence tables | `notes/NEXT_STEPS.md` |
| Build-on-CLIFATRON integration plan | `notes/INTEGRATION.md` |
| Literature/evidence (⚠ pre-pivot design spec is superseded) | `notes/RESEARCH.md`, `notes/METHODS.md` |
| The rendered scientific-workflow docs (diagrams) | `website/docs/` |
| Keeper code | `src/model/heads.py`, `src/model/head_adapter.py`, `src/eval/metrics.py`, `src/eval/method3.py` |

**Precedence:** if any two documents disagree, `MEMORY.md` wins.
