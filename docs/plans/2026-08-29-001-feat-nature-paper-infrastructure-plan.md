---
title: "feat: Nature-paper infrastructure — CI, reproducible locks, docs, synthetic reproduction"
date: 2026-08-29
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
---

# feat: Nature-paper infrastructure for the CLIF ICU foundation model codebase

## Summary

The evidence machinery has landed and merged — U1–U5 (cohort → training → evaluation), U9 (validator
core), U11 (release trust), U13/U14 (attention + resume/DDP verification), and U15 (synthetic
federation harness). What the codebase still lacks is the **research-grade infrastructure** a Nature
methods paper is judged on: continuous integration that actually runs the data-free test suite,
reproducible pinned dependencies, reader-facing documentation and a model card, and a one-command
synthetic reproduction. This plan closes those gaps. It is scoped **strictly to data-free, unblocked
work** — real-data training, GPU qualification (U8, U13-FA2, U14 report), and real-site federation
(U12, and U6/U7/U8 gated behind it) stay out of scope because they are blocked on hardware and
governance, not on code.

## Problem Frame

For a methods paper, the codebase IS part of the contribution: reviewers and readers must be able to
trust that the tests pass, reproduce the synthetic result, and understand the pipeline without reading
every module. Today:

- **No CI runs the tests.** `.github/workflows/` contains only `deploy-docs.yml`. The 358-test repo
  suite and the 32-test `clif-validate` suite run only on a developer's machine. A regression can land
  on `main` unseen — unacceptable for a paper artifact.
- **Dependencies are not reproducibly pinned at the root.** The root `uv.lock` is gitignored; only
  `clif-validate/uv.lock` is committed (from U11). A reader cannot recreate the exact environment the
  results were produced in.
- **No model card, thin reader-facing docs.** `README.md` exists but there is no `MODEL_CARD.md` and no
  single architecture/reproducibility doc that a reviewer can read to understand the CLIF ICU
  foundation-model training + federated-validation pipeline and its disclosure controls.
- **No one-command synthetic reproduction.** The releaser → site → aggregator loop is proven by the
  U15 test, but there is no documented entrypoint that runs it end to end and prints the aggregate —
  the "reproduce our synthetic result in one command" affordance reviewers expect.

## Requirements

- **R1.** CI runs the full data-free repo suite and the `clif-validate` suite on every push and pull
  request, fails the build on any test failure, and is reproducible via a pinned lock.
- **R2.** The root Python environment is reproducibly pinned (committed `uv.lock`) and CI installs from
  it with `--frozen`.
- **R3.** A reader-facing documentation layer — enhanced `README.md`, a `MODEL_CARD.md`, and an
  architecture/reproducibility doc — describes the pipeline, its data-free-by-default and fail-closed
  posture, disclosure controls, intended use, and limitations, grounded in the multi-site
  transferability framing of the surrounding literature.
- **R4.** A single documented, tested command reproduces the synthetic federation result
  (releaser → site → aggregator) end to end and prints the disclosure-controlled aggregate.
- **R5.** Every change is data-free, fail-closed-preserving, and adds no real PHI/data to any output;
  new behavior-bearing code carries tests; the `clif-validate` vendor-drift guard stays green.

## Key Technical Decisions

- **KTD-1 — CI on GitHub Actions with `astral-sh/setup-uv`, matrix over the supported Python floor and
  ceiling.** Use `actions/checkout@v5` + `astral-sh/setup-uv@v10` (pinned) with `enable-cache: true`,
  a matrix of Python `3.11` (the declared floor) and `3.13` (the dev interpreter), installing from the
  committed lock and running `uv run --frozen`. *Rationale:* this is the canonical, cache-fast uv-in-CI
  pattern (Context7 `/astral-sh/setup-uv` v10), it matches the repo's uv-only workflow (per the user's
  global tooling), and 3.11+3.13 brackets the `requires-python = ">=3.11"` range without a wasteful
  full matrix. *Alternative rejected:* `pip`/`venv` CI (diverges from the repo's uv-only stance and
  re-resolves non-reproducibly); a single Python version (misses a floor/ceiling incompatibility that
  a paper artifact should catch).
- **KTD-2 — a `dev` dependency group (pytest + cryptography) plus a committed root `uv.lock`, run with
  `--frozen`, over the ad-hoc `uv run --with pytest --with cryptography`.** The `--with` form
  re-resolves on every invocation and pins nothing; a `[dependency-groups] dev` group captured in a
  committed lock makes CI and local runs install the *same* versions. `cryptography` is already a
  runtime dep (U11), so the group adds only the test runner. *Alternative rejected:* keeping `--with`
  in CI (non-reproducible, the exact failure mode R2 exists to close).
- **KTD-3 — two separate `pytest` invocations in CI (`tests/` and `clif-validate/tests/`), one
  environment.** The two suites each contain a `test_trust.py`; collecting both in one `pytest` run
  fails on the duplicate basename (observed this session). Both suites import `torch`/`transformers`/
  `duckdb`/`polars`/`cryptography`, all in the *root* dependency set, and `clif-validate`'s `conftest`
  puts its package and the repo root on `sys.path` — so one root env runs both, as two steps.
  *Alternative rejected:* installing `clif-validate` as a second environment (duplicate heavy installs
  for no isolation benefit — the vendor-drift guard specifically needs the repo tree present).
- **KTD-4 — the synthetic reproduction is a thin, tested CLI entrypoint that composes the landed
  pieces, not new science.** `python -m src.eval.reproduce_synthetic` builds a synthetic site + signed
  bundle, runs the site CLI (draft → approve) for two synthetic sites, aggregates, and prints the
  panel. It reuses `synthetic_bundle`, `clif_validate.main`, and `aggregator` unchanged. *Rationale:*
  reviewers get a one-command reproduction without any new trusted code path; the U15 test already
  proves the pieces, so this unit's own test is a smoke test that the entrypoint wires them and stays
  data-free.
- **KTD-5 — documentation is data-free and honest about scope.** The model card and architecture doc
  state plainly what is proven (synthetic, CPU, data-free) versus pending (real-data training, GPU
  qualification, real-site federation, governance), so the paper artifact never overclaims. This
  mirrors the repo's fail-closed, "report what actually happened" ethos.

## Implementation Units

Unit IDs continue the project's global numbering (U1–U15 landed) to avoid collision; they are still
plan-local for `ce-work` reference.

### U16. Reproducible root lock + test dependency group + consistency guard

**Goal:** Make the root Python environment reproducibly pinned and give CI a frozen, grouped install.

**Requirements:** R2, R5

**Dependencies:** none

**Files:**
- `pyproject.toml` (modify) — add `[dependency-groups]` with a `dev` group: `pytest`, and
  `cryptography` (already a runtime dep, but keep the test runner grouped with it for clarity).
- `uv.lock` (create) — generated by `uv lock` at the repo root.
- `.gitignore` (modify) — add a negation `!uv.lock` so the root lock is committed (mirrors the U11
  pattern that un-ignored `clif-validate/uv.lock`).
- `tests/test_repro_lock.py` (create) — the consistency guard.

**Approach:** Add the `dev` group; run `uv lock` to produce the root lock; un-ignore it. Mirror the
existing `clif-validate/tests/test_packaging.py` guard so the root lock cannot silently drift from
`pyproject.toml`: assert the lock pins every declared runtime dependency and that the `cryptography`
version clears the U11 advisory floor (`>= 44.0.1`). Keep the guard cheap and offline (parse
`uv.lock` TOML + `pyproject.toml`); skip only via an explicit opt-out env var, otherwise a missing lock
FAILS (not skips), matching the `clif-validate` guard's philosophy.

**Patterns to follow:** `clif-validate/tests/test_packaging.py` (guard shape, fail-not-skip on missing
artifact, cryptography-floor assertion); the U11 `.gitignore` negation for `clif-validate/uv.lock`.

**Test scenarios** (`tests/test_repro_lock.py`):
- Happy path: every dependency name in `pyproject.toml [project.dependencies]` appears as a package in
  the root `uv.lock` (normalized names).
- Security floor: the locked `cryptography` version is `>= 44.0.1` (the U11 advisory floor).
- Fail-closed: a missing `uv.lock` makes the guard FAIL, not skip, unless an explicit
  non-packaging opt-out env var is set.

**Verification:** `uv lock` produces a committed `uv.lock`; `uv run --frozen --group dev pytest
tests/test_repro_lock.py` passes; `git check-ignore uv.lock` reports it is no longer ignored.

### U17. Continuous integration for the data-free suites

**Goal:** Run both data-free test suites on every push and PR, fail on any test failure, reproducibly.

**Requirements:** R1, R2, R5

**Dependencies:** U16 (needs the committed root lock + `dev` group for `--frozen`).

**Files:**
- `.github/workflows/ci.yml` (create) — the CI workflow.

**Approach:** A `ci` workflow triggered on `push` and `pull_request`. One `test` job with a
`strategy.matrix.python-version: ["3.11", "3.13"]`. Steps: `actions/checkout@v5`; `astral-sh/setup-uv@v10`
pinned by commit SHA with `enable-cache: true` and the matrix `python-version`; `uv sync --frozen
--group dev`; then TWO test steps — `uv run --frozen --group dev pytest tests/ -q` and `uv run
--frozen --group dev pytest clif-validate/tests/ -q` (separate invocations per KTD-3). Do not use
`continue-on-error`; a red suite must fail the build. Concurrency: cancel superseded runs on the same
ref. The existing `deploy-docs.yml` is untouched.

**Execution note:** This is CI configuration; the right proof is that the workflow runs green on the PR
that introduces it (runtime/smoke verification), not unit tests of the YAML.

**Patterns to follow:** Context7 `/astral-sh/setup-uv` canonical CI (`checkout@v5` → `setup-uv@v10`
`enable-cache` → `uv run --frozen pytest`, matrix over Python versions); the repo's existing
`.github/workflows/deploy-docs.yml` for workflow conventions and permissions style.

**Test scenarios:** `Test expectation: none -- CI configuration.` Verification is the live workflow
result on the introducing PR.

**Verification:** On the PR that adds `ci.yml`, both matrix legs run and both `pytest` steps pass
(green check); introducing a deliberate failing test locally would turn the job red (do not commit
that — reason about it).

### U18. Reader-facing documentation: README, model card, architecture/reproducibility doc

**Goal:** Give a reviewer the documentation to understand and trust the pipeline without reading source.

**Requirements:** R3, R5

**Dependencies:** none (independent of U16/U17; may land in parallel).

**Files:**
- `README.md` (modify) — audit and enhance: the "one small model → many outcomes → many hospitals →
  one node" thesis, the landed-unit map (training → validator → release trust → federation), how to run
  the data-free test suites, the CI badge, links to `clif-validate/`, `MODEL_CARD.md`, and the
  architecture doc. Preserve any existing accurate content.
- `MODEL_CARD.md` (create) — a data-free model card: intended use (research validation of a CLIF ICU
  foundation model), training-data *class* (CLIF 2.1 tables; no PHI in this repo), outcomes/targets,
  evaluation and calibration approach, disclosure controls (small-cell suppression, cumulative-ledger
  differencing, fail-closed release trust), limitations, and an explicit "proven vs pending" section
  (synthetic/CPU/data-free proven; real-data training, GPU qualification, real-site federation, and
  governance pending).
- `docs/architecture.md` (create) — the pipeline architecture + reproducibility guide: components and
  their trust boundaries (releaser → site → aggregator), the two signing directions (Ed25519
  releaser→site vs HMAC site→aggregator), the disclosure ledger and differencing, and how to reproduce
  the synthetic result (points at U19's entrypoint). Include a `mermaid` component/flow diagram.

**Approach:** Documentation only; every claim must be true against `main` and honest about scope
(KTD-5). Ground the multi-site framing in the surrounding literature (multi-center adaptability,
transferability) without overclaiming. Do NOT restate the full plan; link to it. Keep the model card to
the standard model-card structure. Verify every code/file reference resolves.

**Patterns to follow:** the existing `website/docs/` docs (if present) for tone; the repo's honest
"report what actually happened" convention; `configs/trust_roles.yaml` `pending_governance` for the
honest-scope framing.

**Test scenarios:** `Test expectation: none -- documentation.` Verification is a link/claim audit
(every referenced path exists; every "proven" claim matches a landed, tested unit).

**Verification:** `README.md`, `MODEL_CARD.md`, and `docs/architecture.md` exist; a reviewer can, from
them alone, run the test suites and the synthetic reproduction; no claim overstates what is proven.

### U19. One-command synthetic reproduction entrypoint

**Goal:** A single documented, tested command reproduces the synthetic federation result end to end.

**Requirements:** R4, R5

**Dependencies:** U15 (aggregator + synthetic fixtures + site CLI — all landed). Docs reference it (U18).

**Files:**
- `src/eval/reproduce_synthetic.py` (create) — a thin CLI: build a synthetic site + signed bundle
  (`synthetic_bundle`), run the site CLI (`clif_validate.main`) draft → approve for two synthetic sites
  under a temp working dir, aggregate the two signed reports (`aggregator.aggregate_site_reports`), and
  print the disclosure-controlled multi-site panel plus a "no PHI / all gates passed" summary. `python
  -m src.eval.reproduce_synthetic`.
- `tests/test_reproduce_synthetic.py` (create) — a smoke test.

**Approach:** Compose the landed pieces unchanged — no new trusted code path, no new fail-closed gate
(the gates live in the units it calls). It is the U15 e2e flow packaged as a runnable, reader-facing
entrypoint. It must stay data-free (synthetic fixtures only, temp dir, never touches real paths) and
must not weaken any gate it invokes. Output is the aggregate panel (metrics/summary only, no raw
patient counts) plus a confirmation line.

**Execution note:** Smoke-first — the entrypoint's value is that it *runs*; the test asserts it produces
a two-site panel with no patient-level fields and exits cleanly. Reuse the U15 test's fixture-building
approach (temp CWD, non-UTC TZ to exercise the DuckDB UTC pin).

**Patterns to follow:** `tests/test_federation_e2e.py` (the exact releaser → site → aggregator wiring,
per-site key handling, draft→approve ceremony); `src/eval/clif_forest_plot.py::main` for the CLI
`argparse`/`__main__` shape; the aggregator's no-PHI panel assertions.

**Test scenarios** (`tests/test_reproduce_synthetic.py`):
- Happy path: running the entrypoint (invoked in-process under a temp dir) returns/prints a panel with
  `n_sites == 2` and both site ids, and a success summary.
- Disclosure: the printed/returned panel contains no local filesystem path and no raw patient-level `n`
  (only allow-listed summary stats), mirroring the U15 happy-path leak assertion.
- Fail-closed composition (light): the entrypoint uses the governed path (signature + content-hash +
  rollback verified); it does not pass `--allow-unsigned` — assert the reproduction runs the *governed*
  ceremony, not an escape hatch.

**Verification:** `uv run --with pytest --with cryptography pytest tests/test_reproduce_synthetic.py`
passes; `python -m src.eval.reproduce_synthetic` prints a two-site aggregate with no PHI; the full
data-free suites stay green.

## Scope Boundaries

**In scope:** data-free CI, reproducible root lock, reader-facing docs + model card, and a tested
synthetic-reproduction entrypoint — the infrastructure a methods-paper artifact is judged on.

### Out of scope (blocked on hardware/governance, not code)
- **Real-data training and evaluation** — needs real CLIF tables + `ce-data-qa`.
- **GPU qualification** — U8, U13's FA2 path, U14's 2×L40 hardware report.
- **Real-site federation (U12) and the units behind it (U6/U7/U8)** — external-site onboarding + the
  standing governance question ("may a pre-selection v0 bundle run at an external site?").

### Deferred to Follow-Up Work
- Coverage reporting / a coverage gate in CI (add once CI is green and stable).
- Packaging the coordinating-center aggregator for distribution (only if an operational need appears).
- A Dockerfile / devcontainer for the training environment (valuable but larger; not paper-blocking).

## Verification Contract

- `uv run --frozen --group dev pytest tests/ -q` and `uv run --frozen --group dev pytest
  clif-validate/tests/ -q` both pass locally and in CI.
- `tests/test_repro_lock.py`, `tests/test_reproduce_synthetic.py` pass; the `clif-validate` vendor-drift
  guard stays green.
- The CI workflow runs green on the PR (both matrix legs, both suites).
- `git diff --check` clean; no real-data/PHI in any output; `python -m src.eval.reproduce_synthetic`
  emits only disclosure-controlled aggregates.
- Every documentation claim is true against `main` and honest about proven-vs-pending scope.

## Definition of Done

CI runs the data-free suites on every push/PR from a committed root lock; the root environment is
reproducibly pinned with a drift guard; a reviewer has a README, model card, and architecture/repro doc
that accurately describe the pipeline and its scope; and one command reproduces the synthetic federation
result with no PHI. All data-free, fail-closed-preserving, suites green.

## Sources & Research

- **Context7 `/astral-sh/setup-uv` (v10):** canonical uv-in-CI — `checkout@v5` → `setup-uv@v10`
  (`enable-cache`) → `uv run --frozen pytest`, matrix over Python versions. Shaped KTD-1/KTD-2/KTD-3.
- **Context7 `/pyca/cryptography`:** confirmed the Ed25519 usage the pipeline already ships is canonical
  (no change needed) — informs the model card's disclosure-controls section.
- **Paperclip (EHR/ICU foundation-model literature):** multi-center adaptability of shared EHR
  foundation models (PMC11211479); benchmarking of pre-training strategies (PMC12349770); and the
  authors' own transferability work (Burkhart, Rojas, Parker et al., arXiv 2504.10422). Grounds the
  model card's and architecture doc's multi-site-transferability framing and the "reproducible,
  testable, open infrastructure" quality bar for a methods paper.
