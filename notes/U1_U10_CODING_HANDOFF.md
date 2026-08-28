# Coding Handoff - Evidence-Ready Model Experiments U1-U10

**Status date:** 2026-08-28  
**Current baseline:** `main` includes PR #2 (`feat/evidence-ready-model-experiments`) and PR #3 (`fix/value-head-normalization`).  
**Verification at handoff:** U1-U4 and value-stats fixes were merged after `uv run --with pytest pytest tests/ -q` passed with `108 passed, 3 skipped` on the feature branch and value-stats focused tests passed on the follow-up branch.

This handoff is the coding contract for the remaining evidence-ready experiment work. `MEMORY.md` remains the source of truth for locked scientific decisions. This document is execution guidance, not a replacement for the plan in `docs/plans/2026-08-27-001-feat-evidence-ready-model-experiments-plan.md`.

---

## Locked Decisions To Preserve

- **Novelty:** CLIF-native, open, model-to-data federation; do not claim method invention.
- **Backbone split:** from-scratch headline is Qwen3-architecture plus objective; attach/wedge path remains Qwen2 because it must match CLIFATRON checkpoint compatibility.
- **Size framing:** our compact model is the ~30M model; CLIFATRON Qwen2-0.5B is a larger comparator only.
- **Objective:** threshold hazard + competing-risk CIF + value mark + low-weight NTP with NTP-to-TTE curriculum.
- **Value-head blocker:** real pretraining must use per-token value statistics bound to the exact vocabulary hash.
- **Tokenizer:** fused `code=bin`, frozen reference-site deciles, soft discretization, forced clinical edges, storetime ordering, untied embeddings, 8192 context.
- **PORTER/TextCode:** elevated to a real transfer-robustness arm, not the shipping default.
- **Federation:** aggregate/subgroup metrics only; "label-free" means no local model fitting, not no local evaluation labels.
- **Privacy:** no patient rows, labels, predictions, identifiers, notes, or small cells leave sites.

---

## Current Merged State

### U1. Cohort, Anchor, Splits, Artifact Policy - Done

Implemented files:

- `configs/cohort.yaml`
- `configs/artifact_policy.yaml`
- `src/data/cohort.py`
- `src/data/splits.py`
- `src/data/tokenize.py`
- `src/eval/clif_auto_labeler.py`
- `tests/test_cohort.py`
- `tests/test_splits.py`
- `tests/test_artifact_policy.py`
- updates in `tests/test_data_config.py` and `tests/test_clif_validate.py`

What is now true:

- ICU hour-24 anchor semantics are explicit.
- Feature windows are pre-anchor only.
- Outcomes are represented as `positive`, `negative`, `censored`, `competing_event`, `prevalent`, `not_ascertainable`, or `unsupported_at_site`.
- Patient/grouped splits preserve linked encounters.
- Patient-level artifact destinations are policy-checked.
- Vocabulary manifests carry compatibility hashes for split, vocabulary, numeric edges, target map, outcome spec, and CLIF version.

Important caveat:

- Real site data still needs `ce-data-qa` before relying on any outcome prevalence, unit mapping, storetime field, or follow-up assumption.

### U2. Dataset, Targets, Collator, Document Isolation - Done

Implemented files:

- `src/data/dataset.py`
- `src/data/collate.py`
- `src/data/targets.py`
- `external/clifatron/AR/qwen2/data/packed_dataset.py`
- `external/clifatron/AR/qwen2/scripts/pack_sequences.py`
- `external/clifatron/AR/qwen2/train_sft.py`
- `tests/test_dataset.py`
- `tests/test_collate.py`
- `tests/test_targets.py`
- updates in `tests/test_smoke_arms.py`

What is now true:

- `ModelDataset` adapts decile-token rows and schema-2 CLIFATRON packed rows.
- Packed rows preserve episode keys, source spans, packed spans, continuation flags, and per-document anchors.
- `collate_model_samples()` emits CPU tensors, `document_ids`, `segment_map`, FlashAttention-compatible cumulative lengths, and per-anchor labels.
- Dense `CLIFEncoder` training fails closed on multi-document packed rows rather than leaking across documents.
- Missing numeric values are skipped, not encoded as lowest-value bins.
- Soft discretization weights remain normalized at edge bins.

Important caveat:

- Block-diagonal/varlen attention for Qwen2/Qwen3 training is not implemented. Multi-document packed rows are intentionally rejected in dense training paths until that is done.

### U3. Objective Semantics - Done

Implemented files:

- `src/model/heads.py`
- `tests/test_model_heads.py`
- `tests/test_cr_invariants.py`

What is now true:

- `CompetingRiskHead` uses a normalized `(K causes + no event)` per-time-bin softmax.
- CIFs are monotone and satisfy `sum_k CIF_k(t) + event_free(t) = 1`.
- CR loss supports event and censored rows correctly.
- `next_event_loss()` accepts an explicit mask and no longer requires raw token shifts.
- `ValueRegressionHead.loss_aligned()` consumes precomputed aligned value targets.
- `ThresholdHazardHead.loss()` supports observed-window censoring for no-crossing rows.

### U4. Training Engine, Checkpoints, Manifest - Done

Implemented files:

- `src/train/engine.py`
- `src/train/checkpoint.py`
- `src/train/manifest.py`
- `src/train/pretrain.py`
- `tests/test_train_engine.py`

What is now true:

- Engine uses explicit optimizer-update counting, not epoch-derived checkpoint/validation cadence.
- `max_updates` prevents overshooting `total_steps` by a full epoch.
- Gradient accumulation scales partial final updates by actual accumulated microbatches.
- Ledger counters count per-batch tokens correctly and carry forward on resume.
- RNG state and dataset epoch handling support deterministic restart contracts.
- Real pretraining fails closed without value stats or supervised TTE outcomes.
- `pretrain.py --dry-run` can normalize tokenizer rows for plumbing checks, but real training requires joined outcome/value artifacts.

### Value Statistics Follow-Up - Done

Implemented files:

- `src/data/value_stats.py`
- `tests/test_value_stats.py`

What is now true:

- Value stats artifacts must be bound to the exact vocabulary when `expected_vocab_hash` is supplied.
- Schema-2 artifacts with `vocab_hash: null` are rejected under verification.
- Legacy bare maps are accepted only when no expected vocabulary hash is supplied.

---

## Next Coding Target: U5 Only

Do U5 next. Do not begin U6-U10 until U5 passes review.

### U5 Goal

Repair evaluation, calibration, protocol freezing, aggregate-only validation, small-cell suppression, and fail-closed `clif_validate` behavior.

### U5 Files To Create Or Modify

Create:

- `src/eval/schema.py`
- `tests/test_eval_splits.py`
- `tests/test_eval_metrics.py`

Modify:

- `src/eval/metrics.py`
- `src/eval/method3.py`
- `src/eval/clif_validate.py`
- `src/eval/clif_forest_plot.py`
- `tests/test_clif_validate.py`

Optional docs updates after code is verified:

- `website/docs/evaluation-panel.md`
- `website/docs/federated-validation.md`
- `notes/NEXT_STEPS.md`

### U5 Required Behavior

1. **Fail closed in `clif_validate.py`.**
   - Remove any remaining placeholder or random prediction path.
   - Missing checkpoint, missing head weights, incompatible vocabulary, incompatible target map, missing outcome spec, or unsupported CLIF version must produce a failure status, not benchmark-shaped metrics.
   - Never load partial weights with `strict=False` in a production validation path unless the report explicitly marks the result as non-evaluable.

2. **Define a stable aggregate result schema.**
   - Put the schema in `src/eval/schema.py`.
   - Required fields should include model bundle identifiers, vocabulary hash, outcome spec hash, CLIF version, site role, partition role, metric version, outcome status, and disclosure status.
   - Explicitly distinguish `evaluable`, `unsupported_at_site`, `single_class`, `insufficient_n`, `small_cell_suppressed`, `artifact_mismatch`, and `runtime_failure`.

3. **Separate fit, calibration, and test roles.**
   - `method3.py` must not fit probes, XGBoost, temperature scaling, or model-selection artifacts on the same rows used for final evaluation.
   - At minimum, accept or derive partition labels that separate `train`, `validation`, `calibration`, and `test`.
   - If required partitions are missing, fail closed or return non-evaluable status.

4. **Prevent calibration leakage.**
   - `metrics.full_panel()` currently can fit temperature on the same labels it evaluates if called naively.
   - Add a calibration API that explicitly fits on calibration labels/logits and applies to held-out test logits.
   - Keep single-cell helper behavior available for synthetic tests, but do not let production validation fit on final test labels.

5. **Add small-cell and complementary suppression.**
   - Implement `n < 10` suppression as the baseline rule.
   - Suppress subgroup cells and complementary cells where a hidden value could be recovered.
   - Never include local file paths, patient identifiers, row-level predictions, labels, token sequences, or timestamps in result JSON.

6. **Add confidence intervals without patient export.**
   - Bootstrap or interval estimates must run locally and return only aggregate intervals.
   - If the site has too few samples or single-class labels, report non-evaluable rather than fabricating intervals.

7. **Make external estimand explicit.**
   - External validation outputs are site-specific or meta-analytic aggregates, not pooled patient-level metrics.
   - If metrics cannot be pooled from aggregate-only data, say so in the result schema and forest-plot helper.

### U5 Tests To Write

Add tests covering:

- Missing model/head/vocab/outcome-spec artifacts fail closed.
- Random predictions are impossible in production validator path.
- Temperature calibration fitted on calibration split, applied to test split.
- Final test labels cannot fit calibrators.
- Small cells and complementary subgroup cells are suppressed.
- Output JSON contains no `patient_id`, `hospitalization_id`, `hosp_id`, `sequence`, `token`, `pos_min`, local data paths, row-level predictions, or labels.
- Unsupported outcome returns an explicit non-evaluable status.
- Single-class outcomes return non-evaluable status without AUROC/AUPRC fabrication.
- Forest-plot input loader accepts only allow-listed aggregate schema.
- `method3.py` refuses unsplit site arrays for fit/evaluate workflows.

### U5 Context7 / Paperclip Usage

- Use Context7 for current `scikit-learn` probability calibration, `numpy`, and serialization details if changing metric/calibration implementation.
- Use Paperclip/PubMed only if changing the CR calibration estimand or DCA interpretation. Do not re-research basic AUROC/AUPRC.

### U5 Definition Of Done

- Focused U5 tests pass.
- Full data-free suite passes.
- `git diff --check` passes.
- Code review has no residual P0/P1 findings.
- No real-data output is produced during tests.

---

## U6. Architecture Ablations - Do After U5

Purpose:

- Compare tied vs untied embeddings and separate vs joint objective topology after evaluation is safe.

Files likely involved:

- `configs/architecture_ablation.yaml`
- `src/model/heads.py`
- `src/model/head_adapter.py`
- `src/train/run_arm.py`
- `src/eval/ablation_compare.py`
- `tests/test_architecture_ablation.py`

Coding guidance:

- Do not change defaults based on theory alone.
- Every arm must inherit the same cohort, split, vocabulary, seed, token budget, metric schema, and calibration protocol from U1-U5.
- Add parameter-count reporting so tied/untied effects are not confused with model-capacity effects.
- Keep `NextEventHead.tie_weights` as the mechanism for tying; do not add parallel head classes unless needed.

Definition of done:

- Synthetic experiment matrix resolves all arms.
- Invalid factor combinations fail before training.
- Results schema can represent each arm without changing metric code.

---

## U7. PORTER/TextCode Arm - Do After U5, Parallel With U6 If Desired

Purpose:

- Test language-grounded event inputs as a transfer-robustness arm while keeping frozen mCIDE as the default.

Files likely involved:

- `src/data/tokenize_textcode.py`
- `src/model/event_embeddings.py`
- `src/train/run_tokenization_ablation.py`
- `configs/tokenization_ablation.yaml`
- `tests/test_event_embeddings.py`
- `tests/test_tokenization_ablation.py`

Coding guidance:

- Replace synthetic concept descriptions before evidence-producing runs.
- Require authoritative mCIDE descriptions or a checked-in synthetic fixture clearly marked non-study.
- Cache text encoder embeddings with model name, revision, description hash, and vocabulary hash.
- Keep numeric magnitude in a separate value pathway.
- Do not claim open-vocabulary generation; this is input representation portability only.

Definition of done:

- TextCode cache builds deterministically from a fixture.
- Coverage failures are explicit.
- Learned-ID, text-only, and hybrid arms can be compared under identical split and metric schema.

---

## U8. Scaling, Label Efficiency, Multi-Horizon Studies - Do After U5/U6

Purpose:

- Quantify model-size, data-volume, label-efficiency, and horizon/step degradation without conflating estimands.

Files likely involved:

- `configs/experiment_matrix.yaml`
- `src/train/run_experiment_matrix.py`
- `src/eval/label_efficiency.py`
- `src/eval/multistep.py`
- `src/eval/ablation_compare.py`
- `tests/test_experiment_matrix.py`
- `tests/test_label_efficiency.py`
- `tests/test_multistep.py`

Coding guidance:

- `run_experiment_matrix.py` should orchestrate existing runtime/evaluation code, not create a second training framework.
- Keep fixed-data and fixed-compute comparisons separate.
- Use nested patient-level subsets for data-volume curves.
- Label budgets must include labels used for fitting, model selection, hyperparameter selection, and calibration.
- Multi-step forecasting must separate direct horizon prediction, teacher-forced event-step ranking, and recursive rollout plausibility.
- Do not use future treatments in calibrated anchor-time risk evaluation.

Definition of done:

- Matrix expands deterministic run IDs.
- Invalid combinations are rejected.
- Synthetic curves include uncertainty and preserve site-local disclosure controls.

---

## U9. Standalone `clif-validate` Prototype - Do After U5

Purpose:

- Prove install, bundle compatibility, offline/no-telemetry execution, and aggregate-only output before external sites are asked to run anything.

Files likely involved:

- `clif-validate/pyproject.toml`
- `clif-validate/src/clif_validate/`
- `clif-validate/tests/test_clean_install.py`
- `clif-validate/tests/test_bundle_compatibility.py`
- `clif-validate/tests/test_disclosure.py`
- `website/docs/federated-validation.md`

Coding guidance:

- Start with a synthetic bundle only.
- Support Linux x86_64 Python 3.11 first.
- Require signed release manifests, hashes, artifact-family compatibility, and outcome/vocab/target-map hashes.
- No network/telemetry during validation.
- Do not package trained weights yet unless U10 release criteria are met.

Definition of done:

- Clean offline-ish synthetic install path passes.
- Bundle mismatch fails closed.
- Output schema is allow-listed and disclosure-checked.

---

## U10. Final Bundle Qualification - Do Last

Purpose:

- Convert the selected model into a signed, governed, external-site-ready bundle.

Files likely involved:

- `clif-validate/tests/test_release_bundle.py`
- `clif-validate/src/clif_validate/`
- `website/docs/federated-validation.md`
- `README.md`

Coding guidance:

- Depends on U5, U9, and the selected model family from U6/U7/U8.
- Replace synthetic bundle with frozen model and trained heads.
- Require governance sign-off, artifact minimization, revocation/anti-rollback metadata, and cumulative disclosure ledger checks.
- If a confirmation site is reused after a model change, call it transport evaluation, not untouched external confirmation.

Definition of done:

- Signed bundle manifests validate.
- Privacy/release checks pass.
- External confirmation cannot run until bundle and analysis version are frozen.

---

## Recommended Execution Order

1. `ce-plan` for U5 only if further decomposition is needed.
2. `ce-work` U5.
3. Code review and fix U5 until no P0/P1 remains.
4. Commit/push U5.
5. U9 prototype, because external workflow proof is thesis-critical.
6. U6 and U7.
7. U8.
8. U10.

Do not start real training until:

- `ce-data-qa` has profiled the site data.
- Value stats are generated and vocabulary-bound.
- Outcome artifacts exist and contain supervised TTE labels.
- L40 driver/NVML status is healthy.
- One-batch overfit, checkpoint/resume, and DDP smoke tests pass on the L40 box.
