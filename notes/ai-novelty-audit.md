# AI-novelty audit: tokenizer and training recipe

Audited 2026-10-03 for `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md`.
Sources were retrieved through Paperclip and PubMed; anything not retrieved is marked **[unverified]**.
No ablation results exist in the repo yet, so every verdict is about design, not evidence.

---

## 1. The design being audited

- One fused `concept=bin` token per event; about 2,083 ids on the 5,000-episode verification sample.
- Physician clinical-segment CSV (1,267 segment rows, 92 measurements), with interval closure honoured and a precedence policy that yields a strict partition.
- Forced decision edges (lactate 2/4, MAP 65, SpO2 88/90, creatinine 1.5/2/3), closed on the non-event side and shared with the threshold-hazard queries through one `bin_index`.
- Every numeric concept is binned: CSV, then ordinal, then quantile, then single.
- Soft discretization on the input side only: a Gaussian over the neighbouring bins; the next-event target stays the hard id.
- Zero-aware dose bins, unit conversion, and per-kg dosing from the most recent prior weight.
- Availability-time ordering with a per-table declaration; deterministic, byte-identical output.
- Treatments, respiratory support, CRRT, ECMO, code status, ADT and static tokens are inputs only.
- Admission-relative minute RoPE with no time tokens; untied embeddings; 8192 context.
- Full-hospitalization stream ending in `DISCHARGE//*` terminals.
- Objective: threshold hazard, competing-risk incidence, value mark and low-weight next token, with a next-token-to-time-to-event curriculum.
- A Qwen2-style decoder of about 30M parameters by design (larger as built; the reported size must be measured).

---

## 2. Closest prior work and verdicts

### Numeric tokenization — incremental, with one narrow novel combination

- **Lee et al. 2026 (arXiv 2604.16775)**, 28 matched transformers on MIMIC-IV, 30 outcomes, including CLIF-remapped codes:
  - Fused code-value tokens raise mortality AUROC from 0.891 to 0.915.
  - Event order only and admission-relative RoPE match or exceed time tokens while shortening sequences by 11%.
  - "Finer-than-decile quantization, reference-range anchoring, and soft discretization help in selective outcomes."
  - Soft discretization was tested unfused only; the fused version is named as follow-up.
- **Guo et al. 2026 (arXiv 2603.15644):** joint code-value encoding wins 73 of 74 tasks with 39.5% fewer pretraining FLOPs; time-positions win 71 of 74.
- **Montgomery et al. (arXiv 2607.01391):** hybrid binning before projection is the robust default; clinical gains are modest.
- **McCann et al. (medRxiv 10.64898/2026.08.04.26359713):** continuous-fused gives 34% shorter sequences; discrete is better calibrated.
- **LabTOP (arXiv 2502.14259):** digit-wise tokens beat quantile bins for lab-value prediction.
- **Expert-knowledge precedents:** INTERVenE (arXiv 2608.29901) and HALO (arXiv 2304.02169).

What remains ours:
- Bin edges tied to the threshold-hazard queries.
- Soft discretization on fused tokens.
- Physician decision-zone segments beyond lab reference ranges, covering vitals, doses and ventilator settings.
- Zero-aware dosing is engineering, not ML novelty.

The prior from Lee is a null on averages, with possible gains on tail outcomes.

### Time encoding — already published

Guo 2603.15644, Lee 2604.16775, Al Attrach (arXiv 2512.05217), MIRA (arXiv 2506.07584) and Burkhart (arXiv 2608.02939) cover it. Reviewers will ask for an order-only arm, and RoPE may not beat it.

### Language-grounded codes — already published; incremental here

PORTER (arXiv 2606.24102) recovers 97.1% of target-vocabulary AUROC where a fixed vocabulary dropped 69% of events on an unharmonized transfer. Under mCIDE harmonization that mechanism largely disappears (our non-fit `<unk>` rate is 0.021%; Burkhart's cross-site penalty is 0.025 AUROC). The publishable question is whether harmonization or language grounding buys transport.

### Objectives — a novel combination, not a method

- **ORA (arXiv 2602.00541):** marked time-to-event beats next-token by 10.7% (Transformer) and 11.4% (Mamba) on average over 14 tasks, at a 120M budget.
- **ICareFM (medRxiv 10.1101/2025.07.25.25331635):** threshold-conditioned time-to-event, dual zero-shot median AUROC 0.837.
- **SurvivEHR (PMID 42106492):** competing-risk next-event pretraining.
- **CEHR-XGPT (arXiv 2509.03643):** Gamma time-to-event loss.
- **EveryQuery (arXiv 2603.07900):** task-conditioned queries beat rollouts on 82% of 39 tasks, about 3,000 times faster.
- MOTOR was seen only through citations **[unverified]**.

No retrieved paper trains threshold hazard, competing-risk incidence, value mark and next token together on a tokenized ICU stream, and no controlled test of a next-token-to-time-to-event curriculum was found.

Related training results:
- Patient-aware sampling (arXiv 2607.22114) improves macro AUROC and AUPRC.
- SCOPE/REACH (arXiv 2602.03730): about 10 times fewer samples for mortality and about 1.2 times for ICU admission.
- CoMET (arXiv 2508.12104): an optimal token-to-parameter ratio near 1,000:1.
- FlatASCEND (arXiv 2605.04071): reward post-training took correct associations from 3 of 3 to 0 of 3.

### Evaluation expectations

- **Burkhart 2608.02939:** 12 post-24h CLIF tasks, with gradient-boosted trees on token counts and logistic regression on tokens as baselines. The decile GEM reaches 0.810 within-site AUROC vs 0.793 for LightGBM; the cross-site penalty is 0.025 vs 0.079.
- **Other suites:** EHRSHOT (arXiv 2307.02028), YAIB (arXiv 2306.05109), MEDS-Tab (arXiv 2411.00200), FoMoH (arXiv 2505.16941), ETHOS-ARES (PMID 41026508).
- **Intervention-conditioned generation:** HealthFormer (arXiv 2604.27899; direction agrees in 41 of 41 trial comparisons, mean inside the CI in 30); FlatASCEND (4 of 10 correct mechanistic directions); EHR-MPC (arXiv 2607.08793).

---

## 3. The three defensible claims

| Claim | Experiment | Falsified if |
|---|---|---|
| Threshold-aligned fused tokenization improves zero-shot threshold prediction and calibration | Clinical bins with soft inputs vs hard inputs vs deciles vs deciles with forced edges vs continuous-fused, at matched bin count, at least 3 seeds, on thresholds that sit on a bin edge (MAP 65, lactate 4) and off it (MAP 60, lactate 3) | Deciles match within the interval on-edge, or the gain disappears at matched granularity |
| The combined marked time-to-event objective beats next-token at equal compute | Next-token only vs each head removed vs full, plus a no-curriculum arm; rollouts estimated with SCOPE/REACH | A next-token model with a linear probe, or rollouts, tie on AUROC and calibration |
| Transport under a frozen harmonized vocabulary | Model-to-data penalty at external CLIF sites vs the decile GEM's 0.025; TextCode adds nothing under mCIDE | The penalty exceeds the decile GEM's, or TextCode wins materially |

The plan adopts the first two as claims 1 and 2 and reports transport as an evaluation column; its third claim is the extubation application.

---

## 4. Expected techniques not yet in the recipe

Ranked by payoff over cost for a model of this size on 2× L40:

1. Model-size sweep (3M, 10M, 30M). The site-1 corpus is small relative to CoMET's token-to-parameter ratio.
2. Gradient-boosted trees on counts, logistic regression on tokens, and a decile next-token GEM with a linear probe as the Burkhart replica. Mandatory.
3. An order-only time arm.
4. Output-side ordinal or soft targets (Lee, FlatASCEND, arXiv 2603.07448). Our targets are hard.
5. SCOPE/REACH estimators for rollout evaluation.
6. A continuous time head (zero-inflated log-normal in FlatASCEND, Gamma in CEHR-XGPT).
7. A patient-sampling weight ablation.
8. Scheduled sampling, only if generation claims are made.
9. The Muon optimizer for hidden matrices, as in Lee.

---

## 5. Minimum experiment table

Columns: (a) 12 CLIF post-24h tasks; (b) zero-shot threshold outcomes, on- and off-edge; (c) calibration; (d) value regression; (e) time-to-event; (f) cross-site penalty; (g) generative checks.

| Row | a | b | c | d | e | f | g |
|---|---|---|---|---|---|---|---|
| LightGBM counts / LR tokens | ✓ | ✓ | ✓ | – | – | ✓ | – |
| Decile fused, next-token, order-only (Burkhart replica) | ✓ | rollout | ✓ | – | – | ✓ | ✓ |
| Clinical hard, next-token only | ✓ | rollout | ✓ | – | – | – | ✓ |
| Clinical soft, full objective (primary) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Clinical hard, full | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| Global deciles, full | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| Deciles plus soft with forced edges, full | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| Continuous-fused, full | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| TextCode, full | ✓ | ✓ | ✓ | – | – | ✓ | – |
| Primary minus mark / minus incidence / minus curriculum | ✓ | ✓ | ✓ | ✓ | ✓ | – | – |
| Primary with order-only / time tokens | ✓ | ✓ | – | – | ✓ | – | – |
| Primary at 3M / 10M | ✓ | ✓ | ✓ | – | – | – | – |
| CLIFATRON 0.5B frozen probe (larger comparator) | ✓ | – | ✓ | – | – | ✓ | – |

Three or more seeds, bootstrap intervals and a Benjamini-Hochberg correction, as Lee does.

---

## 6. Repo citations that do not match the retrieved sources

- `configs/model.yaml` says ORA lifts physiology tasks "+33-38% vs NTP"; the retrieved text gives 10.7% and 11.4% average gains.
- `notes/RESEARCH.md` says SCOPE/REACH cut rollout cost "2.5–3.4× (>80×)"; the retrieved abstract gives about 10 times and about 1.2 times.
- `configs/tokenization_ablation.yaml` says "deciles+soft wins tails" (Lee); that is a point-estimate result from unfused arms only.
- `configs/model.yaml` attributes "untied +4–7% AUPRC" to medRxiv 2026.04.24.26351503; the retrieved paper is an LLM-versus-clinical-foundation-model comparison and the claim could not be confirmed there.
- arXiv 2505.22964 (scaling laws) was not retrieved.
- `AGENTS.md` says the CSV has 1268 rows; `website/docs/data-tokenization.md` says 1267.
