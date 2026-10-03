# Clinical decisions of 2026-10-03

**Decided by:** the product authority, J.C. Rojas, on 2026-10-03, in answer to 47 numbered
questions (sections A-K) raised before the first L40 run. Item 7 was revised the same day
(see the correction below). This log records each decision, the default it replaced, the key
references, where it is implemented, and its status. It holds no patient data and no counts.

**Status:** `implemented` = in code or config and test-locked; `provisional` = in place but
to be re-checked before it can bear a claim; `deferred` = decided, not yet built;
`implemented (parallel)` = applied in the same change set to files owned by the
tokenizer/cohort workstream (`src/data/tokenize.py`, `tokenization_report.py`, `units.py`,
`configs/data.yaml`, `configs/cohort.yaml`, `configs/extubation.yaml`,
`src/data/extubation_cohort.py`, `src/eval/extubation_labeler.py`,
`configs/literature_segments/*`); check that file for the final form.

**Margin values** under items 42-45 (preserved fraction 0.5, mortality cap 1.2, equivalence
margin 1.5, negative-control register, positive-control floor) keep `status: proposed` in the
configs until the emulation protocol is registered. The method is decided; the values are
registered with the protocol.

## CLIF 2.1 conformance check (product authority instruction, 2026-10-03)

Every final decision must match CLIF 2.1 mCIDE and CLIF 2.1 units. Checked against the
mCIDE 2.1.1 snapshot (`configs/clif_mcide_2.1.1/`, else
`output/final_no_phi/clif_spec_v2.1.1/`):

- Every concept named in `configs/thresholds.yaml` (decision, control and competing-risk
  cause, including the KDIGO creatinine rule) is a CLIF 2.1 `lab_category` or
  `vital_category`, and every value is in the CLIF 2.1 unit declared under the file's
  `units` block: lab units equal the mCIDE `reference_unit`; `map` (mmHg), `spo2` (%) and
  `temp_c` (Celsius) equal the vitals DDL; `heart_rate`, `sbp` and `respiratory_rate` are
  declared "no unit" by CLIF 2.1 and are used as charted. Test:
  `tests/test_threshold_grid.py::ClifUnitsTest`.
- The extubation arms in `configs/extubation_benchmarks.yaml` are explicit groupings of
  permissible CLIF 2.1.1 `respiratory_support.device_category` values
  (`arm_device_categories`: conventional oxygen = Nasal Cannula, Face Mask, Room Air;
  HFNC = High Flow NC; NIV = NIPPV, CPAP), equal to `configs/extubation.yaml`. Test:
  `tests/test_causal_benchmark.py::test_arm_groupings_are_permissible_clif_2_1_device_categories`.
- The eight negative-control outcomes (item 45) are CLIF 2.1 lab categories in their CLIF
  units. Test: `tests/test_extubation_audit.py::test_negative_control_outcomes_are_registered_with_rationales_in_clif_terms`.

## A. Competing-risk deterioration thresholds (items 1-7)

The organ labs sit on the SOFA score-2 boundary (the Sepsis-3 unit of deterioration);
the vitals without a SOFA component are anchored to NEWS2. Direction is strict (`above` is
`>`, `below` is `<`). All in `configs/thresholds.yaml` `competing_risk_cause`, status
`confirmed`.

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 1 | Respiratory-rate cause | Confirm `> 24` /min for the first L40 screen; RR >= 25 (NEWS2 top band) logged as the secondary sensitivity value (`sensitivity_value: 24.9` under the strict `above` rule; identical to `> 24` for integer-charted rates). See "Sustained crossings" below. | `> 24`, proposed | Yousefi 2024 doi:10.1186/s12873-024-01084-w; Badawy 2017 doi:10.1136/bmjqs-2017-006671; Latten 2019 doi:10.1371/journal.pone.0223155 | `configs/thresholds.yaml` | implemented |
| 2 | Creatinine cause | KDIGO AKI rule instead of a fixed value: event when creatinine rises >= 0.3 mg/dL above the lowest value in the preceding rolling 48 h, or reaches >= 1.5x the lowest value in the preceding rolling 7 days. Baseline: tier 1 of the authority's rule (outpatient creatinine 7-365 days before admission) is not in the token stream, so tier 2 (lowest in-stay value inside the window) is used; tier 3 (back-calculation from an assumed eGFR of 75) is a documented future sensitivity analysis. A measurement with no earlier creatinine in the 7-day window cannot be assessed (`not_ascertainable`, never negative). Labels are computed on the whole stay in availability order. The threshold-hazard QUERY grid for creatinine keeps its fixed decision thresholds 1.5 / 2.0 / 3.0 mg/dL. | `> 2.0` mg/dL, proposed | KDIGO 2012 AKI guideline (Khwaja 2012 doi:10.1159/000339789); criteria as quoted in PMID 33750089; pyAKI doi:10.1371/journal.pone.0315325; De Rosa 2016 doi:10.1186/s13054-016-1218-4; Cooper 2021 doi:10.1016/j.ekir.2020.12.020 | `configs/thresholds.yaml` (`rule: kdigo_aki`), `src/data/threshold_grid.py` (`RuleQuery`), `src/data/targets.py` (`kdigo_aki_flags`, `_label_rule`); tests in `tests/test_targets_gem_tte.py::KdigoCreatinineCauseTest` | implemented |
| 3 | Bilirubin cause | Confirm `> 2.0` mg/dL (SOFA liver score 2) | same, proposed | Singer 2016 doi:10.1001/jama.2016.0287 | `configs/thresholds.yaml` | implemented |
| 4 | Platelet cause | Confirm `< 100` x10^3/uL (SOFA coagulation score 2) | same, proposed | Singer 2016 doi:10.1001/jama.2016.0287 | `configs/thresholds.yaml` | implemented |
| 5 | Heart-rate cause | Confirm `> 130` /min (NEWS2 score 3 at >= 131) | same, proposed | Yousefi 2024 doi:10.1186/s12873-024-01084-w | `configs/thresholds.yaml` | implemented |
| 6 | SBP cause | Confirm `< 90` mmHg. The authority noted it co-fires with MAP < 65 in most hypotensive patients; kept as the SBP-native token. | same, proposed | Woolfe Loftus 2025 doi:10.1111/jocn.17500 | `configs/thresholds.yaml` | implemented |
| 7 | Temperature cause | **Final: `> 39.166` C (102.5 F).** Anchored to NEWS2's top high-temperature band (>= 39.1 C) and snapped to the physician-CSV edge 39.166, so "above 39.166" is exactly the top bin (39.166, 44.0] and the crossing is readable from the token stream, like MAP 65 and lactate 2 / 4. Fever-framing alternative noted, not used: 38.333 C (101 F), the SCCM/IDSA ICU fever definition. | `> 38.3` C, proposed (interim same-day value: `> 39.1`) | Riccalton 2025 doi:10.1186/s12916-025-03943-0; O'Grady 2023 doi:10.1097/CCM.0000000000006022 | `configs/thresholds.yaml`; `tests/test_threshold_grid.py::test_temperature_cause_is_a_physician_csv_edge_and_the_top_bin` | implemented |

**Correction (item 7).** An interim note anchored temperature to "the NEWS2 score-3 band,
consistent with the other vitals". That is wrong: in NEWS2, >= 39.1 C scores **2**; on the
temperature axis only <= 35.0 C scores 3 (Riccalton 2025, doi:10.1186/s12916-025-03943-0,
for the NEWS2 bands). The interim value 39.1 was superseded the same day by 39.166 (above),
which the authority chose for the CSV edge and the token-stream readability, not for a
score-3 equivalence. Respiratory rate (>= 25), heart rate (>= 131) and SBP (<= 90) are
NEWS2 score-3 cut-offs.

### Sustained crossings and per-cause event rates (product authority, 2026-10-03)

Charted respiratory rate is a noisy spot estimate that clusters at 18 and 20 /min (Badawy
2017, BMJ Qual Saf, doi:10.1136/bmjqs-2017-006671) and is measured with only moderate
inter-observer agreement (Latten 2019, PLoS One, doi:10.1371/journal.pone.0223155);
threshold duration changes what a crossing predicts in monitored patients (Aagaard 2026,
Sensors, doi:10.3390/s26175399). All three verified on PubMed (PMIDs 28652259, 31581207,
42740020). So, without changing the confirmed values:

- **Optional sustained-crossing rule, off by default.** Any fixed-value competing-risk cause
  in `configs/thresholds.yaml` may carry `sustained: {min_readings: k, within_minutes: w}`
  (for example RR > 24 or SBP < 90 with k = 2, w = 60). A crossing then counts only when k
  CONSECUTIVE readings beyond the threshold (no non-qualifying reading between them), all
  after the anchor, span at most w minutes; the event time is the reading that completes the
  run, which must fall inside the horizon. An isolated crossing never fires. A crossing that
  is not confirmed (the last reading inside the horizon is beyond the threshold but no run
  completed) is unknown, not event-free: if the stay ends first it is `competing_event` or
  `censored` as usual; otherwise `not_ascertainable` (never `negative`). Labels are still
  computed on the whole stay in availability order; prevalence is unchanged. With the key
  absent the labels are the single-crossing labels, unchanged. Implemented in
  `src/data/threshold_grid.py` (`_sustained`, `GridQuery.sustained`) and
  `src/data/targets.py` ("SUSTAINED CROSSING", `_sustained_completion`); tests in
  `tests/test_targets_gem_tte.py::SustainedCrossingTest`. Status: implemented, off.
- **Per-cause event rates.** `targets.anchor_status_shares` now reports
  `competing_risk_by_cause`: per cause index, the events and the share of supervised anchors
  whose competing-risk event is that cause, so the real-data smoke
  (`src/train/real_data_smoke.py`) can compare RR > 24 vs SpO2 < 88 vs lactate > 4 with and
  without the sustained rule. Status: implemented.

## B. Off-edge control thresholds (item 8)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 8 | Controls landing on bin edges | Creatinine control 1.3 -> **1.35** mg/dL and MAP control 63 -> **62.5** mmHg (bin interiors). Re-verify against the production vocabulary's edge-distance table before any claim-bearing run; the `run_matrix --edge-check` refusal of an on-edge control stays. Also fixed: `--edge-check` crashed (TypeError) when no `--vocab` matched a matrix arm; it now refuses an unknown arm name and reports "not run" when no vocabulary exists. | 1.3 and 63 | none (tokenization geometry) | `configs/thresholds.yaml` `control`; `src/train/run_matrix.py`; `tests/test_experiment_matrix.py::EdgeCheckCommandTest` | provisional |

## C-D. GCS, sedation and blood pressure (items 9-13)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 9 | GCS verbal untestable | Separate "not testable / T" verbal token plus eye and motor | verbal charted as scored | Reith 2016 doi:10.1089/neu.2014.3843; Menon 2025 doi:10.1089/neu.2024.0577; Zhang 2025 doi:10.1111/jocn.17729 | `src/data/tokenize.py`, `configs/data.yaml` | implemented (parallel) |
| 10 | GCS total when verbal untestable | Keep the total, imputed by Brennan from eye + motor (EM 2-6 +1; 7 +2; 8-9 +4; 10 +5) | drop the total | Brennan 2021 doi:10.3171/2020.6.JNS20992; Zhang 2025 doi:10.1111/jocn.17729 | `src/data/tokenize.py`, `configs/data.yaml` | implemented (parallel) |
| 11 | RASS grouping | One token per RASS level; if binary, light (>= -2) vs deep (<= -3) | same | Devlin 2018 doi:10.1097/CCM.0000000000003299; Rakhit 2022 doi:10.1097/SLA.0000000000004484 | `configs/literature_segments/labs.yaml` | implemented (parallel) |
| 12 | MAP source for outcomes | All readings primary; arterial-line-only as sensitivity arm | same (kept) | none | `configs/cohort.yaml`, `configs/data.yaml` | implemented (parallel) |
| 13 | Simultaneous cuff and arterial | Keep both; either crossing fires the event | same | none | `configs/data.yaml` | implemented (parallel) |

## E. Labs - literature-proposed bins (items 14-19)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 14 | (question text not in the decision document) | Not addressed in the answer; the proposed literature bins stand | literature bins | none | `configs/literature_segments/labs.yaml` | implemented (parallel; default) |
| 15 | Troponin T edges | 14 ng/L is the 4th-generation edge; treat MIMIC conventional-assay troponin as a separate concept with data-driven edges; a single hs edge set would be 5 / 15 (or 19) / 52 / 140 | 14 ng/L edge set | Fitzgerald 2020 doi:10.1016/j.cca.2020.01.027; McEvoy 2023 doi:10.1016/j.jacc.2023.03.403; Gore 2014 doi:10.1016/j.jacc.2013.12.032; Koechlin 2026 doi:10.1016/j.jacc.2025.12.052 | `configs/literature_segments/labs.yaml`, `src/data/tokenize.py` | implemented (parallel) |
| 16 | SvO2 / ScvO2 edges | Keep the edges, tag the source; if feasible ScvO2 65/70/80 vs SvO2 60/65/75; keep the 80% upper edge | one edge set | Singh 2020 doi:10.1186/s13054-020-03326-2; Balzer 2015 doi:10.1186/s13054-015-0889-6; Pope 2010 doi:10.1016/j.annemergmed.2009.08.014 | `configs/literature_segments/labs.yaml` | implemented (parallel) |
| 17 | (question text not in the decision document) | Not addressed in the answer; default stands | literature bins | none | `configs/literature_segments/labs.yaml` | implemented (parallel; default) |
| 18 | Eosinophil edge | Dual representation: 2% edge plus absolute 100 / 300 cells/uL | 2% only | Bafadhel 2012 doi:10.1164/rccm.201108-1553OC; Singh 2022 doi:10.1164/rccm.202201-0209PP; GOLD 2026 | `configs/literature_segments/labs.yaml` | implemented (parallel) |
| 19 | (question text not in the decision document) | Not addressed in the answer; default stands | literature bins | none | `configs/literature_segments/labs.yaml` | implemented (parallel; default) |

## F. Medications and fluids (items 20-26)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 20 | Heparin edges | Keep 500 / 1000 / 1500 U/h | same | Holbrook 2012 doi:10.1378/chest.11-2295; Garcia 2012 doi:10.1378/chest.11-2291 | `configs/literature_segments/medications.yaml` | implemented (parallel) |
| 21 | Insulin edges | Keep 3.5 / 7 / 10 U/h | same | Pasquel 2021 doi:10.1016/S2213-8587(20)30381-8 | `configs/literature_segments/medications.yaml` | implemented (parallel) |
| 22 | Normal saline edges | Keep KVO / maintenance / resuscitation / bolus tiers | same | none | `configs/literature_segments/medications.yaml` | implemented (parallel) |
| 23 | Bolus edges | Keep (fentanyl 25-100 ug, hydromorphone 0.2-2 mg, morphine 2-10 mg, midazolam 0.5-5 mg, lorazepam 0.5-4 mg, propofol 10-20 mg) | same | FDA labels; Puntillo 2021 doi:10.1007/s40122-021-00286-5 | `configs/literature_segments/medications.yaml` | implemented (parallel) |
| 24 | Furosemide edges | Keep 40 / 80 / 150 mg bolus and 10 / 20 / 40 mg/h | same | none | `configs/literature_segments/medications.yaml` | implemented (parallel) |
| 25 | Ketamine 0.1-0.3 "mg" | Read as mg/kg; convert with charted weight; quarantine "mcg" rows for review | as charted | Schwenk 2018 doi:10.1097/AAP.0000000000000806; Morgan 2021 doi:10.1080/10903127.2020.1801920; Beaudrie-Nunn 2023 doi:10.1016/j.ajem.2023.05.026 | `src/data/tokenize.py`, `configs/literature_segments/medications.yaml` | implemented (parallel) |
| 26 | Dose plausibility floors | Flag below fentanyl 0.025 mg, hydromorphone 0.2 mg, morphine 1-2 mg, midazolam 0.5 mg, lorazepam 0.5 mg, propofol 10 mg, ketamine 0.1 mg/kg | kept silently | FDA labels; Schwenk 2018 doi:10.1097/AAP.0000000000000806 | `src/data/tokenize.py`, `configs/literature_segments/medications.yaml` | implemented (parallel) |

## G. CRRT, ECMO and mechanical circulatory support (items 27-30)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 27 | `ultrafiltration_out` meaning | Net ultrafiltration (patient fluid removal), labelled as such | total effluent | Neri 2016 doi:10.1186/s13054-016-1489-9; Murugan 2021 doi:10.1038/s41581-020-00358-3 | `configs/literature_segments/support.yaml`, `configs/data.yaml` | implemented (parallel) |
| 28 | Effluent-dose concept | Add 20 / 25 / 35 mL/kg/h (plus a < 13 flag) only if true effluent can be reconstructed | none | Neyra 2021 doi:10.1111/sdi.12974; RENAL 2009 doi:10.1056/NEJMoa0902413; Wang 2018 doi:10.1093/ndt/gfx308; Lumlertgul 2026 doi:10.1186/s13054-026-05987-x; Okamoto 2024 doi:10.1053/j.ajkd.2024.01.526 | `configs/literature_segments/support.yaml` | deferred (needs effluent reconstruction) |
| 29 | ECMO edges | Keep sweep 0/1/2/9, flow 2/4/5/6 L/min, FdO2 0.22 / 1.0 (check 0.22 is not a placeholder) | same | none | `configs/literature_segments/support.yaml` | implemented (parallel) |
| 30 | RVAD flow edges | Data-driven, not LVAD cut-points | 2 / 4 L/min | none | `configs/literature_segments/support.yaml` | implemented (parallel) |

## H. ICU locations and code status (items 31-33)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 31 | `cvicu_icu` | Map to a CLIF cardiac/cardiothoracic ICU type; keep the original label as an attribute | orphaned | none | `configs/data.yaml`, `src/data/extubation_cohort.py` | implemented (parallel) |
| 32 | Home, Chemical Dependency, Shelter, Jail | Fold into the home / other-facility discharge groups; keep the granular code | unmapped | none | `configs/data.yaml` | implemented (parallel) |
| 33 | Comfort-care proxy | Deaths within 24 h without reintubation excluded only in a sensitivity arm | same (kept) | none | `configs/extubation.yaml` | implemented (parallel) |

## I. Extubation study design (items 34-41)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 34 | Grace window | 3 h, with 1 / 2 / 6 h sensitivity; outcomes inside the window handled by clone-censor-weight | same (kept) | Hernandez 2016 doi:10.1001/jama.2016.2711, doi:10.1001/jama.2016.14194; Tran 2026 doi:10.1016/j.jclinepi.2026.112205; Reep 2025 doi:10.1186/s13054-025-05723-x | `configs/extubation.yaml`, `src/eval/causal/estimators.py` | implemented (parallel) |
| 35 | Arm assignment | First device primary; highest support as sensitivity | same (kept) | Liu 2021 doi:10.1186/s12890-021-01526-2; Ren 2026 doi:10.1001/jamanetworkopen.2025.58262 | `configs/extubation.yaml` | implemented (parallel) |
| 36 | Face tents / aerosol masks | Conventional oxygen ("Face Mask") | open | Maggiore 2018 doi:10.1016/S2213-2600(18)30375-8; Yasuda 2021 doi:10.1186/s13054-021-03550-4 | `configs/extubation.yaml` | implemented (parallel) |
| 37 | Post-extubation row with no device | Room air, with a sensitivity analysis excluding no-device rows | dropped | none | `configs/extubation.yaml`, `src/data/extubation_cohort.py` | implemented (parallel) |
| 38 | Single NIV row then IMV | Require >= 2 non-invasive rows (primary); single-row rule as sensitivity | single row | Miltiades 2017 doi:10.1097/CCM.0000000000002327; Tanaka 2023 doi:10.1186/s13054-023-04668-3 | `configs/extubation.yaml`, `src/data/extubation_cohort.py` | implemented (parallel) |
| 39 | Discharged alive before day 7 | Primary `event_free` (emulates the trials' ITT counting); sensitivity with discharge alive as a **competing event** (clone-weighted Aalen-Johansen cumulative incidence) and `censor` | `event_free` proposed, `censor` sensitivity | Hernandez 2016 doi:10.1001/jama.2016.2711; Ionescu 2021 doi:10.1177/08850666211020281; Renard Triche 2025 doi:10.1186/s13054-025-05593-3; Miltiades 2017 doi:10.1097/CCM.0000000000002327 | `configs/extubation_benchmarks.yaml` (`discharge_alive_rule_sensitivity`), `src/eval/causal/emulate.py`, `src/eval/causal/benchmark.py`; `tests/test_causal_emulate.py::test_discharge_alive_as_a_competing_event_is_a_sensitivity_analysis` | implemented |
| 40 | Comorbidity source | Prior-admission codes primary (leakage-safe); index-stay codes as a labelled leakage sensitivity | open | none | `configs/extubation.yaml`, `src/data/extubation_cohort.py` | implemented (parallel) |
| 41 | Hospice then death | Competing event (primary); composite (counted as the endpoint) as sensitivity | competing only | RENOVATE 2025 doi:10.1001/jama.2024.26244; Ionescu 2021 doi:10.1177/08850666211020281 | labeler (parallel); `configs/extubation_benchmarks.yaml` (`hospice_rule`), `src/eval/causal/emulate.py`; `tests/test_causal_emulate.py::test_hospice_then_death_competes_and_the_composite_is_the_sensitivity` | implemented |

## J. Trial matching and stop rules (items 42-45)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 42 | Agreement margins | Per-trial margin by the FDA fixed-margin approach: M1 = the benchmark interval bound nearest no effect; the margin preserves 50% of M1, stated on the relative (log risk ratio, the decision scale) and absolute (risk difference) scales; for mortality-type endpoints (reintubation or death, death) the relative margin is capped at 1.2 (range 1.1-1.2 enforced). Null trials keep the equivalence margin 1.5 (no effect to preserve), capped the same way. Pooled rule: mean of |gap| / own log margin <= 1. Plus the **Roehmel-Kieser second hurdle**: a trial counts as reproduced only if the point estimate is on the benchmark's side of no effect (direction-consistent); the pooled pass needs every scored effect trial to clear it. | one common margin 1.5 | Pong 2021 doi:10.1136/bmjopen-2020-044480 (median relative margin 1.5, IQR 1.3-1.7); Kleber 2025 doi:10.1093/jnci/djae318; Mauri 2017 doi:10.1056/NEJMra1510063; Piaggio 2012 doi:10.1001/jama.2012.87802; Roehmel & Kieser 2013 doi:10.1002/sim.5563; Freund 2025 doi:10.1001/jama.2024.25869 | `configs/extubation_benchmarks.yaml` (`margin_derivation`, `second_hurdle`), `src/eval/causal/benchmark.py` (`derive_margin`, `trial_agreement`, `score_agreement`), `src/eval/causal/simulation.py`, `src/eval/extubation_audit.py`; tests in `tests/test_causal_benchmark.py`, `tests/test_causal_simulation.py` | implemented (values proposed) |
| 43 | Harmful-side stop | Fires when the ENTIRE risk-ratio interval lies above 1.0 (lower bound > 1.0); not a point estimate above 1.0 or 1.5. The optional posterior P(RR > harm threshold) is not implemented. | lower bound > 1.0 vs the equivalence margin (open) | Lewis 2016 doi:10.1001/jama.2016.16070; Pocock 2015 doi:10.1016/j.jacc.2015.10.051 | `configs/extubation_audit.yaml`, `src/eval/extubation_audit.py::harmful_side_rule`; `tests/test_extubation_audit.py` | implemented |
| 44 | Overlap / positivity screen | Measured on the two compared arms only: P(treated given one of the two arms) cross-fitted among those patients; the share outside [0.02, 0.98] is the screen; the share of eligible patients a trim would exclude is reported (of compared and of all trial-eligible patients) | all eligible patients, one-vs-rest | Duong 2024 doi:10.1371/journal.pone.0314761; Simoneau 2022 doi:10.1177/13524585221085733; Zhu 2021 doi:10.1002/pds.5338 | `src/eval/causal/emulate.py::measure_feasibility`, `src/eval/causal/benchmark.py` (`FeasibilityInputs`); `tests/test_causal_emulate.py::test_overlap_is_measured_on_the_two_compared_arms_and_trimming_is_reported` | implemented |
| 45 | Negative and positive controls | Eight proposed negative-control outcomes (new thrombocytopenia, hyperbilirubinemia, hypoglycemia, severe anemia, hyponatremia, hyperkalemia, coagulopathy, hypoalbuminemia; CLIF 2.1 labs, ascertained in 48 h before any reintubation), each with a rationale; an NCO fails when its interval excludes no effect, and failures feed the negative-control stop rule. Positive control: a planted risk ratio of 0.5 must be detected (interval excluding 1) in >= 80% of simulation replicates. | two descriptive candidates, rule not wired | Mac Grory 2026 doi:10.1161/CIR.0000000000001440; Wang 2026 doi:10.1016/j.jclinepi.2026.112188; Levintow 2023 doi:10.1002/pds.5623; Fan 2026 doi:10.1038/s41467-026-74999-6 | `configs/extubation_audit.yaml` (`negative_control_outcomes`, `positive_control`), `src/eval/extubation_audit.py` (`negative_control_rule`, `positive_control_check`), `src/eval/causal/simulation.py::positive_control_detection`; `docs/protocols/extubation-emulation-protocol.md` section 12 | implemented (register proposed; NCO labels not built) |

## K. Held-out share and data timing (items 46-47)

| # | Question | Decision | Default replaced | Key references | Implemented in | Status |
|---|---|---|---|---|---|---|
| 46 | Held-out NIV / HFNC arms too small | **Chosen:** a patient-level stratified split that over-samples patients with an NIV or HFNC first post-extubation device into the held-out partitions (all but `train`) until each arm holds 50% of its patients there, keeping the overall 60/15/10/15 by moving as many other patients the other way, deterministic by seed. The stratum comes only from pre-registered, outcome-blind cohort membership (`patient_id`, `eligible`, `arm_first_device`). Rush is external validation, not the sole held-out source. This changes the training-data composition and is baked into every checkpoint, so it is switched on at the split-freeze step, not before. | 60/15/10/15 unstratified | none (partitioning strategy) | `configs/train.yaml` `data_contract.held_out_stratification` (`enabled: false` until the freeze), `src/data/splits.py`, `src/data/cohort.py` (`--held-out-strata`), `src/train/preflight.py` (freeze records and checks the option), `docs/plans/l40-runbook.md` step 7; tests in `tests/test_splits.py`, `tests/test_preflight.py`, `tests/test_artifact_policy.py` | implemented (activation at the split freeze) |
| 47 | Availability lag for charted-only tables | 30-minute lag primary, 15 and 60 minutes as sensitivity analyses | none | none (operational safeguard for hard rule 4) | `configs/data.yaml`, `src/data/tokenize.py`, `src/data/extubation_cohort.py` | implemented (parallel) |

### Trial-agreement margins: FROZEN, optional, not on the critical path (2026-10-03)

The project goal was re-confirmed: the trial-agreement margin rubric is **frozen as
optional**. The registry, code and tests in the tree keep the item-42 per-trial fixed
margins with the Roehmel-Kieser second hurdle as committed (values `proposed`); nothing
about margins gates a training run or Paper 1's claims.

**Approved method if the rubric is revived** (product authority, 2026-10-03; supersedes the
per-trial 50% margins): (1) pool only exchangeable trials in exposure-outcome FAMILIES
(NIV vs HFNC, reintubation <= 7 d: HIGH-WEAN [approximate] + Hernandez 2022; HFNC vs
conventional oxygen, low risk, 72 h: Hernandez 2016 low risk; Casey 2021 kept out as a
different intervention/population and null; Ferrer not evaluable; null and
non-inferiority trials keep the equivalence margin and are never pooled); (2) per family,
an inverse-variance fixed-effect pooled log RR from the published arm counts, with
DerSimonian-Laird and I^2 as a check, registering the more conservative; margin = 50% of
the pooled 95% bound nearest the null (FDA fixed-margin method: Mauri & D'Agostino 2017,
doi:10.1056/NEJMra1510063; Althunian 2017, doi:10.1186/s13063-017-1859-x); (3) operative
margin = the smaller of that and an SCID of RR 1.20 / 5 percentage points (proposed, ICU
physician sign-off), on both scales (scale caveat: Quartagno 2020,
doi:10.1186/s13063-020-4070-4); (4) a feasibility floor: a trial whose operative margin is
narrower than the emulation's outcome-blind estimate resolution is an
`uninformative_benchmark`, reported separately and never scored as an emulation failure;
the second-hurdle direction check is unchanged. All three references verified on PubMed
(PMIDs 28270184, 32029000; Mauri 2017 via the attachment's list). The work-in-progress
implementation (registry families, `pool_family`, SCID cap, `estimate_resolution`, the
uninformative classification and its tests) is saved as
`output/patches/pooled-margins-wip.patch` and is NOT applied. Computed on the published
counts before it was set aside: NIV vs HFNC pooled (fixed effect; I^2 = 0, so DL is
identical) margin 1.096 / 1.4 percentage points; HFNC vs oxygen low risk 1.152 / 1.3 pp;
Ferrer pooled 1.270, operative 1.20 / 5 pp after the SCID cap. On the synthetic fixture's
size the HFNC-vs-oxygen and NIV-vs-HFNC trials would be classified uninformative.

## Pre-training configuration decisions (2026-10-03)

| Decision | Before | After | Why | Implemented in | Status |
|---|---|---|---|---|---|
| Admission header in continuation windows | A stay longer than one window is cut into contiguous windows; only the first starts with `<bos>`, `ADMISSION//<type>` and the static tokens (age decile, sex, race, ethnicity, admission type). Verified on the synthetic long stay: the second window has none of them. | Optional `continuation_header`: every continuation window gets the stay's header at its start (own positions, masked as targets; labels and anchors' labels unchanged, anchor offsets shift). Counted against `max_tokens`: an over-budget window is refused. | A later window otherwise has no admission context. | `src/data/dataset.py` (`header_token_ids`, `stay_header_length`, `_prepend_header`, `sample_lengths`, `gem_window_bounds_with_header`), `src/train/pretrain.py`, `configs/model.yaml` `trunk.continuation_header`; `tests/test_gem_training_path.py` | implemented, **off** until the tokenizer cuts continuation windows at `max_tokens - header_len` (call `dataset.gem_window_bounds_with_header` in place of `gem_window_bounds` in `src/data/tokenize.py::_gem_records`, then rebuild the shards and switch `continuation_header: true`) |
| RoPE base for minute positions | `rope_base: 10000` with positions in minutes since admission: slowest wavelength 2*pi*10000^(31/32) ~ 47,000 min (~33 days), shorter than long stays (66+ days, ~100,000 min) | `rope_base: 100000`: slowest wavelength ~ 438,000 min (~305 days, ~4x the longest stay); the fastest dimension still turns 1 rad/min, so short-range resolution is unchanged; per-dimension frequency ratio 1.33 -> 1.43. 5e5 was not chosen: it puts the slowest wavelength at ~4 years, beyond any stay. | Distinct angles for every minute of the longest stay in the slowest dimension. Formula checked against the RoPE definition used by Hugging Face Transformers (`inv_freq = 1 / base^(2i/d)`, computed in float32; Context7, /huggingface/transformers). Changes the model: applies to every run trained after it. | `configs/model.yaml`; `tests/test_rope_base.py` | implemented |
| Context-length report | none | `src/eval/context_length.py` (CLI): per shard, share of candidate anchors and ICU+24 h anchors over 4,096 / 8,192 / 16,384 tokens of history, and of the extubation time-zero prompts (aggregate; small cells suppressed). Pre-flight `context:` check per site (warns above 10% of prompts over 8,192; `--extubation-cohort`). `src.eval.threshold_eval --max-context` scores a trained model at 4K vs 8K without retraining. | Size the context decision from the data. | `src/eval/context_length.py`, `src/train/preflight.py`, `src/eval/threshold_eval.py`, runbook step 15; `tests/test_context_length.py` | implemented |

## Simulation operating characteristics after items 42-45

The planted-effect simulation (`src/eval/causal/simulation.py`) on the synthetic fixture of
`tests/test_causal_simulation.py` (1,500 synthetic covariate rows resampled to 3,000 per
replicate; logistic nuisance models; supplied baseline risk 0.25; 100 replicates; seed 3) gave
these single-trial pass rates without a withheld confounder. Synthetic data only.

| Trial | Finding | Derived margin (relative / absolute) | Trial effect: before -> after | Zero: before -> after | Reversed: before -> after | Positive control detection |
|---|---|---|---|---|---|---|
| Hernandez 2016 low risk | effect | 1.152 / 0.0125 | 1.00 -> 0.71 | 0.00 -> 0.00 | 0.00 -> 0.00 | 1.00 |
| Hernandez 2016 high risk | null | 1.500 / 0.0955 | 0.58 -> 0.45 | 0.83 -> 0.41 | 0.41 -> 0.11 | 0.99 |
| HIGH-WEAN 2019 | effect | 1.029 / 0.0043 | 1.00 -> 0.14 | 0.44 -> 0.00 | 0.00 -> 0.00 | 0.95 |
| Hernandez 2022 very high risk | effect | 1.039 / 0.0142 | 0.99 -> 0.17 | 0.17 -> 0.00 | 0.00 -> 0.00 | 0.95 |
| Ferrer 2009 hypercapnic | effect | 1.270 / 0.0835 | 0.97 -> 0.79 | 0.00 -> 0.00 | 0.00 -> 0.00 | 0.96 |
| Casey 2021 all-comers | null | 1.500 / 0.0663 | 0.80 -> 0.78 | 0.94 -> 0.53 | 0.60 -> 0.02 | 1.00 |

Reading it: the second hurdle removes most reversed-effect passes (Casey 0.60 -> 0.02; the
earlier run quoted 57%). For the null trials a zero effect still passes about half the time,
because the hurdle asks the estimate to sit on the side of 1 where the trial's own (null)
point estimate fell. The fixed margins of HIGH-WEAN (1.029) and Hernandez 2022 (1.039) are so
narrow that even the true trial effect passes only 14-17% of the time at this sample size:
those trials are in practice unreproducible under the 50%-preserved fixed margin. With the
withheld confounder no scenario passes more than 1% of the time. Both points go to the
product authority before the values are registered.

## Open items for the product authority (raised by this implementation)

The margin rubric is frozen (above); these remain on record if it is revived. Margins are unchanged; these are reported for decision before the values are registered.

1. **The 50% fixed margin makes two effect trials nearly unreproducible.** HIGH-WEAN 2019
   (relative margin 1.029) and Hernandez 2022 (1.039) have benchmark intervals whose bound
   nearest 1 is close to 1, so half of M1 is a margin of 3-4%. On the synthetic fixture the
   rule passed the TRUE trial effect in only 14-17% of replicates. Options: a smaller
   preserved fraction for agreement (the FDA fraction was written for non-inferiority, not
   for agreement with a benchmark), a floor on the margin, or scoring these trials on the
   absolute scale.
2. **The second hurdle on null trials compares against the trial's noisy point estimate.**
   For a null benchmark (Casey 2021, Hernandez 2016 high risk) "direction-consistent" means
   the side of 1 on which the trial's own non-significant point estimate fell. It removes
   reversed-effect passes (Casey 0.60 -> 0.02) but still passes a zero effect about half the
   time, and could fail a correct emulation of a truly null effect for the same reason.
   Options: apply the hurdle to effect trials only, or define harm for null trials as an
   estimate beyond the equivalence region.
