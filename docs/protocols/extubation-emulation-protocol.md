# Target trial emulation of post-extubation respiratory support: protocol (registration draft)

**Status: DRAFT, not registered.** Every margin, threshold and negative control below is
**proposed**. Nothing here becomes binding until the product authority (J.C. Rojas)
registers the protocol and its sha256 hash is recorded. Until then, the only outcome-by-arm
comparison that may run is the single pre-registration audit comparison described in
section 11 (plan R7, R32, KTD9).

- Plan: `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md` (R7, R8, R9, R10, R12,
  R13, R14, R25, R28, R29, R32, R33, R40; KTD6, KTD8, KTD9, KTD10).
- Evidence: `notes/extubation-evidence-review.md`.
- Machine-readable sources this draft describes, which win if they disagree with the prose:
  `configs/extubation.yaml` (cohort, arms, grace window), `configs/extubation_benchmarks.yaml`
  (trials, estimand parameters, feasibility screen, agreement rule),
  `configs/extubation_audit.yaml` (stop rules, adoption-era trigger).
- Code: `src/data/extubation_cohort.py`, `src/eval/extubation_labeler.py`,
  `src/eval/causal/` (estimators, diagnostics, benchmark, emulate, simulation),
  `src/eval/extubation_audit.py`.

## 1. What this study claims, and what it does not

The study asks whether a classical emulation, and later the foundation model with the
post-extubation device injected at the real extubation, reproduces the pattern that
randomized trials established for high-flow nasal cannula (HFNC) and noninvasive
ventilation (NIV) after extubation.

The claim is **agreement with the trials' pattern** within pre-specified margins. An
emulation that agrees is consistent with the trial; it is not proof that its estimates are
causal, and it is not a device recommendation for any patient. A disagreement is reported
as a disagreement. MIMIC-IV-Ext-CLIF results are exploratory; Rush is the confirmatory site
and is analysed only after the freeze in section 10 (R33, R28).

## 2. Eligibility

**Cohort (every trial).** The patient's first extubation across all hospitalizations:
invasive ventilation for at least 24 hours (12 hours in a sensitivity cohort that reuses the
same extubation), followed by a non-invasive device row. Time zero is the availability time
of that device row; every covariate uses only rows available strictly before time zero.
Excluded, with the first applicable reason recorded: missing or under-18 age; too little
look-back or look-forward to locate the extubation; tracheostomy at or before time zero;
comfort-care code status at time zero; a documented do-not-reintubate status at time zero.
A missing code status does not exclude (about 43% of MIMIC extubations); it is carried as
`code_status_missing`. Details: `configs/extubation.yaml`.

**Per trial.** Each benchmark trial's eligibility is applied on top of the cohort as rules
over the cohort's risk-factor columns (`configs/extubation_benchmarks.yaml`). A risk factor
that is NULL (no PaCO2 before time zero, no usable diagnosis history) counts as absent
(proposed). Criteria with no CLIF proxy (APACHE II, secretions, cough, airway patency,
difficult weaning) are listed per trial as not representable; the proxies used are stated
per trial. Comorbidities come only from earlier hospitalizations, because MIMIC diagnosis
codes are recorded after discharge and carry no present-on-admission flag.

## 3. Treatment strategies, assignment and grace period

- **Arms:** conventional oxygen (nasal cannula, face mask, room air), HFNC, NIV (BiPAP or
  CPAP), in that escalation order.
- **Assignment (primary):** the first arm device row inside the grace window, which is the
  device at the actual extubation (session-settled). Sensitivity: the highest-ranked device
  in the window. NIV alternating with HFNC in the window is the NIV arm (HIGH-WEAN).
- **Grace window:** 3 hours after time zero, right-closed (proposed; sensitivity 1, 2 and
  6 hours). Device rows are a median 180 minutes apart on MIMIC, so 3 hours is the shortest
  window that holds one charting step; longer windows let rescue support count as
  prophylaxis.
- The device-choice model is a nuisance of the estimator. It is never a target of the
  foundation model (hard rule 1).

## 4. Causal estimand

- **Strategy contrast:** start device A at extubation versus start device B, in the
  trial-eligible population at the site (average treatment effect over that population; a
  two-arm contrast keeps patients on the third arm in the target population).
- **Rescue:** support ranked above the assigned arm and started after the grace window is
  rescue. It is part of the strategy as practised (a treatment-policy view), is never the
  assigned arm and is never a covariate. Rescue NIV or HFNC is reported descriptively.
- **Deviation and censoring:** clone-censor-weight. Each patient is cloned into every
  compared arm at time zero; a clone is censored when the first device row inside the grace
  window shows another arm; an event before the first device row counts in every clone.
  Inverse-probability-of-censoring weights come from a cross-fitted device-choice model.
- **Follow-up:** from time zero to the trial's own outcome window (and 168 hours for the
  study primary outcome), to death, or to the index discharge. Readmissions are not
  searched.
- **Discharge alive before the horizon (item 39, decided):** primary rule `event_free`: a
  patient discharged alive with no event inside the window is event-free at the horizon,
  which emulates the trials' intention-to-treat counting and assumes no out-of-hospital
  reintubation or death. Sensitivity analyses: discharge alive as a **competing event**
  (`competing`; clone-weighted Aalen-Johansen cumulative incidence with discharge alive
  competing), and `censor` at discharge (clone-weighted Aalen-Johansen). Rows unresolved under the rule in force (unknown
  disposition) are counted and reported; above 5% (proposed) the emulation refuses rather
  than estimate on the remainder.
- **Competing death:** death prevents later reintubation. Each emulation reports the
  composite (reintubation or death) and the cause-specific cumulative incidence of
  reintubation with death as a competing event; a hospice discharge is a competing event at
  discharge (item 41, decided), with a composite sensitivity (`hospice_rule: composite`)
  that counts hospice-then-death as the endpoint.

## 5. Outcomes

- **Per trial:** that trial's primary reintubation outcome and window (72, 96 or 168 hours;
  table in section 8).
- **Study primary outcome:** reintubation or death within 168 hours.
- Reintubation is the first invasive-ventilation row of the index hospitalization charted
  strictly after time zero; a tracheostomy with no invasive row is not a reintubation. It is
  a label-only study endpoint under the scoped amendment to hard rule 1.

## 6. Estimators

- **Primary:** one-step clone-censor-weight, doubly robust (cross-fitted device-choice and
  outcome models; histogram gradient boosting), with influence-function intervals summed
  within patient.
- **Sensitivity:** point-treatment augmented inverse-probability weighting on the first
  device; overlap weights; the `censor` discharge rule; grace windows of 1, 2 and 6 hours;
  the highest-support assignment; the 12-hour ventilation cohort.
- **Adjustment set (proposed):** age, invasive-ventilation hours, PaCO2, hypercapnia, BMI,
  COPD, chronic respiratory disease, heart failure, chronic cardiac disease, whether
  diagnosis history is available, and missing code status. Unit and calendar period are
  left out on purpose: they shift device choice without a direct path to the outcome, and
  adjusting for them can amplify bias.
- **Per site and pooled** (R14): each development site separately and pooled with site
  indicators.
- The model-based estimators that follow the foundation model use only patients outside its
  pretraining partition (KTD6); the split is frozen by hash before the first L40 run and is
  sized from the audit's blind-stage arm counts.

## 7. Diagnostic gates and the feasibility screen

**Feasibility (R25), read without any outcome by arm.** A trial is evaluable only if its
eligibility, exposure and outcome are representable; every estimator can run; each compared
arm has at least 50 patients; overlap is measured on the **two compared arms only** (item 44,
decided): at most 20% of the patients on the treated or the control arm have a cross-fitted
probability of the treated arm (given one of the two) outside [0.02, 0.98], and the share of
all trial-eligible patients a trim to that range would exclude is reported; each arm's Kish effective sample size is
at least 30; and the projected precision is adequate: for a trial that found an effect, the
smallest detectable risk difference (from the effective sample sizes and the trial's
published control-arm risk) is at most 1.5 times the trial's difference; for a null trial,
the projected half-width of the log risk-ratio interval fits inside the equivalence margin.
All values are proposed. A trial that fails is reported as not evaluable and left out of
the agreement results of every estimator.

**Diagnostic gates (R9), before any effect is read.** Overlap and positivity per arm;
covariate balance after weighting (maximum absolute standardized mean difference, threshold
to be registered); effective sample size; the leakage probe on the model state (model-based
estimators only); and sensitivity to grace-period length and exposure definition. Each
estimate is reported with its E-value and the expected direction of residual confounding.

## 8. Benchmark trials

Benchmark effects are risk ratios (treated over control) computed from the arm counts each
trial published (Katz log interval). The effect as published is kept beside it in the
registry. "Expected bias" is the direction confounding by indication would push the
emulation's risk ratio.

| Trial (registry id) | Contrast (treated vs control) | Emulated outcome | Finding | Benchmark RR (95% CI) | Representability (elig. / exposure / outcome) | Expected bias |
|---|---|---|---|---|---|---|
| Hernández 2016, low risk (`hernandez_2016_low_risk`) | HFNC vs conventional oxygen | reintubation, 72 h | effect | 0.41 (0.22–0.75) | proxied / representable / representable | upward |
| Hernández 2016, high risk (`hernandez_2016_high_risk`) | HFNC vs NIV | reintubation, 72 h | null (non-inferiority) | 1.19 (0.87–1.63) | proxied / representable / representable | downward |
| HIGH-WEAN, Thille 2019 (`high_wean_2019`) | any NIV vs HFNC (highest support; approximate) | reintubation, 7 d | effect | 0.65 (0.45–0.94) | proxied / proxied / representable | upward |
| Hernández 2022, very high risk (`hernandez_2022_very_high_risk`) | NIV vs HFNC | reintubation, 7 d | effect | 0.59 (0.37–0.93) | proxied / representable / representable | upward |
| Ferrer 2009, hypercapnic (`ferrer_2009_hypercapnic`) | NIV vs conventional oxygen | reintubation, 72 h | effect | 0.31 (0.15–0.62), respiratory failure | proxied / representable / **not representable** | upward |
| Casey 2021, all-comers (`casey_2021_all_comers`) | HFNC vs conventional oxygen (protocolized support vs usual care; approximate) | reintubation, 96 h | null | 1.20 (0.85–1.69) | proxied / proxied / representable | upward |

Every emulation is approximate: no trial's eligibility can be fully built from CLIF data.
Ferrer 2009's published effect is for clinical respiratory failure, which the labels do not
build, so it fails the screen until a reintubation benchmark is registered; it still informs
the known-answer pattern (NIV for hypercapnic patients).

## 9. Agreement criteria (proposed)

- **Scale:** log risk ratio; gap = emulation minus trial.
- **Margins (item 42, method decided, values proposed):** each trial that found an effect
  gets its own margin by the FDA fixed-margin approach: M1 is the bound of the benchmark
  risk-ratio interval nearest 1, and the margin preserves 50% of it (relative margin
  exp(0.5 x |log M1|)); the same margin is stated on the absolute scale (half the Wald
  risk-difference bound nearest 0). For a mortality-type endpoint (reintubation or death,
  death) the relative margin is capped at 1.2 (allowed range 1.1-1.2). The derived margins
  are about 1.03-1.27 for the registered effect trials, against the retired common 1.5.
- **Trials that found an effect:** per estimator, each gap is divided by its trial's log
  margin and the mean absolute scaled gap must be at most 1. The inverse-variance signed
  mean gap is reported as a description of systematic bias.
- **Null trials:** reproduced only when the emulation's whole interval lies inside
  [1/1.5, 1.5] (equivalence margin 1.5, proposed; capped like the effect margins for a
  mortality-type endpoint), so a wide interval cannot pass by being wide. A null trial has
  no effect to preserve, so the fixed-margin approach does not apply to it.
- **Second hurdle (Roehmel & Kieser 2013):** a trial counts as reproduced only if, besides
  the interval rule, the emulation's point estimate lies on the benchmark's side of 1, and
  the pooled verdict needs every scored effect trial to clear it. A reversed effect can no
  longer pass by landing inside a wide margin.
- **Operating characteristics (R29):** a planted-effect simulation on each frozen cohort
  reports how often the rule passes when the true effect equals the trial's, is zero, or is
  reversed, with and without a withheld confounder; treatment is drawn from a fitted
  device-choice model and the baseline risk is supplied as a number or fitted without the
  arm. These pass rates are published with the results, with a **positive control**
  (item 45): a planted risk ratio of 0.5 must be detected (interval excluding 1) in at
  least 80% of replicates.
- **Known-answer pattern (R13):** the predicted advantage of NIV over HFNC grows with
  predicted baseline risk, and the lowest-risk device in trial-defined groups matches the
  trials (NIV for hypercapnic, obese and very-high-risk patients; HFNC over conventional
  oxygen for low-risk patients).

## 10. Freeze and sites (R28, R33, KTD9)

Before any confirmatory or external run, the model checkpoint, vocabulary, cohort
definition, estimator code, hyperparameters, thresholds and benchmark registry are frozen
and recorded by sha256. The cohort definition, estimator code and registry hashes are
recomputed on the node and must match. Nothing is tuned on returned aggregates. MIMIC is
exploratory; Rush is confirmatory and is refused by the code until the freeze hashes are
recorded; UChicago is external validation by model-to-data.

## 11. The pre-registration design audit (R32)

Before registration, a go/no-go audit runs on MIMIC with the classical estimator only
(`src/eval/extubation_audit.py`):

1. **Blind stage (no outcome read):** arm sizes per site, pooled, per data partition and for
   the held-out union; effective sample size; minimal detectable effect (baseline risk: each
   trial's published control-arm risk, or a supplied registered number; the report states
   which); covariate coverage; overlap and balance from a device-choice model; the
   feasibility screen per trial; and the share of first devices by calendar period and by
   unit. On MIMIC the period table is not evaluable unless the `anchor_year_group` table is
   staged, because MIMIC dates are shifted per patient and a period is never derived from
   event dates.
2. **Simulation stage:** the planted-effect simulation per evaluable trial.
3. **Unblinded stage:** one all-comer comparison of HFNC versus conventional oxygen against
   Casey 2021, on MIMIC only, exactly as registered in the benchmark registry. It is the only
   outcome-by-arm comparison allowed without a protocol hash.

**Stop rules (proposed; `configs/extubation_audit.yaml`).** The extubation application
stops or is re-scoped with the product authority if any of these fires:

- **Precision:** more than half of the registered trials fail the R25 screen at the site.
- **Harmful side (item 43, decided):** the lower bound of the unblinded all-comer risk ratio
  (HFNC over conventional oxygen) is above 1.0, so the whole interval says HFNC is worse;
  not a point estimate above 1.0 (too noisy) or 1.5 (too permissive).
- **Negative controls (item 45):** more than zero registered negative-control outcomes have
  a device risk-ratio interval that excludes 1 (section 12). These need outcome-by-arm runs
  and so are evaluated only after registration.

**Adoption-era trigger (R40, proposed).** If, for some arm, its share of first devices
differs by more than 0.20 between calendar periods (or units) with at least 50 patients
each, the report flags an adoption-era contrast as a supporting analysis that does not
depend on measured confounders, compared with Casey 2021 and reported as a bound. A shift
by unit may reflect case mix rather than practice and needs clinical review before the
contrast is read. The adoption-era estimator is specified at registration.

## 12. Negative controls (R12; all proposed)

No validated negative controls exist for HFNC, NIV or oxygen exposures, so each candidate
below states why a null is expected and how it could fail for reasons other than
confounding. Each is run with the primary estimator: it passes when the interval covers no
effect, and fails (and counts toward the stop rule) when it excludes it.

**Registered negative-control outcomes (item 45, proposed; `configs/extubation_audit.yaml`
`negative_control_outcomes`).** Eight new laboratory abnormalities, each a CLIF 2.1 lab
category in its CLIF 2.1 unit, absent in the 24 hours before time zero, ascertained in the
first 48 hours and only before any reintubation (so an effect of the device on
reintubation cannot carry into them): platelet count < 100 x10^3/uL, total bilirubin
> 2.0 mg/dL, serum glucose < 70 mg/dL, hemoglobin < 7.0 g/dL, sodium < 130 mmol/L,
potassium > 6.0 mmol/L, INR > 1.5 and albumin < 2.5 g/dL. Each shares confounding by
severity with the primary outcome and has no plausible path from the device. Left out on
purpose because a device path exists: facial pressure injury (NIV mask), aspiration fever
or new antibiotics, delirium, hypoxaemia and lactate, ICU readmission. Shared impurity: a
device can change how often labs are drawn, so lab-draw counts per arm are reported beside
each. Pairing them with the planted positive control (section 9) shows the pipeline can
detect a true effect.

The earlier candidates remain as supporting diagnostics:

| Candidate | Type | Reason to expect a null | Known impurity |
|---|---|---|---|
| Chronic-condition codes recorded on the index stay that cannot arise in hours (for example diabetes mellitus, chronic kidney disease) | negative-control outcome fixed before time zero | the condition predates the extubation, so the device cannot cause it; an association reflects case mix or coding intensity | coding intensity rises with length of stay, which the device can change (length-of-stay-dependent outcomes are impure) |
| A laboratory value fixed early after time zero by pre-existing physiology (for example the first serum creatinine within 6 hours) | negative-control outcome fixed early in the stay | too little time for the device to change it | measurement frequency may differ by arm and severity |
| Balance on pre-time-zero variables held out of adjustment (for example sex, admission type, number of earlier hospitalizations) | balance diagnostic | if the adjustment set captures the confounding, weighting should balance other pre-time-zero variables related to severity | a held-out variable unrelated to severity balances trivially and tests nothing |

Expected bias direction for each benchmark contrast is stated in section 8, and quantitative
bias analysis (E-values) is reported for every estimate. Rescue NIV and HFNC versus
face-mask oxygen in hypoxaemic patients are reported descriptively and are not gates. The
unadjusted contrast was already observed in MIMIC during feasibility (section 13), so it is
a prospective check only at Rush and UChicago.

## 13. Prior data access and disclosure of feasibility results (R7)

During feasibility, before this protocol was drafted, the project computed unadjusted MIMIC
outcome rates by device arm. They informed the expected bias directions in section 8. They
are disclosed at registration, from their release-ledger record, in the space below; this
draft deliberately does not reproduce them.

> **[PLACEHOLDER — to be completed at registration]** Unadjusted MIMIC outcome rates by
> device arm computed during feasibility: definition of arm and window, counts and rates as
> released (suppressed under section 14), and the date they were computed.

Also to be declared at registration: the result of the unblinded audit comparison
(section 11), which by then will have been seen.

## 14. Disclosure (R6, KTD10)

Everything that leaves a node is aggregate. Counts under 10 are suppressed on numerator and
denominator, and suppression holds under differencing: every audit and emulation release
declares which cells nest inside which (trial-eligible within all-comer, partition within
all, site within pooled, arm within unit or period) and which cells sum to a total, and the
export gate refuses any release in which a parent minus a child, or a total minus its
released parts, recovers a suppressed count. Every release, including counts written into
this document, enters the cumulative release ledger before it leaves the node. No
row-level data leaves Rush or UChicago.

## 15. Open items for the product authority

- Register or revise every `proposed` value: margins, the feasibility screen, the discharge
  rule, the maximum unresolved share, the balance threshold, the stop-rule and trigger
  thresholds, and the negative controls.
- Settled by the product authority's decisions of 2026-10-03 (`docs/decisions/2026-10-03-clinical-decisions.md`): the harmful-side bound (1.0, whole interval), the margin method, the
  second hurdle, the two-arm overlap screen, the negative-control register, the discharge
  rule (event-free primary, competing sensitivity) and hospice handling.
- ~~Whether "Face Mask" (which may include aerosol face tents) belongs in the conventional
  oxygen arm.~~ Decided (item 36): face tents and aerosol masks count as conventional
  oxygen ("Face Mask").
- Clinical review of the comorbidity code lists.
- Staging MIMIC-IV `anchor_year_group` so the period table and the adoption-era contrast
  are evaluable on MIMIC.
- A reintubation benchmark for Ferrer 2009, or leaving it not evaluable.
