# Post-extubation respiratory support: evidence review for the extubation counterfactual study

Reviewed 2026-10-03 (sections 9–11 added the same day after the stress test) for `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md`.
Sources were retrieved through PubMed, Paperclip (PMC, arXiv, medRxiv, abstracts) and Tavily.
Every effect size below comes from an abstract or full text that was actually retrieved; claims that were not retrieved are marked **[unverified]**.
MIMIC counts in section 7 are aggregate-only feasibility queries on the local MIMIC-IV-Ext-CLIF tables; no row-level data was read out, and every reported cell is at least 10.

---

## 1. Benchmark trials (emulation targets)

P = prophylactic (support from extubation). R = rescue (support after post-extubation respiratory failure has started). Effects are intervention vs comparator.

| Trial | Population and risk definition | Arms | Primary outcome (window) | Result |
|---|---|---|---|---|
| Hernández 2016a, JAMA (PMID 26975498) P | Low risk: age <65, APACHE II <12, BMI <30, adequate secretions, simple weaning, ≤1 comorbidity, no heart failure, no moderate-to-severe COPD, no airway problem, no prolonged ventilation | HFNC vs conventional O2, 24 h | Reintubation, 72 h | 4.9% vs 12.2% (difference 7.2%, 95% CI 2.5–12.2) |
| Hernández 2016b, JAMA (27706464) P | High risk, ≥1 of: age >65, APACHE II >12, BMI >30, inadequate secretions, difficult or prolonged weaning, >1 comorbidity, heart failure as ventilation indication, moderate-to-severe COPD, airway patency problem, prolonged ventilation | HFNC vs NIV, 24 h | Reintubation, 72 h | 22.8% vs 19.1%; HFNC non-inferior. NIV withdrawn for adverse effects in 42.9% |
| HIGH-WEAN, Thille 2019, JAMA (31577036) P | Age >65 or underlying cardiac or respiratory disease | NIV (≥12 h/day for 48 h) alternating with HFNC vs HFNC alone | Reintubation, day 7 | 11.8% vs 18.2% (difference −6.4%, −12.0 to −0.9) |
| Hernández 2022, ICM (36400984) P | Very high risk: ≥4 of the 2016b factors plus PaCO2 >45 at end of SBT | Humidified NIV vs HFNC, 48 h | Reintubation, day 7 | 23.3% vs 38.8% (difference −15.5%, −28.3 to −1) |
| Ferrer 2009, Lancet (19682735) P | Chronic respiratory disorder with hypercapnia during the SBT | NIV 24 h vs conventional O2 | Post-extubation respiratory failure, 72 h | 15% vs 48% (OR 5.32, 2.11–13.46); 90-day mortality lower |
| Nava 2005, CCM (16276167) P | Ventilated >48 h with ≥1 of hypercapnia, CHF, ineffective cough, >1 failed SBT, >1 comorbidity, airway obstruction | NIV ≥8 h/day for 48 h vs standard care | Reintubation | 4/48 vs 12/49 (p = .027) |
| Maggiore 2014, AJRCCM (25003980) P | PaO2/FiO2 ≤300 before extubation | HFNC vs Venturi, 48 h | PaO2/FiO2 (reintubation secondary) | Reintubation 4% vs 21% |
| RINO, Maggiore 2022 (35849787) P | PaO2/FiO2 ≤300 after extubation | HFNC vs Venturi | Reintubation, 72 h | 13% vs 11% (OR 1.26, 0.70–2.26) — null |
| EXTUBOBESE 2023, Lancet Respir Med (36693403) P | BMI ≥30 | NIV vs oxygen (HFNC or standard) | Treatment failure, day 3 | 13.5% vs 26.5% (RR 0.43); reintubation 10% vs 12% (p = .26) |
| Hernández 2025, AJRCCM (39514845) P | BMI >30, ≤3 risk factors, hypercapnia excluded | Humidified NIV vs HFNC, 48 h | Reintubation, day 7 | 23.6% vs 33.3% (difference 9.7, −4.9 to 24.4); underpowered |
| Casey 2021, AJRCCM (33794131) P | All medical-ICU extubations, one US academic center | Protocol (NIV if suspected hypercapnia, else HFNC) vs usual care | Reintubation, 96 h | 15.9% vs 13.3% (OR 1.23, 0.82–1.84) — null |
| Esteban 2004, NEJM (15190137) R | Respiratory failure within 48 h of elective extubation | NIV vs standard care | Reintubation | 48% in both arms; ICU death 25% vs 14% (RR 1.78, 1.03–3.20) |
| Keenan 2002, JAMA (12076220) R | Respiratory distress within 48 h | NIV vs O2 | Reintubation | 72% vs 69% (RR 1.04) — null |

Common exclusions across trials: tracheostomy, DNR or do-not-reintubate orders, accidental or self-extubation, NIV contraindications; most required a passed SBT and ventilation ≥24 h.

Pooled evidence and guideline:
- Fernando network meta-analysis (34825256, 36 RCTs): vs conventional O2, NIV OR 0.65 (0.52–0.82) and HFNC OR 0.63 (0.45–0.87) for reintubation; NIV vs HFNC OR 1.04 (0.78–1.38). The effect is largest at higher baseline risk and holds for prophylaxis, not rescue.
- Boscolo network meta-analysis (37019458): prophylactic support did not prevent failure in low-risk or hypoxaemic patients.
- HIGH-FLOW OXY 2026 (PMC13446969): HFNC worse than NIV in very-high-risk patients (OR 1.63, 1.05–2.53).
- ATS 2026 guideline (42371750): HFNC for low-risk and NIV for high-risk patients after extubation.

---

## 2. Who benefits (heterogeneity of treatment effect)

| Effect modifier | Evidence | Strength |
|---|---|---|
| Hypercapnia (PaCO2 >45) | Ferrer 2009 (NIV vs O2). Benefit also extends to non-hypercapnic high-risk patients: Thille 2026 post hoc of 829 patients, g-computation −5.6% (−11.0 to −0.5) (41973108) | Strong for hypercapnia; moderate beyond it |
| Obesity | HIGH-WEAN post hoc: interaction p = .007; BMI ≥25 reintubation 7% vs 20% (34813391). EXTUBOBESE surgical analysis: benefit concentrated at BMI ≥40 (41046173). HFNC raised reintubation risk in overweight patients in the Hernández 2016b post hoc, OR 2.47 (36089625) | Moderate; favors NIV |
| Number of risk factors | ≥4 factors: NIV 23.9% vs HFNC 45.7% (36089625); confirmed by Hernández 2022 | Moderate to strong |
| COPD | NIV+HFNC vs HFNC 13% vs 27% (33559765); small HFNC-vs-NIV trials suggest non-inferiority | Conflicting |
| Low risk | HFNC beats O2 (Hernández 2016a), but Boscolo and Casey find no benefit | Weak, contested |
| Hypoxaemia alone | HFNC vs Venturi null for reintubation (RINO) | Moderate (null) |
| Rescue setting | NIV gives no benefit and possible harm (Esteban, Keenan) | Strong for "do not expect benefit" |

Summary of who benefits:
- NIV, usually alternating with HFNC: hypercapnic or COPD patients, BMI ≥25–30 (especially ≥40), and patients with ≥4 risk factors.
- HFNC over conventional oxygen: low-to-intermediate-risk, non-hypercapnic patients, with the weakest evidence.
- Conventional oxygen: reasonable for genuinely low-risk patients.
- Rescue NIV after failure has started: not supported.

Gaps the trials leave open: continuous effect variation across combined risk factors (trials only dichotomize), effects in US populations (all positive NIV trials are French or Spanish), cardiac and neurologic subgroups, and dose (NIV hours, HFNC flow).

---

## 3. Definitions for building the emulation from EHR data

- Extubation failure: reintubation within 7 days (HIGH-WEAN) or reintubation or death within 7 days (WEAN SAFE, 36693401). Windows of 72 h and 96 h are also used.
- Post-extubation respiratory failure (HIGH-WEAN, PMC7871306): ≥2 of RR >25, clinical distress, pH <7.35 with PaCO2 >45, FiO2 ≥50% for SpO2 ≥92% or PaO2/FiO2 <150.
- Hypercapnia: PaCO2 >45 at end of SBT or on extubation day.
- Operational risk factors (Hernández 2022): inadequate secretions = suctioning >2 times in the 8 h before extubation; difficult weaning = ≥1 failed SBT; prolonged ventilation = ≥7 days.
- Composite "treatment failure" (EXTUBOBESE, BiPOP) counts switching devices; in EHR data, switching is a care decision, so the emulation uses reintubation instead.
- Hernández 2016 respiratory-failure thresholds: **[unverified]** (JAMA full text not retrieved).

---

## 4. Observational precedent

- Liu 2021 (33985472), MIMIC-IV hypoxaemic patients: HFNC vs NIV 48-h reintubation 8.6% vs 15.9% after propensity matching.
- Ge 2023 (37553185), MIMIC-IV obese patients: HFNC vs NIV OR 1.50, and at BMI ≥40 HFNC OR 0.06. That direction is the opposite of EXTUBOBESE and HIGH-FLOW OXY — a published case of confounding by indication in this exact dataset.
- No published post-extubation target trial emulation benchmarked against these RCTs was found.

---

## 5. Causal design

**Target trial emulation.** Protocol elements follow Hernán & Robins (26994063); time zero must coincide with eligibility and assignment (27237061). Reporting follows TARGET (40903028, 40899949).
- Time zero is the extubation timestamp; eligibility must be assessable from data available before it.
- Device strategies need a grace period. Patients who start a device late belong to several strategies at time zero, so clone-censor-weight is the standard fix (Maringe 2020, 32386426).
- A long grace period lets rescue NIV count as prophylaxis.
- Competing events: death prevents later reintubation. Report the composite plus cause-specific cumulative incidence (Young 2020, 31985089).

**Benchmarking against RCTs.**
- RCT-DUPLICATE (Wang 2023 JAMA, 37097356): 32 trials, Pearson r = 0.82. Agreement metrics: significance agreement 75%, estimate agreement 66%, standardized-difference agreement 75%. Closely emulable trials reached r = 0.93.
- Wang 2026 BMJ (42156120): respiratory outcomes were overestimated (ratio of ratios 1.20). Treatment started in hospital is a major source of disagreement (Heyard 2024, 38348308), but EHR timestamps help here.
- Protocols and agreement criteria were registered before analysis in RCT-DUPLICATE (33327727).

**ICU emulations against RCTs.**
- Admon 2019 (31038996), PreVent emulation run blind: RR 0.60 vs trial 0.48.
- Hoffman 2022 (36190729), steroids in COVID: target trial emulation with doubly robust estimation matched the meta-analysis, while conventional Cox models ranged from HR 0.50 to 1.08.
- Dai 2026, MIMIC-IV crystalloids (PMC13355737): estimates reversed direction depending on how exposure was defined, even with good propensity diagnostics.

**Confounding diagnostics.**
- Positivity and overlap (21030422).
- Negative controls (20335814) and empirical calibration (ReClaim, arXiv 2605.02740).
- E-values (28693043).
- Avoid adjusting for instruments such as unit, attending or time of day (22025356).

**Estimators.**
- A device-token model is structurally an S-learner and is biased toward a null effect (Künzel, arXiv 1706.03461).
- Doubly robust and orthogonal estimators: Kennedy DR-learner (arXiv 2004.14497); DML (arXiv 1608.00060).
- Representation-induced confounding bias: Melnychuk (arXiv 2311.11321).
- Foundation-model embeddings in propensity models reduced negative-control error from 0.16 to 0.04 (ReClaim), though overlap fell from 69% to 45.7%.
- Data-adaptive adjustment that models both treatment and outcome agreed with RCTs better than investigator-specified models (Weckstein 2026, 41338229).

**Validating per-patient effects.**
- RATE/TOC (Yadlowsky, arXiv 2111.07966).
- CATE calibration (arXiv 2203.13364).
- BLP/GATES/CLAN (Chernozhukov et al.).
- c-for-benefit (29132832); the PATH statement favours risk-based analysis first (31711134).
- Concordance analysis (Lu 2025) mixes treatment effect with confounding and is hypothesis-generating only.

---

## 6. Counterfactuals from generative and foundation models

| Work | What it intervenes on | Did it reproduce an RCT effect? |
|---|---|---|
| G-Net (arXiv 2003.10551), G-Transformer (arXiv 2406.05504) | Monte Carlo g-computation with simulated covariates | Simulation and MIMIC case studies; no RCT benchmark |
| Causal Transformer (arXiv 2204.07258), CRN (arXiv 2002.04083) | Balanced representations | No; balancing did not reliably help (arXiv 2408.08815) |
| EHR-MPC (arXiv 2607.08793) | Drug tokens injected into a GPT-2-style EHR model | No; injected vasopressor tokens raised predicted mortality, consistent with confounding by indication |
| TRIALSCOPE (arXiv 2311.01301) | Language-model-structured EMR plus classical adjustment | Yes, equivalent hazard ratios in 9/9 oncology trials — but with classical estimators, not rollouts |
| ReClaim (arXiv 2605.02740) | Foundation-model embeddings in propensity models | Lower negative-control error; not compared to an RCT |
| Lu 2025, npj Health Syst (41358050) | HFNC vs NIV per-patient effects, acute respiratory failure (not post-extubation) | No; concordance analysis only |
| DINIRS (arXiv 2608.26915) | Non-invasive support vs invasive ventilation, doubly robust learner | Qualitative alignment only |
| ETHOS, Foresight, CoMET, Delphi-2M | None | Not tested; Delphi authors caution against causal reading |

Two works found during the 2026-10-03 stress test come closest. HealthFormer (arXiv 2604.27899) simulated interventions in a generative model of a deeply phenotyped non-ICU cohort and matched the direction of 41 of 41 randomized comparisons, with 30 inside the trial's 95% CI. FlatASCEND (arXiv 2605.04071), a 14.5M autoregressive model on MIMIC-IV, tested intervention tokens against known pharmacology and recovered 4 of 10 expected directions; its authors attribute the result to learned associations, and reward-based post-training destroyed the correct ones. Neither is a pre-registered, estimator-level benchmark with confounding control for an ICU EHR model.

**What the injected-token head estimates.** P(Y | history, device = a) equals the interventional risk only if the history blocks every back-door path, positivity holds, and the history contains nothing caused by the device. Averaging it over eligible patients is single-time-point g-computation, the same estimator Thille 2026 used. Because the head is trained on real futures that include rescue NIV and escalation chosen by clinicians, the estimand is "start device a at extubation, then usual care" — close to an RCT intention-to-treat contrast.

**What rollouts estimate.** Treatments are never sampled, so a rollout contains no future treatments. That is an implicit regime of "no further documented treatment", outside the training support, and the model reads the missing escalation as evidence of improvement. Rollouts are descriptive only unless a policy model or forced regime is added.

**Failure modes to test for:**
- Shrinkage toward the null.
- Charting shortcuts: a device is documented because the patient deteriorated, and whether a lab was ordered predicted survival for 86% of 272 tests (bmj.k1479).
- Positivity loss as embeddings sharpen propensities.
- Storetime leakage after time zero.
- Rollout drift over long horizons.

---

## 7. Data feasibility in MIMIC-IV-Ext-CLIF

CLIF 2.1 supplies device category (IMV, NIPPV, CPAP, High Flow NC, Face Mask, Nasal Cannula, Trach Collar), mode, tracheostomy flag, flow and FiO2 settings in `respiratory_support`; PaCO2 and pH in labs; weight and height in vitals; code status; and discharge disposition. The CLIF consortium's shared extubation definition (IMV→non-IMV transition, first extubation only, trach excluded, withdrawal-of-life-support flag) is in progress in clifpy issue #124.

Cohort funnel (aggregate, local MIMIC):

| Step | Count |
|---|---|
| Hospitalizations | 546,028 |
| With an ICU stay | 85,248 |
| With any invasive ventilation | 35,161 |
| With an extubation event | 24,804 |
| First extubation after ≥12 h ventilation | 15,176 (10,042 after ≥24 h) |

Outcomes in the ≥12 h cohort: reintubation within 48 h 7.4%; within 7 days 12.9%; reintubation or in-hospital death within 7 days 15.9%.

Highest support within 6 h of extubation:

| Arm | n | Reintubation or death ≤7 d |
|---|---|---|
| NIV or CPAP | 633 | 29% |
| HFNC | 829 | 26% |
| Face mask | 10,368 | 16% |
| Nasal cannula | 3,306 | 11% |

The NIV and HFNC arms are small and shrink further with a 2-h window (447 and 737). Their higher outcome rates show confounding by indication.

Feasibility risks:
- "Face Mask" is the first device for 70% of extubations and likely includes aerosol face tents, so the conventional-oxygen arm needs a definition.
- HFNC flow is often missing (7,145 rows) or low (10th percentile 15 L/min).
- Respiratory rows in the 24 h after extubation are a median 180 min apart (90th percentile 300 min), which limits how short a grace period can be.
- `poa_present` is null for every MIMIC diagnosis row and codes are recorded after discharge, so index-stay COPD and heart-failure codes leak. Prior-hospitalization codes or physiology proxies are needed.
- Pre-extubation arterial PaCO2 is present in 81.5% of extubations (missing not at random).
- Documented SBT before extubation: 54.5%.
- Code status at extubation is missing in 43%, so the withdrawal-of-life-support flag is unreliable.
- Cough and secretion burden is not charted; that trial criterion cannot be built.
- Self-extubation is not representable in CLIF 2.1.
- Rush cohort size is **[unverified]**.

---

## 8. What ICU teams find useful from decision support

- Clinicians want actionable recommendations, not just risk scores: "adding treatment recommendations would have been more useful" (Ayorinde 2024, PMC11561443); nurses want "more than a score, guide nurses to action" (Wieben 2024, PMC11771625); physicians ask for an if/then rule tied to the score (Schwartz 2022, PMC9136656).
- Clinicians want to stay in the driver's seat and decide whether to follow advice (Bienefeld 2024, PMC11301121).
- Mistimed, uncontextualized alerts are the leading cause of tool abandonment; alert fatigue matters more than accuracy (sepsis phenotyping review PMC13498814; Joshi 2022, PMC9030109).
- The guideline's operational gap: "do we really know what high risk means?" (Glossop 2016, PMC4937552). A calibrated risk estimate at extubation is what turns the ATS risk-based recommendation into a bedside rule.
- Existing extubation-failure models are mostly single-center risk scores with no device recommendation (Zhao 2021, PMC8165178, MIMIC-based CatBoost, AUC 0.835 internal / 0.803 external; Chen 2026, PMC12870664; Fenske 2025, PMC12307926, next-day extubation at Northwestern).

---

## 9. Scoring agreement with trials

- **RCT-DUPLICATE:** agreement "varies depending on which agreement metric is used" (Franklin 2021, PMID 33327727); Pearson r 0.82 across 32 trials (Wang 2023, PMID 37097356).
- **Heyard 2024 (PMID 38348308):** standardized difference per pair and heterogeneity across 29 pairs; three emulation differences explained most of it.
- **BenchExCal (PMID 40067205):** divergence between emulation and trial estimates, used as a prior with tipping-point analysis; it requires emulation power at least equal to the trial's and calls binary metrics "simplistic".
- **Null trials:** "replication success can virtually always be achieved if the sample sizes are small enough" (Pawel 2024, PMID 38739437); use equivalence tests (Micheloud 2024, PMID 39473139).
- **Sceptical p-value (Köppe 2025, PMID 40413382):** handles non-inferiority margins but cannot separate design, model and population causes.
- **ICU evidence:** Kitsios 2015 (PMID 26086943) found 1 significant difference in 18 comparisons, yet estimates differed by more than 30% in a third.
- **Transportability:** Dahabreh 2022 (arXiv 2203.14857) assumes patient-level data from both sources. Without trial patient-level data, only subgroup reweighting or a homogeneous relative effect are possible (Hong 2019, PMID 30312378); standardizing four emulations to trial age and sex gave "minimal changes" (Htoo 2026, PMID 41733244).
- **Planted-effect simulation:** the common "sample treatment" recipe induces a positivity violation and misranks estimators; draw treatment from a fitted propensity model instead (Shaw 2026, PMID 42487285). Plant unmeasured confounding by withholding simulated variables (Desai 2026, PMID 42093129).
- **Negative controls:** suitability matters more than count ("minimal gains" from 30 versus 5; Hwang 2022, arXiv 2111.04233); length-of-stay-dependent outcomes are impure (Dai 2026, PMID 42436975). No validated negative controls exist for HFNC, NIV or oxygen exposures.
- **Pre-registration:** register before any outcome-by-exposure analysis; declare prior data access and use a hold-out (Baldwin 2022, PMC8791887); TARGET item 18 (PMID 40899949).
- **Gap:** no validated metric exists for ranking estimators over a handful of small trial pairs.

---

## 10. Identification when key confounders are unrecorded

- **Preference instruments are imprecise at this size.** A physician-preference analysis in 476 ICU patients matched the trial direction with "very wide confidence intervals" (Boef 2014, PMID 25051311); in 210,115 arrests, preference instruments gave "implausibly large" effects (Holmberg 2026, PMID 41561319).
- **Adoption-era designs.** Casey 2021 (PMID 33794131) is the randomized version: a protocol moved HFNC use from 2.8% to 74.7%. HFNC use rose from 15.9% to 28.0% over 2018–2022 in one national dataset (Maezawa 2026, PMID 41783135). Usable only if device use shifted sharply for non-patient reasons.
- **Proximal inference** has an ICU time-to-event application and software (Li 2025, PMID 40513053).
- **Strategy definition drives the answer.** In a MIMIC-IV intubation emulation, unrealistic strict strategies gave +7.1 points mortality and realistic ones 0.4 and −0.9 (Wanis 2023, PMID 37150505).
- **Estimators.** Outcome-adaptive adjustment beat investigator-specified adjustment in 73–87% of 15 trial emulations, and models tuned only for treatment prediction "performed poorly" (Weckstein 2026, PMID 41338229). Overlap weights beat inverse probability weights under limited overlap (Zhou 2020, PMID 32693715). Cross-fitted doubly robust estimators outperform but need larger samples (Zivich 2021, PMID 33591058).
- **Learned representations.** Embedding-augmented propensity models gave negative-control estimates more tightly centred on the null (ReClaim, arXiv 2605.02740); double machine learning on pretrained features is valid under conditions (Schulte 2025, arXiv 2506.14329); keep only what predicts both treatment and outcome (Veitch 2019, arXiv 1905.12741).
- **Palliative extubation.** Among 226 palliative extubations, 84.5% were admitted full code and the median extubation-to-death time was 118 minutes (Krinsley 2025, PMID 41016642). No EHR algorithm for finding them without code status was found.
- **Unrecorded confounders matter.** Thille 2020 (PMID 32164739) analysed reintubation or death to day 7 by cough strength and limb weakness; severe weakness independently raised risk.
- **No target trial emulation of post-extubation NIV or HFNC was found.**

---

## 11. Extubation-failure risk models: the bar

- **Outcome definitions inflate published performance.** Zhao 2021 (PMID 34079812) reached AUROC 0.835 on NIV, reintubation or death within 48 h; RMS-EF (PMID 41419550) reached 0.865 on 48-h reintubation. On 7-day reintubation or death, the only retrieved model scored 0.70 (Fleuren 2021, PMID 34961537).
- **Evidence quality is low.** Of 40 studies, 85% lacked external validation, 35% reported calibration and 13% net benefit (Murali 2026, PMID 42712302). None retrieved is in routine clinical use.
- **Transfer warning.** RMS-EF transferred poorly until sedative and vasopressor variables were dropped.
- **The rules to beat:** age over 65 or chronic cardiac or respiratory disease (Thille; a before-after protocol cut reintubation from 28% to 15%, PMID 26926168); the Hernández factor count (≥1 high risk, ≥4 very high risk); the RISC score (PMID 35252224).
- **Bedside judgement is weak.** Only a third of reintubated patients were flagged high risk by caregivers (PMID 25479115).
- **US delivery is low.** Baseline post-extubation NIV or HFNC use was 2.3% at UPMC (PMID 41130691) and 16.8% at Vanderbilt (Casey 2021).
- **Testing benefit in randomized data.** Buell 2024 (PMID 38501205) derived a model in PILOT and validated effect modification in ICU-ROX; Grolleau 2024 (PMID 38452293) validated an EHR-learned policy on the AKIKI trials. PATH statement (PMID 31711134); c-for-benefit (PMID 29132832).
- **Trial data.** The Poitiers group has published pooled analyses of its trials (PMIDs 41973108, 42440109); the Toledo group ran four trials on one risk-factor list. No open repository was identified.
- **Evaluation short of a trial.** TRIPOD+AI (PMID 38626948); DECIDE-AI (PMID 35584845); silent deployments have caught dataset shift before go-live (Kwong 2022, PMID 36052317).

---

## 12. Implications for the plan

1. Extubation is the single clinical application of the three-claim paper; its primary test is the known-answer pattern (NIV's advantage over HFNC grows with baseline risk), with per-trial emulations for every trial precise enough to score.
2. Register the protocol and agreement margin before any adjusted estimate; MIMIC is exploratory and Rush confirmatory.
3. Run three estimators on identical held-out patients (classical clone-censor-weight with doubly robust estimation, doubly robust on the frozen model state, injected-token head standardization), with fair baselines.
4. Score by the gap between emulation and trial estimates against a registered margin; judge null trials by equivalence; publish the rule's operating characteristics from planted-effect simulation.
5. Use negative-control outcomes as diagnostics; report rescue NIV descriptively; add an adoption-era contrast if device use shifted for non-patient reasons.
6. Treat rollouts as descriptive only.
7. MIMIC alone is underpowered for the NIV and HFNC arms; Rush is needed for power, not only replication.
8. Paper 2 is a risk model on the guideline's risk axis, compared with the simple rules on net benefit and reclassification; a device recommendation waits for validation in randomized or prospective data.
