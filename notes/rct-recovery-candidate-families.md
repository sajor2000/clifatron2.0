# RCT-recovery benchmark: candidate ICU trial families

Screened 2026-10-03 for `docs/plans/2026-10-03-0845-feat-icu-gem-rct-recovery-plan.md` (R25, R26).
Post-extubation respiratory support is the deep case and is covered in `notes/extubation-evidence-review.md`.
Effect sizes come from retrieved abstracts; unretrieved claims are marked **[unverified]**.
MIMIC counts are aggregate-only feasibility queries on local MIMIC-IV-Ext-CLIF with rough definitions; every reported cell is at least 10.

---

## Recommendation

| Family | Role in the benchmark | Why |
|---|---|---|
| HFNC vs conventional oxygen or NIV before intubation (FLORALI, HIGH, RECOVERY-RS) | Second device family | Reuses the extubation device machinery; arms three to four times larger (2,589 HFNC and 1,694 NIV/CPAP first escalations in MIMIC); includes null trials |
| Vasopressin (VASST, VANISH) | Null family | A point treatment with tokenized doses; null trials judged by an equivalence margin; the septic-shock label is the gap |
| Prone positioning (PROSEVA) | Declared stress test, not a positive control | Intervention and physiology are tokenized, but only 4.4% of eligible stays were proned within 48 h, mostly as salvage, and there is no imaging for the ARDS definition; expect a null or harmful estimate |

Revised after the 2026-10-03 stress test. Two earlier recommendations were withdrawn:
- **Oxygen targets.** Median achieved SpO2 over 72 h is measured after time zero and depends on severity and survival, which is the error behind the known false-positive study. It is admissible only as a dynamic FiO2-titration regime.
- **RRT timing.** It fails hard gate 1 below: intermittent hemodialysis is not in the tokenized tables, so the never-treated arm is contaminated, and eligibility would rest on creatinine alone.

Each family still has to pass the screen below at each site before registration.

---

## Gaps that apply to every family

- **Post-discharge death.** The token stream carries death only as `DISCHARGE//expired`. `clif_patient.death_dttm` exists (38,301 of 364,627 MIMIC patients) but is not tokenized, so 28-, 60- and 90-day mortality needs a separate labeller, or the trial outcome is replaced with in-hospital death and declared in advance.
- **Untokenized tables.** No urine output, fluid input, imaging (ARDS bilateral infiltrates), microbiology or diagnosis-based septic-shock label. MIMIC `poa_present` is null for every diagnosis.
- **Drug-category gaps in local CLIF.** No corticosteroid, balanced crystalloid or blood-product categories; fluids limited to sodium chloride, dextrose and albumin (CLIF issue #227 notes the fluid lists are incomplete).
- **Intermittent hemodialysis** appears only as billing codes in `patient_procedures`, not in the CRRT table.

---

## Family screen

### Prone positioning in severe ARDS — Strong (positive control)
- Benchmark: PROSEVA (PMID 23688302). P/F <150 with FiO2 ≥0.6, PEEP ≥5 and Vt about 6 mL/kg; proning ≥16 h per session vs supine. 28-day mortality 16.0% vs 32.8%, HR 0.39 (0.25–0.63); 90-day HR 0.44 (0.29–0.67).
- Representable: partial. `position=prone`, P/F (PaO2 matched to FiO2), PEEP, and Vt per predicted body weight are tokenized. Gaps: no imaging for the ARDS definition; the 28-day outcome needs `death_dttm`.
- MIMIC: 7,494 ventilated stays met P/F <150, FiO2 ≥0.6 and PEEP ≥5; 331 (4.4%) were proned within 48 h and 652 ever.
- Bias and precedent: proning is often salvage therapy, so expect confounding by indication and thin positivity. Izdebski (arXiv 2109.06707), Dutch COVID data, found P/F +14.5 to +20.1 mmHg 2–8 h after proning, close to PROSEVA's +15 — agreement on the physiological intermediate only.
- Negative control: none retrieved within the family **[unverified: earlier null proning trials]**.

### HFNC vs conventional oxygen or NIV before intubation — Possible
- Benchmarks:
  - FLORALI (25981908): intubation by day 28 was 38% (HFNC) vs 47% (O2) vs 50% (NIV), P = 0.18; 90-day death HR 2.01 and 2.50 against HFNC (secondary outcome).
  - HIGH (30357270): HR 0.98 (0.77–1.24).
  - RECOVERY-RS (35072713): HFNC vs O2 −1% (−8 to 6); CPAP −8% (−15 to −1).
- Representable: partial. Device, flow and FiO2 are tokenized. P/F on nasal cannula depends on unreliable charted FiO2, and dyspnea is not charted.
- MIMIC: 11,192 stays with P/F ≤300 on non-invasive oxygen. First escalation within 24 h was HFNC in 2,589, NIV/CPAP in 1,694, and intubation in 1,219.
- Shares the device machinery built for the extubation family.

### Oxygen targets — Strong (null control)
- Benchmarks (all null):
  - ICU-ROX (31613432): ventilator-free days −0.3 (−2.1 to 1.6); 180-day mortality OR 1.05.
  - HOT-ICU (33471452): 90-day mortality RR 1.02 (0.94–1.11).
  - PILOT (36278971): ventilator-free days 20 / 21 / 21, P = 0.81.
  - UK-ROX (40501321), n = 16,500: risk difference 0.7 points (−0.7 to 2.0).
- Representable: yes for exposure (dense SpO2 and `fio2_set`); outcomes need `death_dttm`.
- MIMIC, median SpO2 over the first 72 h of ventilation: ≤92% in 756; 93–95% in 4,448; 96–97% in 11,215; ≥98% in 18,658. A PILOT-style intermediate-vs-high contrast is feasible; a conservative (≤92%) arm is not.
- Known observational failure: van den Boom (31589844), MIMIC and eICU, found SpO2 94–98% associated with mortality OR 0.42–0.53 against null RCTs.
- Heterogeneity benchmark: Buell (38501205), a model derived in PILOT and validated in ICU-ROX, found effect modification (P = .02).

### Low tidal volume and PEEP — Poor
- ARMA (10793162): mortality 31.0% vs 39.8%. ART (28973363): recruitment and titrated PEEP harmful, HR 1.20 (1.01–1.42).
- MIMIC Vt over the first 48 h: ≤6.5 mL/kg 7,933; 6.5–8 11,115; 8–10 4,403; ≥10 only 812. ARMA's 12 mL/kg arm has no support; recruitment manoeuvres are not represented.

### Early vs delayed renal replacement therapy — Possible (dynamic regime)
- Benchmarks (null):
  - AKIKI (27181456): 60-day mortality 48.5% vs 49.7%; 49% of the delayed arm never received RRT.
  - STARRT-AKI (32668114): 90-day mortality RR 1.00 (0.93–1.09); RRT dependence among survivors RR 1.74 against accelerated start.
- Representable: partial. CRRT start, creatinine, ventilation and vasopressors are tokenized. Intermittent hemodialysis, urine output and true baseline creatinine are not.
- MIMIC: 4,810 stays with stage-3 creatinine plus ventilation or a vasopressor. CRRT started ≤12 h in 509, 12–72 h in 947, >72 h in 415, never in 2,939 (192 already on CRRT before eligibility).
- Known observational failure: She 2025 (40978732), MIMIC-IV, reported 90-day HR 0.561 favouring early RRT against null RCTs, with internally inconsistent estimators. Grolleau (38452293) learned a policy in MIMIC-III and validated it on AKIKI trial data.

### Balanced crystalloids vs saline — Poor
- SMART (29485925) OR 0.91 (0.84–0.99); BaSICS (34375394) HR 0.97; PLUS (35041780) −0.15 points; SPLIT (26444692) RR 1.04.
- Not representable: no balanced-crystalloid category or fluid volumes. Dai 2026 (42436975) showed opposite answers by exposure definition in MIMIC-IV.

### Vasopressin — Possible (alternate)
- VASST (18305265): 35.4% vs 39.3%, P = 0.26. VANISH (27483065): kidney-failure-free survivors −2.3% (−13.0 to 8.5).
- Doses tokenized; the norepinephrine ≥5 µg/min threshold is buildable; the septic-shock label is not.
- MIMIC: 15,126 stays on norepinephrine; 5,116 also on vasopressin; vasopressin first in 524.
- OVISS (40098600): a policy-value result (OR 0.81 when use matched the learned rule), not a trial emulation.

### Hydrocortisone in septic shock — Poor
- ADRENAL (29347874) OR 0.95; APROCCHSS (29490185) RR 0.88. No corticosteroid categories in local CLIF (Rush coverage **[unverified]**).

### Dexmedetomidine vs propofol — Poor
- MENDS2 (33528922) and SPICE III (31112380) both null. Dexmedetomidine was the first sedative within 12 h of intubation in only 371 stays vs 23,488 propofol.

### Restrictive vs liberal transfusion — Poor
- TRICC (9971864): 18.7% vs 23.3%, P = 0.11. No blood products in local CLIF med tables.

---

## Pre-specified feasibility screen

Apply per family and per site, aggregate counts only, cells under 10 suppressed. A family passes only when every hard gate (H) passes.

1. (H) The intervention maps to a tokenized table and category, with timing at adequate resolution, and every estimator in the plan can run on it (the injected-token head needs a single tokenized treatment started at time zero).
2. (H) Time zero is constructible: eligibility is computable from tokens available at or before one timestamp, and assignment is pinned there with a grace period no longer than the trial's.
3. (H) The trial outcome and window are derivable; if post-discharge death is needed, the `death_dttm` labeller is specified and the in-hospital proxy declared in advance.
4. (H) Positivity: at least 100 initiators per arm per site in the emulated window, and propensity overlap that trims under 5% of eligible patients.
5. At least 70% of trial inclusion and exclusion criteria are directly representable; each proxy is listed with its expected bias direction.
6. Key eligibility variables are present for at least 80% of candidates; missing-not-at-random is checked across outcome groups.
7. Strategy switching within 48 h is reported; above 30%, a per-protocol or dynamic-regime estimand is required.
8. At least one null trial or a pre-specified negative-control outcome (not driven by length of stay) exists.
9. Any prior EHR estimate that agreed or failed is listed, with "recovered" defined in advance.
10. Event rates and treatment prevalence are reported per site (MIMIC, Rush, UChicago).
11. Calendar drift is reported (treatment prevalence before vs after the index trial's publication).

## Unverified

Rush and UChicago prevalence of proning and other interventions; corticosteroid and blood-product coverage at other CLIF sites; earlier null proning trials as negative controls; AKIKI2 and ELAIN results; all MIMIC counts use rough definitions (P/F from the latest FiO2 within 4 h; baseline creatinine as the first value in the hospitalization).
