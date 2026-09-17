# Caprini item → data-source mapping (third clinical comparator)

Implementation: **Caprini risk assessment model, 1-2-3-5 point item set as
published in Caprini JA, *Dis Mon* 2005;51:70-8 and restated in the 2010 update
(Caprini JA, *Clin Appl Thromb Hemost* 2010;16:24S–37S)**.

Script: `scripts/85h_clinical_scores_caprini.py`
Output: `results_vte/clinical_scores_v3_caprini.parquet`
(columns `hadm_id, caprini_score, caprini_high (≥5), caprini_modplus (≥3),
caprini_band (0 = 0 pts, 1 = 1–2, 2 = 3–4, 3 = ≥5)`).

Cohort: all 425,090 admissions of `results_vte/vte_labels_v3.parquet`.
Inputs are exactly the four structured feature parquets already used for the
Padua and IMPROVE mappings in `85e_clinical_scores_v3.py`
(`features_comorbid_v3`, `features_vitals_v3`, `features_treatments_v3`,
`vte_labels_v3`); no new database extraction was performed.

Risk categories (as in the source form): 0 = lowest, 1–2 = low, 3–4 = moderate,
≥5 = high. **The declared high-risk cut-off is ≥5**, because that is the
category the Caprini form and its guidelines label "high risk"; ≥3 is also
reported (`caprini_modplus`) because the score's original use case is to
trigger prophylaxis at ≥3.

## Mapped items

| Caprini item | Pts | Data source | Mapping type | Limitation |
|---|---|---|---|---|
| Age 41–60 y | 1 | `age` at admission (`features_comorbid_v3`) | direct | none |
| Age 61–74 y | 2 | `age` | direct | none |
| Age ≥75 y | 3 | `age` | direct | none |
| BMI > 25 kg/m² | 1 | `bmi` (first-24 h vitals) **or** fallback `obesity_icd` (E66 / 278.0) | direct + fallback | `bmi` missing in 32.3 % of admissions, concentrated outside the ICU; the ICD fallback only recovers BMI ≥30, so BMI 25–30 without a recorded height/weight is scored 0 |
| Varicose veins | 1 | `varicose` (I83 / 454) | direct | ICD under-records mild disease |
| Oral contraceptives / hormone replacement | 1 | `rx_hormone` (in-hospital estrogen, progestin, tamoxifen, aromatase inhibitor, GnRH analogue, "contracept") | direct | only in-hospital administration is captured; outpatient OCP/HRT is invisible |
| Sepsis (<1 month) | 1 | `infection_severe` (A40/A41 / 038, 995.91–995.92) | proxy | = severe sepsis/septic shock only; other acute infections (e.g. pneumonia without sepsis) are missed |
| Abnormal pulmonary function (COPD) | 1 | `copd` (J40–J44 / 490–496) | direct | the overlapping 1-point line "serious lung disease incl. pneumonia" is not separately mappable (no pneumonia flag in the extracted feature set) and is deliberately *not* double-counted |
| Acute MI (<1 month) | 1 | `mi` (I21/I22 / 410) | proxy | admission-level code: acute vs old MI not distinguishable, so prior MI may be scored |
| Congestive heart failure (<1 month) | 1 | `heart_failure` (I50 / 428) | proxy | acute vs chronic not distinguishable |
| Medical patient currently at bed rest | 1 | **proxy**: `icu_los_first24h > 0` | **proxy** | MIMIC has no mobility/activity orders. Same proxy and same caveats as the Padua "reduced mobility" and IMPROVE "paralysis/immobilization" items: ICU admission ≠ documented bed rest, and bed rest on the ward is entirely missed. The proxy is used **once** (not repeated for the 2-point "confined to bed >72 h" item) to avoid double-counting an unvalidated proxy |
| Major surgery (>45 min) | 2 | **proxy**: `surgery_flag` (any ICD-10-PCS procedure code in section 0, length ≥4) | **proxy** | operative duration is not recorded, so "major" is defined as *any* coded operation and the minor/major distinction is lost; the 1-point "minor surgery" line is left unmapped so a given operation is never counted twice |
| Malignancy (present or previous) | 2 | `cancer_active` (C-coded malignancy / ICD-9 140–208 on this admission) | proxy | captures *current* malignancy only, whereas the item also covers previous malignancy → conservative (under-scores) |
| Central venous access | 2 | `proc_cvc` (ICD 02HV*/05HM*/06HM*, 3893*/3895* ∪ procedureevents line-placement items) | direct | bedside placement is under-coded; no window restriction (whole admission) |
| History of DVT/PE | 3 | `prior_vte_any` (prior admission VTE code, or present-on-admission VTE code on this admission) | direct | ICD sensitivity for historical VTE is limited |
| Other congenital/acquired thrombophilia | 3 | `thrombophilia` (D68.5/D68.6 / 289.8) | proxy | used for the single "other thrombophilia" line; because ICD-10 D68.5 covers the primary hypercoagulable states, factor V Leiden and antiphospholipid diagnoses are captured here *when coded*; documented inherited thrombophilia is strongly under-recorded |
| Stroke (<1 month) | 5 | `stroke` (I63/I64 / 433, 434, 436) | proxy | acute vs old stroke not distinguishable; haemorrhagic stroke is not included |

Mapped items cover 17 of the 38 item lines of the form.

## Unmappable items (scored 0, declared)

No source in the structured data extracted for this project. No proxy was
invented for any of them.

| Caprini item | Pts | Why unmappable |
|---|---|---|
| Minor surgery | 1 | no operative magnitude/duration; the `surgery_flag` proxy is spent on the 2-point major-surgery line |
| Swollen legs (current) | 1 | physical-examination finding, no structured record |
| Pregnancy or postpartum | 1 | no pregnancy/obstetric flag in the extracted feature set (would require a separate ICD-O/delivery-code extraction) |
| History of unexplained/recurrent spontaneous abortion | 1 | obstetric history not recorded in structured form |
| Serious lung disease incl. pneumonia (<1 month) | 1 | no pneumonia/exacerbation flag in the extracted feature set; overlaps the COPD line |
| Inflammatory bowel disease | 1 | IBD codes not among the extracted comorbidities |
| Arthroscopic surgery | 2 | procedure approach not recoverable from the aggregated `surgery_flag` |
| Laparoscopic surgery (>45 min) | 2 | procedure approach + duration not recoverable |
| Confined to bed (>72 h) | 2 | no duration of immobilization; the only proxy is the 24-h ICU proxy, already used once at 1 point |
| Immobilizing cast | 2 | no cast/orthosis data |
| Family history of thrombosis | 3 | family history not in MIMIC structured data |
| Factor V Leiden | 3 | no genetic results (partially captured only if coded under D68.5, which is charged to the "other thrombophilia" line instead) |
| Prothrombin 20210A mutation | 3 | no genetic results |
| Lupus anticoagulant | 3 | laboratory not extracted; may be captured via D68.6 codes under "other thrombophilia" |
| Anticardiolipin antibodies | 3 | laboratory not extracted |
| Elevated serum homocysteine | 3 | not among the extracted labs |
| Heparin-induced thrombocytopenia | 3 | no HIT flag (platelet-factor-4 antibody results not extracted) |
| Elective major lower-extremity arthroplasty | 5 | elective-vs-emergent and joint-replacement subclass not recoverable; partly counted as `surgery_flag` (2 pts) instead |
| Hip, pelvis or leg fracture (<1 month) | 5 | fracture site/acuity not recoverable from the trauma flag |
| Multiple trauma (<1 month) | 5 | `trauma_flag` (S/T codes) exists but is *not* used: mapping a generic trauma flag to a 5-point item would add 5 points to 20.4 % of admissions. Sensitivity check: doing so raises the mean score from 3.86 to 4.88 and the high-risk proportion from 35.7 % to 46.6 % (see `output_era/caprini_distribution.json`) |
| Acute spinal cord injury / paralysis (<1 month) | 5 | no paralysis/SCI flag; the 24-h ICU proxy is already used once |

## Overall cautions

1. **The mapping is deliberately conservative (asymmetric).** Missing items
   mostly *subtract* points (family history, HIT, genetic thrombophilias,
   arthroscopy, cast, fracture, multiple trauma), so absolute Caprini scores
   here are lower bounds. The one place where the mapping plausibly inflates
   the score is the `surgery_flag` proxy (any coded operation = 2 points
   "major surgery"), which applies to 24.3 % of admissions.
2. Because it is a lower bound with a single inflated item, the *absolute*
   score and the high-risk proportion must not be compared with published
   Caprini cohorts. Only the within-cohort discrimination and the incremental
   value over Padua/IMPROVE are interpretable.
3. The same data-source caveats that apply to the Padua and IMPROVE mappings in
   `score_mapping.md` (admission-level ICD codes, "acute/within 1 month" time
   windows not verifiable, ICU-based immobility proxy, 24-h medication window)
   apply verbatim here, since the underlying columns are identical.
