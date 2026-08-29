# TRIPOD+AI Checklist — VTE Transportability Manuscript (npj DM)

Completed for the ndm outline/manuscript. Annotations follow the TRIPOD+AI
convention: **P** = Present, **R** = Reported with detail, **D** = Derived
(available from a referenced source / supplement), **A** = Available (permanent
public link/DOI). Model-specification items (2) recorded in a separate box.

Target journal: npj Digital Medicine (Article). This is a **transportability /
external validation + repair** study (development in MIMIC-IV, external
validation in MIMIC-III and eICU), so TRIPOD+AI items for both development and
external validation are completed.

---

## Section 1: Title and abstract

| Item | Recommendation | Status | Location / notes |
|---|---|---|---|
| 1 | Identify as a prediction model (development, evaluation, or both) | P | Title: "Diagnosing and repairing transportability failure of machine learning venous thromboembolism prediction across electronic health record systems" |
| 2 | Model specification box (predictors and outcomes) | P (D) | 58 predictors (first-24h structured EHR); outcome HA-VTE radiology-confirmed (MIMIC) / ICD-9 (eICU). Full predictor dictionary in Supplement S-Tab1; see Model Spec Box below |
| 3 | Structured abstract (objective, methods, results, conclusions) | R | npj DM unstructured abstract (≤200 words) with objective / data / methods / results / conclusion |

## Section 2: Introduction

| Item | Recommendation | Status | Location / notes |
|---|---|---|---|
| 4 | Background: describe problem, existing models, gap | R | ¶1–2: deployment gap, transport failure common but failure not decomposed, no validated repair strategy |
| 5 | Objectives: state the study aim and whether development and/or external validation | R | ¶3: diagnose→attribute→repair→validate cycle across three EHR systems |

## Section 3: Methods

| Item | Recommendation | Status | Location / notes |
|---|---|---|---|
| 6a | Source of data; describe each cohort and whether used for development or validation | R | MIMIC-IV v3.1 (development), MIMIC-III v1.4 (external, cross-version), eICU-CRD v2.0 (external, 208 hospitals). M1 |
| 6b | Eligibility criteria and flow (inclusions/exclusions) | R | age≥18, LOS>24h; exclusion of prevalent / ICD-only / <24h / no-feature; flow in Supplement S-Fig1 / Table 1. M1 |
| 7 | Prediction target/outcome; definition; timing of outcome relative to predictors | R | HA-VTE >24h after admission (MIMIC radiology-confirmed via report text; eICU ICD-9); prevalent & ICD-only excluded; consistent first-24h landmark. M2 |
| 8 | Predictors: all candidate predictors at analysis start, with definitions and timing | R | 56 modelled features from a 57-entry candidate set minus the first-care-unit indicator (58-row S-Tab1 dictionary including coverage-count metadata), first-24h. Full dictionary Supplement S-Tab1. M3 |
| 9 | Sample size justification / events per variable | D | EPV = events/56 features: development 24.8 (1,390/56), de-overlapped repair 18.5 (1,035/56), naive pooled 24.5 (1,373/56), all ≥ conventional 10; external validation powered via cluster bootstrap (208 hospitals, 511 events). Development cohort: 416,638 admissions / 1,623 events (operative v2 labels). M7/M9 |
| 10 | Missing data handling | R | median imputation (LR), native NaN (XGB); missingness reported by feature (BMI/WBC coverage drift). M3/M6 |
| 11 | Development of model: how predictors handled (functional form, interactions, shrinkage) | R | LR (additive, L2) + XGBoost (depth-3 trees, fixed hyperparameters, no feature selection). M4 |
| 12a | Development: training vs test separation; how validation done | R | patient-clustered 15-fold CV (MIMIC-IV OOF); hyperparameters fixed a priori (no tuning on test). M5 |
| 12b | Internal validation results | R | XGB OOF AUC 0.828 (0.818–0.839), LR 0.809 (0.797–0.820); calibration ECE; DCA. ICU-level subgroup aligned with the eICU unit of analysis (67,803 admissions / 794 events): XGB OOF AUC 0.749, LR 0.726, Padua 0.575 (AUPRC 0.040/0.034/0.015; Brier 0.0114/0.0115/0.0116). M6 |
| 13 | External validation: how participants selected; same or different setting | R | MIMIC-III (same institution, earlier era) and eICU (208 hospitals, different EHR); explicit transportability framing. R2 |
| 14 | Statistical methods for model performance (discrimination, calibration, reclassification, net benefit) | R | AUC via Mann–Whitney rank statistic (formula in Methods; tie-corrected); AUPRC, ΔAUC with cluster bootstrap CI + two-sided MC p (paired draws for training-pool contrasts; p=min(1,2·min(P(d≤0),P(d≥0))) with 2/(B+1) floor); ECE/isotonic calibration (5-fold hospital-grouped cross-fitted); NRI(1%), IDI; DCA; negative downsampling NEG_CAP=50,000 (seeds 42/12,345). M3/M4/M7 |
| 15 | Intended use and audience of the model | R | risk-stratification of ICU/admitted patients for VTE prophylaxis; decision context vs Padua/IMPROVE. M8/D4 |
| 16 | Versioning / documentation of model | R | model files + `feature_order.json` + imputer versioned in `eicu_models/`; code + artifacts available to reviewers via private link at submission, public upon publication. Back matter |
| 17 | Ethical approval, data use agreement, reporting guidance | R | PhysioNet credentialed access, de-identified data, IRB exemption; TRIPOD+AI + CLAIM referenced. M8 |

## Section 4: Results

| Item | Recommendation | Status | Location / notes |
|---|---|---|---|
| 18 | Number of participants/events at each stage (development & external); flow diagram | R | MIMIC-IV 416,638 / 1,623 (0.390%); MIMIC-III 46,549 / 598; eICU 154,948 / 511. Table 1, S-Fig1 |
| 19 | Baseline characteristics by outcome | R | Table 1: age, sex, ICU%, LOS, event rate, hospitals per dataset |
| 20a | Model performance in development (discrimination, calibration) with uncertainty | R | R1: XGB 0.828 (CI), LR 0.809 (CI), ECE, DCA; calibration plots Supplement |
| 20b | Model performance in external validation (discrimination, calibration) with uncertainty | R | R6: pooled XGB eICU AUC 0.693 (0.667–0.718); Δ vs Padua +0.043; hospital-clustered CI/p; AUPRC 0.0079 vs Padua 0.0072; calibration main text + Supplement (raw ECE 0.218 / Brier 0.00347; post-recalibration reported from 5-fold hospital-grouped cross-fitting: OOF ECE 8.0×10⁻⁷ / Brier 0.00328) |
| 21 | Model update / revision if applicable (repair) | R | R5–R7 + construct-probe: de-overlapped cross-version pooling repair; paired contrasts vs single-source pools (III-only +0.090 p<0.0002; IV-2014+-only +0.024 p=0.055; naive-pooled +0.022 p=0.003; reverse-combination −0.054 p<0.0002); construct-consistency probe (IV-2014+ → MIMIC-III Δ+0.002, p=0.86); exploratory status + designation timeline disclosed (M7) |
| 22 | Net benefit / decision curve analysis if used | R | DCA in development and on eICU (net benefit 0.1%–5%, cluster-bootstrap at 1%): raw transported model negative NB at 1% (−0.0028); cross-fitted-recalibrated ≈ calibrated Padua ≈ 0. NRI(1%) +0.188/IDI +0.0062 reported as ranking (not decision-relevant) quantities on uncalibrated probabilities. M7/D4, S-Fig3 |

## Section 5: Discussion

| Item | Recommendation | Status | Location / notes |
|---|---|---|---|
| 23 | Clinical implications; comparison with existing models | R | D4: vs Padua/IMPROVE; modest absolute AUC; prophylaxis decision range |
| 24 | Limitations: transportability, missing data, outcome definition, generalizability | R | D5: ICD outcome (with §S2 quantitative bias analysis), single-institution cross-version, era-level de-overlap, retrospective, mobility coding, mechanism inseparability (era vs case-mix), structured-field score approximation fidelity |
| 25 | Interpretation: balanced, avoiding overclaiming; evidence of external validity | R | R6/D1: "diagnosable and repairable"; conditional advantage; no causal inference on mechanism |

## Section 6: Other information

| Item | Recommendation | Status | Location / notes |
|---|---|---|---|
| 26 | Supplementary info: predictor dictionary, hyperparameters, calibration, code | A | Supplement: feature dictionary, fold protocol, calibration plots, robustness, TRIPOD+AI |
| 27 | Funding, competing interests, data availability, authorship, AI-use disclosure | R | Back matter: PhysioNet data availability, GitHub code, CRediT, COI, funding, AI-assistance disclosure |

---

## Model Specification Box (TRIPOD+AI Item 2)

**Model type**: Logistic regression (additive, L2, C=1.0) and gradient-boosted trees (XGBoost, depth-3, 300 trees, lr=0.05, subsample/colsample=0.8, min_child_weight=5, λ=1).

**Target population**: Adults (≥18 y) admitted to hospital (MIMIC-IV) / ICU stays (eICU) with LOS>24h, at risk of hospital-acquired VTE.

**Outcome**: Hospital-acquired VTE (PE and/or DVT) occurring >24h after admission. MIMIC-III/IV: radiology-confirmed from report text (CTPA/VQ for PE; duplex/CTV for DVT), human-adjudicated subset (κ=1.0). eICU: ICD-9 codes (415.1x; 453.4x/453.8/453.9/451.x) with rule-out/prior-history exclusions.

**Time horizon / index**: Predictions at first-24h landmark; features from admission to +24h.

**Inputs**: 58 routinely available structured EHR features (see Supplement S-Tab1).

**Predicted**: probability of HA-VTE (continuous risk; thresholds explored at 0.5% and 1%).

**Comparison**: the model's frozen external performance vs (a) locally retrained model, (b) Padua and IMPROVE scores computed from the same structured data, (c) single-source training pools.

**Intended use**: pre-deployment transportability audit and risk-stratification; not a stand-alone diagnostic.
