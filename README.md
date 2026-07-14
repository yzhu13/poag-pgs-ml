# POAG PGS + ML — Analysis Code

**Multimodal Prediction of Primary Open-Angle Glaucoma Using Polygenic Risk Scores and Clinical Features in a High-Risk African Ancestry Cohort**  
Yan Zhu, Aude Benigne Ikuzwe Sindikubwabo, Yuki Bradford, et al.  
*iScience*, 2026 — Manuscript ISCIENCE-D-26-03991R2

---

## Overview

Python analysis pipeline for POAG risk prediction using:
- Four polygenic risk scores (PGS) matched to African ancestry
- Four ML classifiers (LR, RF, SVM, MLP)
- **5-fold × 20-repeat stratified cross-validation** (5×20 CV; 100 fits per configuration)
- External validation in the Penn Medicine BioBank (PMBB; N = 9,817 AFR)

---

## Repository structure

```
.
├── 01_pgs_standalone.py              Figure 2: standalone PGS performance (5×20 CV + PMBB)
├── 02_training_external_validation.py Figure 3: 12-feature-set training + PMBB external validation
├── 03_suspect_enrichment.py          Figure 4: clinical enrichment in 1,013 suspects
├── 04_asymmetry_analysis.py          Figure 5: inter-eye asymmetry (ΔIOP, ΔCDR) + PMBB
├── 05_shap_calibration.py            Fig S3: SHAP feature importance + calibration curves
├── 06_learning_curves_sex_stratified.py  Fig S4–S5: learning curves + sex-stratified AUC
├── 07_delta_auc_paired.py            Fig S7, Tables S12: paired ΔAUC (DeLong + bootstrap + CV-fold)
├── 08_same_classifier_comparison.py  Table S11: within-classifier Base vs Base+PGS AUC
├── 09_pgs_residualized_on_pc.py      Table S13: PGS residualized on ancestry PCs (sensitivity)
├── requirements.txt
└── data/
    ├── README.md                     Data access instructions
    ├── poaagg/                       Place POAAGG cohort files here
    └── pmbb/                         Place PMBB external validation files here
```

All Excel output tables are written to `outputs/tables/`  
All figures (PNG + PDF) are written to `outputs/figures/`

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11 recommended.

---

## Data

Raw data are not publicly available (IRB / dbGaP restricted). See [`data/README.md`](data/README.md) for full file list and access instructions.

PGS SNP weights: Supplemental Table S2 + [Mendeley Data](https://doi.org/10.17632/9rjn45y65c.1)

---

## Running the analyses

Run scripts **in order** — later scripts read Excel files saved by earlier ones:

```bash
# Step 1: Standalone PGS performance (saves outputs/tables/Table_Figure2_PGS_only.xlsx)
python 01_pgs_standalone.py

# Step 2: Training cohort + PMBB external validation (saves Table_Figure3_*.xlsx)
python 02_training_external_validation.py

# Step 3: Suspect cohort enrichment (saves Table_Figure4_Suspects.xlsx)
python 03_suspect_enrichment.py

# Step 4: Asymmetry analysis (saves Table_Figure5_Asymmetry.xlsx)
python 04_asymmetry_analysis.py

# Step 5: SHAP + calibration (saves Table_SF3_*.xlsx)
python 05_shap_calibration.py

# Step 6: Learning curves + sex-stratified AUC (saves Table_SF4_*.xlsx, Table_SF5_*.xlsx)
python 06_learning_curves_sex_stratified.py

# --- R3 revision analyses (incremental value of PGS beyond age+sex) ---
# Step 7: Paired ΔAUC — DeLong + bootstrap (PMBB) and paired CV-fold diffs (training)
#         → Table_DeltaAUC_*.xlsx, Figure_DeltaAUC_forest.{png,pdf}  (Table S12, Figure S7)
python 07_delta_auc_paired.py

# Step 8: Same-classifier AUC comparison, Base vs Base+PGS  → Table_SameClassifier_AUC.xlsx (Table S11)
python 08_same_classifier_comparison.py

# Step 9: PGS residualized on PC1–PC5 sensitivity  → Table_PGS_residualized_on_PC.xlsx (Table S13)
python 09_pgs_residualized_on_pc.py
```

---

## Key results (iScience R2 submission, June 2026)

### Training cohort (POAAGG; N = 271; 5×20 CV)

| Feature set | Mean AUC | 95% CI |
|---|---|---|
| Base (Age + Sex) | 0.683 | 0.669–0.697 |
| Base + PGS526 | 0.696 | 0.682–0.709 |
| Base + PGS616 | 0.700 | 0.687–0.713 |
| Best: MLP + Base + PGS616 | **0.713** | — |

### PMBB external validation (AFR; N = 9,817; 170 cases, 9,647 controls)

| Feature set | Mean AUC |
|---|---|
| Base | 0.728 |
| Base + PGS526 | 0.735 |
| **Best: MLP + Base + PGS616** | **0.752 (95% CI: 0.723–0.780)** |

### Asymmetry analysis (PMBB; N = 2,786 with IOP/CDR; 120 cases)

| Feature set | External AUC | 95% CI |
|---|---|---|
| ΔCDR + PGS616 | 0.696 | 0.644–0.748 |
| ΔIOP + PGS616 | 0.588 | 0.542–0.635 |

---

## Models

All classifiers implemented in `sklearn.Pipeline` (impute → scale → classify):

| Model | Key settings |
|---|---|
| Logistic Regression (LR) | L2, liblinear, C=1, balanced class weight |
| Random Forest (RF) | 200 trees, max_depth=5, balanced class weight |
| SVM | RBF kernel, balanced class weight, probability=True |
| MLP | hidden=(32,), max_iter=1000, random_state=42 |

---

## Feature sets (12 configurations)

| Set | Features |
|---|---|
| Age only | Age |
| Sex only | Gender |
| Base | Age + Gender |
| Base + PC2/PC5/PC10 | Base + ancestry PCs |
| Base + POAAGG PGS | Base + genome-wide PGS (PRS-CS, POAAGG cohort) |
| Base + MEGA PGS | Base + genome-wide PGS (PRS-CS, MEGA cohort) |
| Base + PGS526 | Base + curated 526-locus score |
| Base + PGS616 | Base + curated 616-locus score |
| Base + PC5 + PGS526/616 | Combined PC + PGS models |

---

## Software

- Python 3.11 · scikit-learn 1.3 · pandas 2.x · numpy 1.x · scipy 1.10
- shap · matplotlib · seaborn · openpyxl
- PGS computed with PLINK 2.0; POAAGG PGS via PRS-CS (African ancestry LD panel)

---

## License

MIT License
