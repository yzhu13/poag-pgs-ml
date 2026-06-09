# POAG PGS + ML — Analysis Code

Code repository for:

> **Multimodal Prediction of Primary Open-Angle Glaucoma Using Polygenic Risk
> Scores and Clinical Features in a High-Risk African Ancestry Cohort**
> Yan Zhu, Aude Benigne Ikuzwe Sindikubwabo, Yuki Bradford, et al.
> *iScience*, 2026.

---

## Overview

This repository contains the Python analysis pipeline used to train and
evaluate machine learning models for POAG risk prediction. The pipeline
integrates ancestry-matched polygenic risk scores (PGS) with demographic and
clinical features in the POAAGG cohort (African ancestry).

---

## Repository structure

```
.
├── config.py                    # Paths, column names, feature sets, model hyperparameters
├── utils.py                     # Shared helper functions (pipeline builder, bootstrap CV, plots)
├── 01_main_analysis.py          # Main pipeline: training, external validation, enrichment, figures
├── 02_asymmetry_analysis.py     # Asymmetry analysis: δIOP/δCDR + PGS, Figure 5
├── requirements.txt
├── data/
│   └── README.md                # Data access instructions and expected column names
├── notebooks/
│   ├── 2026_1_9_paper_271_training_1088_testing.ipynb   # Original analysis notebook
│   └── 2026_1_10_Asymmetry_Training_and_Validation.ipynb
└── revision/                    # Scripts added for iScience major revision (R2, June 2026)
    ├── analysis/                # Core analysis scripts → produce Excel output tables
    │   ├── figure2_pgs_standalone.py         Standalone PGS performance (Fig 2D)
    │   ├── figure3_training_external_val.py  5×20 CV training + PMBB external validation (Fig 3)
    │   ├── figure3_post_cv.py                Post-CV summary from saved results
    │   ├── figure4_suspect_enrichment.py     Clinical enrichment in 1,013 suspects (Fig 4)
    │   ├── figure5_asymmetry.py              Inter-eye asymmetry analysis (Fig 5)
    │   ├── figure5_asymmetry_sep.py          Asymmetry separate ΔIOP/ΔCDR runs
    │   ├── sf2_pc_confounding.py             PC–PGS correlation analysis (Fig S2)
    │   ├── sf3_shap_calibration.py           SHAP + calibration curves (Fig S3)
    │   └── sf4_sf5_learning_curves_sex.py    Learning curves + sex-stratified AUC (Fig S4–S5)
    └── figures/                 # Figure assembly scripts → render PNG/PDF panels
        ├── build_figure3_panels.py
        ├── build_figure5_panels.py
        ├── build_sf2_pc_confounding.py
        ├── build_supplemental_s1.py
        └── compile_supplemental_tables.py
```

---

## Setup

### 1. Install dependencies

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

### 2. Place input data

Copy the required Excel/text files into the `data/` folder (see `data/README.md`
for the full list). Raw data are available via dbGaP accession **phs001312**.

### 3. Edit paths in `config.py` (if needed)

The default configuration expects data files in `data/` and writes all outputs
to `outputs/`. Both paths can be changed at the top of `config.py`.

---

## Running the analyses

### Main pipeline (Tables 2–3, Figures 2–4)

```bash
python 01_main_analysis.py
```

Produces:
- `outputs/bootstrap_cv_results.xlsx` — bootstrap CV metrics (Table 2)
- `outputs/external_validation_results.xlsx` — external test metrics (Table 3)
- `outputs/clinical_enrichment_correlations.xlsx` — Pearson/Spearman r with IOP/CDR/RNFL
- `outputs/predicted_risks_suspects.xlsx` — per-suspect predicted POAG probability
- `outputs/figures/auc_*.png` — AUC bar charts (Figures 2–3)
- `outputs/figures/enrichment_*.png` — clinical enrichment scatter plots (Figure 4)
- `outputs/figures/violin_*.png` — PRS distributions by disease status (Figure 2)

### Asymmetry analysis (Figure 5)

```bash
python 02_asymmetry_analysis.py
```

Produces:
- `outputs/asymmetry_bootstrap_cv_results.xlsx` — bootstrap CV for asymmetry features
- `outputs/suspect_risks_{model}.xlsx` — per-suspect risk using asymmetry + PRS616
- `outputs/asymmetry_validation_stats.xlsx` — Pearson r and KS test statistics
- `outputs/figures/asymmetry_trend_*.png` — binned risk vs |Δ| trend plots
- `outputs/figures/asymmetry_ecdf_*.png` — ECDF top vs bottom 25% risk
- `outputs/figures/auc_asymmetry_*.png` — AUC bar charts for asymmetry feature sets

---

## Models

Four classifiers are used, each wrapped in an impute → scale → classify
`sklearn.Pipeline`:

| Model | Key hyperparameters |
|-------|---------------------|
| Random Forest | 300 trees, max_depth=5, balanced_subsample |
| MLP | (64, 32) hidden layers, ReLU, Adam, early stopping |
| SVM | RBF kernel, C=4, γ=2, balanced class weights |
| Logistic Regression | L2, liblinear solver, balanced class weights |

---

## Feature sets

**Main analysis** — 12 feature sets combining Age, Sex, PCs, and four PGS:

| PGS | Type | Source |
|-----|------|--------|
| POAAGG PGS | Genome-wide (PRS-CS) | POAAGG cohort GWAS (N=7,031) |
| MEGA PGS | Genome-wide (PRS-CS) | MEGA multi-cohort African ancestry GWAS (N=11,275) |
| PGS526 | Curated loci-based | 526 loci from 6 multi-ancestry GWAS, African ancestry weights |
| PGS616 | Curated loci-based | 616 loci from 6 multi-ancestry GWAS, African ancestry weights |

**Asymmetry analysis** — 5 feature sets: ΔIOP only, ΔCDR only, each + PGS526, each + PGS616.
ΔRNFL excluded to ensure consistency across all three cohorts (POAAGG, suspects, PMBB).

---

## Software versions

Analyses were performed with:

- Python 3.11
- scikit-learn 1.3
- pandas 2.x
- numpy 1.x
- scipy 1.10
- matplotlib / seaborn

PRS weights were computed with **PRS-CS** using an African ancestry LD
reference panel. Genotype processing used **PLINK 2.0**.

---

## Key results (R2 submission, June 2026)

### Training cohort (POAAGG; N = 271; 5×20 stratified CV)

| Feature set | Mean AUC | 95% CI |
|-------------|----------|--------|
| Base (Age + Sex) | 0.683 | 0.669–0.697 |
| Base + PGS526 | 0.696 | 0.682–0.709 |
| Base + PGS616 | 0.700 | 0.687–0.713 |
| Best single model (MLP, Base+PGS616) | 0.713 | — |

### External validation (PMBB AFR; N = 9,817; 170 cases, 9,647 controls)

| Feature set | Mean AUC |
|-------------|----------|
| Base | 0.728 |
| Base + PGS526 | 0.735 |
| **Best: MLP + Base + PGS616** | **0.752 (95% CI: 0.723–0.780)** |

### Asymmetry (PMBB; N = 2,786; 120 cases)

| Feature set | External AUC | 95% CI |
|-------------|-------------|--------|
| ΔCDR + PGS616 | 0.696 | 0.644–0.748 |
| ΔIOP + PGS616 | 0.588 | 0.542–0.635 |

---

## Data availability

Raw genotype and phenotype data are not publicly available due to IRB and
dbGaP restrictions. POAAGG data are available via dbGaP accession
[phs001312](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001312)
to researchers with an approved data-use agreement.  
PMBB data: [Penn Medicine BioBank](https://pmbb.med.upenn.edu/) (PMID: 36556195).  
PGS SNP weights are deposited in Supplemental Table S2 and
[Mendeley Data](https://doi.org/10.17632/9rjn45y65c.1).

---

## License

MIT License — see `LICENSE`.
