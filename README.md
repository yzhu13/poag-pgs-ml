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
├── config.py                 # All paths, column names, feature sets, model hyperparameters
├── utils.py                  # Shared helper functions (pipeline builder, bootstrap CV, plots)
├── 01_main_analysis.py       # Main pipeline: training, external validation, enrichment, figures
├── 02_asymmetry_analysis.py  # Asymmetry analysis: δIOP/δCDR/δRNFL + PGS, Figure 5
├── requirements.txt
└── data/
    └── README.md             # Data access instructions and expected column names
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

**Main analysis** — 14 feature sets combining Age, Gender, and four PGS:

| PGS | Source |
|-----|--------|
| POAAGG PRS | POAAGG cohort GWAS |
| MEGA PRS | MEGA-array GWAS |
| PRS526 | MTAG meta-analysis, 526 loci |
| PRS616 | MEGA-array, 616 loci |

**Asymmetry analysis** — 15 feature sets: each of δIOP / δCDR / δRNFL alone,
combined with each of the four PGS, plus δall + PRS616.

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

## Data availability

Raw genotype and phenotype data are not publicly available due to IRB and
dbGaP restrictions. POAAGG data are available via dbGaP accession **phs001312**
to researchers with an approved data-use agreement.

---

## License

MIT License — see `LICENSE`.
