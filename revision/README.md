# Revision Analysis Scripts

Code for the **iScience major revision** (ISCIENCE-D-26-03991R2, submitted June 2026).

All scripts were run in Python 3.11 on Windows.  
Data files are not included (dbGaP restricted); see [Data files](#data-files) below.

---

## Folder structure

```
revision/
├── analysis/          Core analysis scripts — produce Excel output tables
│   ├── figure2_pgs_standalone.py         Standalone PGS performance (5×20 CV; Fig 2D, Table S3)
│   ├── figure3_training_external_val.py  Training + PMBB external validation (Fig 3, Table S3/S4/S9)
│   ├── figure3_post_cv.py                Post-CV summary tables from saved CV results
│   ├── figure4_suspect_enrichment.py     Clinical enrichment in 1,013 suspects (Fig 4, Table S5–S7)
│   ├── figure5_asymmetry.py              Inter-eye asymmetry analysis (Fig 5, Table S10)
│   ├── figure5_asymmetry_sep.py          Asymmetry analysis — separate ΔIOP / ΔCDR runs
│   ├── sf2_pc_confounding.py             PC–PGS correlation analysis (Fig S2, Table S4)
│   ├── sf3_shap_calibration.py           SHAP feature importance + calibration (Fig S3, Table S5–S6)
│   └── sf4_sf5_learning_curves_sex.py    Learning curves + sex-stratified AUC (Fig S4–S5, Table S7–S8)
│
└── figures/           Figure assembly scripts — read Excel outputs, render PNG/PDF panels
    ├── build_figure3_panels.py           Assemble Figure 3 panels (3A all features, 3B dot, 3C PMBB)
    ├── build_figure5_panels.py           Assemble Figure 5 panels
    ├── build_figure5_panels_v2.py        Figure 5 with extended panel D (asymmetry heatmap)
    ├── build_sf2_pc_confounding.py       Build SF2 PC-confounding combined figure
    ├── build_supplemental_s1.py          Build Supplemental Document S1 (all supp figures)
    └── compile_supplemental_tables.py    Compile all supplemental tables into a single Excel workbook
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Set your project root

Each script has a `BASE` variable near the top.  
By default, it is auto-detected as two directory levels above the script
(i.e., the repo root). If your input data is elsewhere, uncomment and edit
the manual override line:

```python
# BASE = r"C:\your\path\here"   # Windows
# BASE = "/your/path/here"      # macOS / Linux
```

### 3. Place input data files

Expect the following subdirectory structure relative to `BASE`:

```
input-data/
  POAAGG_cohort/
    271_training_cohort_4_new_PRS_cleaned.xlsx
    1013_testing_cohort_only_suspect_cleaned.xlsx
  PMBB_external/
    PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_updated_June8.csv
    PMBBv3_GRS_MEGA_616snps_AllSamples.sscore_withSTDscore.txt
    PMBBv3_GRS_QUANT_526snps_AllSamples.sscore_withSTDscore.txt
    PMBB_949_POAG_IOP_CDR_Freeze3.csv
output excel data/   (created automatically by analysis scripts)
figure/              (created automatically by figure scripts)
```

---

## Data files

POAAGG genotype and phenotype data: dbGaP accession [phs001312](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001312)  
PMBB data: [Penn Medicine BioBank](https://pmbb.med.upenn.edu/) (PMID: 36556195) — requires a data-use agreement.

---

## Running the analyses

### Step 1 — Generate output tables (run in order)

```bash
# Training + external validation (required first; saves Excel for figure scripts)
python revision/analysis/figure3_training_external_val.py

# PGS standalone performance
python revision/analysis/figure2_pgs_standalone.py

# PC–PGS correlation (SF2)
python revision/analysis/sf2_pc_confounding.py

# SHAP + calibration (SF3)
python revision/analysis/sf3_shap_calibration.py

# Learning curves + sex-stratified (SF4, SF5)
python revision/analysis/sf4_sf5_learning_curves_sex.py

# Suspect enrichment (Figure 4)
python revision/analysis/figure4_suspect_enrichment.py

# Asymmetry analysis (Figure 5)
python revision/analysis/figure5_asymmetry.py
```

### Step 2 — Build figures

```bash
python revision/figures/build_figure3_panels.py
python revision/figures/build_figure5_panels.py
python revision/figures/build_sf2_pc_confounding.py
python revision/figures/compile_supplemental_tables.py
```

---

## Key results (R2 submission, June 2026)

### Training cohort (N = 271; 5×20 stratified CV)

| Feature set | Mean AUC | 95% CI |
|-------------|----------|--------|
| Base (Age + Sex) | 0.683 | 0.669–0.697 |
| Base + PGS526 | 0.696 | 0.682–0.709 |
| Base + PGS616 | 0.700 | 0.687–0.713 |
| Best single model (MLP, Base+PGS616) | 0.713 | — |

### PMBB external validation (AFR; N = 9,817; 170 cases, 9,647 controls)

| Feature set | Mean AUC | 95% CI |
|-------------|----------|--------|
| Base | 0.728 | — |
| Base + PGS526 | 0.735 | — |
| Base + PGS616 | 0.718 | — |
| **Best (MLP, Base+PGS616)** | **0.752** | **0.723–0.780** |

### Asymmetry analysis (PMBB; N = 2,786; 120 cases)

| Feature set | External AUC | 95% CI |
|-------------|-------------|--------|
| ΔCDR only | 0.670 | 0.620–0.720 |
| ΔCDR + PGS616 | 0.696 | 0.644–0.748 |
| ΔIOP + PGS616 | 0.588 | 0.542–0.635 |

---

## Dependencies

```
numpy pandas scikit-learn scipy shap matplotlib seaborn openpyxl python-docx
```

Install: `pip install -r ../requirements.txt`
