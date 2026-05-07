# Revision Analysis Scripts

Code added for the **iScience major revision** (ISCIENCE-D-26-03991, submitted 2026-05-28).

## Scripts

| File | Description |
|------|-------------|
| `01_revision_analyses.py` | Analyses 1–7: PC sensitivity, 10×10 repeated CV (all models × PGS combos), logistic regression baseline, SHAP feature importance, learning curves, calibration + Brier score, suspect-cohort disentanglement |
| `02_sex_stratified_figures.py` | Analysis 8: sex-stratified AUC; Figure 2A (SNP counts per chromosome); Figure 2B (PGS violin plots); DIOP/DRNFL/DCDR asymmetry AUC with LR primary model |
| `03_pmbb_external_validation.py` | External validation in Penn Medicine Biobank (PMBB) Release 3.0 AFR cohort: OR per SD, bootstrap AUC, DeLong test comparing PGS-augmented vs. Age+Sex baseline |

## Data files required

Place in `../input date/` (repo root `input date/` directory):

- `271_training_cohort_4_new_PRS_cleaned.xlsx` — POAAGG training cohort (N=271)
- `1013_testing_cohort_only_suspect_cleaned.xlsx` — POAAGG suspect cohort (N=1,013)
- `PMBB_AFR_ready_for_PGS.csv` — PMBB AFR phenotype file
- `PMBBv3_GRS_MEGA_616snps.sscore_withSTDscore.txt` — PGS616 scores (PLINK2 output)
- `PMBBv3_GRS_QUANT_526snps.sscore_withSTDscore.txt` — PGS526 scores (PLINK2 output)

## Key results (PMBB external validation)

| Model | AUC | 95% CI | DeLong p vs baseline |
|-------|-----|--------|----------------------|
| Age + Sex (baseline, N=2,594) | 0.690 | 0.660–0.719 | — |
| + PGS616 | 0.707 | 0.677–0.736 | 0.043 |
| + PGS526 | 0.702 | 0.672–0.733 | 0.077 |
| + PGS616 + PGS526 | 0.707 | 0.678–0.737 | 0.036 |
| Age + Sex, full PMBB AFR (N=12,113) | 0.731 | 0.708–0.752 | — |

PGS616 OR per SD: 1.39 (95% CI 1.22–1.58, p<0.0001)  
PGS526 OR per SD: 1.32 (95% CI 1.16–1.50, p<0.0001)

## Dependencies

```
numpy pandas scikit-learn statsmodels scipy shap matplotlib seaborn openpyxl
```

Install: `pip install -r ../requirements.txt`
