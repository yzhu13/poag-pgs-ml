# Data Files

Raw data are **not included** in this repository due to IRB and dbGaP restrictions.

## Required files

Place files in the following locations (relative to the repo root):

### `data/poaagg/`

| File | Description |
|------|-------------|
| `271_training_cohort_4_new_PRS_cleaned.xlsx` | POAAGG training cohort (N=271; 128 cases, 143 controls). Contains Age, Gender, PC1–PC10, POAAGG PGS, MEGA PGS, PGS526, PGS616, delta_IOP, delta_CDR, CaseCtrl label. |
| `1013_testing_cohort_only_suspect_cleaned.xlsx` | POAAGG suspect cohort (N=1,013). Contains same features plus IOP_SEVERE, CDR_SEVERE, RNFL_SEVERE for enrichment analysis. |

**Data access:** dbGaP accession [phs001312](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001312)

---

### `data/pmbb/`

| File | Description |
|------|-------------|
| `PMBB_3.0_pheno_covars_noPOAAGG.csv` | Penn Medicine BioBank Release 3.0 phenotype + covariates for African ancestry individuals, with POAAGG participants removed (N=9,817; 170 POAG cases, 9,647 controls). Columns: PMBB_ID, POAG_cases, PMBB_3.0_Release_AGE, SEX, ANCESTRY, PC1–PC6. |
| `PMBBv3_GRS_MEGA_616snps_AllSamples.sscore_withSTDscore.txt` | PGS616 scores for all PMBB samples (PLINK2 `.sscore` format). Key column: SCORE1_AVG_STD (standardized score). |
| `PMBBv3_GRS_QUANT_526snps_AllSamples.sscore_withSTDscore.txt` | PGS526 scores for all PMBB samples (same format). |
| `PMBB_949_POAG_IOP_CDR_Freeze3.csv` | PMBB IOP and CDR longitudinal data for asymmetry analysis (script `04_asymmetry_analysis.py` only). Columns: PMBB_ID, pheno_eye (OD/OS), pheno_type (IOP/CDR), pheno_value, pheno_date. |

**Data access:** [Penn Medicine BioBank](https://pmbb.med.upenn.edu/) (PMID: 36556195) — requires a data-use agreement.

---

## PGS SNP weights

All four PGS (POAAGG PGS, MEGA PGS, PGS526, PGS616) are custom-built.  
SNP lists and African ancestry effect sizes are deposited in:
- Supplemental Table S2 of the manuscript
- [Mendeley Data DOI: 10.17632/9rjn45y65c.1](https://doi.org/10.17632/9rjn45y65c.1)
