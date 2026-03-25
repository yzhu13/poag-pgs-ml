# Data

Raw genotype and phenotype data are not publicly available due to IRB and
dbGaP data-use restrictions.

POAAGG genotype data are available via dbGaP accession **phs001312**.
Researchers with an approved data-use agreement may request access and
reproduce all analyses using the scripts provided in this repository.

## Expected input files

Place the following files in this `data/` directory before running the
analysis scripts. File names can be updated in `config.py`.

| File | Description |
|------|-------------|
| `271_training_cohort.xlsx` | Training cohort (271 samples; cases and controls) |
| `1088_testing_cohort.xlsx` | Testing cohort (1088 samples; cases, controls, suspects) |
| `1013_testing_suspects.xlsx` | Glaucoma suspects subset of the testing cohort (1013 samples) |
| `MTAG_PRS_526_loci.txt` | PRS526 — MTAG-derived SNP weights (columns: SNP, BETA) |
| `MEGA_PRS_616_loci.txt` | PRS616 — MEGA-array SNP weights (columns: SNP, BETA) |

## Required columns

**Training / testing cohorts** must include the following columns
(names can be adjusted in `config.py`):

| Column | Description |
|--------|-------------|
| `CaseCtrl` | Binary label: 1 = POAG case, 0 = control (training cohort) |
| `Disease status` | String label: "Case", "Control", or "Suspect" (testing cohort) |
| `Age` | Age at enrollment |
| `Gender` | Sex (encoded numerically) |
| `POAAGG PRS` | POAAGG cohort-derived polygenic risk score |
| `MEGA PRS` | MEGA-array polygenic risk score |
| `PRS526` | MTAG-based PRS computed from 526 loci |
| `PRS616` | MEGA-based PRS computed from 616 loci |
| `delta_IOP` | Inter-eye IOP asymmetry (suspects file) |
| `delta_CDR` | Inter-eye CDR asymmetry (suspects file) |
| `delta_RNFL` | Inter-eye RNFL thickness asymmetry (suspects file) |
| `IOP_SEVERE` | Maximum IOP (clinical enrichment) |
| `CDR_SEVERE` | Maximum CDR (clinical enrichment) |
| `RNFL_SEVERE` | Minimum RNFL thickness (clinical enrichment) |
