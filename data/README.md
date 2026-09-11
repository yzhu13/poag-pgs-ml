# Input data

The individual-level data are under controlled access and are **not**
included here.

* POAAGG genotypes and phenotypes — dbGaP accession **phs001312**
* Penn Medicine BioBank — dbGaP accession **phs001453**
* De-identified analysis datasets — from the lead contact under a data use
  agreement

## Expected layout

Set `POAG_DATA_DIR` to the folder containing the cohort subdirectories, or
place them in `./data` next to the scripts. Scripts 01-06 and 42 take the
location from `poag_paths.py`; 07-18 carry the same default.

```
data/
├── POAAGG_cohort/
│   ├── 271_training_cohort_4_new_PRS_cleaned.xlsx
│   └── 1013_testing_cohort_only_suspect_cleaned.xlsx
└── PMBB_external/
    ├── PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_updated_June8.csv
    ├── PMBB_949_POAG_IOP_CDR_Freeze3.csv
    └── PMBBv3_GRS_{MEGA_616,QUANT_526}snps_AllSamples.sscore_withSTDscore.txt
```

`PGS616` in the POAAGG workbooks is the untransformed PLINK score sum. The
analysis scripts transform it on load (`poag_corrections.int_pgs616`), as
described in the paper; the other three scores and the PMBB scores are
stored already transformed.

## Genotype-level files (scripts 14 and 60 only)

The shared-variant sensitivity analysis (60) rescores both cohorts from
PLINK 1 binary files; the provenance table (14) checks the weight files.

```
data/genotypes/
├── POAAGG/    GRS_MainTable1_572snps.{bed,bim,fam}, GRS_{MEGA,MTAG}_MainTable1_572snps.sscore,
│              GRS_weight_{MEGA,QUANT}_572snps.txt
├── PMBB/      PMBB_GRS_572snps.{bed,bim,fam}, PMBBv3_GRS_MEGA_572snps_AllSamples.sscore_withSTDscore.txt
└── weights/   GRS_weight_{MEGA,QUANT}_MainTable1_572snps.txt
```

Script 14 reads the weight files from `POAG_PROJECT_DIR`.
