# POAG PGS + ML — Analysis Code

**Multimodal Prediction of Primary Open-Angle Glaucoma Using Polygenic Risk
Scores and Clinical Features in a High-Risk African Ancestry Cohort**
Yan Zhu, Aude Benigne Ikuzwe Sindikubwabo, Yuki Bradford, et al.
*iScience* — manuscript ISCIENCE-D-26-03991 (new revision)

---

## What this code does

Evaluates four polygenic risk scores for primary open-angle glaucoma in
individuals of African ancestry, alone and added to a demographic baseline,
across four classifiers, with external validation in the Penn Medicine
BioBank.

The headline result is a negative one, and the code is written to make it
checkable: a baseline of age and sex is already strongly predictive, and the
polygenic increment is not distinguishable from zero for the primary model in
either the training cohort or the external cohort, nor does it yield
additional cases detected at fixed specificity.

## Corrections made on 2026-09-05

Two errors in the external validation were found before resubmission and
are corrected in every script here (`poag_corrections.py`):

1. `PGS616` in the POAAGG workbooks is the untransformed PLINK score sum,
   while the PMBB scores are rank-based inverse normal transformed. The
   standardization fitted in POAAGG therefore placed PMBB participants about
   four standard deviations above the training mean. `int_pgs616` now
   transforms PGS616 across the 1,284 POAAGG training and suspect
   participants, as the other three scores already were.
2. The stated PMBB restriction to age >= 35 years had not been applied.
   `restrict_pmbb_age` applies it (9,817 -> 9,084 participants).

With both corrected, the primary model's external increment is -0.001
(DeLong p = 0.55), where the uncorrected analysis gave +0.010 (p < 0.001).
Training-cohort results change only in the third decimal.

## Further changes made on 2026-09-10

* **Inter-eye differences.** Two training participants had a cup-to-disc
  ratio recorded for one eye only, and the stored difference equalled that
  eye's value. `poag_corrections.recompute_training_deltas` recomputes both
  differences from the two eyes (missing when either eye is missing, then
  imputed inside the pipeline), as was already done for the suspect and PMBB
  cohorts. Used by scripts 04, 18 and 42; only the delta-CDR results move.
* **Paired external contrasts.** Script 07 now also compares Base+PC5+PGS526
  and Base+PC5+PGS616 with Base in PMBB, and script 42 gives the paired
  DeLong and bootstrap increment of each score over the phenotype-only
  asymmetry model for every classifier (Table S18, panel B).
* **Shared-variant sensitivity analysis** (`60_common_variant_sensitivity.py`).
  The scores were computed from 568 / 486 variants in POAAGG and 539 / 460 in
  PMBB. Script 60 rescores both cohorts from the PLINK binary files on the
  525 / 448 variants scored in both and repeats the external comparison; the
  primary model's result is unchanged (delta-AUC +0.002, DeLong p = 0.65).
* Table S6 external increments are computed before rounding (script 08).

## Statistical approach

Uncertainty is estimated by a **participant-level bootstrap**, not from
cross-validation folds. Repeated cross-validation is reported only as a
descriptive summary of model stability.

The reason matters. Fold-level estimates from 5-fold × 20-repeat CV are not
independent — the same 271 participants recur in every fold — so their
standard error understates the sampling variability of the cohort by roughly
an order of magnitude. In each of 2,000 bootstrap replicates, participants are
resampled with replacement, stratified on case/control status; the complete
pipeline is refitted on the in-bag sample and evaluated on the out-of-bag
participants. Every feature set and classifier within a replicate shares the
same in-bag and out-of-bag participants, so incremental comparisons are paired
by construction.

External validation uses the DeLong test for correlated ROC curves together
with participant-level bootstrap resampling of the validation cohort.

## Layout

```
.
├── 01_pgs_standalone.py                 Figure 2: standalone PGS performance
├── 02_training_external_validation.py   Figure 3: 12 feature sets, training + PMBB
├── 03_suspect_enrichment.py             Figure 4: enrichment in 1,013 suspects
├── 04_asymmetry_analysis.py             Figure 5: inter-eye asymmetry
├── 05_shap_calibration.py               Figure S3: SHAP, calibration
├── 06_learning_curves_sex_stratified.py Figures S4, S5
├── 07_delta_auc_paired.py               Figure S6, Table S7: incremental AUC, DeLong
├── 08_same_classifier_comparison.py     Table S6: within-classifier Base vs Base+PGS
├── 09_pgs_residualized_on_pc.py         Table S10: PGS residualized on PCs
├── 10_participant_bootstrap_ci.py       Table S8: bootstrap AUC and paired ΔAUC
├── 11_suspect_pgs_partial_association.py Table S17: suspect cohort, age/sex adjusted
├── 12_pgs_standalone_paired.py          Table S4: curated vs genome-wide, paired
├── 13_clinical_utility_pmbb.py          Table S16: sensitivity at fixed specificity
├── 14_pgs_construction_provenance.py    Table S3: variant accounting for PGS616/526
├── 15_leakage_safeguards.py             Table S20: leakage safeguards + live checks
├── 17_regenerate_figures.py             Figures 2C, 3A, 3B, S2B from saved replicates
├── 18_secondary_bootstraps.py           Table S19: sex-stratified, asymmetry, residualized
├── 21_compose_figure_S7.py              Figure S6, two-panel forest plot
├── 24_figure_S2_and_captions.py         Figure S2, both panels
├── 27_rebuild_main_figures.py           Composites bootstrap panels into Figures 2, 3, 5
├── 41_rebuild_external_panels.py        Redraws Figures 2D, 3C, 5C (run after 27)
├── 42_figure5_asymmetry_external.py     Figure 5C / Table S18 external asymmetry AUCs + paired contrasts
├── 60_common_variant_sensitivity.py     Table S3: rescoring on variants scored in both cohorts
├── poag_corrections.py                  PGS616 transform, PMBB age restriction, inter-eye differences
├── poag_paths.py                        data location (POAG_DATA_DIR)
├── data/                                controlled access — see data/README.md
└── outputs/                             tables/ and figures/ written here
```

## Running

```bash
pip install -r requirements.txt
export POAG_DATA_DIR=/path/to/cohort/data     # see data/README.md
python 10_participant_bootstrap_ci.py 2000 -1  # B, n_jobs
```

Scripts write to `outputs/tables/` and `outputs/figures/`. Script 10 saves all
104,000 bootstrap replicates to
`outputs/tables/bootstrap_replicates_training.csv.gz`, so figures can be
redrawn (script 17) without refitting anything.

Runtime: the full bootstrap is about 30 minutes on 16 cores; the secondary
bootstrap (script 18) about the same. Everything else runs in minutes.

Order used for the paper: 01-06, 07, 08, 09, 10, 11-15, 18, 42, 60, then
the figure scripts 17, 21, 24, 27 and 41 (41 after 27). Scripts 27 and 41
paste redrawn panels into the published figure images, because panels 2A
and 5A exist only as images; they expect those images in
`$POAG_PROJECT_DIR/figures/`. All other scripts need only the data.

## Reproducibility

* Seed 42 throughout; classifier settings are fixed literals.
* **No hyperparameter search of any kind was performed.** Settings were fixed
  a priori and held constant across every feature set, cohort and analysis;
  script 15 verifies this by scanning the code for search constructs and by
  confirming that all model-fitting scripts carry identical classifier
  configurations.
* Results reported in this revision were produced with Python 3.13.7 and
  the package versions pinned in `requirements.txt`; other versions can
  differ in the last decimals.
* Preprocessing (median imputation, z-score scaling) sits inside the
  scikit-learn pipeline, so it is refitted within every fold and every
  bootstrap replicate.

## Data

Individual-level data are under controlled access and are not distributed
here. See `data/README.md` for accessions and the expected folder layout.

## Citation

Please cite the paper. `v0.1-submission` corresponds to the original
submission; `v3-revision` corresponds to the revised manuscript
(ISCIENCE-D-26-03991, September 2026).
