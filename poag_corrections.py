# =============================================================
#  POAG — Two corrections to the analysis inputs
#  iScience revision, 2026-09-05
#
#  Both were found in the 2026-09-05 audit of the 0904 submission
#  package and both change reported results, so they are applied here
#  explicitly rather than by editing the source data files. Every
#  analysis script calls these immediately after loading, so the
#  correction is visible in the code path and the raw inputs stay
#  untouched.
#
#  ---------------------------------------------------------------
#  CORRECTION 1 — PGS616 was never rank-INT transformed
#  ---------------------------------------------------------------
#  The POAAGG workbook carries four scores. Three of them (POAAGG PGS,
#  MEGA PGS, PGS526) are rank-based inverse-normal transformed within
#  the combined 1,284 POAAGG participants (271 training + 1,013
#  suspects). This is verifiable from the data alone: pooled over the
#  1,284, PGS526 has mean 0.0003, SD 0.9998 and runs from -3.3604 to
#  +3.3604, against the exact theoretical INT bounds for n = 1,284 of
#  +/-3.3602, with skew 0.000 and excess kurtosis -0.02. The heavily
#  tied POAAGG/MEGA scores show the same signature with tied ranks
#  averaged.
#
#  PGS616 does not. It is still the raw PLINK weighted sum: mean
#  -7.1488, SD 1.7347 in the training cohort. That is the same scale as
#  the PMBB SCORE1_SUM column (mean -6.04, SD 1.51), not the scale of
#  the other three scores.
#
#  This matters because the external validation feeds PMBB's
#  SCORE1_AVG_STD into models fitted on the raw training values.
#  SCORE1_AVG_STD is itself an exact rank-INT of SCORE1_AVG (Pearson r
#  = 1.000000 against a recomputed rank-INT, maximum absolute
#  difference 0.0000), computed within all 56,357 genotyped PMBB
#  samples. The StandardScaler fitted on the training cohort therefore
#  maps every PMBB participant to +4.40 +/- 0.49 SD, with only 10.4% of
#  PMBB values falling inside the training range at all. The PGS
#  feature reaching the external models was not the feature they were
#  trained on.
#
#  The fix restores the transform the other three scores already carry,
#  on the same 1,284 participants. It is monotone but not linear, so it
#  barely moves the training cohort (5x20 CV mean AUC changes by at
#  most 0.0023, against bootstrap intervals 0.14 wide) and shifts the
#  suspect-cohort partial correlations only in the third decimal (CDR
#  0.023 -> 0.025, RNFL 0.034 -> 0.032, IOP 0.029 -> 0.029), leaving
#  every one of them null. It changes the external results materially,
#  because that is where the mismatch was.
#
#  NOTE ON SCOPE. The transform is fitted on training and suspect
#  participants pooled, because that is what was done for the other
#  three scores and the paper already discloses that pooling for the
#  ancestry principal components. It uses no case-control labels. It is
#  not refitted inside bootstrap replicates; the Limitations should say
#  so.
#
#  ---------------------------------------------------------------
#  CORRECTION 2 — the PMBB age restriction was never applied
#  ---------------------------------------------------------------
#  STAR Methods states twice that PMBB participants "aged >= 35 years"
#  were included. No script applied that filter. The analysed set
#  contained 733 participants below 35 (minimum age 24.9), including 2
#  POAG cases. Applying the stated criterion gives N = 9,084 with 168
#  cases and 8,916 controls, against the N = 9,817 / 170 cases reported
#  throughout the 0904 package.
#
#  PMBB_3.0_Release_AGE is the age at data release, which is the only
#  age field available in the supplied phenotype file; the manuscript
#  should say which age it means.
# =============================================================

import numpy as np
from scipy import stats

PMBB_MIN_AGE = 35.0
PMBB_AGE_COL = "PMBB_3.0_Release_AGE"


def int_pgs616(tr, su, col="PGS616", verbose=True):
    """Rank-INT PGS616 across the pooled POAAGG participants.

    Takes the training and suspect frames, returns them with `col`
    replaced by its rank-based inverse-normal transform computed on the
    two pooled — the same construction the other three scores carry.
    Frames are copied, not modified in place.
    """
    tr, su = tr.copy(), su.copy()
    pooled = np.concatenate([tr[col].to_numpy(float), su[col].to_numpy(float)])
    if np.isnan(pooled).any():
        raise ValueError(f"{col} contains NaN; the INT would be ill-defined")
    z = stats.norm.ppf((stats.rankdata(pooled) - 0.5) / len(pooled))
    tr[col], su[col] = z[:len(tr)], z[len(tr):]
    if verbose:
        print(f"  [correction 1] {col} rank-INT over {len(pooled):,} pooled "
              f"POAAGG participants: training mean {tr[col].mean():+.4f} "
              f"SD {tr[col].std():.4f}", flush=True)
    return tr, su


def int_pgs616_training_only(tr, col="PGS616", verbose=True):
    """As `int_pgs616`, for the scripts that never load the suspect frame.

    Reads the suspect workbook itself so the transform is fitted on the
    same 1,284 participants either way — a transform fitted on the 271
    alone would not match the other three scores.
    """
    import os as _os
    import pandas as pd
    from poag_paths import SUSPECT_FILE
    su = pd.read_excel(SUSPECT_FILE)
    tr2, _ = int_pgs616(tr, su, col=col, verbose=verbose)
    return tr2


def restrict_pmbb_age(pmbb, min_age=PMBB_MIN_AGE, age_col=PMBB_AGE_COL,
                      verbose=True):
    """Apply the >= min_age criterion the manuscript states."""
    before = len(pmbb)
    out = pmbb[pmbb[age_col] >= min_age].copy()
    if verbose:
        dropped = before - len(out)
        print(f"  [correction 2] PMBB age >= {min_age:g}: {before:,} -> "
              f"{len(out):,} ({dropped:,} dropped)", flush=True)
    return out


# ---------------------------------------------------------------------
#  Correction 3 (2026-09-10 audit, B02): training inter-eye differences
# ---------------------------------------------------------------------
def recompute_training_deltas(tr):
    """Recompute delta_IOP / delta_CDR in the training data as |OD - OS|.

    The stored delta_CDR equals the single available eye in two cases whose
    fellow-eye CDR is missing (the missing eye had been treated as 0). The
    suspect and PMBB cohorts compute the differences from both eyes; the
    training cohort now does the same, leaving the difference missing when
    either eye is missing (it is then median-imputed inside the pipeline).
    """
    tr = tr.copy()
    pairs = {"delta_IOP": ("OD Baseline IOP (mmHg)", "OS Baseline IOP (mmHg)"),
             "delta_CDR": ("OD Baseline CDR", "OS Baseline CDR")}
    for col, (od, os_) in pairs.items():
        new = (tr[od] - tr[os_]).abs()
        changed = int(((new != tr[col]) & ~(new.isna() & tr[col].isna()))
                      .sum())
        tr[col] = new
        print(f"  [correction 3] {col} recomputed from both eyes "
              f"({changed} value(s) changed)", flush=True)
    return tr
