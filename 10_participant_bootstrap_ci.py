# =============================================================
#  POAG — Participant-Level Bootstrap Confidence Intervals
#  iScience R4 revision (2026-09-04)
#
#  Addresses Editor point 1 and 6, and Reviewer #1 point 1:
#   "Recalculate cross-validation confidence intervals using
#    participant-level bootstrap or report repeated-cross-validation
#    variability descriptively without formal inferential claims."
#   "... These estimates are not independent because the same
#    participants repeatedly contribute to training and testing."
#
#  WHAT THIS DOES
#  --------------
#  R3 reported training-cohort 95% CIs as  mean +/- 1.96 * SD/sqrt(100)
#  over the 100 folds of 5x20 repeated CV.  Those 100 AUCs are NOT
#  independent (the same 271 participants recur in every fold), so the
#  interval is far too narrow.  We replace it with a bootstrap in which
#  the RESAMPLING UNIT IS THE PARTICIPANT:
#
#    for b in 1..B:
#        draw 271 participants WITH replacement, stratified on
#            case/control so prevalence is preserved
#        fit the full pipeline (median impute -> z-score -> clf)
#            on the in-bag sample
#        score the OUT-OF-BAG participants (~37%, never trained on)
#        record AUC
#
#  Because every feature set and classifier is fit on the SAME in-bag
#  sample and scored on the SAME out-of-bag participants within a given
#  replicate, incremental comparisons (Base+PGS vs Base) are PAIRED by
#  construction -- which is what makes the delta-AUC interval valid.
#
#  We report BOTH quantities the editor offered as alternatives:
#    (a) 5x20 CV mean +/- SD  -> DESCRIPTIVE ONLY, no CI, no p-value
#    (b) participant-level bootstrap AUC (95% percentile CI) -> INFERENCE
#
#  Outputs:
#    outputs/tables/Table_Bootstrap_AUC_Training.xlsx
#      A_PerClassifier      per classifier x feature set
#      B_DeltaAUC_vs_Base   incremental comparisons, paired
#      C_CrossClassifier    cross-classifier mean (as quoted in text)
#      D_Notes              method statement for STAR Methods
#    outputs/tables/bootstrap_replicates_training.csv.gz  (raw draws)
#
#  Usage:  python 10_participant_bootstrap_ci.py [B] [n_jobs]
# =============================================================

import os as _os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
_os.environ.setdefault("PYTHONWARNINGS", "ignore")

# ═══════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
from poag_corrections import (int_pgs616, int_pgs616_training_only,
                              restrict_pmbb_age)   # 2026-09-05 audit
OUT_XL = _os.path.join(HERE, "outputs", "tables")
_os.makedirs(OUT_XL, exist_ok=True)

# Data location. Set POAG_DATA_DIR to the folder holding the cohort
# subdirectories; it defaults to ./data next to this script. The data
# themselves are under controlled access (see data/README.md).
DATA_DIR = _os.environ.get(
    "POAG_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   "data"))
TRAIN_F = _os.path.join(DATA_DIR, "POAAGG_cohort",
                        "271_training_cohort_4_new_PRS_cleaned.xlsx")

LABEL = "CaseCtrl"
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
SEED = 42

B      = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else -1

# minimum out-of-bag members of each class for a replicate to count
MIN_OOB_PER_CLASS = 10

# ═══════════════════════════════════════════════════════════════
#  FEATURE SETS  (identical definitions to 02_training_external_validation.py)
# ═══════════════════════════════════════════════════════════════
BASE_COLS = ["Age", "Gender"]
PC = lambda k: [f"PC{i}" for i in range(1, k + 1)]

FS = {
    "Age only":          ["Age"],
    "Sex only":          ["Gender"],
    "Base":              BASE_COLS,
    "Base+PC2":          BASE_COLS + PC(2),
    "Base+PC5":          BASE_COLS + PC(5),
    "Base+PC10":         BASE_COLS + PC(10),
    "Base+PC20":         BASE_COLS + PC(20),
    "Base+POAAGG PGS":   BASE_COLS + ["POAAGG PGS"],
    "Base+MEGA PGS":     BASE_COLS + ["MEGA PGS"],
    "Base+PGS526":       BASE_COLS + ["PGS526"],
    "Base+PGS616":       BASE_COLS + ["PGS616"],
    "Base+PC5+PGS526":   BASE_COLS + PC(5) + ["PGS526"],
    "Base+PC5+PGS616":   BASE_COLS + PC(5) + ["PGS616"],
}
FS_NAMES = list(FS.keys())


def make_pipeline(name):
    """Identical to the pipelines used in scripts 02/07/08/09."""
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(max_iter=1000,
            class_weight="balanced", random_state=SEED))]
    elif name == "SVM":
        steps += [("clf", SVC(kernel="rbf", probability=True,
            class_weight="balanced", random_state=SEED))]
    elif name == "RF":
        steps += [("clf", RandomForestClassifier(n_estimators=200,
            max_depth=5, class_weight="balanced", random_state=SEED))]
    elif name == "MLP":
        steps += [("clf", MLPClassifier(hidden_layer_sizes=(32,),
            max_iter=1000, early_stopping=False, random_state=SEED))]
    return Pipeline(steps)


# ═══════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════
print("Loading training cohort ...", flush=True)
tr = pd.read_excel(TRAIN_F)
tr = int_pgs616_training_only(tr)
y = tr[LABEL].values.astype(int)
N = len(y)
Xall = {fs: tr[cols].values.astype(float) for fs, cols in FS.items()}
print(f"  N = {N}   cases = {y.sum()}   controls = {(1 - y).sum()}", flush=True)
print(f"  feature sets = {len(FS)}   classifiers = {len(MODEL_NAMES)}", flush=True)
print(f"  B = {B}   n_jobs = {N_JOBS}", flush=True)

idx_case = np.where(y == 1)[0]
idx_ctrl = np.where(y == 0)[0]


# ═══════════════════════════════════════════════════════════════
#  ONE BOOTSTRAP REPLICATE
# ═══════════════════════════════════════════════════════════════
def one_replicate(b):
    """Stratified participant-level resample; fit in-bag, score out-of-bag.

    Returns (b, {(model, feature_set): auc}) or (b, None) if the
    out-of-bag sample is too small / single-class.
    """
    warnings.filterwarnings("ignore")
    rng = np.random.RandomState(SEED + b)
    in_bag = np.concatenate([
        rng.choice(idx_case, len(idx_case), replace=True),
        rng.choice(idx_ctrl, len(idx_ctrl), replace=True),
    ])
    oob = np.setdiff1d(np.arange(N), np.unique(in_bag))
    if len(oob) == 0:
        return b, None
    y_oob = y[oob]
    if (y_oob == 1).sum() < MIN_OOB_PER_CLASS or \
       (y_oob == 0).sum() < MIN_OOB_PER_CLASS:
        return b, None

    out = {}
    for m in MODEL_NAMES:
        for fs in FS_NAMES:
            X = Xall[fs]
            pipe = make_pipeline(m)
            pipe.fit(X[in_bag], y[in_bag])
            p = pipe.predict_proba(X[oob])[:, 1]
            out[(m, fs)] = roc_auc_score(y_oob, p)
    return b, out


print("\nRunning participant-level bootstrap ...", flush=True)
t0 = time.time()
results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(one_replicate)(b) for b in range(B))
elapsed = time.time() - t0
print(f"  done in {elapsed/60:.1f} min", flush=True)

valid = [(b, r) for b, r in results if r is not None]
n_valid = len(valid)
print(f"  valid replicates: {n_valid} / {B} "
      f"({B - n_valid} discarded for insufficient out-of-bag cases)", flush=True)

# long-form array:  boot[model][fs] = np.array over replicates
boot = {m: {fs: np.array([r[(m, fs)] for _, r in valid])
            for fs in FS_NAMES} for m in MODEL_NAMES}

# persist raw draws for reproducibility
raw = pd.DataFrame(
    [{"replicate": b, "Classifier": m, "FeatureSet": fs, "AUC": r[(m, fs)]}
     for b, r in valid for m in MODEL_NAMES for fs in FS_NAMES])
raw.to_csv(_os.path.join(OUT_XL, "bootstrap_replicates_training.csv.gz"),
           index=False, compression="gzip")
print(f"  raw draws saved ({len(raw):,} rows)", flush=True)


# ═══════════════════════════════════════════════════════════════
#  DESCRIPTIVE 5x20 CV  (reported alongside, WITHOUT inference)
# ═══════════════════════════════════════════════════════════════
print("\nComputing descriptive 5x20 CV mean +/- SD ...", flush=True)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
folds = list(rskf.split(np.zeros(N), y))


def cv_one(m, fs):
    warnings.filterwarnings("ignore")
    X = Xall[fs]
    aucs = []
    for tri, tei in folds:
        if len(np.unique(y[tei])) < 2:
            continue
        pipe = make_pipeline(m)
        pipe.fit(X[tri], y[tri])
        aucs.append(roc_auc_score(y[tei], pipe.predict_proba(X[tei])[:, 1]))
    return m, fs, np.array(aucs)


# Run serially: the descriptive CV pass is cheap, and a second loky pool
# after the bootstrap pool is unreliable on Windows.
cv_out = {m: {} for m in MODEL_NAMES}
for m in MODEL_NAMES:
    for fs in FS_NAMES:
        _, _, a = cv_one(m, fs)
        cv_out[m][fs] = a
    print(f"    {m} done ({time.time()-t0:.0f}s elapsed)", flush=True)
print("  done.", flush=True)


# ═══════════════════════════════════════════════════════════════
#  SHEET A — per classifier x feature set
# ═══════════════════════════════════════════════════════════════
rows = []
for m in MODEL_NAMES:
    for fs in FS_NAMES:
        cv = cv_out[m][fs]
        bs = boot[m][fs]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({
            "Classifier": m,
            "Feature set": fs,
            "CV mean (descriptive)": round(cv.mean(), 3),
            "CV SD (descriptive)": round(cv.std(ddof=1), 3),
            "CV mean +/- SD": f"{cv.mean():.3f} +/- {cv.std(ddof=1):.3f}",
            "Bootstrap AUC": round(bs.mean(), 3),
            "Bootstrap 95% CI": f"{bs.mean():.3f} ({lo:.3f}-{hi:.3f})",
            "Boot CI width": round(hi - lo, 3),
            "n replicates": len(bs),
        })
sheetA = pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  SHEET B — paired incremental delta-AUC vs Base (within classifier)
# ═══════════════════════════════════════════════════════════════
def boot_p(d):
    """Two-sided bootstrap p: 2 x min(P(d<=0), P(d>=0)), floored at 1/n."""
    n = len(d)
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return max(min(p, 1.0), 1.0 / n)


rows = []
for m in MODEL_NAMES:
    base_b = boot[m]["Base"]
    base_cv = cv_out[m]["Base"]
    for fs in FS_NAMES:
        if fs in ("Age only", "Sex only", "Base"):
            continue
        d = boot[m][fs] - base_b            # paired within replicate
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = boot_p(d)
        rows.append({
            "Classifier": m,
            "Comparison": f"{fs} vs Base",
            "CV delta (descriptive)": round(cv_out[m][fs].mean() - base_cv.mean(), 3),
            "Bootstrap delta-AUC": round(d.mean(), 4),
            "Bootstrap 95% CI": f"{d.mean():+.4f} ({lo:+.4f}, {hi:+.4f})",
            "Bootstrap p": "<0.001" if p < 0.001 else f"{p:.3f}",
            "CI excludes 0": "yes" if (lo > 0 or hi < 0) else "no",
        })
sheetB = pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  SHEET C — cross-classifier mean (the numbers quoted in the text)
# ═══════════════════════════════════════════════════════════════
rows = []
for fs in FS_NAMES:
    # average the four classifiers WITHIN each replicate, then take
    # percentiles -> a CI for the cross-classifier mean
    stacked = np.vstack([boot[m][fs] for m in MODEL_NAMES]).mean(axis=0)
    lo, hi = np.percentile(stacked, [2.5, 97.5])
    cvm = np.mean([cv_out[m][fs].mean() for m in MODEL_NAMES])
    d = stacked - np.vstack([boot[m]["Base"] for m in MODEL_NAMES]).mean(axis=0)
    dlo, dhi = np.percentile(d, [2.5, 97.5])
    p = boot_p(d)
    rows.append({
        "Feature set": fs,
        "CV cross-classifier mean (descriptive)": round(cvm, 3),
        "Bootstrap mean AUC": round(stacked.mean(), 3),
        "Bootstrap 95% CI": f"{stacked.mean():.3f} ({lo:.3f}-{hi:.3f})",
        "delta vs Base": round(d.mean(), 4),
        "delta 95% CI": f"{d.mean():+.4f} ({dlo:+.4f}, {dhi:+.4f})",
        "delta p": "<0.001" if p < 0.001 else f"{p:.3f}",
    })
sheetC = pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  SHEET D — method statement
# ═══════════════════════════════════════════════════════════════
notes = pd.DataFrame({"Note": [
    "Table: Participant-level bootstrap AUC and incremental delta-AUC, "
    "POAAGG training cohort (N=271).",
    "",
    "METHOD. Uncertainty for the training cohort is estimated by a "
    f"participant-level bootstrap with B = {B} replicates (valid: {n_valid}). "
    "In each replicate, participants are resampled with replacement, "
    "stratified on case/control status so that prevalence is preserved; the "
    "full modelling pipeline (median imputation, z-score standardisation, "
    "classifier) is refitted on the in-bag sample and evaluated on the "
    "out-of-bag participants, who never enter that replicate's training set. "
    "Replicates with fewer than "
    f"{MIN_OOB_PER_CLASS} out-of-bag members of either class are discarded. "
    "95% intervals are percentile intervals of the resulting AUC distribution.",
    "",
    "PAIRING. Within a replicate every feature set and classifier is fitted "
    "on the same in-bag participants and evaluated on the same out-of-bag "
    "participants. Incremental delta-AUC is therefore a paired contrast, and "
    "its percentile interval accounts for the correlation between the two "
    "models being compared. Two-sided p-values are 2 x min(P(delta<=0), "
    "P(delta>=0)), bounded below by 1/n_replicates.",
    "",
    "RELATION TO REPEATED CROSS-VALIDATION. The 5-fold x 20-repeat "
    "cross-validation mean and SD are retained as DESCRIPTIVE summaries of "
    "model stability only. They are not used for interval estimation or "
    "hypothesis testing, because the 100 fold-level estimates are not "
    "independent: the same 271 participants recur across folds and repeats, "
    "so SD/sqrt(100) understates the true sampling variability. All "
    "inferential statements about the training cohort derive from the "
    "participant-level bootstrap above.",
    "",
    f"Software: scikit-learn {__import__('sklearn').__version__}, "
    f"numpy {np.__version__}, seed {SEED}.",
]})


with pd.ExcelWriter(_os.path.join(OUT_XL, "Table_Bootstrap_AUC_Training.xlsx"),
                    engine="openpyxl") as w:
    sheetA.to_excel(w, sheet_name="A_PerClassifier", index=False)
    sheetB.to_excel(w, sheet_name="B_DeltaAUC_vs_Base", index=False)
    sheetC.to_excel(w, sheet_name="C_CrossClassifier", index=False)
    notes.to_excel(w, sheet_name="D_Notes", index=False)

print("\nSaved Table_Bootstrap_AUC_Training.xlsx", flush=True)


# ═══════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY — the numbers that go into the manuscript
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 68, flush=True)
print("KEY CONFIGURATIONS — old (5x20 CV) vs new (participant bootstrap)", flush=True)
print("=" * 68, flush=True)
key = ["Base", "Base+PC2", "Base+PGS526", "Base+PGS616"]
for fs in key:
    r = sheetC[sheetC["Feature set"] == fs].iloc[0]
    old_lo = None
    # R3-style (invalid) interval, for the response letter's before/after
    cvs = np.concatenate([cv_out[m][fs] for m in MODEL_NAMES])
    se = cvs.std(ddof=1) / np.sqrt(len(cvs))
    old_lo, old_hi = cvs.mean() - 1.96 * se, cvs.mean() + 1.96 * se
    print(f"\n{fs}", flush=True)
    print(f"  R3  5x20 CV     : {cvs.mean():.3f} ({old_lo:.3f}-{old_hi:.3f})"
          f"   width {old_hi-old_lo:.3f}   [not valid]", flush=True)
    print(f"  R4  bootstrap   : {r['Bootstrap 95% CI']}"
          f"   width {float(r['Bootstrap 95% CI'].split('(')[1].split('-')[1][:-1]) - float(r['Bootstrap 95% CI'].split('(')[1].split('-')[0]):.3f}", flush=True)

print("\n" + "=" * 68, flush=True)
print("INCREMENTAL delta-AUC vs Base (per classifier, paired bootstrap)", flush=True)
print("=" * 68, flush=True)
for m in MODEL_NAMES:
    sub = sheetB[(sheetB.Classifier == m) &
                 (sheetB.Comparison.str.contains("PGS616 vs"))]
    for _, r in sub.iterrows():
        print(f"  {m:4s} {r['Comparison']:26s} {r['Bootstrap 95% CI']:34s} "
              f"p={r['Bootstrap p']:8s} CI excl 0: {r['CI excludes 0']}", flush=True)

print("\nAll done.", flush=True)
