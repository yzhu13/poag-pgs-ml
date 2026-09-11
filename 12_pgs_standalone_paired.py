# =============================================================
#  POAG — Do curated loci-based PGS outperform genome-wide PGS?
#  iScience R4 revision (2026-09-04)
#
#  Addresses Editor point 4 and Reviewer #1 point 4:
#   "Remove or statistically substantiate the claim that curated
#    loci-based PGS outperforms genome-wide PGS, given standalone AUC
#    values near 0.50-0.53."
#
#  R3 reported standalone cross-classifier AUCs of
#     PGS526 0.526 | PGS616 0.505 | MEGA 0.500 | POAAGG 0.498
#  and claimed curated scores "outperform" genome-wide scores.  Those
#  four values are all close to chance and were never formally compared.
#
#  This script performs the comparison the editor asks for, using the
#  same participant-level bootstrap as script 10 so that the contrast is
#  PAIRED: within each replicate all four PGS are fitted on the same
#  in-bag participants and evaluated on the same out-of-bag participants.
#  A paired contrast is more powerful than comparing four marginal CIs,
#  so this is the most favourable valid test of the claim.
#
#  Also reports, for each PGS, whether its standalone AUC is
#  distinguishable from chance (0.50).
#
#  Outputs:
#    outputs/tables/Table_PGS_Standalone_Comparison.xlsx
#      A_StandaloneAUC     per PGS: AUC, 95% CI, test vs chance
#      B_PairedContrasts   curated vs genome-wide, paired delta + CI + p
#      C_Notes
#
#  Usage:  python 12_pgs_standalone_paired.py [B] [n_jobs]
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
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
_os.environ.setdefault("PYTHONWARNINGS", "ignore")

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
MIN_OOB_PER_CLASS = 10

# standalone PGS, one feature each
PGS = {
    "PGS526":      ["PGS526"],        # curated, MTAG weights
    "PGS616":      ["PGS616"],        # curated, MEGA weights
    "MEGA PGS":    ["MEGA PGS"],      # genome-wide
    "POAAGG PGS":  ["POAAGG PGS"],    # genome-wide
}
PGS_NAMES = list(PGS.keys())
CURATED = ["PGS526", "PGS616"]
GENOMEWIDE = ["MEGA PGS", "POAAGG PGS"]


def make_pipeline(name):
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


print("Loading training cohort ...", flush=True)
tr = pd.read_excel(TRAIN_F)
tr = int_pgs616_training_only(tr)
y = tr[LABEL].values.astype(int)
N = len(y)
Xall = {k: tr[c].values.astype(float) for k, c in PGS.items()}
idx_case = np.where(y == 1)[0]
idx_ctrl = np.where(y == 0)[0]
print(f"  N={N}  B={B}  n_jobs={N_JOBS}", flush=True)


def one_replicate(b):
    warnings.filterwarnings("ignore")
    rng = np.random.RandomState(SEED + b)
    in_bag = np.concatenate([
        rng.choice(idx_case, len(idx_case), replace=True),
        rng.choice(idx_ctrl, len(idx_ctrl), replace=True)])
    oob = np.setdiff1d(np.arange(N), np.unique(in_bag))
    if len(oob) == 0:
        return None
    y_oob = y[oob]
    if (y_oob == 1).sum() < MIN_OOB_PER_CLASS or \
       (y_oob == 0).sum() < MIN_OOB_PER_CLASS:
        return None
    out = {}
    for m in MODEL_NAMES:
        for k in PGS_NAMES:
            X = Xall[k]
            pipe = make_pipeline(m)
            pipe.fit(X[in_bag], y[in_bag])
            out[(m, k)] = roc_auc_score(y_oob, pipe.predict_proba(X[oob])[:, 1])
    return out


print("\nRunning paired participant-level bootstrap ...", flush=True)
t0 = time.time()
res = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(one_replicate)(b) for b in range(B))
res = [r for r in res if r is not None]
print(f"  {len(res)} valid replicates in {(time.time()-t0)/60:.1f} min",
      flush=True)

boot = {m: {k: np.array([r[(m, k)] for r in res]) for k in PGS_NAMES}
        for m in MODEL_NAMES}
# cross-classifier mean within each replicate
xmean = {k: np.vstack([boot[m][k] for m in MODEL_NAMES]).mean(axis=0)
         for k in PGS_NAMES}


def boot_p(d):
    n = len(d)
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return max(min(p, 1.0), 1.0 / n)


def fmt_p(p):
    return "<0.001" if (isinstance(p, float) and p < 0.001) else f"{p:.3f}"


# ═══════════════════════════════════════════════════════════════
#  A — standalone AUC per PGS, and test against chance
# ═══════════════════════════════════════════════════════════════
rows = []
for k in PGS_NAMES:
    for scope in MODEL_NAMES + ["Cross-classifier mean"]:
        a = xmean[k] if scope == "Cross-classifier mean" else boot[scope][k]
        lo, hi = np.percentile(a, [2.5, 97.5])
        d = a - 0.5
        rows.append({
            "PGS": k,
            "Type": "curated" if k in CURATED else "genome-wide",
            "Classifier": scope,
            "Bootstrap AUC": round(a.mean(), 3),
            "95% CI": f"{a.mean():.3f} ({lo:.3f}-{hi:.3f})",
            "CI includes 0.50": "yes" if (lo <= 0.5 <= hi) else "no",
            "p vs chance": fmt_p(boot_p(d)),
        })
sheetA = pd.DataFrame(rows)

print("\n=== Standalone AUC (cross-classifier mean) ===", flush=True)
for _, r in sheetA[sheetA.Classifier == "Cross-classifier mean"].iterrows():
    print(f"  {r['PGS']:12s} {r['95% CI']:24s} "
          f"includes 0.50: {r['CI includes 0.50']:4s} p={r['p vs chance']}",
          flush=True)


# ═══════════════════════════════════════════════════════════════
#  B — paired contrasts: curated vs genome-wide
# ═══════════════════════════════════════════════════════════════
rows = []
pairs = [(c, g) for c in CURATED for g in GENOMEWIDE]
pairs += [("PGS526", "PGS616")]     # the two curated scores vs each other
for a_name, b_name in pairs:
    for scope in MODEL_NAMES + ["Cross-classifier mean"]:
        if scope == "Cross-classifier mean":
            da, db = xmean[a_name], xmean[b_name]
        else:
            da, db = boot[scope][a_name], boot[scope][b_name]
        d = da - db
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = boot_p(d)
        rows.append({
            "Contrast": f"{a_name} - {b_name}",
            "Classifier": scope,
            "delta AUC": round(d.mean(), 4),
            "95% CI": f"{d.mean():+.4f} ({lo:+.4f}, {hi:+.4f})",
            "p": fmt_p(p),
            "CI excludes 0": "yes" if (lo > 0 or hi < 0) else "no",
        })
sheetB = pd.DataFrame(rows)

print("\n=== Curated vs genome-wide (cross-classifier mean, paired) ===",
      flush=True)
for _, r in sheetB[sheetB.Classifier == "Cross-classifier mean"].iterrows():
    print(f"  {r['Contrast']:26s} {r['95% CI']:34s} p={r['p']:8s} "
          f"CI excl 0: {r['CI excludes 0']}", flush=True)


notes = pd.DataFrame({"Note": [
    "Table: Standalone PGS discrimination and formal comparison of curated "
    "loci-based versus genome-wide scores, POAAGG training cohort (N=271).",
    "",
    f"METHOD. Participant-level bootstrap, B={B} replicates "
    f"({len(res)} valid). Within each replicate all four scores are fitted "
    "on the same in-bag participants and evaluated on the same out-of-bag "
    "participants, so every contrast is paired and the interval accounts for "
    "the correlation between scores computed in the same individuals. A "
    "paired contrast is more powerful than comparing marginal confidence "
    "intervals; this is therefore a favourable test of the claim.",
    "",
    "PANEL A also tests each score against chance (AUC = 0.50).",
    "",
    "PANEL B reports curated minus genome-wide differences. A confidence "
    "interval containing zero means the data do not support a claim that "
    "curated loci-based scores outperform genome-wide scores.",
]})

with pd.ExcelWriter(_os.path.join(
        OUT_XL, "Table_PGS_Standalone_Comparison.xlsx"),
        engine="openpyxl") as w:
    sheetA.to_excel(w, sheet_name="A_StandaloneAUC", index=False)
    sheetB.to_excel(w, sheet_name="B_PairedContrasts", index=False)
    notes.to_excel(w, sheet_name="C_Notes", index=False)

print("\nSaved Table_PGS_Standalone_Comparison.xlsx", flush=True)
print("All done.", flush=True)
