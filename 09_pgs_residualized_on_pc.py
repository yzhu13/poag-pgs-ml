# =============================================================
#  POAG — PGS Residualized on Ancestry PCs (sensitivity analysis)
#  iScience R3 revision (2026-07-10)
#
#  Addresses Reviewer #1 point 4:
#   "|r| = 0.35 is not negligible ... consider additional analyses,
#    such as residualizing PGS on PCs ..."
#
#  Question: does the curated PGS carry predictive signal INDEPENDENT
#  of ancestry principal components?  We regress each PGS on PC1-PC5
#  (within cohort), take the residual (the part of the PGS orthogonal
#  to the top 5 PCs), and re-run the same-classifier ΔAUC of
#  Base+PGS_resid vs Base.  If the residualized PGS gives ΔAUC similar
#  to the raw PGS, the PGS signal is not merely an ancestry-PC proxy.
#
#  Reported:
#   - variance of each PGS retained after removing PC1-5 (=1-R^2)
#   - training paired-fold ΔAUC: Base+PGS616_resid vs Base (per classifier)
#   - PMBB DeLong ΔAUC: Base+PGS616_resid vs Base (per classifier)
#   side-by-side with the RAW-PGS ΔAUC for direct comparison.
#
#  Output:
#   outputs/tables/Table_PGS_residualized_on_PC.xlsx
# =============================================================

import os as _os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
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

HERE = _os.path.dirname(_os.path.abspath(__file__))
OUT_XL = _os.path.join(HERE, "..", "outputs", "tables")
_os.makedirs(OUT_XL, exist_ok=True)

DATA_DIR = (r"C:\Users\biqiz\iCloudDrive\3_Penn_Postdoc\0_Projects_Ongoing"
            r"\1_MLP\_archive\R1_work_2026-05-08\input-data")
TRAIN_F  = _os.path.join(DATA_DIR, "POAAGG_cohort",
                         "271_training_cohort_4_new_PRS_cleaned.xlsx")
PMBB_PHE = _os.path.join(DATA_DIR, "PMBB_external",
                         "PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_updated_June8.csv")
PMBB_616 = _os.path.join(DATA_DIR, "PMBB_external",
                         "PMBBv3_GRS_MEGA_616snps_AllSamples.sscore_withSTDscore.txt")
PMBB_526 = _os.path.join(DATA_DIR, "PMBB_external",
                         "PMBBv3_GRS_QUANT_526snps_AllSamples.sscore_withSTDscore.txt")

LABEL = "CaseCtrl"
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
SEED, N_BOOT = 42, 1000
PC5_COLS = ["PC1", "PC2", "PC3", "PC4", "PC5"]


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


# ---- DeLong (compact) ---------------------------------------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong_test(y_true, p1, p2):
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")
    yl = y_true[order]; m = int(yl.sum())
    preds = np.vstack((np.asarray(p1)[order], np.asarray(p2)[order]))
    n = preds.shape[1]-m; k = 2
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(preds[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(1)/m - (m+1)/2)/n
    v01 = (tz[:, :m]-tx)/n; v10 = 1-(tz[:, m:]-ty)/m
    cov = np.cov(v01)/m + np.cov(v10)/n
    var = cov[0, 0]+cov[1, 1]-2*cov[0, 1]
    d = aucs[0]-aucs[1]; se = np.sqrt(var) if var > 0 else 0.0
    p = 2*(1-stats.norm.cdf(abs(d/se))) if se > 0 else 1.0
    return float(aucs[0]), float(aucs[1]), float(d), float(d-1.96*se), float(d+1.96*se), float(p)


# ---- LOAD ---------------------------------------------------
print("Loading data ...")
tr = pd.read_excel(TRAIN_F)
y_tr = tr[LABEL].values.astype(int)
phe = pd.read_csv(PMBB_PHE)
p616 = (pd.read_csv(PMBB_616, sep="\t")[["IID", "SCORE1_AVG_STD"]]
        .rename(columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": "PGS616"}))
p526 = (pd.read_csv(PMBB_526, sep="\t")[["IID", "SCORE1_AVG_STD"]]
        .rename(columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": "PGS526"}))
pmbb = phe.merge(p616, on="PMBB_ID").merge(p526, on="PMBB_ID")
pmbb = pmbb[pmbb["ANCESTRY"] == "AFR"].dropna(
    subset=["POAG_cases", "PGS616", "PGS526",
            "PMBB_3.0_Release_AGE", "SEX"]).copy()
pmbb["POAG_cases"] = pmbb["POAG_cases"].astype(int)
pmbb["SEX_bin"] = (pmbb["SEX"] == "Male").astype(int)
y_pmbb = pmbb["POAG_cases"].values
print(f"  Train N={len(tr)}  PMBB AFR N={len(pmbb):,}")


# ---- RESIDUALIZE PGS ON PC1-PC5 (within each cohort) --------
def residualize(df, pgs_col, pc_cols):
    """Return residual of pgs_col after regressing on pc_cols, and R^2."""
    sub = df[pc_cols + [pgs_col]].copy()
    # median-impute PCs/PGS for the regression only
    Xp = sub[pc_cols].apply(lambda c: c.fillna(c.median())).values
    yv = sub[pgs_col].fillna(sub[pgs_col].median()).values
    lr = LinearRegression().fit(Xp, yv)
    pred = lr.predict(Xp)
    resid = yv - pred
    ss_res = np.sum((yv - pred)**2)
    ss_tot = np.sum((yv - yv.mean())**2)
    r2 = 1 - ss_res/ss_tot
    return resid, r2

print("\nVariance of PGS explained by PC1-PC5 (R^2) and retained (1-R^2):")
var_rows = []
for cohort, df in [("Training", tr), ("PMBB", pmbb)]:
    for pgs in ["PGS616", "PGS526"]:
        resid, r2 = residualize(df, pgs, PC5_COLS)
        df[pgs + "_resid"] = resid
        print(f"  {cohort:8s} {pgs}:  R^2(PGS~PC1-5)={r2:.4f}  "
              f"variance retained={1-r2:.4f}")
        var_rows.append({"Cohort": cohort, "PGS": pgs,
                         "R2_PGS_on_PC5": round(r2, 4),
                         "Variance_retained": round(1-r2, 4)})
var_df = pd.DataFrame(var_rows)


# ---- FEATURE SETS (raw vs residualized) ---------------------
FS = {
    "Base":              ["Age", "Gender"],
    "Base+PGS616":       ["Age", "Gender", "PGS616"],
    "Base+PGS616_resid": ["Age", "Gender", "PGS616_resid"],
    "Base+PGS526":       ["Age", "Gender", "PGS526"],
    "Base+PGS526_resid": ["Age", "Gender", "PGS526_resid"],
}
COMPARISONS = [("Base+PGS616", "Base"), ("Base+PGS616_resid", "Base"),
               ("Base+PGS526", "Base"), ("Base+PGS526_resid", "Base")]


# ---- (A) TRAINING paired-fold ΔAUC --------------------------
print("\n(A) Training paired ΔAUC (raw vs residualized PGS) ...")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
folds = list(rskf.split(np.zeros(len(y_tr)), y_tr))
Xcache = {fs: tr[cols].values for fs, cols in FS.items()}
fold_auc = {m: {fs: [] for fs in FS} for m in MODEL_NAMES}
for m in MODEL_NAMES:
    for (tri, tei) in folds:
        yt = y_tr[tei]
        if len(np.unique(yt)) < 2:
            for fs in FS: fold_auc[m][fs].append(np.nan)
            continue
        for fs in FS:
            X = Xcache[fs]
            pipe = make_pipeline(m); pipe.fit(X[tri], y_tr[tri])
            fold_auc[m][fs].append(roc_auc_score(yt, pipe.predict_proba(X[tei])[:, 1]))
    print(f"  {m}: folds done")

train_rows = []
for m in MODEL_NAMES:
    for aug, ref in COMPARISONS:
        a = np.array(fold_auc[m][aug]); b = np.array(fold_auc[m][ref])
        mask = ~(np.isnan(a) | np.isnan(b)); a, b = a[mask], b[mask]
        d = a - b; lo, hi = np.percentile(d, [2.5, 97.5])
        _, p_t = stats.ttest_rel(a, b)
        train_rows.append({"Classifier": m, "Comparison": f"{aug} vs {ref}",
            "AUC_ref": round(b.mean(), 4), "AUC_aug": round(a.mean(), 4),
            "Delta_AUC": round(d.mean(), 4),
            "Delta_95CI": f"{d.mean():+.4f} ({lo:+.4f}, {hi:+.4f})",
            "p_paired_t": round(float(p_t), 4)})
train_df = pd.DataFrame(train_rows)


# ---- (B) PMBB DeLong ΔAUC -----------------------------------
print("\n(B) PMBB external ΔAUC (raw vs residualized PGS) ...")
def pmbb_mat(fs):
    ct = FS[fs][:]  # training cols
    cp = []
    for col in ct:
        if col == "Age": cp.append("PMBB_3.0_Release_AGE")
        elif col == "Gender": cp.append("SEX_bin")
        else: cp.append(col)  # PGS616/PGS526/_resid all exist in pmbb
    return tr[ct].values, pmbb[cp].values

pred = {m: {} for m in MODEL_NAMES}
for m in MODEL_NAMES:
    for fs in FS:
        Xt, Xp = pmbb_mat(fs)
        pipe = make_pipeline(m); pipe.fit(Xt, y_tr)
        pred[m][fs] = pipe.predict_proba(Xp)[:, 1]

pmbb_rows = []
for m in MODEL_NAMES:
    for aug, ref in COMPARISONS:
        aa, ab, d, lo, hi, p = delong_test(y_pmbb, pred[m][aug], pred[m][ref])
        pmbb_rows.append({"Classifier": m, "Comparison": f"{aug} vs {ref}",
            "AUC_ref": round(ab, 4), "AUC_aug": round(aa, 4),
            "Delta_AUC": round(d, 4),
            "DeLong_95CI": f"{d:+.4f} ({lo:+.4f}, {hi:+.4f})",
            "DeLong_p": round(p, 4)})
pmbb_df = pd.DataFrame(pmbb_rows)


# ---- SAVE + SUMMARY -----------------------------------------
with pd.ExcelWriter(_os.path.join(OUT_XL, "Table_PGS_residualized_on_PC.xlsx"),
                    engine="openpyxl") as w:
    var_df.to_excel(w, sheet_name="PGS_variance_after_PC", index=False)
    train_df.to_excel(w, sheet_name="Training_dAUC", index=False)
    pmbb_df.to_excel(w, sheet_name="PMBB_dAUC", index=False)

def side_by_side(df, ci_col, p_col, tag):
    print(f"\n{tag}: ΔAUC vs Base — RAW PGS616 vs RESIDUALIZED PGS616")
    for m in MODEL_NAMES:
        raw = df[(df.Classifier==m)&(df.Comparison=="Base+PGS616 vs Base")]
        res = df[(df.Classifier==m)&(df.Comparison=="Base+PGS616_resid vs Base")]
        print(f"  {m:4s} raw {raw[ci_col].iloc[0]}  (p={raw[p_col].iloc[0]})"
              f"   |  resid {res[ci_col].iloc[0]}  (p={res[p_col].iloc[0]})")

print("\n=== RESULTS ===")
print(var_df.to_string(index=False))
side_by_side(train_df, "Delta_95CI", "p_paired_t", "Training (paired CV folds)")
side_by_side(pmbb_df,  "DeLong_95CI", "DeLong_p",  "PMBB external (DeLong)")
print("\nSaved Table_PGS_residualized_on_PC.xlsx")
