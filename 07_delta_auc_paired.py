# =============================================================
#  POAG — Incremental AUC (ΔAUC) Paired Comparisons
#  iScience R3 revision (2026-07-10)
#
#  Addresses Reviewer #1, point 3 and Editor points 2, 3, 7:
#   "provide paired comparisons for AUC differences between Base
#    and Base+PGS models ... paired bootstrap, DeLong where
#    appropriate, or resampling-based paired differences across
#    CV folds and in PMBB. Confidence intervals for ΔAUC ..."
#
#  Two settings, both SAME-CLASSIFIER (Base vs Base+PGS, identical model):
#   (A) Training cohort (POAAGG, N=271):
#         paired ΔAUC across the 100 identical 5×20 CV folds
#         → mean ΔAUC, 95% percentile CI, one-sided empirical p
#   (B) PMBB external (AFR, N≈9,817):
#         - DeLong test for two correlated AUCs (p + analytic 95% CI)
#         - paired bootstrap ΔAUC (1000 resamples) 95% percentile CI
#
#  Outputs:
#    outputs/tables/Table_DeltaAUC_Training_CV.xlsx
#    outputs/tables/Table_DeltaAUC_PMBB_External.xlsx
#    outputs/figures/Figure_DeltaAUC_forest.png/.pdf
# =============================================================

import os as _os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
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

# ═══════════════════════════════════════════════════════════════
#  PATH CONFIGURATION
#  Data live in the project _archive (R1 working set, June-8 update).
#  Edit DATA_DIR if the data are elsewhere.
# ═══════════════════════════════════════════════════════════════
HERE = _os.path.dirname(_os.path.abspath(__file__))
OUT_XL  = _os.path.join(HERE, "..", "outputs", "tables")
OUT_FIG = _os.path.join(HERE, "..", "outputs", "figures")
_os.makedirs(OUT_XL,  exist_ok=True)
_os.makedirs(OUT_FIG, exist_ok=True)

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

LABEL       = "CaseCtrl"
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
SEED        = 42
N_BOOT      = 1000

BASE_COLS = ["Age", "Gender"]
PC5_COLS  = ["PC1", "PC2", "PC3", "PC4", "PC5"]

# feature sets used here (must exist as columns in training data)
FS = {
    "Base":         BASE_COLS,
    "Base+PGS616":  BASE_COLS + ["PGS616"],
    "Base+PGS526":  BASE_COLS + ["PGS526"],
    "Base+PC5":     BASE_COLS + PC5_COLS,
}
# comparisons: (augmented, reference)
COMPARISONS = [
    ("Base+PGS616", "Base"),
    ("Base+PGS526", "Base"),
    ("Base+PC5",    "Base"),
]


# =============================================================
#  HELPERS
# =============================================================
def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED))]
    elif name == "SVM":
        steps += [("clf", SVC(kernel="rbf", probability=True,
            class_weight="balanced", random_state=SEED))]
    elif name == "RF":
        steps += [("clf", RandomForestClassifier(
            n_estimators=200, max_depth=5,
            class_weight="balanced", random_state=SEED))]
    elif name == "MLP":
        steps += [("clf", MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=1000,
            early_stopping=False, random_state=SEED))]
    return Pipeline(steps)


# ---- DeLong test for two correlated ROC AUCs ---------------
#  Fast midrank implementation (Sun & Xu, 2014).
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    # preds_sorted: (k, n) predictions, positives first (m positives)
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_test(y_true, p1, p2):
    """Return AUC1, AUC2, ΔAUC (=AUC1-AUC2), CI on ΔAUC, two-sided p."""
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")  # positives first
    label_1 = y_true[order]
    m = int(label_1.sum())
    preds = np.vstack((np.asarray(p1)[order], np.asarray(p2)[order]))
    aucs, cov = _fast_delong(preds, m)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    d = aucs[0] - aucs[1]
    se = np.sqrt(var_diff) if var_diff > 0 else 0.0
    if se == 0:
        z, p = 0.0, 1.0
    else:
        z = d / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    lo, hi = d - 1.96 * se, d + 1.96 * se
    return float(aucs[0]), float(aucs[1]), float(d), float(lo), float(hi), float(p)


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


# =============================================================
#  LOAD DATA
# =============================================================
print("Loading data ...")
tr = pd.read_excel(TRAIN_F)
y_tr = tr[LABEL].values.astype(int)
print(f"  Train N={len(tr)}  cases={y_tr.sum()}  ctrl={(y_tr==0).sum()}")

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
print(f"  PMBB AFR N={len(pmbb):,}  cases={y_pmbb.sum()}  "
      f"ctrl={(y_pmbb==0).sum()}")


# =============================================================
#  (A) TRAINING — PAIRED ΔAUC ACROSS IDENTICAL 5×20 CV FOLDS
# =============================================================
print("\n(A) Training paired ΔAUC across 100 identical CV folds ...")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
folds = list(rskf.split(np.zeros(len(y_tr)), y_tr))  # identical for all FS

# per-fold AUC:  fold_auc[model][fs] = list over 100 folds
fold_auc = {m: {fs: [] for fs in FS} for m in MODEL_NAMES}
Xcache = {fs: tr[cols].values for fs, cols in FS.items()}

for m in MODEL_NAMES:
    for (tri, tei) in folds:
        yt = y_tr[tei]
        if len(np.unique(yt)) < 2:
            for fs in FS:
                fold_auc[m][fs].append(np.nan)
            continue
        for fs in FS:
            X = Xcache[fs]
            pipe = make_pipeline(m)
            pipe.fit(X[tri], y_tr[tri])
            yp = pipe.predict_proba(X[tei])[:, 1]
            fold_auc[m][fs].append(roc_auc_score(yt, yp))
    print(f"  {m}: folds done")

train_rows = []
for m in MODEL_NAMES:
    for aug, ref in COMPARISONS:
        a = np.array(fold_auc[m][aug], dtype=float)
        b = np.array(fold_auc[m][ref], dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        d = a - b                       # paired per-fold differences
        mean_d = d.mean()
        lo, hi = np.percentile(d, [2.5, 97.5])
        # one-sided empirical p: fraction of folds where augmented is NOT better
        p_emp = np.mean(d <= 0)
        # paired t-test across folds (folds not independent → report as descriptive)
        t_stat, p_t = stats.ttest_rel(a, b)
        train_rows.append({
            "Classifier": m, "Comparison": f"{aug} vs {ref}",
            "AUC_ref":  round(b.mean(), 4),
            "AUC_aug":  round(a.mean(), 4),
            "Delta_AUC": round(mean_d, 4),
            "Delta_95CI": f"{mean_d:+.4f} ({lo:+.4f}, {hi:+.4f})",
            "CI_lower": round(lo, 4), "CI_upper": round(hi, 4),
            "p_empirical_1sided": round(float(p_emp), 4),
            "p_paired_t": round(float(p_t), 4),
            "n_folds": int(mask.sum()),
        })
        print(f"  {m:4s} {aug:12s}-{ref:5s}  "
              f"ΔAUC={mean_d:+.4f} ({lo:+.4f},{hi:+.4f})  p_t={p_t:.3f}")

train_df = pd.DataFrame(train_rows)


# =============================================================
#  (B) PMBB EXTERNAL — DeLong + PAIRED BOOTSTRAP ΔAUC
# =============================================================
print("\n(B) PMBB external ΔAUC (DeLong + paired bootstrap) ...")

def pmbb_matrix(fs):
    cols_tr, cols_pm = ["Age", "Gender"], ["PMBB_3.0_Release_AGE", "SEX_bin"]
    if fs == "Base+PGS616":
        cols_tr += ["PGS616"]; cols_pm += ["PGS616"]
    elif fs == "Base+PGS526":
        cols_tr += ["PGS526"]; cols_pm += ["PGS526"]
    elif fs == "Base+PC5":
        cols_tr += PC5_COLS;   cols_pm += PC5_COLS
    return tr[cols_tr].values, pmbb[cols_pm].values

# get fixed prediction vectors per (model, fs)
pred = {m: {} for m in MODEL_NAMES}
for m in MODEL_NAMES:
    for fs in FS:
        Xt, Xp = pmbb_matrix(fs)
        pipe = make_pipeline(m)
        pipe.fit(Xt, y_tr)
        pred[m][fs] = pipe.predict_proba(Xp)[:, 1]

rng = np.random.RandomState(SEED)
# pre-draw bootstrap index sets (shared across comparisons for coherence)
boot_idx = [rng.choice(len(y_pmbb), len(y_pmbb), replace=True)
            for _ in range(N_BOOT)]

pmbb_rows = []
for m in MODEL_NAMES:
    for aug, ref in COMPARISONS:
        pa, pb = pred[m][aug], pred[m][ref]
        auc_a, auc_b, d_delong, lo_dl, hi_dl, p_dl = delong_test(y_pmbb, pa, pb)
        # paired bootstrap ΔAUC
        dboot = []
        for idx in boot_idx:
            yy = y_pmbb[idx]
            if len(np.unique(yy)) < 2:
                continue
            dboot.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
        dboot = np.array(dboot)
        lo_b, hi_b = np.percentile(dboot, [2.5, 97.5])
        pmbb_rows.append({
            "Classifier": m, "Comparison": f"{aug} vs {ref}",
            "AUC_ref": round(auc_b, 4), "AUC_aug": round(auc_a, 4),
            "Delta_AUC": round(d_delong, 4),
            "DeLong_95CI": f"{d_delong:+.4f} ({lo_dl:+.4f}, {hi_dl:+.4f})",
            "DeLong_p": round(p_dl, 4),
            "Bootstrap_95CI": f"{dboot.mean():+.4f} ({lo_b:+.4f}, {hi_b:+.4f})",
            "boot_lo": round(lo_b, 4), "boot_hi": round(hi_b, 4),
        })
        print(f"  {m:4s} {aug:12s}-{ref:5s}  "
              f"ΔAUC={d_delong:+.4f}  DeLong p={p_dl:.3g}  "
              f"boot95=({lo_b:+.4f},{hi_b:+.4f})")

pmbb_df = pd.DataFrame(pmbb_rows)


# =============================================================
#  SAVE TABLES
# =============================================================
print("\nSaving tables ...")
with pd.ExcelWriter(_os.path.join(OUT_XL, "Table_DeltaAUC_Training_CV.xlsx"),
                    engine="openpyxl") as w:
    train_df.to_excel(w, sheet_name="Training_paired_dAUC", index=False)
with pd.ExcelWriter(_os.path.join(OUT_XL, "Table_DeltaAUC_PMBB_External.xlsx"),
                    engine="openpyxl") as w:
    pmbb_df.to_excel(w, sheet_name="PMBB_dAUC", index=False)
print("  tables saved.")


# =============================================================
#  FOREST PLOT — ΔAUC (PGS616 vs Base) both settings
# =============================================================
print("Building forest plot ...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 10})

def forest(ax, df, comp, ci_lo, ci_hi, title):
    sub = df[df["Comparison"] == comp].set_index("Classifier").reindex(MODEL_NAMES)
    y = np.arange(len(MODEL_NAMES))[::-1]
    d = sub["Delta_AUC"].values
    lo = sub[ci_lo].values; hi = sub[ci_hi].values
    ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="o", color="#E15759",
                capsize=3, markersize=6, linewidth=1.3)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(MODEL_NAMES)
    ax.set_xlabel("ΔAUC (Base+PGS616 − Base)")
    ax.set_title(title, fontweight="bold", fontsize=10)
    for yi, di, l, h in zip(y, d, lo, hi):
        ax.text(h + 0.002, yi, f"{di:+.3f}", va="center", fontsize=7.5)

forest(axes[0], train_df, "Base+PGS616 vs Base",
       "CI_lower", "CI_upper",
       "Training (POAAGG, N=271)\n100 paired 5×20 CV folds")
forest(axes[1], pmbb_df, "Base+PGS616 vs Base",
       "boot_lo", "boot_hi",
       "PMBB external (AFR, N=9,817)\npaired bootstrap")
fig.tight_layout()
fig.savefig(_os.path.join(OUT_FIG, "Figure_DeltaAUC_forest.png"),
            dpi=300, bbox_inches="tight")
fig.savefig(_os.path.join(OUT_FIG, "Figure_DeltaAUC_forest.pdf"),
            bbox_inches="tight")
plt.close(fig)
print("  forest plot saved.")


# =============================================================
#  SUMMARY
# =============================================================
print("\n=== SUMMARY: ΔAUC (Base+PGS616 − Base), same classifier ===")
print("\nTraining (paired CV folds):")
for _, r in train_df[train_df["Comparison"] == "Base+PGS616 vs Base"].iterrows():
    print(f"  {r['Classifier']:4s} {r['Delta_95CI']}  (paired-t p={r['p_paired_t']})")
print("\nPMBB external (DeLong):")
for _, r in pmbb_df[pmbb_df["Comparison"] == "Base+PGS616 vs Base"].iterrows():
    print(f"  {r['Classifier']:4s} {r['DeLong_95CI']}  DeLong p={r['DeLong_p']}")
print("\nAll done.")
