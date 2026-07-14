# =============================================================
#  POAG — Same-Classifier Comparison Table
#  iScience R3 revision (2026-07-10)
#
#  Addresses Reviewer #1 point 2 (and Editor points 2, 6):
#   "The 'best AUC' framing may be misleading because baseline
#    logistic regression already performs similarly ... report
#    same-classifier comparisons and explicitly state whether PGS
#    improves performance over the best age/sex-only model, not
#    just over the cross-classifier mean."
#
#  Produces ONE table: for each classifier (LR/SVM/RF/MLP),
#  the AUC of Base, Base+PGS526, Base+PGS616 side by side, in
#  BOTH the training cohort (5×20 CV mean, 95% CI) and PMBB
#  external (bootstrap 95% CI).  Makes explicit that the age/sex
#  baseline is already strong and PGS adds little within-classifier.
#
#  Output: outputs/tables/Table_SameClassifier_AUC.xlsx
# =============================================================

import os as _os
import numpy as np
import pandas as pd
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
BASE_COLS = ["Age", "Gender"]
FS = {"Base": BASE_COLS,
      "Base+PGS526": BASE_COLS + ["PGS526"],
      "Base+PGS616": BASE_COLS + ["PGS616"]}


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

# ---- Training 5x20 CV mean AUC + 95% CI (normal approx) ------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
folds = list(rskf.split(np.zeros(len(y_tr)), y_tr))
train_auc = {m: {fs: [] for fs in FS} for m in MODEL_NAMES}
for m in MODEL_NAMES:
    for (tri, tei) in folds:
        yt = y_tr[tei]
        if len(np.unique(yt)) < 2:
            continue
        for fs, cols in FS.items():
            X = tr[cols].values
            pipe = make_pipeline(m)
            pipe.fit(X[tri], y_tr[tri])
            yp = pipe.predict_proba(X[tei])[:, 1]
            train_auc[m][fs].append(roc_auc_score(yt, yp))

# ---- PMBB external AUC + bootstrap 95% CI --------------------
def pmbb_mat(fs):
    ct, cp = ["Age", "Gender"], ["PMBB_3.0_Release_AGE", "SEX_bin"]
    if fs == "Base+PGS616":
        ct += ["PGS616"]; cp += ["PGS616"]
    elif fs == "Base+PGS526":
        ct += ["PGS526"]; cp += ["PGS526"]
    return tr[ct].values, pmbb[cp].values

rng = np.random.RandomState(SEED)
boot_idx = [rng.choice(len(y_pmbb), len(y_pmbb), replace=True)
            for _ in range(N_BOOT)]
pmbb_auc = {m: {} for m in MODEL_NAMES}
for m in MODEL_NAMES:
    for fs in FS:
        Xt, Xp = pmbb_mat(fs)
        pipe = make_pipeline(m); pipe.fit(Xt, y_tr)
        yp = pipe.predict_proba(Xp)[:, 1]
        point = roc_auc_score(y_pmbb, yp)
        bs = [roc_auc_score(y_pmbb[i], yp[i]) for i in boot_idx
              if len(np.unique(y_pmbb[i])) > 1]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pmbb_auc[m][fs] = (point, lo, hi)

# ---- assemble table -----------------------------------------
rows = []
for m in MODEL_NAMES:
    for fs in FS:
        a = np.array(train_auc[m][fs])
        mn, sd, n = a.mean(), a.std(ddof=1), len(a)
        tlo, thi = mn - 1.96*sd/np.sqrt(n), mn + 1.96*sd/np.sqrt(n)
        pt, plo, phi = pmbb_auc[m][fs]
        rows.append({
            "Classifier": m, "FeatureSet": fs,
            "Train_AUC": round(mn, 3),
            "Train_95CI": f"{mn:.3f} ({tlo:.3f}-{thi:.3f})",
            "PMBB_AUC": round(pt, 3),
            "PMBB_95CI": f"{pt:.3f} ({plo:.3f}-{phi:.3f})",
        })
tbl = pd.DataFrame(rows)

# add ΔAUC-vs-Base column within classifier
tbl["Train_dAUC_vs_Base"] = np.nan
tbl["PMBB_dAUC_vs_Base"]  = np.nan
for m in MODEL_NAMES:
    base_tr = tbl[(tbl.Classifier == m) & (tbl.FeatureSet == "Base")]["Train_AUC"].iloc[0]
    base_pm = tbl[(tbl.Classifier == m) & (tbl.FeatureSet == "Base")]["PMBB_AUC"].iloc[0]
    mask = tbl.Classifier == m
    tbl.loc[mask, "Train_dAUC_vs_Base"] = (tbl.loc[mask, "Train_AUC"] - base_tr).round(3)
    tbl.loc[mask, "PMBB_dAUC_vs_Base"]  = (tbl.loc[mask, "PMBB_AUC"]  - base_pm).round(3)

# wide pivot for readability (AUC with CI)
piv_train = tbl.pivot(index="Classifier", columns="FeatureSet",
                      values="Train_95CI").reindex(MODEL_NAMES)
piv_pmbb  = tbl.pivot(index="Classifier", columns="FeatureSet",
                      values="PMBB_95CI").reindex(MODEL_NAMES)
col_order = ["Base", "Base+PGS526", "Base+PGS616"]
piv_train = piv_train[col_order]
piv_pmbb  = piv_pmbb[col_order]

with pd.ExcelWriter(_os.path.join(OUT_XL, "Table_SameClassifier_AUC.xlsx"),
                    engine="openpyxl") as w:
    piv_train.to_excel(w, sheet_name="Training_5x20CV_AUC")
    piv_pmbb.to_excel(w, sheet_name="PMBB_External_AUC")
    tbl.to_excel(w, sheet_name="Long_form", index=False)

print("\n=== SAME-CLASSIFIER AUC (Training 5x20 CV) ===")
print(piv_train.to_string())
print("\n=== SAME-CLASSIFIER AUC (PMBB external) ===")
print(piv_pmbb.to_string())
print("\nBaseline (Base, age+sex) already reaches, per classifier:")
for m in MODEL_NAMES:
    b = tbl[(tbl.Classifier==m)&(tbl.FeatureSet=="Base")]
    print(f"  {m}: train {b['Train_AUC'].iloc[0]:.3f}, PMBB {b['PMBB_AUC'].iloc[0]:.3f}")
print("\nSaved Table_SameClassifier_AUC.xlsx")
