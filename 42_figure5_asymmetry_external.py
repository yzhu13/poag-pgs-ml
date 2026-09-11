# =============================================================
#  POAG - Figure 5C / Table S18 external asymmetry analysis
#  iScience revision, 2026-09-05
#
#  Recovered from _archive/R1_work_2026-05-08/code/analysis_figure5_sep.py.
#  This analysis produces the PMBB columns of Table S18 and all of Figure
#  5C, and is the only source for them, but it was never carried into
#  _working/code or into the public repository: nothing in github/
#  reproduces the "IOP_PHE" feature sets or the external asymmetry AUCs.
#
#  Two corrections applied here, as elsewhere this round:
#    1. PGS616 rank-INT over the pooled POAAGG participants
#    2. the stated PMBB age >= 35 restriction
#
#  A third change is specific to this script. It read the PGS from
#  PMBBv3_GRS_*snps.sscore_withSTDscore.txt, whose SCORE1_AVG_STD is an
#  inverse-normal transform computed within 2,594 samples (bounds
#  +/-3.550), while every other analysis reads the _AllSamples file,
#  transformed within 56,357 (bounds +/-4.292). Two different definitions
#  of the same score were in use across the figures. This now reads
#  _AllSamples, matching the rest.
# =============================================================

# =============================================================
#  Figure 5B/5C — Separate delta-feature models
#  Each delta (IOP, CDR) trained independently + 4 PGS options
#  5x20 CV on POAAGG 271  +  PMBB external AUC
#  Appends new sheets to Table_Figure5_Asymmetry.xlsx
# =============================================================
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import openpyxl

import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
from poag_paths import DATA_DIR, TRAINING_FILE, SUSPECT_FILE
from poag_corrections import int_pgs616, restrict_pmbb_age, recompute_training_deltas

_PM = _os.path.join(DATA_DIR, "PMBB_external")
TRAIN_F  = TRAINING_FILE
SUSP_F   = SUSPECT_FILE
PMBB_F   = _os.path.join(_PM, "PMBB_949_POAG_IOP_CDR_Freeze3.csv")
PMBB_MF  = _os.path.join(
    _PM, "PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_updated_June8.csv")
# _AllSamples, to match every other analysis (see the header note)
PGS616_F = _os.path.join(
    _PM, "PMBBv3_GRS_MEGA_616snps_AllSamples.sscore_withSTDscore.txt")
PGS526_F = _os.path.join(
    _PM, "PMBBv3_GRS_QUANT_526snps_AllSamples.sscore_withSTDscore.txt")
OUT_XL   = _os.path.join(_HERE, "outputs", "tables",
                         "Table_Figure5_Asymmetry.xlsx")

LABEL    = "CaseCtrl"
N_SPLITS = 5
N_REPEATS = 20
RNG      = 42
N_BOOT   = 2000   # was 1000; STAR Methods states 2,000 externally

# ------------------------------------------------------------------
# Feature sets — delta_IOP and delta_CDR trained SEPARATELY
# Group label | tr cols | su cols | pm cols (None = skip PMBB)
# ------------------------------------------------------------------
FEATURE_SETS_SEP = {
    # IOP group
    "IOP_PHE":    {"delta": "IOP", "tr": ["delta_IOP"],               "su": ["delta_IOP_su"],                    "pm": ["delta_IOP_pm"]},
    "IOP+POAAGG": {"delta": "IOP", "tr": ["delta_IOP","POAAGG PGS"],  "su": ["delta_IOP_su","POAAGG PGS"],    "pm": None},
    "IOP+MEGA":   {"delta": "IOP", "tr": ["delta_IOP","MEGA PGS"],    "su": ["delta_IOP_su","MEGA PGS"],      "pm": None},
    "IOP+PGS526": {"delta": "IOP", "tr": ["delta_IOP","PGS526"],      "su": ["delta_IOP_su","PGS526"],         "pm": ["delta_IOP_pm","PGS526_pm"]},
    "IOP+PGS616": {"delta": "IOP", "tr": ["delta_IOP","PGS616"],      "su": ["delta_IOP_su","PGS616"],         "pm": ["delta_IOP_pm","PGS616_pm"]},
    # CDR group
    "CDR_PHE":    {"delta": "CDR", "tr": ["delta_CDR"],               "su": ["delta_CDR_su"],                    "pm": ["delta_CDR_pm"]},
    "CDR+POAAGG": {"delta": "CDR", "tr": ["delta_CDR","POAAGG PGS"],  "su": ["delta_CDR_su","POAAGG PGS"],    "pm": None},
    "CDR+MEGA":   {"delta": "CDR", "tr": ["delta_CDR","MEGA PGS"],    "su": ["delta_CDR_su","MEGA PGS"],      "pm": None},
    "CDR+PGS526": {"delta": "CDR", "tr": ["delta_CDR","PGS526"],      "su": ["delta_CDR_su","PGS526"],         "pm": ["delta_CDR_pm","PGS526_pm"]},
    "CDR+PGS616": {"delta": "CDR", "tr": ["delta_CDR","PGS616"],      "su": ["delta_CDR_su","PGS616"],         "pm": ["delta_CDR_pm","PGS616_pm"]},
}
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]

# Display labels for the 5 bars within each delta group
PGS_SLOT_LABELS = {
    "PHE":    "PHE\nOnly",
    "POAAGG": "PHE+\nPOAAGG",
    "MEGA":   "PHE+\nMEGA",
    "PGS526": "PHE+\nPGS526",
    "PGS616": "PHE+\nPGS616",
}
PGS_SLOT_COLORS = {
    "PHE":    "#888888",
    "POAAGG": "#7B9EC9",
    "MEGA":   "#E07B54",
    "PGS526": "#5BAD72",
    "PGS616": "#2196A6",
}

def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RNG))]
    elif name == "SVM":
        steps += [("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG))]
    elif name == "RF":
        steps += [("clf", RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=RNG))]
    elif name == "MLP":
        steps += [("clf", MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, early_stopping=False, random_state=RNG))]
    return Pipeline(steps)


# ==================================================================
# Load data
# ==================================================================
print("Loading data ...")
tr = pd.read_excel(TRAIN_F)
su = pd.read_excel(SUSP_F)
tr, su = int_pgs616(tr, su)
tr = recompute_training_deltas(tr)      # 2026-09-10 audit B02
y_tr = tr[LABEL].values.astype(int)

su["delta_IOP_su"] = (su["OD Baseline IOP (mmHg)"] - su["OS Baseline IOP (mmHg)"]).abs()
su["delta_CDR_su"] = (su["OD Baseline CDR"]         - su["OS Baseline CDR"]).abs()

# PMBB
pmbb_raw  = pd.read_csv(PMBB_F)
pmbb_main = pd.read_csv(PMBB_MF)
pgs616_df = pd.read_csv(PGS616_F, sep="\t").rename(columns={"IID":"PMBB_ID","SCORE1_AVG_STD":"PGS616_pm"})
pgs526_df = pd.read_csv(PGS526_F, sep="\t").rename(columns={"IID":"PMBB_ID","SCORE1_AVG_STD":"PGS526_pm"})

pmbb_afr  = pmbb_main[pmbb_main["ANCESTRY"] == "AFR"].copy()
pmbb_afr  = restrict_pmbb_age(pmbb_afr)
pmbb_raw["value_clean"] = pmbb_raw["pheno_value"].astype(str).str.split(",").str[0].str.strip()
pmbb_raw["value_num"]   = pd.to_numeric(pmbb_raw["value_clean"], errors="coerce")
pmbb_raw = pmbb_raw[pmbb_raw["PMBB_ID"].isin(pmbb_afr["PMBB_ID"])]
pmbb_raw["pheno_date"]  = pd.to_datetime(pmbb_raw["pheno_date"], errors="coerce")
first = (pmbb_raw.sort_values("pheno_date")
         .groupby(["PMBB_ID","pheno_eye","pheno_type"]).first().reset_index())
wide  = first.pivot_table(index="PMBB_ID", columns=["pheno_eye","pheno_type"],
                           values="value_num")
wide.columns = ["_".join(c) for c in wide.columns]
wide = wide.reset_index()
wide["delta_IOP_pm"] = (wide["OD_IOP"] - wide["OS_IOP"]).abs()
wide["delta_CDR_pm"] = (wide["OD_CDR"] - wide["OS_CDR"]).abs()

pmbb = (wide
        .merge(pmbb_afr[["PMBB_ID","POAG_cases"]], on="PMBB_ID", how="inner")
        .merge(pgs616_df[["PMBB_ID","PGS616_pm"]], on="PMBB_ID", how="left")
        .merge(pgs526_df[["PMBB_ID","PGS526_pm"]], on="PMBB_ID", how="left"))
y_pm = pmbb["POAG_cases"].values.astype(int)
print(f"  Train N={len(tr)}, Suspects N={len(su)}, PMBB AFR N={len(pmbb)} (cases={y_pm.sum()})")


# ==================================================================
# 5x20 CV — training AUC per separate delta feature
# ==================================================================
print("\n5x20 CV (separate delta features) ...")
cv  = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RNG)
cv_rows = []

for fs_name, cfg in FEATURE_SETS_SEP.items():
    X_tr = tr[cfg["tr"]].values
    for mn in MODEL_NAMES:
        aucs = cross_val_score(make_pipeline(mn), X_tr, y_tr,
                               cv=cv, scoring="roc_auc", n_jobs=-1)
        m, s = aucs.mean(), aucs.std()
        se   = s / np.sqrt(len(aucs))
        cv_rows.append({
            "FeatureSet": fs_name, "DeltaGroup": cfg["delta"], "Model": mn,
            "Mean_AUC": round(m, 4), "Std_AUC": round(s, 4),
            "SE": round(se, 4),
            "CI_lo_95": round(m - 1.96*se, 4),
            "CI_hi_95": round(m + 1.96*se, 4),
        })
        print(f"  {fs_name} x {mn}: {m:.4f} +/- {s:.4f}")

cv_sep_df = pd.DataFrame(cv_rows)


# ==================================================================
# PMBB external AUC — separate delta features
# ==================================================================
print("\nPMBB external AUC (separate delta features) ...")
np.random.seed(RNG)
pmbb_rows = []
scores_pm = {}

for fs_name, cfg in FEATURE_SETS_SEP.items():
    if cfg["pm"] is None:
        continue
    X_tr = tr[cfg["tr"]].values
    X_pm = pmbb[cfg["pm"]].values

    for mn in MODEL_NAMES:
        pipe = make_pipeline(mn)
        pipe.fit(X_tr, y_tr)
        sc_pm = pipe.predict_proba(X_pm)[:, 1]
        auc_pt = roc_auc_score(y_pm, sc_pm)
        scores_pm[(fs_name, mn)] = sc_pm

        # (2026-09-10: a first list comprehension here drew and discarded
        # 4,000 bootstrap samples per model before the loop below; removed)
        boot_aucs = []
        for _ in range(N_BOOT):
            idx = np.random.choice(len(y_pm), len(y_pm), replace=True)
            if y_pm[idx].sum() == 0 or y_pm[idx].sum() == len(idx):
                continue
            boot_aucs.append(roc_auc_score(y_pm[idx], sc_pm[idx]))

        pmbb_rows.append({
            "FeatureSet": fs_name, "DeltaGroup": cfg["delta"], "Model": mn,
            "PMBB_N": len(y_pm), "PMBB_Cases": int(y_pm.sum()),
            "AUC": round(auc_pt, 4),
            "CI_lo_95": round(float(np.percentile(boot_aucs, 2.5)), 4),
            "CI_hi_95": round(float(np.percentile(boot_aucs, 97.5)), 4),
        })
        print(f"  {fs_name} x {mn}: AUC={auc_pt:.4f}")

pmbb_sep_df = pd.DataFrame(pmbb_rows)


# ==================================================================
# PMBB paired contrasts: PHE+PGS vs PHE-only, same classifier
# (2026-09-10 audit: Table S18 gave separate AUCs with overlapping
#  intervals but no paired test of the PGS increment)
# ==================================================================
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong(y, p1, p2):
    """DeLong test for two correlated AUCs: dAUC (p1 - p2), 95% CI, p."""
    order = (-np.asarray(y)).argsort(kind="mergesort")
    m = int(np.asarray(y).sum())
    P = np.vstack((np.asarray(p1)[order], np.asarray(p2)[order]))
    n = P.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in P])
    ty = np.array([_midrank(r[m:]) for r in P])
    tz = np.array([_midrank(r) for r in P])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    d = aucs[0] - aucs[1]
    se = np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 0.0))
    p = 2 * (1 - stats.norm.cdf(abs(d / se))) if se > 0 else 1.0
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se), float(p)


print("\nPMBB paired contrasts (PHE+PGS vs PHE-only) ...")
rs = np.random.RandomState(RNG)
pair_idx = [rs.choice(len(y_pm), len(y_pm), replace=True) for _ in range(N_BOOT)]
pair_rows = []
for dg in ("IOP", "CDR"):
    for pgs in ("PGS616", "PGS526"):
        for mn in MODEL_NAMES:
            pa = scores_pm[(f"{dg}+{pgs}", mn)]
            pb = scores_pm[(f"{dg}_PHE", mn)]
            d, lo, hi, p = delong(y_pm, pa, pb)
            db = [roc_auc_score(y_pm[i], pa[i]) - roc_auc_score(y_pm[i], pb[i])
                  for i in pair_idx if 0 < y_pm[i].sum() < len(i)]
            blo, bhi = np.percentile(db, [2.5, 97.5])
            pair_rows.append({
                "DeltaGroup": dg, "Comparison": f"PHE+{pgs} vs PHE-only",
                "Model": mn, "dAUC": round(d, 4),
                "DeLong_CI_lo": round(lo, 4), "DeLong_CI_hi": round(hi, 4),
                "DeLong_p": round(p, 4),
                "Boot_CI_lo": round(float(blo), 4),
                "Boot_CI_hi": round(float(bhi), 4)})
            print(f"  {dg} {pgs} {mn}: dAUC={d:+.4f} ({lo:+.4f},{hi:+.4f}) p={p:.3g}")
pmbb_pair_df = pd.DataFrame(pair_rows)


# ==================================================================
# Append new sheets to existing Excel
# ==================================================================
print("\nAppending to Excel ...")
# The original appended to a workbook an earlier step had created; run on
# its own it failed on a missing file. Write the file when it is absent.
_mode = "a" if _os.path.exists(OUT_XL) else "w"
_kw = {"if_sheet_exists": "replace"} if _mode == "a" else {}
with pd.ExcelWriter(OUT_XL, engine="openpyxl", mode=_mode, **_kw) as w:
    cv_sep_df.to_excel(w,   sheet_name="CV_AUC_Sep_Delta",   index=False)
    pmbb_sep_df.to_excel(w, sheet_name="PMBB_AUC_Sep_Delta", index=False)
    pmbb_pair_df.to_excel(w, sheet_name="PMBB_Paired_dAUC", index=False)

print(f"  Saved sheets: CV_AUC_Sep_Delta, PMBB_AUC_Sep_Delta, PMBB_Paired_dAUC")


# ==================================================================
# Summary
# ==================================================================
print("\n=== MLP Training AUC (separate delta) ===")
mlp_cv = cv_sep_df[cv_sep_df["Model"]=="MLP"][["FeatureSet","DeltaGroup","Mean_AUC","CI_lo_95","CI_hi_95"]]
print(mlp_cv.to_string(index=False))

print("\n=== MLP PMBB AUC (separate delta) ===")
mlp_pm = pmbb_sep_df[pmbb_sep_df["Model"]=="MLP"][["FeatureSet","DeltaGroup","AUC","CI_lo_95","CI_hi_95"]]
print(mlp_pm.to_string(index=False))

print("\nAll done.")
