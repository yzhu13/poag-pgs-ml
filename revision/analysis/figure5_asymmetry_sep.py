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

# ── Configure your project root ─────────────────────────────────────────
# Set BASE to the folder containing input-data/, output excel data/, figure/
# Default: auto-detected as the repo root (2 levels above this script).
BASE = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")
)
# To override manually, uncomment:
# BASE = r"C:\path	o\your\project"   # Windows
# BASE = "/path/to/your/project"          # macOS / Linux
TRAIN_F  = fr"{BASE}\input-data\POAAGG_cohort\271_training_cohort_4_new_PRS_cleaned.xlsx"
SUSP_F   = fr"{BASE}\input-data\POAAGG_cohort\1013_testing_cohort_only_suspect_cleaned.xlsx"
PMBB_F   = fr"{BASE}\input-data\PMBB_external\PMBB_949_POAG_IOP_CDR_Freeze3.csv"
PMBB_MF  = fr"{BASE}\input-data\PMBB_external\PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_updated_June8.csv"
PGS616_F = fr"{BASE}\input-data\PMBB_external\PMBBv3_GRS_MEGA_616snps.sscore_withSTDscore.txt"
PGS526_F = fr"{BASE}\input-data\PMBB_external\PMBBv3_GRS_QUANT_526snps.sscore_withSTDscore.txt"
OUT_XL   = fr"{BASE}\output excel data\Table_Figure5_Asymmetry.xlsx"

LABEL    = "CaseCtrl"
N_SPLITS = 5
N_REPEATS = 20
RNG      = 42
N_BOOT   = 1000

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
y_tr = tr[LABEL].values.astype(int)

su["delta_IOP_su"] = (su["OD Baseline IOP (mmHg)"] - su["OS Baseline IOP (mmHg)"]).abs()
su["delta_CDR_su"] = (su["OD Baseline CDR"]         - su["OS Baseline CDR"]).abs()

# PMBB
pmbb_raw  = pd.read_csv(PMBB_F)
pmbb_main = pd.read_csv(PMBB_MF)
pgs616_df = pd.read_csv(PGS616_F, sep="\t").rename(columns={"IID":"PMBB_ID","SCORE1_AVG_STD":"PGS616_pm"})
pgs526_df = pd.read_csv(PGS526_F, sep="\t").rename(columns={"IID":"PMBB_ID","SCORE1_AVG_STD":"PGS526_pm"})

pmbb_afr  = pmbb_main[pmbb_main["ANCESTRY"] == "AFR"].copy()
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

        boot = [roc_auc_score(y_pm[idx := np.random.choice(len(y_pm), len(y_pm), replace=True)],
                              sc_pm[idx])
                for _ in range(N_BOOT)
                if 0 < y_pm[np.random.choice(len(y_pm), len(y_pm), replace=True)].sum() < len(y_pm)]
        # redo properly
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
# Append new sheets to existing Excel
# ==================================================================
print("\nAppending to Excel ...")
with pd.ExcelWriter(OUT_XL, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as w:
    cv_sep_df.to_excel(w,   sheet_name="CV_AUC_Sep_Delta",   index=False)
    pmbb_sep_df.to_excel(w, sheet_name="PMBB_AUC_Sep_Delta", index=False)

print(f"  Saved sheets: CV_AUC_Sep_Delta, PMBB_AUC_Sep_Delta")


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
