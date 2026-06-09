# =============================================================
#  Figure 5 — Phenotypic Asymmetry Analysis
#  Features: delta_IOP, delta_CDR (|OD − OS|)  [RNFL excluded
#             so features are consistent across all 3 cohorts]
#
#  Step 1: 5×20 CV on POAAGG 271 training cohort
#  Step 2: Apply to POAAGG 1013 suspects → enrichment
#  Step 3: Apply to PMBB AFR (N~3,281) → external AUC validation
#          (PMBB has case/control labels; first-visit IOP & CDR used)
#
#  Export all metrics to Excel (no figures yet)
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

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
TRAIN_F = fr"{BASE}\input-data\POAAGG_cohort\271_training_cohort_4_new_PRS_cleaned.xlsx"
SUSP_F  = fr"{BASE}\input-data\POAAGG_cohort\1013_testing_cohort_only_suspect_cleaned.xlsx"
PMBB_F  = fr"{BASE}\input-data\PMBB_external\PMBB_949_POAG_IOP_CDR_Freeze3.csv"
PMBB_MAIN_F = fr"{BASE}\input-data\PMBB_external\PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_updated_June8.csv"
PGS616_F    = fr"{BASE}\input-data\PMBB_external\PMBBv3_GRS_MEGA_616snps.sscore_withSTDscore.txt"
PGS526_F    = fr"{BASE}\input-data\PMBB_external\PMBBv3_GRS_QUANT_526snps.sscore_withSTDscore.txt"
OUT_XL  = fr"{BASE}\output excel data\Table_Figure5_Asymmetry.xlsx"

LABEL      = "CaseCtrl"
N_SPLITS   = 5
N_REPEATS  = 20
RNG        = 42

# ------------------------------------------------------------------
# Feature sets — delta_IOP and delta_CDR only
# Training col names → Suspect col names → PMBB col names
# ------------------------------------------------------------------
DELTA_TR = ["delta_IOP", "delta_CDR"]
DELTA_SU = ["delta_IOP_su", "delta_CDR_su"]   # computed below
DELTA_PM = ["delta_IOP_pm", "delta_CDR_pm"]   # computed below

FEATURE_SETS = {
    "Delta":            {"tr": DELTA_TR,                     "su": DELTA_SU,               "pm": DELTA_PM},
    "Delta+POAAGG PGS": {"tr": DELTA_TR + ["POAAGG PGS"],    "su": DELTA_SU + ["POAAGG PGS"], "pm": None},
    "Delta+MEGA PGS":   {"tr": DELTA_TR + ["MEGA PGS"],      "su": DELTA_SU + ["MEGA PGS"],   "pm": None},
    "Delta+PGS526":     {"tr": DELTA_TR + ["PGS526"],        "su": DELTA_SU + ["PGS526"],      "pm": DELTA_PM + ["PGS526_pm"]},
    "Delta+PGS616":     {"tr": DELTA_TR + ["PGS616"],        "su": DELTA_SU + ["PGS616"],      "pm": DELTA_PM + ["PGS616_pm"]},
}

MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]


def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RNG))]
    elif name == "SVM":
        steps += [("clf", SVC(kernel="rbf", probability=True,
                              class_weight="balanced", random_state=RNG))]
    elif name == "RF":
        steps += [("clf", RandomForestClassifier(
            n_estimators=200, max_depth=5,
            class_weight="balanced", random_state=RNG))]
    elif name == "MLP":
        steps += [("clf", MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=1000,
            early_stopping=False, random_state=RNG))]
    return Pipeline(steps)


# ==================================================================
# 1.  LOAD & PREPARE DATA
# ==================================================================
print("=" * 60)
print("Loading POAAGG training cohort ...")
tr = pd.read_excel(TRAIN_F)
su = pd.read_excel(SUSP_F)
y_tr = tr[LABEL].values.astype(int)
print(f"  Train N={len(tr)}  cases={y_tr.sum()}  ctrl={(y_tr==0).sum()}")
print(f"  Suspects N={len(su)}")

# Delta for suspects (absolute inter-eye difference)
su["delta_IOP_su"] = (su["OD Baseline IOP (mmHg)"] - su["OS Baseline IOP (mmHg)"]).abs()
su["delta_CDR_su"] = (su["OD Baseline CDR"]         - su["OS Baseline CDR"]).abs()

print("\nSuspect delta stats:")
for col in DELTA_SU:
    n = su[col].notna().sum()
    print(f"  {col}: N={n} ({n/len(su)*100:.1f}%)  mean={su[col].mean():.3f}")

print("\nTraining delta stats:")
for col in DELTA_TR:
    n = tr[col].notna().sum()
    print(f"  {col}: N={n}  mean={tr[col].mean():.3f}")

# ------------------------------------------------------------------
# Load and prepare PMBB external cohort
# ------------------------------------------------------------------
print("\nLoading PMBB external cohort ...")
pmbb_raw  = pd.read_csv(PMBB_F)
pmbb_main = pd.read_csv(PMBB_MAIN_F)
pgs616_df = pd.read_csv(PGS616_F, sep="\t")
pgs526_df = pd.read_csv(PGS526_F, sep="\t")

# AFR only
pmbb_afr = pmbb_main[pmbb_main["ANCESTRY"] == "AFR"].copy()

# Clean CDR value: some entries are "0.6, 0.5" — take first number
pmbb_raw["value_clean"] = (pmbb_raw["pheno_value"].astype(str)
                           .str.split(",").str[0].str.strip())
pmbb_raw["value_num"] = pd.to_numeric(pmbb_raw["value_clean"], errors="coerce")

# Keep only AFR individuals
pmbb_raw = pmbb_raw[pmbb_raw["PMBB_ID"].isin(pmbb_afr["PMBB_ID"])]

# First visit per (ID, eye, type)
pmbb_raw["pheno_date"] = pd.to_datetime(pmbb_raw["pheno_date"], errors="coerce")
first = (pmbb_raw.sort_values("pheno_date")
         .groupby(["PMBB_ID", "pheno_eye", "pheno_type"])
         .first()
         .reset_index())

# Pivot to wide: OD_IOP, OS_IOP, OD_CDR, OS_CDR
wide = first.pivot_table(index="PMBB_ID",
                          columns=["pheno_eye", "pheno_type"],
                          values="value_num")
wide.columns = ["_".join(c) for c in wide.columns]
wide = wide.reset_index()

# Compute deltas
wide["delta_IOP_pm"] = (wide["OD_IOP"] - wide["OS_IOP"]).abs()
wide["delta_CDR_pm"] = (wide["OD_CDR"] - wide["OS_CDR"]).abs()

# Merge covariates
pmbb = wide.merge(
    pmbb_afr[["PMBB_ID", "POAG_cases", "PMBB_3.0_Release_AGE", "SEX"]],
    on="PMBB_ID", how="inner")

# Merge PGS (join on IID = PMBB_ID)
pgs616_df = pgs616_df.rename(columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": "PGS616_pm"})
pgs526_df = pgs526_df.rename(columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": "PGS526_pm"})
pmbb = pmbb.merge(pgs616_df[["PMBB_ID", "PGS616_pm"]], on="PMBB_ID", how="left")
pmbb = pmbb.merge(pgs526_df[["PMBB_ID", "PGS526_pm"]], on="PMBB_ID", how="left")

y_pm = pmbb["POAG_cases"].values.astype(int)

print(f"  PMBB AFR total with IOP/CDR: N={len(pmbb)}")
print(f"  Cases={y_pm.sum()}  Controls={(y_pm==0).sum()}")
print(f"  delta_IOP_pm: N={pmbb['delta_IOP_pm'].notna().sum()}")
print(f"  delta_CDR_pm: N={pmbb['delta_CDR_pm'].notna().sum()}")
print(f"  Both deltas:  N={(pmbb['delta_IOP_pm'].notna() & pmbb['delta_CDR_pm'].notna()).sum()}")
print(f"  PGS616_pm:    N={pmbb['PGS616_pm'].notna().sum()}")
print(f"  PGS526_pm:    N={pmbb['PGS526_pm'].notna().sum()}")


# Outcome columns for suspect enrichment
DELTA_OUTCOMES = {
    "ΔIOP (mmHg)": "delta_IOP_su",
    "ΔCDR":        "delta_CDR_su",
}
SEVERE_OUTCOMES = {
    "IOP_SEVERE":  "IOP_SEVERE",
    "CDR_SEVERE":  "CDR_SEVERE",
    "RNFL_SEVERE": "RNFL_SEVERE",
}


# ==================================================================
# 2.  5×20 CV — AUC on POAAGG training cohort
# ==================================================================
print("\n" + "=" * 60)
print("5x20 CV on POAAGG training cohort ...")
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                              random_state=RNG)
cv_rows = []
for fs_name, cfg in FEATURE_SETS.items():
    X_tr = tr[cfg["tr"]].values
    for mn in MODEL_NAMES:
        pipe  = make_pipeline(mn)
        aucs  = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                scoring="roc_auc", n_jobs=-1)
        mean_auc = float(aucs.mean())
        std_auc  = float(aucs.std())
        se       = std_auc / np.sqrt(len(aucs))
        cv_rows.append({
            "FeatureSet": fs_name, "Model": mn,
            "N_CV_fits":  len(aucs),
            "Mean_AUC":   round(mean_auc, 4),
            "Std_AUC":    round(std_auc, 4),
            "SE":         round(se, 4),
            "CI_lo_95":   round(mean_auc - 1.96 * se, 4),
            "CI_hi_95":   round(mean_auc + 1.96 * se, 4),
        })
        print(f"  {fs_name} x {mn}: {mean_auc:.4f} +/- {std_auc:.4f} "
              f"(95% CI {mean_auc-1.96*se:.4f}-{mean_auc+1.96*se:.4f})")

cv_df = pd.DataFrame(cv_rows)


# ==================================================================
# 3.  TRAIN ON 271, APPLY TO SUSPECTS & PMBB
# ==================================================================
print("\n" + "=" * 60)
print("Training on POAAGG 271 and applying to suspects + PMBB ...")

pred_su   = {}   # (fs, model) -> scores for 1013 suspects
pred_pm   = {}   # (fs, model) -> scores for PMBB
score_log = []

for fs_name, cfg in FEATURE_SETS.items():
    X_tr = tr[cfg["tr"]].values
    X_su = su[cfg["su"]].values

    # PMBB columns (None if not applicable)
    pm_cols  = cfg["pm"]
    X_pm     = pmbb[pm_cols].values if pm_cols else None

    for mn in MODEL_NAMES:
        pipe = make_pipeline(mn)
        pipe.fit(X_tr, y_tr)

        # Suspects
        sc_su = pipe.predict_proba(X_su)[:, 1]
        pred_su[(fs_name, mn)] = sc_su

        # PMBB
        if X_pm is not None:
            sc_pm = pipe.predict_proba(X_pm)[:, 1]
            pred_pm[(fs_name, mn)] = sc_pm
            pm_auc = roc_auc_score(y_pm, sc_pm)
            print(f"  {fs_name} x {mn} -> suspect score {sc_su.mean():.3f}+/-{sc_su.std():.3f} | PMBB AUC={pm_auc:.4f}")
            score_log.append({
                "FeatureSet": fs_name, "Model": mn,
                "Suspect_score_mean": round(float(sc_su.mean()), 4),
                "Suspect_score_std":  round(float(sc_su.std()), 4),
                "PMBB_AUC": round(pm_auc, 4),
            })
        else:
            print(f"  {fs_name} x {mn} -> suspect score {sc_su.mean():.3f}+/-{sc_su.std():.3f} | PMBB: skipped (no PMBB PGS)")
            score_log.append({
                "FeatureSet": fs_name, "Model": mn,
                "Suspect_score_mean": round(float(sc_su.mean()), 4),
                "Suspect_score_std":  round(float(sc_su.std()), 4),
                "PMBB_AUC": None,
            })


# ==================================================================
# 4.  PMBB EXTERNAL AUC — bootstrap 95% CI
# ==================================================================
print("\n" + "=" * 60)
print("PMBB external AUC with bootstrap 95% CI ...")
np.random.seed(RNG)
N_BOOT = 1000

pmbb_auc_rows = []
for (fs_name, mn), sc_pm in pred_pm.items():
    # point estimate
    auc_pt = roc_auc_score(y_pm, sc_pm)
    # bootstrap
    boot_aucs = []
    for _ in range(N_BOOT):
        idx = np.random.choice(len(y_pm), len(y_pm), replace=True)
        if y_pm[idx].sum() == 0 or y_pm[idx].sum() == len(idx):
            continue
        boot_aucs.append(roc_auc_score(y_pm[idx], sc_pm[idx]))
    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))
    pmbb_auc_rows.append({
        "FeatureSet": fs_name, "Model": mn,
        "PMBB_N":     len(y_pm),
        "PMBB_Cases": int(y_pm.sum()),
        "AUC":        round(auc_pt, 4),
        "CI_lo_95":   round(ci_lo, 4),
        "CI_hi_95":   round(ci_hi, 4),
    })
    print(f"  {fs_name} x {mn}: AUC={auc_pt:.4f} (95% CI {ci_lo:.4f}-{ci_hi:.4f})")

pmbb_auc_df = pd.DataFrame(pmbb_auc_rows)


# ==================================================================
# 5.  SUSPECT CORRELATIONS — predicted risk vs delta outcomes
# ==================================================================
print("\n" + "=" * 60)
print("Suspect correlations ...")
corr_rows = []
all_outcomes = {**DELTA_OUTCOMES, **SEVERE_OUTCOMES}

for (fs_name, mn), sc in pred_su.items():
    for out_label, col in all_outcomes.items():
        valid = su[col].notna().values
        if valid.sum() < 10:
            continue
        r_val, p_val = stats.pearsonr(sc[valid], su[col].values[valid])
        sig = ("***" if p_val < 0.001 else
               ("**" if p_val < 0.01 else
               ("*"  if p_val < 0.05 else "ns")))
        corr_rows.append({
            "FeatureSet": fs_name, "Model": mn,
            "Outcome": out_label, "N": int(valid.sum()),
            "Pearson_r": round(r_val, 4),
            "p_value":   round(p_val, 6),
            "Sig":       sig,
        })

corr_df = pd.DataFrame(corr_rows)

print("\nMLP x all feature sets (delta outcomes):")
sub = corr_df[(corr_df["Model"] == "MLP") &
              corr_df["Outcome"].isin(DELTA_OUTCOMES.keys())]
print(sub[["FeatureSet","Outcome","Pearson_r","p_value","Sig","N"]].to_string(index=False))


# ==================================================================
# 6.  QUINTILE ENRICHMENT — MLP x Delta+PGS616
# ==================================================================
print("\n" + "=" * 60)
print("Quintile enrichment (MLP x Delta+PGS616) ...")
mlp_scores   = pred_su[("Delta+PGS616", "MLP")]
quint_labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
quintiles    = pd.Categorical(pd.qcut(mlp_scores, 5, labels=quint_labels))

quint_rows = []
for out_label, col in all_outcomes.items():
    for q in quint_labels:
        mask = np.array(quintiles == q) & su[col].notna().values
        vals = su[col].values[mask]
        if len(vals) == 0:
            continue
        quint_rows.append({
            "Outcome":  out_label, "Quintile": q,
            "N":        len(vals),
            "Mean":     round(float(vals.mean()), 4),
            "SEM":      round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 4),
        })

quint_df = pd.DataFrame(quint_rows)
print(quint_df[quint_df["Outcome"].isin(DELTA_OUTCOMES.keys())].to_string(index=False))


# ==================================================================
# 7.  KS TEST — low 25% vs high 25% MLP risk (suspects)
# ==================================================================
print("\n" + "=" * 60)
print("KS test: low vs high 25% MLP risk (Delta+PGS616, suspects) ...")
p25 = np.percentile(mlp_scores, 25)
p75 = np.percentile(mlp_scores, 75)
mask_lo = mlp_scores <= p25
mask_hi = mlp_scores >= p75
print(f"  Low 25%: n={mask_lo.sum()}, High 25%: n={mask_hi.sum()}")

ks_rows = []
for out_label, col in all_outcomes.items():
    valid = su[col].notna().values
    v_lo  = su[col].values[mask_lo & valid]
    v_hi  = su[col].values[mask_hi & valid]
    if len(v_lo) < 5 or len(v_hi) < 5:
        continue
    ks_stat, ks_p = stats.ks_2samp(v_lo, v_hi)
    sig = ("***" if ks_p < 0.001 else
           ("**" if ks_p < 0.01 else
           ("*"  if ks_p < 0.05 else "ns")))
    ks_rows.append({
        "Outcome": out_label,
        "N_lo": len(v_lo), "N_hi": len(v_hi),
        "Mean_lo": round(float(v_lo.mean()), 4),
        "Mean_hi": round(float(v_hi.mean()), 4),
        "KS_stat": round(ks_stat, 4),
        "KS_p":    round(ks_p, 6),
        "Sig":     sig,
    })
    print(f"  {out_label}: lo={v_lo.mean():.3f}  hi={v_hi.mean():.3f}  "
          f"KS={ks_stat:.4f}  p={ks_p:.6f}  {sig}")

ks_df = pd.DataFrame(ks_rows)


# ==================================================================
# 8.  EXPORT TO EXCEL
# ==================================================================
print("\nSaving Excel ...")
with pd.ExcelWriter(OUT_XL, engine="openpyxl") as w:

    # --- CV AUC (training) ---
    cv_df.to_excel(w, sheet_name="CV_AUC_Training", index=False)

    piv_auc = cv_df.copy()
    piv_auc["AUC_summary"] = (piv_auc["Mean_AUC"].map(lambda x: f"{x:.4f}") + " (" +
                               piv_auc["CI_lo_95"].map(lambda x: f"{x:.4f}") + "-" +
                               piv_auc["CI_hi_95"].map(lambda x: f"{x:.4f}") + ")")
    piv_wide = piv_auc.pivot(index="FeatureSet", columns="Model",
                              values="AUC_summary")
    piv_wide = piv_wide.reindex(index=list(FEATURE_SETS.keys()),
                                 columns=MODEL_NAMES)
    piv_wide.to_excel(w, sheet_name="CV_AUC_Pivot")

    # --- PMBB external AUC ---
    pmbb_auc_df.to_excel(w, sheet_name="PMBB_External_AUC", index=False)

    piv_pm = pmbb_auc_df.copy()
    piv_pm["AUC_summary"] = (piv_pm["AUC"].map(lambda x: f"{x:.4f}") + " (" +
                              piv_pm["CI_lo_95"].map(lambda x: f"{x:.4f}") + "-" +
                              piv_pm["CI_hi_95"].map(lambda x: f"{x:.4f}") + ")")
    piv_pm_wide = piv_pm.pivot(index="FeatureSet", columns="Model",
                                values="AUC_summary")
    piv_pm_wide.to_excel(w, sheet_name="PMBB_AUC_Pivot")

    # --- Suspect correlations ---
    corr_df.to_excel(w, sheet_name="Suspect_Correlations", index=False)

    delta_corr = corr_df[corr_df["Outcome"].isin(DELTA_OUTCOMES.keys())].copy()
    delta_corr["r_sig"] = (delta_corr["Pearson_r"].map(lambda x: f"{x:.3f}") +
                            " " + delta_corr["Sig"])
    piv_corr = delta_corr.pivot_table(
        index=["FeatureSet", "Model"], columns="Outcome",
        values="r_sig", aggfunc="first")
    piv_corr.to_excel(w, sheet_name="Delta_Correlation_Pivot")

    # --- Quintile enrichment ---
    quint_df.to_excel(w, sheet_name="Quintile_MLP_Delta_PGS616", index=False)

    # --- KS test ---
    ks_df.to_excel(w, sheet_name="KS_LowVsHigh_25pct", index=False)

    # --- Score stats ---
    pd.DataFrame(score_log).to_excel(w, sheet_name="Score_Statistics", index=False)

print(f"  Saved: {OUT_XL}")


# ==================================================================
# 9.  FINAL SUMMARY
# ==================================================================
print("\n" + "=" * 60)
print("FULL CV AUC SUMMARY (POAAGG training, 5x20 CV):")
print(cv_df[["FeatureSet","Model","Mean_AUC","CI_lo_95","CI_hi_95"]].to_string(index=False))

print("\nPMBB EXTERNAL AUC SUMMARY:")
print(pmbb_auc_df[["FeatureSet","Model","AUC","CI_lo_95","CI_hi_95"]].to_string(index=False))

print("\nBEST training config:", cv_df.loc[cv_df["Mean_AUC"].idxmax(), ["FeatureSet","Model","Mean_AUC"]].to_dict())
print("BEST PMBB config:    ", pmbb_auc_df.loc[pmbb_auc_df["AUC"].idxmax(), ["FeatureSet","Model","AUC"]].to_dict())

print("\nAll done.")
