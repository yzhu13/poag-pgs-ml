# Re-run everything AFTER CV (correlation, PMBB validation, figures, Excel)
# using already-computed CV results embedded here.
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, recall_score)
import warnings, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

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
PMBB_PHE = fr"{BASE}\input-data\PMBB_external\PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_updated_June8.csv"
PMBB_616 = fr"{BASE}\input-data\PMBB_external\PMBBv3_GRS_MEGA_616snps_AllSamples.sscore_withSTDscore.txt"
PMBB_526 = fr"{BASE}\input-data\PMBB_external\PMBBv3_GRS_QUANT_526snps_AllSamples.sscore_withSTDscore.txt"
OUT_XL   = fr"{BASE}\output excel data"
OUT_FIG  = fr"{BASE}\figure"

LABEL       = "CaseCtrl"
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
MODEL_ORDER = ["LR", "SVM", "RF", "MLP", "Average"]
MODEL_COLORS = {
    "LR": "#4C72B0", "SVM": "#DD8452",
    "RF": "#55A868", "MLP": "#C44E52", "Average": "#8172B3",
}

TRAIN_PGS_COLS = {
    "POAAGG PGS": "POAAGG PGS",
    "MEGA PGS":   "MEGA PGS",
    "PGS526":     "PGS526",
    "PGS616":     "PGS616",
}
BASE_COLS  = ["Age", "Gender"]
PC2_COLS   = ["PC1", "PC2"]
PC5_COLS   = ["PC1", "PC2", "PC3", "PC4", "PC5"]
PC10_COLS  = [f"PC{i}" for i in range(1, 11)]

FEATURE_SETS = {
    "Age only":          ["Age"],
    "Sex only":          ["Gender"],
    "Base":              BASE_COLS,
    "Base+PC2":          BASE_COLS + PC2_COLS,
    "Base+PC5":          BASE_COLS + PC5_COLS,
    "Base+PC10":         BASE_COLS + PC10_COLS,
    "Base+POAAGG PGS":   BASE_COLS + ["POAAGG PGS"],
    "Base+MEGA PGS":     BASE_COLS + ["MEGA PGS"],
    "Base+PGS526":       BASE_COLS + ["PGS526"],
    "Base+PGS616":       BASE_COLS + ["PGS616"],
    "Base+PC5+PGS526":   BASE_COLS + PC5_COLS + ["PGS526"],
    "Base+PC5+PGS616":   BASE_COLS + PC5_COLS + ["PGS616"],
}

PMBB_FEAT_SETS = {
    "Base":              {"tr_cols": ["Age","Gender"],
                         "pm_cols": ["PMBB_3.0_Release_AGE","SEX_bin"]},
    "Base+PGS526":       {"tr_cols": ["Age","Gender","PGS526"],
                         "pm_cols": ["PMBB_3.0_Release_AGE","SEX_bin","PGS526"]},
    "Base+PGS616":       {"tr_cols": ["Age","Gender","PGS616"],
                         "pm_cols": ["PMBB_3.0_Release_AGE","SEX_bin","PGS616"]},
    "Base+PC5":          {"tr_cols": ["Age","Gender"]+PC5_COLS,
                         "pm_cols": ["PMBB_3.0_Release_AGE","SEX_bin"]+PC5_COLS},
    "Base+PC5+PGS616":   {"tr_cols": ["Age","Gender"]+PC5_COLS+["PGS616"],
                         "pm_cols": ["PMBB_3.0_Release_AGE","SEX_bin"]+PC5_COLS+["PGS616"]},
    "Base+PC5+PGS526":   {"tr_cols": ["Age","Gender"]+PC5_COLS+["PGS526"],
                         "pm_cols": ["PMBB_3.0_Release_AGE","SEX_bin"]+PC5_COLS+["PGS526"]},
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})


def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]
    elif name == "SVM":
        steps += [("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))]
    elif name == "RF":
        steps += [("clf", RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=42))]
    elif name == "MLP":
        steps += [("clf", MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, early_stopping=False, random_state=42))]
    return Pipeline(steps)


def specificity(yt, yp):
    tn = np.sum((yt == 0) & (yp == 0))
    fp = np.sum((yt == 0) & (yp == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def summarise(vals):
    n, mn, sd = len(vals), np.mean(vals), np.std(vals, ddof=1)
    lo, hi = mn - 1.96*sd/np.sqrt(n), mn + 1.96*sd/np.sqrt(n)
    return dict(Mean=round(mn,4), SD=round(sd,4),
                CI_lower=round(lo,4), CI_upper=round(hi,4),
                Mean_SD=f"{mn:.3f} ± {sd:.3f}",
                Mean_95CI=f"{mn:.3f} ({lo:.3f}–{hi:.3f})")


def boot_auc(yt, ys, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        idx = rng.choice(len(yt), len(yt), replace=True)
        if len(np.unique(yt[idx])) > 1:
            bs.append(roc_auc_score(yt[idx], ys[idx]))
    bs = np.array(bs)
    return (round(float(bs.mean()),4),
            round(float(np.percentile(bs,2.5)),4),
            round(float(np.percentile(bs,97.5)),4))


# ── load data ─────────────────────────────────────────────────
print("Loading data ...")
tr   = pd.read_excel(TRAIN_F)
y_tr = tr[LABEL].values.astype(int)

pmbb_phe = pd.read_csv(PMBB_PHE)
p616 = (pd.read_csv(PMBB_616,sep="\t")[["IID","SCORE1_AVG_STD"]]
          .rename(columns={"IID":"PMBB_ID","SCORE1_AVG_STD":"PGS616"}))
p526 = (pd.read_csv(PMBB_526,sep="\t")[["IID","SCORE1_AVG_STD"]]
          .rename(columns={"IID":"PMBB_ID","SCORE1_AVG_STD":"PGS526"}))
pmbb = pmbb_phe.merge(p616,on="PMBB_ID").merge(p526,on="PMBB_ID")
pmbb = pmbb[pmbb["ANCESTRY"]=="AFR"].dropna(
       subset=["POAG_cases","PGS616","PGS526","PMBB_3.0_Release_AGE","SEX"]).copy()
pmbb["POAG_cases"] = pmbb["POAG_cases"].astype(int)
pmbb["SEX_bin"]    = (pmbb["SEX"]=="Male").astype(int)
y_pmbb = pmbb["POAG_cases"].values
print(f"  Train N={len(tr)}  PMBB AFR N={len(pmbb):,}  cases={y_pmbb.sum()}")

# ── Re-run 5x20 CV ───────────────────────────────────────────
print("Re-running 5x20 CV ...")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
cv_rows = []
for fs_name, cols in FEATURE_SETS.items():
    missing = [c for c in cols if c not in tr.columns]
    if missing:
        continue
    X = tr[cols].values
    for mn in MODEL_NAMES:
        fm = {m: [] for m in ["AUC","Accuracy","F1","Sensitivity","Specificity"]}
        for tr_i, te_i in rskf.split(X, y_tr):
            if len(np.unique(y_tr[te_i])) < 2:
                continue
            pipe = make_pipeline(mn)
            pipe.fit(X[tr_i], y_tr[tr_i])
            yp  = pipe.predict_proba(X[te_i])[:,1]
            ypb = pipe.predict(X[te_i])
            yt  = y_tr[te_i]
            fm["AUC"].append(roc_auc_score(yt, yp))
            fm["Accuracy"].append(accuracy_score(yt, ypb))
            fm["F1"].append(f1_score(yt, ypb, zero_division=0))
            fm["Sensitivity"].append(recall_score(yt, ypb, zero_division=0))
            fm["Specificity"].append(specificity(yt, ypb))
        for metric, vals in fm.items():
            row = {"FeatureSet":fs_name,"Model":mn,"Metric":metric,"N_features":len(cols)}
            row.update(summarise(vals))
            cv_rows.append(row)
        print(f"  {fs_name} × {mn}  AUC={np.mean(fm['AUC']):.3f}")

cv_df = pd.DataFrame(cv_rows)

# Average across models
avg_rows = []
for fs_name in FEATURE_SETS:
    for metric in cv_df["Metric"].unique():
        sub = cv_df[(cv_df["FeatureSet"]==fs_name)&(cv_df["Metric"]==metric)]
        if len(sub)==0: continue
        mn  = sub["Mean"].mean()
        sd  = float(np.sqrt(np.mean(sub["SD"].values**2)))
        lo, hi = mn-1.96*sd/np.sqrt(100), mn+1.96*sd/np.sqrt(100)
        avg_rows.append({"FeatureSet":fs_name,"Model":"Average","Metric":metric,
                         "N_features":sub["N_features"].iloc[0],
                         "Mean":round(mn,4),"SD":round(sd,4),
                         "CI_lower":round(lo,4),"CI_upper":round(hi,4),
                         "Mean_SD":f"{mn:.3f} ± {sd:.3f}",
                         "Mean_95CI":f"{mn:.3f} ({lo:.3f}–{hi:.3f})"})
cv_df = pd.concat([cv_df, pd.DataFrame(avg_rows)], ignore_index=True)

# ── PC–PGS correlations ───────────────────────────────────────
print("\nPC–PGS correlation ...")
pgs_labels = list(TRAIN_PGS_COLS.keys())
pgs_cols   = list(TRAIN_PGS_COLS.values())
corr_rows  = []
for pc in PC5_COLS:
    for pgs_label, pgs_col in zip(pgs_labels, pgs_cols):
        tmp = tr[[pc, pgs_col]].dropna()
        r, p = stats.pearsonr(tmp[pc], tmp[pgs_col])
        corr_rows.append({"PC":pc,"PGS":pgs_label,"Pearson_r":round(r,3),"p_value":round(p,4)})
        print(f"  {pc} vs {pgs_label}: r={r:.3f}, p={p:.4f}")
corr_df = pd.DataFrame(corr_rows)

# ── PMBB external validation ──────────────────────────────────
print("\nPMBB external validation ...")
pmbb_rows = []
for fs_name, cfg in PMBB_FEAT_SETS.items():
    tr_cols = cfg["tr_cols"]
    pm_cols = cfg["pm_cols"]
    X_tr = tr[tr_cols].values
    X_pm = pmbb[pm_cols].values
    for mn in MODEL_NAMES:
        pipe = make_pipeline(mn)
        pipe.fit(X_tr, y_tr)
        yp   = pipe.predict_proba(X_pm)[:,1]
        ypb  = (yp >= 0.5).astype(int)
        am, alo, ahi = boot_auc(y_pmbb, yp)
        pmbb_rows.append({
            "FeatureSet":fs_name,"Model":mn,
            "N_cases":int(y_pmbb.sum()),"N_controls":int((y_pmbb==0).sum()),
            "AUC":am,"AUC_CI_lower":alo,"AUC_CI_upper":ahi,
            "AUC_95CI":f"{am:.3f} ({alo:.3f}–{ahi:.3f})",
            "Accuracy":round(accuracy_score(y_pmbb,ypb),4),
            "F1":round(f1_score(y_pmbb,ypb,zero_division=0),4),
            "Sensitivity":round(recall_score(y_pmbb,ypb,zero_division=0),4),
            "Specificity":round(specificity(y_pmbb,ypb),4),
        })
        print(f"  PMBB {fs_name} × {mn}  AUC={am:.3f} ({alo:.3f}–{ahi:.3f})")
    sub = [r for r in pmbb_rows if r["FeatureSet"]==fs_name and r["Model"]!="Average"]
    avg_auc = np.mean([r["AUC"] for r in sub])
    pmbb_rows.append({
        "FeatureSet":fs_name,"Model":"Average",
        "N_cases":int(y_pmbb.sum()),"N_controls":int((y_pmbb==0).sum()),
        "AUC":round(avg_auc,4),"AUC_CI_lower":np.nan,"AUC_CI_upper":np.nan,
        "AUC_95CI":f"{avg_auc:.3f}",
        "Accuracy":round(np.mean([r["Accuracy"] for r in sub]),4),
        "F1":round(np.mean([r["F1"] for r in sub]),4),
        "Sensitivity":round(np.mean([r["Sensitivity"] for r in sub]),4),
        "Specificity":round(np.mean([r["Specificity"] for r in sub]),4),
    })
pmbb_df = pd.DataFrame(pmbb_rows)

# ── Save Excel ────────────────────────────────────────────────
print("\nSaving Excel ...")
with pd.ExcelWriter(fr"{OUT_XL}\Table_Figure3_Training_5x20CV.xlsx",engine="openpyxl") as w:
    for metric in ["AUC","Accuracy","F1","Sensitivity","Specificity"]:
        sub = cv_df[cv_df["Metric"]==metric]
        piv = sub.pivot(index="FeatureSet",columns="Model",values="Mean_95CI")
        piv = piv.reindex(columns=MODEL_ORDER)
        piv.to_excel(w, sheet_name=metric)
    cv_df.to_excel(w, sheet_name="Full_Results", index=False)

with pd.ExcelWriter(fr"{OUT_XL}\Table_Figure3_PMBB_External.xlsx",engine="openpyxl") as w:
    pmbb_df.to_excel(w, sheet_name="PMBB_AFR", index=False)

with pd.ExcelWriter(fr"{OUT_XL}\Table_PC_PGS_Correlations.xlsx",engine="openpyxl") as w:
    corr_df.to_excel(w, sheet_name="PC_PGS_Corr", index=False)
    corr_df.pivot(index="PC",columns="PGS",values="Pearson_r").to_excel(w, sheet_name="Pivot_r")
print("  Excel saved.")

# ── Figures ───────────────────────────────────────────────────
auc_cv = cv_df[cv_df["Metric"]=="AUC"]

def auc_bar_panel(ax, data_df, fs_list, title,
                  ylabel="AUC (mean ± 95% CI)",
                  ylim=(0.40, 0.92), show_legend=True, highlight_fs=None):
    n_fs  = len(fs_list)
    n_mod = len(MODEL_ORDER)
    bar_w = 0.13
    x     = np.arange(n_fs)
    for i, mod in enumerate(MODEL_ORDER):
        means, errs = [], []
        for fs in fs_list:
            row = data_df[(data_df["FeatureSet"]==fs)&(data_df["Model"]==mod)]
            if len(row)==0:
                means.append(np.nan); errs.append(np.nan)
            else:
                mn  = float(row["Mean"].iloc[0])
                lo  = float(row["CI_lower"].iloc[0]) if "CI_lower" in row.columns else np.nan
                means.append(mn)
                errs.append(mn - lo if not np.isnan(lo) else 0)
        offset = (i - (n_mod-1)/2) * bar_w
        ax.bar(x+offset, means, bar_w, color=MODEL_COLORS[mod], alpha=0.85,
               label=mod, zorder=3)
        ax.errorbar(x+offset, means, yerr=errs, fmt="none", color="black",
                    capsize=2.5, linewidth=0.8, zorder=4)
        for xpos, mv, err in zip(x+offset, means, errs):
            if np.isnan(mv): continue
            top = mv + (err if not np.isnan(err) else 0) + 0.010
            ax.text(xpos, top, f"{mv:.3f}", ha="center", va="bottom",
                    fontsize=5.5, rotation=0, color="black")
    ax.axhline(0.5, color="dimgrey", linewidth=0.9, linestyle=":", label="Chance (0.5)", zorder=2)
    if highlight_fs:
        for j, fs in enumerate(fs_list):
            if fs in highlight_fs:
                ax.axvspan(j-0.42, j+0.42, color="lightyellow", alpha=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(fs_list, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)

# Fig 3A — training: base vs PC vs PGS
FS_MAIN = ["Base","Base+PC2","Base+PC5","Base+POAAGG PGS","Base+MEGA PGS","Base+PGS526","Base+PGS616"]
fig3a, ax3a = plt.subplots(figsize=(13,5))
fig3a.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.25)
auc_bar_panel(ax3a, auc_cv, FS_MAIN,
              title="Training Cohort (N=271) — 5×20 CV AUC\nBase vs Base+PC vs Base+PGS",
              highlight_fs=["Base+PC2","Base+PC5"])
fig3a.savefig(fr"{OUT_FIG}\Figure_3A_Training_Base_PC_PGS.png", bbox_inches="tight", dpi=300)
fig3a.savefig(fr"{OUT_FIG}\Figure_3A_Training_Base_PC_PGS.pdf", bbox_inches="tight", dpi=300)
plt.close(fig3a)
print("  Fig 3A saved.")

# Fig 3B — PC+PGS combined
FS_COMB = ["Base","Base+PGS526","Base+PGS616","Base+PC5+PGS526","Base+PC5+PGS616"]
fig3b, ax3b = plt.subplots(figsize=(10,5))
fig3b.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.25)
auc_bar_panel(ax3b, auc_cv, FS_COMB,
              title="Training Cohort — PC+PGS Combined Models\nDoes adding PC change PGS model performance?",
              highlight_fs=["Base+PC5+PGS526","Base+PC5+PGS616"])
fig3b.savefig(fr"{OUT_FIG}\Figure_3B_Training_PCplusPGS.png", bbox_inches="tight", dpi=300)
fig3b.savefig(fr"{OUT_FIG}\Figure_3B_Training_PCplusPGS.pdf", bbox_inches="tight", dpi=300)
plt.close(fig3b)
print("  Fig 3B saved.")

# Fig 3C — PMBB external validation
pmbb_auc = pmbb_df.rename(columns={"AUC":"Mean","AUC_CI_lower":"CI_lower","AUC_CI_upper":"CI_upper"})
PMBB_FS  = ["Base","Base+PGS526","Base+PGS616","Base+PC5","Base+PC5+PGS526","Base+PC5+PGS616"]
fig3c, ax3c = plt.subplots(figsize=(11,5))
fig3c.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.25)
auc_bar_panel(ax3c, pmbb_auc, PMBB_FS,
              title=f"PMBB External Validation — African Ancestry (N={len(pmbb):,})\n"
                    f"Cases={int(y_pmbb.sum())}, Controls={int((y_pmbb==0).sum())}",
              ylim=(0.40, 0.92),
              highlight_fs=["Base+PC5","Base+PC5+PGS526","Base+PC5+PGS616"])
fig3c.savefig(fr"{OUT_FIG}\Figure_3C_PMBB_External.png", bbox_inches="tight", dpi=300)
fig3c.savefig(fr"{OUT_FIG}\Figure_3C_PMBB_External.pdf", bbox_inches="tight", dpi=300)
plt.close(fig3c)
print("  Fig 3C saved.")

# Supp: PC–PGS correlation heatmap
corr_mat = corr_df.pivot(index="PC",columns="PGS",values="Pearson_r")
pval_mat = corr_df.pivot(index="PC",columns="PGS",values="p_value")
fig_corr, ax_corr = plt.subplots(figsize=(6,4))
im = ax_corr.imshow(corr_mat.values, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
plt.colorbar(im, ax=ax_corr, label="Pearson r")
ax_corr.set_xticks(range(len(corr_mat.columns)))
ax_corr.set_xticklabels(corr_mat.columns, rotation=30, ha="right", fontsize=9)
ax_corr.set_yticks(range(len(corr_mat.index)))
ax_corr.set_yticklabels(corr_mat.index, fontsize=9)
ax_corr.set_title("PC–PGS Pearson Correlations (Training Cohort, N=271)", fontsize=10, fontweight="bold")
for i, pc in enumerate(corr_mat.index):
    for j, pgs in enumerate(corr_mat.columns):
        r_val = corr_mat.loc[pc, pgs]
        p_val = pval_mat.loc[pc, pgs]
        star  = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else ""))
        ax_corr.text(j, i, f"{r_val:.2f}{star}", ha="center", va="center",
                     fontsize=8, color="white" if abs(r_val)>0.25 else "black")
fig_corr.tight_layout()
fig_corr.savefig(fr"{OUT_FIG}\SuppFig_PC_PGS_Correlation.png", bbox_inches="tight", dpi=300)
fig_corr.savefig(fr"{OUT_FIG}\SuppFig_PC_PGS_Correlation.pdf", bbox_inches="tight", dpi=300)
plt.close(fig_corr)
print("  Supp Fig PC-PGS correlation saved.")

# Supp: Full feature comparison
FS_FULL = ["Age only","Sex only","Base","Base+PC2","Base+PC5","Base+PC10",
           "Base+PGS616","Base+PC5+PGS616"]
fig_full, ax_full = plt.subplots(figsize=(14,5))
fig_full.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.25)
auc_bar_panel(ax_full, auc_cv, FS_FULL,
              title="Full Feature Set Comparison — 5×20 CV (Training Cohort, N=271)",
              ylim=(0.35, 0.92),
              highlight_fs=["Base+PC2","Base+PC5","Base+PC10"])
fig_full.savefig(fr"{OUT_FIG}\SuppFig_Full_Feature_Comparison.png", bbox_inches="tight", dpi=300)
fig_full.savefig(fr"{OUT_FIG}\SuppFig_Full_Feature_Comparison.pdf", bbox_inches="tight", dpi=300)
plt.close(fig_full)
print("  Supp Fig full comparison saved.")

# ── Key numbers ───────────────────────────────────────────────
print("\n=== KEY NUMBERS ===")
print("\nTraining AUC (average across classifiers):")
for fs in FEATURE_SETS:
    row = auc_cv[(auc_cv["FeatureSet"]==fs)&(auc_cv["Model"]=="Average")]
    if len(row)>0:
        print(f"  {fs:30s}: {row['Mean_95CI'].values[0]}")

print("\nPMBB AUC (average across classifiers):")
for fs in PMBB_FEAT_SETS:
    row = pmbb_df[(pmbb_df["FeatureSet"]==fs)&(pmbb_df["Model"]=="Average")]
    if len(row)>0:
        print(f"  {fs:30s}: {row['AUC_95CI'].values[0]}")

print("\nPC–PGS max |r| per PGS:")
for pgs_label in pgs_labels:
    sub = corr_df[corr_df["PGS"]==pgs_label]
    mx = sub.loc[sub["Pearson_r"].abs().idxmax()]
    print(f"  {pgs_label}: max |r|={mx['Pearson_r']:.3f} ({mx['PC']}, p={mx['p_value']:.4f})")

print("\nAll done.")
