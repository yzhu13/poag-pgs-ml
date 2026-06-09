# =============================================================
#  POAG-PGS  —  Figure 2  (Revised for iScience revision)
#  Validation : 5-fold × 20-repeat stratified CV  (100 fits)
#  Feature    : PGS only  (standalone, no age/sex)
#  Panels     : 2A chromosome bar chart  (loaded from PNG)
#               2B violin plots (cases / controls / suspects)
#               2C training AUC  (5×20 CV, error bars + labels)
#               2D PMBB external AUC  (bootstrap CI + labels)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
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
import warnings
warnings.filterwarnings("ignore")

import os as _os

# ═══════════════════════════════════════════════════════════════
#  PATH CONFIGURATION  —  edit BASE if your data is elsewhere
# ═══════════════════════════════════════════════════════════════
#  Expected folder layout (relative to BASE):
#    data/poaagg/   271_training_cohort_4_new_PRS_cleaned.xlsx
#                   1013_testing_cohort_only_suspect_cleaned.xlsx
#    data/pmbb/     PMBB_3.0_pheno_covars_noPOAAGG.csv
#                   PMBBv3_GRS_MEGA_616snps_AllSamples.sscore_withSTDscore.txt
#                   PMBBv3_GRS_QUANT_526snps_AllSamples.sscore_withSTDscore.txt
#                   PMBB_949_POAG_IOP_CDR_Freeze3.csv   (asymmetry script only)
#    outputs/tables/   ← Excel files written here
#    outputs/figures/  ← PNG / PDF figures written here
# ────────────────────────────────────────────────────────────────
BASE = _os.path.dirname(_os.path.abspath(__file__))
# To override:  BASE = r"C:\your\path"   (Windows)
#               BASE = "/your/path"        (macOS / Linux)

POAAGG_DIR = _os.path.join(BASE, "data", "poaagg")
PMBB_DIR   = _os.path.join(BASE, "data", "pmbb")
OUT_XL     = _os.path.join(BASE, "outputs", "tables")
OUT_FIG    = _os.path.join(BASE, "outputs", "figures")
_os.makedirs(OUT_XL,  exist_ok=True)
_os.makedirs(OUT_FIG, exist_ok=True)

TRAIN_F  = _os.path.join(POAAGG_DIR, "271_training_cohort_4_new_PRS_cleaned.xlsx")
SUSP_F   = _os.path.join(POAAGG_DIR, "1013_testing_cohort_only_suspect_cleaned.xlsx")
PMBB_PHE = _os.path.join(PMBB_DIR,   "PMBB_3.0_pheno_covars_noPOAAGG.csv")
PMBB_616 = _os.path.join(PMBB_DIR,   "PMBBv3_GRS_MEGA_616snps_AllSamples.sscore_withSTDscore.txt")
PMBB_526 = _os.path.join(PMBB_DIR,   "PMBBv3_GRS_QUANT_526snps_AllSamples.sscore_withSTDscore.txt")
PMBB_IOP_CDR = _os.path.join(PMBB_DIR, "PMBB_949_POAG_IOP_CDR_Freeze3.csv")
# ═══════════════════════════════════════════════════════════════

TRAIN_PGS = {
    "POAAGG PGS": "POAAGG PGS",
    "MEGA PGS":   "MEGA PGS",
    "PGS526":     "PGS526",
    "PGS616":     "PGS616",
}
SUSP_PGS = {
    "POAAGG PGS": "POAAGG PGS",
    "MEGA PGS":   "MEGA PGS",
    "PGS526":     "PGS526",
    "PGS616":     "PGS616",
}
PMBB_PGS = {
    "PGS616": ("PGS616", "PGS616"),
    "PGS526": ("PGS526", "PGS526"),
}
LABEL       = "CaseCtrl"
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
PGS_ORDER   = ["POAAGG PGS", "MEGA PGS", "PGS526", "PGS616"]
MODEL_ORDER = ["LR", "SVM", "RF", "MLP", "Average"]

MODEL_COLORS = {
    "LR": "#4E79A7", "SVM": "#F28E2B",
    "RF": "#59A14F", "MLP": "#E15759", "Average": "#B07AA1",
}
GROUP_COLORS = {"GLA": "#E15759", "CON": "#4E79A7", "SUS": "#59A14F"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})


# =============================================================
# 1.  HELPERS
# =============================================================
def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42))]
    elif name == "SVM":
        steps += [("clf", SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=42))]
    elif name == "RF":
        steps += [("clf", RandomForestClassifier(
            n_estimators=200, max_depth=5,
            class_weight="balanced", random_state=42))]
    elif name == "MLP":
        steps += [("clf", MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=1000,
            early_stopping=False, random_state=42))]
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
                Mean_SD=f"{mn:.3f} +/- {sd:.3f}",
                Mean_95CI=f"{mn:.3f} ({lo:.3f}-{hi:.3f})")


def boot_auc(yt, ys, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    bs  = [roc_auc_score(yt[i:=rng.choice(len(yt),len(yt),replace=True)], ys[i])
           for _ in range(n) if len(np.unique(yt[
               i:=rng.choice(len(yt),len(yt),replace=True)])) > 1]
    bs = np.array(bs)
    return round(float(bs.mean()),4), round(float(np.percentile(bs,2.5)),4), \
           round(float(np.percentile(bs,97.5)),4)


# =============================================================
# 2.  LOAD DATA
# =============================================================
print("Loading data ...")
tr   = pd.read_excel(TRAIN_F)
su   = pd.read_excel(SUSP_F)
y_tr = tr[LABEL].values.astype(int)
n_tr = len(y_tr)
print(f"  Train N={n_tr}  cases={y_tr.sum()}  ctrl={(y_tr==0).sum()}")

pmbb_phe = pd.read_csv(PMBB_PHE)
p616 = (pd.read_csv(PMBB_616,sep="\t")[["IID","SCORE1_AVG_STD"]]
          .rename(columns={"IID":"PMBB_ID","SCORE1_AVG_STD":"PGS616"}))
p526 = (pd.read_csv(PMBB_526,sep="\t")[["IID","SCORE1_AVG_STD"]]
          .rename(columns={"IID":"PMBB_ID","SCORE1_AVG_STD":"PGS526"}))
pmbb = (pmbb_phe.merge(p616,on="PMBB_ID").merge(p526,on="PMBB_ID"))
pmbb = pmbb[pmbb["ANCESTRY"]=="AFR"].dropna(
       subset=["POAG_cases","PGS616","PGS526"]).copy()
pmbb["POAG_cases"] = pmbb["POAG_cases"].astype(int)
y_pmbb = pmbb["POAG_cases"].values
print(f"  PMBB AFR N={len(pmbb):,}  cases={y_pmbb.sum()}  "
      f"ctrl={(y_pmbb==0).sum()}")


# =============================================================
# 3.  5-FOLD × 20-REPEAT CV  (PGS only)
# =============================================================
print("\n5-fold x 20-repeat stratified CV ...")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

cv_rows = []
for pgs_label, pgs_col in TRAIN_PGS.items():
    X = tr[[pgs_col]].values
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
            row = {"PGS":pgs_label,"Model":mn,"Metric":metric}
            row.update(summarise(vals))
            cv_rows.append(row)
        print(f"  {pgs_label} x {mn}  "
              f"AUC={np.mean(fm['AUC']):.3f}+/-{np.std(fm['AUC']):.3f}")

cv_df = pd.DataFrame(cv_rows)

# Average across models
avg_rows = []
for pgs_label in PGS_ORDER:
    for metric in cv_df["Metric"].unique():
        sub = cv_df[(cv_df["PGS"]==pgs_label)&(cv_df["Metric"]==metric)]
        mn  = sub["Mean"].mean()
        sd  = float(np.sqrt(np.mean(sub["SD"].values**2)))
        lo, hi = mn-1.96*sd/np.sqrt(100), mn+1.96*sd/np.sqrt(100)
        avg_rows.append({"PGS":pgs_label,"Model":"Average","Metric":metric,
            "Mean":round(mn,4),"SD":round(sd,4),
            "CI_lower":round(lo,4),"CI_upper":round(hi,4),
            "Mean_SD":f"{mn:.3f} +/- {sd:.3f}",
            "Mean_95CI":f"{mn:.3f} ({lo:.3f}-{hi:.3f})"})
cv_df = pd.concat([cv_df, pd.DataFrame(avg_rows)], ignore_index=True)


# =============================================================
# 4.  PMBB EXTERNAL VALIDATION
# =============================================================
print("\nPMBB external validation ...")
pmbb_rows = []
for pgs_label, (tr_col, pm_col) in PMBB_PGS.items():
    X_tr = tr[[tr_col]].values
    X_pm = pmbb[[pm_col]].values
    for mn in MODEL_NAMES:
        pipe = make_pipeline(mn)
        pipe.fit(X_tr, y_tr)
        yp   = pipe.predict_proba(X_pm)[:,1]
        ypb  = (yp >= 0.5).astype(int)
        am, alo, ahi = boot_auc(y_pmbb, yp)
        pmbb_rows.append({
            "PGS":pgs_label,"Model":mn,
            "N_cases":int(y_pmbb.sum()),"N_controls":int((y_pmbb==0).sum()),
            "AUC":am,"AUC_CI_lower":alo,"AUC_CI_upper":ahi,
            "AUC_95CI":f"{am:.3f} ({alo:.3f}-{ahi:.3f})",
            "Accuracy":round(accuracy_score(y_pmbb,ypb),4),
            "F1":round(f1_score(y_pmbb,ypb,zero_division=0),4),
            "Sensitivity":round(recall_score(y_pmbb,ypb,zero_division=0),4),
            "Specificity":round(specificity(y_pmbb,ypb),4),
        })
        print(f"  PMBB {pgs_label} x {mn}  AUC={am:.3f} ({alo:.3f}-{ahi:.3f})")
    sub = [r for r in pmbb_rows if r["PGS"]==pgs_label and r["Model"]!="Average"]
    avg_auc = np.mean([r["AUC"] for r in sub])
    pmbb_rows.append({
        "PGS":pgs_label,"Model":"Average",
        "N_cases":int(y_pmbb.sum()),"N_controls":int((y_pmbb==0).sum()),
        "AUC":round(avg_auc,4),"AUC_CI_lower":np.nan,"AUC_CI_upper":np.nan,
        "AUC_95CI":f"{avg_auc:.3f}",
        "Accuracy":round(np.mean([r["Accuracy"]    for r in sub]),4),
        "F1":round(np.mean([r["F1"]          for r in sub]),4),
        "Sensitivity":round(np.mean([r["Sensitivity"] for r in sub]),4),
        "Specificity":round(np.mean([r["Specificity"] for r in sub]),4),
    })
pmbb_df = pd.DataFrame(pmbb_rows)


# =============================================================
# 5.  SAVE EXCEL
# =============================================================
print("\nSaving Excel ...")
with pd.ExcelWriter(
        _os.path.join(OUT_XL, "Table_Training_5x20CV_PGSonly.xlsx"),
        engine="openpyxl") as w:
    for metric in ["AUC","Accuracy","F1","Sensitivity","Specificity"]:
        sub = cv_df[cv_df["Metric"]==metric]
        piv = (sub.pivot(index="PGS",columns="Model",values="Mean_95CI")
                  .reindex(index=PGS_ORDER, columns=MODEL_ORDER))
        piv.to_excel(w, sheet_name=metric)
    cv_df.to_excel(w, sheet_name="Full_Results", index=False)

with pd.ExcelWriter(
        _os.path.join(OUT_XL, "Table_PMBB_External_5x20CV_PGSonly.xlsx"),
        engine="openpyxl") as w:
    pmbb_df.to_excel(w, sheet_name="PMBB_AFR", index=False)
print("  Excel saved.")


# =============================================================
# 6.  FIGURE 2  (4 panels)
# =============================================================
print("\nBuilding Figure 2 ...")

auc_cv  = cv_df[cv_df["Metric"]=="AUC"]

# ── layout ───────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
gs_outer = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.42, wspace=0.32,
    left=0.06, right=0.97, top=0.95, bottom=0.07
)

# Panel B occupies top-right with 4 internal violin axes
gs_b = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs_outer[0, 1], wspace=0.45)

ax_a  = fig.add_subplot(gs_outer[0, 0])
axes_b = [fig.add_subplot(gs_b[0, j]) for j in range(4)]
ax_c  = fig.add_subplot(gs_outer[1, 0])
ax_d  = fig.add_subplot(gs_outer[1, 1])

# panel label helper
def panel_label(ax, lbl, dx=-0.10, dy=1.04):
    ax.text(dx, dy, lbl, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top")

# ── 2A : chromosome bar chart (load existing PNG) ─────────────
img = mpimg.imread(FIG_2A)
ax_a.imshow(img)
ax_a.axis("off")
panel_label(ax_a, "A", dx=-0.03)

# ── 2B : violin plots ─────────────────────────────────────────
n_gla = int((y_tr == 1).sum())
n_con = int((y_tr == 0).sum())
n_sus = len(su)
g_order  = ["GLA", "CON", "SUS"]
g_labels = {
    "GLA": f"Cases\n(n={n_gla})",
    "CON": f"Controls\n(n={n_con})",
    "SUS": f"Suspects\n(n={n_sus})",
}

viol_rows = []
for pgs_label, col in TRAIN_PGS.items():
    for _, row in tr.iterrows():
        viol_rows.append({"PGS": pgs_label,
                          "Group": "GLA" if row[LABEL]==1 else "CON",
                          "Score": row[col]})
for pgs_label, col in SUSP_PGS.items():
    for _, row in su.iterrows():
        viol_rows.append({"PGS": pgs_label, "Group": "SUS",
                          "Score": row[col]})
vdf = pd.DataFrame(viol_rows)

# Z-score normalize each PGS across all subjects (training + suspects)
# for visual comparability. PGS616 is stored on a raw PLINK log-odds scale
# (-12 to -0.6) while POAAGG/MEGA/PGS526 are already INT-normalized.
# Normalization here is for DISPLAY ONLY; ML models use StandardScaler internally.
for _pgs in PGS_ORDER:
    _mask = vdf["PGS"] == _pgs
    _vals = vdf.loc[_mask, "Score"]
    vdf.loc[_mask, "Score"] = (_vals - _vals.mean()) / _vals.std()

for j, (ax_b, pgs_label) in enumerate(zip(axes_b, PGS_ORDER)):
    sub  = vdf[vdf["PGS"] == pgs_label]
    data = [sub[sub["Group"]==g]["Score"].dropna().values for g in g_order]
    pts  = ax_b.violinplot(data, positions=[1,2,3],
                           showmedians=True, showextrema=False, widths=0.7)
    for body, g in zip(pts["bodies"], g_order):
        body.set_facecolor(GROUP_COLORS[g])
        body.set_alpha(0.75)
        body.set_edgecolor("grey")
        body.set_linewidth(0.5)
    pts["cmedians"].set_color("black")
    pts["cmedians"].set_linewidth(1.5)
    ax_b.set_xticks([1,2,3])
    ax_b.set_xticklabels([g_labels[g] for g in g_order], fontsize=8)
    ax_b.set_title(pgs_label, fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Z-score normalized PGS" if j==0 else "", fontsize=8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

# legend for violins on the last violin ax
patches_b = [mpatches.Patch(color=GROUP_COLORS[g], label=l, alpha=0.75)
             for g, l in [("GLA","Cases"),("CON","Controls"),("SUS","Suspects")]]
axes_b[-1].legend(handles=patches_b, fontsize=7.5, frameon=False,
                  loc="lower left")
panel_label(axes_b[0], "B", dx=-0.18)

# ── helper: bar chart with error bars + AUC labels on top ─────
def auc_bar_panel(ax, data_df, pgs_list, title, xlabel,
                  pmbb_ref=None, ylim=(0.35, 0.82)):
    n_pgs  = len(pgs_list)
    n_mod  = len(MODEL_ORDER)
    bar_w  = 0.13
    x      = np.arange(n_pgs)

    for i, mod in enumerate(MODEL_ORDER):
        means, errs = [], []
        for pgs in pgs_list:
            row = data_df[(data_df["PGS"]==pgs) & (data_df["Model"]==mod)]
            if len(row) == 0:
                means.append(np.nan); errs.append(np.nan)
            else:
                mn = float(row["Mean"].iloc[0])
                lo = float(row["CI_lower"].iloc[0]) \
                     if "CI_lower" in row.columns else np.nan
                means.append(mn)
                errs.append(mn - lo if not np.isnan(lo) else 0)

        offset = (i - (n_mod-1)/2) * bar_w
        bars = ax.bar(x + offset, means, bar_w,
                      color=MODEL_COLORS[mod], alpha=0.85,
                      label=mod, zorder=3)
        ax.errorbar(x + offset, means, yerr=errs,
                    fmt="none", color="black",
                    capsize=2.5, linewidth=0.8, zorder=4)

        # AUC label above each bar
        for xpos, mn, err in zip(x + offset, means, errs):
            if np.isnan(mn):
                continue
            top = mn + (err if not np.isnan(err) else 0) + 0.012
            ax.text(xpos, top, f"{mn:.3f}",
                    ha="center", va="bottom",
                    fontsize=6.2, rotation=0, color="black")

    # chance line
    ax.axhline(0.5, color="dimgrey", linewidth=0.9,
               linestyle=":", label="Chance (0.5)", zorder=2)

    # optional PMBB reference band
    if pmbb_ref is not None:
        lo_r = min(pmbb_ref); hi_r = max(pmbb_ref)
        ax.axhspan(lo_r, hi_r, color="gold", alpha=0.20, zorder=1)
        ax.axhline(np.mean(pmbb_ref), color="goldenrod",
                   linewidth=1.2, linestyle="--", zorder=2,
                   label=f"PMBB mean AUC ({np.mean(pmbb_ref):.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(pgs_list, fontsize=9)
    ax.set_ylabel("AUC (mean ± 95% CI)", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)


# ── 2C : training AUC (5×20 CV) ──────────────────────────────
# reference: average PMBB AUC per PGS (for PGS616 and PGS526 only)
pmbb_ref_vals = [
    float(pmbb_df[(pmbb_df["PGS"]=="PGS616")&
                  (pmbb_df["Model"]=="Average")]["AUC"].values[0]),
    float(pmbb_df[(pmbb_df["PGS"]=="PGS526")&
                  (pmbb_df["Model"]=="Average")]["AUC"].values[0]),
]
pmbb_ref_c = [float(pmbb_df[(pmbb_df["PGS"]==p)&
                             (pmbb_df["Model"]=="Average")]["AUC"].values[0])
              for p in PGS_ORDER if p in ["PGS616","PGS526"]]

auc_bar_panel(
    ax_c, auc_cv, PGS_ORDER,
    title="",
    xlabel="PGS (standalone, no age/sex)",
    ylim=(0.35, 0.70),
)
panel_label(ax_c, "C", dx=-0.10, dy=1.08)

# ── 2D : PMBB external validation AUC ────────────────────────
pmbb_auc_df = pmbb_df[["PGS","Model","AUC","AUC_CI_lower","AUC_CI_upper"]].copy()
pmbb_auc_df = pmbb_auc_df.rename(
    columns={"AUC":"Mean","AUC_CI_lower":"CI_lower","AUC_CI_upper":"CI_upper"})

auc_bar_panel(
    ax_d, pmbb_auc_df, ["PGS616","PGS526"],
    title="",
    xlabel="PGS (standalone, no age/sex)",
    ylim=(0.35, 0.70),
)
panel_label(ax_d, "D", dx=-0.10, dy=1.08)

# ── save combined ────────────────────────────────────────────
fig.savefig(_os.path.join(OUT_FIG, "Figure_2_Combined.pdf"),
            bbox_inches="tight", dpi=300)
fig.savefig(_os.path.join(OUT_FIG, "Figure_2_Combined.png"),
            bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Figure 2 combined saved.")

# ── save Fig 2B separately (4 violin subplots) ───────────────
fig_b, axes_b2 = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
fig_b.subplots_adjust(wspace=0.45, left=0.07, right=0.97,
                      top=0.88, bottom=0.15)
for j, (ax_b2, pgs_label) in enumerate(zip(axes_b2, PGS_ORDER)):
    sub  = vdf[vdf["PGS"] == pgs_label]
    data = [sub[sub["Group"]==g]["Score"].dropna().values for g in g_order]
    pts  = ax_b2.violinplot(data, positions=[1,2,3],
                            showmedians=True, showextrema=False, widths=0.7)
    for body, g in zip(pts["bodies"], g_order):
        body.set_facecolor(GROUP_COLORS[g])
        body.set_alpha(0.75)
        body.set_edgecolor("grey")
        body.set_linewidth(0.5)
    pts["cmedians"].set_color("black")
    pts["cmedians"].set_linewidth(1.5)
    ax_b2.set_xticks([1,2,3])
    ax_b2.set_xticklabels([g_labels[g] for g in g_order], fontsize=8)
    ax_b2.set_title(pgs_label, fontsize=10, fontweight="bold")
    ax_b2.set_ylabel("Z-score normalized PGS" if j==0 else "", fontsize=8)
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)
patches_b2 = [mpatches.Patch(color=GROUP_COLORS[g], label=l, alpha=0.75)
              for g, l in [("GLA","Cases"),("CON","Controls"),("SUS","Suspects")]]
axes_b2[-1].legend(handles=patches_b2, fontsize=7.5, frameon=False, loc="lower left")
fig_b.suptitle("Figure 2B — PGS Distributions Across Groups", fontsize=11, fontweight="bold")
fig_b.savefig(_os.path.join(OUT_FIG, "Figure_2B_Violin.pdf"), bbox_inches="tight", dpi=300)
fig_b.savefig(_os.path.join(OUT_FIG, "Figure_2B_Violin.png"), bbox_inches="tight", dpi=300)
plt.close(fig_b)
print("  Figure 2B saved.")

# ── save Fig 2C separately (training AUC bar chart) ──────────
fig_c, ax_c2 = plt.subplots(figsize=(8, 5))
fig_c.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)
auc_bar_panel(
    ax_c2, auc_cv, PGS_ORDER,
    title="",
    xlabel="PGS (standalone, no age/sex)",
    ylim=(0.35, 0.70),
)
fig_c.savefig(_os.path.join(OUT_FIG, "Figure_2C_Training_AUC.pdf"), bbox_inches="tight", dpi=300)
fig_c.savefig(_os.path.join(OUT_FIG, "Figure_2C_Training_AUC.png"), bbox_inches="tight", dpi=300)
plt.close(fig_c)
print("  Figure 2C saved.")

# ── save Fig 2D separately (PMBB external AUC bar chart) ─────
fig_d, ax_d2 = plt.subplots(figsize=(6, 5))
fig_d.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.12)
auc_bar_panel(
    ax_d2, pmbb_auc_df, ["PGS616","PGS526"],
    title="",
    xlabel="PGS (standalone, no age/sex)",
    ylim=(0.35, 0.70),
)
fig_d.savefig(_os.path.join(OUT_FIG, "Figure_2D_PMBB_AUC.pdf"), bbox_inches="tight", dpi=300)
fig_d.savefig(_os.path.join(OUT_FIG, "Figure_2D_PMBB_AUC.png"), bbox_inches="tight", dpi=300)
plt.close(fig_d)
print("  Figure 2D saved.")

# =============================================================
# 7.  PRINT KEY NUMBERS for manuscript
# =============================================================
print("\n=== KEY NUMBERS FOR MANUSCRIPT ===")
print("\nTraining AUC (5x20 CV, Average across LR/SVM/RF/MLP):")
for pgs in PGS_ORDER:
    row = auc_cv[(auc_cv["PGS"]==pgs)&(auc_cv["Model"]=="Average")]
    print(f"  {pgs}: {row['Mean_95CI'].values[0]}")

print("\nTraining AUC per model (PGS616):")
for mod in MODEL_ORDER:
    row = auc_cv[(auc_cv["PGS"]=="PGS616")&(auc_cv["Model"]==mod)]
    print(f"  {mod}: {row['Mean_95CI'].values[0]}")

print("\nPMBB AUC (bootstrap 95% CI):")
for pgs in ["PGS616","PGS526"]:
    for mod in MODEL_ORDER:
        row = pmbb_df[(pmbb_df["PGS"]==pgs)&(pmbb_df["Model"]==mod)]
        print(f"  {pgs} x {mod}: {row['AUC_95CI'].values[0]}")

print("\nAll done.")
