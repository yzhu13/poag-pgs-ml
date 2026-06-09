# =============================================================
#  Testing Cohort Analysis — Clinical Enrichment (1013 Suspects)
#  Train 5 feature sets × 4 classifiers on 271 cohort
#  Apply to 1013 suspects → predicted risk scores
#  Correlate with IOP_SEVERE, CDR_SEVERE, RNFL_SEVERE
#  Main figure: MLP × Base+PGS616 — quintile enrichment (3 panels)
#  Supp figure: Pearson r heatmap (all classifiers × all feature sets)
# =============================================================
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

LABEL = "CaseCtrl"

# Column mapping: feature name → training col, suspect col
FEATURE_SETS = {
    "Base":           {"tr": ["Age","Gender"],                       "su": ["Age","Gender"]},
    "Base+POAAGG PGS":{"tr": ["Age","Gender","POAAGG PGS"],          "su": ["Age","Gender","POAAGG PGS"]},
    "Base+MEGA PGS":  {"tr": ["Age","Gender","MEGA PGS"],            "su": ["Age","Gender","MEGA PGS"]},
    "Base+PGS526":    {"tr": ["Age","Gender","PGS526"],               "su": ["Age","Gender","PGS526"]},
    "Base+PGS616":    {"tr": ["Age","Gender","PGS616"],               "su": ["Age","Gender","PGS616"]},
}
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
MODEL_COLORS = {"LR":"#4E79A7","SVM":"#F28E2B","RF":"#59A14F","MLP":"#E15759"}

# Clinical outcomes in suspect file
OUTCOMES = {
    "IOP (mmHg)":    "IOP_SEVERE",
    "CDR":           "CDR_SEVERE",
    "RNFL (μm)":     "RNFL_SEVERE",
}
OUTCOME_DIRECTION = {"IOP (mmHg)": "+", "CDR": "+", "RNFL (μm)": "-"}
OUTCOME_COLORS    = {"IOP (mmHg)":"#E05C5C","CDR":"#5B8DB8","RNFL (μm)":"#7DC77A"}

plt.rcParams.update({
    "font.family":"Arial","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":0.8,
})


# =============================================================
# HELPERS
# =============================================================
def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42))]
    elif name == "SVM":
        steps += [("clf", SVC(kernel="rbf", probability=True,
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


# =============================================================
# 1.  LOAD DATA
# =============================================================
print("Loading data ...")
tr   = pd.read_excel(TRAIN_F)
su   = pd.read_excel(SUSP_F)
y_tr = tr[LABEL].values.astype(int)
print(f"  Train N={len(tr)}  cases={y_tr.sum()}  ctrl={(y_tr==0).sum()}")
print(f"  Suspects N={len(su)}")
for out_label, col in OUTCOMES.items():
    n = su[col].notna().sum()
    print(f"  {out_label} ({col}): {n} non-missing ({n/len(su)*100:.1f}%)")


# =============================================================
# 2.  TRAIN ON 271, APPLY TO 1013 SUSPECTS
# =============================================================
print("\nTraining & applying models ...")
# Store predicted scores: dict[(fs_name, model_name)] → array of len 1013
pred_scores = {}
train_log   = []

for fs_name, cfg in FEATURE_SETS.items():
    X_tr  = tr[cfg["tr"]].values
    X_su  = su[cfg["su"]].values
    for mn in MODEL_NAMES:
        pipe = make_pipeline(mn)
        pipe.fit(X_tr, y_tr)
        scores = pipe.predict_proba(X_su)[:, 1]
        pred_scores[(fs_name, mn)] = scores
        train_log.append({
            "FeatureSet": fs_name, "Model": mn,
            "Train_N": len(tr), "Suspect_N": len(su),
            "Score_mean": round(float(scores.mean()), 4),
            "Score_std":  round(float(scores.std()), 4),
            "Score_min":  round(float(scores.min()), 4),
            "Score_max":  round(float(scores.max()), 4),
        })
        print(f"  {fs_name} × {mn}  score: {scores.mean():.3f}±{scores.std():.3f}")


# =============================================================
# 3.  CORRELATIONS: predicted risk vs clinical outcomes
# =============================================================
print("\nCorrelation analysis ...")
corr_rows = []
for (fs_name, mn), scores in pred_scores.items():
    for out_label, col in OUTCOMES.items():
        valid = su[col].notna().values
        r_val, p_val = stats.pearsonr(scores[valid], su[col].values[valid])
        corr_rows.append({
            "FeatureSet": fs_name, "Model": mn,
            "Outcome": out_label,
            "N": int(valid.sum()),
            "Pearson_r": round(r_val, 4),
            "p_value": round(p_val, 6),
            "Significant": "***" if p_val < 0.001 else
                           ("**" if p_val < 0.01 else
                           ("*"  if p_val < 0.05 else "ns")),
        })
        print(f"  {fs_name} × {mn} vs {out_label}: "
              f"r={r_val:.3f}, p={p_val:.4f}, N={valid.sum()}")

corr_df = pd.DataFrame(corr_rows)


# =============================================================
# 4.  QUINTILE ENRICHMENT (for MLP × Base+PGS616 — main figure)
# =============================================================
print("\nQuintile enrichment (MLP × Base+PGS616) ...")
mlp_scores = pred_scores[("Base+PGS616", "MLP")]
quint_labels_cat = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
quintiles  = pd.Categorical(
    pd.qcut(mlp_scores, 5, labels=quint_labels_cat))

quint_rows = []
for out_label, col in OUTCOMES.items():
    for q in quint_labels_cat:
        mask = np.array(quintiles == q) & su[col].notna().values
        vals = su[col].values[mask]
        quint_rows.append({
            "Outcome": out_label, "Quintile": q,
            "N": len(vals),
            "Mean": round(float(vals.mean()), 3),
            "SEM":  round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 3),
        })
quint_df = pd.DataFrame(quint_rows)
print(quint_df.to_string(index=False))


# =============================================================
# 5.  SAVE EXCEL
# =============================================================
print("\nSaving Excel ...")
with pd.ExcelWriter(
        _os.path.join(OUT_XL, "Table_Suspects_AllModels_ClinicalCorrelations.xlsx"),
        engine="openpyxl") as w:

    # Pivot: rows=FeatureSet×Model, cols=Outcome, values=r (p)
    summary_rows = []
    for _, row in corr_df.iterrows():
        summary_rows.append({
            "FeatureSet": row["FeatureSet"],
            "Model":      row["Model"],
            "Outcome":    row["Outcome"],
            "r_p":        f"{row['Pearson_r']:.3f} ({row['p_value']:.4f}) {row['Significant']}",
            "N":          row["N"],
        })
    sum_df = pd.DataFrame(summary_rows)
    piv = sum_df.pivot_table(index=["FeatureSet","Model"],
                              columns="Outcome", values="r_p",
                              aggfunc="first")
    piv.to_excel(w, sheet_name="Correlation_Summary")

    corr_df.to_excel(w, sheet_name="All_Correlations", index=False)
    quint_df.to_excel(w, sheet_name="Quintile_MLP_PGS616", index=False)

    # Score stats
    pd.DataFrame(train_log).to_excel(w, sheet_name="Score_Statistics", index=False)

print("  Excel saved.")


# =============================================================
# 6.  FIGURE 4A — MLP × Base+PGS616 quintile line plots (1×3)
# =============================================================
print("\nBuilding Figure 4A (MLP quintile line plots) ...")

fig4a, axes4a = plt.subplots(1, 3, figsize=(13, 4.5))
fig4a.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.20,
                      wspace=0.40)

quint_labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]
quint_x_num  = np.array([1, 2, 3, 4, 5])   # numeric x for OLS

for ax, (out_label, col) in zip(axes4a, OUTCOMES.items()):
    sub   = quint_df[quint_df["Outcome"] == out_label]
    means = sub["Mean"].values
    sems  = sub["SEM"].values
    ns    = sub["N"].values
    color = OUTCOME_COLORS[out_label]

    # ---- line + error bars ----
    ax.errorbar(quint_x_num, means, yerr=sems,
                fmt="o-", color=color,
                markersize=7, linewidth=1.8,
                capsize=4, capthick=1.2,
                elinewidth=1.0, zorder=4,
                label="Mean ± SEM")

    # ---- OLS regression dashed line ----
    slope, intercept, r_line, p_line, _ = stats.linregress(quint_x_num, means)
    x_fit = np.linspace(0.7, 5.3, 100)
    ax.plot(x_fit, slope * x_fit + intercept,
            linestyle="--", color="#D2691E",
            linewidth=1.4, zorder=3, label="OLS trend")

    # ---- Pearson r annotation ----
    valid = su[col].notna().values
    r_val, p_val = stats.pearsonr(mlp_scores[valid], su[col].values[valid])
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
          ("*" if p_val < 0.05 else "ns"))

    # OLS 95% CI for slope (N=5 quintile means, but use individual-level for annotation)
    ci_lo = r_val - 1.96 * np.sqrt((1 - r_val**2) / max(valid.sum() - 2, 1))
    ci_hi = r_val + 1.96 * np.sqrt((1 - r_val**2) / max(valid.sum() - 2, 1))

    annot_text = (f"r = {r_val:.3f}{sig}\n"
                  f"β = {slope:.4f}\n"
                  f"N = {valid.sum()}")
    ax.text(0.97, 0.97, annot_text,
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#cccccc", alpha=0.88))

    # ---- n labels above/below each point ----
    y_range = means.max() - means.min()
    for xi, m, s, n in zip(quint_x_num, means, sems, ns):
        ax.text(xi, m + s + y_range * 0.05,
                f"n={n}", ha="center", va="bottom",
                fontsize=6, color="#666666")

    # ---- y-axis: tight range to magnify trend ----
    y_lo = means.min() - sems.max() * 3 - y_range * 0.25
    y_hi = means.max() + sems.max() * 3 + y_range * 0.55
    ax.set_ylim(y_lo, y_hi)

    ax.set_xticks(quint_x_num)
    ax.set_xticklabels(quint_labels, fontsize=8.5)
    ax.set_xlim(0.5, 5.5)
    ax.set_xlabel("Predicted POAG Risk Quintile (MLP)", fontsize=9)
    ax.set_ylabel(out_label, fontsize=9)
    direction = OUTCOME_DIRECTION[out_label]
    ax.set_title(f"{out_label}  ({direction} with risk)",
                 fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

# suptitle removed per journal style

fig4a.savefig(_os.path.join(OUT_FIG, "Figure_4A_MLP_Quintile_LinePlot.png"),
              bbox_inches="tight", dpi=300)
fig4a.savefig(_os.path.join(OUT_FIG, "Figure_4A_MLP_Quintile_LinePlot.pdf"),
              bbox_inches="tight", dpi=300)
plt.close(fig4a)
print("  Figure 4A saved.")


# =============================================================
# 6B. FIGURE 4B — ECDF: Low 25% vs High 25% MLP risk (1×3)
# =============================================================
print("Building Figure 4B (ECDF low vs high 25% risk) ...")

# Define risk groups
threshold_lo = np.percentile(mlp_scores, 25)
threshold_hi = np.percentile(mlp_scores, 75)
mask_lo = mlp_scores <= threshold_lo
mask_hi = mlp_scores >= threshold_hi

CLR_LO = "#2CA02C"   # green  — low risk
CLR_HI = "#FF7F0E"   # orange — high risk

fig4b, axes4b = plt.subplots(1, 3, figsize=(13, 4.5))
fig4b.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.18,
                      wspace=0.40)

for ax, (out_label, col) in zip(axes4b, OUTCOMES.items()):
    valid = su[col].notna().values

    vals_lo = su[col].values[mask_lo & valid]
    vals_hi = su[col].values[mask_hi & valid]

    # KS test
    ks_stat, ks_p = stats.ks_2samp(vals_lo, vals_hi)
    ks_sig = "***" if ks_p < 0.001 else ("**" if ks_p < 0.01 else
             ("*" if ks_p < 0.05 else "ns"))

    # Plot ECDF
    for vals, clr, lbl in [(vals_lo, CLR_LO, f"Low 25%  (n={len(vals_lo)})"),
                            (vals_hi, CLR_HI, f"High 25%  (n={len(vals_hi)})")]:
        sorted_v = np.sort(vals)
        ecdf     = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.step(sorted_v, ecdf, where="post",
                color=clr, linewidth=1.8, label=lbl)

    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    ax.set_xlabel(out_label, fontsize=9)
    ax.set_ylabel("Cumulative proportion", fontsize=9)
    ax.set_title(f"{out_label}\nKS p = {ks_p:.4f}{ks_sig}",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

# suptitle removed per journal style

fig4b.savefig(_os.path.join(OUT_FIG, "Figure_4B_MLP_ECDF_LowVsHigh.png"),
              bbox_inches="tight", dpi=300)
fig4b.savefig(_os.path.join(OUT_FIG, "Figure_4B_MLP_ECDF_LowVsHigh.pdf"),
              bbox_inches="tight", dpi=300)
plt.close(fig4b)
print("  Figure 4B saved.")


# =============================================================
# 6C. COMBINED Figure 4 (4A on top, 4B on bottom)
# =============================================================
print("Building combined Figure 4 ...")

fig4c = plt.figure(figsize=(13, 9.5))
gs4   = gridspec.GridSpec(2, 3, figure=fig4c,
                           hspace=0.55, wspace=0.40,
                           left=0.08, right=0.97,
                           top=0.93, bottom=0.08)

panel_labels_done = set()

# ---- Row 0: quintile line plots ----
for ci, (out_label, col) in enumerate(OUTCOMES.items()):
    ax = fig4c.add_subplot(gs4[0, ci])
    sub   = quint_df[quint_df["Outcome"] == out_label]
    means = sub["Mean"].values
    sems  = sub["SEM"].values
    ns    = sub["N"].values
    color = OUTCOME_COLORS[out_label]

    ax.errorbar(quint_x_num, means, yerr=sems,
                fmt="o-", color=color,
                markersize=7, linewidth=1.8,
                capsize=4, capthick=1.2, elinewidth=1.0, zorder=4)

    slope, intercept, _, _, _ = stats.linregress(quint_x_num, means)
    x_fit = np.linspace(0.7, 5.3, 100)
    ax.plot(x_fit, slope * x_fit + intercept,
            linestyle="--", color="#D2691E", linewidth=1.4, zorder=3)

    valid = su[col].notna().values
    r_val, p_val = stats.pearsonr(mlp_scores[valid], su[col].values[valid])
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
          ("*" if p_val < 0.05 else "ns"))
    ax.text(0.97, 0.97,
            f"r = {r_val:.3f}{sig}\nβ = {slope:.4f}\nN = {valid.sum()}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#cccccc", alpha=0.88))

    y_range = means.max() - means.min()
    for xi, m, s, n in zip(quint_x_num, means, sems, ns):
        ax.text(xi, m + s + y_range * 0.05,
                f"n={n}", ha="center", va="bottom",
                fontsize=6, color="#666666")

    y_lo = means.min() - sems.max() * 3 - y_range * 0.25
    y_hi = means.max() + sems.max() * 3 + y_range * 0.55
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(quint_x_num)
    ax.set_xticklabels(quint_labels, fontsize=8)
    ax.set_xlim(0.5, 5.5)
    ax.set_xlabel("Risk Quintile (MLP)", fontsize=8.5)
    ax.set_ylabel(out_label, fontsize=8.5)
    direction = OUTCOME_DIRECTION[out_label]
    ax.set_title(f"{out_label}  ({direction} with risk)",
                 fontsize=9.5, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    if ci == 0:
        ax.text(-0.18, 1.12, "A", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

# ---- Row 1: ECDF ----
for ci, (out_label, col) in enumerate(OUTCOMES.items()):
    ax = fig4c.add_subplot(gs4[1, ci])
    valid = su[col].notna().values
    vals_lo = su[col].values[mask_lo & valid]
    vals_hi = su[col].values[mask_hi & valid]
    ks_stat, ks_p = stats.ks_2samp(vals_lo, vals_hi)
    ks_sig = "***" if ks_p < 0.001 else ("**" if ks_p < 0.01 else
             ("*" if ks_p < 0.05 else "ns"))

    for vals, clr, lbl in [(vals_lo, CLR_LO, f"Low 25%  (n={len(vals_lo)})"),
                            (vals_hi, CLR_HI, f"High 25%  (n={len(vals_hi)})")]:
        sorted_v = np.sort(vals)
        ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.step(sorted_v, ecdf, where="post",
                color=clr, linewidth=1.8, label=lbl)

    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    ax.set_xlabel(out_label, fontsize=8.5)
    ax.set_ylabel("Cumulative proportion", fontsize=8.5)
    ax.set_title(f"{out_label}\nKS p = {ks_p:.4f}{ks_sig}",
                 fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    if ci == 0:
        ax.text(-0.18, 1.18, "B", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

# suptitle removed per journal style

fig4c.savefig(_os.path.join(OUT_FIG, "Figure_4_Combined.png"),
              bbox_inches="tight", dpi=300)
fig4c.savefig(_os.path.join(OUT_FIG, "Figure_4_Combined.pdf"),
              bbox_inches="tight", dpi=300)
plt.close(fig4c)
print("  Combined Figure 4 saved.")


# =============================================================
# 7.  SUPPLEMENTAL — Pearson r heatmap (all models × all feature sets)
# =============================================================
print("Building supplemental heatmap ...")

# One heatmap per clinical outcome, side by side
fig_s, axes_s = plt.subplots(1, 3, figsize=(15, 5))
fig_s.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.28,
                      wspace=0.5)

for ax_s, (out_label, col) in zip(axes_s, OUTCOMES.items()):
    sub = corr_df[corr_df["Outcome"] == out_label].copy()
    piv_r = sub.pivot(index="FeatureSet", columns="Model", values="Pearson_r")
    piv_p = sub.pivot(index="FeatureSet", columns="Model", values="p_value")

    # row order
    row_order = list(FEATURE_SETS.keys())
    col_order  = MODEL_NAMES
    piv_r = piv_r.reindex(index=row_order, columns=col_order)
    piv_p = piv_p.reindex(index=row_order, columns=col_order)

    vmax = max(corr_df["Pearson_r"].abs().max(), 0.25)
    im = ax_s.imshow(piv_r.values, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax_s, label="Pearson r", shrink=0.75)

    ax_s.set_xticks(range(len(col_order)))
    ax_s.set_xticklabels(col_order, fontsize=9)
    ax_s.set_yticks(range(len(row_order)))
    ax_s.set_yticklabels(row_order, fontsize=8)
    ax_s.set_title(f"{out_label}", fontsize=10, fontweight="bold")

    for i, fs in enumerate(row_order):
        for j, mn in enumerate(col_order):
            r_v = piv_r.loc[fs, mn]
            p_v = piv_p.loc[fs, mn]
            sig = "***" if p_v < 0.001 else ("**" if p_v < 0.01 else
                  ("*" if p_v < 0.05 else ""))
            ax_s.text(j, i, f"{r_v:.2f}{sig}",
                      ha="center", va="center", fontsize=7.5,
                      color="white" if abs(r_v) > vmax * 0.5 else "black")

# suptitle removed per journal style
fig_s.savefig(_os.path.join(OUT_FIG, "SF6A_SuppFig_Suspects_AllModels_Heatmap.png"),
              bbox_inches="tight", dpi=300)
fig_s.savefig(_os.path.join(OUT_FIG, "SF6A_SuppFig_Suspects_AllModels_Heatmap.pdf"),
              bbox_inches="tight", dpi=300)
plt.close(fig_s)
print("  Supplemental heatmap saved.")


# =============================================================
# 8.  SUPPLEMENTAL — Scatter plots for MLP across all feature sets
# =============================================================
print("Building supplemental scatter plots (MLP, all feature sets) ...")
n_fs  = len(FEATURE_SETS)
n_out = len(OUTCOMES)

fig_sc, axes_sc = plt.subplots(n_fs, n_out,
                                figsize=(4.5 * n_out, 4 * n_fs),
                                squeeze=False)
fig_sc.subplots_adjust(hspace=0.55, wspace=0.38,
                        left=0.07, right=0.97, top=0.95, bottom=0.05)

for ri, (fs_name, cfg) in enumerate(FEATURE_SETS.items()):
    scores_fs = pred_scores[(fs_name, "MLP")]
    for ci, (out_label, col) in enumerate(OUTCOMES.items()):
        ax_sc = axes_sc[ri][ci]
        valid = su[col].notna().values
        x_plot = scores_fs[valid]
        y_plot = su[col].values[valid]
        r_val, p_val = stats.pearsonr(x_plot, y_plot)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
              ("*" if p_val < 0.05 else "ns"))

        ax_sc.scatter(x_plot, y_plot, alpha=0.25, s=6,
                      color=OUTCOME_COLORS[out_label], zorder=2)
        # regression line
        m, b = np.polyfit(x_plot, y_plot, 1)
        xr = np.array([x_plot.min(), x_plot.max()])
        ax_sc.plot(xr, m * xr + b, color="#333333",
                   linewidth=1.2, zorder=3)

        ax_sc.text(0.97, 0.97,
                   f"r = {r_val:.3f}{sig}\nN = {valid.sum()}",
                   transform=ax_sc.transAxes,
                   ha="right", va="top", fontsize=7.5,
                   bbox=dict(boxstyle="round,pad=0.25",
                             facecolor="white", edgecolor="#cccccc",
                             alpha=0.85))
        if ri == 0:
            ax_sc.set_title(out_label, fontsize=9, fontweight="bold")
        if ci == 0:
            ax_sc.set_ylabel(fs_name, fontsize=8)
        ax_sc.set_xlabel("Predicted risk (MLP)", fontsize=7.5)
        ax_sc.spines["top"].set_visible(False)
        ax_sc.spines["right"].set_visible(False)

# suptitle removed per journal style
fig_sc.savefig(_os.path.join(OUT_FIG, "SF6B_SuppFig_Suspects_MLP_Scatter.png"),
               bbox_inches="tight", dpi=300)
fig_sc.savefig(_os.path.join(OUT_FIG, "SF6B_SuppFig_Suspects_MLP_Scatter.pdf"),
               bbox_inches="tight", dpi=300)
plt.close(fig_sc)
print("  Supplemental scatter plots saved.")


# =============================================================
# 9.  PRINT KEY NUMBERS
# =============================================================
print("\n=== KEY NUMBERS ===")
print("\nMLP × Base+PGS616 correlations with clinical outcomes:")
for out_label in OUTCOMES:
    row = corr_df[(corr_df["FeatureSet"]=="Base+PGS616") &
                  (corr_df["Model"]=="MLP") &
                  (corr_df["Outcome"]==out_label)]
    print(f"  {out_label}: r={row['Pearson_r'].values[0]:.3f}, "
          f"p={row['p_value'].values[0]:.6f}, N={row['N'].values[0]}, "
          f"{row['Significant'].values[0]}")

print("\nAll models × Base+PGS616 correlations:")
sub = corr_df[corr_df["FeatureSet"]=="Base+PGS616"]
print(sub[["Model","Outcome","Pearson_r","p_value","Significant","N"]].to_string(index=False))

print("\nQuintile means (MLP × Base+PGS616):")
print(quint_df.to_string(index=False))

print("\nAll done.")
