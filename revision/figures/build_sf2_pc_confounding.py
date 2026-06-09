# =============================================================
#  Build Supplemental Figure SF_PC_Confounding
#  Addresses R2-KI1: PC-only model (age+sex+PC1-20) claimed AUC=0.87
#
#  Panel A: Training AUC under 5x20 CV — PC progression vs PGS616
#           with reference line at 0.87 (prior non-CV estimate)
#  Panel B: PMBB external AUC — PGS generalizes better than PCs
# =============================================================
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

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
OUT_XL = fr"{BASE}\output excel data"
OUT_FIG= fr"{BASE}\figure"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

MODEL_ORDER  = ["LR", "SVM", "RF", "MLP", "Average"]
MODEL_COLORS = {
    "LR": "#4E79A7", "SVM": "#F28E2B",
    "RF": "#59A14F", "MLP": "#E15759", "Average": "#B07AA1",
}

# ── Load data ─────────────────────────────────────────────────
print("Loading data ...")
cv_main = pd.read_excel(fr"{OUT_XL}\Table_Figure3_Training_5x20CV.xlsx",
                         sheet_name="Full_Results")
cv_pc20 = pd.read_excel(fr"{OUT_XL}\Table_Figure3_Training_5x20CV.xlsx",
                         sheet_name="PC20_Results")
pmbb_df = pd.read_excel(fr"{OUT_XL}\Table_Figure3_PMBB_External.xlsx",
                         sheet_name="PMBB_AFR")

# Combine training data (AUC metric only)
auc_main = cv_main[cv_main["Metric"] == "AUC"].copy()
auc_pc20 = cv_pc20[cv_pc20["Metric"] == "AUC"].copy()
auc_all  = pd.concat([auc_main, auc_pc20], ignore_index=True)

# PMBB rename for consistency
pmbb_auc = pmbb_df.rename(columns={
    "AUC": "Mean", "AUC_CI_lower": "CI_lower", "AUC_CI_upper": "CI_upper"})


# ── Panel A: Dot plot — PC progression vs PGS616 ──────────────
# Feature sets to compare (PC1-20 range + PGS models)
FS_A = [
    "Base",
    "Base+PC2",
    "Base+PC5",
    "Base+PC10",
    "Base+PC20",        # the exact configuration reviewer cited
    "Base+PGS616",
]
FS_COLORS_A = {
    "Base":        "#888888",
    "Base+PC2":    "#F0A500",
    "Base+PC5":    "#E07B54",
    "Base+PC10":   "#C44E52",
    "Base+PC20":   "#8B1A1A",   # darkest red — most PCs
    "Base+PGS616": "#2196A6",   # teal — PGS
}
FS_MARKERS_A = {
    "Base":        "o",
    "Base+PC2":    "s",
    "Base+PC5":    "^",
    "Base+PC10":   "D",
    "Base+PC20":   "P",
    "Base+PGS616": "D",
}

CLF_ORDER = ["LR", "SVM", "RF", "MLP", "Average"]
PRIOR_CLAIM = 0.87   # reviewer-cited AUC from prior non-CV analysis

def dot_panel_A(ax, data_df, fs_list, clf_list, title, xlim=(0.55, 0.95)):
    n_clf    = len(clf_list)
    n_fs     = len(fs_list)
    spacing  = 0.16
    row_gap  = 1.0

    yticks, yticklabs = [], []

    for ci, clf in enumerate(clf_list):
        y_center = ci * row_gap
        yticks.append(y_center)
        yticklabs.append(clf)

        offsets = np.linspace(-(n_fs - 1) * spacing / 2,
                               (n_fs - 1) * spacing / 2, n_fs)

        for fi, fs in enumerate(fs_list):
            row = data_df[(data_df["FeatureSet"] == fs) & (data_df["Model"] == clf)]
            if len(row) == 0:
                continue
            mn  = float(row["Mean"].iloc[0])
            lo  = float(row["CI_lower"].iloc[0]) \
                  if not pd.isna(row["CI_lower"].iloc[0]) else mn
            hi  = float(row["CI_upper"].iloc[0]) \
                  if not pd.isna(row["CI_upper"].iloc[0]) else mn

            ypos  = y_center + offsets[fi]
            color = FS_COLORS_A[fs]
            mrk   = FS_MARKERS_A[fs]

            ax.plot(mn, ypos, marker=mrk,
                    color=color, markersize=7, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.errorbar(mn, ypos, xerr=[[mn - lo], [hi - mn]],
                        fmt="none", color=color,
                        capsize=3, linewidth=1.2, zorder=3)
            ax.text(hi + (xlim[1] - xlim[0]) * 0.012, ypos, f"{mn:.3f}",
                    va="center", ha="left", fontsize=7, color=color)

        if ci % 2 == 0:
            ax.axhspan(y_center - row_gap / 2, y_center + row_gap / 2,
                       color="#f5f5f5", zorder=0)

    # Reference line: prior non-CV claim
    ax.axvline(PRIOR_CLAIM, color="#CC0000", linewidth=1.4,
               linestyle="--", zorder=2, alpha=0.8)
    ax.text(PRIOR_CLAIM + (xlim[1] - xlim[0]) * 0.01,
            (n_clf - 1) * row_gap + 0.45,
            f"Prior non-CV estimate\n(AUC = {PRIOR_CLAIM})",
            fontsize=7.5, color="#CC0000", va="top", ha="left")

    # Chance line
    ax.axvline(0.5, color="dimgrey", linewidth=0.9,
               linestyle=":", zorder=1)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabs, fontsize=9.5)
    ax.set_xlabel("AUC (mean, 95% CI)", fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, (n_clf - 1) * row_gap + 0.6)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Legend: PC models (warm) and PGS model (teal)
    handles = []
    for fs in fs_list:
        handles.append(mlines.Line2D(
            [], [], color=FS_COLORS_A[fs],
            marker=FS_MARKERS_A[fs], markersize=7,
            markeredgecolor="white", markeredgewidth=0.6,
            linewidth=0, label=fs))
    handles.append(mlines.Line2D(
        [], [], color="#CC0000", linewidth=1.4,
        linestyle="--", label=f"Prior estimate ({PRIOR_CLAIM})"))
    ax.legend(handles=handles, frameon=True, framealpha=0.9,
              fontsize=8, loc="lower right")


# ── Panel B: PMBB external — PC vs PGS bar chart ──────────────
FS_B = ["Base", "Base+PGS526", "Base+PGS616",
        "Base+PC5", "Base+PC5+PGS616"]
FS_LABELS_B = ["Base", "Base\n+PGS526", "Base\n+PGS616",
               "Base\n+PC5", "Base+PC5\n+PGS616"]
HIGHLIGHT_PC  = [3, 4]   # indices of PC-augmented bars
HIGHLIGHT_PGS = [1, 2]   # indices of PGS-augmented bars

def bar_panel_B(ax, data_df, fs_list, fs_labels, title, ylim=(0.40, 0.92)):
    n_fs  = len(fs_list)
    n_mod = len(MODEL_ORDER)
    bar_w = 0.13
    x     = np.arange(n_fs)

    for i, mod in enumerate(MODEL_ORDER):
        means, errs = [], []
        for fs in fs_list:
            row = data_df[(data_df["FeatureSet"] == fs) & (data_df["Model"] == mod)]
            if len(row) == 0:
                means.append(np.nan); errs.append(np.nan)
            else:
                mn = float(row["Mean"].iloc[0])
                lo = float(row["CI_lower"].iloc[0]) \
                     if not pd.isna(row["CI_lower"].iloc[0]) else np.nan
                means.append(mn)
                errs.append(mn - lo if not np.isnan(lo) else 0)

        offset = (i - (n_mod - 1) / 2) * bar_w
        ax.bar(x + offset, means, bar_w,
               color=MODEL_COLORS[mod], alpha=0.85,
               label=mod, zorder=3)
        ax.errorbar(x + offset, means, yerr=errs,
                    fmt="none", color="black",
                    capsize=2.0, linewidth=0.7, zorder=4)
        for xpos, mv, err in zip(x + offset, means, errs):
            if np.isnan(mv):
                continue
            top = mv + (err if not np.isnan(err) else 0) + 0.008
            ax.text(xpos, top, f"{mv:.3f}",
                    ha="center", va="bottom", fontsize=5.5, color="black")

    # Shade PC-augmented bars (orange) and PGS bars (blue)
    for j in HIGHLIGHT_PGS:
        ax.axvspan(j - 0.44, j + 0.44, color="#2196A6", alpha=0.07, zorder=0)
    for j in HIGHLIGHT_PC:
        ax.axvspan(j - 0.44, j + 0.44, color="#FF8C00", alpha=0.10, zorder=0)

    ax.axhline(0.5, color="dimgrey", linewidth=0.9, linestyle=":", zorder=2)

    ax.axvline(2.5, color="#cccccc", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(1.5, 0.945, "PGS augmented", ha="center", va="bottom",
            fontsize=7.5, color="#2196A6", fontstyle="italic",
            transform=ax.get_xaxis_transform())
    ax.text(3.5, 0.945, "PC augmented", ha="center", va="bottom",
            fontsize=7.5, color="#FF8C00", fontstyle="italic",
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(fs_labels, fontsize=8.5)
    ax.set_ylabel("AUC (mean ± 95% CI)", fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=3)


# =============================================================
# Build figure
# =============================================================
print("Building SF_PC_Confounding ...")
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 5.5))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.16,
                    wspace=0.38)

n_cases    = int(pmbb_df["N_cases"].iloc[0])
n_controls = int(pmbb_df["N_controls"].iloc[0])
n_total    = n_cases + n_controls

dot_panel_A(
    ax_a, auc_all, FS_A, CLF_ORDER,
    title="",
    xlim=(0.55, 0.97),
)
ax_a.text(-0.10, 1.08, "A", transform=ax_a.transAxes,
          fontsize=14, fontweight="bold", va="top")

bar_panel_B(
    ax_b, pmbb_auc, FS_B, FS_LABELS_B,
    title="",
    ylim=(0.38, 0.95),
)
ax_b.text(-0.10, 1.08, "B", transform=ax_b.transAxes,
          fontsize=14, fontweight="bold", va="top")

# suptitle removed per journal style

for ext in ["png", "pdf"]:
    fig.savefig(fr"{OUT_FIG}\SF2B_PC_Confounding.{ext}",
                bbox_inches="tight", dpi=300)
    print(f"  Saved SF2B_PC_Confounding.{ext}")

plt.close(fig)
print("Done.")
