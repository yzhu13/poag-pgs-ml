# =============================================================
#  Build Figure 3 panels from saved Excel data
#  3A — all 12 feature sets (bar chart)
#  3B — key comparison: Base / Base+PC2 / Base+PC10 / Base+PGS616
#        format: dot plot (forest-plot style), classifiers on y-axis
#  3C — PMBB external validation (bar chart)
# =============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

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

MODEL_ORDER  = ["LR", "SVM", "RF", "MLP", "Average"]
MODEL_COLORS = {
    "LR": "#4E79A7", "SVM": "#F28E2B",
    "RF": "#59A14F", "MLP": "#E15759", "Average": "#B07AA1",
}

# Feature set display order for 3A
FS_ORDER_3A = [
    "Age only", "Sex only", "Base",
    "Base+PC2", "Base+PC5", "Base+PC10",
    "Base+POAAGG PGS", "Base+MEGA PGS", "Base+PGS526", "Base+PGS616",
    "Base+PC5+PGS526", "Base+PC5+PGS616",
]

# Shorter x-tick labels for 3A
FS_LABELS_3A = [
    "Age", "Sex", "Base",
    "Base\n+PC2", "Base\n+PC5", "Base\n+PC10",
    "Base\n+POAAGG", "Base\n+MEGA", "Base\n+PGS526", "Base\n+PGS616",
    "Base+PC5\n+PGS526", "Base+PC5\n+PGS616",
]

# Key feature sets for 3B dot plot
FS_KEY  = ["Base", "Base+PC2", "Base+PC10", "Base+PGS616"]
FS_COLORS_B = {
    "Base":         "#888888",
    "Base+PC2":     "#E07B54",
    "Base+PC10":    "#C44E52",
    "Base+PGS616":  "#2196A6",
}

# PMBB feature sets for 3C
FS_ORDER_3C = [
    "Base", "Base+PGS526", "Base+PGS616",
    "Base+PC5", "Base+PC5+PGS526", "Base+PC5+PGS616",
]
FS_LABELS_3C = [
    "Base", "Base\n+PGS526", "Base\n+PGS616",
    "Base\n+PC5", "Base+PC5\n+PGS526", "Base+PC5\n+PGS616",
]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# ── load data ─────────────────────────────────────────────────
print("Loading Excel data ...")
cv_full  = pd.read_excel(fr"{OUT_XL}\Table_Figure3_Training_5x20CV.xlsx",
                          sheet_name="Full_Results")
pmbb_df  = pd.read_excel(fr"{OUT_XL}\Table_Figure3_PMBB_External.xlsx",
                          sheet_name="PMBB_AFR")
auc_cv   = cv_full[cv_full["Metric"] == "AUC"].copy()

# ── helper: bar chart (used for 3A and 3C) ────────────────────
def bar_panel(ax, data_df, fs_list, fs_labels, title,
              ylim=(0.38, 0.95), highlight_groups=None,
              show_legend=True, label_size=5.5, legend_kw=None):
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
                     if "CI_lower" in row.columns and not pd.isna(row["CI_lower"].iloc[0]) \
                     else np.nan
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
                    ha="center", va="bottom",
                    fontsize=label_size, rotation=0, color="black")

    ax.axhline(0.5, color="dimgrey", linewidth=0.9,
               linestyle=":", label="Chance (0.5)", zorder=2)

    # optional shading for PC groups
    if highlight_groups:
        for grp_indices, color in highlight_groups:
            for j in grp_indices:
                ax.axvspan(j - 0.44, j + 0.44,
                           color=color, alpha=0.10, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(fs_labels, fontsize=8.5)
    ax.set_ylabel("AUC (mean ± 95% CI)", fontsize=9)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    if show_legend:
        kw = dict(loc="upper left", frameon=False, fontsize=7.5, ncol=3)
        if legend_kw:
            kw.update(legend_kw)
        ax.legend(**kw)


# ── helper: dot plot (used for 3B) ────────────────────────────
def dot_panel(ax, data_df, fs_list, clf_list, title, xlim=(0.55, 0.82)):
    """
    Horizontal dot-plot (forest-plot style).
    y-axis  = classifiers (one row per classifier)
    x-axis  = AUC
    series  = feature sets (different colours / symbols)
    """
    n_clf = len(clf_list)
    n_fs  = len(fs_list)

    # vertical spacing: each classifier block is 1 unit; feature sets offset within block
    spacing   = 0.18          # vertical gap between feature-set dots within one clf row
    row_gap   = 1.0           # gap between classifier groups

    markers  = ["o", "s", "^", "D"]     # one marker per feature set
    yticks   = []
    yticklabs= []

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
                  if "CI_lower" in row.columns and not pd.isna(row["CI_lower"].iloc[0]) \
                  else mn
            hi  = float(row["CI_upper"].iloc[0]) \
                  if "CI_upper" in row.columns and not pd.isna(row["CI_upper"].iloc[0]) \
                  else mn

            ypos = y_center + offsets[fi]
            color = FS_COLORS_B[fs]

            ax.plot(mn, ypos, marker=markers[fi],
                    color=color, markersize=7, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.errorbar(mn, ypos, xerr=[[mn - lo], [hi - mn]],
                        fmt="none", color=color,
                        capsize=3, linewidth=1.2, zorder=3)
            ax.text(hi + 0.004, ypos, f"{mn:.3f}",
                    va="center", ha="left",
                    fontsize=7.5, color=color)

        # light horizontal band every other classifier
        if ci % 2 == 0:
            ax.axhspan(y_center - row_gap / 2, y_center + row_gap / 2,
                       color="#f5f5f5", zorder=0)

    # chance line
    ax.axvline(0.5, color="dimgrey", linewidth=0.9,
               linestyle=":", label="Chance (0.5)", zorder=2)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabs, fontsize=9.5)
    ax.set_xlabel("AUC (mean, 95% CI)", fontsize=9)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.6, (n_clf - 1) * row_gap + 0.6)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # legend
    legend_handles = [
        mlines.Line2D([], [], color=FS_COLORS_B[fs],
                      marker=markers[fi], markersize=7,
                      markeredgecolor="white", markeredgewidth=0.6,
                      linewidth=0, label=fs)
        for fi, fs in enumerate(fs_list)
    ]
    ax.legend(handles=legend_handles, frameon=False,
              fontsize=8, loc="lower right")


# =============================================================
# FIGURE 3A — all 12 feature sets, bar chart
# =============================================================
print("Building Figure 3A ...")
fig3a, ax3a = plt.subplots(figsize=(16, 5.5))
fig3a.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.20)

bar_panel(
    ax3a, auc_cv, FS_ORDER_3A, FS_LABELS_3A,
    title="Training Cohort (N=271) — 5×20 CV AUC\n"
          "All Feature Sets: Age / Sex / Base / PC / PGS / Combined",
    ylim=(0.38, 0.95),
    highlight_groups=[
        ([3, 4, 5], "#FF8C00"),      # PC groups: orange tint
        ([10, 11], "#9C27B0"),        # PC+PGS: purple tint
    ],
    label_size=5.2,
    legend_kw=dict(bbox_to_anchor=(0.0, 0.88), loc="upper left"),
)
# add vertical dividers to separate groups visually
for xpos in [2.5, 5.5, 9.5]:
    ax3a.axvline(xpos, color="#cccccc", linewidth=1.0,
                 linestyle="--", zorder=1)

# group labels above x-axis
group_info = [
    (1, "Demographics"),
    (4, "Demographics + PC"),
    (7.5, "Demographics + PGS"),
    (10.5, "+ PC & PGS"),
]
for xpos, label in group_info:
    ax3a.text(xpos, 0.945, label,
              ha="center", va="bottom",
              fontsize=7.5, color="#555555",
              fontstyle="italic",
              transform=ax3a.get_xaxis_transform())

ax3a.text(-0.03, 1.06, "A", transform=ax3a.transAxes,
          fontsize=15, fontweight="bold", va="top")
fig3a.savefig(fr"{OUT_FIG}\Figure_3A_All_FeatureSets.png",
              bbox_inches="tight", dpi=300)
fig3a.savefig(fr"{OUT_FIG}\Figure_3A_All_FeatureSets.pdf",
              bbox_inches="tight", dpi=300)
plt.close(fig3a)
print("  Fig 3A saved.")


# =============================================================
# FIGURE 3B — dot plot: Base / Base+PC2 / Base+PC10 / Base+PGS616
# =============================================================
print("Building Figure 3B ...")
fig3b, ax3b = plt.subplots(figsize=(7, 5))
fig3b.subplots_adjust(left=0.16, right=0.88, top=0.88, bottom=0.12)

# classifiers bottom→top so Average is at top
CLF_ORDER_B = ["LR", "SVM", "RF", "MLP", "Average"]

dot_panel(
    ax3b, auc_cv, FS_KEY, CLF_ORDER_B,
    title="Key Comparison — Training Cohort\n"
          "Base vs PC-augmented vs PGS-augmented",
    xlim=(0.55, 0.84),
)
ax3b.text(-0.12, 1.06, "B", transform=ax3b.transAxes,
          fontsize=15, fontweight="bold", va="top")
fig3b.savefig(fr"{OUT_FIG}\Figure_3B_KeyComparison_DotPlot.png",
              bbox_inches="tight", dpi=300)
fig3b.savefig(fr"{OUT_FIG}\Figure_3B_KeyComparison_DotPlot.pdf",
              bbox_inches="tight", dpi=300)
plt.close(fig3b)
print("  Fig 3B saved.")


# =============================================================
# FIGURE 3C — PMBB external validation, bar chart
# =============================================================
print("Building Figure 3C ...")
pmbb_auc = pmbb_df.rename(columns={
    "AUC": "Mean", "AUC_CI_lower": "CI_lower", "AUC_CI_upper": "CI_upper"})
n_cases    = int(pmbb_df["N_cases"].iloc[0])
n_controls = int(pmbb_df["N_controls"].iloc[0])
n_total    = n_cases + n_controls

fig3c, ax3c = plt.subplots(figsize=(11, 5))
fig3c.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.22)

bar_panel(
    ax3c, pmbb_auc, FS_ORDER_3C, FS_LABELS_3C,
    title=f"PMBB External Validation — African Ancestry (N={n_total:,})\n"
          f"Cases = {n_cases}, Controls = {n_controls}",
    ylim=(0.38, 0.95),
    highlight_groups=[
        ([3, 4, 5], "#FF8C00"),   # PC-augmented: orange tint
    ],
)
# divider between base-only and PC-augmented sets
ax3c.axvline(2.5, color="#cccccc", linewidth=1.0, linestyle="--", zorder=1)
ax3c.text(1.0, 0.945, "PGS augmented",
          ha="center", va="bottom", fontsize=7.5,
          color="#555555", fontstyle="italic",
          transform=ax3c.get_xaxis_transform())
ax3c.text(4.0, 0.945, "PC augmented",
          ha="center", va="bottom", fontsize=7.5,
          color="#555555", fontstyle="italic",
          transform=ax3c.get_xaxis_transform())

ax3c.text(-0.08, 1.06, "C", transform=ax3c.transAxes,
          fontsize=15, fontweight="bold", va="top")
fig3c.savefig(fr"{OUT_FIG}\Figure_3C_PMBB_External.png",
              bbox_inches="tight", dpi=300)
fig3c.savefig(fr"{OUT_FIG}\Figure_3C_PMBB_External.pdf",
              bbox_inches="tight", dpi=300)
plt.close(fig3c)
print("  Fig 3C saved.")


# =============================================================
# COMBINED Figure 3 (A / B / C stacked vertically)
# =============================================================
print("Building combined Figure 3 ...")
fig3 = plt.figure(figsize=(18, 16))
gs = fig3.add_gridspec(3, 2,
                        height_ratios=[1, 1, 1],
                        hspace=0.55, wspace=0.38,
                        left=0.07, right=0.97,
                        top=0.95, bottom=0.07)

# 3A spans both columns on row 0
ax3a2 = fig3.add_subplot(gs[0, :])
bar_panel(
    ax3a2, auc_cv, FS_ORDER_3A, FS_LABELS_3A,
    title="A  All Feature Sets — 5×20 CV AUC (Training Cohort, N=271)",
    ylim=(0.38, 0.95), label_size=5.0,
    highlight_groups=[
        ([3, 4, 5], "#FF8C00"),
        ([10, 11], "#9C27B0"),
    ],
    legend_kw=dict(bbox_to_anchor=(0.0, 0.88), loc="upper left"),
)
for xpos in [2.5, 5.5, 9.5]:
    ax3a2.axvline(xpos, color="#cccccc", linewidth=1.0, linestyle="--", zorder=1)
for xpos, label in group_info:
    ax3a2.text(xpos, 0.945, label, ha="center", va="bottom",
               fontsize=7.5, color="#555555", fontstyle="italic",
               transform=ax3a2.get_xaxis_transform())
ax3a2.text(-0.025, 1.05, "A", transform=ax3a2.transAxes,
           fontsize=15, fontweight="bold", va="top")

# 3B: dot plot, left column row 1
ax3b2 = fig3.add_subplot(gs[1, 0])
dot_panel(
    ax3b2, auc_cv, FS_KEY, CLF_ORDER_B,
    title="B  Key Comparison: Base vs PC vs PGS (Training)",
    xlim=(0.55, 0.84),
)
ax3b2.text(-0.14, 1.06, "B", transform=ax3b2.transAxes,
           fontsize=15, fontweight="bold", va="top")

# 3C: PMBB, right column row 1 and spans row 2 as well?
# Actually put 3C spanning both cols on row 2 for consistency
ax3c2 = fig3.add_subplot(gs[1, 1])
bar_panel(
    ax3c2, pmbb_auc, FS_ORDER_3C, FS_LABELS_3C,
    title=f"C  PMBB External Validation (N={n_total:,})",
    ylim=(0.38, 0.95), label_size=5.5,
    highlight_groups=[([3, 4, 5], "#FF8C00")],
    show_legend=False,
)
ax3c2.axvline(2.5, color="#cccccc", linewidth=1.0, linestyle="--", zorder=1)
ax3c2.text(-0.10, 1.06, "C", transform=ax3c2.transAxes,
           fontsize=15, fontweight="bold", va="top")

fig3.savefig(fr"{OUT_FIG}\Figure_3_Combined.png",
             bbox_inches="tight", dpi=300)
fig3.savefig(fr"{OUT_FIG}\Figure_3_Combined.pdf",
             bbox_inches="tight", dpi=300)
plt.close(fig3)
print("  Combined Figure 3 saved.")

print("\nAll done. Files in figure/:")
for f in sorted(os.listdir(fr"{BASE}\figure")):
    if "Figure_3" in f or "Supp" in f:
        sz = os.path.getsize(fr"{BASE}\figure\{f}")
        print(f"  {f}  ({sz:,} bytes)")
