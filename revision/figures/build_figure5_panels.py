# =============================================================
#  Build Figure 5  (loads from Table_Figure5_Asymmetry.xlsx)
#  5A  – user-supplied eyeball illustration (loaded as image)
#  5B  – MLP training AUC bar chart (5 feature sets, 5x20 CV)
#  5C  – PMBB external AUC bar chart (Delta / +PGS526 / +PGS616)
#  5D  – Suspect quintile line plots (ΔIOP, ΔCDR)
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
import matplotlib.image as mpimg
from scipy import stats

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
XL     = fr"{BASE}\output excel data\Table_Figure5_Asymmetry.xlsx"
FIG5A  = fr"{BASE}\5-18-revision-for-submission\Figures\figure\Figure 5\Figure_5A.png"
OUT    = fr"{BASE}\5-18-revision-for-submission\Figures\figure\Figure 5"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# ------------------------------------------------------------------
# Load tables
# ------------------------------------------------------------------
quint_df  = pd.read_excel(XL, sheet_name="Quintile_MLP_Delta_PGS616")
ks_df     = pd.read_excel(XL, sheet_name="KS_LowVsHigh_25pct")
cv_sep    = pd.read_excel(XL, sheet_name="CV_AUC_Sep_Delta")
pmbb_sep  = pd.read_excel(XL, sheet_name="PMBB_AUC_Sep_Delta")

# PGS slot order and display labels (x-axis within each delta group)
PGS_SLOTS  = ["PHE", "POAAGG", "MEGA", "PGS526", "PGS616"]
PGS_LABELS = ["PHE\nOnly", "PHE+\nPOAAGG", "PHE+\nMEGA", "PHE+\nPGS526", "PHE+\nPGS616"]
PGS_PM_SLOTS  = ["PHE", "PGS526", "PGS616"]   # PMBB only has these
PGS_PM_LABELS = ["PHE\nOnly", "PHE+\nPGS526", "PHE+\nPGS616"]

PGS_COLORS = {
    "PHE":    "#888888",
    "POAAGG": "#7B9EC9",
    "MEGA":   "#E07B54",
    "PGS526": "#5BAD72",
    "PGS616": "#2196A6",
}

# Map FeatureSet name → (DeltaGroup, PGS slot)
def parse_slot(fs):
    if "POAAGG" in fs: return "POAAGG"
    if "MEGA"   in fs: return "MEGA"
    if "PGS526" in fs: return "PGS526"
    if "PGS616" in fs: return "PGS616"
    return "PHE"

DELTA_COLORS = {"ΔIOP (mmHg)": "#E05C5C", "ΔCDR": "#5B8DB8"}
QUINT_X      = np.array([1, 2, 3, 4, 5])
QUINT_LABELS = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]


# ==================================================================
# Panel 5B — two subplots: ΔIOP (left) and ΔCDR (right)
#            5 bars each: PHE Only, PHE+POAAGG, PHE+MEGA, +PGS526, +PGS616
# ==================================================================
DELTA_DOT = {
    "IOP": {"color": "#C0392B", "marker": "o", "label": "ΔIOP"},
    "CDR": {"color": "#2471A3", "marker": "s", "label": "ΔCDR"},
}

def _dot_panel(ax, cv_data, pmbb_data,
               slots, slot_labels,
               cv_col, pm_col,
               title, xlim, ref_line=None, legend_loc="upper left"):
    """
    Horizontal dot/forest plot.
    y-axis  = feature sets (rows, bottom→top)
    x-axis  = AUC
    series  = ΔIOP (red circle) and ΔCDR (blue square)
    cv_data / pmbb_data : already filtered to MLP
    cv_col  : column name for AUC value in cv_data
    pm_col  : column name for AUC value in pmbb_data (or None)
    """
    n   = len(slots)
    offset = 0.12     # vertical offset between the two dot series

    for yi, slot in enumerate(slots):
        for di, (grp, dstyle) in enumerate(DELTA_DOT.items()):
            df   = cv_data if pm_col is None else (
                   cv_data if cv_col != "AUC" else pmbb_data)
            # pick correct dataframe
            df   = cv_data if cv_col == "Mean_AUC" else pmbb_data
            row  = df[(df["DeltaGroup"] == grp) &
                      (df["FeatureSet"].apply(parse_slot) == slot)]
            if len(row) == 0:
                continue
            row  = row.iloc[0]
            mn   = float(row[cv_col])
            lo   = float(row["CI_lo_95"])
            hi   = float(row["CI_hi_95"])
            ypos = yi + (0.5 - di) * offset

            ax.errorbar(mn, ypos,
                        xerr=[[mn - lo], [hi - mn]],
                        fmt="none",
                        color=dstyle["color"],
                        capsize=3.5, linewidth=1.2, zorder=3)
            ax.plot(mn, ypos,
                    marker=dstyle["marker"],
                    color=dstyle["color"],
                    markersize=8, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=dstyle["label"] if yi == 0 else "_")
            ax.text(hi + (xlim[1] - xlim[0]) * 0.012, ypos,
                    f"{mn:.3f}", va="center", ha="left",
                    fontsize=8, color=dstyle["color"])

    if ref_line is not None:
        ax.axvline(ref_line, color="#aaaaaa", linewidth=0.9,
                   linestyle=":", zorder=1)
        ax.text(ref_line + (xlim[1]-xlim[0])*0.01, -0.5,
                "chance", fontsize=7, color="#aaaaaa", va="bottom")

    ax.set_yticks(range(n))
    ax.set_yticklabels(slot_labels, fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("AUC (mean ± 95% CI)", fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False)

    handles = [plt.Line2D([0], [0], marker=DELTA_DOT[g]["marker"],
                           color=DELTA_DOT[g]["color"],
                           markersize=7, linestyle="",
                           markeredgecolor="white", markeredgewidth=0.5,
                           label=DELTA_DOT[g]["label"])
               for g in DELTA_DOT]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9,
              loc=legend_loc)


def build_5B(ax, title="Training Cohort (N=271) — MLP, 5×20 CV\nΔIOP and ΔCDR as Outcomes"):
    mlp = cv_sep[cv_sep["Model"] == "MLP"].copy()
    _dot_panel(ax, mlp, None,
               PGS_SLOTS, PGS_LABELS,
               "Mean_AUC", None,
               title,
               xlim=(0.72, 0.88), legend_loc="lower right")


def build_5C(ax, title="PMBB External Validation — MLP\nΔIOP and ΔCDR as Outcomes"):
    mlp = pmbb_sep[pmbb_sep["Model"] == "MLP"].copy()
    _dot_panel(ax, mlp, mlp,
               PGS_PM_SLOTS, PGS_PM_LABELS,
               "AUC", "AUC",
               title,
               xlim=(0.49, 0.78), ref_line=0.5, legend_loc="lower right")


# ==================================================================
# Panel 5D  — Suspect quintile line plots (ΔIOP, ΔCDR)
# ==================================================================
def build_5D(axes_pair):
    for ax, (out_label, col_color) in zip(axes_pair,
            [("ΔIOP (mmHg)", "#E05C5C"), ("ΔCDR", "#5B8DB8")]):
        sub   = quint_df[quint_df["Outcome"] == out_label]
        means = sub["Mean"].values
        sems  = sub["SEM"].values
        ns    = sub["N"].values

        ax.errorbar(QUINT_X, means, yerr=sems,
                    fmt="o-", color=col_color,
                    markersize=7, linewidth=1.8,
                    capsize=4, capthick=1.2, elinewidth=1.0, zorder=4)

        # OLS regression line
        slope, intercept, _, _, _ = stats.linregress(QUINT_X, means)
        x_fit = np.linspace(0.6, 5.4, 100)
        ax.plot(x_fit, slope * x_fit + intercept,
                linestyle="--", color="#D2691E", linewidth=1.4, zorder=3)

        # n labels
        y_range = means.max() - means.min()
        for xi, m, s, n in zip(QUINT_X, means, sems, ns):
            ax.text(xi, m + s + y_range * 0.06,
                    f"n={n}", ha="center", va="bottom",
                    fontsize=6, color="#666666")

        # Pearson r from KS table — load from corr sheet instead
        # (use linregress slope as proxy annotation)
        corr_df2 = pd.read_excel(XL, sheet_name="Suspect_Correlations")
        row = corr_df2[(corr_df2["FeatureSet"] == "Delta+PGS616") &
                       (corr_df2["Model"] == "MLP") &
                       (corr_df2["Outcome"] == out_label)]
        if len(row):
            r_val = row["Pearson_r"].values[0]
            p_val = row["p_value"].values[0]
            sig   = row["Sig"].values[0]
            n_val = row["N"].values[0]
            ax.text(0.97, 0.97,
                    f"r = {r_val:.3f}{sig}\nβ = {slope:.4f}\nN = {n_val}",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="#cccccc",
                              alpha=0.88))

        y_lo = means.min() - sems.max() * 3 - y_range * 0.25
        y_hi = means.max() + sems.max() * 3 + y_range * 0.65
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(QUINT_X)
        ax.set_xticklabels(QUINT_LABELS, fontsize=8.5)
        ax.set_xlim(0.5, 5.5)
        ax.set_xlabel("Predicted POAG Risk Quintile (MLP)", fontsize=9)
        ax.set_ylabel(out_label, fontsize=9)
        title_map = {
            "ΔIOP (mmHg)": "Predicted POAG Risk vs ΔIOP\n(Suspects, MLP Quintile)",
            "ΔCDR":        "Predicted POAG Risk vs ΔCDR\n(Suspects, MLP Quintile)",
        }
        ax.set_title(title_map.get(out_label, out_label),
                     fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)


# ==================================================================
# Build individual panels
# ==================================================================
# 5B — single dot-plot panel
fig5B, ax5B = plt.subplots(figsize=(7, 4.5))
fig5B.subplots_adjust(left=0.18, right=0.88, top=0.88, bottom=0.14)
build_5B(ax5B)
fig5B.savefig(fr"{OUT}\Figure_5B_Training_AUC.png", bbox_inches="tight", dpi=300)
fig5B.savefig(fr"{OUT}\Figure_5B_Training_AUC.pdf", bbox_inches="tight", dpi=300)
plt.close(fig5B)
print("Figure 5B saved.")

# 5C — single dot-plot panel
fig5C, ax5C = plt.subplots(figsize=(6, 3.5))
fig5C.subplots_adjust(left=0.18, right=0.88, top=0.82, bottom=0.16)
build_5C(ax5C)
fig5C.savefig(fr"{OUT}\Figure_5C_PMBB_External.png", bbox_inches="tight", dpi=300)
fig5C.savefig(fr"{OUT}\Figure_5C_PMBB_External.pdf", bbox_inches="tight", dpi=300)
plt.close(fig5C)
print("Figure 5C saved.")

# 5D
fig5D, axes5D = plt.subplots(1, 2, figsize=(10, 4.5))
fig5D.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.20, wspace=0.38)
build_5D(axes5D)
# suptitle removed per journal style
fig5D.savefig(fr"{OUT}\Figure_5D_Suspect_Quintile.png", bbox_inches="tight", dpi=300)
fig5D.savefig(fr"{OUT}\Figure_5D_Suspect_Quintile.pdf", bbox_inches="tight", dpi=300)
plt.close(fig5D)
print("Figure 5D saved.")


# ==================================================================
# Combined Figure 5
# Layout (12 cols):
#   Row 0: A (cols 0-3) | B-IOP (cols 4-7) | B-CDR (cols 8-11)
#   Row 1: C-IOP (0-2) | C-CDR (3-5) | D-IOP (6-8) | D-CDR (9-11)
# ==================================================================
print("Building combined Figure 5 ...")

fig5  = plt.figure(figsize=(18, 10))
gs    = gridspec.GridSpec(2, 12, figure=fig5,
                           hspace=0.60, wspace=0.55,
                           left=0.05, right=0.98,
                           top=0.93, bottom=0.09)

# --- 5A ---
ax5a = fig5.add_subplot(gs[0, 0:4])
try:
    img = mpimg.imread(FIG5A)
    ax5a.imshow(img)
except Exception:
    ax5a.text(0.5, 0.5, "Figure 5A", ha="center", va="center",
              fontsize=11, transform=ax5a.transAxes)
ax5a.axis("off")
ax5a.text(-0.05, 1.06, "A", transform=ax5a.transAxes,
          fontsize=14, fontweight="bold", va="top")

# --- 5B: dot plot (cols 4-7) ---
ax5b = fig5.add_subplot(gs[0, 4:8])
build_5B(ax5b)
ax5b.text(-0.22, 1.10, "B", transform=ax5b.transAxes,
          fontsize=14, fontweight="bold", va="top")

# --- 5C: dot plot (cols 9-11, col 8 left as gap from 5B) ---
ax5c = fig5.add_subplot(gs[0, 9:12])
build_5C(ax5c)
ax5c.text(-0.22, 1.10, "C", transform=ax5c.transAxes,
          fontsize=14, fontweight="bold", va="top")

# --- 5D: quintile line plots (cols 0-5 bottom row) ---
ax5d_iop = fig5.add_subplot(gs[1, 0:4])
ax5d_cdr = fig5.add_subplot(gs[1, 4:8])
build_5D([ax5d_iop, ax5d_cdr])
ax5d_iop.text(-0.16, 1.12, "D", transform=ax5d_iop.transAxes,
              fontsize=14, fontweight="bold", va="top")

# suptitle removed per journal style

fig5.savefig(fr"{OUT}\Figure_5_Combined.png", bbox_inches="tight", dpi=300)
fig5.savefig(fr"{OUT}\Figure_5_Combined.pdf", bbox_inches="tight", dpi=300)
plt.close(fig5)
print("Combined Figure 5 saved.")
print("\nAll done.")
