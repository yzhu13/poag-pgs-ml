# =============================================================
#  Rebuild Figure 5 — Panel D replaced with Base+PGS616 model
#  5A  – user-supplied eyeball illustration
#  5B  – MLP training AUC dot plot (5×20 CV)
#  5C  – PMBB external AUC dot plot
#  5D  – Suspect quintile line plots (ΔIOP, ΔCDR)
#         NOW uses Base+PGS616 MLP risk scores (age+sex+PGS616)
#         instead of the old circular ΔIOP+ΔCDR+PGS616 model
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

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
DATA   = r"C:\Users\biqiz\Dropbox\3_Penn_Postdoc\0_Projects_Ongoing\1_MLP\2026-4-29-Revision-to-iScience\01_input_data\POAAGG_cohort"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# ------------------------------------------------------------------
# Load panel B/C tables (unchanged)
# ------------------------------------------------------------------
cv_sep   = pd.read_excel(XL, sheet_name="CV_AUC_Sep_Delta")
pmbb_sep = pd.read_excel(XL, sheet_name="PMBB_AUC_Sep_Delta")

PGS_SLOTS     = ["PHE", "POAAGG", "MEGA", "PGS526", "PGS616"]
PGS_LABELS    = ["PHE\nOnly", "PHE+\nPOAAGG", "PHE+\nMEGA", "PHE+\nPGS526", "PHE+\nPGS616"]
PGS_PM_SLOTS  = ["PHE", "PGS526", "PGS616"]
PGS_PM_LABELS = ["PHE\nOnly", "PHE+\nPGS526", "PHE+\nPGS616"]
PGS_COLORS    = {"PHE": "#888888", "POAAGG": "#7B9EC9", "MEGA": "#E07B54",
                 "PGS526": "#5BAD72", "PGS616": "#2196A6"}

def parse_slot(fs):
    if "POAAGG" in fs: return "POAAGG"
    if "MEGA"   in fs: return "MEGA"
    if "PGS526" in fs: return "PGS526"
    if "PGS616" in fs: return "PGS616"
    return "PHE"

DELTA_DOT = {
    "IOP": {"color": "#C0392B", "marker": "o", "label": "ΔIOP"},
    "CDR": {"color": "#2471A3", "marker": "s", "label": "ΔCDR"},
}
QUINT_X      = np.array([1, 2, 3, 4, 5])
QUINT_LABELS = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]


# ------------------------------------------------------------------
# Compute Base+PGS616 MLP quintile data for Panel 5D
# ------------------------------------------------------------------
print("Computing Base+PGS616 MLP risk scores for suspects ...")

train = pd.read_excel(fr"{DATA}\271_training_cohort_4_new_PRS_cleaned.xlsx")
susp  = pd.read_excel(fr"{DATA}\1013_testing_cohort_only_suspect_cleaned.xlsx")

# Train on Base+PGS616 (Age, Gender, PRS616)
y      = train["CaseCtrl"].values
X_tr   = train[["Age", "Gender", "PRS616"]].values
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(imputer.fit_transform(X_tr))
mlp = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam",
                    alpha=0.01, random_state=42, max_iter=500)
mlp.fit(X_tr_s, y)

# Apply to suspects (MEGA PRS in suspects = PRS616 in training, same raw log-odds scale)
X_su  = susp[["Age", "Gender", "MEGA PRS"]].values
X_su_s = scaler.transform(imputer.transform(X_su))
risk   = mlp.predict_proba(X_su_s)[:, 1]

susp = susp.copy()
susp["risk_score"] = risk
susp["delta_IOP"]  = (susp["OD Baseline IOP (mmHg)"] - susp["OS Baseline IOP (mmHg)"]).abs()
susp["delta_CDR"]  = (susp["OD Baseline CDR"]         - susp["OS Baseline CDR"]).abs()

# Assign quintiles based on risk score
susp["quintile"] = pd.qcut(susp["risk_score"], q=5,
                            labels=["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"])

# Pearson correlations (individual level)
mask_iop = susp["delta_IOP"].notna()
mask_cdr = susp["delta_CDR"].notna()
r_iop, p_iop = stats.pearsonr(susp.loc[mask_iop, "risk_score"],
                               susp.loc[mask_iop, "delta_IOP"])
r_cdr, p_cdr = stats.pearsonr(susp.loc[mask_cdr, "risk_score"],
                               susp.loc[mask_cdr, "delta_CDR"])
n_iop = mask_iop.sum()
n_cdr = mask_cdr.sum()
print(f"  ΔIOP: r={r_iop:.3f}, p={p_iop:.3f}, N={n_iop}")
print(f"  ΔCDR: r={r_cdr:.3f}, p={p_cdr:.3f}, N={n_cdr}")

# Quintile means and SEMs
def quintile_stats(df, outcome_col, quint_col="quintile"):
    rows = []
    for q in ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]:
        sub = df[df[quint_col] == q][outcome_col].dropna()
        rows.append({"Quintile": q, "N": len(sub),
                     "Mean": sub.mean(), "SEM": sub.sem()})
    return pd.DataFrame(rows)

quint_iop = quintile_stats(susp, "delta_IOP")
quint_cdr = quintile_stats(susp, "delta_CDR")
print("\nQuintile ΔIOP stats:")
print(quint_iop.to_string(index=False))
print("\nQuintile ΔCDR stats:")
print(quint_cdr.to_string(index=False))


# ------------------------------------------------------------------
# Panel builders
# ------------------------------------------------------------------
def _dot_panel(ax, cv_data, pmbb_data, slots, slot_labels,
               cv_col, pm_col, title, xlim, ref_line=None, legend_loc="upper left"):
    n      = len(slots)
    offset = 0.12
    for yi, slot in enumerate(slots):
        for di, (grp, dstyle) in enumerate(DELTA_DOT.items()):
            df  = cv_data if cv_col == "Mean_AUC" else pmbb_data
            row = df[(df["DeltaGroup"] == grp) &
                     (df["FeatureSet"].apply(parse_slot) == slot)]
            if len(row) == 0:
                continue
            row  = row.iloc[0]
            mn   = float(row[cv_col])
            lo   = float(row["CI_lo_95"])
            hi   = float(row["CI_hi_95"])
            ypos = yi + (0.5 - di) * offset
            ax.errorbar(mn, ypos, xerr=[[mn - lo], [hi - mn]],
                        fmt="none", color=dstyle["color"],
                        capsize=3.5, linewidth=1.2, zorder=3)
            ax.plot(mn, ypos, marker=dstyle["marker"],
                    color=dstyle["color"], markersize=8, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=dstyle["label"] if yi == 0 else "_")
            ax.text(hi + (xlim[1] - xlim[0]) * 0.012, ypos,
                    f"{mn:.3f}", va="center", ha="left",
                    fontsize=8, color=dstyle["color"])

    if ref_line is not None:
        ax.axvline(ref_line, color="#aaaaaa", linewidth=0.9, linestyle=":", zorder=1)
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
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9, loc=legend_loc)


def build_5B(ax, title="Training Cohort (N=271) — MLP, 5×20 CV\nΔIOP and ΔCDR as Outcomes"):
    mlp = cv_sep[cv_sep["Model"] == "MLP"].copy()
    _dot_panel(ax, mlp, None, PGS_SLOTS, PGS_LABELS, "Mean_AUC", None,
               title, xlim=(0.72, 0.88), legend_loc="lower right")


def build_5C(ax, title="PMBB External Validation — MLP\nΔIOP and ΔCDR as Outcomes"):
    mlp = pmbb_sep[pmbb_sep["Model"] == "MLP"].copy()
    _dot_panel(ax, mlp, mlp, PGS_PM_SLOTS, PGS_PM_LABELS, "AUC", "AUC",
               title, xlim=(0.49, 0.78), ref_line=0.5, legend_loc="lower right")


def build_5D(axes_pair):
    """Panel D: quintile line plots using BASE+PGS616 MLP risk scores."""
    configs = [
        ("ΔIOP (mmHg)", quint_iop, "#E05C5C", r_iop, p_iop, n_iop),
        ("ΔCDR",        quint_cdr, "#5B8DB8", r_cdr, p_cdr, n_cdr),
    ]
    for ax, (out_label, qdf, col_color, r_val, p_val, n_val) in zip(axes_pair, configs):
        means = qdf["Mean"].values
        sems  = qdf["SEM"].values
        ns    = qdf["N"].values

        ax.errorbar(QUINT_X, means, yerr=sems,
                    fmt="o-", color=col_color,
                    markersize=7, linewidth=1.8,
                    capsize=4, capthick=1.2, elinewidth=1.0, zorder=4)

        # OLS trend line
        slope, intercept, _, _, _ = stats.linregress(QUINT_X, means)
        x_fit = np.linspace(0.6, 5.4, 100)
        ax.plot(x_fit, slope * x_fit + intercept,
                linestyle="--", color="#D2691E", linewidth=1.4, zorder=3)

        # n labels above each point
        y_range = means.max() - means.min() if means.max() != means.min() else 1e-6
        for xi, m, s, n in zip(QUINT_X, means, sems, ns):
            ax.text(xi, m + s + y_range * 0.06,
                    f"n={n}", ha="center", va="bottom",
                    fontsize=6, color="#666666")

        # Significance label
        if p_val < 0.001:
            sig_str = "***"
            p_str   = "p < 0.001"
        elif p_val < 0.01:
            sig_str = "**"
            p_str   = f"p = {p_val:.3f}"
        elif p_val < 0.05:
            sig_str = "*"
            p_str   = f"p = {p_val:.3f}"
        else:
            sig_str = "ns"
            p_str   = f"p = {p_val:.3f}"

        ax.text(0.97, 0.97,
                f"r = {r_val:.3f} ({sig_str})\n{p_str}\nN = {n_val}",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#cccccc", alpha=0.88))

        y_lo = means.min() - sems.max() * 3 - y_range * 0.25
        y_hi = means.max() + sems.max() * 3 + y_range * 0.80
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(QUINT_X)
        ax.set_xticklabels(QUINT_LABELS, fontsize=8.5)
        ax.set_xlim(0.5, 5.5)
        ax.set_xlabel("Predicted POAG Risk Quintile\n(Base+PGS616, MLP)", fontsize=9)
        ax.set_ylabel(out_label, fontsize=9)

        title_map = {
            "ΔIOP (mmHg)": "Base+PGS616 Risk vs ΔIOP\n(Suspects, N=1,013)",
            "ΔCDR":        "Base+PGS616 Risk vs ΔCDR\n(Suspects, N=1,013)",
        }
        ax.set_title(title_map[out_label], fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)


# ------------------------------------------------------------------
# Save individual panels
# ------------------------------------------------------------------
fig5D, axes5D = plt.subplots(1, 2, figsize=(10, 4.5))
fig5D.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.20, wspace=0.38)
build_5D(axes5D)
fig5D.savefig(fr"{OUT}\Figure_5D_Suspect_Quintile.png", bbox_inches="tight", dpi=300)
fig5D.savefig(fr"{OUT}\Figure_5D_Suspect_Quintile.pdf", bbox_inches="tight", dpi=300)
plt.close(fig5D)
print("\nFigure 5D (Base+PGS616) saved.")

# ------------------------------------------------------------------
# Combined Figure 5
# ------------------------------------------------------------------
print("Building combined Figure 5 ...")

fig5 = plt.figure(figsize=(18, 10))
gs   = gridspec.GridSpec(2, 12, figure=fig5,
                          hspace=0.60, wspace=0.55,
                          left=0.05, right=0.98,
                          top=0.93, bottom=0.09)

# 5A
ax5a = fig5.add_subplot(gs[0, 0:4])
try:
    img = mpimg.imread(FIG5A)
    ax5a.imshow(img)
except Exception:
    ax5a.text(0.5, 0.5, "Figure 5A\n(schematic)", ha="center", va="center",
              fontsize=11, transform=ax5a.transAxes)
ax5a.axis("off")
ax5a.text(-0.05, 1.06, "A", transform=ax5a.transAxes,
          fontsize=14, fontweight="bold", va="top")

# 5B
ax5b = fig5.add_subplot(gs[0, 4:8])
build_5B(ax5b)
ax5b.text(-0.22, 1.10, "B", transform=ax5b.transAxes,
          fontsize=14, fontweight="bold", va="top")

# 5C
ax5c = fig5.add_subplot(gs[0, 9:12])
build_5C(ax5c)
ax5c.text(-0.22, 1.10, "C", transform=ax5c.transAxes,
          fontsize=14, fontweight="bold", va="top")

# 5D
ax5d_iop = fig5.add_subplot(gs[1, 0:4])
ax5d_cdr = fig5.add_subplot(gs[1, 4:8])
build_5D([ax5d_iop, ax5d_cdr])
ax5d_iop.text(-0.16, 1.12, "D", transform=ax5d_iop.transAxes,
              fontsize=14, fontweight="bold", va="top")

fig5.savefig(fr"{OUT}\Figure_5_Combined.png", bbox_inches="tight", dpi=300)
fig5.savefig(fr"{OUT}\Figure_5_Combined.pdf", bbox_inches="tight", dpi=300)
plt.close(fig5)
print("Combined Figure 5 saved.")

# Also copy to main Figures folder for easy access
import shutil
shutil.copy(fr"{OUT}\Figure_5_Combined.png",
            fr"{BASE}\5-18-revision-for-submission\Figures\Figure_5_Combined.png")
print("Copied to main Figures folder.")
print("\nAll done.")
