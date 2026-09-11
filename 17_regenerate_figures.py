# =============================================================
#  POAG — Regenerate figures with participant-level bootstrap intervals
#  iScience R4 revision (2026-09-04)
#
#  Addresses Editor points 1 and 6 at the figure level. Every error bar
#  in the training-cohort panels previously showed
#  mean +/- 1.96 * SD/sqrt(100) over cross-validation folds, which is not
#  a valid interval. They are redrawn as participant-level bootstrap
#  percentile intervals.
#
#  No model is refitted: everything is computed from the 2,000 saved
#  bootstrap replicates written by script 10
#  (outputs/tables/bootstrap_replicates_training.csv.gz, 104,000 rows =
#  2,000 replicates x 4 classifiers x 13 feature sets).
#
#  Figures produced:
#    Figure_3A_All_FeatureSets_R4        12 feature sets x 4 classifiers
#    Figure_3B_KeyComparison_DotPlot_R4  key configurations
#    SF2B_PC_Confounding_R4              PC- vs PGS-augmented
#    SF7_DeltaAUC_Forest_R4              training panel of the forest plot
#
#  Figure 2C (standalone PGS) is redrawn by script 12, which holds those
#  replicates; Figure S5 (sex-stratified) by script 18.
# =============================================================

import os as _os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = _os.path.dirname(_os.path.abspath(__file__))
TBL = _os.path.join(HERE, "outputs", "tables")
FIG = _os.path.join(HERE, "outputs", "figures")
_os.makedirs(FIG, exist_ok=True)

RAW = _os.path.join(TBL, "bootstrap_replicates_training.csv.gz")
if not _os.path.exists(RAW):
    raise SystemExit("Run script 10 first (missing bootstrap replicates).")

MODEL_NAMES = ["LR", "RF", "MLP", "SVM"]
MODEL_COLORS = {"LR": "#4E79A7", "SVM": "#F28E2B",
                "RF": "#59A14F", "MLP": "#E15759"}

FS_ORDER = ["Age only", "Sex only", "Base",
            "Base+PC2", "Base+PC5", "Base+PC10", "Base+PC20",
            "Base+POAAGG PGS", "Base+MEGA PGS",
            "Base+PGS526", "Base+PGS616",
            "Base+PC5+PGS526", "Base+PC5+PGS616"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 300,
})

print("Loading bootstrap replicates ...", flush=True)
raw = pd.read_csv(RAW)
print(f"  {len(raw):,} rows, {raw.replicate.nunique()} replicates", flush=True)

# wide: index=(replicate), columns=(Classifier, FeatureSet)
W = raw.pivot_table(index="replicate", columns=["Classifier", "FeatureSet"],
                    values="AUC")


def ci(vals):
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def stats_for(m, fs):
    return ci(W[(m, fs)].values)


def xmean_for(fs):
    return ci(np.mean([W[(m, fs)].values for m in MODEL_NAMES], axis=0))


# ═══════════════════════════════════════════════════════════════
#  Figure 3A — all feature sets x classifiers + cross-classifier mean
# ═══════════════════════════════════════════════════════════════
print("Figure 3A ...", flush=True)
fig, ax = plt.subplots(figsize=(11, 5))
n_grp = len(FS_ORDER)
width = 0.16
xs = np.arange(n_grp)

for i, m in enumerate(MODEL_NAMES):
    means, los, his = [], [], []
    for fs in FS_ORDER:
        mu, lo, hi = stats_for(m, fs)
        means.append(mu); los.append(mu - lo); his.append(hi - mu)
    ax.bar(xs + (i - 1.5) * width, means, width * 0.9,
           label=m, color=MODEL_COLORS[m], alpha=0.9,
           yerr=[los, his], capsize=1.8,
           error_kw={"lw": 0.7, "alpha": 0.65})

xm = [xmean_for(fs)[0] for fs in FS_ORDER]
ax.plot(xs, xm, "k_", markersize=16, markeredgewidth=1.6,
        label="Cross-classifier mean", zorder=5)

ax.axhline(0.5, color="grey", ls="--", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(FS_ORDER, rotation=38, ha="right", fontsize=7.5)
ax.set_ylabel("Cross-validated AUC")
ax.set_ylim(0.40, 0.90)
ax.set_title("Training cohort (POAAGG, N = 271) — "
             "participant-level bootstrap 95% CI (B = 2,000)",
             fontsize=9.5, fontweight="bold")
ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(_os.path.join(FIG, f"Figure_3A_All_FeatureSets_R4.{ext}"),
                bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Figure 3B — key configurations, dot plot
# ═══════════════════════════════════════════════════════════════
print("Figure 3B ...", flush=True)
KEY = ["Base", "Base+PC5", "Base+PGS526", "Base+PGS616", "Base+PC5+PGS616"]
fig, ax = plt.subplots(figsize=(7.5, 4))
yy = np.arange(len(KEY))[::-1]
off = {m: (i - 1.5) * 0.14 for i, m in enumerate(MODEL_NAMES)}

for m in MODEL_NAMES:
    for k, fs in enumerate(KEY):
        mu, lo, hi = stats_for(m, fs)
        y = yy[k] + off[m]
        ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt="o",
                    color=MODEL_COLORS[m], markersize=4.5, capsize=2,
                    lw=1.1, label=m if k == 0 else None)
for k, fs in enumerate(KEY):
    mu, lo, hi = xmean_for(fs)
    ax.errorbar(mu, yy[k] + 0.30, xerr=[[mu - lo], [hi - mu]], fmt="D",
                color="black", markersize=4.5, capsize=2, lw=1.2,
                label="Cross-classifier mean" if k == 0 else None)

ax.set_yticks(yy); ax.set_yticklabels(KEY, fontsize=8.5)
ax.set_xlabel("Cross-validated AUC (participant-level bootstrap 95% CI)")
ax.axvline(0.5, color="grey", ls="--", lw=0.8)
ax.set_title("Key feature-set comparisons, training cohort",
             fontsize=9.5, fontweight="bold")
ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(_os.path.join(FIG, f"Figure_3B_KeyComparison_DotPlot_R4.{ext}"),
                bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Figure S2B — PC-augmented vs PGS-augmented
# ═══════════════════════════════════════════════════════════════
print("Figure S2B ...", flush=True)
SET = ["Base", "Base+PC2", "Base+PC5", "Base+PC10", "Base+PC20",
       "Base+PGS526", "Base+PGS616"]
fig, ax = plt.subplots(figsize=(7.5, 4))
cols = ["#888888"] + ["#4E79A7"] * 4 + ["#E15759"] * 2
for i, fs in enumerate(SET):
    mu, lo, hi = xmean_for(fs)
    ax.bar(i, mu, 0.62, color=cols[i], alpha=0.9,
           yerr=[[mu - lo], [hi - mu]], capsize=3,
           error_kw={"lw": 0.9})
    ax.text(i, hi + 0.006, f"{mu:.3f}", ha="center", fontsize=7.5)
ax.axhline(xmean_for("Base")[0], color="grey", ls=":", lw=0.9)
ax.set_xticks(range(len(SET)))
ax.set_xticklabels(SET, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Cross-classifier mean AUC")
ax.set_ylim(0.40, 0.85)
ax.set_title("Ancestry PC- versus PGS-augmented models\n"
             "participant-level bootstrap 95% CI",
             fontsize=9.5, fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(_os.path.join(FIG, f"SF2B_PC_Confounding_R4.{ext}"),
                bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Figure S7 (training panel) — paired incremental delta-AUC forest
# ═══════════════════════════════════════════════════════════════
print("Figure S7 (training panel) ...", flush=True)
fig, ax = plt.subplots(figsize=(6, 3.6))
order = ["LR", "SVM", "RF", "MLP"]
yy = np.arange(len(order))[::-1]
for k, m in enumerate(order):
    d = W[(m, "Base+PGS616")].values - W[(m, "Base")].values
    mu, lo, hi = ci(d)
    ax.errorbar(mu, yy[k], xerr=[[mu - lo], [hi - mu]], fmt="o",
                color="#E15759", capsize=3, markersize=6, lw=1.3)
    ax.text(hi + 0.004, yy[k], f"{mu:+.3f}", va="center", fontsize=7.5)
ax.axvline(0, color="grey", ls="--", lw=1)
ax.set_yticks(yy); ax.set_yticklabels(order)
ax.set_xlabel("ΔAUC (Base+PGS616 − Base)")
ax.set_title("Training (POAAGG, N = 271)\n"
             "paired participant-level bootstrap, B = 2,000",
             fontsize=9, fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(_os.path.join(FIG, f"SF7_DeltaAUC_Forest_training_R4.{ext}"),
                bbox_inches="tight")
plt.close(fig)

print("\nSaved 4 figures to outputs/figures/", flush=True)
print("Still required: Figure 2C (script 12) and Figure S5 (script 18).",
      flush=True)


# ═══════════════════════════════════════════════════════════════
#  Figure 2C — standalone PGS discrimination
#  Built from script 12's summary table (which holds those replicates).
# ═══════════════════════════════════════════════════════════════
PGSC = _os.path.join(TBL, "Table_PGS_Standalone_Comparison.xlsx")
if _os.path.exists(PGSC):
    print("Figure 2C ...", flush=True)
    sa = pd.read_excel(PGSC, sheet_name="A_StandaloneAUC")

    def parse_ci(s):
        mu = float(s.split(" (")[0])
        lo, hi = s.split("(")[1].rstrip(")").split("-")
        return mu, float(lo), float(hi)

    PGS_ORDER = ["PGS526", "PGS616", "MEGA PGS", "POAAGG PGS"]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(PGS_ORDER))
    for i, m in enumerate(MODEL_NAMES):
        mus, los, his = [], [], []
        for p in PGS_ORDER:
            row = sa[(sa.PGS == p) & (sa.Classifier == m)].iloc[0]
            mu, lo, hi = parse_ci(row["95% CI"])
            mus.append(mu); los.append(mu - lo); his.append(hi - mu)
        ax.bar(xs + (i - 1.5) * 0.19, mus, 0.17, label=m,
               color=MODEL_COLORS[m], alpha=0.9,
               yerr=[los, his], capsize=2, error_kw={"lw": 0.8})
    xm, xlo, xhi = [], [], []
    for p in PGS_ORDER:
        row = sa[(sa.PGS == p) &
                 (sa.Classifier == "Cross-classifier mean")].iloc[0]
        mu, lo, hi = parse_ci(row["95% CI"])
        xm.append(mu); xlo.append(mu - lo); xhi.append(hi - mu)
    ax.plot(xs, xm, "k_", markersize=18, markeredgewidth=1.8,
            label="Cross-classifier mean", zorder=5)

    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.text(len(PGS_ORDER) - 0.45, 0.503, "chance", fontsize=7.5,
            color="grey", va="bottom", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p}\n({'curated' if p.startswith('PGS') else 'genome-wide'})"
                        for p in PGS_ORDER], fontsize=8.5)
    ax.set_ylabel("Standalone AUC")
    ax.set_ylim(0.35, 0.72)
    ax.set_title("Standalone PGS discrimination, training cohort\n"
                 "participant-level bootstrap 95% CI — every interval "
                 "includes chance",
                 fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(_os.path.join(FIG, f"Figure_2C_Training_AUC_R4.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  saved Figure_2C_Training_AUC_R4", flush=True)
else:
    print("Figure 2C skipped (run script 12 first).", flush=True)
