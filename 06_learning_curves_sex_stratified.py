# =============================================================
#  SF4: Learning Curves  (E3 / R1-C1)
#  SF5: Sex-Stratified AUC  (R1-C8)
#
#  Both analyses use the POAAGG training cohort (N=271, 5×20 CV)
#  Exports:
#    output excel data/Table_SF4_LearningCurves.xlsx
#    output excel data/Table_SF5_SexStratified.xlsx
#  Figures:
#    figure/Figure 3 and SF2/SF4_LearningCurves.png/.pdf
#    figure/Figure 3 and SF2/SF5_SexStratified.png/.pdf
# =============================================================
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedKFold, learning_curve)
from sklearn.metrics import roc_auc_score

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

LABEL   = "CaseCtrl"
RNG     = 42

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

MODEL_COLORS = {
    "LR": "#4E79A7", "SVM": "#F28E2B",
    "RF": "#59A14F", "MLP": "#E15759", "Average": "#B07AA1",
}

# ── Feature sets ──────────────────────────────────────────────
BASE_COLS  = ["Age", "Gender"]
PC5_COLS   = BASE_COLS + [f"PC{i}" for i in range(1, 6)]
PGS616_COLS = BASE_COLS + ["PGS616"]
PC5PGS_COLS = PC5_COLS + ["PGS616"]

FEATURE_SETS = {
    "Base":            BASE_COLS,
    "Base+PGS616":     PGS616_COLS,
    "Base+PC5+PGS616": PC5PGS_COLS,
}

FS_COLORS = {
    "Base":            "#888888",
    "Base+PGS616":     "#2196A6",
    "Base+PC5+PGS616": "#C44E52",
}
FS_STYLES = {
    "Base":            "-",
    "Base+PGS616":     "--",
    "Base+PC5+PGS616": "-.",
}

MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]


# ── Pipeline factory ──────────────────────────────────────────
def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RNG))]
    elif name == "SVM":
        steps += [("clf", SVC(
            kernel="rbf", probability=True,
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


def ci95(vals):
    n, mn, sd = len(vals), np.mean(vals), np.std(vals, ddof=1)
    se = sd / np.sqrt(n)
    return mn, mn - 1.96 * se, mn + 1.96 * se


# ── Load data ─────────────────────────────────────────────────
print("Loading training data ...")
tr   = pd.read_excel(TRAIN_F)
y_tr = tr[LABEL].values.astype(int)
print(f"  N={len(tr)}, cases={y_tr.sum()}, controls={(y_tr==0).sum()}")

sex_col = "Gender"   # 1=Male, 0=Female (or check actual coding)
sex_vals = tr[sex_col].values
print(f"  Sex distribution: {pd.Series(sex_vals).value_counts().to_dict()}")

# ============================================================
#  SECTION 1: LEARNING CURVES  (SF4)
# ============================================================
print("\n" + "="*60)
print("SECTION 1: Learning Curves (SF4)")
print("="*60)

# Use sklearn learning_curve with 5-fold × 4-repeat CV (20 fits per size)
# train_sizes = fractions of the training set within each CV fold
TRAIN_FRACS = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
LC_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=RNG)
# n_splits=5, n_repeats=4 → 20 fits per configuration (same total as 5×20 but faster for LC)

lc_rows = []

for fs_name, cols in FEATURE_SETS.items():
    missing = [c for c in cols if c not in tr.columns]
    if missing:
        print(f"  SKIP {fs_name} — missing: {missing}")
        continue
    X = tr[cols].values
    print(f"\n  {fs_name} ({len(cols)} features):")

    all_train_scores = {frac: [] for frac in TRAIN_FRACS}
    all_test_scores  = {frac: [] for frac in TRAIN_FRACS}
    per_model_test   = {mn: {frac: [] for frac in TRAIN_FRACS} for mn in MODEL_NAMES}

    for mn in MODEL_NAMES:
        pipe = make_pipeline(mn)
        train_sizes_abs, tr_sc, te_sc = learning_curve(
            pipe, X, y_tr,
            train_sizes=TRAIN_FRACS,
            cv=LC_CV,
            scoring="roc_auc",
            n_jobs=1,
            return_times=False,
        )
        # tr_sc, te_sc shape: (n_sizes, n_splits)
        for fi, frac in enumerate(TRAIN_FRACS):
            all_train_scores[frac].extend(tr_sc[fi].tolist())
            all_test_scores[frac].extend(te_sc[fi].tolist())
            per_model_test[mn][frac].extend(te_sc[fi].tolist())

        mn_tr = tr_sc.mean(axis=1)
        mn_te = te_sc.mean(axis=1)
        print(f"    {mn}: train AUC={mn_tr[-1]:.3f}  CV AUC={mn_te[-1]:.3f}")

    # Compute absolute N at each fraction (fraction of 80% of N)
    n_train_total = int(len(tr) * 0.80)   # approximate per-fold training N

    for fi, frac in enumerate(TRAIN_FRACS):
        n_abs = int(round(n_train_total * frac))
        # All-model average
        mn_tr, lo_tr, hi_tr = ci95(all_train_scores[frac])
        mn_te, lo_te, hi_te = ci95(all_test_scores[frac])
        lc_rows.append({
            "FeatureSet": fs_name, "Model": "Average",
            "TrainFrac": round(frac, 2), "N_train_approx": n_abs,
            "Train_AUC": round(mn_tr, 4),
            "Train_CI_lo": round(lo_tr, 4), "Train_CI_hi": round(hi_tr, 4),
            "CV_AUC": round(mn_te, 4),
            "CV_CI_lo": round(lo_te, 4), "CV_CI_hi": round(hi_te, 4),
        })
        # Per-model rows
        for mn in MODEL_NAMES:
            mn_te_m, lo_te_m, hi_te_m = ci95(per_model_test[mn][frac])
            lc_rows.append({
                "FeatureSet": fs_name, "Model": mn,
                "TrainFrac": round(frac, 2), "N_train_approx": n_abs,
                "Train_AUC": None,  # not tracked per model separately here
                "Train_CI_lo": None, "Train_CI_hi": None,
                "CV_AUC": round(mn_te_m, 4),
                "CV_CI_lo": round(lo_te_m, 4), "CV_CI_hi": round(hi_te_m, 4),
            })

lc_df = pd.DataFrame(lc_rows)
lc_df.to_excel(_os.path.join(OUT_XL, "Table_SF4_LearningCurves.xlsx"), index=False)
print(f"\n  Saved Table_SF4_LearningCurves.xlsx")


# ============================================================
#  SECTION 2: SEX-STRATIFIED AUC  (SF5)
# ============================================================
print("\n" + "="*60)
print("SECTION 2: Sex-Stratified AUC (SF5)")
print("="*60)

# Custom 5×20 CV loop: within each fold, compute AUC for each sex in test set
RSKF = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RNG)

# Determine sex coding
sex_unique = np.unique(sex_vals)
print(f"  Unique sex values: {sex_unique}")
# Assume: 1=Male, 0=Female (common coding for Gender column)
# Verify via label counts
for sv in sex_unique:
    idx = sex_vals == sv
    print(f"    sex={sv}: N={idx.sum()}, cases={y_tr[idx].sum()}")

SEX_LABELS = {1: "Male", 0: "Female"}
# Handle if coded differently (e.g. "M"/"F")
if not set(sex_unique).issubset({0, 1}):
    # recode
    uniq_sorted = sorted(sex_unique)
    SEX_LABELS = {uniq_sorted[0]: "Female", uniq_sorted[1]: "Male"}
    print(f"  Recoding: {SEX_LABELS}")

sex_rows = []

for fs_name, cols in FEATURE_SETS.items():
    missing = [c for c in cols if c not in tr.columns]
    if missing:
        print(f"  SKIP {fs_name} — missing: {missing}")
        continue
    X = tr[cols].values
    print(f"\n  {fs_name}:")

    for mn in MODEL_NAMES:
        aucs_overall = []
        aucs_sex     = {sv: [] for sv in sex_unique}

        for tr_i, te_i in RSKF.split(X, y_tr):
            if len(np.unique(y_tr[te_i])) < 2:
                continue
            pipe = make_pipeline(mn)
            pipe.fit(X[tr_i], y_tr[tr_i])
            yp = pipe.predict_proba(X[te_i])[:, 1]

            # Overall AUC
            if len(np.unique(y_tr[te_i])) >= 2:
                aucs_overall.append(roc_auc_score(y_tr[te_i], yp))

            # Sex-stratified AUC
            sex_te = sex_vals[te_i]
            for sv in sex_unique:
                idx_sv = sex_te == sv
                if idx_sv.sum() < 5:
                    continue
                y_sv  = y_tr[te_i][idx_sv]
                yp_sv = yp[idx_sv]
                if len(np.unique(y_sv)) < 2:
                    continue
                aucs_sex[sv].append(roc_auc_score(y_sv, yp_sv))

        # Summarize overall
        mn_o, lo_o, hi_o = ci95(aucs_overall)
        sex_rows.append({
            "FeatureSet": fs_name, "Model": mn, "Group": "Overall",
            "N_folds": len(aucs_overall),
            "Mean_AUC": round(mn_o, 4),
            "CI_lo": round(lo_o, 4), "CI_hi": round(hi_o, 4),
            "Mean_95CI": f"{mn_o:.3f} ({lo_o:.3f}–{hi_o:.3f})",
        })

        # Summarize per sex
        for sv in sex_unique:
            label = SEX_LABELS[sv]
            vals_sv = aucs_sex[sv]
            if len(vals_sv) < 5:
                print(f"    {mn} × {label}: too few valid folds ({len(vals_sv)})")
                continue
            mn_s, lo_s, hi_s = ci95(vals_sv)
            sex_rows.append({
                "FeatureSet": fs_name, "Model": mn, "Group": label,
                "N_folds": len(vals_sv),
                "Mean_AUC": round(mn_s, 4),
                "CI_lo": round(lo_s, 4), "CI_hi": round(hi_s, 4),
                "Mean_95CI": f"{mn_s:.3f} ({lo_s:.3f}–{hi_s:.3f})",
            })
            print(f"    {mn} × {label}: {mn_s:.3f} ({lo_s:.3f}–{hi_s:.3f})  n_folds={len(vals_sv)}")

    # Average across models
    for grp in ["Overall"] + [SEX_LABELS[sv] for sv in sex_unique]:
        sub = [r for r in sex_rows
               if r["FeatureSet"] == fs_name
               and r["Model"] in MODEL_NAMES
               and r["Group"] == grp]
        if not sub:
            continue
        mn_avg = np.mean([r["Mean_AUC"] for r in sub])
        sd_avg = float(np.sqrt(np.mean(
            [((r["CI_hi"] - r["CI_lo"]) / (2 * 1.96) * np.sqrt(r["N_folds"]))**2
             for r in sub])))
        n_avg  = int(np.mean([r["N_folds"] for r in sub]))
        lo_avg = mn_avg - 1.96 * sd_avg / np.sqrt(n_avg)
        hi_avg = mn_avg + 1.96 * sd_avg / np.sqrt(n_avg)
        sex_rows.append({
            "FeatureSet": fs_name, "Model": "Average", "Group": grp,
            "N_folds": n_avg,
            "Mean_AUC": round(mn_avg, 4),
            "CI_lo": round(lo_avg, 4), "CI_hi": round(hi_avg, 4),
            "Mean_95CI": f"{mn_avg:.3f} ({lo_avg:.3f}–{hi_avg:.3f})",
        })

sex_df = pd.DataFrame(sex_rows)
sex_df.to_excel(_os.path.join(OUT_XL, "Table_SF5_SexStratified.xlsx"), index=False)
print(f"\n  Saved Table_SF5_SexStratified.xlsx")


# ============================================================
#  SECTION 3: BUILD FIGURES
# ============================================================
print("\n" + "="*60)
print("SECTION 3: Building SF4 and SF5")
print("="*60)

# ── SF4: Learning Curves ──────────────────────────────────────
print("Building SF4 (Learning Curves) ...")

lc_avg = lc_df[lc_df["Model"] == "Average"].copy()

fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
fig4.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.14, wspace=0.38)

# Panel A: CV AUC vs training size (Average across classifiers)
ax4a = axes4[0]
for fs_name in FEATURE_SETS:
    sub = lc_avg[lc_avg["FeatureSet"] == fs_name].sort_values("N_train_approx")
    if sub.empty:
        continue
    x  = sub["N_train_approx"].values
    y  = sub["CV_AUC"].values
    lo = sub["CV_CI_lo"].values
    hi = sub["CV_CI_hi"].values
    color = FS_COLORS[fs_name]
    ls    = FS_STYLES[fs_name]
    ax4a.plot(x, y, color=color, linestyle=ls, linewidth=2.0,
              marker="o", markersize=6, label=fs_name, zorder=4)
    ax4a.fill_between(x, lo, hi, color=color, alpha=0.12, zorder=2)

ax4a.axhline(0.5, color="dimgrey", linewidth=0.9, linestyle=":", zorder=1)
ax4a.set_xlabel("Approximate Training Set Size (N)", fontsize=10)
ax4a.set_ylabel("Cross-Validation AUC (mean ± 95% CI)", fontsize=10)
ax4a.set_title("Learning Curves — Average Classifier\n(5-fold × 4-repeat CV, POAAGG N=271)",
               fontsize=10, fontweight="bold")
ax4a.legend(frameon=True, framealpha=0.9, fontsize=8.5, loc="lower right")
ax4a.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax4a.set_axisbelow(True)
ax4a.text(-0.12, 1.06, "A", transform=ax4a.transAxes,
          fontsize=14, fontweight="bold", va="top")

# Panel B: Train AUC vs CV AUC for each FS (gap = overfitting indicator)
ax4b = axes4[1]
for fs_name in FEATURE_SETS:
    sub = lc_avg[lc_avg["FeatureSet"] == fs_name].sort_values("N_train_approx")
    if sub.empty:
        continue
    x     = sub["N_train_approx"].values
    y_cv  = sub["CV_AUC"].values
    y_tr  = sub["Train_AUC"].values
    color = FS_COLORS[fs_name]
    ls    = FS_STYLES[fs_name]
    # CV AUC (solid)
    ax4b.plot(x, y_cv, color=color, linestyle=ls, linewidth=2.0,
              marker="o", markersize=6,
              label=f"{fs_name} (CV)", zorder=4)
    # Training AUC (lighter, dashed variant)
    ax4b.plot(x, y_tr, color=color, linestyle=":",
              linewidth=1.4, marker="s", markersize=4,
              alpha=0.6, label=f"{fs_name} (Train)", zorder=3)

ax4b.axhline(0.5, color="dimgrey", linewidth=0.9, linestyle=":", zorder=1)
ax4b.set_xlabel("Approximate Training Set Size (N)", fontsize=10)
ax4b.set_ylabel("AUC (mean)", fontsize=10)
ax4b.set_title("Training vs. CV AUC — Overfitting Assessment\n(solid = CV; dotted = Training)",
               fontsize=10, fontweight="bold")
ax4b.legend(frameon=True, framealpha=0.9, fontsize=7.5, loc="lower right",
            ncol=2)
ax4b.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax4b.set_axisbelow(True)
ax4b.text(-0.12, 1.06, "B", transform=ax4b.transAxes,
          fontsize=14, fontweight="bold", va="top")

# suptitle removed per journal style

for ext in ["png", "pdf"]:
    fig4.savefig(_os.path.join(OUT_FIG, "SF4_LearningCurves.{ext}"),
                 bbox_inches="tight", dpi=300)
    print(f"  Saved SF4_LearningCurves.{ext}")
plt.close(fig4)


# ── SF5: Sex-Stratified AUC ───────────────────────────────────
print("Building SF5 (Sex-Stratified AUC) ...")

sex_avg = sex_df[sex_df["Model"] == "Average"].copy()

# Identify sex group labels
sex_groups = [g for g in sex_avg["Group"].unique() if g != "Overall"]
grp_colors = {"Overall": "#555555"}
# Assign colors to sex groups
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for i, g in enumerate(sorted(sex_groups)):
    grp_colors[g] = palette[i % len(palette)]

ALL_GROUPS  = ["Overall"] + sorted(sex_groups)
FS_LIST     = list(FEATURE_SETS.keys())
N_FS        = len(FS_LIST)
N_GRP       = len(ALL_GROUPS)
SPACING     = 0.20

fig5, ax5 = plt.subplots(figsize=(9, 4.5))
fig5.subplots_adjust(left=0.22, right=0.90, top=0.88, bottom=0.14)

yticks, yticklabs = [], []
xlim = (0.40, 0.90)

GRP_MARKERS = {"Overall": "D", "Male": "o", "Female": "s",
               "1": "o", "0": "s"}

for fi, fs_name in enumerate(FS_LIST):
    y_center = fi * 1.0
    yticks.append(y_center)
    yticklabs.append(fs_name)
    offsets = np.linspace(-(N_GRP - 1) * SPACING / 2,
                           (N_GRP - 1) * SPACING / 2, N_GRP)

    for gi, grp in enumerate(ALL_GROUPS):
        row = sex_avg[(sex_avg["FeatureSet"] == fs_name) &
                      (sex_avg["Group"] == grp)]
        if row.empty:
            continue
        mn = float(row["Mean_AUC"].iloc[0])
        lo = float(row["CI_lo"].iloc[0])
        hi = float(row["CI_hi"].iloc[0])
        ypos   = y_center + offsets[gi]
        color  = grp_colors.get(grp, "#888888")
        marker = GRP_MARKERS.get(grp, "o")

        ax5.plot(mn, ypos, marker=marker, color=color,
                 markersize=8, zorder=4,
                 markeredgecolor="white", markeredgewidth=0.7)
        ax5.errorbar(mn, ypos, xerr=[[mn - lo], [hi - mn]],
                     fmt="none", color=color,
                     capsize=3.5, linewidth=1.3, zorder=3)
        ax5.text(hi + (xlim[1] - xlim[0]) * 0.013, ypos,
                 f"{mn:.3f}", va="center", ha="left",
                 fontsize=8, color=color)

    if fi % 2 == 0:
        ax5.axhspan(y_center - 0.5, y_center + 0.5,
                    color="#f5f5f5", zorder=0)

ax5.axvline(0.5, color="dimgrey", linewidth=0.9, linestyle=":", zorder=1)
ax5.set_yticks(yticks)
ax5.set_yticklabels(yticklabs, fontsize=9.5)
ax5.set_xlim(*xlim)
ax5.set_ylim(-0.6, (N_FS - 1) + 0.6)
ax5.set_xlabel("AUC (mean ± 95% CI)", fontsize=10)
ax5.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax5.set_axisbelow(True)
ax5.spines["left"].set_visible(False)
ax5.tick_params(axis="y", length=0)
ax5.set_title(
    "Sex-Stratified Cross-Validation AUC — Average Classifier\n"
    "(5×20 CV, POAAGG N=271)",
    fontsize=10, fontweight="bold")

# Legend
handles = []
for grp in ALL_GROUPS:
    handles.append(mlines.Line2D(
        [], [], color=grp_colors.get(grp, "#888888"),
        marker=GRP_MARKERS.get(grp, "o"),
        markersize=8, markeredgecolor="white",
        markeredgewidth=0.7, linewidth=0, label=grp))
ax5.legend(handles=handles, frameon=True, framealpha=0.9,
           fontsize=9, loc="lower right")

# suptitle removed per journal style

for ext in ["png", "pdf"]:
    fig5.savefig(_os.path.join(OUT_FIG, "SF5_SexStratified.{ext}"),
                 bbox_inches="tight", dpi=300)
    print(f"  Saved SF5_SexStratified.{ext}")
plt.close(fig5)

print("\nAll done.")
