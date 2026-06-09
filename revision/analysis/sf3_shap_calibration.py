# =============================================================
#  SF3 — SHAP Feature Importance + Calibration Curves
#  Addresses: E6 / R1-C4 (SHAP), R2-KI4 (calibration + Brier)
#
#  Panel A: SHAP beeswarm — MLP × Base+PGS616 (N=271, full model)
#  Panel B: SHAP mean |SHAP| bar — compare Base, Base+PGS616, Base+PC5+PGS616
#  Panel C: Calibration curves — MLP, 5-fold CV, key feature sets
#  Panel D: Brier score bar — all classifiers × key feature sets
#
#  Exports:
#    output excel data/Table_SF3_SHAP.xlsx
#    output excel data/Table_SF3_Calibration.xlsx
#    figure/Figure 3 and SF2/SF3_SHAP_Calibration.png/.pdf
# =============================================================
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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
OUT_XL  = fr"{BASE}\output excel data"
OUT_FIG = fr"{BASE}\figure\Figure 3 and SF2"

LABEL      = "CaseCtrl"
RNG        = 42
MODEL_NAMES= ["LR", "SVM", "RF", "MLP"]
MODEL_COLORS = {
    "LR": "#4E79A7", "SVM": "#F28E2B", "RF": "#59A14F", "MLP": "#E15759",
}

BASE_COLS   = ["Age", "Gender"]
PC5_COLS    = [f"PC{i}" for i in range(1, 6)]

# Feature sets for SHAP and calibration
FEAT_SETS = {
    "Base":              BASE_COLS,
    "Base+PGS616":       BASE_COLS + ["PGS616"],
    "Base+PC5+PGS616":   BASE_COLS + PC5_COLS + ["PGS616"],
}

# Display names for features
FEAT_DISPLAY = {
    "Age":    "Age (years)",
    "Gender": "Sex (male=1)",
    "PGS616": "PGS616",
    "PC1":    "PC1", "PC2": "PC2", "PC3": "PC3",
    "PC4":    "PC4", "PC5": "PC5",
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
        steps += [("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RNG))]
    elif name == "SVM":
        steps += [("clf", SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=RNG))]
    elif name == "RF":
        steps += [("clf", RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=RNG))]
    elif name == "MLP":
        steps += [("clf", MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=1000,
            early_stopping=False, random_state=RNG))]
    return Pipeline(steps)


# ── Load data ─────────────────────────────────────────────────
print("Loading data ...")
tr   = pd.read_excel(TRAIN_F)
y_tr = tr[LABEL].values.astype(int)
print(f"  N={len(tr)}, cases={y_tr.sum()}, controls={(y_tr==0).sum()}")


# ==================================================================
# SECTION 1 — SHAP feature importance (MLP, trained on full N=271)
# ==================================================================
print("\n=== SHAP Analysis ===")

shap_rows  = []   # for Excel export
shap_store = {}   # {fs_name: (shap_vals array, X_display, feature_display_names)}

for fs_name, cols in FEAT_SETS.items():
    print(f"  Computing SHAP for {fs_name} ...")
    X = tr[cols].values.astype(float)
    feat_names = [FEAT_DISPLAY.get(c, c) for c in cols]

    # Fit full pipeline on all N=271
    pipe = make_pipeline("MLP")
    pipe.fit(X, y_tr)

    # Transform X through preprocessor (all steps except final clf)
    prep = Pipeline(pipe.steps[:-1])
    X_t  = prep.transform(X)

    # KernelExplainer with kmeans background (k=20 for speed)
    k_bg = min(20, len(X_t))
    bg   = shap.kmeans(X_t, k_bg)
    clf  = pipe.named_steps["clf"]

    explainer   = shap.KernelExplainer(clf.predict_proba, bg)
    shap_values = explainer.shap_values(X_t, nsamples=150)
    # Handle both old API (list of arrays) and new API (3D array)
    if isinstance(shap_values, list):
        sv_pos = shap_values[1]          # list[class1], shape (N, n_features)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_pos = shap_values[:, :, 1]   # (N, n_features, n_classes) → class 1
    else:
        sv_pos = shap_values             # fallback

    shap_store[fs_name] = (sv_pos, X, feat_names)

    # Mean |SHAP| per feature
    mean_abs = np.abs(sv_pos).mean(axis=0)
    for fi, (col, fname) in enumerate(zip(cols, feat_names)):
        shap_rows.append({
            "FeatureSet": fs_name,
            "Feature_col": col,
            "Feature_display": fname,
            "Mean_abs_SHAP": round(float(mean_abs[fi]), 5),
            "Mean_SHAP": round(float(sv_pos[:, fi].mean()), 5),
            "SD_SHAP": round(float(sv_pos[:, fi].std()), 5),
        })
    print(f"    Mean |SHAP|: " +
          ", ".join(f"{f}={v:.4f}" for f, v in zip(feat_names, mean_abs)))

shap_df = pd.DataFrame(shap_rows)

# Save SHAP Excel
with pd.ExcelWriter(fr"{OUT_XL}\Table_SF3_SHAP.xlsx", engine="openpyxl") as w:
    shap_df.to_excel(w, sheet_name="Mean_AbsSHAP", index=False)
    for fs_name, (sv, X, fnames) in shap_store.items():
        sv_out = pd.DataFrame(sv, columns=fnames)
        sv_out.to_excel(w, sheet_name=fs_name[:28], index=False)
print("  SHAP Excel saved.")


# ==================================================================
# SECTION 2 — Calibration curves + Brier scores
# ==================================================================
print("\n=== Calibration + Brier Analysis ===")

# Use 5-fold CV (1 repeat) to get unbiased out-of-fold probabilities
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

# Also compute Brier from 5x20 CV for reporting (mean ± SD)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RNG)

calib_rows  = []   # calibration curve points
brier_rows  = []   # Brier scores

for fs_name, cols in FEAT_SETS.items():
    X = tr[cols].values.astype(float)

    for mn in MODEL_NAMES:
        pipe = make_pipeline(mn)

        # --- Calibration curve (5-fold CV predicted probabilities) ---
        y_prob_cv = cross_val_predict(pipe, X, y_tr,
                                      cv=cv5, method="predict_proba")[:, 1]
        frac_pos, mean_pred = calibration_curve(y_tr, y_prob_cv,
                                                n_bins=10, strategy="uniform")
        brier_cv5 = brier_score_loss(y_tr, y_prob_cv)

        for fp, mp in zip(frac_pos, mean_pred):
            calib_rows.append({
                "FeatureSet": fs_name, "Model": mn,
                "Mean_predicted": round(float(mp), 4),
                "Fraction_positive": round(float(fp), 4),
                "Brier_5fold": round(brier_cv5, 5),
            })

        # --- Brier score across 5x20 CV folds ---
        brier_vals = []
        for tr_i, te_i in rskf.split(X, y_tr):
            if len(np.unique(y_tr[te_i])) < 2:
                continue
            p = make_pipeline(mn)
            p.fit(X[tr_i], y_tr[tr_i])
            yp = p.predict_proba(X[te_i])[:, 1]
            brier_vals.append(brier_score_loss(y_tr[te_i], yp))

        mn_b = np.mean(brier_vals)
        sd_b = np.std(brier_vals, ddof=1)
        se_b = sd_b / np.sqrt(len(brier_vals))
        brier_rows.append({
            "FeatureSet": fs_name, "Model": mn,
            "Mean_Brier": round(mn_b, 5),
            "SD_Brier":   round(sd_b, 5),
            "SE_Brier":   round(se_b, 6),
            "CI_lo_95":   round(mn_b - 1.96 * se_b, 5),
            "CI_hi_95":   round(mn_b + 1.96 * se_b, 5),
            "Mean_Brier_CI": f"{mn_b:.4f} ± {sd_b:.4f}",
        })
        print(f"  {fs_name} × {mn}: Brier = {mn_b:.4f} ± {sd_b:.4f}")

calib_df = pd.DataFrame(calib_rows)
brier_df  = pd.DataFrame(brier_rows)

# Save calibration Excel
with pd.ExcelWriter(fr"{OUT_XL}\Table_SF3_Calibration.xlsx",
                    engine="openpyxl") as w:
    calib_df.to_excel(w, sheet_name="Calibration_Curves", index=False)
    brier_df.to_excel(w, sheet_name="Brier_Scores_5x20CV", index=False)
    # pivot: Brier mean by FeatureSet × Model
    piv = brier_df.pivot(index="FeatureSet", columns="Model",
                         values="Mean_Brier_CI")
    piv.to_excel(w, sheet_name="Brier_Pivot")
print("  Calibration Excel saved.")


# ==================================================================
# SECTION 3 — Build SF3 figure (4 panels)
# ==================================================================
print("\nBuilding SF3 figure ...")

fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.38,
                        left=0.08, right=0.97,
                        top=0.93, bottom=0.09)
ax_a = fig.add_subplot(gs[0, 0])   # SHAP beeswarm (Base+PGS616)
ax_b = fig.add_subplot(gs[0, 1])   # SHAP mean|SHAP| bar (3 feature sets)
ax_c = fig.add_subplot(gs[1, 0])   # Calibration curves (MLP)
ax_d = fig.add_subplot(gs[1, 1])   # Brier score bar (all classifiers)


# ── Panel A: SHAP beeswarm (Base+PGS616, MLP) ─────────────────
sv_b616, X_b616, fnames_b616 = shap_store["Base+PGS616"]
n_feat = sv_b616.shape[1]
n_samp = sv_b616.shape[0]

# Standardise X for coloring (0–1 per feature)
X_norm = (X_b616 - X_b616.min(0)) / (X_b616.max(0) - X_b616.min(0) + 1e-9)

jitter = 0.08
cmap_beeswarm = plt.cm.RdBu_r

for fi in range(n_feat):
    sv_col = sv_b616[:, fi]
    x_col  = X_norm[:, fi]
    # Jitter y positions
    rng_j  = np.random.default_rng(fi)
    y_jit  = fi + rng_j.uniform(-jitter, jitter, n_samp)
    colors = cmap_beeswarm(x_col)
    ax_a.scatter(sv_col, y_jit,
                 c=colors, s=18, alpha=0.7,
                 linewidths=0, zorder=3)

ax_a.axvline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=1)
ax_a.set_yticks(range(n_feat))
ax_a.set_yticklabels(fnames_b616, fontsize=9.5)
ax_a.set_xlabel("SHAP value  (impact on POAG risk score)", fontsize=9)
ax_a.set_title("A  SHAP Beeswarm — MLP × Base+PGS616\n"
               "(N=271; each dot = one participant)",
               fontsize=10, fontweight="bold")
ax_a.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax_a.set_axisbelow(True)
ax_a.spines["left"].set_visible(False)
ax_a.tick_params(axis="y", length=0)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_beeswarm,
                            norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_a, pad=0.02, fraction=0.03, aspect=25)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Low", "Mid", "High"], fontsize=7.5)
cbar.set_label("Feature value", fontsize=8)


# ── Panel B: Mean |SHAP| bar chart — 3 feature sets ───────────
FS_ORDER_B = ["Base", "Base+PGS616", "Base+PC5+PGS616"]
FS_COLORS_B = {
    "Base":            "#888888",
    "Base+PGS616":     "#2196A6",
    "Base+PC5+PGS616": "#9C27B0",
}
FS_LABELS_B = ["Base\n(Age+Sex)", "Base\n+PGS616", "Base+PC5\n+PGS616"]

# Collect all unique features in order
all_feats_ordered = ["Age (years)", "Sex (male=1)",
                     "PC1", "PC2", "PC3", "PC4", "PC5", "PGS616"]

x_pos   = np.arange(len(all_feats_ordered))
bar_w   = 0.22
offsets = np.linspace(-(len(FS_ORDER_B)-1)*bar_w/2,
                       (len(FS_ORDER_B)-1)*bar_w/2,
                       len(FS_ORDER_B))

for bi, fs_name in enumerate(FS_ORDER_B):
    sub = shap_df[shap_df["FeatureSet"] == fs_name]
    means = []
    for feat in all_feats_ordered:
        row = sub[sub["Feature_display"] == feat]
        means.append(float(row["Mean_abs_SHAP"].iloc[0]) if len(row) > 0 else 0.0)
    ax_b.bar(x_pos + offsets[bi], means, bar_w,
             color=FS_COLORS_B[fs_name], alpha=0.85,
             label=FS_LABELS_B[bi], zorder=3)
    for xp, mv in zip(x_pos + offsets[bi], means):
        if mv > 0:
            ax_b.text(xp, mv + 0.0005, f"{mv:.3f}",
                      ha="center", va="bottom", fontsize=6.5, rotation=0)

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(all_feats_ordered, fontsize=8.5, rotation=20, ha="right")
ax_b.set_ylabel("Mean |SHAP value|", fontsize=9)
ax_b.set_title("B  Feature Importance (Mean |SHAP|)\n"
               "MLP model — comparison across feature sets",
               fontsize=10, fontweight="bold")
ax_b.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax_b.set_axisbelow(True)
ax_b.legend(fontsize=8, framealpha=0.9, loc="upper right")


# ── Panel C: Calibration curves (MLP, 3 feature sets) ─────────
FS_COLORS_C = {
    "Base":            "#888888",
    "Base+PGS616":     "#2196A6",
    "Base+PC5+PGS616": "#9C27B0",
}
FS_LABELS_C = {
    "Base":            "Base (Age+Sex)",
    "Base+PGS616":     "Base+PGS616",
    "Base+PC5+PGS616": "Base+PC5+PGS616",
}

ax_c.plot([0, 1], [0, 1], "k--", linewidth=0.9,
          label="Perfect calibration", zorder=1)

for fs_name in FEAT_SETS:
    sub = calib_df[(calib_df["FeatureSet"] == fs_name) &
                   (calib_df["Model"] == "MLP")]
    brier_val = sub["Brier_5fold"].iloc[0] if len(sub) > 0 else np.nan
    mp  = sub["Mean_predicted"].values
    fop = sub["Fraction_positive"].values
    lbl = f"{FS_LABELS_C[fs_name]}  (Brier={brier_val:.4f})"
    ax_c.plot(mp, fop, "o-",
              color=FS_COLORS_C[fs_name],
              markersize=6, linewidth=1.6,
              label=lbl, zorder=3)

ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.set_xlabel("Mean predicted probability", fontsize=9)
ax_c.set_ylabel("Fraction of positives (observed)", fontsize=9)
ax_c.set_title("C  Calibration Curves — MLP (5-fold CV)\n"
               "Reliability diagram: predicted vs. observed POAG risk",
               fontsize=10, fontweight="bold")
ax_c.legend(fontsize=7.5, framealpha=0.9, loc="upper left")
ax_c.xaxis.grid(True, linestyle="--", alpha=0.35)
ax_c.yaxis.grid(True, linestyle="--", alpha=0.35)
ax_c.set_axisbelow(True)


# ── Panel D: Brier score bar — all classifiers × feature sets ──
FS_ORDER_D = ["Base", "Base+PGS616", "Base+PC5+PGS616"]
FS_LABELS_D = ["Base\n(Age+Sex)", "Base\n+PGS616", "Base+PC5\n+PGS616"]
x_pos_d = np.arange(len(FS_ORDER_D))
bar_w_d = 0.16
offsets_d = np.linspace(-(len(MODEL_NAMES)-1)*bar_w_d/2,
                         (len(MODEL_NAMES)-1)*bar_w_d/2,
                         len(MODEL_NAMES))

for mi, mn in enumerate(MODEL_NAMES):
    means_d, errs_d = [], []
    for fs_name in FS_ORDER_D:
        row = brier_df[(brier_df["FeatureSet"] == fs_name) &
                       (brier_df["Model"] == mn)]
        if len(row) == 0:
            means_d.append(np.nan); errs_d.append(0)
        else:
            m = float(row["Mean_Brier"].iloc[0])
            e = float(row["SE_Brier"].iloc[0]) * 1.96
            means_d.append(m); errs_d.append(e)

    ax_d.bar(x_pos_d + offsets_d[mi], means_d, bar_w_d,
             color=MODEL_COLORS[mn], alpha=0.85,
             label=mn, zorder=3, yerr=errs_d,
             error_kw={"capsize": 3, "linewidth": 0.8, "capthick": 0.8},
             ecolor="black")
    for xp, mv in zip(x_pos_d + offsets_d[mi], means_d):
        if not np.isnan(mv):
            ax_d.text(xp, mv + 0.002, f"{mv:.4f}",
                      ha="center", va="bottom", fontsize=6, rotation=90)

# Reference: Brier score for no-skill model (predicts prevalence for all)
prev  = y_tr.mean()
brier_ref = prev * (1 - prev)   # for a constant prediction model
ax_d.axhline(brier_ref, color="dimgrey", linewidth=1.0,
             linestyle=":", label=f"No-skill ({brier_ref:.3f})")

ax_d.set_xticks(x_pos_d)
ax_d.set_xticklabels(FS_LABELS_D, fontsize=9)
ax_d.set_ylabel("Brier Score (5×20 CV mean ± 95% CI)\n← lower is better", fontsize=9)
ax_d.set_ylim(0, 0.38)
ax_d.set_title("D  Brier Score — All Classifiers × Key Feature Sets\n"
               "(5-fold × 20-repeat CV; lower = better calibration)",
               fontsize=10, fontweight="bold")
ax_d.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax_d.set_axisbelow(True)
ax_d.legend(fontsize=8, framealpha=0.9, loc="upper right", ncol=2)

# suptitle removed per journal style

for ext in ["png", "pdf"]:
    fig.savefig(fr"{OUT_FIG}\SF3_SHAP_Calibration.{ext}",
                bbox_inches="tight", dpi=300)
    print(f"  Saved SF3_SHAP_Calibration.{ext}")
plt.close(fig)

# ==================================================================
# Print key numbers for manuscript
# ==================================================================
print("\n=== KEY NUMBERS FOR MANUSCRIPT ===")
print("\nSHAP Feature Importance (MLP × Base+PGS616, mean |SHAP|):")
sub = shap_df[shap_df["FeatureSet"] == "Base+PGS616"]
for _, row in sub.sort_values("Mean_abs_SHAP", ascending=False).iterrows():
    print(f"  {row['Feature_display']:20s}: {row['Mean_abs_SHAP']:.4f}")

print("\nBrier Scores (5×20 CV, MLP):")
mlp_brier = brier_df[brier_df["Model"] == "MLP"]
for _, row in mlp_brier.iterrows():
    print(f"  {row['FeatureSet']:25s}: {row['Mean_Brier']:.4f} ± {row['SD_Brier']:.4f}")

print(f"\n  Null model (predict prevalence): {brier_ref:.4f}")
print("\nAll done.")
