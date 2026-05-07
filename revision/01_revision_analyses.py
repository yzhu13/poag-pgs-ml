"""
iScience Revision Analysis Pipeline
Manuscript: ISCIENCE-D-26-03991
Date: 2026-04-29

Addresses reviewer comments:
  Analysis 1  - PC Sensitivity Analysis (ancestry vs PGS confounding)
  Analysis 2  - Repeated 10x10 K-Fold CV for all models/PGS combos (validation rigor)
  Analysis 3  - Logistic Regression as formal baseline
  Analysis 4  - SHAP Feature Importance
  Analysis 5  - Learning Curves
  Analysis 6  - Calibration Curves + Brier Score
  Analysis 7  - Suspect Cohort Disentanglement (baseline vs PGS signal)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold, cross_validate, learning_curve
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, RocCurveDisplay
)
from sklearn.base import clone as sklearn_clone
import shap

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
TRAIN_FILE  = "../input date/271_training_cohort_4_new_PRS_cleaned.xlsx"
SUSPECT_FILE = "../input date/1013_testing_cohort_only_suspect_cleaned.xlsx"
OUT_DIR     = "./"   # outputs go into revision output/

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# Load training data
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_excel(TRAIN_FILE)
suspect = pd.read_excel(SUSPECT_FILE)

# Column mapping - training cohort
# col2=POAAGG PRS, col3=MEGA PRS, col4=PRS526, col5=PRS616
# col8-27=PC1-PC20, col28=Gender, col29=Age, col7=CaseCtrl
PGS_COLS = {
    "POAAGG PRS": "POAAGG_PRS",
    "MEGA PRS":   "MEGA_PRS",
    "PRS526":     "PRS526",
    "PRS616":     "PRS616",
}
train = train.rename(columns=PGS_COLS)
PC_COLS = [f"PC{i}" for i in range(1, 21)]
DEMO_COLS = ["Age", "Gender"]
LABEL_COL = "CaseCtrl"

# Suspect cohort PGS column mapping
# col2=PRS-CS POAAGG, col3=PRS-CS MEGA, col4=MTAG PRS, col5=MEGA PRS
sus_pgs_map = {
    "PRS-CS POAAGG": "POAAGG_PRS",
    "PRS-CS MEGA":   "MEGA_PRS",
    "MTAG PRS":      "PRS526",
    "MEGA PRS":      "PRS616",
}
suspect = suspect.rename(columns=sus_pgs_map)

y = train[LABEL_COL].values

# ─────────────────────────────────────────────
# Helper: build sklearn Pipeline
# ─────────────────────────────────────────────
def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

def lr():
    return make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))

def rf():
    return make_pipeline(RandomForestClassifier(n_estimators=200, random_state=SEED))

def svm():
    return make_pipeline(SVC(probability=True, kernel="rbf", random_state=SEED))

def mlp():
    return make_pipeline(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                       random_state=SEED, early_stopping=True))

CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)

def run_cv(X, y, clf_fn, label=""):
    pipe = clf_fn()
    scores = cross_validate(pipe, X, y, cv=CV, scoring="roc_auc", n_jobs=-1)
    aucs = scores["test_score"]
    print(f"  {label:45s}  AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")
    return aucs.mean(), aucs.std()

# ─────────────────────────────────────────────
# ANALYSIS 1 — PC Sensitivity Analysis
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 1: PC Sensitivity (ancestry vs PGS confounding)")
print("="*60)

pc_configs = {
    "Age + Sex (no PCs)": DEMO_COLS,
    "Age + Sex + PC1-2":  DEMO_COLS + PC_COLS[:2],
    "Age + Sex + PC1-5":  DEMO_COLS + PC_COLS[:5],
    "Age + Sex + PC1-10": DEMO_COLS + PC_COLS[:10],
    "Age + Sex + PC1-20": DEMO_COLS + PC_COLS[:20],
    "Age + Sex + PC1-5 + PRS616":  DEMO_COLS + PC_COLS[:5]  + ["PRS616"],
    "Age + Sex + PC1-20 + PRS616": DEMO_COLS + PC_COLS[:20] + ["PRS616"],
}

pc_results = {}
for label, cols in pc_configs.items():
    X = train[cols].values
    m, s = run_cv(X, y, lr, label)
    pc_results[label] = (m, s)

# PGS vs PC correlation
print("\nPGS vs PC Spearman correlations:")
corr_rows = []
for pgs in ["POAAGG_PRS", "MEGA_PRS", "PRS526", "PRS616"]:
    for pc in PC_COLS[:5]:
        r, p = stats.spearmanr(train[pgs], train[pc])
        corr_rows.append({"PGS": pgs, "PC": pc, "r": round(r, 4), "p": round(p, 4)})
        if abs(r) > 0.15:
            print(f"  {pgs} vs {pc}: r={r:.4f}, p={p:.4f}  ← notable")
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT_DIR + "A1_PGS_PC_correlations.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = list(pc_results.keys())
means  = [pc_results[l][0] for l in labels]
stds   = [pc_results[l][1] for l in labels]
colors = ["#4878CF"] * 5 + ["#E87D4B", "#D43F3A"]

ax = axes[0]
bars = ax.barh(labels, means, xerr=stds, color=colors, edgecolor="black",
               linewidth=0.7, capsize=4, error_kw={"elinewidth": 1.2})
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Mean AUC (10x10 repeated CV)", fontsize=11)
ax.set_title("PC Sensitivity: LR models\n(orange/red = with PRS616)", fontsize=11)
ax.set_xlim(0.45, 1.0)
for bar, m, s in zip(bars, means, stds):
    ax.text(m + s + 0.005, bar.get_y() + bar.get_height()/2,
            f"{m:.3f}±{s:.3f}", va="center", fontsize=8)

# Heatmap of PGS-PC correlations
pivot = corr_df.pivot(index="PGS", columns="PC", values="r")
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            ax=axes[1], linewidths=0.5, cbar_kws={"label": "Spearman r"})
axes[1].set_title("PGS vs PC1-5 Spearman correlations\n(|r|>0.15 indicates shared signal)", fontsize=11)
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(OUT_DIR + "A1_PC_sensitivity.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved A1_PC_sensitivity.png + A1_PGS_PC_correlations.csv")

# ─────────────────────────────────────────────
# ANALYSIS 2 — Repeated 10x10 CV: All models x All PGS
# (includes logistic regression as formal baseline)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 2+3: Repeated 10x10 CV — All models x All PGS")
print("="*60)

feature_sets = {
    "Age + Sex":              DEMO_COLS,
    "Age + Sex + POAAGG_PRS": DEMO_COLS + ["POAAGG_PRS"],
    "Age + Sex + MEGA_PRS":   DEMO_COLS + ["MEGA_PRS"],
    "Age + Sex + PRS526":     DEMO_COLS + ["PRS526"],
    "Age + Sex + PRS616":     DEMO_COLS + ["PRS616"],
}

model_fns = {
    "Logistic Regression": lr,
    "Random Forest":       rf,
    "SVM":                 svm,
    "MLP":                 mlp,
}

cv_rows = []
for feat_label, cols in feature_sets.items():
    X = train[cols].values
    for model_label, clf_fn in model_fns.items():
        tag = f"{model_label} | {feat_label}"
        m, s = run_cv(X, y, clf_fn, tag)
        cv_rows.append({
            "Model": model_label, "Features": feat_label,
            "Mean AUC": round(m, 4), "SD AUC": round(s, 4),
            "AUC_str": f"{m:.3f}±{s:.3f}"
        })

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUT_DIR + "A2_CV_results.csv", index=False)

# Pivot for heatmap
pivot_mean = cv_df.pivot(index="Model", columns="Features", values="Mean AUC")
pivot_str  = cv_df.pivot(index="Model", columns="Features", values="AUC_str")
model_order = ["Logistic Regression", "SVM", "Random Forest", "MLP"]
feat_order  = list(feature_sets.keys())
pivot_mean = pivot_mean.loc[model_order, feat_order]
pivot_str  = pivot_str.loc[model_order, feat_order]

fig, ax = plt.subplots(figsize=(13, 4.5))
sns.heatmap(pivot_mean, annot=pivot_str, fmt="", cmap="YlOrRd",
            vmin=0.5, vmax=0.9, ax=ax, linewidths=0.5,
            cbar_kws={"label": "Mean AUC"})
ax.set_title("Mean AUC ± SD — 10x10 Repeated Stratified K-Fold CV\n"
             "(each cell = mean ± SD across 100 CV folds)", fontsize=11)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=25, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR + "A2_CV_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved A2_CV_heatmap.png + A2_CV_results.csv")

# ─────────────────────────────────────────────
# ANALYSIS 4 — SHAP Feature Importance
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 4: SHAP Feature Importance (SVM + PRS616 + Age + Sex)")
print("="*60)

shap_features = DEMO_COLS + ["PRS616"]
X_shap = train[shap_features].values
scaler = StandardScaler()
X_shap_scaled = scaler.fit_transform(X_shap)

# Also run RF for TreeExplainer (faster, complementary)
rf_shap_features = DEMO_COLS + ["POAAGG_PRS", "MEGA_PRS", "PRS526", "PRS616"]
X_rf = train[rf_shap_features].values
X_rf_scaled = scaler.fit_transform(X_rf)

# RF SHAP (TreeExplainer — fast, all features)
imputer_rf = SimpleImputer(strategy="median")
X_rf_imp = imputer_rf.fit_transform(X_rf)
scaler_rf = StandardScaler()
X_rf_scaled = scaler_rf.fit_transform(X_rf_imp)

rf_model = RandomForestClassifier(n_estimators=200, random_state=SEED)
rf_model.fit(X_rf_scaled, y)
explainer_rf = shap.TreeExplainer(rf_model)
shap_obj_rf = explainer_rf(X_rf_scaled)  # returns Explanation object (new API)
# shap_obj_rf.values shape: (n_samples, n_features, n_classes) for RF binary
sv_rf_raw = shap_obj_rf.values
if sv_rf_raw.ndim == 3:
    sv_rf = sv_rf_raw[:, :, 1]   # class-1 SHAP values
elif isinstance(sv_rf_raw, list):
    sv_rf = sv_rf_raw[1]
else:
    sv_rf = sv_rf_raw

# SVM: use sklearn permutation importance (avoids KernelExplainer shape issues)
from sklearn.inspection import permutation_importance
imputer_svm = SimpleImputer(strategy="median")
X_shap_imp = imputer_svm.fit_transform(X_shap)
scaler_svm = StandardScaler()
X_shap_scaled = scaler_svm.fit_transform(X_shap_imp)

svm_model = SVC(probability=True, kernel="rbf", random_state=SEED)
svm_model.fit(X_shap_scaled, y)
perm_result = permutation_importance(svm_model, X_shap_scaled, y,
                                     n_repeats=30, scoring="roc_auc",
                                     random_state=SEED, n_jobs=-1)
svm_perm_means = perm_result.importances_mean
svm_perm_stds  = perm_result.importances_std

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF SHAP beeswarm
plt.sca(axes[0])
shap.summary_plot(sv_rf, X_rf_scaled, feature_names=rf_shap_features,
                  show=False, max_display=8, plot_type="dot")
axes[0].set_title("RF SHAP — All PGS + Demographics\n(all 271 samples, TreeExplainer)", fontsize=10)

# SVM permutation importance bar
order = np.argsort(svm_perm_means)
axes[1].barh([shap_features[i] for i in order], svm_perm_means[order],
             xerr=svm_perm_stds[order], color="#4878CF", edgecolor="black",
             capsize=4, error_kw={"elinewidth": 1.2})
axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("Mean decrease in AUC (permutation)", fontsize=10)
axes[1].set_title("SVM Permutation Importance\n(Age + Sex + PRS616, 30 repeats)", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR + "A4_SHAP.png", dpi=200, bbox_inches="tight")
plt.close()

# Save RF SHAP values
rf_imp = pd.DataFrame({
    "Feature": rf_shap_features,
    "Mean_abs_SHAP": np.abs(sv_rf).mean(axis=0)
}).sort_values("Mean_abs_SHAP", ascending=False)
rf_imp.to_csv(OUT_DIR + "A4_RF_SHAP_importance.csv", index=False)

# Save SVM permutation importance
svm_imp = pd.DataFrame({
    "Feature": shap_features,
    "Mean_AUC_decrease": svm_perm_means,
    "SD": svm_perm_stds
}).sort_values("Mean_AUC_decrease", ascending=False)
svm_imp.to_csv(OUT_DIR + "A4_SVM_permutation_importance.csv", index=False)
print("  -> Saved A4_SHAP.png + A4_RF_SHAP_importance.csv + A4_SVM_permutation_importance.csv")

# ─────────────────────────────────────────────
# ANALYSIS 5 — Learning Curves
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 5: Learning Curves")
print("="*60)

lc_features = DEMO_COLS + ["PRS616"]
X_lc = train[lc_features].values

lc_models = {
    "Logistic Regression": lr(),
    "SVM":                 svm(),
    "Random Forest":       rf(),
    "MLP":                 mlp(),
}
train_sizes = np.linspace(0.15, 1.0, 10)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, (name, pipe) in zip(axes, lc_models.items()):
    tr_sizes, tr_scores, cv_scores = learning_curve(
        pipe, X_lc, y,
        train_sizes=train_sizes,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED),
        scoring="roc_auc", n_jobs=-1
    )
    tr_mean = tr_scores.mean(axis=1)
    tr_std  = tr_scores.std(axis=1)
    cv_mean = cv_scores.mean(axis=1)
    cv_std  = cv_scores.std(axis=1)

    ax.plot(tr_sizes, tr_mean, "o-", color="#2C7BB6", label="Training AUC")
    ax.fill_between(tr_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#2C7BB6")
    ax.plot(tr_sizes, cv_mean, "o-", color="#D7191C", label="CV AUC")
    ax.fill_between(tr_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color="#D7191C")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{name}\n(Age + Sex + PRS616)", fontsize=10)
    ax.set_xlabel("Training set size (N)", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8)
    ax.set_xticks(tr_sizes)
    ax.set_xticklabels([f"{int(s*271)}" for s in train_sizes], rotation=45, fontsize=7)

fig.suptitle("Learning Curves — Age + Sex + PRS616\n"
             "(gap between train/CV lines indicates overfitting)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR + "A5_learning_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("  -> Saved A5_learning_curves.png")

# ─────────────────────────────────────────────
# ANALYSIS 6 — Calibration Curves + Brier Score
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 6: Calibration Curves + Brier Score")
print("="*60)

cal_features = DEMO_COLS + ["PRS616"]
X_cal = train[cal_features].values

cal_models_raw = {
    "Logistic Regression": lr(),
    "SVM (calibrated)":    make_pipeline(CalibratedClassifierCV(
                               SVC(kernel="rbf", random_state=SEED), cv=5)),
    "Random Forest":       rf(),
    "MLP":                 mlp(),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
brier_rows = []

cv5 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)

for ax, (name, pipe) in zip(axes, cal_models_raw.items()):
    all_probs, all_true = [], []
    for tr_idx, te_idx in cv5.split(X_cal, y):
        pipe_clone = sklearn_clone(pipe)
        pipe_clone.fit(X_cal[tr_idx], y[tr_idx])
        probs = pipe_clone.predict_proba(X_cal[te_idx])[:, 1]
        all_probs.extend(probs)
        all_true.extend(y[te_idx])

    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)
    bs = brier_score_loss(all_true, all_probs)
    brier_rows.append({"Model": name, "Brier Score": round(bs, 4)})
    print(f"  {name}: Brier Score = {bs:.4f}")

    frac_pos, mean_pred = calibration_curve(all_true, all_probs, n_bins=8, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#2C7BB6", lw=2, label=f"Model (Brier={bs:.3f})")
    ax.set_title(f"Calibration — {name}\n(Age + Sex + PRS616, 5x5 CV)", fontsize=10)
    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_ylabel("Fraction of positives", fontsize=9)
    ax.legend(fontsize=8)

fig.suptitle("Calibration Curves (quantile binning, 5x5 repeated CV)\n"
             "Points near diagonal = well-calibrated", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR + "A6_calibration.png", dpi=200, bbox_inches="tight")
plt.close()

pd.DataFrame(brier_rows).to_csv(OUT_DIR + "A6_brier_scores.csv", index=False)
print("  -> Saved A6_calibration.png + A6_brier_scores.csv")

# ─────────────────────────────────────────────
# ANALYSIS 7 — Suspect Cohort: Disentangle Baseline vs PGS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 7: Suspect Cohort — Disentangle Age/Sex vs PGS Signal")
print("="*60)

# Train three models on full 271 training set, apply to 1013 suspects
# (i) age+sex only, (ii) PRS616 only, (iii) age+sex+PRS616

sus_clinical = {
    "IOP_SEVERE":  "IOP_SEVERE",
    "CDR_SEVERE":  "CDR_SEVERE",
    "RNFL_SEVERE": "RNFL_SEVERE",
}

# Map suspect column names
sus = suspect.copy()

models_7 = {
    "Age + Sex":          DEMO_COLS,
    "PRS616 only":        ["PRS616"],
    "Age + Sex + PRS616": DEMO_COLS + ["PRS616"],
}

# Confirm clinical columns exist in suspect file
avail_clinical = [c for c in sus_clinical.values() if c in sus.columns]
if not avail_clinical:
    # Try alternate names
    for col in sus.columns:
        print("  suspect col:", col)

scaler_full = StandardScaler()
corr_out = []

fig, axes = plt.subplots(len(models_7), len(avail_clinical),
                         figsize=(4.5 * len(avail_clinical), 4 * len(models_7)))
if axes.ndim == 1:
    axes = axes.reshape(1, -1)

for row_i, (model_label, feat_cols) in enumerate(models_7.items()):
    # Train on 271
    X_train = train[feat_cols].values
    pipe_7 = svm()
    pipe_7.fit(X_train, y)

    # Predict on suspects
    X_sus_feats = sus[feat_cols].dropna()
    valid_idx = X_sus_feats.index
    probs = pipe_7.predict_proba(X_sus_feats.values)[:, 1]
    sus_scores = pd.Series(probs, index=valid_idx, name="risk_score")

    for col_j, clin_col in enumerate(avail_clinical):
        clin_vals = sus.loc[valid_idx, clin_col].dropna()
        common_idx = sus_scores.index.intersection(clin_vals.index)
        scores_c = sus_scores.loc[common_idx]
        clin_c   = clin_vals.loc[common_idx]

        r, p = stats.spearmanr(scores_c, clin_c)
        p_str = f"p={p:.4f}" if p >= 0.001 else "p<0.001"
        corr_out.append({
            "Model": model_label, "Clinical": clin_col,
            "Spearman_r": round(r, 4), "p": round(p, 6), "N": len(common_idx)
        })
        print(f"  {model_label} vs {clin_col}: r={r:.4f}, {p_str}, N={len(common_idx)}")

        ax = axes[row_i, col_j]
        ax.scatter(scores_c, clin_c, alpha=0.3, s=8, color="#4878CF", rasterized=True)
        m_fit, b_fit = np.polyfit(scores_c, clin_c, 1)
        x_line = np.linspace(scores_c.min(), scores_c.max(), 100)
        ax.plot(x_line, m_fit * x_line + b_fit, color="#D43F3A", lw=1.5)
        ax.set_xlabel("Predicted POAG risk", fontsize=8)
        ax.set_ylabel(clin_col, fontsize=8)
        ax.set_title(f"{model_label}\nvs {clin_col}\n(r={r:.3f}, {p_str})", fontsize=8)

fig.suptitle("Suspect Cohort: Predicted Risk vs Clinical Features\n"
             "(SVM models trained on 271; applied to 1,013 suspects)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR + "A7_suspect_disentangle.png", dpi=200, bbox_inches="tight")
plt.close()

pd.DataFrame(corr_out).to_csv(OUT_DIR + "A7_suspect_correlations.csv", index=False)
print("  -> Saved A7_suspect_disentangle.png + A7_suspect_correlations.csv")

# ─────────────────────────────────────────────
# ANALYSIS 7b — Inter-eye Asymmetry (consistent primary model)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 7b: Asymmetry — Fix Model Inconsistency (use SVM consistently)")
print("="*60)

asym_cols = ["delta_IOP", "delta_CDR", "delta_RNFL"]
avail_asym = [c for c in asym_cols if c in sus.columns]

if avail_asym:
    asym_feat_sets = {
        "Age + Sex":          DEMO_COLS,
        "Age + Sex + PRS616": DEMO_COLS + ["PRS616"],
    }
    asym_rows = []
    for feat_label, feat_cols in asym_feat_sets.items():
        X_train = train[feat_cols].values
        pipe_asym = svm()
        pipe_asym.fit(X_train, y)
        X_sus_feats = sus[feat_cols].dropna()
        valid_idx = X_sus_feats.index
        probs = pipe_asym.predict_proba(X_sus_feats.values)[:, 1]
        sus_scores = pd.Series(probs, index=valid_idx)
        for col in avail_asym:
            clin_vals = sus.loc[valid_idx, col].dropna().abs()
            common_idx = sus_scores.index.intersection(clin_vals.index)
            r, p = stats.spearmanr(sus_scores.loc[common_idx], clin_vals.loc[common_idx])
            p_str = f"{p:.4f}" if p >= 0.001 else "<0.001"
            asym_rows.append({
                "Features": feat_label, "Asymmetry": col,
                "Spearman_r": round(r, 4), "p": p_str, "N": len(common_idx)
            })
            print(f"  SVM | {feat_label} vs |{col}|: r={r:.4f}, p={p_str}")
    pd.DataFrame(asym_rows).to_csv(OUT_DIR + "A7b_asymmetry_SVM_consistent.csv", index=False)
    print("  -> Saved A7b_asymmetry_SVM_consistent.csv")
else:
    print("  delta columns not found in suspect file — skipping")

# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ALL ANALYSES COMPLETE")
print("="*60)
summary = [
    ("A1_PC_sensitivity.png",          "Analysis 1: PC sensitivity bar chart + AUC values"),
    ("A1_PGS_PC_correlations.csv",     "Analysis 1: PGS vs PC Spearman correlations"),
    ("A2_CV_heatmap.png",              "Analysis 2+3: Repeated CV AUC heatmap (all models x PGS)"),
    ("A2_CV_results.csv",              "Analysis 2+3: Full CV table with mean ± SD"),
    ("A4_SHAP.png",                    "Analysis 4: SHAP beeswarm (RF) + bar (SVM)"),
    ("A4_RF_SHAP_importance.csv",      "Analysis 4: RF SHAP importance values"),
    ("A5_learning_curves.png",         "Analysis 5: Learning curves for all 4 models"),
    ("A6_calibration.png",             "Analysis 6: Calibration curves"),
    ("A6_brier_scores.csv",            "Analysis 6: Brier scores"),
    ("A7_suspect_disentangle.png",     "Analysis 7: Suspect cohort baseline vs PGS scatter plots"),
    ("A7_suspect_correlations.csv",    "Analysis 7: Spearman r table for all model x clinical combos"),
    ("A7b_asymmetry_SVM_consistent.csv","Analysis 7b: Asymmetry analysis using consistent SVM model"),
]
for fname, desc in summary:
    print(f"  {fname:45s}  {desc}")
