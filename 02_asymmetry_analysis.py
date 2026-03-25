"""
02_asymmetry_analysis.py
------------------------
Asymmetry-based ML analysis for POAG prediction.

Analyses produced
-----------------
1. Bootstrap CV with asymmetry feature sets (δIOP, δCDR, δRNFL ± PGS)
   -> outputs/asymmetry_bootstrap_cv_results.xlsx

2. Risk prediction in 1013 glaucoma suspects using
   (δIOP, δCDR, δRNFL, PRS616), for each model (RF, MLP, SVM)
   -> outputs/suspect_risks_{model}.xlsx

3. Validation plots (Figure 5-style)
   3a. Binned predicted risk vs mean |Δ| ± SEM (trend plot)
   3b. ECDF of |Δ| in top vs bottom 25% risk suspects + KS test
   -> outputs/figures/asymmetry_trend_{model}.png
   -> outputs/figures/asymmetry_ecdf_{model}.png
   -> outputs/asymmetry_validation_stats.xlsx

4. AUC bar charts per model for asymmetry feature sets
   -> outputs/figures/auc_asymmetry_{model}.png

Usage
-----
    python 02_asymmetry_analysis.py

Edit DATA_DIR and file name constants in config.py before running.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, ks_2samp

from config import (
    TRAIN_FILE, TEST_1013_FILE,
    OUTPUT_DIR, FIGURE_DIR,
    LABEL_COL, STATUS_COL,
    DELTA_IOP_COL, DELTA_CDR_COL, DELTA_RNFL_COL,
    ASYMMETRY_FEATURE_SETS, ASYMMETRY_RISK_FEATURES,
    MODELS, N_BOOTSTRAPS_ASYMMETRY,
)
from utils import bootstrap_evaluate, train_and_predict, plot_auc_bars

sns.set_theme(style="whitegrid", font_scale=1.1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DELTA_COLS = [DELTA_IOP_COL, DELTA_CDR_COL, DELTA_RNFL_COL]

# Only RF / MLP / SVM for asymmetry analysis
ASYMMETRY_MODELS = {k: v for k, v in MODELS.items()
                    if k in ("Random Forest", "MLP", "SVM")}


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load training cohort and 1013 suspect cohort.

    Returns
    -------
    train_df  : 271-sample training cohort
    suspect_df: 1013 glaucoma suspects
    """
    print("Loading data...")
    train_df   = pd.read_excel(TRAIN_FILE)
    suspect_df = pd.read_excel(TEST_1013_FILE)

    train_df = train_df.dropna(subset=[LABEL_COL])
    train_df[LABEL_COL] = train_df[LABEL_COL].astype(int)

    missing_delta = [c for c in DELTA_COLS if c not in train_df.columns]
    if missing_delta:
        print(f"  WARNING: asymmetry columns not found in training data: {missing_delta}")

    print(f"  Training cohort : {len(train_df)} samples")
    print(f"  Suspects        : {len(suspect_df)} samples")
    return train_df, suspect_df


# ---------------------------------------------------------------------------
# 2. Bootstrap CV with asymmetry feature sets
# ---------------------------------------------------------------------------

def run_asymmetry_bootstrap_cv(train_df):
    """Bootstrap CV for all asymmetry feature sets.

    Returns
    -------
    results_df : DataFrame (Feature Set, Model, Metric, Mean, SEM, CI95)
    """
    print(f"\nAsymmetry bootstrap CV ({N_BOOTSTRAPS_ASYMMETRY} iterations)...")
    records = []

    for fs_name, features in ASYMMETRY_FEATURE_SETS.items():
        missing = [f for f in features if f not in train_df.columns]
        if missing:
            print(f"  Skipping '{fs_name}' — missing columns: {missing}")
            continue
        X = train_df[features]
        y = train_df[LABEL_COL]

        for model_name, model in ASYMMETRY_MODELS.items():
            summary = bootstrap_evaluate(
                X, y, model, n_bootstraps=N_BOOTSTRAPS_ASYMMETRY
            )
            for metric, stats in summary.items():
                records.append({
                    "Feature Set": fs_name,
                    "Model":       model_name,
                    "Metric":      metric,
                    "Mean":        stats["mean"],
                    "SEM":         stats["sem"],
                    "CI95":        stats["ci95"],
                })
            print(f"  {fs_name} | {model_name} | "
                  f"AUC = {summary['AUC']['ci95']}")

    results_df = pd.DataFrame(records)
    out = OUTPUT_DIR / "asymmetry_bootstrap_cv_results.xlsx"
    results_df.to_excel(out, index=False)
    print(f"\n  Saved -> {out}")
    return results_df


# ---------------------------------------------------------------------------
# 3. Predict glaucoma risk in suspects
# ---------------------------------------------------------------------------

def predict_suspect_risk(train_df, suspect_df):
    """Train on full training cohort, predict risk for 1013 suspects.

    Feature set: (δIOP, δCDR, δRNFL, PRS616)

    Returns
    -------
    risk_dict : {model_name: np.ndarray}  predicted probabilities
    """
    print("\nPredicting risk in glaucoma suspects...")

    features = ASYMMETRY_RISK_FEATURES
    missing = [f for f in features if f not in train_df.columns
               or f not in suspect_df.columns]
    if missing:
        print(f"  Cannot predict — columns missing: {missing}")
        return {}

    X_train = train_df[features]
    y_train = train_df[LABEL_COL]

    risk_dict = {}
    for model_name, model in ASYMMETRY_MODELS.items():
        _, y_prob, _ = train_and_predict(
            X_train, y_train, suspect_df[features], model
        )
        risk_dict[model_name] = y_prob

        # Save per-model risk file
        out_df = suspect_df.copy()
        out_df["Predicted_Risk"] = y_prob
        out = OUTPUT_DIR / f"suspect_risks_{model_name.replace(' ', '_')}.xlsx"
        out_df.to_excel(out, index=False)
        print(f"  {model_name} — saved -> {out}")

    return risk_dict


# ---------------------------------------------------------------------------
# 4. Validation plots: trend + ECDF (Figure 5-style)
# ---------------------------------------------------------------------------

def _compute_asymmetry_magnitude(suspect_df):
    """Return mean absolute asymmetry across available delta columns."""
    available = [c for c in DELTA_COLS if c in suspect_df.columns]
    if not available:
        return None
    return suspect_df[available].abs().mean(axis=1)


def plot_trend(suspect_df, risk_prob, model_name, bin_width=0.05):
    """Binned predicted risk vs mean |Δ| ± SEM trend plot.

    Parameters
    ----------
    suspect_df : DataFrame with asymmetry columns
    risk_prob  : np.ndarray  predicted risk probabilities
    model_name : str
    bin_width  : float  risk bin size
    """
    mag = _compute_asymmetry_magnitude(suspect_df)
    if mag is None:
        return None

    df = pd.DataFrame({"Risk": risk_prob, "AsymMag": mag.values}).dropna()
    pearson_r, pearson_p = pearsonr(df["Risk"], df["AsymMag"])

    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    df["Bin"] = pd.cut(df["Risk"], bins, include_lowest=True)
    grouped = (
        df.groupby("Bin")["AsymMag"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped["sem"]    = grouped["std"] / np.sqrt(grouped["count"])
    grouped["center"] = grouped["Bin"].apply(lambda b: b.mid)
    grouped = grouped.dropna(subset=["mean"])

    x_fit = grouped["center"].values
    y_fit = grouped["mean"].values
    if len(x_fit) >= 2:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
        y_line = intercept + slope * x_line
    else:
        x_line = y_line = None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(grouped["center"], grouped["mean"], yerr=grouped["sem"],
                fmt="o", capsize=4, label="Binned mean ± SEM")
    if x_line is not None:
        ax.plot(x_line, y_line, "r--", label="Linear fit")
    ax.set_xlabel("Predicted POAG risk (binned)")
    ax.set_ylabel("Mean |Δ| asymmetry")
    ax.set_title(
        f"{model_name} — predicted risk vs asymmetry magnitude\n"
        f"Pearson r = {pearson_r:.2f},  p = {pearson_p:.2e}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = FIGURE_DIR / f"asymmetry_trend_{model_name.replace(' ', '_')}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Trend plot saved -> {out}")

    return {"Model": model_name, "Plot": "Trend",
            "Pearson_r": round(pearson_r, 3), "Pearson_p": f"{pearson_p:.2e}",
            "N": len(df)}


def plot_ecdf(suspect_df, risk_prob, model_name, quantile=0.25):
    """ECDF of asymmetry magnitude in top vs bottom risk quartile + KS test.

    Parameters
    ----------
    suspect_df : DataFrame
    risk_prob  : np.ndarray
    model_name : str
    quantile   : float  fraction defining top/bottom groups
    """
    mag = _compute_asymmetry_magnitude(suspect_df)
    if mag is None:
        return None

    df = pd.DataFrame({"Risk": risk_prob, "AsymMag": mag.values}).dropna()
    n = len(df)
    k = max(1, int(n * quantile))

    top_idx    = df["Risk"].nlargest(k).index
    bottom_idx = df["Risk"].nsmallest(k).index
    top_vals    = df.loc[top_idx,    "AsymMag"].values
    bottom_vals = df.loc[bottom_idx, "AsymMag"].values

    ks_stat, ks_p = ks_2samp(top_vals, bottom_vals)

    def ecdf(arr):
        x = np.sort(arr)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(*ecdf(top_vals),    label=f"Top {int(quantile*100)}% risk (n={k})",
            color="#d62728")
    ax.plot(*ecdf(bottom_vals), label=f"Bottom {int(quantile*100)}% risk (n={k})",
            color="#1f77b4")
    ax.set_xlabel("Mean |Δ| asymmetry")
    ax.set_ylabel("ECDF")
    ax.set_title(
        f"{model_name} — asymmetry ECDF by risk group\n"
        f"KS stat = {ks_stat:.3f},  p = {ks_p:.2e}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = FIGURE_DIR / f"asymmetry_ecdf_{model_name.replace(' ', '_')}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ECDF plot saved -> {out}")

    return {"Model": model_name, "Plot": "ECDF",
            "KS_stat": round(ks_stat, 3), "KS_p": f"{ks_p:.2e}",
            "N_top": k, "N_bottom": k}


def run_validation_plots(suspect_df, risk_dict):
    """Generate trend + ECDF plots for all models and save stats."""
    print("\nGenerating asymmetry validation plots...")
    stat_rows = []
    for model_name, risk_prob in risk_dict.items():
        row1 = plot_trend(suspect_df, risk_prob, model_name)
        row2 = plot_ecdf(suspect_df,  risk_prob, model_name)
        if row1:
            stat_rows.append(row1)
        if row2:
            stat_rows.append(row2)

    if stat_rows:
        stats_df = pd.DataFrame(stat_rows)
        out = OUTPUT_DIR / "asymmetry_validation_stats.xlsx"
        stats_df.to_excel(out, index=False)
        print(f"  Stats saved -> {out}")


# ---------------------------------------------------------------------------
# 5. AUC bar charts for asymmetry feature sets
# ---------------------------------------------------------------------------

def plot_asymmetry_auc(results_df):
    """Grouped bar chart of AUC for asymmetry feature sets, per model."""
    print("\nGenerating asymmetry AUC figures...")

    # Groups: (delta_measure, PRS sets for that measure)
    groups = {
        DELTA_IOP_COL:  [f"delta_IOP+{p}"
                         for p in ["POAAGG PRS", "MEGA PRS", "PRS526", "PRS616"]],
        DELTA_CDR_COL:  [f"delta_CDR+{p}"
                         for p in ["POAAGG PRS", "MEGA PRS", "PRS526", "PRS616"]],
        DELTA_RNFL_COL: [f"delta_RNFL+{p}"
                         for p in ["POAAGG PRS", "MEGA PRS", "PRS526", "PRS616"]],
    }

    for model_name in ASYMMETRY_MODELS:
        # Build feature order: all 12 combined sets + standalone deltas
        feature_order = (
            ["delta_IOP"] + groups[DELTA_IOP_COL] +
            ["delta_CDR"] + groups[DELTA_CDR_COL] +
            ["delta_RNFL"] + groups[DELTA_RNFL_COL]
        )
        out = FIGURE_DIR / f"auc_asymmetry_{model_name.replace(' ', '_')}.png"
        plot_auc_bars(results_df, model_name, feature_order,
                      output_path=out, ylim=(0.4, 1.0))
        print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_df, suspect_df = load_data()

    # --- Analysis 1: Bootstrap CV with asymmetry features ---
    cv_results = run_asymmetry_bootstrap_cv(train_df)

    # --- Analysis 2: Predict risk in suspects ---
    risk_dict = predict_suspect_risk(train_df, suspect_df)

    # --- Analysis 3: Trend + ECDF validation plots ---
    if risk_dict:
        run_validation_plots(suspect_df, risk_dict)

    # --- Analysis 4: AUC figures ---
    plot_asymmetry_auc(cv_results)

    print("\nAsymmetry analysis complete. Outputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
