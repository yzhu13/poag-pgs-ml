"""
01_main_analysis.py
-------------------
Main ML pipeline for POAG prediction using polygenic risk scores (PGS)
and clinical features in an African ancestry cohort.

Analyses produced
-----------------
1. Bootstrap cross-validation on the 271-sample training cohort
   -> outputs/bootstrap_cv_results.xlsx          (Table 2)

2. External validation on the 1088-sample testing cohort
   -> outputs/external_validation_results.xlsx   (Table 3)

3. Clinical enrichment: predicted risk vs IOP / CDR / RNFL in suspects
   -> outputs/figures/enrichment_*.png
   -> outputs/clinical_enrichment_correlations.xlsx
   -> outputs/predicted_risks_suspects.xlsx

4. AUC bar charts per model and across models
   -> outputs/figures/auc_*.png                  (Figure 2 / Figure 3)

5. Violin plots of PRS distributions by disease status
   -> outputs/figures/violin_prs_*.png           (Figure 2)

Usage
-----
    python 01_main_analysis.py

Edit DATA_DIR and file name constants in config.py before running.

Requirements
------------
See requirements.txt
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from config import (
    TRAIN_FILE, TEST_1088_FILE, TEST_1013_FILE,
    PRS526_LOCI, PRS616_LOCI,
    OUTPUT_DIR, FIGURE_DIR,
    LABEL_COL, STATUS_COL,
    AGE_COL, SEX_COL, POAAGG_PRS, MEGA_PRS, PRS526_COL, PRS616_COL,
    CLINICAL_MARKERS, CLINICAL_YLIMS,
    IOP_SEVERE, CDR_SEVERE, RNFL_SEVERE,
    FEATURE_SETS, ENRICHMENT_FEATURE_SET, PRS_ONLY_SETS,
    MODELS, N_BOOTSTRAPS,
)
from utils import (
    bootstrap_evaluate, train_and_predict, external_test_metrics,
    plot_auc_bars, plot_auc_grouped,
)

sns.set_theme(style="whitegrid", font_scale=1.1)

# ---------------------------------------------------------------------------
# 0. Setup output directories
# ---------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load training and testing cohorts.

    Returns
    -------
    train_df    : 271-sample training cohort
    test_1088   : 1088-sample testing cohort (cases + controls + suspects)
    test_1013   : 1013 glaucoma suspects subset
    """
    print("Loading data...")
    train_df  = pd.read_excel(TRAIN_FILE)
    test_1088 = pd.read_excel(TEST_1088_FILE)
    test_1013 = pd.read_excel(TEST_1013_FILE)

    # Ensure binary label
    train_df = train_df.dropna(subset=[LABEL_COL])
    train_df[LABEL_COL] = train_df[LABEL_COL].astype(int)

    print(f"  Training cohort : {len(train_df)} samples "
          f"({train_df[LABEL_COL].sum()} cases, "
          f"{(train_df[LABEL_COL] == 0).sum()} controls)")
    print(f"  Testing cohort  : {len(test_1088)} samples")
    print(f"  Suspects (1013) : {len(test_1013)} samples")
    return train_df, test_1088, test_1013


# ---------------------------------------------------------------------------
# 2. Bootstrap cross-validation on training cohort
# ---------------------------------------------------------------------------

def run_bootstrap_cv(train_df):
    """Run bootstrap CV for all feature sets and models.

    Returns
    -------
    results_df : DataFrame with columns
                 Feature Set, Model, Metric, Mean, SEM, CI95
    """
    print(f"\nBootstrap CV ({N_BOOTSTRAPS} iterations) on training cohort...")
    records = []

    for fs_name, features in FEATURE_SETS.items():
        missing = [f for f in features if f not in train_df.columns]
        if missing:
            print(f"  Skipping '{fs_name}' — missing columns: {missing}")
            continue
        X = train_df[features]
        y = train_df[LABEL_COL]

        for model_name, model in MODELS.items():
            summary = bootstrap_evaluate(X, y, model, n_bootstraps=N_BOOTSTRAPS)
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
    out = OUTPUT_DIR / "bootstrap_cv_results.xlsx"
    results_df.to_excel(out, index=False)
    print(f"\n  Saved -> {out}")
    return results_df


# ---------------------------------------------------------------------------
# 3. External validation on the 1088-sample testing cohort
# ---------------------------------------------------------------------------

def run_external_validation(train_df, test_1088):
    """Train on 271, test on cases+controls within 1088 cohort.

    Returns
    -------
    external_df : DataFrame with per-(feature-set, model) test metrics
    """
    print("\nExternal validation on 1088-sample testing cohort...")

    # Separate test cases and controls
    test_cc = test_1088[
        test_1088[STATUS_COL].astype(str).str.contains("Case|Control", case=False, na=False)
    ].copy()
    y_test = test_cc[STATUS_COL].apply(
        lambda x: 1 if "Case" in str(x) else 0
    )

    records = []
    for fs_name, features in FEATURE_SETS.items():
        if not set(features).issubset(test_1088.columns):
            print(f"  Skipping '{fs_name}' — features absent in test cohort")
            continue

        X_train = train_df[features]
        y_train = train_df[LABEL_COL]
        X_test  = test_cc[features].replace([np.inf, -np.inf], np.nan)

        for model_name, model in MODELS.items():
            y_pred, y_prob, _ = train_and_predict(X_train, y_train, X_test, model)
            m = external_test_metrics(y_test, y_pred, y_prob)
            m.update({"Feature Set": fs_name, "Model": model_name})
            records.append(m)
            print(f"  {fs_name} | {model_name} | AUC = {m['AUC']:.3f}")

    external_df = pd.DataFrame(records)
    out = OUTPUT_DIR / "external_validation_results.xlsx"
    external_df.to_excel(out, index=False)
    print(f"\n  Saved -> {out}")
    return external_df


# ---------------------------------------------------------------------------
# 4. Clinical enrichment: predicted risk vs clinical severity in suspects
# ---------------------------------------------------------------------------

def run_clinical_enrichment(train_df, test_1013):
    """Correlate model-predicted POAG risk with optic nerve damage markers.

    Uses the ENRICHMENT_FEATURE_SET (base+PRS616) trained on the 271-sample
    cohort to predict risk probabilities for glaucoma suspects, then
    correlates those probabilities with IOP_SEVERE, CDR_SEVERE, RNFL_SEVERE.

    Outputs
    -------
    - outputs/figures/enrichment_{model}_{marker}.png
    - outputs/clinical_enrichment_correlations.xlsx
    - outputs/predicted_risks_suspects.xlsx
    """
    print(f"\nClinical enrichment analysis (feature set: {ENRICHMENT_FEATURE_SET})...")

    features = FEATURE_SETS[ENRICHMENT_FEATURE_SET]
    missing  = [f for f in features if f not in test_1013.columns]
    if missing:
        print(f"  Cannot run enrichment — columns missing in suspect data: {missing}")
        return

    X_train = train_df[features]
    y_train = train_df[LABEL_COL]

    risk_df    = test_1013.copy()
    corr_rows  = []

    for model_name, model in MODELS.items():
        _, y_prob, _ = train_and_predict(X_train, y_train, test_1013[features], model)
        risk_col = f"{model_name}_Risk"
        risk_df[risk_col] = y_prob

        for marker_label in CLINICAL_MARKERS.values():
            if marker_label not in test_1013.columns:
                continue
            valid = test_1013[[*features]].copy()
            valid[risk_col]    = y_prob
            valid[marker_label] = test_1013[marker_label].values
            valid = valid[[risk_col, marker_label]].dropna()

            if len(valid) < 10:
                continue

            x = valid[risk_col].astype(float)
            y = valid[marker_label].astype(float)
            pearson_r, pearson_p   = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)

            corr_rows.append({
                "Model":       model_name,
                "Feature Set": ENRICHMENT_FEATURE_SET,
                "Marker":      marker_label,
                "Pearson_r":   round(pearson_r, 3),
                "Pearson_p":   f"{pearson_p:.2e}",
                "Spearman_r":  round(spearman_r, 3),
                "Spearman_p":  f"{spearman_p:.2e}",
                "N":           len(valid),
            })
            print(f"  {model_name} | {marker_label} | "
                  f"r={pearson_r:.2f}, p={pearson_p:.1e}")

            # Scatter + regression plot
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.4}, ax=ax)
            ax.set_xlabel(f"{model_name} ({ENRICHMENT_FEATURE_SET}) predicted risk")
            ylabel = marker_label
            if "RNFL" in marker_label:
                ylabel += " (µm)"
            elif "IOP" in marker_label:
                ylabel += " (mmHg)"
            ax.set_ylabel(ylabel)
            if marker_label in CLINICAL_YLIMS:
                ax.set_ylim(*CLINICAL_YLIMS[marker_label])
            ax.set_title(
                f"Pearson r={pearson_r:.2f}, p={pearson_p:.1e}\n"
                f"Spearman ρ={spearman_r:.2f}, p={spearman_p:.1e}"
            )
            plt.tight_layout()
            fig_path = FIGURE_DIR / f"enrichment_{model_name.replace(' ', '_')}_{marker_label}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    # Save correlations table
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_excel(OUTPUT_DIR / "clinical_enrichment_correlations.xlsx", index=False)

    # Save predicted risks for suspects
    risk_df.to_excel(OUTPUT_DIR / "predicted_risks_suspects.xlsx", index=False)

    print(f"  Saved -> {OUTPUT_DIR / 'clinical_enrichment_correlations.xlsx'}")
    print(f"  Saved -> {OUTPUT_DIR / 'predicted_risks_suspects.xlsx'}")
    return risk_df, corr_df


# ---------------------------------------------------------------------------
# 5. AUC visualizations
# ---------------------------------------------------------------------------

def plot_auc_figures(results_df):
    """Generate AUC bar charts for the paper figures.

    Figure 2 (top row): base vs base+{each PRS} for RF, MLP, SVM separately.
    Figure 3 (grouped): four PRS-only feature sets, all models side by side.
    """
    print("\nGenerating AUC figures...")

    selected_fs = ["base", "base+POAAGG PRS", "base+MEGA PRS",
                   "base+PRS526", "base+PRS616"]

    # Per-model bar charts
    for model_name in ["Random Forest", "MLP", "SVM"]:
        out = FIGURE_DIR / f"auc_{model_name.replace(' ', '_')}_base_plus_prs.png"
        plot_auc_bars(results_df, model_name, selected_fs, output_path=out)
        print(f"  Saved -> {out}")

    # Grouped: 4 PRS-only sets, all models + average
    prs_df = results_df[results_df["Feature Set"].isin(PRS_ONLY_SETS)].copy()

    # Compute average across RF / MLP / SVM
    avg_rows = []
    for fs in PRS_ONLY_SETS:
        sub = prs_df[
            (prs_df["Feature Set"] == fs) &
            (prs_df["Metric"] == "AUC") &
            (prs_df["Model"].isin(["Random Forest", "MLP", "SVM"]))
        ]
        if sub.empty:
            continue
        avg_rows.append({
            "Feature Set": fs,
            "Model":       "Average",
            "Metric":      "AUC",
            "Mean":        round(sub["Mean"].mean(), 4),
            "SEM":         round(np.sqrt((sub["SEM"] ** 2).sum()) / len(sub), 4),
        })
    if avg_rows:
        prs_df = pd.concat([prs_df, pd.DataFrame(avg_rows)], ignore_index=True)

    out = FIGURE_DIR / "auc_4prs_all_models.png"
    plot_auc_grouped(prs_df, PRS_ONLY_SETS,
                     ["Random Forest", "MLP", "SVM", "Average"],
                     output_path=out)
    print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# 6. Violin plots — PRS distributions by disease status
# ---------------------------------------------------------------------------

def plot_prs_violins(test_1088):
    """Violin plots of each PRS score split by Case / Control / Suspect."""
    print("\nGenerating PRS violin plots...")

    prs_cols = [POAAGG_PRS, MEGA_PRS, PRS526_COL, PRS616_COL]
    existing = [c for c in prs_cols if c in test_1088.columns]
    if not existing:
        print("  No PRS columns found in testing cohort — skipping violin plots.")
        return

    df = test_1088[existing + [STATUS_COL]].copy()
    df[STATUS_COL] = df[STATUS_COL].astype(str).str.strip()
    df = df[df[STATUS_COL].isin(["Case", "Control", "Suspect"])]

    palette = {"Case": "#d62728", "Control": "#1f77b4", "Suspect": "#ff7f0e"}

    for prs in existing:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.violinplot(
            data=df, x=STATUS_COL, y=prs,
            order=["Control", "Suspect", "Case"],
            palette=palette, inner="quartile", ax=ax,
        )
        ax.set_xlabel("Disease Status")
        ax.set_ylabel(prs)
        ax.set_title(f"{prs} Distribution by Disease Status")
        plt.tight_layout()
        out = FIGURE_DIR / f"violin_{prs.replace(' ', '_')}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_df, test_1088, test_1013 = load_data()

    # --- Analysis 1: Bootstrap CV ---
    cv_results = run_bootstrap_cv(train_df)

    # --- Analysis 2: External validation ---
    ext_results = run_external_validation(train_df, test_1088)

    # --- Analysis 3: Clinical enrichment ---
    suspects = test_1088[
        test_1088[STATUS_COL].astype(str).str.contains("Suspect", case=False, na=False)
    ].copy()
    run_clinical_enrichment(train_df, suspects)

    # --- Analysis 4: AUC figures ---
    plot_auc_figures(cv_results)

    # --- Analysis 5: PRS violin plots ---
    plot_prs_violins(test_1088)

    print("\nAll analyses complete. Outputs saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
