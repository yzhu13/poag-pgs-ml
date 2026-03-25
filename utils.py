"""
utils.py
--------
Shared helper functions used across analysis scripts.

Functions
---------
build_pipeline          Build a sklearn Pipeline (impute → scale → clf).
bootstrap_evaluate      Bootstrap cross-validation returning mean ± SEM metrics.
train_and_predict       Fit a pipeline on training data, return predictions on test data.
plot_auc_bars           Grouped bar chart of AUC by feature set for a single model.
plot_auc_grouped        Side-by-side bar chart comparing multiple models across PRS sets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(model):
    """Return a Pipeline: mean imputation -> standard scaling -> classifier.

    Parameters
    ----------
    model : sklearn estimator (unfitted)

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("clf",     deepcopy(model)),
    ])


# ---------------------------------------------------------------------------
# Bootstrap cross-validation
# ---------------------------------------------------------------------------

def bootstrap_evaluate(X, y, model, n_bootstraps=10, test_size=0.2):
    """Bootstrap cross-validation for a single feature set + model combination.

    Each iteration: resample with replacement (stratified) -> 80/20 split ->
    fit on train -> evaluate on held-out test.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary labels (0/1).
    model : sklearn estimator
        Unfitted model; a deep copy is made each iteration.
    n_bootstraps : int
        Number of resampling iterations.
    test_size : float
        Fraction of resampled data held out for evaluation.

    Returns
    -------
    dict
        Keys: "Accuracy", "F1", "AUC", "Sensitivity", "Specificity".
        Values: dicts with "mean", "std", "sem", "ci95".
    """
    metrics = {k: [] for k in ("Accuracy", "F1", "AUC", "Sensitivity", "Specificity")}

    for _ in range(n_bootstraps):
        X_rs, y_rs = resample(X, y, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_rs, y_rs, test_size=test_size, stratify=y_rs, random_state=None
        )
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics["Accuracy"].append(report["accuracy"])
        metrics["F1"].append(report.get("1", {}).get("f1-score", np.nan))
        metrics["Sensitivity"].append(report.get("1", {}).get("recall", np.nan))
        metrics["Specificity"].append(report.get("0", {}).get("recall", np.nan))
        metrics["AUC"].append(roc_auc_score(y_test, y_prob))

    summary = {}
    for k, vals in metrics.items():
        arr  = np.array(vals, dtype=float)
        mean = np.nanmean(arr)
        std  = np.nanstd(arr)
        sem  = std / np.sqrt(len(arr))
        summary[k] = {
            "mean":  round(mean, 4),
            "std":   round(std,  4),
            "sem":   round(sem,  4),
            "ci95":  f"{mean:.3f} ± {1.96 * sem:.3f}",
        }
    return summary


# ---------------------------------------------------------------------------
# Fit on full training set, predict on external test set
# ---------------------------------------------------------------------------

def train_and_predict(X_train, y_train, X_test, model):
    """Fit pipeline on training data and return predictions for test data.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_test           : test features
    model            : unfitted sklearn estimator

    Returns
    -------
    y_pred : np.ndarray  (hard predictions)
    y_prob : np.ndarray  (probability of class 1)
    pipe   : fitted Pipeline
    """
    pipe = build_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return y_pred, y_prob, pipe


def external_test_metrics(y_test, y_pred, y_prob):
    """Compute AUC, accuracy, sensitivity, specificity, F1 for external test.

    Returns
    -------
    dict
    """
    return {
        "AUC":         round(roc_auc_score(y_test, y_prob), 4),
        "Accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "Sensitivity": round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "Specificity": round(recall_score(y_test, y_pred, pos_label=0, zero_division=0), 4),
        "F1":          round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_auc_bars(results_df, model_name, feature_order, output_path=None,
                  ylim=(0.6, 1.0)):
    """Bar chart of AUC (mean ± SEM) for a single model across feature sets.

    Parameters
    ----------
    results_df   : DataFrame with columns Feature Set, Model, Metric, Mean, SEM
    model_name   : str  (must match a value in results_df["Model"])
    feature_order: list of str  (x-axis ordering)
    output_path  : Path or str  (if given, figure is saved here)
    ylim         : tuple  (y-axis limits)
    """
    df = results_df[
        (results_df["Metric"] == "AUC") &
        (results_df["Model"] == model_name) &
        (results_df["Feature Set"].isin(feature_order))
    ].set_index("Feature Set").reindex(feature_order).reset_index()

    x    = np.arange(len(df))
    y    = df["Mean"].values
    yerr = df["SEM"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, y, yerr=yerr, capsize=4, color="orange", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Feature Set"], rotation=30, ha="right")
    ax.set_ylim(*ylim)
    ax.set_ylabel("AUC")
    ax.set_title(f"{model_name} — AUC (mean ± SEM)")
    pad = (ylim[1] - ylim[0]) * 0.02
    for i, (val, err) in enumerate(zip(y, yerr)):
        ax.text(i, val + err + pad, f"{val:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_auc_grouped(results_df, feature_order, model_names, output_path=None,
                     ylim=(0.5, 1.0)):
    """Side-by-side grouped bar chart comparing models across feature sets.

    Parameters
    ----------
    results_df   : DataFrame with columns Feature Set, Model, Metric, Mean, SEM
    feature_order: list of str  (feature sets to include, in order)
    model_names  : list of str  (models to include)
    output_path  : Path or str
    ylim         : tuple
    """
    df = results_df[
        (results_df["Metric"] == "AUC") &
        (results_df["Feature Set"].isin(feature_order)) &
        (results_df["Model"].isin(model_names))
    ].copy()

    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#9c9c9c"]
    n_models   = len(model_names)
    n_features = len(feature_order)
    width  = 0.8 / n_models
    x      = np.arange(n_features)
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(max(10, n_features * 1.5), 5))
    for i, (model, offset, color) in enumerate(zip(model_names, offsets, palette)):
        sub = df[df["Model"] == model].set_index("Feature Set").reindex(feature_order)
        y    = sub["Mean"].values.astype(float)
        yerr = sub["SEM"].values.astype(float)
        bars = ax.bar(x + offset, y, yerr=yerr, capsize=3, width=width,
                      label=model, color=color)
        pad = (ylim[1] - ylim[0]) * 0.015
        for xi, (val, err) in enumerate(zip(y, yerr)):
            if np.isfinite(val):
                ax.text(xi + offset, val + err + pad, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_order, rotation=30, ha="right")
    ax.set_ylim(*ylim)
    ax.set_ylabel("AUC (mean ± SEM)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
