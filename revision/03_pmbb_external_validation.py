"""
PMBB External Validation Analysis
Manuscript: ISCIENCE-D-26-03991
Date: 2026-05-07

PGS616 (617 SNPs, MEGA African-ancestry GWAS) and
PGS526 (527 SNPs, quantitative MTAG meta-analysis)
for POAG prediction — external validation in the
Penn Medicine Biobank (PMBB) Release 3.0, AFR cohort.

Expected input files (in ../input date/):
  PMBB_AFR_ready_for_PGS.csv
  PMBBv3_GRS_MEGA_616snps.sscore_withSTDscore.txt
  PMBBv3_GRS_QUANT_526snps.sscore_withSTDscore.txt

Outputs (saved alongside this script in revision/):
  PMBB_validation_AUC_summary.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# Paths  (relative to this script in revision/)
# -----------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent      # .../poag-pgs-ml/revision/
REPO     = _HERE.parent                         # .../poag-pgs-ml/
DATA_DIR = REPO / "input date"
OUT_DIR  = _HERE

# -----------------------------------------------------------------
# 1. Load and merge data
# -----------------------------------------------------------------
pheno = pd.read_csv(DATA_DIR / "PMBB_AFR_ready_for_PGS.csv")

pgs616 = pd.read_csv(
    DATA_DIR / "PMBBv3_GRS_MEGA_616snps.sscore_withSTDscore.txt",
    sep="\t"
)[["IID", "SCORE1_AVG_STD"]].rename(
    columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": "PGS616"}
)

pgs526 = pd.read_csv(
    DATA_DIR / "PMBBv3_GRS_QUANT_526snps.sscore_withSTDscore.txt",
    sep="\t"
)[["IID", "SCORE1_AVG_STD"]].rename(
    columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": "PGS526"}
)

df = (
    pheno
    .merge(pgs616, on="PMBB_ID", how="inner")
    .merge(pgs526, on="PMBB_ID", how="inner")
)
df = df.dropna(subset=["POAG_cases", "Age", "sex_bin", "PGS616", "PGS526"])
df["POAG"] = df["POAG_cases"].astype(int)

print("=" * 60)
print("PMBB AFR External Validation — Sample Descriptives")
print("=" * 60)
n_total = len(df)
n_cases = df["POAG"].sum()
n_ctrl  = n_total - n_cases
print(f"Total N:   {n_total}")
print(f"Cases:     {n_cases} ({100*n_cases/n_total:.1f}%)")
print(f"Controls:  {n_ctrl}  ({100*n_ctrl/n_total:.1f}%)")
print(f"\nAge (mean+/-SD): {df['Age'].mean():.1f} +/- {df['Age'].std():.1f}")
print(f"  Cases:    {df.loc[df.POAG==1,'Age'].mean():.1f} +/- {df.loc[df.POAG==1,'Age'].std():.1f}")
print(f"  Controls: {df.loc[df.POAG==0,'Age'].mean():.1f} +/- {df.loc[df.POAG==0,'Age'].std():.1f}")
print(f"\nMale: {df['sex_bin'].sum()} ({100*df['sex_bin'].mean():.1f}%)")
print(f"\nPGS616 (mean+/-SD): {df['PGS616'].mean():.3f} +/- {df['PGS616'].std():.3f}")
print(f"  Cases:    {df.loc[df.POAG==1,'PGS616'].mean():.3f} +/- {df.loc[df.POAG==1,'PGS616'].std():.3f}")
print(f"  Controls: {df.loc[df.POAG==0,'PGS616'].mean():.3f} +/- {df.loc[df.POAG==0,'PGS616'].std():.3f}")
print(f"\nPGS526 (mean+/-SD): {df['PGS526'].mean():.3f} +/- {df['PGS526'].std():.3f}")
print(f"  Cases:    {df.loc[df.POAG==1,'PGS526'].mean():.3f} +/- {df.loc[df.POAG==1,'PGS526'].std():.3f}")
print(f"  Controls: {df.loc[df.POAG==0,'PGS526'].mean():.3f} +/- {df.loc[df.POAG==0,'PGS526'].std():.3f}")


# -----------------------------------------------------------------
# 2. Logistic regression — OR per SD for each PGS (adjusted Age+Sex)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("Logistic Regression: OR per SD for PGS (adjusted Age+Sex)")
print("=" * 60)

for pgs_col, label in [("PGS616", "PGS616 (MEGA AFR)"), ("PGS526", "PGS526 (QUANT MTAG)")]:
    formula = f"POAG ~ Age + sex_bin + {pgs_col}"
    model   = smf.logit(formula, data=df).fit(disp=False)
    coef    = model.params[pgs_col]
    se      = model.bse[pgs_col]
    pval    = model.pvalues[pgs_col]
    OR      = np.exp(coef)
    CI_lo   = np.exp(coef - 1.96 * se)
    CI_hi   = np.exp(coef + 1.96 * se)
    print(f"\n{label}:")
    print(f"  OR per SD = {OR:.3f} (95% CI: {CI_lo:.3f}-{CI_hi:.3f}), p = {pval:.4f}")


# -----------------------------------------------------------------
# 3. AUC with 95% CI via bootstrap
# -----------------------------------------------------------------
def bootstrap_auc(y, probs, n_boot=1000, seed=42):
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0 or y[idx].sum() == len(y[idx]):
            continue
        aucs.append(roc_auc_score(y[idx], probs[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def fit_predict(formula, df):
    m = smf.logit(formula, data=df).fit(disp=False)
    return m.predict(df)


models = {
    "Age + Sex (baseline)":        "POAG ~ Age + sex_bin",
    "Age + Sex + PGS616":          "POAG ~ Age + sex_bin + PGS616",
    "Age + Sex + PGS526":          "POAG ~ Age + sex_bin + PGS526",
    "Age + Sex + PGS616 + PGS526": "POAG ~ Age + sex_bin + PGS616 + PGS526",
}

y_arr = df["POAG"].values
preds = {}

print("\n" + "=" * 60)
print("AUC Results (with 95% bootstrap CI, 1000 iterations)")
print("=" * 60)

for name, formula in models.items():
    probs = fit_predict(formula, df).values
    auc   = roc_auc_score(y_arr, probs)
    lo, hi = bootstrap_auc(y_arr, probs)
    preds[name] = probs
    print(f"\n{name}:")
    print(f"  AUC = {auc:.4f} (95% CI: {lo:.4f}-{hi:.4f})")


# -----------------------------------------------------------------
# 4. DeLong test for AUC comparison
# -----------------------------------------------------------------
def delong_auc_var(y_true, y_pred):
    """Returns (AUC, variance) using DeLong structural components."""
    pos  = y_pred[y_true == 1]
    neg  = y_pred[y_true == 0]
    m, n = len(pos), len(neg)
    auc  = roc_auc_score(y_true, y_pred)
    v10  = np.array([np.mean(pi > neg) for pi in pos])
    v01  = np.array([np.mean(pos > ni) for ni in neg])
    var  = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    return auc, var


def delong_compare(y, p1, p2):
    """DeLong test comparing two correlated ROC curves."""
    auc1, var1 = delong_auc_var(y, p1)
    auc2, var2 = delong_auc_var(y, p2)
    pos1, neg1 = p1[y == 1], p1[y == 0]
    pos2, neg2 = p2[y == 1], p2[y == 0]
    m, n = (y == 1).sum(), (y == 0).sum()
    v10_1 = np.array([np.mean(pos1[i] > neg1) for i in range(m)])
    v10_2 = np.array([np.mean(pos2[i] > neg2) for i in range(m)])
    v01_1 = np.array([np.mean(pos1 > neg1[j]) for j in range(n)])
    v01_2 = np.array([np.mean(pos2 > neg2[j]) for j in range(n)])
    cov   = np.cov(v10_1, v10_2)[0, 1] / m + np.cov(v01_1, v01_2)[0, 1] / n
    se_diff = np.sqrt(var1 + var2 - 2 * cov)
    z = (auc1 - auc2) / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, auc1 - auc2, z, p


print("\n" + "=" * 60)
print("DeLong Test: PGS models vs. Age+Sex baseline")
print("=" * 60)

p_base = preds["Age + Sex (baseline)"]
comparisons = [
    ("Age + Sex + PGS616",          "PGS616"),
    ("Age + Sex + PGS526",          "PGS526"),
    ("Age + Sex + PGS616 + PGS526", "PGS616+PGS526"),
]
for model_name, label in comparisons:
    p_new = preds[model_name]
    auc1, auc2, diff, z, p = delong_compare(y_arr, p_new, p_base)
    direction = "improvement" if diff > 0 else "decrease"
    print(f"\n{label} vs. baseline:")
    print(f"  dAUC = {diff:+.4f}  (z = {z:.3f}, p = {p:.4f})  [{direction}]")


# -----------------------------------------------------------------
# 5. PGS mean difference: cases vs. controls (t-test + Cohen's d)
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("PGS Mean Difference: Cases vs. Controls")
print("=" * 60)

for pgs_col, label in [("PGS616", "PGS616"), ("PGS526", "PGS526")]:
    cases = df.loc[df.POAG == 1, pgs_col]
    ctrls = df.loc[df.POAG == 0, pgs_col]
    t, p  = stats.ttest_ind(cases, ctrls)
    d     = (cases.mean() - ctrls.mean()) / df[pgs_col].std()
    print(f"\n{label}: cases {cases.mean():.3f} vs controls {ctrls.mean():.3f}")
    print(f"  t = {t:.3f}, p = {p:.4f}, Cohen's d = {d:.3f}")


# -----------------------------------------------------------------
# 6. Save AUC summary
# -----------------------------------------------------------------
summary_rows = []
for name, formula in models.items():
    probs = preds[name]
    auc   = roc_auc_score(y_arr, probs)
    lo, hi = bootstrap_auc(y_arr, probs)
    summary_rows.append({
        "Model":  name,
        "AUC":    round(auc, 4),
        "CI_lo":  round(lo, 4),
        "CI_hi":  round(hi, 4),
    })

out_path = OUT_DIR / "PMBB_validation_AUC_summary.csv"
pd.DataFrame(summary_rows).to_csv(out_path, index=False)
print(f"\n\nSummary saved to: {out_path}")
print("\nDone.")
