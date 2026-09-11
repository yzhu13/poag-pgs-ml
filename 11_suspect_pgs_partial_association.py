# =============================================================
#  POAG — Suspect Cohort: PGS616 vs CDR / RNFL, adjusted for age and sex
#  iScience R4 revision (2026-09-04)
#
#  Addresses Editor point 2 and Reviewer #1 point 2:
#   "Test the association of PGS616 with CDR and RNFL after adjustment
#    for age and sex, or directly compare Base versus Base+PGS616
#    performance in the suspect cohort."
#   "The model includes age and sex, which themselves are strongly
#    related to glaucoma-related phenotypes. Therefore, these
#    associations should not be described simply as validation of
#    'genetic risk'."
#
#  R3 reported Pearson r between MLP(Base+PGS616) PREDICTED RISK and
#  each phenotype (CDR r=0.121, RNFL r=-0.205).  Because the predictor
#  contains age and sex, that correlation mostly re-expresses the age
#  effect.  This script separates the two questions:
#
#  ANALYSIS A — is the PGS ITSELF associated with the phenotype?
#      phenotype ~ PGS616 + Age + Sex        (OLS)
#      report beta, 95% CI, p, and the partial correlation of PGS616
#      with the phenotype given age and sex.  Unadjusted Pearson r is
#      shown alongside so the shrinkage is visible.
#
#  ANALYSIS B — does adding PGS616 to the model change what the
#      predicted risk explains?
#      compare predicted risk from Base vs Base+PGS616 (same
#      classifier), correlation with each phenotype, and the nested
#      model gain  phenotype ~ age + sex  vs  + PGS616
#      (delta R-squared, partial F test).
#
#  Outputs:
#    outputs/tables/Table_Suspect_PGS_Adjusted.xlsx
#      A_PartialAssociation   per phenotype: crude vs adjusted
#      B_BaseVsPGS            predicted-risk comparison + nested test
#      C_Notes                wording guidance for Results/Discussion
# =============================================================

import os as _os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
from poag_corrections import (int_pgs616, int_pgs616_training_only,
                              restrict_pmbb_age)   # 2026-09-05 audit
OUT_XL = _os.path.join(HERE, "outputs", "tables")
_os.makedirs(OUT_XL, exist_ok=True)

# Data location. Set POAG_DATA_DIR to the folder holding the cohort
# subdirectories; it defaults to ./data next to this script. The data
# themselves are under controlled access (see data/README.md).
DATA_DIR = _os.environ.get(
    "POAG_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   "data"))
TRAIN_F = _os.path.join(DATA_DIR, "POAAGG_cohort",
                        "271_training_cohort_4_new_PRS_cleaned.xlsx")
SUSP_F = _os.path.join(DATA_DIR, "POAAGG_cohort",
                       "1013_testing_cohort_only_suspect_cleaned.xlsx")

LABEL = "CaseCtrl"
SEED = 42
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
PRIMARY = "MLP"          # primary model used in Figure 4 / Results

OUTCOMES = {
    "IOP (mmHg)": "IOP_SEVERE",
    "CDR":        "CDR_SEVERE",
    "RNFL (um)":  "RNFL_SEVERE",
}


def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    if name == "LR":
        steps += [("clf", LogisticRegression(max_iter=1000,
            class_weight="balanced", random_state=SEED))]
    elif name == "SVM":
        steps += [("clf", SVC(kernel="rbf", probability=True,
            class_weight="balanced", random_state=SEED))]
    elif name == "RF":
        steps += [("clf", RandomForestClassifier(n_estimators=200,
            max_depth=5, class_weight="balanced", random_state=SEED))]
    elif name == "MLP":
        steps += [("clf", MLPClassifier(hidden_layer_sizes=(32,),
            max_iter=1000, early_stopping=False, random_state=SEED))]
    return Pipeline(steps)


# ═══════════════════════════════════════════════════════════════
#  OLS helper (no statsmodels dependency)
# ═══════════════════════════════════════════════════════════════
def ols(X, yv):
    """Least squares with intercept. Returns beta, se, t, p, r2, n, k."""
    n = len(yv)
    Xd = np.column_stack([np.ones(n), X])
    k = Xd.shape[1]
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    resid = yv - Xd @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    return beta, se, t, p, r2, n, k


def partial_r(x, yv, Z):
    """Partial correlation of x with y controlling for columns of Z."""
    def resid_of(v):
        Zd = np.column_stack([np.ones(len(v)), Z])
        b, *_ = np.linalg.lstsq(Zd, v, rcond=None)
        return v - Zd @ b
    rx, ry = resid_of(x), resid_of(yv)
    r, _ = stats.pearsonr(rx, ry)
    n, k = len(x), Z.shape[1]
    dof = n - k - 2
    if dof <= 0 or abs(r) >= 1:
        return r, np.nan, dof
    tstat = r * np.sqrt(dof / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(tstat), dof), dof


# ═══════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════
print("Loading data ...", flush=True)
tr = pd.read_excel(TRAIN_F)
su = pd.read_excel(SUSP_F)
tr, su = int_pgs616(tr, su)
y_tr = tr[LABEL].values.astype(int)
print(f"  Train N={len(tr)}   Suspects N={len(su)}", flush=True)
for lab, col in OUTCOMES.items():
    print(f"  {lab:12s} non-missing: {su[col].notna().sum()}", flush=True)


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS A — PGS616 vs phenotype, adjusted for age and sex
# ═══════════════════════════════════════════════════════════════
print("\n=== A. PGS616 -> phenotype, adjusted for age + sex ===", flush=True)
rowsA = []
for lab, col in OUTCOMES.items():
    m = su[[col, "PGS616", "PGS526", "Age", "Gender"]].dropna()
    yv = m[col].values.astype(float)
    age = m["Age"].values.astype(float)
    sex = m["Gender"].values.astype(float)

    for pgs_name in ["PGS616", "PGS526"]:
        g = m[pgs_name].values.astype(float)

        # crude
        r_crude, p_crude = stats.pearsonr(g, yv)

        # adjusted: phenotype ~ PGS + age + sex
        # The PGS is standardised within the analysis sample so that the
        # reported coefficient really is per standard deviation. Before
        # 2026-09-05 the raw column went in unchanged while the column was
        # headed "per SD"; p, partial r, R-squared and the partial F test
        # are invariant to this rescaling, so only the coefficient and its
        # interval change.
        g = (g - g.mean()) / g.std(ddof=0)
        X = np.column_stack([g, age, sex])
        beta, se, t, p, r2, n, k = ols(X, yv)
        b_pgs, se_pgs, p_pgs = beta[1], se[1], p[1]
        crit = stats.t.ppf(0.975, n - k)
        lo, hi = b_pgs - crit * se_pgs, b_pgs + crit * se_pgs

        # partial correlation given age + sex
        pr, pr_p, _ = partial_r(g, yv, np.column_stack([age, sex]))

        # nested model gain from adding the PGS
        _, _, _, _, r2_base, _, k0 = ols(np.column_stack([age, sex]), yv)
        f_stat = ((r2 - r2_base) / 1) / ((1 - r2) / (n - k))
        f_p = stats.f.sf(f_stat, 1, n - k)

        rowsA.append({
            "Phenotype": lab,
            "PGS": pgs_name,
            "N": n,
            "Crude Pearson r": round(r_crude, 4),
            "Crude p": f"{p_crude:.3g}",
            "Adjusted beta (per SD)": round(b_pgs, 4),
            "Adjusted 95% CI": f"({lo:+.4f}, {hi:+.4f})",
            "Adjusted p": f"{p_pgs:.3g}",
            "Partial r (| age,sex)": round(pr, 4),
            "Partial p": f"{pr_p:.3g}",
            "R2 age+sex": round(r2_base, 4),
            "R2 age+sex+PGS": round(r2, 4),
            "delta R2": round(r2 - r2_base, 5),
            "Partial F p": f"{f_p:.3g}",
        })
        print(f"  {lab:12s} {pgs_name}: crude r={r_crude:+.3f} -> "
              f"partial r={pr:+.3f} (p={pr_p:.3g}), dR2={r2-r2_base:.5f}",
              flush=True)
sheetA = pd.DataFrame(rowsA)


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS B — predicted risk: Base vs Base+PGS616
# ═══════════════════════════════════════════════════════════════
print("\n=== B. Predicted risk, Base vs Base+PGS616 ===", flush=True)
FS = {
    "Base":        ["Age", "Gender"],
    "Base+PGS616": ["Age", "Gender", "PGS616"],
}
pred = {}
for fs, cols in FS.items():
    for mn in MODEL_NAMES:
        pipe = make_pipeline(mn)
        pipe.fit(tr[cols].values, y_tr)
        pred[(fs, mn)] = pipe.predict_proba(su[cols].values)[:, 1]

rowsB = []
for mn in MODEL_NAMES:
    for lab, col in OUTCOMES.items():
        ok = su[col].notna().values
        yv = su[col].values[ok].astype(float)
        r_base, p_base = stats.pearsonr(pred[("Base", mn)][ok], yv)
        r_pgs, p_pgs = stats.pearsonr(pred[("Base+PGS616", mn)][ok], yv)
        rowsB.append({
            "Classifier": mn,
            "Phenotype": lab,
            "N": int(ok.sum()),
            "r (Base)": round(r_base, 4),
            "p (Base)": f"{p_base:.3g}",
            "r (Base+PGS616)": round(r_pgs, 4),
            "p (Base+PGS616)": f"{p_pgs:.3g}",
            "delta r": round(r_pgs - r_base, 4),
            "Primary model": "yes" if mn == PRIMARY else "",
        })
        if mn == PRIMARY:
            print(f"  {lab:12s} r(Base)={r_base:+.3f}  "
                  f"r(Base+PGS616)={r_pgs:+.3f}  delta={r_pgs-r_base:+.4f}",
                  flush=True)
sheetB = pd.DataFrame(rowsB)


# ═══════════════════════════════════════════════════════════════
#  NOTES
# ═══════════════════════════════════════════════════════════════
notes = pd.DataFrame({"Note": [
    "Table: Suspect cohort (N=1,013) — association of PGS with ocular "
    "phenotypes after adjustment for age and sex.",
    "",
    "PANEL A. Each ocular phenotype was regressed on the PGS together with "
    "age and sex (ordinary least squares). Reported are the crude Pearson "
    "correlation of the PGS with the phenotype, the adjusted regression "
    "coefficient with 95% confidence interval, the partial correlation of "
    "the PGS with the phenotype given age and sex, and the incremental "
    "variance explained (delta R-squared) with its partial F test. PGS were "
    "standardised, so coefficients are per standard deviation.",
    "",
    "PANEL B. Predicted POAG risk was generated in the suspect cohort from "
    "models trained in the labelled cohort (N=271) using age and sex alone "
    "(Base) and using age, sex and PGS616 (Base+PGS616), and correlated with "
    "each phenotype. The difference between the two correlations isolates "
    "the contribution of the PGS to the predicted-risk association, which the "
    "crude correlation of the Base+PGS616 predictions confounds with age.",
    "",
    "INTERPRETATION. Correlations between predicted risk and ocular "
    "phenotypes reported previously were obtained from a model containing age "
    "and sex, and therefore cannot be attributed to genetic risk. Panel A "
    "tests the genetic component directly; Panel B quantifies how much of the "
    "predicted-risk association is attributable to adding the PGS.",
]})

with pd.ExcelWriter(_os.path.join(OUT_XL, "Table_Suspect_PGS_Adjusted.xlsx"),
                    engine="openpyxl") as w:
    sheetA.to_excel(w, sheet_name="A_PartialAssociation", index=False)
    sheetB.to_excel(w, sheet_name="B_BaseVsPGS", index=False)
    notes.to_excel(w, sheet_name="C_Notes", index=False)

print("\nSaved Table_Suspect_PGS_Adjusted.xlsx", flush=True)
print("\nAll done.", flush=True)
