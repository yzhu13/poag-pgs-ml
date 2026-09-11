# =============================================================
#  POAG — Statistical significance vs clinical utility (PMBB)
#  iScience R4 revision (2026-09-04)
#
#  Addresses Editor point 9 and Reviewer #1 point 3:
#   "Revise the interpretation to distinguish statistical improvement
#    from clinically meaningful benefit and to acknowledge the limited
#    incremental value of PGS616 beyond age and sex."
#
#  The external increment is delta-AUC = +0.010 (DeLong p < 0.001):
#  statistically significant in N=9,817, but AUC is not a clinical
#  quantity.  This script translates it into screening terms that a
#  clinician can weigh:
#
#    - sensitivity at FIXED specificity (90% and 95%), Base vs Base+PGS616
#    - cases detected and missed per 10,000 people screened
#    - number of additional cases found by adding the PGS
#    - paired bootstrap CI on every difference
#
#  The operating points are chosen on the SAME cohort for both models,
#  so the comparison is like-for-like; bootstrap resampling is at the
#  participant level and paired (both models scored on the same resample).
#
#  Outputs:
#    outputs/tables/Table_ClinicalUtility_PMBB.xlsx
#      A_FixedSpecificity   sens/PPV/counts at each operating point
#      B_Differences        paired differences with 95% CI
#      C_Notes
#
#  Usage:  python 13_clinical_utility_pmbb.py [n_boot]
# =============================================================

import os as _os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
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
PMBB_PHE = _os.path.join(DATA_DIR, "PMBB_external",
                         "PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_updated_June8.csv")
PMBB_616 = _os.path.join(DATA_DIR, "PMBB_external",
                         "PMBBv3_GRS_MEGA_616snps_AllSamples.sscore_withSTDscore.txt")
PMBB_526 = _os.path.join(DATA_DIR, "PMBB_external",
                         "PMBBv3_GRS_QUANT_526snps_AllSamples.sscore_withSTDscore.txt")

LABEL = "CaseCtrl"
SEED = 42
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
PRIMARY = "MLP"
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SPECS = [0.90, 0.95]          # fixed specificity operating points
PER = 10000                   # report counts per this many screened


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
#  LOAD  (identical to 07/08)
# ═══════════════════════════════════════════════════════════════
print("Loading data ...", flush=True)
tr = pd.read_excel(TRAIN_F)
tr = int_pgs616_training_only(tr)
y_tr = tr[LABEL].values.astype(int)
phe = pd.read_csv(PMBB_PHE)
p616 = (pd.read_csv(PMBB_616, sep="\t")[["IID", "SCORE1_AVG_STD"]]
        .rename(columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": "PGS616"}))
p526 = (pd.read_csv(PMBB_526, sep="\t")[["IID", "SCORE1_AVG_STD"]]
        .rename(columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": "PGS526"}))
pmbb = phe.merge(p616, on="PMBB_ID").merge(p526, on="PMBB_ID")
pmbb = pmbb[pmbb["ANCESTRY"] == "AFR"].dropna(
    subset=["POAG_cases", "PGS616", "PGS526",
            "PMBB_3.0_Release_AGE", "SEX"]).copy()
pmbb["POAG_cases"] = pmbb["POAG_cases"].astype(int)
pmbb = restrict_pmbb_age(pmbb)
pmbb["SEX_bin"] = (pmbb["SEX"] == "Male").astype(int)
y = pmbb["POAG_cases"].values
n_case, n_ctrl = int(y.sum()), int((1 - y).sum())
prev = y.mean()
print(f"  PMBB AFR N={len(y):,}  cases={n_case}  controls={n_ctrl}  "
      f"prevalence={prev*100:.2f}%", flush=True)

FS = {
    "Base":        (["Age", "Gender"], ["PMBB_3.0_Release_AGE", "SEX_bin"]),
    "Base+PGS616": (["Age", "Gender", "PGS616"],
                    ["PMBB_3.0_Release_AGE", "SEX_bin", "PGS616"]),
}

# fit once on the full training cohort, score PMBB
pred = {}
for m in MODEL_NAMES:
    for fs, (ct, cp) in FS.items():
        pipe = make_pipeline(m)
        pipe.fit(tr[ct].values, y_tr)
        pred[(m, fs)] = pipe.predict_proba(pmbb[cp].values)[:, 1]
        if m == PRIMARY:
            print(f"  {fs:12s} AUC = {roc_auc_score(y, pred[(m, fs)]):.4f}",
                  flush=True)


# ═══════════════════════════════════════════════════════════════
#  Metrics at a fixed specificity
# ═══════════════════════════════════════════════════════════════
def at_fixed_spec(scores, yv, spec):
    """Threshold at the given specificity; return sens, ppv, counts."""
    ctrl = scores[yv == 0]
    if len(ctrl) == 0:
        return None
    thr = np.quantile(ctrl, spec)          # spec of controls fall below
    flagged = scores >= thr
    tp = int((flagged & (yv == 1)).sum())
    fp = int((flagged & (yv == 0)).sum())
    fn = int(((~flagged) & (yv == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    achieved_spec = 1 - fp / (yv == 0).sum()
    return {"threshold": thr, "sens": sens, "ppv": ppv,
            "tp": tp, "fp": fp, "fn": fn, "spec": achieved_spec}


rowsA = []
for m in MODEL_NAMES:
    for fs in FS:
        s = pred[(m, fs)]
        auc = roc_auc_score(y, s)
        for spec in SPECS:
            r = at_fixed_spec(s, y, spec)
            # scale counts to a screened population of PER people
            cases_per = prev * PER
            rowsA.append({
                "Classifier": m,
                "Model": fs,
                "AUC": round(auc, 4),
                "Target specificity": f"{spec*100:.0f}%",
                "Achieved specificity": round(r["spec"], 4),
                "Sensitivity": round(r["sens"], 4),
                "PPV": round(r["ppv"], 4),
                f"Cases detected per {PER:,}": round(r["sens"] * cases_per, 1),
                f"Cases missed per {PER:,}": round((1 - r["sens"]) * cases_per, 1),
                f"False positives per {PER:,}":
                    round((1 - r["spec"]) * (1 - prev) * PER, 1),
                "Primary": "yes" if m == PRIMARY else "",
            })
sheetA = pd.DataFrame(rowsA)

print(f"\n=== {PRIMARY}: sensitivity at fixed specificity ===", flush=True)
for _, r in sheetA[(sheetA.Classifier == PRIMARY)].iterrows():
    print(f"  {r['Model']:12s} spec={r['Target specificity']:4s}  "
          f"sens={r['Sensitivity']:.4f}  "
          f"cases detected/{PER:,}={r[f'Cases detected per {PER:,}']}",
          flush=True)


# ═══════════════════════════════════════════════════════════════
#  Paired bootstrap on the DIFFERENCE (Base+PGS616 - Base)
# ═══════════════════════════════════════════════════════════════
print("\nPaired bootstrap on differences ...", flush=True)
rng = np.random.RandomState(SEED)
idx_c = np.where(y == 1)[0]
idx_n = np.where(y == 0)[0]
boot_idx = [np.concatenate([rng.choice(idx_c, len(idx_c), replace=True),
                            rng.choice(idx_n, len(idx_n), replace=True)])
            for _ in range(N_BOOT)]


def boot_p(d):
    n = len(d)
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return max(min(p, 1.0), 1.0 / n)


rowsB = []
for m in MODEL_NAMES:
    sb, sp = pred[(m, "Base")], pred[(m, "Base+PGS616")]
    # AUC difference
    d_auc = np.array([roc_auc_score(y[i], sp[i]) - roc_auc_score(y[i], sb[i])
                      for i in boot_idx])
    lo, hi = np.percentile(d_auc, [2.5, 97.5])
    rowsB.append({
        "Classifier": m, "Metric": "AUC",
        "Base": round(roc_auc_score(y, sb), 4),
        "Base+PGS616": round(roc_auc_score(y, sp), 4),
        "Difference": round(roc_auc_score(y, sp) - roc_auc_score(y, sb), 4),
        "95% CI": f"({lo:+.4f}, {hi:+.4f})",
        "p": "<0.001" if boot_p(d_auc) < 0.001 else f"{boot_p(d_auc):.3f}",
    })
    for spec in SPECS:
        rb = at_fixed_spec(sb, y, spec)
        rp = at_fixed_spec(sp, y, spec)
        d_sens = []
        for i in boot_idx:
            a = at_fixed_spec(sb[i], y[i], spec)
            b = at_fixed_spec(sp[i], y[i], spec)
            if a and b:
                d_sens.append(b["sens"] - a["sens"])
        d_sens = np.array(d_sens)
        lo, hi = np.percentile(d_sens, [2.5, 97.5])
        cases_per = prev * PER
        rowsB.append({
            "Classifier": m,
            "Metric": f"Sensitivity at {spec*100:.0f}% specificity",
            "Base": round(rb["sens"], 4),
            "Base+PGS616": round(rp["sens"], 4),
            "Difference": round(rp["sens"] - rb["sens"], 4),
            "95% CI": f"({lo:+.4f}, {hi:+.4f})",
            "p": "<0.001" if boot_p(d_sens) < 0.001 else f"{boot_p(d_sens):.3f}",
            f"Extra cases per {PER:,}":
                round((rp["sens"] - rb["sens"]) * cases_per, 2),
        })
sheetB = pd.DataFrame(rowsB)

print(f"\n=== {PRIMARY}: differences (Base+PGS616 - Base) ===", flush=True)
for _, r in sheetB[sheetB.Classifier == PRIMARY].iterrows():
    extra = r.get(f"Extra cases per {PER:,}", "")
    if pd.isna(extra):
        extra = ""
    print(f"  {r['Metric']:34s} {r['Base']:.4f} -> {r['Base+PGS616']:.4f}  "
          f"diff={r['Difference']:+.4f} {r['95% CI']}  p={r['p']}"
          + (f"  extra cases/{PER:,}: {extra}" if extra != "" else ""),
          flush=True)


notes = pd.DataFrame({"Note": [
    "Table: Statistical versus clinical significance of the PGS616 increment, "
    "PMBB African ancestry external cohort.",
    "",
    f"COHORT. N = {len(y):,} ({n_case} POAG cases, {n_ctrl} controls; "
    f"prevalence {prev*100:.2f}%). Models were trained in the POAAGG cohort "
    "(N=271) and applied without retraining.",
    "",
    "METHOD. For each classifier the decision threshold was set at fixed "
    "specificity (90% and 95%) in this cohort, and sensitivity, positive "
    "predictive value, and the numbers of cases detected, cases missed and "
    "false positives per "
    f"{PER:,} people screened were computed at that operating point. "
    "Differences between Base and Base+PGS616 were assessed by a paired, "
    "participant-level bootstrap "
    f"(B = {N_BOOT}, stratified on case status), in which both models are "
    "scored on the same resampled individuals.",
    "",
    "PURPOSE. An area under the curve difference of about 0.01 is "
    "statistically distinguishable from zero in a cohort of this size but is "
    "not by itself a clinical quantity. Expressing the same increment as "
    "additional cases detected per "
    f"{PER:,} screened at a fixed false-positive rate makes the practical "
    "magnitude explicit and supports the distinction the Discussion draws "
    "between statistical improvement and clinically meaningful benefit.",
]})

with pd.ExcelWriter(_os.path.join(OUT_XL, "Table_ClinicalUtility_PMBB.xlsx"),
                    engine="openpyxl") as w:
    sheetA.to_excel(w, sheet_name="A_FixedSpecificity", index=False)
    sheetB.to_excel(w, sheet_name="B_Differences", index=False)
    notes.to_excel(w, sheet_name="C_Notes", index=False)

print("\nSaved Table_ClinicalUtility_PMBB.xlsx", flush=True)
print("All done.", flush=True)
