# =============================================================
#  POAG — Participant-level bootstrap for the remaining analyses
#  iScience R4 revision (2026-09-04)
#
#  Addresses Editor point 6 ("...for every classifier, cohort, and
#  incremental model comparison") in the three places where R3 intervals
#  were still derived from non-independent cross-validation folds and
#  which script 10 did not cover:
#
#    (A) Sex-stratified AUC          -> Table S8, Figure S5
#    (B) Inter-eye asymmetry models  -> Table S10, Figure 5B
#                                       (main-text CIs: 0.806, 0.813, 0.766)
#    (C) PGS residualised on PCs     -> Table S13 panel B
#
#  All three are computed in ONE pass over the bootstrap replicates: each
#  replicate draws a stratified participant resample once and every model
#  needed by (A), (B) and (C) is fitted on that same in-bag sample and
#  evaluated on the same out-of-bag participants. This keeps every
#  contrast paired and costs roughly the same as a single analysis.
#
#  RESIDUALISATION NOTE. In script 09 the PGS was residualised on PC1-PC5
#  once, using the whole cohort. Here the residualising regression is fitted
#  on the IN-BAG sample only and applied to the out-of-bag participants, so
#  no out-of-bag information enters the transformation.
#
#  Outputs:
#    outputs/tables/Table_Secondary_Bootstraps.xlsx
#      A_SexStratified / B_Asymmetry / C_Residualised / D_Notes
#    outputs/figures/SF5_SexStratified_R4.{png,pdf}
#    outputs/figures/Figure_5B_Asymmetry_R4.{png,pdf}
#    outputs/tables/bootstrap_replicates_secondary.csv.gz
#
#  Usage:  python 18_secondary_bootstraps.py [B] [n_jobs]
# =============================================================

import os as _os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
_os.environ.setdefault("PYTHONWARNINGS", "ignore")

HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
from poag_corrections import (int_pgs616, int_pgs616_training_only,
                              restrict_pmbb_age,
                              recompute_training_deltas)   # 2026-09-05 audit
TBL = _os.path.join(HERE, "outputs", "tables")
FIG = _os.path.join(HERE, "outputs", "figures")
_os.makedirs(TBL, exist_ok=True)
_os.makedirs(FIG, exist_ok=True)

# Data location. Set POAG_DATA_DIR to the folder holding the cohort
# subdirectories; it defaults to ./data next to this script. The data
# themselves are under controlled access (see data/README.md).
DATA_DIR = _os.environ.get(
    "POAG_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   "data"))
TRAIN_F = _os.path.join(DATA_DIR, "POAAGG_cohort",
                        "271_training_cohort_4_new_PRS_cleaned.xlsx")

LABEL = "CaseCtrl"
MODEL_NAMES = ["LR", "SVM", "RF", "MLP"]
SEED = 42
B      = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else -1
MIN_OOB_PER_CLASS = 10
MIN_OOB_PER_SEX = 8          # for the sex-stratified panel

BASE = ["Age", "Gender"]
PC5 = [f"PC{i}" for i in range(1, 6)]
PGS_ALL = ["POAAGG PGS", "MEGA PGS", "PGS526", "PGS616"]

# (A) sex-stratified feature sets
FS_SEX = {
    "Base":              BASE,
    "Base+PGS616":       BASE + ["PGS616"],
    "Base+PC5+PGS616":   BASE + PC5 + ["PGS616"],
}
# (B) asymmetry feature sets
FS_ASYM = {"dIOP PHE only": ["delta_IOP"], "dCDR PHE only": ["delta_CDR"]}
for p in PGS_ALL:
    FS_ASYM[f"dIOP + {p}"] = ["delta_IOP", p]
    FS_ASYM[f"dCDR + {p}"] = ["delta_CDR", p]


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


print("Loading training cohort ...", flush=True)
tr = pd.read_excel(TRAIN_F)
tr = int_pgs616_training_only(tr)
tr = recompute_training_deltas(tr)      # 2026-09-10 audit B02
y = tr[LABEL].values.astype(int)
N = len(y)
sex = tr["Gender"].values.astype(int)
idx_case = np.where(y == 1)[0]
idx_ctrl = np.where(y == 0)[0]

X_sex = {k: tr[c].values.astype(float) for k, c in FS_SEX.items()}
X_asym = {k: tr[c].values.astype(float) for k, c in FS_ASYM.items()}
X_base = tr[BASE].values.astype(float)
PC5_M = tr[PC5].values.astype(float)
PGS616_V = tr["PGS616"].values.astype(float)

print(f"  N={N}  males={int((sex==1).sum())}  females={int((sex==0).sum())}",
      flush=True)
print(f"  sex-stratified sets={len(FS_SEX)}  asymmetry sets={len(FS_ASYM)}  "
      f"B={B}", flush=True)


def one_replicate(b):
    warnings.filterwarnings("ignore")
    rng = np.random.RandomState(SEED + b)
    in_bag = np.concatenate([
        rng.choice(idx_case, len(idx_case), replace=True),
        rng.choice(idx_ctrl, len(idx_ctrl), replace=True)])
    oob = np.setdiff1d(np.arange(N), np.unique(in_bag))
    if len(oob) == 0:
        return None
    y_oob = y[oob]
    if (y_oob == 1).sum() < MIN_OOB_PER_CLASS or \
       (y_oob == 0).sum() < MIN_OOB_PER_CLASS:
        return None
    out = {}

    # ---- (A) sex-stratified -------------------------------------
    s_oob = sex[oob]
    for m in MODEL_NAMES:
        for fs, X in X_sex.items():
            pipe = make_pipeline(m)
            pipe.fit(X[in_bag], y[in_bag])
            p = pipe.predict_proba(X[oob])[:, 1]
            out[("SEX", m, fs, "all")] = roc_auc_score(y_oob, p)
            for lab, mask in (("male", s_oob == 1), ("female", s_oob == 0)):
                yy, pp = y_oob[mask], p[mask]
                if mask.sum() >= MIN_OOB_PER_SEX and len(np.unique(yy)) > 1:
                    out[("SEX", m, fs, lab)] = roc_auc_score(yy, pp)

    # ---- (B) asymmetry ------------------------------------------
    for m in MODEL_NAMES:
        for fs, X in X_asym.items():
            pipe = make_pipeline(m)
            pipe.fit(X[in_bag], y[in_bag])
            p = pipe.predict_proba(X[oob])[:, 1]
            out[("ASYM", m, fs, "all")] = roc_auc_score(y_oob, p)

    # ---- (C) residualised PGS616 --------------------------------
    #  fit PGS616 ~ PC1-5 on the IN-BAG sample only, apply to OOB.
    #  PC columns carry 4 missing values; impute with the in-bag median so
    #  the residualising regression sees no out-of-bag information.
    imp = SimpleImputer(strategy="median").fit(PC5_M[in_bag])
    PC5_i = imp.transform(PC5_M)
    lin = LinearRegression().fit(PC5_i[in_bag], PGS616_V[in_bag])
    resid = PGS616_V - lin.predict(PC5_i)
    X_raw = np.column_stack([X_base, PGS616_V])
    X_res = np.column_stack([X_base, resid])
    for m in MODEL_NAMES:
        for tag, X in (("Base", X_base), ("Base+PGS616_raw", X_raw),
                       ("Base+PGS616_resid", X_res)):
            pipe = make_pipeline(m)
            pipe.fit(X[in_bag], y[in_bag])
            out[("RESID", m, tag, "all")] = roc_auc_score(
                y_oob, pipe.predict_proba(X[oob])[:, 1])
    return out


print("\nRunning bootstrap (three analyses, one pass) ...", flush=True)
t0 = time.time()
res = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(one_replicate)(b) for b in range(B))
res = [r for r in res if r is not None]
print(f"  {len(res)} valid replicates in {(time.time()-t0)/60:.1f} min",
      flush=True)


def collect(key):
    vals = [r[key] for r in res if key in r]
    return np.array(vals)


def ci(a):
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)


def boot_p(d):
    n = len(d)
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return max(min(p, 1.0), 1.0 / n)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ═══════════════════════════════════════════════════════════════
#  A — sex-stratified
# ═══════════════════════════════════════════════════════════════
rowsA = []
for m in MODEL_NAMES:
    for fs in FS_SEX:
        for grp in ("all", "male", "female"):
            a = collect(("SEX", m, fs, grp))
            if len(a) == 0:
                continue
            mu, lo, hi = ci(a)
            rowsA.append({
                "Classifier": m, "Feature set": fs, "Group": grp,
                "Bootstrap AUC": round(mu, 3),
                "95% CI": f"{mu:.3f} ({lo:.3f}-{hi:.3f})",
                "n replicates": len(a),
            })
# male vs female difference, paired within replicate
for m in MODEL_NAMES:
    for fs in FS_SEX:
        am = collect(("SEX", m, fs, "male"))
        af = collect(("SEX", m, fs, "female"))
        k = min(len(am), len(af))
        if k < 50:
            continue
        d = am[:k] - af[:k]
        mu, lo, hi = ci(d)
        rowsA.append({
            "Classifier": m, "Feature set": fs, "Group": "male - female",
            "Bootstrap AUC": round(mu, 3),
            "95% CI": f"{mu:+.3f} ({lo:+.3f}, {hi:+.3f}) p={fmt_p(boot_p(d))}",
            "n replicates": k,
        })
sheetA = pd.DataFrame(rowsA)

# ═══════════════════════════════════════════════════════════════
#  B — asymmetry
# ═══════════════════════════════════════════════════════════════
rowsB = []
for m in MODEL_NAMES:
    for fs in FS_ASYM:
        a = collect(("ASYM", m, fs, "all"))
        mu, lo, hi = ci(a)
        row = {"Classifier": m, "Feature set": fs,
               "Bootstrap AUC": round(mu, 3),
               "95% CI": f"{mu:.3f} ({lo:.3f}-{hi:.3f})"}
        stem = "dIOP" if fs.startswith("dIOP") else "dCDR"
        base_key = ("ASYM", m, f"{stem} PHE only", "all")
        if fs != f"{stem} PHE only":
            d = a - collect(base_key)
            dm, dlo, dhi = ci(d)
            row["delta vs PHE only"] = f"{dm:+.4f} ({dlo:+.4f}, {dhi:+.4f})"
            row["delta p"] = fmt_p(boot_p(d))
        rowsB.append(row)
sheetB = pd.DataFrame(rowsB)

# ═══════════════════════════════════════════════════════════════
#  C — residualised PGS
# ═══════════════════════════════════════════════════════════════
rowsC = []
for m in MODEL_NAMES:
    base = collect(("RESID", m, "Base", "all"))
    for tag in ("Base+PGS616_raw", "Base+PGS616_resid"):
        a = collect(("RESID", m, tag, "all"))
        d = a - base
        mu, lo, hi = ci(a)
        dm, dlo, dhi = ci(d)
        rowsC.append({
            "Classifier": m,
            "Model": tag.replace("Base+PGS616_", "PGS616 "),
            "Bootstrap AUC": round(mu, 3),
            "AUC 95% CI": f"{mu:.3f} ({lo:.3f}-{hi:.3f})",
            "delta vs Base": round(dm, 4),
            "delta 95% CI": f"{dm:+.4f} ({dlo:+.4f}, {dhi:+.4f})",
            "delta p": fmt_p(boot_p(d)),
        })
sheetC = pd.DataFrame(rowsC)

notes = pd.DataFrame({"Note": [
    "Table: Participant-level bootstrap intervals for the sex-stratified, "
    "inter-eye asymmetry, and PGS-residualisation analyses.",
    "",
    f"METHOD. B = {B} replicates ({len(res)} valid); participants resampled "
    "with replacement, stratified on case-control status; models refitted on "
    "the in-bag sample and evaluated on out-of-bag participants. All three "
    "analyses share the same replicates, so every contrast is paired.",
    "",
    "PANEL A. Sex-stratified AUC is computed within the out-of-bag sample; "
    f"replicates leaving fewer than {MIN_OOB_PER_SEX} out-of-bag members of a "
    "sex, or only one outcome class within a sex, are omitted for that "
    "stratum. This replaces the fold-level aggregation used previously.",
    "",
    "PANEL B. Replaces the cross-validation intervals previously reported for "
    "the asymmetry models, including the main-text values for delta-IOP "
    "(0.806, 0.813) and delta-CDR (0.766).",
    "",
    "PANEL C. The residualising regression of PGS616 on PC1-PC5 is fitted on "
    "the in-bag sample only and applied to out-of-bag participants, so the "
    "transformation carries no out-of-bag information. This differs from the "
    "whole-cohort residualisation used previously and is the more conservative "
    "construction.",
]})

with pd.ExcelWriter(_os.path.join(TBL, "Table_Secondary_Bootstraps.xlsx"),
                    engine="openpyxl") as w:
    sheetA.to_excel(w, sheet_name="A_SexStratified", index=False)
    sheetB.to_excel(w, sheet_name="B_Asymmetry", index=False)
    sheetC.to_excel(w, sheet_name="C_Residualised", index=False)
    notes.to_excel(w, sheet_name="D_Notes", index=False)

# raw draws
raw_rows = []
for i, r in enumerate(res):
    for (grp, m, fs, sub), v in r.items():
        raw_rows.append({"replicate": i, "Analysis": grp, "Classifier": m,
                         "FeatureSet": fs, "Stratum": sub, "AUC": v})
pd.DataFrame(raw_rows).to_csv(
    _os.path.join(TBL, "bootstrap_replicates_secondary.csv.gz"),
    index=False, compression="gzip")

# ═══════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})

# --- Figure S5: sex-stratified ---
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, fs in zip(axes, FS_SEX):
    xs = np.arange(len(MODEL_NAMES))
    for j, (grp, col) in enumerate((("male", "#4E79A7"),
                                    ("female", "#E15759"))):
        mus, los, his = [], [], []
        for m in MODEL_NAMES:
            a = collect(("SEX", m, fs, grp))
            mu, lo, hi = ci(a)
            mus.append(mu); los.append(mu - lo); his.append(hi - mu)
        ax.bar(xs + (j - 0.5) * 0.34, mus, 0.32, color=col, alpha=0.9,
               label=grp.capitalize(), yerr=[los, his], capsize=2,
               error_kw={"lw": 0.8})
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(MODEL_NAMES)
    ax.set_title(fs, fontsize=9, fontweight="bold")
    ax.set_ylim(0.35, 0.90)
axes[0].set_ylabel("AUC")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Sex-stratified performance — participant-level bootstrap 95% CI",
             fontsize=10, fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(_os.path.join(FIG, f"SF5_SexStratified_R4.{ext}"),
                bbox_inches="tight")
plt.close(fig)

# --- Figure 5B: asymmetry ---
fig, ax = plt.subplots(figsize=(7.5, 4.2))
groups = ["PHE only"] + PGS_ALL
yy = np.arange(len(groups))[::-1]
for stem, col, off in (("dIOP", "#E05C5C", -0.16), ("dCDR", "#5B8DB8", 0.16)):
    mus, los, his = [], [], []
    for g in groups:
        fs = f"{stem} PHE only" if g == "PHE only" else f"{stem} + {g}"
        a = collect(("ASYM", "MLP", fs, "all"))
        mu, lo, hi = ci(a)
        mus.append(mu); los.append(mu - lo); his.append(hi - mu)
    ax.errorbar(mus, yy + off, xerr=[los, his], fmt="o", color=col,
                capsize=3, markersize=6, lw=1.3,
                label="ΔIOP" if stem == "dIOP" else "ΔCDR")
ax.axvline(0.5, color="grey", ls="--", lw=0.9)
ax.set_yticks(yy); ax.set_yticklabels(groups)
ax.set_xlabel("Cross-validated AUC (participant-level bootstrap 95% CI)")
ax.set_title("Inter-eye asymmetry models, MLP, training cohort",
             fontsize=9.5, fontweight="bold")
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(_os.path.join(FIG, f"Figure_5B_Asymmetry_R4.{ext}"),
                bbox_inches="tight")
plt.close(fig)

print("\nSaved Table_Secondary_Bootstraps.xlsx and 2 figures", flush=True)

print("\n=== (B) Asymmetry, MLP — R3 main-text values vs R4 ===", flush=True)
for fs in ["dIOP PHE only", "dIOP + PGS616", "dCDR PHE only", "dCDR + PGS616"]:
    a = collect(("ASYM", "MLP", fs, "all"))
    mu, lo, hi = ci(a)
    print(f"  {fs:20s} {mu:.3f} ({lo:.3f}-{hi:.3f})", flush=True)

print("\n=== (C) Residualised PGS616, delta vs Base ===", flush=True)
for _, r in sheetC.iterrows():
    print(f"  {r['Classifier']:4s} {r['Model']:16s} {r['delta 95% CI']:32s} "
          f"p={r['delta p']}", flush=True)
print("\nAll done.", flush=True)
