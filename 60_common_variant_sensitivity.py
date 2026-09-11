# =============================================================
#  POAG — Common-variant sensitivity analysis for the external PGS
#  iScience revision, 2026-09-10
#
#  PGS616 and PGS526 were scored in POAAGG from 568 and 486 variants and
#  in PMBB from 539 and 460 (the PMBB genotype extract did not include
#  every variant of the final panel). This script rescores BOTH cohorts on
#  the variants scored in both (525 and 448), with the same weights and
#  the same transformation, and repeats the external validation, so the
#  external comparison rests on identically constructed scores.
#
#  Scoring is re-implemented from the PLINK genotype subsets. In POAAGG it
#  reproduces the existing score files exactly. The PMBB subset available
#  here is hard-called (the original PMBB scores used imputed dosages), so
#  PMBB scores rebuilt from it correlate r = 0.99 with the originals rather
#  than matching exactly. The comparison is therefore built so that the
#  hard-calling cannot be confused with the variant set:
#    as reported        POAAGG 568/486 variants; PMBB original dosage scores
#    PMBB hard calls    same variant sets, PMBB rebuilt from hard calls
#    common variants    both cohorts on the shared 525/448 variants, PMBB
#                       from hard calls
#  The effect of variant coverage is the difference between the last two.
#
#  Output: outputs/tables/Table_CommonVariant_Sensitivity.xlsx
# =============================================================

import os as _os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import poag_paths as P                                     # noqa: E402
from poag_corrections import restrict_pmbb_age, int_pgs616  # noqa: E402

# Genotype and weight files. With POAG_DATA_DIR/genotypes present (the
# layout in data/README.md) they are read from there; otherwise from the
# authors' archive layout.
_G = _os.path.join(P.DATA_DIR, "genotypes")
if _os.path.isdir(_G):
    PQ = _os.path.join(_G, "POAAGG")        # GRS_MainTable1_572snps.* + old weights
    PM = _os.path.join(_G, "PMBB")          # PMBB_GRS_572snps.* + scoring logs
    WDIR = _os.path.join(_G, "weights")     # current weight files
else:
    PROJ = _os.path.abspath(_os.path.join(HERE, "..", "..", ".."))
    ARCH = _os.path.join(PROJ, "_archive", "04_iscience_R1_work_2026-04-05",
                         "2026-4-29-Revision-to-iScience")
    PQ = _os.path.join(ARCH, "01_input_data", "POAAGG_PGS_data")
    PM = _os.path.join(ARCH, "07_PMBB_documentation", "PMBBv3")
    WDIR = _os.path.join(PROJ, "data")
OUT = _os.path.join(HERE, "outputs", "tables",
                    "Table_CommonVariant_Sensitivity.xlsx")
SEED = 42
MODELS = ["LR", "SVM", "RF", "MLP"]

# weights: (file, allele column, beta column)
W = {"PGS616": (_os.path.join(WDIR, "GRS_weight_MEGA_MainTable1_572snps.txt"),
                "A2", "BETA"),
     "PGS526": (_os.path.join(WDIR, "GRS_weight_QUANT_MainTable1_572snps.txt"),
                "A1", "BETA")}
W_OLD = {"PGS616": (_os.path.join(PQ, "GRS_weight_MEGA_572snps.txt"),
                    "A2", "BETA"),
         "PGS526": (_os.path.join(PQ, "GRS_weight_QUANT_572snps.txt"),
                    "A1", "BETA")}


# ---- genotype reading and PLINK-style scoring -----------------------
def read_bed(prefix):
    bim = pd.read_csv(prefix + ".bim", sep=r"\s+", header=None,
                      names=["chr", "id", "cm", "pos", "a1", "a2"])
    fam = pd.read_csv(prefix + ".fam", sep=r"\s+", header=None,
                      names=["fid", "iid", "p", "m", "sex", "ph"])
    n, v = len(fam), len(bim)
    bpv = (n + 3) // 4
    raw = np.fromfile(prefix + ".bed", dtype=np.uint8)
    assert raw[0] == 0x6C and raw[1] == 0x1B and raw[2] == 0x01
    mat = raw[3:3 + bpv * v].reshape(v, bpv)
    codes = np.stack([(mat >> (2 * k)) & 3 for k in range(4)], axis=2)
    codes = codes.reshape(v, bpv * 4)[:, :n]
    lut = np.array([2, -1, 1, 0], dtype=np.int8)      # count of bim a1
    g = lut[codes].T                                   # samples x variants
    bim["key"] = bim["chr"].astype(str) + ":" + bim["pos"].astype(str)
    return bim, fam, g


def score(bim, g, wfile, acol, bcol, keep=None):
    w = pd.read_csv(wfile, sep=r"\s+")
    w = w[w["SNP"].isin(set(bim["key"]))]
    m = w.merge(bim.reset_index(), left_on="SNP", right_on="key")
    ok = (m[acol] == m["a1"]) | (m[acol] == m["a2"])
    m = m[ok]
    if keep is not None:
        m = m[m["SNP"].isin(keep)]
    cols = m["index"].values
    G = g[:, cols].astype(float)
    G[G < 0] = np.nan
    dose = np.where((m[acol] == m["a1"]).values, G, 2 - G)
    freq = np.nanmean(dose, axis=0)
    dose = np.where(np.isnan(dose), freq, dose)
    s = dose @ m[bcol].values
    return pd.Series(s), set(m["SNP"]), 2 * len(m)


def inv_normal(x):
    x = np.asarray(x, dtype=float)
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def pid(iid):
    return pd.Series(iid).str.extract(r"SCHE[-_](\d+)")[0].astype(float)


# ---- model helpers (as in script 07) ---------------------------------
def make_pipeline(name):
    steps = [("imp", SimpleImputer(strategy="median")),
             ("scl", StandardScaler())]
    clf = {"LR": LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=SEED),
           "SVM": SVC(kernel="rbf", probability=True,
                      class_weight="balanced", random_state=SEED),
           "RF": RandomForestClassifier(n_estimators=200, max_depth=5,
                                        class_weight="balanced",
                                        random_state=SEED),
           "MLP": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000,
                                early_stopping=False,
                                random_state=SEED)}[name]
    return Pipeline(steps + [("clf", clf)])


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong(y, p1, p2):
    order = (-np.asarray(y)).argsort(kind="mergesort")
    m = int(np.asarray(y).sum())
    preds = np.vstack((np.asarray(p1)[order], np.asarray(p2)[order]))
    n = preds.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    d = aucs[0] - aucs[1]
    se = np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(d / se))) if se > 0 else 1.0
    return aucs[0], aucs[1], d, d - 1.96 * se, d + 1.96 * se, p


# =============================================================
#  1. validate the scorer against the existing score files
# =============================================================
print("Reading genotypes ...", flush=True)
bq, fq, gq = read_bed(_os.path.join(PQ, "GRS_MainTable1_572snps"))
bp, fp_, gp = read_bed(_os.path.join(PM, "PMBB_GRS_572snps"))
print(f"  POAAGG {gq.shape}, PMBB {gp.shape}", flush=True)

checks = []


def check(label, mine, file, keep_ids=None):
    s = pd.read_csv(file, sep=r"\s+")
    ref = s.set_index("IID")["SCORE1_SUM"]
    got = pd.Series(mine.values, index=keep_ids)
    common = ref.index.intersection(got.index)
    d = float(np.max(np.abs(ref[common].values - got[common].values)))
    checks.append((label, len(common), d))
    print(f"  [{'ok  ' if d < 1e-3 else 'FAIL'}] {label}: {len(common)} "
          f"samples, max |diff| = {d:.2e}", flush=True)
    return d < 1e-3


ok = True
s, used_q616, ac = score(bq, gq, *W["PGS616"])
ok &= check("POAAGG PGS616 (616 weights)", s,
            _os.path.join(PQ, "GRS_MEGA_MainTable1_572snps.sscore"),
            fq["iid"].values)
s, used_q526, _ = score(bq, gq, *W["PGS526"])
ok &= check("POAAGG PGS526 (526 weights)", s,
            _os.path.join(PQ, "GRS_MTAG_MainTable1_572snps.sscore"),
            fq["iid"].values)
# PMBB: hard calls vs the original dosage scores (correlation, not identity)
s, used_p616, _ = score(bp, gp, *W_OLD["PGS616"])
ref = pd.read_csv(_os.path.join(PM, "PMBBv3_GRS_MEGA_572snps_AllSamples"
                                ".sscore"), sep=r"\s+").set_index("IID")
r_pm = float(np.corrcoef(ref.loc[fp_["iid"].values, "SCORE1_SUM"].values,
                         s.values)[0, 1])
checks.append(("PMBB PGS616 hard calls vs original dosage score",
               len(s), r_pm))
print(f"  PMBB PGS616 hard-call vs dosage score: r = {r_pm:.4f}", flush=True)
ok &= r_pm > 0.98
if not ok:
    raise SystemExit("scorer does not reproduce the existing scores")


# =============================================================
#  2. common-variant scores in both cohorts
# =============================================================
_, used_p616_cur, _ = score(bp, gp, *W["PGS616"])
_, used_p526_cur, _ = score(bp, gp, *W["PGS526"])
common = {"PGS616": used_q616 & used_p616_cur,
          "PGS526": used_q526 & used_p526_cur}
counts = pd.DataFrame([
    {"Score": k, "Weights": len(pd.read_csv(W[k][0], sep=r"\s+")),
     "Scored in POAAGG": len(used_q616 if k == "PGS616" else used_q526),
     "Scored in PMBB": len(used_p616_cur if k == "PGS616"
                          else used_p526_cur),
     "Scored in both (sensitivity panel)": len(common[k])}
    for k in ("PGS616", "PGS526")])
print(counts.to_string(index=False), flush=True)

q_c, p_c, p_h = {}, {}, {}
for k in ("PGS616", "PGS526"):
    sq, _, _ = score(bq, gq, *W[k], keep=common[k])
    sp, _, _ = score(bp, gp, *W[k], keep=common[k])
    sh, _, _ = score(bp, gp, *W[k])                 # PMBB variant set, hard
    q_c[k] = pd.Series(sq.values, index=fq["iid"].values)
    p_c[k] = pd.Series(inv_normal(sp.values), index=fp_["iid"].values)
    p_h[k] = pd.Series(inv_normal(sh.values), index=fp_["iid"].values)

# training and suspect rows -> genotype IIDs
tr = pd.read_excel(P.TRAINING_FILE)
su = pd.read_excel(P.SUSPECT_FILE)
map_pid = pd.Series(fq["iid"].values, index=pid(fq["iid"].values).values)
map_pid = map_pid[~map_pid.index.duplicated()]
tr_iid = tr["POAAGG ID"].astype(float).map(map_pid)
if tr_iid.isna().any():
    raise SystemExit(f"{int(tr_iid.isna().sum())} training rows unmatched")
su_iid = su["IID"]
for k in ("PGS616", "PGS526"):
    tr[k + "_c"] = q_c[k].reindex(tr_iid.values).values
    su[k + "_c"] = q_c[k].reindex(su_iid.values).values
    pooled = inv_normal(np.r_[tr[k + "_c"].values, su[k + "_c"].values])
    tr[k + "_c"] = pooled[:len(tr)]
tr, su = int_pgs616(tr, su)                    # current-analysis PGS616

# correlation of common-panel and full scores (training cohort)
corr = {k: float(np.corrcoef(tr[k], tr[k + "_c"])[0, 1])
        for k in ("PGS616", "PGS526")}
print("  training r(full, common):", {k: round(v, 3) for k, v in corr.items()},
      flush=True)

# PMBB analysis cohort
phe = pd.read_csv(_os.path.join(P.DATA_DIR, "PMBB_external",
                                "PMBB_3.0_pheno_covars_for_Yan_noPOAAGG_"
                                "updated_June8.csv"))
cur = {}
for k, f in (("PGS616", "PMBBv3_GRS_MEGA_616snps_AllSamples"),
             ("PGS526", "PMBBv3_GRS_QUANT_526snps_AllSamples")):
    cur[k] = (pd.read_csv(_os.path.join(P.DATA_DIR, "PMBB_external",
                                        f + ".sscore_withSTDscore.txt"),
                          sep="\t")[["IID", "SCORE1_AVG_STD"]]
              .rename(columns={"IID": "PMBB_ID", "SCORE1_AVG_STD": k}))
pm = phe.merge(cur["PGS616"], on="PMBB_ID").merge(cur["PGS526"],
                                                   on="PMBB_ID")
pm = pm[pm["ANCESTRY"] == "AFR"].dropna(
    subset=["POAG_cases", "PGS616", "PGS526", "PMBB_3.0_Release_AGE",
            "SEX"]).copy()
pm = restrict_pmbb_age(pm)
pm["SEX_bin"] = (pm["SEX"] == "Male").astype(int)
for k in ("PGS616", "PGS526"):
    pm[k + "_c"] = p_c[k].reindex(pm["PMBB_ID"].values).values
    pm[k + "_h"] = p_h[k].reindex(pm["PMBB_ID"].values).values
    tr[k + "_h"] = tr[k]
y_tr = tr["CaseCtrl"].values.astype(int)
y_pm = pm["POAG_cases"].values.astype(int)
print(f"  PMBB N = {len(pm):,} ({y_pm.sum()} cases); missing common score: "
      f"{int(pm['PGS616_c'].isna().sum())}", flush=True)

# =============================================================
#  3. external validation: full (as reported) vs common panel
# =============================================================
rows = []
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
folds = list(rskf.split(np.zeros(len(y_tr)), y_tr))
for mname in MODELS:
    base = make_pipeline(mname).fit(tr[["Age", "Gender"]].values, y_tr)
    pb = base.predict_proba(pm[["PMBB_3.0_Release_AGE", "SEX_bin"]]
                            .values)[:, 1]
    for k in ("PGS616", "PGS526"):
        for panel, col in (("as reported", k), ("PMBB hard calls", k + "_h"),
                           ("common variants", k + "_c")):
            pipe = make_pipeline(mname).fit(tr[["Age", "Gender", col]].values,
                                            y_tr)
            pa = pipe.predict_proba(pm[["PMBB_3.0_Release_AGE", "SEX_bin",
                                        col]].values)[:, 1]
            a1, a0, d, lo, hi, p = delong(y_pm, pa, pb)
            cv = []
            for tri, tei in folds:
                pp = make_pipeline(mname).fit(
                    tr[["Age", "Gender", col]].values[tri], y_tr[tri])
                cv.append(roc_auc_score(
                    y_tr[tei], pp.predict_proba(
                        tr[["Age", "Gender", col]].values[tei])[:, 1]))
            rows.append({"Classifier": mname, "Score": k, "Panel": panel,
                         "Training CV AUC (descriptive)": round(np.mean(cv), 3),
                         "PMBB AUC Base": round(a0, 4),
                         "PMBB AUC Base+PGS": round(a1, 4),
                         "PMBB delta-AUC": round(d, 4),
                         "DeLong 95% CI": f"{d:+.4f} ({lo:+.4f}, {hi:+.4f})",
                         "DeLong p": round(p, 4)})
            print(f"  {mname:4s} {k} {panel:16s} dAUC={d:+.4f} "
                  f"({lo:+.4f},{hi:+.4f}) p={p:.3f}", flush=True)
res = pd.DataFrame(rows)

notes = pd.DataFrame({"Note": [
    "Common-variant sensitivity analysis for the external validation.",
    "",
    "PGS616 and PGS526 were rescored in both cohorts on the variants scored in "
    "both (see counts), with the same weights and allele coding, and "
    "rank-based inverse normal transformed within cohort (POAAGG training "
    "and suspect cohorts pooled; all PMBB participants with genotypes). "
    "Models were refitted in the training cohort and applied to the PMBB "
    "analysis cohort (N = 9,084) without refitting; delta-AUC against the "
    "age and sex model by the DeLong test.",
    "",
    "Scoring was re-implemented from the PLINK genotype subsets. POAAGG: "
    "the existing score files are reproduced exactly ("
    + "; ".join(f"{a}, {n} samples, max |diff| {d:.1e}"
                for a, n, d in checks[:2]) + "). PMBB: the subset available "
    "is hard-called whereas the original scores used imputed dosages; "
    f"rebuilt scores correlate r = {checks[2][2]:.3f} with the originals. "
    "'PMBB hard calls' repeats the reported analysis with PMBB scores rebuilt "
    "from hard calls on the same variants, so that the comparison of "
    "'common variants' with it isolates the effect of the variant set.",
    "",
    "Correlation of the common-panel with the full score in the training "
    "cohort: " + ", ".join(f"{k} r = {v:.3f}" for k, v in corr.items()) + "."]})
with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    counts.to_excel(w, sheet_name="A_VariantCounts", index=False)
    res.to_excel(w, sheet_name="B_External", index=False)
    notes.to_excel(w, sheet_name="C_Notes", index=False)
print(f"\nSaved {_os.path.basename(OUT)}")
