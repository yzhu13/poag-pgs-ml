# =============================================================
#  POAG — Rebuild Figure S2 and repair the supplemental captions
#  iScience R4 revision (2026-09-04)
#
#  TWO JOBS.
#
#  (1) Figure S2 is regenerated in full — BOTH panels — rather than
#      grafting a new panel B onto the old image:
#        A  PC1-PC5 x four PGS Pearson correlations (recomputed; the
#           values are unchanged from R3, this only redraws them)
#        B  six feature sets x four classifiers + cross-classifier mean,
#           with PARTICIPANT-LEVEL BOOTSTRAP intervals replacing the
#           invalid 5x20 CV intervals            (editor points 1, 6)
#
#  (2) The supplemental figure CAPTIONS are repaired. These live in
#      Supplemental_Figures_*.docx, a separate submitted file, and were
#      never swept in R3 or R4 — every earlier scan covered the
#      manuscript only. They still contain claims the reviewer objected
#      to in the previous round and claims this round's analyses have
#      since falsified, including:
#        S2A  "largely non-redundant signals"      (|r|max = 0.35)
#        S2B  "consistent benefit of PGS integration"
#        S2B  "declined monotonically" / "matched or outperformed"
#        S3D  "substantially outperform" / "consistently lower"
#        S4   "consistently outperforms Base across all training sizes"
#        S5   "males showed substantially higher AUC than females"
#        S6A  "confirming that higher predicted risk is associated with
#              greater retinal nerve fiber loss"   (now known to be
#              carried by age and sex)
#      and four "mean +/- 95% CI, 5x20 CV" interval labels.
#
#  Outputs:
#    outputs/figures/SF2_PC_PGS_R4.{png,pdf}
#    Supplemental_Figures_0904.docx   (image2 replaced, captions edited)
# =============================================================

import os as _os
import sys
import re
import numpy as np
import pandas as pd
import docx
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = _os.path.dirname(_os.path.abspath(__file__))
TBL = _os.path.join(HERE, "outputs", "tables")
FIG = _os.path.join(HERE, "outputs", "figures")
ROOT = _os.environ.get("POAG_PROJECT_DIR", HERE)
SUPF = _os.path.join(ROOT, "Supplemental_Figures_0904.docx")
_os.makedirs(FIG, exist_ok=True)

# Data location. Set POAG_DATA_DIR to the folder holding the cohort
# subdirectories; it defaults to ./data next to this script. The data
# themselves are under controlled access (see data/README.md).
DATA_DIR = _os.environ.get(
    "POAG_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                   "data"))
TRAIN_F = _os.path.join(DATA_DIR, "POAAGG_cohort",
                        "271_training_cohort_4_new_PRS_cleaned.xlsx")
RAW = _os.path.join(TBL, "bootstrap_replicates_training.csv.gz")

PGS_ORDER = ["MEGA PGS", "PGS526", "PGS616", "POAAGG PGS"]
PCS = [f"PC{i}" for i in range(1, 6)]
MODELS = ["LR", "SVM", "RF", "MLP"]
FS_B = ["Base", "Base+PC2", "Base+PC5", "Base+PC10", "Base+PC20",
        "Base+PGS616"]
FS_COL = {"Base": "#8C8C8C", "Base+PC2": "#F0A73A", "Base+PC5": "#E8836B",
          "Base+PC10": "#C0392B", "Base+PC20": "#7B1E24",
          "Base+PGS616": "#2AA0A8"}
FS_MK = {"Base": "o", "Base+PC2": "s", "Base+PC5": "^", "Base+PC10": "D",
         "Base+PC20": "P", "Base+PGS616": "D"}
PRIOR = 0.87

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "figure.dpi": 300})

# ═══════════════════════════════════════════════════════════════
#  PANEL A — correlations (recomputed, values unchanged)
# ═══════════════════════════════════════════════════════════════
# 0905: PGS616 on the transformed scale, as in every other analysis
sys.path.insert(0, HERE)
from poag_corrections import int_pgs616                       # noqa: E402
import poag_paths                                             # noqa: E402
tr, _su = int_pgs616(pd.read_excel(TRAIN_F),
                     pd.read_excel(poag_paths.SUSPECT_FILE))
R = np.zeros((len(PCS), len(PGS_ORDER)))
P = np.zeros_like(R)
for i, pc in enumerate(PCS):
    for j, g in enumerate(PGS_ORDER):
        m = tr[[pc, g]].dropna()
        R[i, j], P[i, j] = stats.pearsonr(m[pc], m[g])


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ═══════════════════════════════════════════════════════════════
#  PANEL B — bootstrap intervals
# ═══════════════════════════════════════════════════════════════
raw = pd.read_csv(RAW)
W = raw.pivot_table(index="replicate", columns=["Classifier", "FeatureSet"],
                    values="AUC")


def ci(v):
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def get(scope, fs):
    if scope == "Average":
        return ci(np.mean([W[(m, fs)].values for m in MODELS], axis=0))
    return ci(W[(scope, fs)].values)


# ═══════════════════════════════════════════════════════════════
#  DRAW
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(9.6, 10.5))
gs = GridSpec(2, 1, height_ratios=[1.0, 1.45], hspace=0.28)

# --- A ---
axA = fig.add_subplot(gs[0])
im = axA.imshow(R, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
axA.set_xticks(range(len(PGS_ORDER)))
axA.set_xticklabels(PGS_ORDER, rotation=22, ha="right", style="italic",
                    fontsize=9)
axA.set_yticks(range(len(PCS)))
axA.set_yticklabels(PCS, fontsize=9)
for i in range(len(PCS)):
    for j in range(len(PGS_ORDER)):
        # three decimals to match Table S4 exactly; the R3 figure showed
        # 0.35 for a value Table S4 reports as 0.355, which at two decimals
        # is 0.36 — a display mismatch we avoid by matching the table.
        axA.text(j, i, f"{R[i,j]:.3f}{stars(P[i,j])}", ha="center",
                 va="center", fontsize=7.8,
                 color="white" if abs(R[i, j]) > 0.28 else "black")
axA.set_title("PC–PGS Pearson Correlations\n(Training Cohort, N = 271)",
              fontsize=10, fontweight="bold")
cb = fig.colorbar(im, ax=axA, fraction=0.030, pad=0.02)
cb.set_label("Pearson r", fontsize=9)
axA.text(-0.13, 1.10, "A", transform=axA.transAxes, fontsize=15,
         fontweight="bold", va="top")

# --- B ---
axB = fig.add_subplot(gs[1])
scopes = ["Average", "MLP", "RF", "SVM", "LR"]
band = 1.0
row_y = {}
for k, sc in enumerate(scopes):
    base_y = (len(scopes) - 1 - k) * band
    row_y[sc] = base_y
    if k % 2 == 0:
        axB.axhspan(base_y - band / 2, base_y + band / 2, color="#F2F2F2",
                    zorder=0)
    for t, fs in enumerate(FS_B):
        mu, lo, hi = get(sc, fs)
        y = base_y + (t - (len(FS_B) - 1) / 2) * (band / (len(FS_B) + 1.4))
        axB.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt=FS_MK[fs],
                     color=FS_COL[fs], markersize=5, capsize=2, lw=1.1,
                     zorder=3, label=fs if k == 0 else None)
        axB.text(hi + 0.004, y, f"{mu:.3f}", va="center", fontsize=6.6,
                 color=FS_COL[fs])

axB.axvline(PRIOR, color="#C0392B", ls="--", lw=1.4, zorder=2)
axB.text(PRIOR - 0.004, (len(scopes) - 0.62) * band,
         "Prior non-CV estimate\n(AUC = 0.87)", color="#C0392B",
         fontsize=7.6, ha="right", va="top")
axB.set_yticks([row_y[s] for s in scopes])
axB.set_yticklabels(scopes, fontsize=10)
axB.set_xlabel("AUC (participant-level bootstrap mean, 95% CI)", fontsize=9.5)
axB.set_xlim(0.53, 0.95)
axB.set_ylim(-band / 2, (len(scopes) - 0.5) * band)
axB.grid(axis="x", ls=":", color="#CCCCCC", lw=0.7, zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.legend(frameon=True, fontsize=7.8, loc="lower right", ncol=1,
           framealpha=0.95)
axB.text(-0.13, 1.045, "B", transform=axB.transAxes, fontsize=15,
         fontweight="bold", va="top")

fig.savefig(_os.path.join(FIG, "SF2_PC_PGS_R4.png"), bbox_inches="tight")
fig.savefig(_os.path.join(FIG, "SF2_PC_PGS_R4.pdf"), bbox_inches="tight")
plt.close(fig)
print("Figure S2 rebuilt (both panels).", flush=True)
print("  panel A max |r| = "
      f"{np.abs(R).max():.2f} (PC{np.unravel_index(np.abs(R).argmax(), R.shape)[0]+1}"
      f"–{PGS_ORDER[np.unravel_index(np.abs(R).argmax(), R.shape)[1]]})",
      flush=True)
for fs in FS_B:
    mu, lo, hi = get("Average", fs)
    print(f"  panel B  {fs:14s} {mu:.3f} ({lo:.3f}-{hi:.3f})", flush=True)


# 0905: the caption repairs below were applied to the 0904 captions, which
# the 0905 document inherits; "--figure-only" regenerates the image alone.
if "--figure-only" in sys.argv:
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════
#  CAPTION REPAIRS
# ═══════════════════════════════════════════════════════════════
CAPTION_EDITS = [
    # ---- S2A: the PC-independence claim the reviewer objected to ----
    ("S2A non-redundant",
     "The maximum observed |r| = 0.35 (PC3–PGS616), indicating that ancestry PCs and PGS capture largely non-redundant signals.",
     "The maximum observed |r| = 0.35 (PC3–PGS616). Because a correlation of "
     "0.35 is not negligible, we do not interpret this as evidence that "
     "ancestry PCs and PGS capture distinct signals; a residualisation "
     "analysis is reported in Table S13 and Table S20."),

    # ---- S2B: interval label + three overclaims ----
    ("S2B interval label",
     "(B) Horizontal dot plot showing cross-validated AUC (mean ± 95% CI, 5×20 CV) for six feature configurations",
     "(B) Horizontal dot plot showing AUC (participant-level bootstrap mean "
     "and 95% percentile confidence interval, B = 2,000) for six feature "
     "configurations"),

    ("S2B monotonic + outperform + consistent benefit",
     "Under rigorous 5×20 CV, Base+PC2 achieved the highest AUC among PC-augmented models (average: 0.704) but performance declined monotonically with additional PCs (Base+PC5: 0.688; Base+PC10: 0.672; Base+PC20: 0.657), consistent with overfitting as feature dimensionality increases relative to N = 271. Base+PGS616 (average AUC = 0.700) matched or outperformed all PC-augmented baselines. Among individual classifiers, MLP Base+PGS616 achieved AUC = 0.713, compared to RF Base = 0.616 (lowest), demonstrating the consistent benefit of PGS integration across model classes.",
     "Base+PC2 gave the highest cross-validated AUC among PC-augmented models "
     "(average 0.704) with lower values as further components were added "
     "(Base+PC5: 0.688; Base+PC10: 0.672; Base+PC20: 0.657), a pattern "
     "consistent with overfitting as feature dimensionality increases "
     "relative to N = 271, although under participant-level bootstrap "
     "resampling these differences are not statistically distinguishable. "
     "Base+PGS616 reached a nominally higher average AUC (0.700) than the "
     "PC-augmented baselines, but no feature set differed significantly from "
     "the age and sex baseline in this cohort (Table S14); the confidence "
     "intervals shown overlap almost completely. The red dashed reference "
     "line marks the value reported in a prior non-cross-validated analysis "
     "and is retained to show the extent to which that estimate overstated "
     "achievable performance."),

    # ---- S3D ----
    ("S3D outperform/consistently",
     "All classifiers and feature sets substantially outperform the no-skill baseline. Base+PGS616 shows consistently lower Brier scores than Base+PC5+PGS616 across all classifiers, indicating better calibration with PGS alone",
     "All classifiers and feature sets improve on the no-skill baseline. "
     "Base+PGS616 shows lower Brier scores than Base+PC5+PGS616 across "
     "classifiers, indicating better calibration with PGS alone",),

    # ---- S4 ----
    ("S4 consistently outperforms",
     "Base+PGS616 (teal) consistently outperforms Base (gray) across all training sizes.",
     "Base+PGS616 (teal) lies above Base (gray) across training sizes, "
     "although the curves are close relative to their dispersion and the "
     "learning-curve summaries are descriptive rather than inferential."),

    # ---- S5 ----
    ("S5 substantially higher in males",
     "males showed substantially higher AUC than females (Male: 0.702; Female: 0.641; Overall: 0.688), consistent with ancestry PC axes disproportionately capturing sex-correlated cohort structure rather than disease-specific",
     "males showed a nominally higher AUC than females (Male: 0.702; Female: "
     "0.641; Overall: 0.688). Under participant-level bootstrap resampling no "
     "sex difference in any feature set was statistically distinguishable "
     "(all p ≥ 0.48; Table S20), so this pattern is suggestive only and may "
     "reflect ancestry PC axes capturing sex-correlated cohort structure "
     "rather than disease-specific",),

    # ---- S6A: the association now known to be age/sex driven ----
    ("S6A confirming RNFL",
     "RNFL thickness showed the most consistent significant negative associations, confirming that higher predicted risk is associated with greater retinal n",
     "RNFL thickness showed the strongest negative associations with "
     "predicted risk. These associations are attributable to the age and sex "
     "components of the models: after adjustment for age and sex the "
     "polygenic scores show no association with any phenotype (Table S15). "
     "Higher predicted risk is associated with greater retinal n"),

    ("S6A demonstrates not driven by",
     "The broad consistency of CDR and RNFL enrichment patterns across all four classifiers and all five feature sets demonstrates that the observed clinical associations are not driven by a specific model architecture or genetic feature co",
     "The similarity of CDR and RNFL enrichment patterns across all four "
     "classifiers and all five feature sets indicates that the observed "
     "clinical associations do not depend on a specific model architecture "
     "or genetic feature co"),
# ---- S5 title + interval label (second instance) ----
    # S5 title and body are separate paragraphs -> two edits
    ("S5 title",
     "Figure S5. Sex-Stratified Cross-Validation AUC, Related to Figure 3",
     "Figure S5. Sex-Stratified AUC, Related to Figure 3"),

    ("S5 interval label",
     "Horizontal dot plot showing cross-validation AUC (mean ± 95% CI, 5×20 CV) stratified by sex",
     "Horizontal dot plot showing AUC (participant-level bootstrap mean and "
     "95% percentile confidence interval, B = 2,000) stratified by sex"),

    # ---- S6B: the same age/sex-driven association, second instance ----
    ("S6B confirming RNFL",
     "RNFL thickness showed significant negative correlations across all feature sets (r = −0.205 to −0.238, all p < 0.001; N = 483), confirming that higher predicted risk is associated with progressive retinal nerve fiber layer thinning. The consistency of CDR and RNFL associations across all five f",
     "RNFL thickness showed significant negative correlations across all "
     "feature sets (r = −0.205 to −0.238, all p < 0.001; N = 483). Note that "
     "the association is strongest for the age-and-sex model (r = −0.238) and "
     "weaker once PGS616 is added (r = −0.205); after adjustment for age and "
     "sex the polygenic scores show no association with any phenotype "
     "(Table S15), so these correlations reflect demographic rather than "
     "polygenic risk. The similarity of CDR and RNFL associations across all "
     "five f"),

]


def norm(s):
    return " ".join(s.split())


def replace_in_par(par, old, new):
    runs = par.runs
    if not runs:
        return False
    text = "".join(r.text for r in runs)
    idx = text.find(old)
    target = old
    if idx < 0:
        nt, no = norm(text), norm(old)
        j = nt.find(no)
        if j < 0:
            return False
        map_idx, prev_space = [], False
        for i, ch in enumerate(text):
            if ch.isspace():
                if prev_space:
                    continue
                map_idx.append(i); prev_space = True
            else:
                map_idx.append(i); prev_space = False
        start = map_idx[j]
        end = map_idx[j + len(no) - 1] + 1
        idx, target = start, text[start:end]
    end = idx + len(target)
    pos, first = 0, None
    for r in runs:
        rs, re_ = pos, pos + len(r.text)
        pos = re_
        if re_ <= idx or rs >= end:
            continue
        ls, le = max(0, idx - rs), min(len(r.text), end - rs)
        if first is None:
            first = r
            r.text = r.text[:ls] + new + r.text[le:]
        else:
            r.text = r.text[:ls] + r.text[le:]
    return first is not None


doc = docx.Document(SUPF)

# swap the Figure S2 image (image2.png)
new_png = open(_os.path.join(FIG, "SF2_PC_PGS_R4.png"), "rb").read()
swapped = False
for part in doc.part.package.parts:
    if str(part.partname) == "/word/media/image2.png":
        print(f"\nreplacing image2.png (Figure S2): "
              f"{len(part._blob):,} -> {len(new_png):,} bytes", flush=True)
        part._blob = new_png
        swapped = True

print("\nCaption repairs:", flush=True)
ok_all = swapped
doc_text = norm(" ".join(p.text for p in doc.paragraphs))
for label, old, new in CAPTION_EDITS:
    # An edit is already applied iff the OLD text is gone. Testing for the
    # presence of the NEW text is wrong: its opening words often duplicate
    # the old text, or appear elsewhere in the document, which silently
    # skips a real edit.
    if norm(old) not in doc_text:
        print(f"  [done] {label} (already applied)", flush=True)
        continue
    hit = any(replace_in_par(p, old, new) for p in doc.paragraphs)
    print(f"  [{'HIT ' if hit else 'MISS'}] {label}", flush=True)
    ok_all &= hit

doc.save(SUPF)
print(f"\nSaved {_os.path.basename(SUPF)}", flush=True)
if not ok_all:
    sys.exit(1)
