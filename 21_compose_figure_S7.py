# =============================================================
#  POAG — Recompose Figure S7 (two-panel incremental AUC forest)
#  iScience R4 revision (2026-09-04)
#
#  Figure S7 in R3 had two panels:
#    left   training (POAAGG, N=271)  — paired CV-fold differences
#    right  PMBB external (N=9,817)   — DeLong / paired bootstrap
#
#  Only the LEFT panel was invalid: fold-level differences are not
#  independent (Editor point 1). Replacing the whole figure with the
#  regenerated training panel alone would silently delete the external
#  panel, so both are recomposed here into a single figure:
#
#    left   training  — paired PARTICIPANT-LEVEL bootstrap (script 10)
#    right  PMBB      — DeLong with analytic CI (script 07), unchanged
#                       methodology, redrawn to match
#
#  The contrast between the panels is the point of the figure and is
#  stated in the caption: the increment is indistinguishable from zero in
#  the training cohort and small but significant in the external cohort.
#
#  Inputs:
#    outputs/tables/bootstrap_replicates_training.csv.gz   (script 10)
#    outputs/tables/Table_DeltaAUC_PMBB_External.xlsx      (script 07)
#  Output:
#    outputs/figures/SF7_DeltaAUC_Forest_R4.{png,pdf}
#    outputs/figures/SF7_caption_R4.txt
# =============================================================

import os as _os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = _os.path.dirname(_os.path.abspath(__file__))
TBL = _os.path.join(HERE, "outputs", "tables")
FIG = _os.path.join(HERE, "outputs", "figures")
_os.makedirs(FIG, exist_ok=True)

RAW = _os.path.join(TBL, "bootstrap_replicates_training.csv.gz")
PMBB = _os.path.join(TBL, "Table_DeltaAUC_PMBB_External.xlsx")

for p, who in ((RAW, "script 10"), (PMBB, "script 07")):
    if not _os.path.exists(p):
        raise SystemExit(f"Missing {_os.path.basename(p)} — run {who} first.")

MODELS = ["LR", "SVM", "RF", "MLP"]
COMP = "Base+PGS616 vs Base"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 300,
})

# ── training panel, from the saved bootstrap replicates ──────────
raw = pd.read_csv(RAW)
W = raw.pivot_table(index="replicate", columns=["Classifier", "FeatureSet"],
                    values="AUC")
train = {}
for m in MODELS:
    d = W[(m, "Base+PGS616")].values - W[(m, "Base")].values
    lo, hi = np.percentile(d, [2.5, 97.5])
    n = len(d)
    p = max(min(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0), 1.0 / n)
    train[m] = (d.mean(), lo, hi, p)

# ── external panel, from script 07 ───────────────────────────────
pm = pd.read_excel(PMBB)
pm = pm[pm["Comparison"] == COMP].set_index("Classifier")


def _parse_ci(s):
    """'+0.0102 (+0.0062, +0.0141)' -> (0.0102, 0.0062, 0.0141)"""
    head, rest = str(s).split("(")
    lo, hi = rest.rstrip(")").split(",")
    return float(head), float(lo), float(hi)


ext = {}
for m in MODELS:
    if m not in pm.index:
        continue
    row = pm.loc[m]
    if "DeLong_95CI" in pm.columns:
        mu, lo, hi = _parse_ci(row["DeLong_95CI"])
    else:
        mu = float(row["Delta_AUC"])
        lo, hi = float(row["boot_lo"]), float(row["boot_hi"])
    pv = float(row["DeLong_p"]) if "DeLong_p" in pm.columns else np.nan
    ext[m] = (mu, lo, hi, pv)

# ── draw ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
yy = np.arange(len(MODELS))[::-1]


def fmt_p(p):
    if np.isnan(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def panel(ax, data, title, colour):
    for k, m in enumerate(MODELS):
        if m not in data:
            continue
        mu, lo, hi, p = data[m]
        ax.errorbar(mu, yy[k], xerr=[[mu - lo], [hi - mu]], fmt="o",
                    color=colour, capsize=3, markersize=6, lw=1.4)
        sig = (lo > 0 or hi < 0)
        ax.text(hi + 0.004, yy[k], f"{mu:+.3f}  {fmt_p(p)}",
                va="center", fontsize=7.2,
                fontweight="bold" if sig else "normal")
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("ΔAUC (Base+PGS616 − Base)")
    ax.set_title(title, fontsize=9, fontweight="bold")


panel(axes[0], train,
      "Training (POAAGG, N = 271)\npaired participant-level bootstrap, "
      "B = 2,000", "#E15759")
panel(axes[1], ext,
      "PMBB external (AFR, N = 9,084)\nDeLong test for correlated ROC curves",
      "#4E79A7")

# give the annotation text room
for ax, data in ((axes[0], train), (axes[1], ext)):
    his = [v[2] for v in data.values()]
    los = [v[1] for v in data.values()]
    span = max(his) - min(los)
    ax.set_xlim(min(los) - 0.06 * span, max(his) + 0.55 * span)

# NOTE (2026-09-04 re-audit): this panel is captioned "Figure S6" in the
# submitted Supplemental_Figures_0904.docx (script 29 swapped S6<->S7 to
# fix citation order), but this baked-in title still read "S7" — a
# text-substitution pass over the docx captions cannot reach pixels
# rendered into an image. Corrected to S6 here so the file this script
# produces matches the document it is placed into; the filename and
# in-code identifiers keep the original "S7" name to avoid a wider
# rename across scripts that reference them.
fig.suptitle("Figure S6. Incremental AUC of PGS616 over the age + sex "
             "baseline", fontsize=10, fontweight="bold", y=1.02)
fig.tight_layout()
for ext_ in ("png", "pdf"):
    fig.savefig(_os.path.join(FIG, f"SF7_DeltaAUC_Forest_R4.{ext_}"),
                bbox_inches="tight")
plt.close(fig)

# ── caption ──────────────────────────────────────────────────────
caption = (
    "Figure S6. Incremental AUC (ΔAUC) of PGS616 over the age + sex "
    "baseline, related to Figure 3.\n"
    "Within-classifier incremental AUC of Base+PGS616 relative to Base, shown "
    "for each of the four classifiers. Left: POAAGG training cohort "
    "(N = 271); points are the mean paired difference across 2,000 "
    "participant-level bootstrap replicates and bars are percentile 95% "
    "confidence intervals. Within each replicate both models are fitted on "
    "the same in-bag participants and evaluated on the same out-of-bag "
    "participants, so the contrast is paired. This replaces the paired "
    "cross-validation-fold differences reported previously, which were based "
    "on non-independent fold estimates. Right: PMBB African ancestry external "
    "cohort (N = 9,084); points are the observed ΔAUC and bars are analytic "
    "95% confidence intervals from the DeLong test for two correlated ROC "
    "curves, with models trained in POAAGG and applied without refitting. "
    "Dashed line marks no difference. Intervals excluding zero are shown in "
    "bold. The increment is not distinguishable from zero for any classifier "
    "in the training cohort; in the external cohort it is small and varies "
    "in sign across classifiers, and for the multilayer perceptron, the "
    "primary model, it is not distinguishable from zero.\n"
)
with open(_os.path.join(FIG, "SF7_caption_R4.txt"), "w",
          encoding="utf-8") as fh:
    fh.write(caption)

print("=== Figure S7 recomposed ===", flush=True)
print(f"{'':6s} {'training (bootstrap)':34s} {'PMBB (DeLong)'}", flush=True)
for m in MODELS:
    t = train[m]
    e = ext.get(m)
    ts = f"{t[0]:+.4f} ({t[1]:+.4f},{t[2]:+.4f}) {fmt_p(t[3])}"
    es = (f"{e[0]:+.4f} ({e[1]:+.4f},{e[2]:+.4f}) {fmt_p(e[3])}"
          if e else "n/a")
    print(f"  {m:4s} {ts:34s} {es}", flush=True)
print("\nSaved SF7_DeltaAUC_Forest_R4.png/.pdf and SF7_caption_R4.txt",
      flush=True)
