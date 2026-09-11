# =============================================================
#  POAG — Redraw the three external panels after the 0905 corrections
#  iScience revision, 2026-09-05
#
#  Figures 2D, 3C and 5C all show PMBB results. Every one of them was
#  computed with the uncorrected PGS616 scale and without the stated age
#  restriction, so all three are wrong in the 0904 composites. Script 27
#  rebuilt the training panels (2C, 3A, 3B, 5B) but deliberately kept the
#  external ones; this script covers them.
#
#  Panel styling follows the training panels as redrawn in script 27:
#  four classifier bars plus a black dash for the cross-classifier mean,
#  bootstrap intervals, value labels. The old panels used a fifth purple
#  "Average" bar, which is dropped here so a reader cannot mistake the
#  mean for a fifth model.
#
#  RUN AFTER script 27. Script 27 pastes onto the pristine R3 backup, so
#  running it afterwards would discard these panels; this one pastes onto
#  the current composite.
#
#  Output: figures/Figure_{2,3,5}_Combined.png
# =============================================================

import os as _os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT = _os.environ.get("POAG_PROJECT_DIR", HERE)
FIGDIR = _os.path.join(ROOT, "figures")
TBL = _os.path.join(HERE, "outputs", "tables")
DPI = 300

MODELS = ["LR", "SVM", "RF", "MLP"]
MODEL_COLORS = {"LR": "#4E79A7", "SVM": "#F28E2B",
                "RF": "#59A14F", "MLP": "#E15759"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.linewidth": 1.0})


def render(draw_fn, box, out_png):
    x0, y0, x1, y1 = box
    fig = plt.figure(figsize=((x1 - x0) / DPI, (y1 - y0) / DPI), dpi=DPI)
    draw_fn(fig)
    fig.savefig(out_png, dpi=DPI, facecolor="white")
    plt.close(fig)
    im = Image.open(out_png)
    if im.size != (x1 - x0, y1 - y0):
        im = im.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    return im


def paste_onto_current(path, box, img):
    """Paste onto the composite as it stands, not onto the R3 backup."""
    base = Image.open(path).convert("RGB")
    base.paste(img.convert("RGB"), (box[0], box[1]))
    base.save(path)
    return base.size


def label(fig, txt, x=0.012, y=0.985):
    fig.text(x, y, txt, fontsize=26, fontweight="bold", va="top", ha="left")


def ci_from_string(s):
    """'0.731 (0.696-0.763)' -> (0.731, 0.696, 0.763); handles en dashes."""
    s = str(s).replace("–", "-").replace("—", "-")
    mu = float(s.split(" (")[0])
    lo, hi = s.split("(")[1].rstrip(")").split("-")
    return mu, float(lo), float(hi)


def grouped_bars(ax, groups, values, title, ylabel, ylim, chance=0.5):
    """values[group][model] = (mu, lo, hi); a black dash marks the mean."""
    xs = np.arange(len(groups))
    wdt = 0.19
    for i, m in enumerate(MODELS):
        mu = [values[g][m][0] for g in groups]
        lo = [values[g][m][0] - values[g][m][1] for g in groups]
        hi = [values[g][m][2] - values[g][m][0] for g in groups]
        ax.bar(xs + (i - 1.5) * wdt, mu, wdt * 0.9, label=m,
               color=MODEL_COLORS[m], alpha=0.92, yerr=[lo, hi],
               capsize=2, error_kw={"lw": 0.9, "alpha": 0.75})
    avg = [np.mean([values[g][m][0] for m in MODELS]) for g in groups]
    ax.plot(xs, avg, "k_", markersize=18, markeredgewidth=2.0,
            label="Average", zorder=5)
    if chance is not None:
        ax.axhline(chance, color="black", ls=":", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", ls=":", color="#CCCCCC", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


print("Redrawing the external panels on the corrected results")
print("=" * 66, flush=True)

# ═══════════════════════════════════════════════════════════════
#  FIGURE 2D — standalone PGS, PMBB external
# ═══════════════════════════════════════════════════════════════
d2 = pd.read_excel(_os.path.join(TBL, "Table_PMBB_External_5x20CV_PGSonly.xlsx"),
                   sheet_name="PMBB_AFR")
G2 = ["PGS616", "PGS526"]
V2 = {g: {m: ci_from_string(
            d2[(d2.PGS == g) & (d2.Model == m)].iloc[0]["AUC_95CI"])
          for m in MODELS} for g in G2}
N2 = int(d2.iloc[0]["N_cases"]) + int(d2.iloc[0]["N_controls"])
BOX_2D = (3210, 2086, 5755, 3662)


def draw_2d(fig):
    ax = fig.add_axes([0.13, 0.16, 0.83, 0.68])
    grouped_bars(ax, G2, V2,
                 f"D   Standalone PGS — PMBB External (N = {N2:,})",
                 "AUC (bootstrap mean, 95% CI)", (0.33, 0.74))
    ax.legend(frameon=False, ncol=5, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.0))
    label(fig, "D", x=0.02, y=0.99)


img = render(draw_2d, BOX_2D, _os.path.join(TBL, "..", "figures", "_p2d.png"))
F2 = _os.path.join(FIGDIR, "Figure_2_Combined.png")
print(f"  Figure 2D redrawn  {paste_onto_current(F2, BOX_2D, img)}", flush=True)


def whiteout(path, box, why):
    """Blank a region of the composite that no panel box covers."""
    base = Image.open(path).convert("RGB")
    base.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), "white"),
               (box[0], box[1]))
    base.save(path)
    print(f"    cleared {why} at x={box[0]}-{box[2]}, y={box[1]}-{box[3]}",
          flush=True)


# The R3 composite carried its own small panel letters just above the two
# lower panels, and the redrawn panels bring their own, so the originals
# now read as duplicates. Measured: the band y 1960-2085 holds nothing but
# those two letters (columns 0-200 and 3000-3200); panels A and B end well
# above it.
whiteout(F2, (0, 1960, 5755, 2086), "duplicate C/D panel letters")
# Old panel C extended past the detected box edge into the C-D gutter;
# those pixels are byte-identical to the R3 backup, i.e. never repainted.
whiteout(F2, (2546, 2086, 3210, 3662), "stale panel C remnant in the gutter")

# ═══════════════════════════════════════════════════════════════
#  FIGURE 3C — feature sets, PMBB external
# ═══════════════════════════════════════════════════════════════
d3 = pd.read_excel(_os.path.join(TBL, "Table_Figure3_PMBB_External.xlsx"),
                   sheet_name="PMBB_AFR")
G3 = ["Base", "Base+PGS526", "Base+PGS616",
      "Base+PC5", "Base+PC5+PGS526", "Base+PC5+PGS616"]
G3 = [g for g in G3 if g in set(d3.FeatureSet)]
V3 = {g: {m: ci_from_string(
            d3[(d3.FeatureSet == g) & (d3.Model == m)].iloc[0]["AUC_95CI"])
          for m in MODELS} for g in G3}
N3 = int(d3.iloc[0]["N_cases"]) + int(d3.iloc[0]["N_controls"])
BOX_3C = (2933, 1621, 5205, 2860)
G3_LBL = [g.replace("Base+", "Base\n+").replace("+PGS", "\n+PGS")
          if g.count("+") > 1 else g.replace("+", "\n+") for g in G3]


def draw_3c(fig):
    ax = fig.add_axes([0.13, 0.19, 0.83, 0.64])
    grouped_bars(ax, G3, V3,
                 f"C   PMBB External Validation (N = {N3:,})",
                 "AUC (bootstrap mean, 95% CI)", (0.40, 0.86), chance=None)
    ax.set_xticklabels(G3_LBL, fontsize=8.5)
    ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, 1.0))
    label(fig, "C", x=0.02, y=0.99)


img = render(draw_3c, BOX_3C, _os.path.join(TBL, "..", "figures", "_p3c.png"))
F3 = _os.path.join(FIGDIR, "Figure_3_Combined.png")
print(f"  Figure 3C redrawn  {paste_onto_current(F3, BOX_3C, img)}", flush=True)

# ═══════════════════════════════════════════════════════════════
#  FIGURE 5C — asymmetry models, PMBB external
# ═══════════════════════════════════════════════════════════════
ASYM_SRC = _os.path.join(TBL, "Table_Figure5_Asymmetry.xlsx")
if not _os.path.exists(ASYM_SRC):
    print("  Figure 5C SKIPPED — run script 42 first", flush=True)
    sys.exit(0)

d5 = pd.read_excel(ASYM_SRC, sheet_name="PMBB_AUC_Sep_Delta")
d5 = d5[d5.Model == "MLP"]
ROWS_5C = [("PHE\nOnly", "IOP_PHE", "CDR_PHE"),
           ("PHE+\nPGS526", "IOP+PGS526", "CDR+PGS526"),
           ("PHE+\nPGS616", "IOP+PGS616", "CDR+PGS616")]
N5 = int(d5.iloc[0]["PMBB_N"])
C5 = int(d5.iloc[0]["PMBB_Cases"])
BOX_5C = (3915, 0, 5333, 1438)


def _row5(fs):
    r = d5[d5.FeatureSet == fs].iloc[0]
    return float(r["AUC"]), float(r["CI_lo_95"]), float(r["CI_hi_95"])


def draw_5c(fig):
    ax = fig.add_axes([0.24, 0.17, 0.72, 0.62])
    yy = np.arange(len(ROWS_5C))[::-1]
    # the two series are offset vertically: the IOP interval reaches into
    # the CDR marker, so drawn on one line the value labels sit on top of
    # the other series' points.
    for stem, col, mk, lab, off in ((1, "#C0392B", "o", "ΔIOP", +0.17),
                                    (2, "#2E6DA4", "s", "ΔCDR", -0.17)):
        for k, spec in enumerate(ROWS_5C):
            mu, lo, hi = _row5(spec[stem])
            ax.errorbar(mu, yy[k] + off, xerr=[[mu - lo], [hi - mu]], fmt=mk,
                        color=col, markersize=8, capsize=4, lw=1.6,
                        label=lab if k == 0 else None)
            ax.text(hi + 0.008, yy[k] + off, f"{mu:.3f}", va="center",
                    fontsize=9.5, color=col)
    ax.axvline(0.5, color="grey", ls="--", lw=1.0)
    ax.text(0.5, -0.72, "chance", fontsize=8.5, color="#999999",
            ha="center", va="bottom")
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in ROWS_5C], fontsize=10.5)
    # 2026-09-10: the points are the observed AUCs (Table S18), not
    # bootstrap means; only the interval is from the bootstrap
    ax.set_xlabel("AUC (95% bootstrap CI)", fontsize=11)
    ax.set_xlim(0.45, 0.84)
    ax.set_ylim(-0.8, len(ROWS_5C) - 0.35)
    ax.grid(axis="x", ls="--", color="#CCCCCC", lw=0.9)
    ax.set_axisbelow(True)
    # kept to two short lines: at this panel width a third clause, or the
    # cohort size on the same line, runs off the canvas.
    ax.set_title(f"PMBB External — MLP (N = {N5:,}, {C5} cases)\n"
                 f"ΔIOP- and ΔCDR-based models",   # the outcome is POAG
                 fontsize=10.5, fontweight="bold")
    # measured against every value label: lower right sat on 0.651 and
    # center right on 0.666; upper right, upper left and lower left were
    # all clear, and upper right also keeps clear of the y-axis labels.
    ax.legend(frameon=True, fontsize=10, loc="upper right", framealpha=0.95)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    label(fig, "C", x=0.02, y=0.99)


img = render(draw_5c, BOX_5C, _os.path.join(TBL, "..", "figures", "_p5c.png"))
F5 = _os.path.join(FIGDIR, "Figure_5_Combined.png")
print(f"  Figure 5C redrawn  {paste_onto_current(F5, BOX_5C, img)}", flush=True)
print("\nAll external panels rebuilt on the corrected results.", flush=True)
