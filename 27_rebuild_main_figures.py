# =============================================================
#  POAG — Rebuild the combined main figures with bootstrap intervals
#  iScience R4 revision (2026-09-04)
#
#  The submitted combined figures (figures/Figure_*_Combined.png) were
#  still the R3 renderings: panels 2C, 3A, 3B and 5B showed the
#  cross-validation intervals that editor point 1 asked us to abandon.
#  The regenerated panels existed only as separate files and had never
#  been composited back.
#
#  Panels 2A (SNP distribution) and 5A (asymmetry schematic) exist only
#  inside the combined images — there is no separate source file and no
#  code that reproduces them — so the combined figures cannot simply be
#  re-rendered from scratch. Instead each changed panel is redrawn at the
#  exact pixel geometry of the region it replaces, and pasted in. Panel
#  regions are located by projecting ink onto the row and column axes and
#  finding the blank gutters, not by hard-coded coordinates.
#
#  Panel 3A is redrawn with TWELVE feature sets, dropping Base+PC20, to
#  match the figure legend ("all twelve feature sets ... Base+PC2 through
#  Base+PC10"). Base+PC20 remains in Figure S2 and Table S3, which is
#  where the text says it is shown.
#
#  Output: figures/Figure_{2,3,5}_Combined.png  (originals backed up)
# =============================================================

import os as _os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT = _os.environ.get("POAG_PROJECT_DIR", HERE)
FIGDIR = _os.path.join(ROOT, "figures")
BACKUP = _os.path.join(FIGDIR, "_R3_originals")
TBL = _os.path.join(HERE, "outputs", "tables")
_os.makedirs(BACKUP, exist_ok=True)

MODELS = ["LR", "SVM", "RF", "MLP"]
MODEL_COLORS = {"LR": "#4E79A7", "SVM": "#F28E2B",
                "RF": "#59A14F", "MLP": "#E15759"}
DPI = 300


# ═══════════════════════════════════════════════════════════════
#  data
# ═══════════════════════════════════════════════════════════════
raw = pd.read_csv(_os.path.join(TBL, "bootstrap_replicates_training.csv.gz"))
W = raw.pivot_table(index="replicate", columns=["Classifier", "FeatureSet"],
                    values="AUC")
sec = pd.read_csv(_os.path.join(TBL, "bootstrap_replicates_secondary.csv.gz"))
SW = sec[sec.Analysis == "ASYM"].pivot_table(
    index="replicate", columns=["Classifier", "FeatureSet"], values="AUC")
stand = pd.read_excel(_os.path.join(TBL,
                      "Table_PGS_Standalone_Comparison.xlsx"),
                      sheet_name="A_StandaloneAUC")


def ci(v):
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def tr_ci(m, fs):
    return ci(W[(m, fs)].values)


def tr_avg(fs):
    return ci(np.mean([W[(m, fs)].values for m in MODELS], axis=0))


# ═══════════════════════════════════════════════════════════════
#  panel-region detection
# ═══════════════════════════════════════════════════════════════
def gutters(arr, axis, min_gap=40, thresh=240, ink=3):
    ink_proj = (arr < thresh).sum(axis=axis)
    blank, out, s = ink_proj < ink, [], None
    for i, b in enumerate(blank):
        if b and s is None:
            s = i
        if not b and s is not None:
            if i - s > min_gap:
                out.append((s, i))
            s = None
    if s is not None and len(blank) - s > min_gap:
        out.append((s, len(blank)))
    return out


def panel_boxes(path):
    a = np.array(Image.open(path).convert("L"))
    H, Wd = a.shape
    rows = gutters(a, 1)
    inner = [b for b in rows if b[0] > 20 and b[1] < H - 20]
    if not inner:
        return None
    y_split = inner[-1]
    top = (0, 0, Wd, y_split[0])
    low = a[y_split[1]:, :]
    cols = [b for b in gutters(low, 0) if b[0] > 20 and b[1] < Wd - 20]
    if cols:
        x = cols[-1]
        return {"top": top,
                "bl": (0, y_split[1], x[0], H),
                "br": (x[1], y_split[1], Wd, H)}
    return {"top": top, "bottom": (0, y_split[1], Wd, H)}


def render(draw_fn, box, out_png):
    """Render a panel at exactly the pixel geometry of `box`."""
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) / DPI, (y1 - y0) / DPI
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    draw_fn(fig)
    fig.savefig(out_png, dpi=DPI, facecolor="white")
    plt.close(fig)
    im = Image.open(out_png)
    if im.size != (x1 - x0, y1 - y0):
        im = im.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    return im


def paste(combined_path, replacements):
    """Paste onto the pristine R3 original, never onto a previous rebuild.

    The backup is written only once; thereafter it is the source. Without
    this, a second run would back up the already-modified file and the
    untouched panels would be lost.
    """
    bak = _os.path.join(BACKUP, _os.path.basename(combined_path))
    if not _os.path.exists(bak):
        shutil.copy2(combined_path, bak)
    base = Image.open(bak).convert("RGB")
    for box, img in replacements:
        base.paste(img.convert("RGB"), (box[0], box[1]))
    base.save(combined_path)
    return base.size


def label(fig, txt, x=0.012, y=0.985):
    fig.text(x, y, txt, fontsize=26, fontweight="bold", va="top", ha="left")


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.linewidth": 1.0})

print("Rebuilding combined main figures\n" + "=" * 58, flush=True)

# ═══════════════════════════════════════════════════════════════
#  FIGURE 3 — panel A (12 feature sets) and panel B
# ═══════════════════════════════════════════════════════════════
F3 = _os.path.join(FIGDIR, "Figure_3_Combined.png")
b3 = panel_boxes(F3)
print(f"Figure 3 panels: {b3}", flush=True)

FS12 = ["Age only", "Sex only", "Base", "Base+PC2", "Base+PC5", "Base+PC10",
        "Base+POAAGG PGS", "Base+MEGA PGS", "Base+PGS526", "Base+PGS616",
        "Base+PC5+PGS526", "Base+PC5+PGS616"]
FS12_LBL = ["Age", "Sex", "Base", "Base\n+PC2", "Base\n+PC5", "Base\n+PC10",
            "Base\n+POAAGG", "Base\n+MEGA", "Base\n+PGS526", "Base\n+PGS616",
            "Base+PC5\n+PGS526", "Base+PC5\n+PGS616"]


GROUPS = [(0, 3, "Demographics", "#FFFFFF"),
          (3, 6, "Demographics + PC", "#FDF0DC"),
          (6, 10, "Demographics + PGS", "#FFFFFF"),
          (10, 12, "+ PC & PGS", "#EFE6F7")]


def draw_3a(fig):
    ax = fig.add_axes([0.045, 0.17, 0.94, 0.72])
    xs = np.arange(len(FS12))
    wdt = 0.17
    for i, m in enumerate(MODELS):
        mu, lo, hi = [], [], []
        for fs in FS12:
            a, b, c = tr_ci(m, fs)
            mu.append(a); lo.append(a - b); hi.append(c - a)
        ax.bar(xs + (i - 1.5) * wdt, mu, wdt * 0.9, label=m,
               color=MODEL_COLORS[m], alpha=0.92,
               yerr=[lo, hi], capsize=1.6,
               error_kw={"lw": 0.8, "alpha": 0.7})
    avg = [tr_avg(fs)[0] for fs in FS12]
    ax.plot(xs, avg, "k_", markersize=20, markeredgewidth=2.0,
            label="Average", zorder=5)
    # category bands, matching the legend's description of the grouping.
    # y=0.895 sat inside the ncol=5 "upper left" legend's bounding box
    # (measured: legend y in [0.807,0.872] of the axes, text at 0.895 maps
    # to 0.855) and "Demographics" rendered underneath "LR"/"SVM"; 0.83
    # clears the legend by ~45px.
    for i0, i1, lbl, col in GROUPS:
        if col != "#FFFFFF":
            ax.axvspan(i0 - 0.5, i1 - 0.5, color=col, zorder=0)
        ax.text((i0 + i1 - 1) / 2, 0.83, lbl, ha="center", va="top",
                fontsize=9.5, style="italic", color="#555555")
        if i0 > 0:
            ax.axvline(i0 - 0.5, color="#AAAAAA", ls="--", lw=0.9, zorder=1)
    ax.set_axisbelow(True)
    ax.axhline(0.5, color="black", ls=":", lw=1.0)
    ax.set_xlim(-0.6, len(FS12) - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(FS12_LBL, fontsize=8.5)
    ax.set_ylabel("AUC (bootstrap mean, 95% CI)", fontsize=11)
    ax.set_ylim(0.40, 0.92)
    ax.set_title("A   All Feature Sets — participant-level bootstrap AUC "
                 "(Training Cohort, N = 271)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=5, fontsize=9.5, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    label(fig, "A")


KEY5 = ["Base", "Base+PC5", "Base+PGS526", "Base+PGS616", "Base+PC5+PGS616"]
KEY5_COL = {"Base": "#8C8C8C", "Base+PC5": "#E8836B",
            "Base+PGS526": "#F0A73A", "Base+PGS616": "#2AA0A8",
            "Base+PC5+PGS616": "#8E6BAF"}
KEY5_MK = {"Base": "o", "Base+PC5": "^", "Base+PGS526": "s",
           "Base+PGS616": "D", "Base+PC5+PGS616": "P"}


def draw_3b(fig):
    ax = fig.add_axes([0.16, 0.11, 0.80, 0.78])
    scopes = ["Average"] + MODELS
    for k, sc in enumerate(scopes):
        base_y = (len(scopes) - 1 - k)
        if k % 2 == 0:
            ax.axhspan(base_y - 0.5, base_y + 0.5, color="#F2F2F2", zorder=0)
        for t, fs in enumerate(KEY5):
            mu, lo, hi = (tr_avg(fs) if sc == "Average" else tr_ci(sc, fs))
            y = base_y + (t - 2) * 0.16
            ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]],
                        fmt=KEY5_MK[fs], color=KEY5_COL[fs], markersize=7,
                        capsize=3, lw=1.5, zorder=3,
                        label=fs if k == 0 else None)
            ax.text(hi + 0.004, y, f"{mu:.3f}", va="center", fontsize=8,
                    color=KEY5_COL[fs])
    ax.set_yticks(range(len(scopes)))
    ax.set_yticklabels(scopes[::-1], fontsize=11)
    ax.set_xlabel("AUC (bootstrap mean, 95% CI)", fontsize=11)
    ax.set_xlim(0.53, 0.83)
    ax.set_ylim(-0.5, len(scopes) - 0.5)
    ax.grid(axis="x", ls=":", color="#CCCCCC", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("B   Key Comparison: Base vs PC vs PGS (Training)",
                 fontsize=12, fontweight="bold")
    # "lower right" sat on top of the RF/MLP rows' PGS616 and PGS526 value
    # labels (measured 7 of 25 text labels inside the legend's bbox);
    # "center left" is the only corner/side tested with zero label overlap.
    ax.legend(frameon=True, fontsize=8.5, loc="center left", framealpha=0.95)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    label(fig, "B")


tmp = _os.path.join(HERE, "outputs", "figures")
img_a = render(draw_3a, b3["top"], _os.path.join(tmp, "_p3a.png"))
img_b = render(draw_3b, b3["bl"], _os.path.join(tmp, "_p3b.png"))
sz = paste(F3, [(b3["top"], img_a), (b3["bl"], img_b)])

# The auto-detected "top" box (0,0,5205,1101) undershoots the pristine R3
# panel A's true extent by ~100-500px: the R3 original had thirteen bars
# per group (Base+PC20 included) and its own x-tick-label row extending to
# roughly y=1200, versus the shorter twelve-bar redraw whose real content
# ends near y=1010. The gap between them (1101 to the real gutter at
# b3["bl"][1]) was left showing untouched, misaligned R3 pixels — visible
# as a stray row of tick marks and colour swatches under the new panel A.
# Whited out here because this region is only ever gutter once panel A and
# B are both current.
base = Image.open(F3).convert("RGB")
gap = (0, b3["top"][3], base.width, b3["bl"][1])
if gap[3] > gap[1]:
    white = Image.new("RGB", (gap[2] - gap[0], gap[3] - gap[1]), "white")
    base.paste(white, (gap[0], gap[1]))
    base.save(F3)
    print(f"  cleared stale R3 remnant at y={gap[1]}-{gap[3]}", flush=True)

# Same problem, second location: the vertical gutter between panel B
# ("bl", redrawn, narrower than R3's original) and panel C ("br", kept
# pristine) still showed fragments of R3's old Base+PC5+PGS616 tick label
# ("616", "0", "80") bleeding through at x=2270-2400 — inside what is
# meant to be blank space between the two panels for every run seen so
# far. Whited out for the same reason as above.
base = Image.open(F3).convert("RGB")
vgap = (b3["bl"][2], b3["bl"][1], b3["br"][0], b3["bl"][3])
if vgap[2] > vgap[0]:
    white = Image.new("RGB", (vgap[2] - vgap[0], vgap[3] - vgap[1]), "white")
    base.paste(white, (vgap[0], vgap[1]))
    base.save(F3)
    print(f"  cleared stale R3 remnant at x={vgap[0]}-{vgap[2]}", flush=True)

print(f"  Figure_3_Combined.png rebuilt (A + B replaced, C kept) {sz}",
      flush=True)


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2 — panel C only
# ═══════════════════════════════════════════════════════════════
F2 = _os.path.join(FIGDIR, "Figure_2_Combined.png")
b2 = panel_boxes(F2)
print(f"Figure 2 panels: {b2}", flush=True)

PGS4 = ["PGS526", "PGS616", "MEGA PGS", "POAAGG PGS"]


def parse_ci(s):
    mu = float(str(s).split(" (")[0])
    lo, hi = str(s).split("(")[1].rstrip(")").split("-")
    return mu, float(lo), float(hi)


def draw_2c(fig):
    ax = fig.add_axes([0.11, 0.20, 0.86, 0.66])
    xs = np.arange(len(PGS4))
    wdt = 0.19
    for i, m in enumerate(MODELS):
        mu, lo, hi = [], [], []
        for g in PGS4:
            row = stand[(stand.PGS == g) & (stand.Classifier == m)].iloc[0]
            a, b, c = parse_ci(row["95% CI"])
            mu.append(a); lo.append(a - b); hi.append(c - a)
        ax.bar(xs + (i - 1.5) * wdt, mu, wdt * 0.9, label=m,
               color=MODEL_COLORS[m], alpha=0.92, yerr=[lo, hi],
               capsize=2, error_kw={"lw": 0.9})
    ax.axhline(0.5, color="black", ls="--", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(PGS4, fontsize=10)
    ax.set_ylabel("AUC (bootstrap mean, 95% CI)", fontsize=11)
    ax.set_ylim(0.33, 0.74)
    ax.set_title("C   Standalone PGS — Training Cohort (N = 271)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=4, fontsize=9.5, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    label(fig, "C")


if b2 and "bl" in b2:
    img_2c = render(draw_2c, b2["bl"], _os.path.join(tmp, "_p2c.png"))
    sz = paste(F2, [(b2["bl"], img_2c)])
    print(f"  Figure_2_Combined.png rebuilt (C replaced) {sz}", flush=True)
else:
    print("  Figure 2: panel layout not resolved — left unchanged", flush=True)


# ═══════════════════════════════════════════════════════════════
#  FIGURE 5 — panel B only
# ═══════════════════════════════════════════════════════════════
F5 = _os.path.join(FIGDIR, "Figure_5_Combined.png")
b5 = panel_boxes(F5)
print(f"Figure 5 panels: {b5}", flush=True)

ASYM_G = ["PHE only", "POAAGG PGS", "MEGA PGS", "PGS526", "PGS616"]
# rows are drawn bottom-to-top: PHE Only lowest, PGS616 highest


def asym_ci(stem, g):
    fs = f"{stem} PHE only" if g == "PHE only" else f"{stem} + {g}"
    return ci(SW[("MLP", fs)].values)


# Figure 5 is a 1x3 horizontal layout, so the two-row detector returns
# None. Panel boundaries come from the vertical gutters measured on the
# pristine original: A 0-1701, gutter 1701-1818, B 1818-3617, gutter
# 3617-3915, C 3915-5333.
# 2026-09-10: the box began at x = 1620 to cover the old 'B' label, but
# that also cut the "Hg" of panel A's "IOP=20mmHg" and the right edge of
# the schematic. The panel now starts at the gutter, and only the old
# label (x 1640-1705, above the schematic) is blanked.
B5_BOX = (1705, 0, 3617, 1438)
B5_OLD_LABEL = (1640, 0, 1705, 115)


def draw_5b(fig):
    ax = fig.add_axes([0.30, 0.135, 0.665, 0.70])
    yy = np.arange(len(ASYM_G))          # PHE Only lowest row
    # 2026-09-10: the two series are offset vertically, as in panel C; on
    # one line the ΔCDR value labels were struck through by the ΔIOP bars
    for stem, col, mk, lab, off in (("dIOP", "#C0392B", "o", "ΔIOP", +0.17),
                                    ("dCDR", "#2E6DA4", "s", "ΔCDR", -0.17)):
        for k, g in enumerate(ASYM_G):
            mu, lo, hi = asym_ci(stem, g)
            ax.errorbar(mu, yy[k] + off, xerr=[[mu - lo], [hi - mu]], fmt=mk,
                        color=col, markersize=9, capsize=4, lw=1.8,
                        label=lab if k == 0 else None)
            ax.text(hi + 0.006, yy[k] + off, f"{mu:.3f}", va="center",
                    fontsize=10, color=col)
    ax.set_yticks(yy)
    ax.set_yticklabels(["PHE\nOnly", "PHE+\nPOAAGG", "PHE+\nMEGA",
                        "PHE+\nPGS526", "PHE+\nPGS616"], fontsize=11)
    ax.set_xlabel("AUC (participant-level bootstrap mean, 95% CI)",
                  fontsize=12)
    ax.set_xlim(0.54, 0.96)   # 2026-09-10: room for the legend left of the lowest bars
    ax.set_ylim(-0.6, len(ASYM_G) - 0.4)
    ax.grid(axis="x", ls="--", color="#CCCCCC", lw=0.9)
    ax.set_axisbelow(True)
    # keep the title inside the panel width; the full method name is in
    # the figure legend and the axis label.
    # fontsize 12.5 measured 15px wider than the panel canvas and clipped
    # the closing "I" of "95% CI"; 11.5 leaves a ~46px margin.
    ax.set_title("Training Cohort (N = 271) — MLP, bootstrap 95% CI\n"
                 "ΔIOP- and ΔCDR-based models",   # outcome is POAG
                 fontsize=11.5, fontweight="bold")
    # "lower right" sat on top of the PHE Only row's ΔIOP label (0.806,
    # the bottom row); "lower left" is clear of every marker and label.
    ax.legend(frameon=True, fontsize=11, loc="lower left", framealpha=0.95)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    label(fig, "B", x=0.02, y=0.99)


img_5b = render(draw_5b, B5_BOX, _os.path.join(tmp, "_p5b.png"))
_blank = Image.new("RGB", (B5_OLD_LABEL[2] - B5_OLD_LABEL[0],
                            B5_OLD_LABEL[3] - B5_OLD_LABEL[1]), "white")
sz = paste(F5, [(B5_OLD_LABEL, _blank), (B5_BOX, img_5b)])
print(f"  Figure_5_Combined.png rebuilt (B replaced, A and C kept) {sz}",
      flush=True)

print(f"\nR3 originals backed up to {_os.path.basename(BACKUP)}/", flush=True)
