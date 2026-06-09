# =============================================================
#  Build Document S1 — Supplemental Figures PDF
#  iScience format: one figure per page, legend text below each
#  Output: 5-8-revision-for-submission/Document_S1_Supplemental_Figures.pdf
# =============================================================
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import textwrap, os

import os as _os

# ── Configure your project root ─────────────────────────────────────────
# Set BASE to the folder containing input-data/, output excel data/, figure/
# Default: auto-detected as the repo root (2 levels above this script).
BASE = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")
)
# To override manually, uncomment:
# BASE = r"C:\path	o\your\project"   # Windows
# BASE = "/path/to/your/project"          # macOS / Linux
FIG_DIR = fr"{BASE}\5-18-revision-for-submission\Figures"
OUT_PDF = fr"{BASE}\5-18-revision-for-submission\Document_S1_Supplemental_Figures.pdf"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "pdf.fonttype": 42,   # TrueType embedding — makes text searchable/copyable
    "ps.fonttype": 42,
})

# ── Figure definitions: (filename, panel_label, full_legend) ──
# iScience legend format: bold label, colon, then description.
FIGURES = [
    (
        "SF1_circos_plot.png",
        "Figure S1. Chromosomal Distribution of Curated PGS Variants, Related to Figure 2",
        (
            "Circos plot showing the genomic distribution of SNPs included in the two "
            "curated locus-based polygenic risk scores (PGS526 and PGS616) across all 22 "
            "autosomes. The outermost ring displays the chromosomal ideogram. The second "
            "ring (teal ticks) shows the positions of PGS526 SNPs (n = 526) selected from "
            "six independent African ancestry POAG GWAS. The third ring (orange ticks) "
            "shows the positions of MEGA PGS variants. The fourth ring displays stacked "
            "variant density counts per chromosomal segment. The innermost ring shows the "
            "fraction of positive-effect-size (β > 0) variants per segment, split by score. "
            "SNP density is concentrated on chromosomes 1, 2, 4, 6, and 9, consistent with "
            "known POAG-associated loci. The circos layout confirms broad genomic coverage "
            "across all autosomes without regional clustering that might indicate spurious "
            "selection."
        ),
    ),
    (
        "SF2_Combined.png",
        "Figure S2. PC–PGS Correlations and Ancestry PC vs. PGS Performance, Related to Figure 3",
        (
            "(A) Heatmap of Pearson correlation coefficients between the first five ancestry "
            "principal components (PC1–PC5) and the four polygenic risk scores (MEGA PGS, "
            "PGS526, PGS616, POAAGG PGS) in the POAAGG training cohort (N = 271). "
            "Asterisks indicate statistical significance: *p < 0.05, **p < 0.01, "
            "***p < 0.001. The maximum observed correlation is |r| = 0.355 (PC3–PGS616), "
            "indicating that ancestry PCs and PGS capture largely distinct biological "
            "signals. PC3 shows the strongest correlations with the curated loci-based "
            "scores (PGS526: r = 0.29; PGS616: r = 0.35), while PC1, PC2, PC4, and PC5 "
            "show weak or non-significant correlations with all PGS. These results indicate "
            "that PGS and PCs are not redundant. "
            "(B) Horizontal dot plot showing cross-validated AUC (mean ± 95% CI, 5×20 CV) "
            "for six feature configurations across four classifiers and their average: Base "
            "(age+sex), Base+PC2, Base+PC5, Base+PC10, Base+PC20, and Base+PGS616. The red "
            "dashed reference line indicates AUC = 0.87, the value reported in a prior "
            "non-cross-validated (bootstrap) analysis for the age+sex+PC1–20 model. Under "
            "rigorous 5×20 CV, Base+PC20 achieves a mean AUC of only 0.657 (95% CI: "
            "0.645–0.669). Adding more PCs progressively reduces cross-validated AUC "
            "(Base+PC2: 0.704; Base+PC5: 0.688; Base+PC10: 0.672; Base+PC20: 0.657), "
            "consistent with overfitting as feature dimensionality grows relative to N = 271. "
            "Base+PGS616 (AUC = 0.700) outperforms Base+PC20 by +0.043. "
            "(C) Bar chart of PMBB external validation AUC (African ancestry, N = 12,042) "
            "for Base, Base+PGS526, Base+PGS616, Base+PC5, and Base+PC5+PGS616. PGS-"
            "augmented models maintain external performance (Base+PGS616: 0.700), while "
            "PC-augmented models show reduced external AUC (Base+PC5: 0.696 vs. Base: 0.716; "
            "Base+PC5+PGS616: 0.655), consistent with ancestry PCs derived from a small, "
            "ancestry-homogeneous training cohort not transferring robustly to an "
            "independent biobank."
        ),
    ),
    (
        "SF3_SHAP_Calibration.png",
        "Figure S3. SHAP Feature Importance and Model Calibration, Related to Figure 3",
        (
            "(A) SHAP beeswarm plot for the MLP classifier trained on the Base+PGS616 "
            "feature set (N = 271). Each dot represents one participant. The x-axis shows "
            "the SHAP value (impact on predicted POAG risk); the color indicates the "
            "feature value (red = high, blue = low). Age (years) is the most influential "
            "feature, showing a positive SHAP gradient consistent with age as the dominant "
            "POAG risk factor. Sex (male = 1) shows a bimodal pattern, with males receiving "
            "higher risk scores. PGS616 shows a predominantly positive SHAP contribution "
            "at higher PGS values, confirming that higher polygenic burden increases "
            "predicted risk. "
            "(B) Mean |SHAP| bar chart comparing feature importance across three MLP "
            "configurations: Base (age+sex), Base+PGS616, and Base+PC5+PGS616. For "
            "Base+PGS616, Age (0.127) > Sex (0.109) > PGS616 (0.042). When PCs are "
            "included (Base+PC5+PGS616), PC1 dominates (0.218), substantially attenuating "
            "the relative contribution of PGS616 (0.042 → 0.040), confirming that "
            "consistent with the possibility that ancestry PCs capture cohort structure "
            "or ancestry-related signal. "
            "(C) Calibration curves (reliability diagrams) for three MLP feature "
            "configurations under 5-fold cross-validation. The dashed diagonal represents "
            "perfect calibration. All models show reasonable calibration across the "
            "predicted probability range, with no systematic over- or under-estimation. "
            "Brier scores are shown in parentheses. "
            "(D) Brier score bar chart for all four classifiers (LR, SVM, RF, MLP) across "
            "three feature sets under 5×20 CV. Lower scores indicate better calibration. "
            "The dotted line at 0.249 represents the null model (predicting marginal "
            "prevalence for all subjects). All models substantially outperform the null "
            "model. In the 5-fold MLP calibration analysis, Base+PGS616 had the lowest "
            "Brier score (0.2140). In the 5×20 repeated-CV analysis, MLP showed a similar "
            "pattern, with Base+PGS616 slightly lower than Base."
        ),
    ),
    (
        "SF4_LearningCurves.png",
        "Figure S4. Learning Curves: Model Performance vs. Training Set Size, Related to Figure 3",
        (
            "(A) Learning curves showing cross-validation AUC (mean ± 95% CI) as a "
            "function of approximate training set size for three feature configurations "
            "(Base, Base+PGS616, Base+PC5+PGS616), averaged across all four classifiers. "
            "CV was performed using 5-fold × 4-repeat stratified cross-validation (20 fits "
            "per training-size step). Training fractions ranged from 30% to 100% of the "
            "per-fold training set (approximate N = 65 to 217). Base+PGS616 (teal) "
            "consistently outperforms Base (gray) across all training sizes. "
            "Base+PC5+PGS616 (red) shows low early performance that improves with more "
            "data, reflecting higher-dimensional overfitting at small N. CV AUC for "
            "Base+PGS616 plateaus around 60–70% of training data (N ≈ 130–150). "
            "(B) Training AUC (dotted lines) vs. cross-validation AUC (solid lines) for "
            "all three feature sets. Learning curves were averaged across the four classifiers. "
            "All models show small train-to-CV gaps at full N, indicating well-controlled "
            "overfitting. Base+PC5+PGS616 showed the largest train-to-CV gap, suggesting "
            "reduced stability when ancestry PCs were added."
        ),
    ),
    (
        "SF5_SexStratified.png",
        "Figure S5. Sex-Stratified Cross-Validation AUC, Related to Figure 3",
        (
            "Horizontal dot plot showing cross-validation AUC (mean ± 95% CI, 5×20 CV) "
            "stratified by sex (Female: N = 160, 60 cases; Male: N = 111, 68 cases) "
            "and overall (N = 271), averaged across all four classifiers (Average). "
            "Three feature configurations are shown: Base (age+sex), Base+PGS616, and "
            "Base+PC5+PGS616. In the Base model, females achieve higher AUC than males "
            "(Average: Female > Male), likely reflecting steeper age-related risk gradients "
            "in the female subgroup. Adding PGS616 narrows the sex gap, with the polygenic "
            "score providing proportionally greater lift in males. Notably, the "
            "Base+PC5+PGS616 model reverses this pattern — yielding substantially higher "
            "AUC in males than females — consistent with ancestry PC axes capturing "
            "may reflect sex-correlated cohort structure or ancestry-related variation. "
            "This sex-differential PC effect provides additional evidence supporting PGS "
            "over PCs as the primary genetic feature for generalizable POAG prediction."
        ),
    ),
    (
        "SF6A_SuppFig_Suspects_AllModels_Heatmap.png",
        "Figure S6A. Predicted Risk Correlations with Clinical Outcomes Across All Classifiers, Related to Figure 4",
        (
            "Heatmap of Pearson correlation coefficients between model-predicted POAG risk "
            "scores and three clinical outcome measures (IOP, CDR, RNFL thickness) in the "
            "POAAGG suspect cohort (N = 1,013), shown for all four classifiers (LR, SVM, "
            "RF, MLP) and five feature sets (Base, Base+POAAGG PGS, Base+MEGA PGS, "
            "Base+PGS526, Base+PGS616). CDR and RNFL thickness showed the most consistent "
            "significant positive and negative associations, respectively, with predicted "
            "risk across classifiers. IOP associations were generally weaker and "
            "non-significant in most configurations. Associations were broadly consistent "
            "across classifiers and feature sets, with CDR and RNFL showing the strongest "
            "clinical enrichment patterns, whereas IOP associations were weak. Color "
            "intensity represents the magnitude of the correlation; asterisks indicate "
            "statistical significance (*p < 0.05, **p < 0.01, ***p < 0.001)."
        ),
    ),
    (
        "SF6B_SuppFig_Suspects_MLP_Scatter.png",
        "Figure S6B. MLP Predicted Risk vs. Clinical Outcomes in Suspect Cohort, Related to Figure 4",
        (
            "Scatter plots showing the relationship between MLP-predicted POAG risk scores "
            "(x-axis) and three clinical measures (IOP, CDR, RNFL thickness; y-axis) in "
            "the POAAGG suspect cohort (N = 1,013) using the Base+PGS616 feature set. "
            "Each point represents one suspect participant. Regression lines with 95% "
            "confidence bands are shown. CDR showed a significant positive correlation "
            "with predicted risk (r = 0.121, p < 0.001), indicating that suspects with "
            "higher predicted POAG risk tended to have larger optic nerve cups. RNFL "
            "thickness showed a significant negative correlation (r = −0.205, p < 0.001, N = 483), "
            "consistent with progressive retinal nerve fiber layer thinning in higher-risk "
            "individuals. IOP did not reach statistical significance in the suspect cohort, "
            "consistent with the known decoupling of IOP from structural damage in "
            "glaucoma suspects. These associations support the biological plausibility of "
            "PGS-augmented risk scores for identifying suspects at elevated structural risk."
        ),
    ),
]


def wrap(text, width=115):
    return "\n".join(textwrap.wrap(text, width=width))


print(f"Building Document S1 ({len(FIGURES)} figures) ...")

with PdfPages(OUT_PDF) as pdf:
    for i, (fname, label, legend) in enumerate(FIGURES, 1):
        fpath = os.path.join(FIG_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found — skipping")
            continue

        img = mpimg.imread(fpath)
        h_px, w_px = img.shape[:2]
        aspect = w_px / h_px

        # Page layout: figure takes upper 78%, legend takes lower 22%
        fig = plt.figure(figsize=(11, 8.5))   # US letter landscape

        # Figure axes
        ax_img = fig.add_axes([0.03, 0.22, 0.94, 0.76])
        ax_img.imshow(img)
        ax_img.axis("off")

        # Legend text area (bottom 20%)
        ax_leg = fig.add_axes([0.03, 0.01, 0.94, 0.20])
        ax_leg.axis("off")

        # Bold label + regular body
        legend_full = legend
        label_short = label  # e.g. "Figure S1. Chromosomal..."

        ax_leg.text(0, 1.0, label_short,
                    transform=ax_leg.transAxes,
                    ha="left", va="top",
                    fontsize=8.5, fontweight="bold",
                    wrap=False)
        ax_leg.text(0, 0.82, wrap(legend_full, width=160),
                    transform=ax_leg.transAxes,
                    ha="left", va="top",
                    fontsize=7.8,
                    wrap=False)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i}/{len(FIGURES)}] {fname}")

    # PDF metadata
    d = pdf.infodict()
    d["Title"] = "Document S1 — Supplemental Figures"
    d["Author"] = "Yan Zhu et al."
    d["Subject"] = "POAG PGS ML — Supplemental Figures SF1–SF6B"
    d["Keywords"] = "POAG; polygenic risk score; machine learning; supplemental"

print(f"\nSaved: {OUT_PDF}")
