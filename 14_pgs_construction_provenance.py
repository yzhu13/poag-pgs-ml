# =============================================================
#  POAG — PGS616 / PGS526 construction provenance
#  iScience R4 revision (2026-09-04)
#
#  Addresses Editor point 7 (and supports point 8):
#   "Fully describe PGS616 construction, including variant selection,
#    linkage-disequilibrium pruning, effect-size weighting,
#    ancestry-specific procedures, and training-data separation."
#
#  Builds an auditable variant-count waterfall from the discovery GWAS
#  through to the variants actually scored, plus a verification pass over
#  the deposited weight files.
#
#  SOURCES OF THE COUNTS (all documented, none inferred):
#   - per-paper locus counts and MEGA-overlap counts: analysis memo from
#     Y. Bradford (liftOver + extract_range.py step)
#   - deduplication: `sort -u -k7,7` over the concatenated overlap files
#   - genotype-level QC and scoring: PLINK log files retained under
#     _archive/04_iscience_R1_work_2026-04-05/.../POAAGG_PGS_data/
#       GRS_MainTable1_572snps.log        (622 -> 568 after --exclude)
#       GRS_MEGA_MainTable1_572snps.log   (568 processed, 49 unmatched)
#       GRS_MTAG_MainTable1_572snps.log   (486 processed, 41 unmatched)
#       missing.log                        (genotyping rate 0.930894)
#
#  NOTE ON FILE NAMING: the weight files are named "..._572snps.txt" for
#  historical reasons (572 was the deduplicated pool before the Verma
#  loci were added). They contain 616 and 526 variants respectively, and
#  are byte-identical to the files distributed as "..._616snps.txt".
#  This script verifies that directly.
#
#  Outputs:
#    outputs/tables/Table_PGS_Construction_Provenance.xlsx
#      A_Waterfall        variant counts at every step
#      B_ScoreComparison  PGS616 vs PGS526 set relationship
#      C_Software         tool versions and commands
#      D_Notes
# =============================================================

import os as _os
import hashlib
import numpy as np
import pandas as pd

HERE = _os.path.dirname(_os.path.abspath(__file__))
OUT_XL = _os.path.join(HERE, "outputs", "tables")
_os.makedirs(OUT_XL, exist_ok=True)

ROOT = _os.environ.get("POAG_PROJECT_DIR", ".")
W_MEGA = _os.path.join(ROOT, "_archive", "R1_work_2026-05-08", "input-data",
                       "POAAGG_cohort", "GRS_weight_MEGA_MainTable1_616snps.txt")
W_QUANT = _os.path.join(ROOT, "_archive", "R1_work_2026-05-08", "input-data",
                        "POAAGG_cohort", "GRS_weight_QUANT_MainTable1_616snps.txt")
W_MEGA_ALT = _os.path.join(ROOT, "data", "GRS_weight_MEGA_MainTable1_572snps.txt")
W_QUANT_ALT = _os.path.join(ROOT, "data", "GRS_weight_QUANT_MainTable1_572snps.txt")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════
#  A — variant-count waterfall
# ═══════════════════════════════════════════════════════════════
papers = pd.DataFrame([
    ("Craig et al. 2020",        "multi-ancestry POAG meta-analysis", 114, 104),
    ("Lo Faro et al. 2024",      "GBMI POAG meta-analysis (already GRCh38)", 62, 54),
    ("Gharahkhani et al. 2021",  "POAG/IOP, European + Asian",        127, 110),
    ("Han et al. 2023 (multi-ancestry)", "multi-ancestry meta-analysis", 312, 286),
    ("Han et al. 2023 (EUR)",    "European ancestry multitrait (MTAG)", 263, 236),
], columns=["Source GWAS", "Description",
            "Loci reported", "Retained after liftOver + MEGA overlap"])

tot_reported = int(papers["Loci reported"].sum())
tot_overlap = int(papers["Retained after liftOver + MEGA overlap"].sum())

waterfall = pd.DataFrame([
    ("1. Genome-wide significant loci extracted from five discovery "
     "datasets (Han et al. reported a multi-ancestry and a European "
     "ancestry analysis separately)", tot_reported,
     "Craig, Lo Faro, Gharahkhani, Han multi-ancestry, Han EUR (upper block)"),
    ("2. Coordinates lifted GRCh37 -> GRCh38", tot_reported,
     "UCSC liftOver, chain hg19ToHg38.over.chain; Lo Faro table already GRCh38"),
    ("3. Retained if present in the MEGA African ancestry summary "
     "statistics", tot_overlap,
     "extract_range.py against the MEGA SAIGE dosage file"),
    ("4. Deduplicated across the five sources", 572,
     "sort -u -k7,7 over the concatenated per-paper overlap files"),
    ("5. Verma et al. 2024 African ancestry POAG loci added", 618,
     "572 + 46 POAAGG/MEGA loci"),
    ("6a. Variants with usable MEGA case-control effect sizes -> PGS616",
     616, "weight file GRS_weight_MEGA_MainTable1 (verified below)"),
    ("6b. Variants with usable POAAGG MTAG effect sizes -> PGS526",
     526, "weight file GRS_weight_QUANT_MainTable1 (verified below)"),
    ("7. Genotype records extracted from the POAAGG panel by position (a "
     "record count: a multi-allelic site contributes one record per "
     "alternate allele)", 622,
     "PLINK extract, GRS_MainTable1_572snps_temp (616 distinct positions)"),
    ("8. Records passing genotype QC (records removed chiefly for "
     "missingness >= 15%)", 568,
     "PLINK1.9 --missing, then --exclude exclude_missing.txt (54 records); "
     "overall genotyping rate 0.9309"),
    ("9a. Variants actually scored for PGS616 in POAAGG", 568,
     "PLINK2 --score, 49 weight-file entries unmatched"),
    ("9b. Variants actually scored for PGS526 in POAAGG", 486,
     "PLINK2 --score, 41 weight-file entries unmatched"),
], columns=["Step", "Variant count", "Provenance"])


# ═══════════════════════════════════════════════════════════════
#  B — set relationship between the two scores  (verified from files)
# ═══════════════════════════════════════════════════════════════
mega = pd.read_csv(W_MEGA, sep=r"\s+")
quant = pd.read_csv(W_QUANT, sep=r"\s+")
s_mega, s_quant = set(mega["SNP"]), set(quant["SNP"])

rel = pd.DataFrame([
    ("Variants in PGS616 weight file", len(mega), ""),
    ("Variants in PGS526 weight file", len(quant), ""),
    ("Unique variant IDs, PGS616", len(s_mega),
     "no duplicates" if len(s_mega) == len(mega) else "DUPLICATES PRESENT"),
    ("Unique variant IDs, PGS526", len(s_quant),
     "no duplicates" if len(s_quant) == len(quant) else "DUPLICATES PRESENT"),
    ("In PGS526 but not PGS616", len(s_quant - s_mega),
     "PGS526 is a strict subset of PGS616"
     if not (s_quant - s_mega) else "not nested"),
    ("In PGS616 but not PGS526", len(s_mega - s_quant),
     "disease-status loci without a usable MTAG weight"),
    ("Shared variants", len(s_mega & s_quant),
     "same variants, different effect-size source"),
], columns=["Quantity", "Count", "Comment"])

files = pd.DataFrame([
    ("GRS_weight_MEGA_MainTable1_616snps.txt", md5(W_MEGA), len(mega)),
    ("GRS_weight_MEGA_MainTable1_572snps.txt", md5(W_MEGA_ALT), None),
    ("GRS_weight_QUANT_MainTable1_616snps.txt", md5(W_QUANT), len(quant)),
    ("GRS_weight_QUANT_MainTable1_572snps.txt", md5(W_QUANT_ALT), None),
], columns=["File", "MD5", "Variants"])
files["Note"] = [
    "PGS616 weights (MEGA African ancestry case-control betas)",
    "identical to the file above; '572snps' is a legacy filename",
    "PGS526 weights (POAAGG MTAG betas)",
    "identical to the file above; '572snps' is a legacy filename",
]


# ═══════════════════════════════════════════════════════════════
#  C — software and commands
# ═══════════════════════════════════════════════════════════════
software = pd.DataFrame([
    ("Coordinate conversion", "UCSC liftOver",
     "liftOver <in> hg19ToHg38.over.chain <out> <unmapped>"),
    ("Overlap with MEGA summary statistics", "extract_range.py (in-house)",
     "--chrom-col CHR --pos-col POS against the MEGA SAIGE dosage file"),
    ("Deduplication", "GNU coreutils", "sort -u -k7,7"),
    ("Genotype QC", "PLINK v1.90p (16 Apr 2021)",
     "--missing ; --exclude exclude_missing.txt --make-bed"),
    ("Scoring", "PLINK v2.00a6LM (4 Aug 2024)",
     "--score <weights> 1 3 4 cols=+scoresums   (PGS616); "
     "--score <weights> 1 2 3 cols=+scoresums   (PGS526)"),
    ("Missing dosage handling", "PLINK v2.00 --score default",
     "mean imputation to the cohort allele frequency for variants "
     "surviving QC"),
    ("Standardisation", "R",
     "SCORE1_AVG_STD = qnorm((rank(SCORE1_AVG, na.last='keep') - 0.5) / "
     "sum(!is.na(SCORE1_AVG)))"),
    ("Genome-wide scores (POAAGG PGS, MEGA PGS)", "PRS-CS",
     "phi=auto, 1000 burn-in, 2000 MCMC, 1000 Genomes AFR LD panel"),
], columns=["Step", "Tool", "Command / parameters"])


notes = pd.DataFrame({"Note": [
    "Table: Construction provenance for the curated loci-based polygenic "
    "scores PGS616 and PGS526.",
    "",
    "LINKAGE DISEQUILIBRIUM. No additional LD pruning was applied. The "
    "curated variants are lead SNPs reported by the discovery GWAS, each of "
    "which applied its own LD-based clumping, so the panel is approximately "
    "LD-independent by construction. No pruning step appears anywhere in the "
    "processing pipeline, and none is claimed.",
    "",
    "EFFECT-SIZE WEIGHTING. PGS616 uses African ancestry case-control effect "
    "sizes from the MEGA GWAS (N=11,275). PGS526 uses effect sizes from the "
    "POAAGG multi-trait analysis of GWAS (MTAG), which integrates IOP, CDR "
    "and RNFL endophenotypes with POAG case-control status. The two scores "
    "therefore differ principally in the SOURCE OF THE WEIGHTS rather than in "
    "variant membership: PGS526 is a strict subset of PGS616 (sheet B).",
    "",
    "STANDARDISATION. Scores were rank-based inverse normal transformed "
    "WITHIN each cohort in which they were computed (sheet C). This is a "
    "within-sample rank transform; no transformation parameters are carried "
    "across cohorts, so no information passes from the training cohort to the "
    "suspect or external validation cohorts through the standardisation step. "
    "Scores are consequently cohort-relative and are not comparable on an "
    "absolute scale between cohorts.",
    "",
    "TRAINING-DATA SEPARATION. The 271-subject machine learning training "
    "cohort was excluded from the POAAGG GWAS used to derive the POAAGG PGS "
    "and from the POAAGG component of the MEGA mega-analysis used to derive "
    "the MEGA PGS and the PGS616 weights, so no individual in the training "
    "cohort contributed to any effect-size estimate used to score them. The "
    "curated variant selection derives from external GWAS and was not "
    "optimised against the training cohort.",
    "",
    "SCORED VERSUS NOMINAL VARIANT COUNTS. The scores are named for the "
    "number of variants carrying usable weights (616 and 526). In the POAAGG "
    "genotype panel, 568 and 486 of those variants respectively were present "
    "and passed missingness QC, and those are the variants actually "
    "contributing to each score (sheet A, steps 7-9).",
    "",
    "OPEN ITEM. The PLINK2 scoring log reports 49 unmatched weight-file "
    "entries and 568 variants processed for PGS616 (616 - 49 = 567). The "
    "one-variant discrepancy most likely reflects a variant identifier "
    "matching more than one entry in the genotype panel. Rerunning --score "
    "with the 'list-variants' modifier would enumerate the scored variants "
    "exactly; this should be done before the counts in step 9a are quoted in "
    "the manuscript to more than one significant figure.",
]})

with pd.ExcelWriter(_os.path.join(
        OUT_XL, "Table_PGS_Construction_Provenance.xlsx"),
        engine="openpyxl") as w:
    papers.to_excel(w, sheet_name="A_Waterfall", index=False, startrow=0)
    waterfall.to_excel(w, sheet_name="A_Waterfall", index=False,
                       startrow=len(papers) + 3)
    rel.to_excel(w, sheet_name="B_ScoreComparison", index=False)
    files.to_excel(w, sheet_name="B_ScoreComparison", index=False,
                   startrow=len(rel) + 3)
    software.to_excel(w, sheet_name="C_Software", index=False)
    notes.to_excel(w, sheet_name="D_Notes", index=False)

print("=== Variant-count waterfall ===", flush=True)
for _, r in waterfall.iterrows():
    print(f"  {r['Step'][:68]:70s} {r['Variant count']}", flush=True)
print("\n=== Score set relationship (verified from weight files) ===",
      flush=True)
for _, r in rel.iterrows():
    print(f"  {r['Quantity']:36s} {r['Count']:6d}   {r['Comment']}", flush=True)
print("\n=== File identity check ===", flush=True)
same_mega = files.loc[0, "MD5"] == files.loc[1, "MD5"]
same_quant = files.loc[2, "MD5"] == files.loc[3, "MD5"]
print(f"  MEGA  '616snps' == '572snps' : {same_mega}", flush=True)
print(f"  QUANT '616snps' == '572snps' : {same_quant}", flush=True)

print("\nSaved Table_PGS_Construction_Provenance.xlsx", flush=True)
