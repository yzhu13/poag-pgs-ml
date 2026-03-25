"""
config.py
---------
Central configuration for the POAG PGS + ML analysis pipeline.

Edit the DATA_DIR and file name constants below to match your local setup
before running any analysis script.
"""

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths — set DATA_DIR to the folder containing your input Excel files
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"

# Input file names (place files in DATA_DIR)
TRAIN_FILE   = DATA_DIR / "271_training_cohort.xlsx"
TEST_1088_FILE = DATA_DIR / "1088_testing_cohort.xlsx"
TEST_1013_FILE = DATA_DIR / "1013_testing_suspects.xlsx"
PRS526_LOCI  = DATA_DIR / "MTAG_PRS_526_loci.txt"
PRS616_LOCI  = DATA_DIR / "MEGA_PRS_616_loci.txt"

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
# Training cohort
LABEL_COL     = "CaseCtrl"          # 1 = case, 0 = control

# Testing cohort
STATUS_COL    = "Disease status"    # values: "Case", "Control", "Suspect"

# Feature columns
AGE_COL       = "Age"
SEX_COL       = "Gender"
POAAGG_PRS    = "POAAGG PRS"
MEGA_PRS      = "MEGA PRS"
PRS526_COL    = "PRS526"
PRS616_COL    = "PRS616"

# Asymmetry columns
DELTA_IOP_COL  = "delta_IOP"
DELTA_CDR_COL  = "delta_CDR"
DELTA_RNFL_COL = "delta_RNFL"

# Clinical severity markers (for enrichment analysis)
IOP_SEVERE  = "IOP_SEVERE"
CDR_SEVERE  = "CDR_SEVERE"
RNFL_SEVERE = "RNFL_SEVERE"

CLINICAL_MARKERS = {
    "IOP_max":  IOP_SEVERE,
    "CDR_max":  CDR_SEVERE,
    "RNFL_min": RNFL_SEVERE,
}

CLINICAL_YLIMS = {
    IOP_SEVERE:  (10, 30),
    CDR_SEVERE:  (0.4, 0.8),
    RNFL_SEVERE: (70, 105),
}

# ---------------------------------------------------------------------------
# Feature sets for main analysis (Table 2 / Table 3)
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "Age":              [AGE_COL],
    "Gender":           [SEX_COL],
    "base":             [AGE_COL, SEX_COL],
    "POAAGG PRS":       [POAAGG_PRS],
    "MEGA PRS":         [MEGA_PRS],
    "PRS526":           [PRS526_COL],
    "PRS616":           [PRS616_COL],
    "PRS":              [POAAGG_PRS, MEGA_PRS, PRS526_COL, PRS616_COL],
    "base+POAAGG PRS":  [AGE_COL, SEX_COL, POAAGG_PRS],
    "base+MEGA PRS":    [AGE_COL, SEX_COL, MEGA_PRS],
    "base+PRS526":      [AGE_COL, SEX_COL, PRS526_COL],
    "base+PRS616":      [AGE_COL, SEX_COL, PRS616_COL],
    "base+PRS":         [AGE_COL, SEX_COL, POAAGG_PRS, MEGA_PRS, PRS526_COL, PRS616_COL],
    "all":              [AGE_COL, SEX_COL, POAAGG_PRS, MEGA_PRS, PRS526_COL, PRS616_COL],
}

# Feature set used for clinical enrichment scatter plots
ENRICHMENT_FEATURE_SET = "base+PRS616"

# The four PRS-only feature sets (for Figure 2 grouped bar chart)
PRS_ONLY_SETS = ["POAAGG PRS", "MEGA PRS", "PRS526", "PRS616"]

# ---------------------------------------------------------------------------
# Feature sets for asymmetry analysis (Figure 5)
# ---------------------------------------------------------------------------
ASYMMETRY_FEATURE_SETS = {
    # Each asymmetry measure alone
    "delta_IOP":  [DELTA_IOP_COL],
    "delta_CDR":  [DELTA_CDR_COL],
    "delta_RNFL": [DELTA_RNFL_COL],
    # Each asymmetry measure + each PRS
    "delta_IOP+POAAGG PRS":  [DELTA_IOP_COL,  POAAGG_PRS],
    "delta_IOP+MEGA PRS":    [DELTA_IOP_COL,  MEGA_PRS],
    "delta_IOP+PRS526":      [DELTA_IOP_COL,  PRS526_COL],
    "delta_IOP+PRS616":      [DELTA_IOP_COL,  PRS616_COL],
    "delta_CDR+POAAGG PRS":  [DELTA_CDR_COL,  POAAGG_PRS],
    "delta_CDR+MEGA PRS":    [DELTA_CDR_COL,  MEGA_PRS],
    "delta_CDR+PRS526":      [DELTA_CDR_COL,  PRS526_COL],
    "delta_CDR+PRS616":      [DELTA_CDR_COL,  PRS616_COL],
    "delta_RNFL+POAAGG PRS": [DELTA_RNFL_COL, POAAGG_PRS],
    "delta_RNFL+MEGA PRS":   [DELTA_RNFL_COL, MEGA_PRS],
    "delta_RNFL+PRS526":     [DELTA_RNFL_COL, PRS526_COL],
    "delta_RNFL+PRS616":     [DELTA_RNFL_COL, PRS616_COL],
    # All three asymmetry measures + best PRS
    "delta_all+PRS616":      [DELTA_IOP_COL, DELTA_CDR_COL, DELTA_RNFL_COL, PRS616_COL],
}

# Feature set used to predict suspect risk in the asymmetry validation
ASYMMETRY_RISK_FEATURES = [DELTA_IOP_COL, DELTA_CDR_COL, DELTA_RNFL_COL, PRS616_COL]

# ---------------------------------------------------------------------------
# Model definitions (shared across analyses)
# ---------------------------------------------------------------------------
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=1.0,
        bootstrap=True,
        class_weight="balanced_subsample",
        ccp_alpha=0.001,
        random_state=42,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        batch_size=32,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.20,
        random_state=42,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=4.0,
        gamma=2.0,
        class_weight="balanced",
        probability=True,
        tol=1e-4,
        cache_size=1000,
        random_state=42,
    ),
    "Logistic Regression": LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
}

# Number of bootstrap iterations for internal cross-validation
N_BOOTSTRAPS = 10

# Number of bootstrap iterations for asymmetry analysis
N_BOOTSTRAPS_ASYMMETRY = 50
