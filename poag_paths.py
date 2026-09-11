# =============================================================
#  POAG - one definition of where the input data lives
#  Set POAG_DATA_DIR to the folder holding POAAGG_cohort/ and
#  PMBB_external/; defaults to ./data next to this file.
# =============================================================

import os as _os

DATA_DIR = _os.environ.get(
    "POAG_DATA_DIR",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data"))

TRAINING_FILE = _os.path.join(DATA_DIR, "POAAGG_cohort",
                              "271_training_cohort_4_new_PRS_cleaned.xlsx")
SUSPECT_FILE = _os.path.join(DATA_DIR, "POAAGG_cohort",
                             "1013_testing_cohort_only_suspect_cleaned.xlsx")
