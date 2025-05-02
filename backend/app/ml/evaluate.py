from pathlib import Path
import csv

# Model paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PNEU_MODEL_PATH  = MODELS_DIR / "pneumonia_classifier1.keras"
OSTEO_MODEL_PATH = MODELS_DIR / "osteo_efficientnetb0.keras"
ANAT_MODEL_PATH  = MODELS_DIR / "anatomical_classifier.keras"

# Input sizes
PNEU_SIZE  = (97, 132)
OSTEO_SIZE = (224, 224)

# Last convolutional layer names
PNEU_LAST_CONV  = 'block7a_project_conv'
OSTEO_LAST_CONV = 'top_conv'

# Contoured spot visualization parameters
CONTOUR_PARAMS = {
    "pneumonia": {"threshold": 0.35, "alpha": 0.6,  "color_scheme": "hot",     "max_spots": 8},
    "osteoarthritis": {"threshold": 0.4,  "alpha": 0.55, "color_scheme": "viridis", "max_spots": 8}
}

# Classification thresholds
CLASSIFICATION_THRESHOLDS = {
    "pneumonia":     {"confident": 0.88, "probable": 0.75, "uncertain": 0.60},
    "osteoarthritis": {"confident": 0.90, "probable": 0.80, "uncertain": 0.70}
}

# Feedback logging
FEEDBACK_DIR = BASE_DIR / "feedback"
LOG_PATH     = FEEDBACK_DIR / "feedback_log.csv"

# Initialize feedback directories and log file
for sub in ("pneu", "osteo", "normal"):
    (FEEDBACK_DIR / sub).mkdir(parents=True, exist_ok=True)
if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["filename", "predicted", "corrected"])
