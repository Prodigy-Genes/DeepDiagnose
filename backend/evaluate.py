import os
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODELS_DIR   = Path("models")
PNEU_MODEL   = load_model(MODELS_DIR / "pneumonia_classifier.keras")
OSTEO_MODEL  = load_model(MODELS_DIR / "osteo_efficientnetb0.keras")

# Input sizes (must match your preprocess logic)
PNEU_SIZE  = (97, 132)
OSTEO_SIZE = (224, 224)

# Test data root
TEST_DIR = Path("test")

# ─── PREPROCESS FUNCTIONS ─────────────────────────────────────────────────────
def preprocess_pneu(img_path):
    img = load_img(img_path, target_size=PNEU_SIZE, color_mode='grayscale')
    arr = img_to_array(img).astype("float32") / 255.0
    return arr.reshape(1, *PNEU_SIZE, 1)

def preprocess_osteo(img_path):
    img = load_img(img_path, target_size=OSTEO_SIZE, color_mode='grayscale')
    arr = img_to_array(img).astype("float32")
    arr = np.stack([arr]*3, axis=-1)  # make 3 channels
    from tensorflow.keras.applications.efficientnet import preprocess_input
    arr = preprocess_input(arr)
    return arr.reshape(1, *OSTEO_SIZE, 3)

# ─── EVALUATION ────────────────────────────────────────────────────────────────
def evaluate_model(model, disease, preprocess_fn, pos_label):
    """
    disease: subfolder name under TEST_DIR (e.g. "pneumonia" or "osteoarthritis")
    pos_label: the name of the positive class folder inside that (e.g. "PNEUMONIA" or "Osteoarthritis")
    """
    folder = TEST_DIR / disease
    # Determine negative class folder (the only other one)
    neg_labels = [d.name for d in folder.iterdir() if d.is_dir() and d.name != pos_label]
    if len(neg_labels) != 1:
        raise RuntimeError(f"Expected exactly one negative class folder in {folder}")
    neg_label = neg_labels[0]

    y_true, y_pred, y_prob = [], [], []
    for label, true_val in [(pos_label, 1), (neg_label, 0)]:
        for img_file in (folder/label).iterdir():
            x = preprocess_fn(str(img_file))
            prob = float(model.predict(x)[0][0])
            pred = int(prob >= 0.5)
            y_prob.append(prob)
            y_pred.append(pred)
            y_true.append(true_val)

    # Metrics
    print(f"\n=== {disease.upper()} MODEL ===")
    target_names = [neg_label, pos_label]
    print(classification_report(y_true, y_pred, target_names=target_names))
    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC AUC: {auc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{disease.capitalize()} ROC Curve")
    plt.legend()
    plt.savefig(f"{disease}_roc.png")
    plt.close()

if __name__ == "__main__":
    # Pneumonia uses uppercase folder names
    evaluate_model(
        model=PNEU_MODEL,
        disease="pneumonia",
        preprocess_fn=preprocess_pneu,
        pos_label="PNEUMONIA"
    )

    # Osteoarthritis uses “Normal” and “Osteoarthritis”
    evaluate_model(
        model=OSTEO_MODEL,
        disease="osteoarthritis",
        preprocess_fn=preprocess_osteo,
        pos_label="Osteoarthritis"
    )
