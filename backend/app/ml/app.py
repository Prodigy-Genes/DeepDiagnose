"""
Streamlit application to accept multiple X-ray image uploads and automatically detect anatomy (chest vs joint)
then run the appropriate disease classifier:
- Pneumonia vs Normal (Chest model)
- Osteoarthritis vs Normal (Joint model)

Supports up to 5 images at once, and captures user feedback for fine-tuning.
"""

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from uuid import uuid4
import csv

# ----------------------
# CONFIGURATION
# ----------------------
st.set_page_config(page_title="X-ray Multi-Disease Classifier", layout="centered")

# Model paths
MODELS_DIR        = Path(__file__).resolve().parent.parent / "models"
PNEU_MODEL_PATH   = MODELS_DIR / "pneumonia_classifier.keras"
OSTEO_MODEL_PATH  = MODELS_DIR / "osteo_efficientnetb0.keras"
ANAT_MODEL_PATH   = MODELS_DIR / "anatomical_classifier.keras"

# Input sizes for disease models
PNEU_SIZE  = (97, 132)
OSTEO_SIZE = (224, 224)

# Feedback dirs & log
FEEDBACK_DIR = Path(__file__).resolve().parent.parent / "feedback"
LOG_PATH     = FEEDBACK_DIR / "feedback_log.csv"
for sub in ("pneu", "osteo", "normal"): 
    (FEEDBACK_DIR / sub).mkdir(parents=True, exist_ok=True)
if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted", "corrected"])

# ----------------------
# LOAD MODELS
# ----------------------
@st.cache_resource
def load_models():
    pneu_model = tf.keras.models.load_model(str(PNEU_MODEL_PATH))
    osteo_model = tf.keras.models.load_model(str(OSTEO_MODEL_PATH))
    anat_model = tf.keras.models.load_model(str(ANAT_MODEL_PATH))
    return pneu_model, osteo_model, anat_model

pneumonia_model, osteoarthritis_model, anatomical_model = load_models()

# Determine anatomical model input size dynamically
anat_input_shape = anatomical_model.input_shape  # (None, height, width, channels)
anat_height, anat_width = anat_input_shape[1], anat_input_shape[2]

# ----------------------
# HELPER: VALIDATE IMAGE
# ----------------------
def is_likely_xray(image: Image.Image) -> bool:
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        if np.std([r-g, r-b, g-b]) > 15:
            return False
    gray = np.mean(arr, axis=-1) if arr.ndim == 3 else arr
    if gray.mean() < 30 or gray.mean() > 220:
        return False
    return True

# ----------------------
# PREPROCESSING FUNCTIONS
# ----------------------
def preprocess_pneumonia(img: Image.Image) -> np.ndarray:
    img = img.convert('L').resize(PNEU_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, PNEU_SIZE[0], PNEU_SIZE[1], 1)


def preprocess_osteo(img: Image.Image) -> np.ndarray:
    img = img.convert('L').resize(OSTEO_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.stack([arr]*3, axis=-1)
    arr = preprocess_efficientnet(arr)
    return arr.reshape(1, OSTEO_SIZE[0], OSTEO_SIZE[1], 3)


def preprocess_anatomy(img: Image.Image) -> np.ndarray:
    img = img.convert('L').resize((anat_width, anat_height))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, anat_height, anat_width, 1)

# ----------------------
# UI
# ----------------------
st.title("X-ray Multi-Disease Classifier 🩺")
st.write(
    "Upload up to 5 X-ray images; we'll detect anatomy and run the appropriate disease classifier."
)

uploaded_files = st.file_uploader(
    "Choose up to 5 X-ray images (PNG/JPEG)",
    type=["png", "jpeg", "jpg"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files[:5]:
        st.markdown("---")
        st.write(f"## Processing: {f.name}")
        img = Image.open(f)

        if not is_likely_xray(img):
            st.warning("Not a valid X-ray. Skipping.")
            continue
        st.image(img, use_column_width=True)

        # Anatomical classification (output=prob of 'joint')
        x_anat = preprocess_anatomy(img)
        p_anat = float(anatomical_model.predict(x_anat)[0][0])
        is_joint = p_anat >= 0.5
        anatomy = "Joint" if is_joint else "Chest"
        st.subheader(f"Detected Anatomy: {anatomy}")
        st.write(f"Confidence: {p_anat:.2%}")

        # Route to the correct disease model
        if not is_joint:  # Chest
            x = preprocess_pneumonia(img)
            prob = float(pneumonia_model.predict(x)[0][0])
            pred = "Pneumonia" if prob >= 0.5 else "Normal"
            st.subheader("Pneumonia Results")
        else:            # Joint
            x = preprocess_osteo(img)
            prob = float(osteoarthritis_model.predict(x)[0][0])
            pred = "Osteoarthritis" if prob >= 0.5 else "Normal"
            st.subheader("Osteoarthritis Results")

        st.write(f"**Prediction:** {pred}")
        st.write(f"**Confidence:** {prob:.2%}")
        st.progress(int(prob*100))

        # Feedback options
        options = [pred, "Normal", "Pneumonia", "Osteoarthritis"]
        feedback = st.selectbox("Is this prediction correct?", options, key=f.name)
        if feedback != pred:
            if st.button("Submit Correction", key=f.name+"btn"):
                actual = feedback.lower().replace(" ", "")
                folder = "pneu" if "pneu" in actual else "osteo" if "osteo" in actual else "normal"
                fname = f"{actual}_{uuid4().hex}.png"
                path = FEEDBACK_DIR / folder / fname
                img.convert('L').save(path)
                with open(LOG_PATH, "a", newline="") as log:
                    writer = csv.writer(log)
                    writer.writerow([fname, pred.lower(), actual])
                st.success("Saved correction.")

st.markdown("---")
st.write("Built with Streamlit & TensorFlow. Your feedback drives improvement.")
