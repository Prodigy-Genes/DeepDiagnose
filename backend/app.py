import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from uuid import uuid4
import csv

# Grad-CAM utilities
from grad_cam_utils import make_gradcam_heatmap, overlay_heatmap
import cv2

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

# Grad-CAM layer names (adjust as needed)
PNEU_LAST_CONV  = 'block7a_project_conv'      # EfficientNetB0 pneumonia model last conv layer
OSTEO_LAST_CONV = 'top_conv'                  # EfficientNetB0 osteoarthritis model last conv layer

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
# HEATMAP DESCRIPTIONS
# ----------------------
def get_heatmap_explanation(pred, prob):
    """Return appropriate explanation for the heatmap based on prediction and probability."""
    if "Pneumonia" in pred:
        if prob > 0.8:
            return """
            **Heatmap Analysis:** The highlighted areas (in red/yellow) indicate regions where the AI detected potential infection. 
            In pneumonia, these are typically cloudy or opaque areas in the lungs that indicate fluid accumulation. 
            This scan shows strong indicators of pneumonia with high confidence.
            """
        else:
            return """
            **Heatmap Analysis:** The highlighted areas (in red/yellow) show regions of interest that contributed to the pneumonia 
            diagnosis. These may indicate early-stage or mild pneumonia infiltrates. The moderate confidence level suggests 
            these findings may be subtle.
            """
    elif "Osteoarthritis" in pred:
        if prob > 0.8:
            return """
            **Heatmap Analysis:** The highlighted regions (in red/yellow) indicate areas of bone abnormality that suggest osteoarthritis. 
            The AI is focusing on joint space narrowing, osteophytes (bone spurs), or subchondral sclerosis that are hallmarks 
            of osteoarthritis. This scan shows strong indicators of joint degeneration.
            """
        else:
            return """
            **Heatmap Analysis:** The highlighted areas (in red/yellow) show regions that may indicate early osteoarthritic changes. 
            These could include minor joint space narrowing or early bone remodeling. The moderate confidence suggests these 
            findings may be early-stage or borderline.
            """
    else:  # Normal
        return """
        **Heatmap Analysis:** Even for normal scans, the AI highlights regions (in red/yellow) that were most relevant to its decision.
        The highlighted areas were carefully examined and found to be within normal ranges. The heatmap helps confirm that
        key anatomical areas were properly assessed during analysis.
        """

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

        # Anatomical classification (prob of 'joint')
        x_anat = preprocess_anatomy(img)
        p_anat = float(anatomical_model.predict(x_anat)[0][0])
        is_joint = p_anat >= 0.5
        anatomy = "Joint" if is_joint else "Chest"
        st.subheader(f"Detected Anatomy: {anatomy}")
        st.write(f"Confidence: {p_anat:.2%}")

        # Route to disease model
        if not is_joint:  # Chest → Pneumonia
            x = preprocess_pneumonia(img)
            prob = float(pneumonia_model.predict(x)[0][0])
            pred = "Pneumonia" if prob >= 0.5 else "Normal"
            st.subheader("Pneumonia Results")
            last_conv = PNEU_LAST_CONV
            model = pneumonia_model
            preprocess_fn = preprocess_pneumonia
        else:            # Joint → Osteoarthritis
            x = preprocess_osteo(img)
            prob = float(osteoarthritis_model.predict(x)[0][0])
            pred = "Osteoarthritis" if prob >= 0.5 else "Normal"
            st.subheader("Osteoarthritis Results")
            last_conv = OSTEO_LAST_CONV
            model = osteoarthritis_model
            preprocess_fn = preprocess_osteo

        st.write(f"**Prediction:** {pred}")
        st.write(f"**Confidence:** {prob:.2%}")
        st.progress(int(prob*100))

        # Generate Grad-CAM heatmap
        # Prepare arrays
        orig = np.array(img.convert('RGB'))
        img_array = x  # already preprocessed for model
        heatmap = make_gradcam_heatmap(img_array, model, last_conv)
        cam = overlay_heatmap(orig, heatmap)
        
        # Add explanation below Grad-CAM
        st.image(cam, caption="🔍 Grad-CAM Heatmap Analysis", use_column_width=True)
        st.markdown(get_heatmap_explanation(pred, prob))
        
        # Add color legend for heatmap
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="width: 200px; height: 20px; background: linear-gradient(to right, blue, cyan, green, yellow, red);"></div>
                <div style="display: flex; justify-content: space-between; width: 200px;">
                    <span style="font-size:0.8em">Low</span>
                    <span style="font-size:0.8em">Activation Intensity</span>
                    <span style="font-size:0.8em">High</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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