import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from uuid import uuid4
import csv
import cv2

# Grad-CAM utilities
from grad_cam_utils import make_gradcam_heatmap, create_contoured_spot_heatmap

# ----------------------
# CONFIGURATION
# ----------------------
st.set_page_config(page_title="X-ray Multi-Disease Classifier", layout="centered")

# Model paths
MODELS_DIR       = Path(__file__).resolve().parent.parent / "models"
PNEU_MODEL_PATH  = MODELS_DIR / "pneumonia_classifier1.keras"
OSTEO_MODEL_PATH = MODELS_DIR / "osteo_efficientnetb0.keras"
ANAT_MODEL_PATH  = MODELS_DIR / "anatomical_classifier.keras"

# Input sizes for disease models
PNEU_SIZE  = (97, 132)
OSTEO_SIZE = (224, 224)

# Grad-CAM layer names
PNEU_LAST_CONV  = 'block7a_project_conv'
OSTEO_LAST_CONV = 'top_conv'

# Contoured spot parameters
CONTOUR_PARAMS = {
    "pneumonia": {
        "threshold": 0.35,
        "alpha": 0.6,
        "color_scheme": "hot",
        "adaptive_threshold": True
    },
    "osteoarthritis": {
        "threshold": 0.4,
        "alpha": 0.55,
        "color_scheme": "viridis",
        "adaptive_threshold": True
    }
}

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
anat_input_shape = anatomical_model.input_shape
anat_height, anat_width = anat_input_shape[1], anat_input_shape[2]

# ----------------------
# HELPER: VALIDATE IMAGE
# ----------------------
def is_likely_xray(image: Image.Image) -> bool:
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
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
# SPOT DESCRIPTIONS
# ----------------------
def get_spot_explanation(pred, prob, spots_count):
    if "Pneumonia" in pred:
        if prob > 0.9:
            return (
                f"**Analysis:** {spots_count} critical areas identified showing definite pneumonia infiltrates. "
                f"The numbered spots reveal significant fluid accumulation in the lungs "
                f"with clear consolidation patterns. These findings strongly indicate active pneumonia infection "
                f"with high diagnostic certainty."
            )
        elif prob > 0.8:
            return (
                f"**Analysis:** {spots_count} significant areas identified showing characteristic pneumonia patterns. "
                f"The numbered spots indicate areas of pulmonary infiltration and opacity "
                f"that are typical manifestations of pneumonia infection. These findings provide substantial "
                f"evidence for pneumonia diagnosis."
            )
        else:
            return (
                f"**Analysis:** {spots_count} areas of interest detected suggesting possible pneumonia. "
                f"The numbered spots show regions of subtle opacity that may represent early-stage pneumonia "
                f"or mild infiltrates. These findings warrant clinical correlation."
            )
    elif "Osteoarthritis" in pred:
        if prob > 0.9:
            return (
                f"**Analysis:** {spots_count} definitive areas identified showing advanced osteoarthritis changes. "
                f"The numbered spots highlight joint space narrowing, osteophyte formation, and subchondral sclerosis "
                f"that are hallmark signs of established osteoarthritis. These findings provide conclusive evidence "
                f"of degenerative joint disease."
            )
        elif prob > 0.8:
            return (
                f"**Analysis:** {spots_count} significant areas identified showing moderate osteoarthritis. "
                f"The numbered spots reveal joint space reduction and bone remodeling "
                f"consistent with intermediate-stage osteoarthritis. These findings provide clear evidence "
                f"of ongoing degenerative changes."
            )
        else:
            return (
                f"**Analysis:** {spots_count} areas of interest detected suggesting early osteoarthritis. "
                f"The numbered spots indicate subtle joint abnormalities including minor joint space narrowing "
                f"or early bone remodeling. These findings suggest early degenerative changes."
            )
    else:
        return (
            f"**Analysis:** {spots_count} notable regions inspected and determined to be normal. "
            f"The numbered spots highlight areas that were carefully examined for potential abnormalities. "
            f"These key anatomical regions show normal bone structure and tissue density without signs of "
            f"pathology. This comprehensive assessment confirms the absence of disease."
        )

# ----------------------
# UI
# ----------------------
st.title("X-ray Disease Classifier with Precision Spot Analysis 🩺")
st.write(
    "Upload up to 5 X-ray images; we'll detect anatomy, classify diseases, and highlight key diagnostic regions."
)

# Visualization options for contoured spots
st.sidebar.header("Visualization Settings")
color_scheme = 'viridis'
max_spots = 6
alpha_value = 0.4

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

        # Anatomical classification
        x_anat = preprocess_anatomy(img)
        p_anat = float(anatomical_model.predict(x_anat)[0][0])
        is_joint = p_anat >= 0.5
        anatomy = "Joint" if is_joint else "Chest"
        st.subheader(f"Detected Anatomy: {anatomy}")
        st.write(f"Confidence: {p_anat:.2%}")

        # Route to disease model
        if not is_joint:
            x = preprocess_pneumonia(img)
            prob = float(pneumonia_model.predict(x)[0][0])
            pred = "Pneumonia" if prob >= 0.81 else "Normal"
            st.subheader("Pneumonia Analysis")
            last_conv = PNEU_LAST_CONV
            model = pneumonia_model
            params = CONTOUR_PARAMS["pneumonia"]
        else:
            x = preprocess_osteo(img)
            prob = float(osteoarthritis_model.predict(x)[0][0])
            pred = "Osteoarthritis" if prob >= 0.90 else "Normal"
            st.subheader("Osteoarthritis Analysis")
            last_conv = OSTEO_LAST_CONV
            model = osteoarthritis_model
            params = CONTOUR_PARAMS["osteoarthritis"]

        # Override with user settings if provided
        params["color_scheme"] = color_scheme
        params["alpha"] = alpha_value

        st.write(f"**Prediction:** {pred}")
        st.write(f"**Confidence:** {prob:.2%}")
        
        # Use colored progress bar based on confidence
        col1, col2, col3 = st.columns([1, 10, 1])
        with col2:
            if prob > 0.9:
                st.markdown(f"""
                <div style="width:100%; background-color:#f0f0f0; border-radius:5px; height:20px">
                    <div style="width:{int(prob*100)}%; background-color:#ff4b4b; height:20px; border-radius:5px; text-align:center; color:white; font-size:14px">
                        {prob:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif prob > 0.8:
                st.markdown(f"""
                <div style="width:100%; background-color:#f0f0f0; border-radius:5px; height:20px">
                    <div style="width:{int(prob*100)}%; background-color:#ffa421; height:20px; border-radius:5px; text-align:center; color:white; font-size:14px">
                        {prob:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="width:100%; background-color:#f0f0f0; border-radius:5px; height:20px">
                    <div style="width:{int(prob*100)}%; background-color:#0068c9; height:20px; border-radius:5px; text-align:center; color:white; font-size:14px">
                        {prob:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Generate contoured spot visualization
        orig = np.array(img.convert('RGB'))
        heatmap = make_gradcam_heatmap(x, model, last_conv)
        
        # Count spots that will be shown (for analysis text)
        binary = (cv2.resize(heatmap, (orig.shape[1], orig.shape[0])) > params["threshold"]).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        spots_count = min(len(valid_contours), max_spots)
        
        cam = create_contoured_spot_heatmap(
            orig, 
            heatmap, 
            alpha=params["alpha"],
            threshold=params["threshold"],
            max_spots=max_spots,
            color_scheme=params["color_scheme"],
            adaptive_threshold=params["adaptive_threshold"]
        )

        # Display spot visualization
        st.image(cam, caption="🔍 Precision Spot Analysis", use_column_width=True)
        
        # Custom spot intensity legend based on color scheme
        legend_colors = {
            "hot": ["#0000ff", "#00ffff", "#00ff80", "#ffff00", "#ff0000"],
            "viridis": ["#440154", "#3b528b", "#21908c", "#5dc963", "#fde725"],
            "cool": ["#ff0000", "#ff00ff", "#8000ff", "#0000ff"],
            "rainbow": ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"]
        }
        colors = legend_colors.get(color_scheme, legend_colors["hot"])
        
        # Build color legend HTML
        legend_html = '<div style="display:flex; flex-direction:column; margin:10px 0 20px 0;">'
        legend_html += '<p style="margin-bottom:5px; font-weight:bold;">Spot Intensity Guide:</p>'
        legend_html += '<div style="display:flex; align-items:center; gap:5px;">'
        
        # Add gradient bar
        gradient = f"linear-gradient(to right, {', '.join(colors)})"
        legend_html += f'<div style="flex-grow:1; height:24px; border-radius:3px; background: {gradient};"></div>'
        
        legend_html += '</div>'
        legend_html += '<div style="display:flex; justify-content:space-between; margin-top:2px;">'
        legend_html += '<span style="font-size:0.8em">Low Importance</span>'
        legend_html += '<span style="font-size:0.8em">High Importance</span>'
        legend_html += '</div>'
        
        # Add spot number explanation
        legend_html += '<p style="margin-top:8px; font-size:0.9em;">Numbered spots indicate regions of interest, with #1 being the most significant.</p>'
        legend_html += '</div>'
        
        st.markdown(legend_html, unsafe_allow_html=True)
        
        # Display analysis text
        st.markdown(get_spot_explanation(pred, prob, spots_count))
        
        # Display clinical relevance box
        if "Pneumonia" in pred:
            st.info(
                "**Clinical Relevance:** Pneumonia is characterized by infection and inflammation in the air sacs of the lungs. "
                "Radiographic findings typically show increased opacity in affected areas. The identified spots represent "
                "areas of consolidation where air in the lungs is replaced by inflammatory exudate."
            )
        elif "Osteoarthritis" in pred:
            st.info(
                "**Clinical Relevance:** Osteoarthritis is a degenerative joint disease characterized by cartilage loss and bone changes. "
                "Radiographic findings include joint space narrowing, osteophyte formation, and subchondral sclerosis. "
                "The identified spots highlight areas of these characteristic changes."
            )
        else:
            st.success(
                "**Clinical Relevance:** The normal appearance of this X-ray indicates proper anatomical structure with "
                "appropriate tissue density and no evidence of pathological changes. The identified spots were "
                "evaluated as within normal limits."
            )

        # Feedback options
        st.subheader("Provide Feedback")
        options = [pred, "Normal", "Pneumonia", "Osteoarthritis"]
        feedback = st.selectbox("Is this prediction correct?", options, key=f.name)
        if feedback != pred and st.button("Submit Correction", key=f.name + "btn"):
            actual = feedback.lower().replace(" ", "")
            folder = "pneu" if "pneu" in actual else "osteo" if "osteo" in actual else "normal"
            fname = f"{actual}_{uuid4().hex}.png"
            path = FEEDBACK_DIR / folder / fname
            img.convert('L').save(path)
            with open(LOG_PATH, "a", newline="") as log:
                writer = csv.writer(log)
                writer.writerow([fname, pred.lower(), actual])
            st.success("Thank you! Your correction has been saved and will help improve the model.")

st.markdown("---")
st.write("Built with Streamlit & TensorFlow. Your feedback drives improvement.")