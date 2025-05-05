import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
import json

# Grad-CAM utilities
from grad_cam_utils import make_gradcam_heatmap, create_contoured_spot_heatmap, overlay_heatmap

# ----------------------
# CONFIGURATION & METRICS LOAD
# ----------------------
st.set_page_config(page_title="X-ray Disease Classifier 🩺", layout="centered")
# project structure
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
PNEU_METRICS = BASE_DIR / "pneu_metrics"
ANAT_METRICS = BASE_DIR / "ana_metrics"

# Load pneumonia metrics
def load_pneu_metrics():
    info = json.loads((PNEU_METRICS / "dataset_info.json").read_text())
    norm = json.loads((PNEU_METRICS / "normalization_stats.json").read_text())
    line = next(l for l in (PNEU_METRICS / "thresholds.txt").read_text().splitlines() if l.startswith("opt_threshold="))
    threshold = float(line.split('=')[1])
    size = (info['resize_to']['height'], info['resize_to']['width'])
    return {
        'size': size,
        'mean': norm.get('train_pixel_mean', 0.0),
        'std': norm.get('train_pixel_std', 1.0),
        'threshold': threshold,
        'last_conv': info.get('last_conv_layer', 'conv2d_2')
    }

# Load anatomy metrics
def load_anat_metrics():
    info = json.loads((ANAT_METRICS / "dataset_info.json").read_text())
    norm = json.loads((ANAT_METRICS / "normalization_stats.json").read_text())
    line = next(l for l in (ANAT_METRICS / "thresholds.txt").read_text().splitlines() if l.startswith("opt_threshold="))
    threshold = float(line.split('=')[1])
    size = (info['resize_to']['height'], info['resize_to']['width'])
    return {
        'size': size,
        'mean': norm.get('train_pixel_mean', 0.0),
        'std': norm.get('train_pixel_std', 1.0),
        'threshold': threshold
    }

pneu_m = load_pneu_metrics()
anat_m = load_anat_metrics()

# ----------------------
# MODEL LOADING
# ----------------------
@st.cache_resource
def load_models():
    anat = tf.keras.models.load_model(str(MODELS_DIR / "anatomical_classifier.keras"))
    pneu = tf.keras.models.load_model(str(MODELS_DIR / "pneumonia_classifier.keras"))
    osteo = tf.keras.models.load_model(str(MODELS_DIR / "osteo_efficientnetb0.keras"))
    return anat, pneu, osteo

anatomical_model, pneumonia_model, osteoarthritis_model = load_models()

# ----------------------
# PREPROCESSING & VALIDATION
# ----------------------
def is_likely_xray(img: Image.Image) -> bool:
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        diffs = [arr[:,:,0] - arr[:,:,1], arr[:,:,0] - arr[:,:,2], arr[:,:,1] - arr[:,:,2]]
        if np.std(diffs) > 15:
            return False
    gray = arr.mean(axis=-1) if arr.ndim == 3 else arr
    return 30 <= gray.mean() <= 220

# Pneumonia preprocessing
def preprocess_pneumonia(img: Image.Image):
    h, w = pneu_m['size']
    im = img.convert('L').resize((w, h))
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = (arr - pneu_m['mean']) / (pneu_m['std'] + 1e-8)
    return arr.reshape(1, h, w, 1)

# Osteoarthritis preprocessing
def preprocess_osteo(img: Image.Image):
    im = img.convert('L').resize((224, 224))
    arr_gray = np.array(im, dtype=np.float32)
    arr = np.stack([arr_gray] * 3, axis=-1)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(arr).reshape(1, 224, 224, 3)

# Anatomy preprocessing
def preprocess_anatomy(img: Image.Image):
    h, w = anat_m['size']
    im = img.convert('L').resize((w, h))
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = (arr - anat_m['mean']) / (anat_m['std'] + 1e-8)
    return arr.reshape(1, h, w, 1)

# Contour parameters
iCONTOUR_PARAMS = {
    'pneumonia': {'threshold': 0.2, 'alpha': 0.6, 'color_scheme': 'viridis', 'adaptive_threshold': True, 'min_spot_area': 5},
    'osteoarthritis': {'threshold': 0.4, 'alpha': 0.55, 'color_scheme': 'viridis', 'adaptive_threshold': True, 'min_spot_area': 50}
}

# ----------------------
# STREAMLIT LAYOUT
# ----------------------
ANAT_CONF_MARGIN = 0.85

st.title('X-ray Disease Classifier 🩺')
st.write('Upload up to 5 chest or knee X-ray images')
files = st.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if files:
    for f in files[:5]:
        img = Image.open(f)
        st.image(img, use_column_width=True)

        if not is_likely_xray(img):
            st.warning('Invalid X-ray. Please upload a chest or knee X-ray.')
            continue

        # Anatomy classification
        xa = preprocess_anatomy(img)
        pa = float(anatomical_model.predict(xa)[0, 0])
        # Determine class and confidence
        if pa >= anat_m['threshold']:
            is_joint = True
            an_conf = pa
        else:
            is_joint = False
            an_conf = 1 - pa
        anatomy = 'Joint' if is_joint else 'Chest'
        # Reject low-confidence anatomy
        if an_conf < ANAT_CONF_MARGIN:
            st.error(f"Anatomy uncertain ({an_conf:.1%}). Please upload a clear chest or knee X-ray.")
            continue

        st.write(f"**Anatomy:** {anatomy} ({an_conf:.1%})")

        # Select disease model
        if is_joint:
            x = preprocess_osteo(img)
            model = osteoarthritis_model
            params = iCONTOUR_PARAMS['osteoarthritis']
            label = 'Osteoarthritis'
            threshold = anat_m['threshold']
        else:
            x = preprocess_pneumonia(img)
            model = pneumonia_model
            params = iCONTOUR_PARAMS['pneumonia']
            label = 'Pneumonia'
            threshold = pneu_m['threshold']

        # Disease prediction
        p = float(model.predict(x)[0, 0])
        disease = label if p >= threshold else 'Normal'
        conf = p if p >= threshold else 1 - p
        st.write(f"**Prediction:** {disease} ({conf:.1%})")

        # Grad-CAM visualization
        orig = np.array(img.convert('RGB'))
        heat = make_gradcam_heatmap(x, model, pneu_m['last_conv'] if not is_joint else None)
        thr = params['threshold']
        if params['adaptive_threshold']:
            nz = heat[heat > 0]
            thr = float(np.clip(np.percentile(nz, 70) if nz.size else thr, 0.2, 0.7))
        cam_spots = create_contoured_spot_heatmap(
            orig, heat,
            alpha=params['alpha'], threshold=thr,
            max_spots=8, color_scheme=params['color_scheme'],
            adaptive_threshold=False, min_spot_area=params['min_spot_area']
        )
        cam = overlay_heatmap(orig, heat, alpha=0.4) if np.array_equal(cam_spots, orig) else cam_spots
        st.image(cam, caption='🔍 Model Focus', use_column_width=True)

st.markdown('---')
st.write('Built with Streamlit & TensorFlow.')

