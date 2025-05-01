import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from uuid import uuid4
import csv
import cv2
import json
import pandas as pd

# Grad-CAM utilities
from grad_cam_utils import make_gradcam_heatmap, create_contoured_spot_heatmap

# ----------------------
# CONFIGURATION & METRICS LOAD
# ----------------------
st.set_page_config(page_title="X-ray Disease Classifier", layout="centered")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "pneu_metrics"

# Load sizing & normalization
dataset_info = json.loads((METRICS_DIR / "dataset_info.json").read_text())
norm_stats = json.loads((METRICS_DIR / "normalization_stats.json").read_text())
PNEU_SIZE = (dataset_info["resize_to"]["height"], dataset_info["resize_to"]["width"])
PIXEL_MEAN = norm_stats.get("train_pixel_mean", 0.0)
PIXEL_STD = norm_stats.get("train_pixel_std", 1.0)
PNEU_LAST_CONV = 'conv2d_2'

# Load thresholds
threshold = float([l for l in (METRICS_DIR / "thresholds.txt").read_text().splitlines() if l.startswith("opt_threshold=")][0].split('=')[1])
# Best-F1 contour threshold
scan_df = pd.read_csv(METRICS_DIR / "threshold_scan.csv")
contour_thresh = scan_df.loc[scan_df['f1_score'].idxmax(), 'threshold']

# Set static visualization params

CONTOUR_PARAMS = {
    "pneumonia": {"threshold": 0.35,  
                   "alpha": 0.6,
                   "color_scheme": "viridis",
                   "adaptive_threshold": True},
    "osteoarthritis": {
                     "threshold": 0.4,  
                     "alpha": 0.55,
                     "color_scheme": "viridis",
                     "adaptive_threshold": True}
     }
 


# Feedback dirs
FEEDBACK_DIR = BASE_DIR / "feedback"
LOG_PATH = FEEDBACK_DIR / "feedback_log.csv"
for sub in ("pneu","osteo","normal"): (FEEDBACK_DIR/sub).mkdir(exist_ok=True)
if not LOG_PATH.exists(): (LOG_PATH).write_text("filename,predicted,corrected\n")

# ----------------------
# LOAD MODELS
# ----------------------
@st.cache_resource
def load_models():
    pneu = tf.keras.models.load_model(str(MODELS_DIR / "pneumonia_classifier.keras"))
    osteo = tf.keras.models.load_model(str(MODELS_DIR / "osteo_efficientnetb0.keras"))
    anat = tf.keras.models.load_model(str(MODELS_DIR / "anatomical_classifier.keras"))
    return pneu, osteo, anat
pneumonia_model, osteoarthritis_model, anatomical_model = load_models()
ANAT_H, ANAT_W = anatomical_model.input_shape[1:3]

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def is_likely_xray(img):
    arr = np.array(img)
    if arr.ndim==3 and arr.shape[2]==3:
        if np.std([arr[:,:,0]-arr[:,:,1],arr[:,:,0]-arr[:,:,2],arr[:,:,1]-arr[:,:,2]])>15: return False
    gray = arr.mean(axis=-1) if arr.ndim==3 else arr
    return 30<=gray.mean()<=220


def get_spot_explanation(pred, prob, count):
    if "Pneumonia" in pred:
        return f"**Analysis:** {count} highlighted regions indicate pneumonia focus areas."
    if "Osteoarthritis" in pred:
        return f"**Analysis:** {count} highlighted regions indicate osteoarthritis focus areas."
    return f"**Analysis:** {count} regions inspected, no pathology detected."


def preprocess_pneumonia(img):
    im = img.convert('L').resize((PNEU_SIZE[1], PNEU_SIZE[0]))
    arr = np.array(im, np.float32)/255.0
    arr = (arr-PIXEL_MEAN)/(PIXEL_STD+1e-8)
    return arr.reshape(1,*PNEU_SIZE,1)


def preprocess_osteo(img):
    im = img.convert('L').resize((224,224))
    arr = np.stack([np.array(im)]*3, axis=-1).astype(np.float32)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(arr).reshape(1,224,224,3)


def preprocess_anatomy(img):
    im = img.convert('L').resize((ANAT_W,ANAT_H))
    return (np.array(im,np.float32)/255.0).reshape(1,ANAT_H,ANAT_W,1)

# ----------------------
# UI & INFERENCE
# ----------------------
st.title("X-ray Disease Classifier 🩺")
st.write("Upload up to 5 X-rays. We'll show model focus and analyses.")
files = st.file_uploader("Upload X-ray images", type=["png","jpg","jpeg"], accept_multiple_files=True)
if files:
    for f in files[:5]:
        img = Image.open(f)
        st.image(img,use_column_width=True)
        if not is_likely_xray(img): st.warning("Not a valid X-ray."); continue

        # anatomy check
        xa = preprocess_anatomy(img)
        pa = anatomical_model.predict(xa)[0,0]
        joint = pa>=0.5
        st.write(f"**Anatomy:** { 'Joint' if joint else 'Chest' } ({pa:.1%})")

        # select model
        if not joint:
            x = preprocess_pneumonia(img); model=pneumonia_model; last=PNEU_LAST_CONV; params=CONTOUR_PARAMS['pneumonia']
            thresh=threshold
        else:
            x = preprocess_osteo(img); model=osteoarthritis_model; last='top_conv'; params=CONTOUR_PARAMS['osteoarthritis']
            thresh=0.9
        p = float(model.predict(x)[0,0])
        pred = "Pneumonia" if p>thresh else "Normal" if not joint else "Osteoarthritis"
        st.write(f"**Prediction:** {pred}  **Confidence:** {p:.1%}")
        max_spots = 6
        

        # Grad-CAM spots
        orig = np.array(img.convert('RGB'))
        heat = make_gradcam_heatmap(x, model, last)
        # start with your static contour threshold
        thr = params['threshold']
        if params['adaptive_threshold']:
            # compute an image-specific cutoff at the 90th percentile,
            # or you could use cv2.threshold(..., cv2.THRESH_OTSU)
            thr = np.percentile(heat.flatten(), 90)

        # build mask & clean it
        mask = (cv2.resize(heat, (orig.shape[1], orig.shape[0])) > thr).astype(np.uint8)
        kern = np.ones((5,5), np.uint8)
        clean = cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern),
            cv2.MORPH_CLOSE,
            kern
        )

        # find contours and limit to max_spots
        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spots = [c for c in cnts if cv2.contourArea(c) > 50][:max_spots]
        spots_count = len(spots)

        # overlay
        cam = create_contoured_spot_heatmap(
            orig, heat,
            alpha=params['alpha'],
            threshold=thr,
            max_spots=spots_count,
            color_scheme=params['color_scheme'],
            adaptive_threshold=params['adaptive_threshold']
        )
        st.image(cam, caption="🔍 Precision Spot Analysis", use_column_width=True)


        # explanation
        st.markdown(get_spot_explanation(pred, p, spots_count))

        # feedback
        fb = st.radio("Is this correct?", [pred,'Normal','Pneumonia','Osteoarthritis'], key=str(uuid4()))
        if fb!=pred and st.button("Submit Correction",key=str(uuid4())):
            act=fb.lower().replace(' ','')
            fold=('pneu' if 'pneu' in act else 'osteo' if 'osteo' in act else 'normal')
            fn=f"{act}_{uuid4().hex}.png"
            img.convert('L').save(FEEDBACK_DIR/fold/fn)
            with open(LOG_PATH,'a') as log: log.write(f"{fn},{pred.lower()},{act}\n")
            st.success("Feedback saved.")

st.markdown("---")
st.write("Built with Streamlit & TensorFlow.")
