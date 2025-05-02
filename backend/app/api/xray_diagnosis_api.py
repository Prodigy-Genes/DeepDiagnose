import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import tensorflow as tf
import json
import base64

# Grad-CAM utilities
from backend.app.ml.grad_cam_utils import (
    make_gradcam_heatmap,
    create_contoured_spot_heatmap,
    overlay_heatmap
)

# ----------------------
# PATH CONFIGURATION
# ----------------------
API_DIR      = Path(__file__).resolve().parent
APP_DIR      = API_DIR.parent
ML_DIR       = APP_DIR / "ml"
MODELS_DIR   = ML_DIR / "models"
PNEU_METRICS = ML_DIR / "pneu_metrics"
ANAT_METRICS = ML_DIR / "ana_metrics"

# ----------------------
# LOAD METRICS & MODELS
# ----------------------
def load_json(path: Path):
    return json.loads(path.read_text())

# Pneumonia metrics
pneu_info      = load_json(PNEU_METRICS / 'dataset_info.json')
pneu_norm      = load_json(PNEU_METRICS / 'normalization_stats.json')
line_pneu      = next(l for l in (PNEU_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
pneu_thresh    = float(line_pneu.split('=')[1])
pneu_last_conv = pneu_info.get('last_conv_layer', 'conv2d_2')
pneu_size      = (pneu_info['resize_to']['height'], pneu_info['resize_to']['width'])

# Anatomy metrics (also reused as osteoarthritis threshold)
anat_info   = load_json(ANAT_METRICS / 'dataset_info.json')
anat_norm   = load_json(ANAT_METRICS / 'normalization_stats.json')
line_anat   = next(l for l in (ANAT_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
anat_thresh = float(line_anat.split('=')[1])
anat_size   = (anat_info['resize_to']['height'], anat_info['resize_to']['width'])

# Load models once
anat_model  = tf.keras.models.load_model(str(MODELS_DIR / 'anatomical_classifier.keras'))
pneu_model  = tf.keras.models.load_model(str(MODELS_DIR / 'pneumonia_classifier.keras'))
osteo_model = tf.keras.models.load_model(str(MODELS_DIR / 'osteo_efficientnetb0.keras'))

# ----------------------
# FASTAPI APP SETUP
# ----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# UTILITIES
# ----------------------
def is_likely_xray(arr: np.ndarray) -> bool:
    if arr.ndim == 3 and arr.shape[2] == 3:
        diffs = [arr[:,:,0] - arr[:,:,1],
                 arr[:,:,0] - arr[:,:,2],
                 arr[:,:,1] - arr[:,:,2]]
        if np.std(diffs) > 15:
            return False
    gray = arr.mean(axis=-1) if arr.ndim == 3 else arr
    return 30 <= gray.mean() <= 220

def preprocess_pneumonia(img: Image.Image):
    h, w = pneu_size
    im = img.convert('L').resize((w, h))
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = (arr - pneu_norm['train_pixel_mean']) / (pneu_norm['train_pixel_std'] + 1e-8)
    return arr.reshape(1, h, w, 1)

def preprocess_osteo(img: Image.Image):
    # replicate Streamlit: grayscale → 224×224 → stack to 3 → EfficientNet preprocess
    im = img.convert('L').resize((224, 224))
    arr_gray = np.array(im, dtype=np.float32)
    arr_rgb  = np.stack([arr_gray]*3, axis=-1)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(arr_rgb).reshape(1, 224, 224, 3)

def preprocess_anatomy(img: Image.Image):
    h, w = anat_size
    im = img.convert('L').resize((w, h))
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = (arr - anat_norm['train_pixel_mean']) / (anat_norm['train_pixel_std'] + 1e-8)
    return arr.reshape(1, h, w, 1)

# contour parameters
CONTOURS = {
    'pneumonia':     {'threshold': 0.2,  'alpha': 0.6,  'color_scheme': 'viridis', 'adaptive_threshold': True,  'min_spot_area': 5},
    'osteoarthritis':{'threshold': 0.4,  'alpha': 0.55, 'color_scheme': 'viridis', 'adaptive_threshold': True,  'min_spot_area': 50},
}

# ----------------------
# PREDICTION ENDPOINT
# ----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load & validate image
    try:
        data = await file.read()
        img  = Image.open(BytesIO(data)).convert('RGB')
    except:
        raise HTTPException(400, "Invalid image file")
    arr = np.array(img)
    if not is_likely_xray(arr):
        return JSONResponse(status_code=400, content={"error": "Not a valid X-ray"})

    # Anatomy classification
    xa = preprocess_anatomy(img)
    pa = float(anat_model.predict(xa)[0,0])
    if pa >= anat_thresh:
        anatomy, an_conf = 'joint', pa
    else:
        anatomy, an_conf = 'chest', 1 - pa
    if an_conf < 0.9:
        return JSONResponse(status_code=400, content={"error": "Anatomy uncertain"})

    # Disease selection & preprocessing
    if anatomy == 'joint':
        x       = preprocess_osteo(img)
        model   = osteo_model
        threshold = anat_thresh          # exactly as in Streamlit
        label     = 'Osteoarthritis'
        last_conv = None
    else:
        x         = preprocess_pneumonia(img)
        model     = pneu_model
        threshold = pneu_thresh
        label     = 'Pneumonia'
        last_conv = pneu_last_conv

    # Prediction + threshold logic
    p = float(model.predict(x)[0,0])
    if p >= threshold:
        disease, d_conf = label, p
    else:
        disease, d_conf = 'Normal', 1 - p

    # Grad-CAM + contour overlay
    heat = make_gradcam_heatmap(x, model, last_conv)
    params = CONTOURS[label.lower()]
    thr    = params['threshold']
    if params['adaptive_threshold']:
        nz = heat[heat > 0]
        thr = float(np.clip(np.percentile(nz, 70) if nz.size else thr, 0.2, 0.7))
    spots = create_contoured_spot_heatmap(
        np.array(img.convert('RGB')), heat,
        alpha=params['alpha'], threshold=thr,
        max_spots=8, color_scheme=params['color_scheme'],
        adaptive_threshold=False, min_spot_area=params['min_spot_area']
    )
    overlay = overlay_heatmap(np.array(img.convert('RGB')), heat, alpha=0.4) \
        if np.array_equal(spots, np.array(img.convert('RGB'))) else spots

    # Encode PNG → Base64
    buf = BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "anatomy":             anatomy,
        "anatomy_confidence":  round(an_conf, 3),
        "disease":             disease,
        "disease_confidence":  round(d_conf, 3),
        "overlay_image":       f"data:image/png;base64,{img_b64}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
