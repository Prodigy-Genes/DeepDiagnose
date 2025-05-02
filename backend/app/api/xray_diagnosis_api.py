# xray_diagnosis_api.py
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

# ----------------------
# PATH CONFIGURATION
# ----------------------
# Assuming this script lives in "backend/" and both "models/" and metrics folders
# (pneu_metrics, ana_metrics) are siblings of "backend/"
ROOT_DIR = Path(__file__).resolve().parent  # backend/
PROJECT_ROOT = ROOT_DIR.parent          # project root
MODELS_DIR = PROJECT_ROOT / "models"
PNEU_METRICS = PROJECT_ROOT / "pneu_metrics"
ANAT_METRICS = PROJECT_ROOT / "ana_metrics"

# ----------------------
# LOAD METRICS & MODELS
# ----------------------
def load_json(path: Path):
    return json.loads(path.read_text())

# Pneumonia metrics
pneu_info = load_json(PNEU_METRICS / 'dataset_info.json')
pneu_norm = load_json(PNEU_METRICS / 'normalization_stats.json')
line = next(l for l in (PNEU_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
pneu_thresh = float(line.split('=')[1])
pneu_last_conv = pneu_info.get('last_conv_layer', 'conv2d_2')
pneu_size = (pneu_info['resize_to']['height'], pneu_info['resize_to']['width'])

# Anatomy metrics
anat_info = load_json(ANAT_METRICS / 'dataset_info.json')
anat_norm = load_json(ANAT_METRICS / 'normalization_stats.json')
line = next(l for l in (ANAT_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
anat_thresh = float(line.split('=')[1])
anat_size = (anat_info['resize_to']['height'], anat_info['resize_to']['width'])

# Load models once
anat_model = tf.keras.models.load_model(str(MODELS_DIR / 'anatomical_classifier.keras'))
pneu_model = tf.keras.models.load_model(str(MODELS_DIR / 'pneumonia_classifier.keras'))
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
        diffs = [arr[:,:,0] - arr[:,:,1], arr[:,:,0] - arr[:,:,2], arr[:,:,1] - arr[:,:,2]]
        if np.std(diffs) > 15:
            return False
    gray = arr.mean(axis=-1) if arr.ndim == 3 else arr
    return 30 <= gray.mean() <= 220

def preprocess(img: Image.Image, size: tuple, mean: float, std: float, channels: int = 1):
    im = img.convert('L').resize((size[1], size[0]))
    arr = np.array(im, np.float32) / 255.0
    arr = (arr - mean) / (std + 1e-8)
    if channels == 3:
        arr = np.stack([arr] * 3, axis=-1)
    else:
        arr = arr.reshape(size[0], size[1], 1)
    return arr[np.newaxis]

# ----------------------
# PREDICTION ENDPOINT
# ----------------------
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # load image
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image file')
    arr = np.array(img)

    # validate
    if not is_likely_xray(arr):
        return JSONResponse(status_code=400, content={'error': 'Not a valid X-ray'})

    # anatomy
    xa = preprocess(img, anat_size, anat_norm['train_pixel_mean'], anat_norm['train_pixel_std'], channels=1)
    pa = float(anat_model.predict(xa)[0,0])
    if pa >= anat_thresh:
        anatomy = 'joint'; an_conf = pa
    else:
        anatomy = 'chest'; an_conf = 1 - pa
    if an_conf < 0.8:
        return JSONResponse(status_code=400, content={'error': 'Anatomy uncertain'})

    # select disease model
    if anatomy == 'joint':
        x = preprocess(img, (224,224), 0.0, 1.0, channels=3)
        model, thresh, label, last_conv = osteo_model, anat_thresh, 'osteoarthritis', None
        contour = {'threshold':0.4,'alpha':0.55,'color_scheme':'viridis','adaptive_threshold':True,'min_spot_area':50}
    else:
        x = preprocess(img, pneu_size, pneu_norm['train_pixel_mean'], pneu_norm['train_pixel_std'], channels=1)
        model, thresh, label, last_conv = pneu_model, pneu_thresh, 'pneumonia', pneu_last_conv
        contour = {'threshold':0.2,'alpha':0.6,'color_scheme':'viridis','adaptive_threshold':True,'min_spot_area':5}

    # disease prediction
    p = float(model.predict(x)[0,0])
    if p >= thresh:
        disease = label; d_conf = p
    else:
        disease = 'normal'; d_conf = 1 - p

    # grad-cam + contours
    from grad_cam_utils import make_gradcam_heatmap, create_contoured_spot_heatmap, overlay_heatmap
    heat = make_gradcam_heatmap(x, model, last_conv)
    thr = contour['threshold']
    if contour['adaptive_threshold']:
        nz = heat[heat>0]
        thr = float(np.clip(np.percentile(nz,70) if nz.size else thr,0.2,0.7))
    cam_spots = create_contoured_spot_heatmap(
        np.array(img.convert('RGB')), heat,
        alpha=contour['alpha'], threshold=thr,
        max_spots=8, color_scheme=contour['color_scheme'],
        adaptive_threshold=False, min_spot_area=contour['min_spot_area']
    )
    overlay = overlay_heatmap(np.array(img.convert('RGB')), heat, alpha=0.4) if np.array_equal(cam_spots, np.array(img.convert('RGB'))) else cam_spots

    # encode overlay
    buf = BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'anatomy': anatomy,
        'anatomy_confidence': round(an_conf,3),
        'disease': disease,
        'disease_confidence': round(d_conf,3),
        'overlay_image': f"data:image/png;base64,{img_b64}"
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
