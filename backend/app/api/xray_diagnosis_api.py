import uvicorn  # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import JSONResponse # type: ignore
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import tensorflow as tf
import json
import base64
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
import cv2
from scipy import ndimage
from skimage import filters, measure, morphology


# Grad-CAM utilities
from ml.grad_cam_utils import (
    make_gradcam_heatmap,
    create_contoured_spot_heatmap,
    overlay_heatmap
)

# Import explanation utilities
from api.explanation_utils import generate_patient_explanation
# ----------------------
# PATH CONFIGURATION
# ----------------------
API_DIR      = Path(__file__).resolve().parent
APP_DIR      = API_DIR.parent
ML_DIR       = APP_DIR / "ml"
MODELS_DIR   = ML_DIR / "models"
PNEU_METRICS = ML_DIR / "pneu_metrics"
ANAT_METRICS = ML_DIR / "ana_metrics"
MEDICAL_SCAN_TYPE_METRICS = ML_DIR/"medical_scan_type_metrics"
COVID_METRICS = ML_DIR / "covid_metrics"


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

# COVID metrics
covid_info   = load_json(COVID_METRICS / 'dataset_info.json')
covid_norm   = load_json(COVID_METRICS / 'normalization_stats.json')
try:
    covid_thresh_data = load_json(COVID_METRICS / 'optimal_threshold.json')
    covid_thresh = float(covid_thresh_data.get('optimal_threshold', 0.592))
except Exception as e:
    print(f"Warning: Could not load COVID threshold, using default: {e}")
    covid_thresh = 0.592
covid_last_conv = covid_info.get('last_conv_layer', 'conv2d_2')
covid_size   = (covid_info['resize_to']['height'], covid_info['resize_to']['width'])

# Medical scan type metrics
med_scan_info   = load_json(MEDICAL_SCAN_TYPE_METRICS / 'dataset_info.json')
med_scan_norm   = load_json(MEDICAL_SCAN_TYPE_METRICS / 'normalization_stats.json')
line_med_scan   = next(l for l in (MEDICAL_SCAN_TYPE_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
med_scan_thresh = float(line_med_scan.split('=')[1])
med_scan_last_conv = med_scan_info.get('last_conv_layer', 'conv2d_2')
med_scan_size   = (med_scan_info['resize_to']['height'], med_scan_info['resize_to']['width'])

# Anatomy metrics (also reused as osteoarthritis threshold)
anat_info   = load_json(ANAT_METRICS / 'dataset_info.json')
anat_norm   = load_json(ANAT_METRICS / 'normalization_stats.json')
line_anat   = next(l for l in (ANAT_METRICS / 'thresholds.txt').read_text().splitlines() if 'opt_threshold' in l)
anat_thresh = float(line_anat.split('=')[1])
anat_size   = (anat_info['resize_to']['height'], anat_info['resize_to']['width'])

# COVID confidence thresholds - match Streamlit exactly
COVID_CONFIDENCE_THRESHOLD = 0.90  # 90% confidence for COVID prediction
NORMAL_CONFIDENCE_THRESHOLD = 0.80  # 80% confidence for Normal prediction

# Load models once
try:
    anat_model  = tf.keras.models.load_model(str(MODELS_DIR / 'anatomical_classifier.keras'))
    pneu_model  = tf.keras.models.load_model(str(MODELS_DIR / 'pneumonia_classifier.keras'))
    osteo_model = tf.keras.models.load_model(str(MODELS_DIR / 'osteo_efficientnetb0.keras'))
    covid_model = tf.keras.models.load_model(str(MODELS_DIR / 'covid19_model.keras'))
    med_scan_model = tf.keras.models.load_model(str(MODELS_DIR / 'medical_scan_type_classifier.keras'))
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise


# ----------------------
# FASTAPI APP SETUP
# ----------------------
app = FastAPI()
app.add_middleware(
    # CORS, to allow requests from the frontend 
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# MEDICAL IMAGE VALIDATION
# ----------------------
def validate_medical_image(img: Image.Image) -> tuple[bool, str]:
    """
    Enhanced validation for medical scans (X-ray or CT) - rejects colored photos and non-medical images.
    Accommodates CT scans which often have high contrast black/white regions.
    Returns (is_valid, reason_if_invalid)
    """
    try:
        # First, analyze the original image for color characteristics
        img_array_color = np.array(img)
        
        # 1. Color analysis - medical images should be grayscale or near-grayscale
        if len(img_array_color.shape) == 3:  # Color image
            h, w, channels = img_array_color.shape
            
            if channels >= 3:  # RGB or RGBA
                r, g, b = img_array_color[:,:,0], img_array_color[:,:,1], img_array_color[:,:,2]
                
                # Calculate color variance - medical images should have low color variance
                rg_diff = np.abs(r.astype(np.float32) - g.astype(np.float32))
                rb_diff = np.abs(r.astype(np.float32) - b.astype(np.float32))
                gb_diff = np.abs(g.astype(np.float32) - b.astype(np.float32))
                
                avg_color_diff = (np.mean(rg_diff) + np.mean(rb_diff) + np.mean(gb_diff)) / 3
                
                # If there's significant color variation, it's likely not a medical scan
                if avg_color_diff > 20:  # Slightly more lenient threshold
                    return False, "Image appears to be a colored photo, not a medical scan"
                
                # Check for high saturation - medical images should be low saturation
                # Convert to HSV to check saturation
                try:
                    img_hsv = img.convert('HSV')
                    hsv_array = np.array(img_hsv)
                    saturation = hsv_array[:,:,1]  # S channel
                    avg_saturation = np.mean(saturation)
                    
                    if avg_saturation > 40:  # More lenient for medical images
                        return False, "Image has high color saturation, appears to be a photo"
                except:
                    pass  # If HSV conversion fails, continue with other checks
        
        # Convert to grayscale for remaining analysis
        img_array = np.array(img.convert('L'))
        h, w = img_array.shape
        
        # 2. Basic dimension checks - more lenient
        if h < 64 or w < 64:
            return False, "Image too small for medical scan (minimum 64x64)"
        
        if h > 8000 or w > 8000:
            return False, "Image unusually large for medical scan"
        
        # 3. Aspect ratio check - more flexible
        aspect_ratio = max(h, w) / min(h, w)
        if aspect_ratio > 6.0:  # More lenient for CT scans
            return False, "Unusual aspect ratio for medical scan"
        
        # 4. Intensity distribution analysis - ADAPTED FOR CT SCANS
        hist, _ = np.histogram(img_array, bins=256, range=(0, 255))
        
        # Check for completely uniform images
        if np.sum(hist[:3]) > 0.99 * img_array.size:
            return False, "Image appears to be completely black"
        
        if np.sum(hist[252:]) > 0.99 * img_array.size:
            return False, "Image appears to be completely white"
        
        # 5. RELAXED intensity distribution for CT scans
        # CT scans can have dominant black/white regions with less mid-range
        dark_pixels = np.sum(hist[:80]) / img_array.size  # Expanded dark range
        mid_pixels = np.sum(hist[80:180]) / img_array.size  # Mid-range
        bright_pixels = np.sum(hist[180:]) / img_array.size  # Expanded bright range
        
        # More lenient check - CT scans can have very little mid-range
        if mid_pixels < 0.1 and (dark_pixels < 0.1 or bright_pixels < 0.1):
            # Only reject if there's almost no variation at all
            return False, "Image lacks sufficient intensity variation"
        
        # 6. Dynamic range check - more lenient for CT
        img_std = np.std(img_array.astype(np.float32))
        if img_std < 3:  # Reduced from 5 to accommodate high-contrast CT
            return False, "Image lacks sufficient contrast"
        
        # 7. Edge analysis - adapted for CT scans
        edges = cv2.Canny(img_array, 20, 100)  # Lower thresholds for CT
        edge_density = np.sum(edges > 0) / (h * w)
        
        if edge_density < 0.002:  # More lenient
            return False, "Image lacks anatomical structure"
        
        if edge_density > 0.7:  # More lenient for high-contrast CT
            return False, "Image appears to be text or diagram"
        
        # 8. Texture analysis - more lenient for CT
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        if laplacian_var < 10:  # Reduced from 20
            return False, "Image appears too uniform (lacks medical texture)"
        
        # 9. SKIP complex texture analysis for CT compatibility
        # CT scans can have different texture characteristics than X-rays
        
        # 10. RELAXED photographic characteristics check
        # Skip smooth gradient check as CT scans can have legitimate smooth regions
        
        # 11. MODIFIED frequency domain analysis
        try:
            f_transform = np.fft.fft2(img_array)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # More lenient frequency analysis for CT scans
            center_h, center_w = h//2, w//2
            center_region = magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30]
            edge_region = np.concatenate([
                magnitude_spectrum[:15, :].flatten(),
                magnitude_spectrum[-15:, :].flatten(),
                magnitude_spectrum[:, :15].flatten(),
                magnitude_spectrum[:, -15:].flatten()
            ])
            
            center_energy = np.mean(center_region)
            edge_energy = np.mean(edge_region)
            
            # More lenient threshold for CT scans
            if edge_energy > center_energy * 2.0:  # Increased from 1.5
                return False, "Frequency characteristics suggest natural photo rather than medical scan"
        except:
            pass  # Continue if frequency analysis fails
        
        # 12. RELAXED structure analysis for CT scans
        try:
            binary = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            structure_density = np.sum(binary > 0) / (h * w)
            # More lenient bounds for CT scans
            if structure_density < 0.05 or structure_density > 0.98:
                return False, "Image appears to lack anatomical structures"
        except:
            pass  # Skip if adaptive threshold fails
        
        # 13. MODIFIED entropy check for CT scans
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        if entropy < 2.0:  # Reduced from 3.0 for high-contrast CT
            return False, "Image too simple (lacks medical complexity)"
        
        if entropy > 8.5:  # Slightly more lenient
            return False, "Image too noisy or complex"
        
        # 14. RELAXED gradient check for CT
        grad_magnitude = np.sqrt(
            cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)**2 + 
            cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)**2
        )
        avg_gradient = np.mean(grad_magnitude)
        
        if avg_gradient < 1.5:  # Reduced from 2.0
            return False, "Image too smooth for medical scan"
        
        # 15. RELAXED unique values check
        unique_values = len(np.unique(img_array))
        if unique_values < 8:  # Reduced from 10 for high-contrast CT
            return False, "Image appears to be artificial or non-medical"
        
        # 16. ADDITIONAL CT-specific validation
        # Check for reasonable contrast distribution in CT scans
        # CT scans often have bimodal distributions (air/background vs tissue/bone)
        hist_peaks = []
        for i in range(10, 246):  # Avoid extreme values
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > img_array.size * 0.01:
                hist_peaks.append(i)
        
        # CT scans typically have at least some structure (peaks in histogram)
        if len(hist_peaks) == 0 and np.max(hist[10:246]) < img_array.size * 0.05:
            return False, "Image lacks characteristic medical intensity patterns"
        
        # If all checks pass
        return True, "Valid medical image"
        
    except Exception as e:
        return False, f"Error during validation: {str(e)}"

# ----------------------
# UTILITIES
# ----------------------
def preprocess_medical_scan_type(img: Image.Image):
    """Preprocess image for medical scan type classification (X-ray vs CTI)"""
    try:
        h, w = med_scan_size
        im = img.convert('L').resize((w, h))
        arr = np.array(im, dtype=np.float32) / 255.0
        arr = (arr - med_scan_norm['train_pixel_mean']) / (med_scan_norm['train_pixel_std'] + 1e-8)
        return arr.reshape(1, h, w, 1)
    except Exception as e:
        raise ValueError(f"Error preprocessing image for medical scan type: {e}")

def preprocess_covid(img: Image.Image):
    """Preprocess image for COVID-19 classification - EXACT match to Streamlit"""
    try:
        h, w = covid_size
        # Convert to grayscale EXACTLY like Streamlit
        if img.mode != 'L':
            img = img.convert('L')
        
        img = img.resize((w, h))
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype("float32") / 255.0
        img_array = img_array.reshape(1, h, w, 1)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image for COVID-19: {e}")

def make_enhanced_prediction(model, image_array, threshold, covid_conf_thresh=0.90, normal_conf_thresh=0.80):
    """Make prediction with confidence-based classification - EXACT match to Streamlit"""
    # Get raw prediction probability - EXACT match to Streamlit
    prediction_prob = 1 - model.predict(image_array)[0][0]
    
    # Determine prediction based on threshold
    basic_prediction = "COVID-19" if prediction_prob >= threshold else "Normal"
    
    # Apply confidence thresholding - EXACT match to Streamlit logic
    if basic_prediction == "COVID-19":
        # For COVID prediction, need high confidence
        if prediction_prob >= covid_conf_thresh:
            final_prediction = "COVID-19"
            confidence = prediction_prob
            certainty_level = "High Confidence"
        else:
            final_prediction = "Uncertain - Consult Specialist"
            confidence = prediction_prob
            certainty_level = "Low Confidence"
    else:
        # For Normal prediction, need moderate confidence
        normal_confidence = 1 - prediction_prob
        if normal_confidence >= normal_conf_thresh:
            final_prediction = "Normal"
            confidence = normal_confidence
            certainty_level = "High Confidence"
        else:
            final_prediction = "Uncertain - Consult Specialist"
            confidence = max(prediction_prob, normal_confidence)
            certainty_level = "Low Confidence"
    
    return {
        'prediction_prob': prediction_prob,
        'final_prediction': final_prediction,
        'basic_prediction': basic_prediction,
        'confidence': confidence,
        'certainty_level': certainty_level,
        'covid_confidence': prediction_prob,
        'normal_confidence': 1 - prediction_prob
    }

def preprocess_pneumonia(img: Image.Image):
    try:
        h, w = pneu_size
        im = img.convert('L').resize((w, h))
        arr = np.array(im, dtype=np.float32) / 255.0
        arr = (arr - pneu_norm['train_pixel_mean']) / (pneu_norm['train_pixel_std'] + 1e-8)
        return arr.reshape(1, h, w, 1)
    except Exception as e:
        raise ValueError(f"Error preprocessing image for pneumonia: {e}")

def preprocess_osteo(img: Image.Image):
    try:
        # replicate Streamlit: grayscale → 224×224 → stack to 3 → EfficientNet preprocess
        im = img.convert('L').resize((224, 224))
        arr_gray = np.array(im, dtype=np.float32)
        arr_rgb  = np.stack([arr_gray]*3, axis=-1)
        return preprocess_input(arr_rgb).reshape(1, 224, 224, 3)
    except Exception as e:
        raise ValueError(f"Error preprocessing image for osteoarthritis: {e}")

def preprocess_anatomy(img: Image.Image):
    try:
        h, w = anat_size
        im = img.convert('L').resize((w, h))
        arr = np.array(im, dtype=np.float32) / 255.0
        arr = (arr - anat_norm['train_pixel_mean']) / (anat_norm['train_pixel_std'] + 1e-8)
        return arr.reshape(1, h, w, 1)
    except Exception as e:
        raise ValueError(f"Error preprocessing image for anatomy: {e}")

# contour parameters
CONTOURS = {
    'pneumonia':     {'threshold': 0.2,  'alpha': 0.6,  'color_scheme': 'viridis', 'adaptive_threshold': True,  'min_spot_area': 5},
    'osteoarthritis':{'threshold': 0.4,  'alpha': 0.55, 'color_scheme': 'viridis', 'adaptive_threshold': True,  'min_spot_area': 50},
    'covid-19':      {'threshold': 0.3,  'alpha': 0.6,  'color_scheme': 'viridis', 'adaptive_threshold': True,  'min_spot_area': 10},
}

def generate_gradcam_overlay(img: Image.Image, x, model, last_conv, label: str):
    """Generate Grad-CAM overlay for visualization"""
    try:
        # Generate Grad-CAM heatmap
        heat = make_gradcam_heatmap(x, model, last_conv)
        
        # Get contour parameters for the specific condition
        params = CONTOURS.get(label.lower(), CONTOURS['pneumonia'])  # Default to pneumonia params
        thr = params['threshold']
        
        if params['adaptive_threshold']:
            nz = heat[heat > 0]
            thr = float(np.clip(np.percentile(nz, 70) if nz.size else thr, 0.2, 0.7))
        
        # Create contoured spot heatmap
        spots = create_contoured_spot_heatmap(
            np.array(img.convert('RGB')), heat,
            alpha=params['alpha'], threshold=thr,
            max_spots=8, color_scheme=params['color_scheme'],
            adaptive_threshold=False, min_spot_area=params['min_spot_area']
        )
        
        # Create overlay
        overlay = overlay_heatmap(np.array(img.convert('RGB')), heat, alpha=0.4) \
            if np.array_equal(spots, np.array(img.convert('RGB'))) else spots
        
        return overlay, heat
    except Exception as e:
        print(f"Warning: Could not generate Grad-CAM overlay: {e}")
        # Return original image if Grad-CAM fails
        return np.array(img.convert('RGB')), None

# Utility function to convert numpy types to Python native types
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# ----------------------
# PREDICTION ENDPOINT
# ----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        # Load & validate image
        try:
            data = await file.read()
            img = Image.open(BytesIO(data)).convert('RGB')
        except Exception as e:
            raise HTTPException(400, f"Invalid image file: {str(e)}")

        # RIGOROUS MEDICAL IMAGE VALIDATION
        is_valid, validation_message = validate_medical_image(img)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Invalid medical image: {validation_message}. Please upload a valid X-ray or CTI scan."
                }
            )

        # Step 1: Medical Scan Type Classification (X-ray vs CTI)
        try:
            x_scan_type = preprocess_medical_scan_type(img)
            scan_pred = med_scan_model.predict(x_scan_type)[0, 0].item()
            
            # Determine scan type based on threshold
            if scan_pred >= med_scan_thresh:
                scan_type, scan_conf = 'X-ray', scan_pred
            else:
                scan_type, scan_conf = 'CTI', 1 - scan_pred
            
            # Check confidence threshold
            if scan_conf < 0.8:
                return JSONResponse(
                    status_code=400, 
                    content={"error": "Unable to determine scan type with sufficient confidence. Please upload a clearer medical image."}
                )
                
        except Exception as e:
            raise HTTPException(500, f"Error in medical scan type classification: {str(e)}")

        # Step 2: Route based on scan type
        if scan_type == 'CTI':
            try:
                # Preprocess image for COVID-19 model - EXACT match to Streamlit
                x_covid = preprocess_covid(img)
                
                # Use EXACT Streamlit prediction logic
                covid_result = make_enhanced_prediction(
                    covid_model, 
                    x_covid, 
                    covid_thresh, 
                    COVID_CONFIDENCE_THRESHOLD, 
                    NORMAL_CONFIDENCE_THRESHOLD
                )
                
                # Handle uncertain predictions the same way as Streamlit
                if covid_result['final_prediction'] == "Uncertain - Consult Specialist":
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": f"Inconclusive result - specialist consultation required. COVID probability: {covid_result['covid_confidence']:.1%} (needs ≥90% for confident COVID diagnosis), Normal probability: {covid_result['normal_confidence']:.1%} (needs ≥80% for confident normal diagnosis). Please consult a healthcare professional."
                        }
                    )
                
                disease = covid_result['final_prediction']
                disease_conf = covid_result['confidence']
                
                # Generate Grad-CAM overlay
                overlay, heat = generate_gradcam_overlay(img, x_covid, covid_model, covid_last_conv, 'covid-19')
                
                # Clean CTI output
                output = {
                    "scan_type": scan_type,
                    "scan_type_confidence": round(scan_conf, 3),
                    "anatomy": "CTI Scan",
                    "anatomy_confidence": round(scan_conf, 3),
                    "disease": disease,
                    "disease_confidence": round(disease_conf, 3),
                }
                
            except Exception as e:
                raise HTTPException(500, f"Error in COVID-19 classification: {str(e)}")
                
        else:  # X-ray
            # Route to anatomy classification first
            try:
                x_anat = preprocess_anatomy(img)
                anat_pred = anat_model.predict(x_anat)[0, 0].item()
                
                if anat_pred >= anat_thresh:
                    anatomy, anat_conf = 'Joint-scan', anat_pred
                else:
                    anatomy, anat_conf = 'Chest-scan', 1 - anat_pred
                
                # Check confidence threshold
                if anat_conf < 0.8:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Unable to classify X-ray anatomy with sufficient confidence. Please upload a clearer X-ray image."}
                    )
                    
            except Exception as e:
                raise HTTPException(500, f"Error in anatomy classification: {str(e)}")
            
            # Route to appropriate disease classification
            if anatomy == 'Joint-scan':
                # Route to Osteoarthritis detection
                try:
                    x_osteo = preprocess_osteo(img)
                    osteo_pred = osteo_model.predict(x_osteo)[0, 0].item()
                    
                    if osteo_pred >= anat_thresh:  # Using anatomy threshold for osteo
                        disease, disease_conf = 'Osteoarthritis', osteo_pred
                    else:
                        disease, disease_conf = 'Normal', 1 - osteo_pred
                    
                    # Check confidence threshold
                    if disease_conf < 0.8:
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Unable to classify osteoarthritis with sufficient confidence. Please consult a medical professional."}
                        )
                    
                    # Generate Grad-CAM overlay (Note: osteo model might not have conv layers for Grad-CAM)
                    overlay, heat = generate_gradcam_overlay(img, x_osteo, osteo_model, None, 'osteoarthritis')
                    
                except Exception as e:
                    raise HTTPException(500, f"Error in osteoarthritis classification: {str(e)}")
                    
            else:  # Chest-scan
                # Route to Pneumonia detection
                try:
                    x_pneu = preprocess_pneumonia(img)
                    pneu_pred = pneu_model.predict(x_pneu)[0, 0].item()
                    
                    if pneu_pred >= pneu_thresh:
                        disease, disease_conf = 'Pneumonia', pneu_pred
                    else:
                        disease, disease_conf = 'Normal', 1 - pneu_pred
                    
                    # Check confidence threshold
                    if disease_conf < 0.8:
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Unable to classify pneumonia with sufficient confidence. Please consult a medical professional."}
                        )
                    
                    # Generate Grad-CAM overlay
                    overlay, heat = generate_gradcam_overlay(img, x_pneu, pneu_model, pneu_last_conv, 'pneumonia')
                    
                except Exception as e:
                    raise HTTPException(500, f"Error in pneumonia classification: {str(e)}")
            
            output = {
                "scan_type": scan_type,
                "scan_type_confidence": round(scan_conf, 3),
                "anatomy": anatomy,
                "anatomy_confidence": round(anat_conf, 3),
                "disease": disease,
                "disease_confidence": round(disease_conf, 3),
            }

        # Encode overlay image to base64
        try:
            buf = BytesIO()
            Image.fromarray(overlay).save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            output["overlay_image"] = f"data:image/png;base64,{img_b64}"
        except Exception as e:
            print(f"Warning: Could not encode overlay image: {e}")
            output["overlay_image"] = None

        # Generate patient explanation
        try:
            explanation_text = generate_patient_explanation(
                model_output=output,
                heatmap=heat
            )
            output['explanation'] = explanation_text
        except Exception as e:
            print(f"Warning: Could not generate explanation: {e}")
            output['explanation'] = "Analysis completed. Please consult with a healthcare professional for detailed interpretation."

        return convert_numpy_types(output)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(500, f"Unexpected error during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)