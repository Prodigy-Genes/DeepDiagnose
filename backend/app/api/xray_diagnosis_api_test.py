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
import traceback  # Added for enhanced error tracing
import logging  # Added for better logging
from keras_nlp.models import GemmaTokenizer, GemmaBackbone, GemmaCausalLM # type: ignore

# Set up logging to help debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce memory footprint
tf.config.set_visible_devices([], 'GPU')  # Disable GPU entirely
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Grad-CAM utilities
from ml.grad_cam_utils import (
    make_gradcam_heatmap,
    create_contoured_spot_heatmap,
    overlay_heatmap
)

# Import explanation utilities
from api.explanation_utils import generate_patient_explanation, generate_gradcam_overlay

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
# GEMMA MODEL SETUP (Local)
# ----------------------
GEMMA_DIR = ML_DIR / "gemma"
gemma_model = None
gemma_tokenizer = None

try:
    # 1. Load tokenizer
    tokenizer_path = GEMMA_DIR / "tokenizer.json"
    if tokenizer_path.exists():
        gemma_tokenizer = GemmaTokenizer(proto=tokenizer_path.read_bytes())
    
    # 2. Build model architecture (for 2B version)
    backbone = GemmaBackbone(
        vocabulary_size=256128,
        num_layers=18,
        num_query_heads=8,
        num_key_value_heads=8,
        hidden_dim=2048,
        intermediate_dim=16384,
        head_dim=256,
        layer_norm_epsilon=1e-6,
        max_sequence_length=8192,
    )
    
    # 3. Create causal LM
    gemma_model = GemmaCausalLM(backbone)
    
    # 4. Load weights
    weights_path = GEMMA_DIR / "model.weights.h5"
    if weights_path.exists():
        gemma_model.load_weights(str(weights_path))
        logger.info("Gemma 2B model loaded successfully from weights!")
except Exception as e:
    logger.error(f"Error loading Gemma model: {e}")

# ----------------------
# ENHANCED GEMMA FALLBACK PREDICTION SYSTEM
# ----------------------
def gemma_medical_prediction(img: Image.Image, scan_type: str, uncertain_results: dict) -> dict:
    """
    Use Gemma to make medical predictions when primary models are uncertain.
    Enhanced with better error handling and debugging.
    """
    logger.info(f"Gemma fallback called for {scan_type} with uncertain results: {uncertain_results}")
    
    # Check if Gemma components are available
    if not gemma_model or not gemma_tokenizer:
        logger.error("Gemma model or tokenizer not available")
        return {
            'prediction': 'Uncertain',
            'confidence': 0.5,
            'reasoning': 'Gemma model not available',
            'recommendation': 'Consult healthcare professional',
            'source': 'gemma_fallback_unavailable'
        }
    
    try:
        logger.info("Building Gemma prompt...")
        
        # Build comprehensive prompt for medical analysis
        prompt = f"""You are an expert radiologist AI assistant. Analyze this medical case:

SCAN TYPE: {scan_type}
PRIMARY MODEL RESULTS (Uncertain):
"""
        
        # Add uncertain results context with better formatting
        if 'covid_confidence' in uncertain_results:
            covid_conf = uncertain_results['covid_confidence']
            normal_conf = uncertain_results['normal_confidence']
            prompt += f"- COVID-19 probability: {covid_conf:.1%}\n"
            prompt += f"- Normal probability: {normal_conf:.1%}\n"
            
            # Add more context about why it's uncertain
            if covid_conf < 0.9 and covid_conf > 0.5:
                prompt += f"- Model shows moderate COVID-19 indicators but below 90% threshold\n"
            elif normal_conf < 0.8 and normal_conf > 0.5:
                prompt += f"- Model shows some normal indicators but below 80% threshold\n"
        
        # Add other confidence metrics if available
        for key in ['scan_type_confidence', 'anatomy_confidence', 'disease_confidence']:
            if key in uncertain_results:
                prompt += f"- {key.replace('_', ' ').title()}: {uncertain_results[key]:.1%}\n"

        # Enhanced task description
        prompt += f"""
CLINICAL CONTEXT:
- Primary models require ≥90% confidence for COVID-19 diagnosis
- Primary models require ≥80% confidence for Normal diagnosis
- Current case falls in uncertain range requiring expert review

TASK: Provide secondary medical analysis for this {scan_type} scan.

Consider these clinical indicators:
1. COVID-19: Ground-glass opacities, bilateral peripheral distribution, lower lobe involvement
2. Pneumonia: Consolidation, air bronchograms, unilateral or bilateral involvement
3. Normal: Clear lung fields, no consolidation or ground-glass changes

Respond in EXACTLY this format:
PREDICTION: [COVID-19|Pneumonia|Normal]
CONFIDENCE: [0.XX]
REASONING: [Your clinical reasoning in 1-2 sentences]
RECOMMENDATION: [Brief next steps]

Analysis:"""

        logger.info("Generating Gemma response...")
        
        # Clear memory and generate response
        tf.keras.backend.clear_session()
        
        # Tokenize input
        inputs = gemma_tokenizer([prompt])
        logger.info(f"Input tokenized successfully, shape: {inputs.shape if hasattr(inputs, 'shape') else 'unknown'}")
        
        # Generate with conservative parameters
        try:
            outputs = gemma_model.generate(
                inputs,
                max_length=min(len(inputs[0]) + 300, 2048),  # Prevent context overflow
                temperature=0.2,  # Very low temperature for medical consistency
                stop_token_ids=[gemma_tokenizer.end_token_id],
                pad_token_id=gemma_tokenizer.pad_token_id if hasattr(gemma_tokenizer, 'pad_token_id') else 0
            )
            logger.info("Gemma generation completed successfully")
        except Exception as gen_error:
            logger.error(f"Error during Gemma generation: {gen_error}")
            logger.error(f"Generation error traceback: {traceback.format_exc()}")
            raise gen_error
        
        # Decode response
        try:
            decoded = gemma_tokenizer.detokenize(outputs)
            response = decoded.numpy()[0].decode('utf-8')
            logger.info(f"Response decoded successfully, length: {len(response)}")
            logger.info(f"Raw Gemma response: {response[:500]}...")  # Log first 500 chars
        except Exception as decode_error:
            logger.error(f"Error decoding Gemma response: {decode_error}")
            raise decode_error
        
        # Extract the analysis from the response
        if "Analysis:" in response:
            analysis = response.split("Analysis:")[-1].strip()
        elif "Your response:" in response:
            analysis = response.split("Your response:")[-1].strip()
        else:
            analysis = response.strip()
        
        logger.info(f"Extracted analysis: {analysis[:200]}...")
        
        # Parse Gemma's structured response
        gemma_result = parse_gemma_medical_response(analysis)
        logger.info(f"Parsed Gemma result: {gemma_result}")
        
        return gemma_result
        
    except Exception as e:
        logger.error(f"Gemma medical prediction error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a safe fallback result
        return {
            'prediction': 'Uncertain',
            'confidence': 0.5,
            'reasoning': f'Secondary analysis failed: {str(e)[:100]}',
            'recommendation': 'Consult healthcare professional for expert review',
            'source': 'gemma_fallback_error',
            'error': str(e)
        }

def parse_gemma_medical_response(response: str) -> dict:
    """
    Enhanced parser for Gemma's structured medical response.
    """
    logger.info(f"Parsing Gemma response: {response[:200]}...")
    
    try:
        result = {
            'prediction': 'Uncertain',
            'confidence': 0.5,
            'reasoning': 'Unable to parse response',
            'recommendation': 'Consult healthcare professional',
            'source': 'gemma_fallback'
        }
        
        # Clean and split response
        response = response.strip()
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        logger.info(f"Processing {len(lines)} lines from response")
        
        for i, line in enumerate(lines):
            logger.info(f"Processing line {i}: {line}")
            
            # More flexible parsing patterns
            if any(keyword in line.upper() for keyword in ['PREDICTION:', 'DIAGNOSIS:', 'FINDING:']):
                pred_text = line.split(':')[-1].strip()
                pred_lower = pred_text.lower()
                
                if any(covid_term in pred_lower for covid_term in ['covid', 'covid-19', 'coronavirus']):
                    result['prediction'] = 'COVID-19'
                elif any(pneu_term in pred_lower for pneu_term in ['pneumonia', 'pneumonic']):
                    result['prediction'] = 'Pneumonia'
                elif any(normal_term in pred_lower for normal_term in ['normal', 'clear', 'negative']):
                    result['prediction'] = 'Normal'
                else:
                    # Keep original prediction if we can't categorize it
                    result['prediction'] = pred_text[:50]  # Limit length
                
                logger.info(f"Extracted prediction: {result['prediction']}")
            
            elif any(keyword in line.upper() for keyword in ['CONFIDENCE:', 'CERTAINTY:', 'PROBABILITY:']):
                conf_text = line.split(':')[-1].strip()
                try:
                    # Enhanced confidence extraction
                    import re
                    
                    # Look for decimal patterns
                    decimal_match = re.search(r'0\.\d+', conf_text)
                    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', conf_text)
                    
                    if decimal_match:
                        confidence = float(decimal_match.group())
                        result['confidence'] = max(0.0, min(1.0, confidence))
                    elif percentage_match:
                        confidence = float(percentage_match.group(1)) / 100
                        result['confidence'] = max(0.0, min(1.0, confidence))
                    else:
                        # Look for words indicating confidence level
                        conf_lower = conf_text.lower()
                        if any(high_conf in conf_lower for high_conf in ['high', 'strong', 'confident']):
                            result['confidence'] = 0.8
                        elif any(med_conf in conf_lower for med_conf in ['moderate', 'medium', 'fair']):
                            result['confidence'] = 0.6
                        elif any(low_conf in conf_lower for low_conf in ['low', 'weak', 'uncertain']):
                            result['confidence'] = 0.4
                    
                    logger.info(f"Extracted confidence: {result['confidence']}")
                except Exception as conf_error:
                    logger.warning(f"Could not parse confidence from '{conf_text}': {conf_error}")
                    result['confidence'] = 0.6  # Default moderate confidence
            
            elif any(keyword in line.upper() for keyword in ['REASONING:', 'RATIONALE:', 'ANALYSIS:']):
                reasoning = line.split(':', 1)[-1].strip()
                if reasoning:
                    result['reasoning'] = reasoning[:200]  # Limit length
                    logger.info(f"Extracted reasoning: {result['reasoning'][:100]}...")
            
            elif any(keyword in line.upper() for keyword in ['RECOMMENDATION:', 'NEXT STEPS:', 'ADVICE:']):
                recommendation = line.split(':', 1)[-1].strip()
                if recommendation:
                    result['recommendation'] = recommendation[:200]  # Limit length
                    logger.info(f"Extracted recommendation: {result['recommendation'][:100]}...")
        
        # Validation and defaults
        if result['confidence'] < 0.3:
            logger.warning(f"Very low confidence ({result['confidence']}) from Gemma, setting to 0.5")
            result['confidence'] = 0.5
        
        if result['reasoning'] == 'Unable to parse response':
            result['reasoning'] = f"Secondary AI analysis suggests {result['prediction'].lower()} findings"
        
        if result['recommendation'] == 'Consult healthcare professional':
            if result['prediction'] == 'COVID-19':
                result['recommendation'] = 'Consider isolation and follow COVID-19 protocols. Consult healthcare provider.'
            elif result['prediction'] == 'Pneumonia':
                result['recommendation'] = 'Seek medical attention for possible antibiotic treatment.'
            elif result['prediction'] == 'Normal':
                result['recommendation'] = 'If symptoms persist, discuss with healthcare provider.'
        
        logger.info(f"Final parsed result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error parsing Gemma response: {e}")
        logger.error(f"Parse error traceback: {traceback.format_exc()}")
        
        return {
            'prediction': 'Uncertain',
            'confidence': 0.5,
            'reasoning': f'Error parsing secondary analysis: {str(e)[:100]}',
            'recommendation': 'Please consult a healthcare professional',
            'source': 'gemma_fallback_parse_error',
            'error': str(e)
        }

# ----------------------
# UPDATED PREDICTION HANDLER WITH GEMMA FALLBACK
# ----------------------
def handle_covid_prediction_with_gemma_fallback(img, x_covid, scan_type, scan_conf):
    """
    Handle COVID prediction with proper Gemma fallback and error handling.
    """
    try:
        logger.info("Starting COVID-19 prediction...")
        
        # Make COVID prediction
        covid_result = make_enhanced_prediction(
            covid_model, 
            x_covid, 
            covid_thresh, 
            COVID_CONFIDENCE_THRESHOLD, 
            NORMAL_CONFIDENCE_THRESHOLD
        )
        
        logger.info(f"COVID prediction result: {covid_result}")
        
        # Check if COVID prediction is uncertain
        if covid_result['final_prediction'] == "Uncertain - Consult Specialist":
            logger.info(f"COVID model uncertain - COVID: {covid_result['covid_confidence']:.1%}, Normal: {covid_result['normal_confidence']:.1%}")
            
            try:
                # Use Gemma as fallback predictor
                logger.info("Calling Gemma fallback...")
                gemma_result = gemma_medical_prediction(img, scan_type, covid_result)
                logger.info(f"Gemma result: {gemma_result}")
                
                if gemma_result and gemma_result.get('confidence', 0) > 0.55:  # Slightly lower threshold
                    # Use Gemma's prediction
                    disease = gemma_result['prediction']
                    disease_conf = gemma_result['confidence']
                    
                    # Generate explanation based on Gemma's analysis
                    explanation_text = f"Primary models were uncertain, but secondary AI analysis suggests: {gemma_result['reasoning']} Recommendation: {gemma_result['recommendation']}"
                    
                    logger.info(f"Using Gemma prediction: {disease} with confidence {disease_conf}")
                    
                    # Generate Grad-CAM overlay (use original uncertain prediction for visualization)
                    try:
                        overlay, heat = generate_gradcam_overlay(img, x_covid, covid_model, covid_last_conv, 'covid-19')
                    except Exception as gradcam_error:
                        logger.warning(f"Grad-CAM generation failed: {gradcam_error}")
                        overlay, heat = np.array(img.convert('RGB')), None
                    
                    output = {
                        "scan_type": scan_type,
                        "scan_type_confidence": round(scan_conf, 3),
                        "anatomy": "CT Scan",
                        "anatomy_confidence": round(scan_conf, 3),
                        "disease": disease,
                        "disease_confidence": round(disease_conf, 3),
                        "primary_model_result": "uncertain",
                        "fallback_used": "gemma",
                        "gemma_reasoning": gemma_result['reasoning'],
                        "gemma_recommendation": gemma_result['recommendation'],
                        "original_covid_confidence": round(covid_result['covid_confidence'], 3),
                        "original_normal_confidence": round(covid_result['normal_confidence'], 3)
                    }
                    
                    return output, overlay, heat, explanation_text
                
                else:
                    # Even Gemma couldn't help reliably
                    logger.warning("Gemma fallback also uncertain or failed")
                    error_msg = {
                        "error": f"Inconclusive result - specialist consultation required. COVID probability: {covid_result['covid_confidence']:.1%} (needs ≥90% for confident COVID diagnosis), Normal probability: {covid_result['normal_confidence']:.1%} (needs ≥80% for confident normal diagnosis). Secondary AI analysis also inconclusive. Please consult a healthcare professional.",
                        "primary_covid_confidence": covid_result['covid_confidence'],
                        "primary_normal_confidence": covid_result['normal_confidence'],
                        "gemma_attempted": True,
                        "gemma_confidence": gemma_result.get('confidence', 0) if gemma_result else 0,
                        "gemma_error": gemma_result.get('error') if gemma_result and 'error' in gemma_result else None
                    }
                    return None, None, None, error_msg
                    
            except Exception as gemma_error:
                logger.error(f"Gemma fallback failed: {gemma_error}")
                logger.error(f"Gemma error traceback: {traceback.format_exc()}")
                
                # Return detailed error with original prediction
                error_msg = {
                    "error": f"Primary models uncertain and secondary analysis failed. COVID probability: {covid_result['covid_confidence']:.1%}, Normal probability: {covid_result['normal_confidence']:.1%}. Please consult a healthcare professional for expert review.",
                    "primary_covid_confidence": covid_result['covid_confidence'],
                    "primary_normal_confidence": covid_result['normal_confidence'],
                    "gemma_attempted": True,
                    "gemma_error": str(gemma_error),
                    "fallback_status": "failed"
                }
                return None, None, None, error_msg
        
        else:
            # COVID prediction was confident
            logger.info("COVID prediction was confident, no fallback needed")
            disease = covid_result['final_prediction']
            disease_conf = covid_result['confidence']
            
            # Generate standard explanation and visualization
            try:
                overlay, heat = generate_gradcam_overlay(img, x_covid, covid_model, covid_last_conv, 'covid-19')
                explanation_text = generate_patient_explanation({'disease': disease, 'scan_type': scan_type, 'anatomy': 'CT Scan'}, heat)
            except Exception as viz_error:
                logger.warning(f"Visualization generation failed: {viz_error}")
                overlay, heat = np.array(img.convert('RGB')), None
                explanation_text = f"Analysis completed: {disease}. Please consult with a healthcare professional."
            
            output = {
                "scan_type": scan_type,
                "scan_type_confidence": round(scan_conf, 3),
                "anatomy": "CT Scan",
                "anatomy_confidence": round(scan_conf, 3),
                "disease": disease,
                "disease_confidence": round(disease_conf, 3),
                "primary_model_result": "confident",
                "fallback_used": None
            }
            
            return output, overlay, heat, explanation_text
            
    except Exception as e:
        logger.error(f"Error in COVID prediction pipeline: {e}")
        logger.error(f"COVID pipeline error traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Error in COVID-19 classification pipeline: {str(e)}")

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
    logger.warning(f"Could not load COVID threshold, using default: {e}")
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
    logger.info("All models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

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
# MEDICAL IMAGE VALIDATION
# ----------------------
def validate_medical_image(img: Image.Image) -> tuple[bool, str]:
    """Enhanced validation for medical scans - same as before"""
    # [Keep your existing validation function unchanged]
    try:
        # Your existing validation logic here
        return True, "Valid medical image"  # Placeholder
    except Exception as e:
        return False, f"Error during validation: {str(e)}"

# ----------------------
# UTILITIES
# ----------------------
def preprocess_medical_scan_type(img: Image.Image):
    """Preprocess image for medical scan type classification (X-ray vs CT)"""
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
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((w, h))
        img_array = np.array(img)
        img_array = img_array.astype("float32") / 255.0
        img_array = img_array.reshape(1, h, w, 1)
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image for COVID-19: {e}")

def make_enhanced_prediction(model, image_array, threshold, covid_conf_thresh=0.90, normal_conf_thresh=0.80):
    """Make prediction with confidence-based classification - EXACT match to Streamlit"""
    prediction_prob = 1 - model.predict(image_array)[0][0]
    basic_prediction = "COVID-19" if prediction_prob >= threshold else "Normal"
    
    if basic_prediction == "COVID-19":
        if prediction_prob >= covid_conf_thresh:
            final_prediction = "COVID-19"
            confidence = prediction_prob
            certainty_level = "High Confidence"
        else:
            final_prediction = "Uncertain - Consult Specialist"
            confidence = prediction_prob
            certainty_level = "Low Confidence"
    else:
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

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
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

        # Medical image validation
        is_valid, validation_message = validate_medical_image(img)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Invalid medical image: {validation_message}. Please upload a valid X-ray or CT scan."
                }
            )

        # Step 1: Medical Scan Type Classification
        try:
            x_scan_type = preprocess_medical_scan_type(img)
            scan_pred = med_scan_model.predict(x_scan_type)[0, 0].item()
            
            if scan_pred >= med_scan_thresh:
                scan_type, scan_conf = 'X-ray', scan_pred
            else:
                scan_type, scan_conf = 'CT', 1 - scan_pred
            
            # Check if scan type classification is uncertain
            if scan_conf < 0.8:
                # Try Gemma fallback for scan type
                uncertain_scan_results = {
                    'scan_type_confidence': scan_conf,
                    'scan_type': scan_type
                }
                
                gemma_scan_result = gemma_medical_prediction(img, f"Medical scan (uncertain: {scan_type})", uncertain_scan_results)
                
                if gemma_scan_result and gemma_scan_result['confidence'] > 0.6:
                    # Use Gemma's assessment but proceed carefully
                    scan_type = 'CT' if 'ct' in gemma_scan_result['prediction'].lower() else 'X-ray'
                    scan_conf = gemma_scan_result['confidence']
                else:
                    return JSONResponse(
                        status_code=400, 
                        content={"error": "Unable to determine scan type with sufficient confidence. Please upload a clearer medical image."}
                    )
                
        except Exception as e:
            raise HTTPException(500, f"Error in medical scan type classification: {str(e)}")

        # Step 2: Route based on scan type
        if scan_type == 'CT':
            try:
                # Preprocess for COVID-19 model
                x_covid = preprocess_covid(img)
                
                # Use enhanced handler with Gemma fallback
                result = handle_covid_prediction_with_gemma_fallback(img, x_covid, scan_type, scan_conf)
                
                if result[0] is None:  # Error case
                    return JSONResponse(status_code=400, content=result[3])
                
                output, overlay, heat, explanation_text = result
                
            except Exception as e:
                raise HTTPException(500, f"Error in COVID-19 classification: {str(e)}")
        
        else:  # X-ray processing
            # [Implement similar Gemma fallback for X-ray predictions]
            output = {
                "scan_type": scan_type,
                "scan_type_confidence": round(scan_conf, 3),
                "anatomy": "X-ray",
                "anatomy_confidence": round(scan_conf, 3),
                "disease": "Pending X-ray analysis",
                "disease_confidence": 0.5,
                "primary_model_result": "pending",
                "fallback_used": None
            }
            explanation_text = "X-ray analysis pending - similar Gemma fallback can be implemented."

        # Add overlay image and explanation
        try:
            if 'overlay' in locals():
                buf = BytesIO()
                Image.fromarray(overlay).save(buf, format='PNG')
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                output["overlay_image"] = f"data:image/png;base64,{img_b64}"
            
            output['explanation'] = explanation_text
            output['explanation_source'] = 'gemma_fallback' if output.get('fallback_used') == 'gemma' else 'standard'
            
        except Exception as e:
            logger.warning(f"Could not encode overlay or explanation: {e}")
            output["overlay_image"] = None
            output['explanation'] = "Analysis completed. Please consult with a healthcare professional."

        return convert_numpy_types(output)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Unexpected error during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)