from PIL import Image
import numpy as np
from fastapi import HTTPException
import base64
from io import BytesIO
from app.core.config import ConfidenceThresholds
from app.utils.image_utils import encode_image_to_base64
from app.ml.grad_cam_utils import make_gradcam_heatmap, create_contoured_spot_heatmap, overlay_heatmap
from app.services.image_service import (
    med_scan_model, med_scan_thresh,
    covid_model, covid_thresh, covid_last_conv,
    pneu_model, pneu_thresh, pneu_last_conv,
    anat_model, anat_thresh,
    osteo_model
)

from app.services.image_service import (
    validate_medical_image,
    preprocess_medical_scan_type,
    preprocess_covid,
    preprocess_anatomy,
    preprocess_osteo,
    preprocess_pneumonia
)

from app.services.explanation_service import generate_patient_explanation





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


def handle_prediction(img: Image.Image) -> dict:
    try:
        overlay = None
        output = {}
        # Step 1 - Validate Image
        is_valid, message = validate_medical_image(img)
        if not is_valid:
            raise ValueError(f"Invalid medical image: {message}")

        # Step 2 - Scan Type Classification
        x_scan = preprocess_medical_scan_type(img)
        scan_pred = med_scan_model.predict(x_scan)[0, 0].item()
        scan_type, scan_conf = ('X-ray', scan_pred) if scan_pred >= med_scan_thresh else ('CT', 1 - scan_pred)
        if scan_conf < 0.8:
            raise ValueError("Uncertain scan type. Please upload a clearer medical image.")

        # Step 3 - Route Based on Scan Type
        if scan_type == 'CT':
            x_covid = preprocess_covid(img)
            covid_result = make_enhanced_prediction(covid_model, x_covid, covid_thresh,
                                                    ConfidenceThresholds.COVID_CONFIDENCE_THRESHOLD,
                                                    ConfidenceThresholds.NORMAL_CONFIDENCE_THRESHOLD)
            if covid_result['final_prediction'] == "Uncertain - Consult Specialist":
                raise ValueError("Inconclusive result. Please consult a healthcare professional.")
            disease = covid_result['final_prediction']
            disease_conf = covid_result['confidence']
            overlay, heat = generate_gradcam_overlay(img, x_covid, covid_model, covid_last_conv, 'covid-19')
            anatomy = "CT Scan"
            anat_conf = scan_conf

        else:  # X-ray
            x_anat = preprocess_anatomy(img)
            anat_pred = anat_model.predict(x_anat)[0, 0].item()
            anatomy, anat_conf = ('Joint-scan', anat_pred) if anat_pred >= anat_thresh else ('Chest-scan', 1 - anat_pred)
            if anat_conf < 0.8:
                raise ValueError("Anatomy classification too uncertain.")

            if anatomy == "Joint-scan":
                x_osteo = preprocess_osteo(img)
                osteo_pred = osteo_model.predict(x_osteo)[0, 0].item()
                disease, disease_conf = ('Osteoarthritis', osteo_pred) if osteo_pred >= anat_thresh else ('Normal', 1 - osteo_pred)
                if disease_conf < 0.8:
                    raise ValueError("Osteoarthritis classification uncertain.")
                overlay, heat = generate_gradcam_overlay(img, x_osteo, osteo_model, None, 'osteoarthritis')

            else:
                x_pneu = preprocess_pneumonia(img)
                pneu_pred = pneu_model.predict(x_pneu)[0, 0].item()
                disease, disease_conf = ('Pneumonia', pneu_pred) if pneu_pred >= pneu_thresh else ('Normal', 1 - pneu_pred)
                if disease_conf < 0.8:
                    raise ValueError("Pneumonia classification uncertain.")
                overlay, heat = generate_gradcam_overlay(img, x_pneu, pneu_model, pneu_last_conv, 'pneumonia')

        # Final response structure
        output = {
            "scan_type": scan_type,
            "scan_type_confidence": round(scan_conf, 3),
            "anatomy": anatomy,
            "anatomy_confidence": round(anat_conf, 3),
            "disease": disease,
            "disease_confidence": round(disease_conf, 3)
        }

        # Overlay Image (base64)
        output["overlay_image"] = encode_image_to_base64(overlay)

        # Explanation
        try:
            explanation = generate_patient_explanation(model_output=output, heatmap=heat)
            output["explanation"] = explanation
        except:
            output["explanation"] = "Analysis completed. Consult a healthcare professional."

        return convert_numpy_types(output)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
