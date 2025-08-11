import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any
from pathlib import Path

# Import Gemma components (will be available from main.py context)
try:
    from keras_nlp.models import GemmaBackbone, GemmaCausalLM # type: ignore
    from keras_nlp.tokenizers import GemmaTokenizer # type: ignore
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

def generate_gradcam_overlay(img, x, model, last_conv, label: str):
    """Generate Grad-CAM overlay for visualization"""
    try:
        # Import here to avoid circular imports
        from ml.grad_cam_utils import (
            make_gradcam_heatmap,
            create_contoured_spot_heatmap,
            overlay_heatmap
        )
        
        # Generate Grad-CAM heatmap
        heat = make_gradcam_heatmap(x, model, last_conv)
        
        # Contour parameters
        CONTOURS = {
            'pneumonia':     {'threshold': 0.2,  'alpha': 0.6,  'color_scheme': 'viridis', 'adaptive_threshold': True,  'min_spot_area': 5},
            'osteoarthritis':{'threshold': 0.4,  'alpha': 0.55, 'color_scheme': 'viridis', 'adaptive_threshold': True,  'min_spot_area': 50},
            'covid-19':      {'threshold': 0.3,  'alpha': 0.6,  'color_scheme': 'viridis', 'adaptive_threshold': True,  'min_spot_area': 10},
        }
        
        # Get contour parameters for the specific condition
        params = CONTOURS.get(label.lower(), CONTOURS['pneumonia'])
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
        return np.array(img.convert('RGB')), None

def analyze_activation_patterns(heatmap: np.ndarray, disease: str) -> dict:
    """
    Analyze activation patterns from heatmap to provide medical insights.
    
    Parameters:
    - heatmap: 2D numpy array of activation intensities (normalized 0–1)
    - disease: predicted disease label
    
    Returns:
    - dict with analysis results
    """
    try:
        hm = np.asarray(heatmap, dtype=np.float32)
    except Exception:
        return {"doctor_note": "Invalid heatmap data provided."}
        
    if hm.ndim != 2:
        return {"doctor_note": "Heatmap must be a 2D array."}
    
    # Analyze activation patterns
    positives = hm[hm > 0]
    if positives.size == 0:
        if disease.lower() == "normal":
            return {"doctor_note": "The scan appears normal with no concerning areas highlighted."}
        severity = "mild"
    else:
        max_intensity = float(np.max(positives))
        area_fraction = positives.size / hm.size
        mean_intensity = float(np.mean(positives))
        
        # Determine severity based on activation characteristics
        if max_intensity >= 0.8 and area_fraction >= 0.15:
            severity = "severe"
        elif max_intensity >= 0.6 or area_fraction >= 0.08:
            severity = "moderate"
        else:
            severity = "mild"
    
    # Disease-specific analysis
    analysis = {"severity": severity, "max_intensity": max_intensity if positives.size > 0 else 0.0}
    
    if disease.lower() == "covid-19":
        analysis["pattern_type"] = "ground-glass opacities" if area_fraction > 0.1 else "focal consolidation"
        analysis["distribution"] = "bilateral" if _check_bilateral_pattern(hm) else "unilateral"
        analysis["doctor_note"] = f"COVID-19 findings show {severity} {analysis['pattern_type']} with {analysis['distribution']} distribution."
        
    elif disease.lower() == "pneumonia":
        analysis["consolidation_type"] = "extensive" if area_fraction > 0.12 else "localized"
        analysis["doctor_note"] = f"Pneumonia findings indicate {severity} {analysis['consolidation_type']} consolidation."
        
    elif disease.lower() == "osteoarthritis":
        analysis["joint_involvement"] = "multiple joints" if area_fraction > 0.15 else "localized joint"
        analysis["doctor_note"] = f"Osteoarthritis shows {severity} changes in {analysis['joint_involvement']}."
        
    elif disease.lower() == "normal":
        analysis["doctor_note"] = "Scan appears within normal limits with no significant pathological findings."
    
    else:
        analysis["doctor_note"] = f"Analysis shows {severity} findings for {disease}."
    
    return analysis

def _check_bilateral_pattern(heatmap: np.ndarray) -> bool:
    """Check if activation pattern suggests bilateral involvement"""
    try:
        h, w = heatmap.shape
        left_half = heatmap[:, :w//2]
        right_half = heatmap[:, w//2:]
        
        left_activation = np.mean(left_half[left_half > 0.3]) if np.any(left_half > 0.3) else 0
        right_activation = np.mean(right_half[right_half > 0.3]) if np.any(right_half > 0.3) else 0
        
        return left_activation > 0.1 and right_activation > 0.1
    except:
        return False

def generate_patient_explanation(prediction_data: dict, heatmap: Optional[np.ndarray] = None, 
                               gemma_model=None, gemma_tokenizer=None) -> str:
    """
    Generate comprehensive patient explanation with optional Gemma enhancement.
    
    Parameters:
    - prediction_data: Dictionary containing prediction results
    - heatmap: Optional heatmap for activation analysis
    - gemma_model: Optional Gemma model for enhanced explanations
    - gemma_tokenizer: Optional Gemma tokenizer
    
    Returns:
    - Detailed patient explanation string
    """
    disease = prediction_data.get('disease', 'Unknown')
    scan_type = prediction_data.get('scan_type', 'Medical scan')
    anatomy = prediction_data.get('anatomy', 'Anatomical region')
    confidence = prediction_data.get('disease_confidence', 0.5)
    
    # Base explanation templates
    base_explanations = {
        "covid-19": {
            "condition": "COVID-19 pneumonia",
            "description": "a viral infection affecting the lungs, typically showing ground-glass opacities and bilateral involvement",
            "symptoms": "fever, cough, shortness of breath, and fatigue",
            "next_steps": "isolation, monitoring oxygen levels, and following healthcare provider guidance"
        },
        "pneumonia": {
            "condition": "bacterial or viral pneumonia",
            "description": "an infection causing inflammation in the lung tissue, typically showing consolidation and air bronchograms",
            "symptoms": "fever, productive cough, chest pain, and difficulty breathing",
            "next_steps": "antibiotic treatment (if bacterial), rest, and follow-up imaging"
        },
        "osteoarthritis": {
            "condition": "osteoarthritis",
            "description": "degenerative joint disease showing joint space narrowing, bone spurs, and cartilage loss",
            "symptoms": "joint pain, stiffness, and reduced range of motion",
            "next_steps": "pain management, physical therapy, and lifestyle modifications"
        },
        "normal": {
            "condition": "normal findings",
            "description": "no significant pathological abnormalities detected",
            "symptoms": "if symptoms persist, they may be due to other non-radiographic causes",
            "next_steps": "discuss symptoms with healthcare provider if present"
        }
    }
    
    # Get base explanation
    base_info = base_explanations.get(disease.lower(), {
        "condition": disease,
        "description": "requires further evaluation by a healthcare professional",
        "symptoms": "variable depending on the specific condition",
        "next_steps": "consultation with a specialist for proper diagnosis and treatment"
    })
    
    # Analyze heatmap if available
    heatmap_analysis = {}
    if heatmap is not None:
        heatmap_analysis = analyze_activation_patterns(heatmap, disease)
    
    # Generate enhanced explanation with Gemma if available
    enhanced_explanation = ""
    if gemma_model and gemma_tokenizer and GEMMA_AVAILABLE:
        enhanced_explanation = _generate_gemma_explanation(
            prediction_data, heatmap_analysis, base_info, gemma_model, gemma_tokenizer
        )
    
    # Construct final explanation
    if enhanced_explanation:
        explanation = enhanced_explanation
    else:
        # Fallback to template-based explanation
        explanation = _generate_template_explanation(prediction_data, heatmap_analysis, base_info)
    
    # Add standard medical disclaimer
    disclaimer = ("\n\n⚠️ IMPORTANT MEDICAL DISCLAIMER: This AI analysis is for educational purposes only "
                 "and should not replace professional medical diagnosis or treatment. Please consult "
                 "with a qualified healthcare provider for proper medical evaluation and care.")
    
    return explanation + disclaimer

def _generate_gemma_explanation(prediction_data: dict, heatmap_analysis: dict, 
                               base_info: dict, gemma_model, gemma_tokenizer) -> str:
    """Generate enhanced explanation using Gemma model"""
    try:
        disease = prediction_data.get('disease', 'Unknown')
        scan_type = prediction_data.get('scan_type', 'Medical scan')
        confidence = prediction_data.get('disease_confidence', 0.5)
        
        # Build comprehensive prompt for patient explanation
        prompt = f"""
You are a medical AI assistant explaining scan results to a patient in simple, reassuring terms.

SCAN RESULTS:
- Scan Type: {scan_type}
- Finding: {disease}
- Confidence: {confidence:.1%}
- Condition: {base_info['condition']}
- Medical Description: {base_info['description']}

ANALYSIS DETAILS:
"""
        
        # Add heatmap analysis if available
        if heatmap_analysis:
            if 'severity' in heatmap_analysis:
                prompt += f"- Severity: {heatmap_analysis['severity']}\n"
            if 'doctor_note' in heatmap_analysis:
                prompt += f"- Clinical Note: {heatmap_analysis['doctor_note']}\n"
        
        prompt += f"""
TASK: Create a patient-friendly explanation that:
1. Explains what was found in simple terms
2. Describes what this means for the patient
3. Mentions expected symptoms: {base_info['symptoms']}
4. Provides next steps: {base_info['next_steps']}
5. Offers reassurance where appropriate
6. Emphasizes the importance of healthcare provider consultation

Keep the tone professional but warm and reassuring. Use everyday language, not medical jargon.

Patient Explanation:"""

        # Generate explanation
        inputs = gemma_tokenizer([prompt])
        outputs = gemma_model.generate(
            inputs,
            max_length=400,  # Reasonable length for patient explanation
            temperature=0.4,  # Slightly higher for more natural language
            stop_token_ids=[gemma_tokenizer.end_token_id]
        )
        
        decoded = gemma_tokenizer.detokenize(outputs)
        response = decoded.numpy()[0].decode('utf-8')
        
        # Extract the explanation
        if "Patient Explanation:" in response:
            explanation = response.split("Patient Explanation:")[-1].strip()
        else:
            explanation = response.strip()
        
        # Clean up the explanation
        explanation = explanation.replace("DISCLAIMER:", "").strip()
        
        # Ensure reasonable length
        sentences = explanation.split('. ')
        if len(sentences) > 12:  # Limit to reasonable length
            explanation = '. '.join(sentences[:12]) + '.'
        
        return explanation
        
    except Exception as e:
        print(f"Error generating Gemma explanation: {e}")
        return ""

def _generate_template_explanation(prediction_data: dict, heatmap_analysis: dict, base_info: dict) -> str:
    """Generate explanation using templates when Gemma is not available"""
    disease = prediction_data.get('disease', 'Unknown')
    scan_type = prediction_data.get('scan_type', 'Medical scan')
    confidence = prediction_data.get('disease_confidence', 0.5)
    
    # Confidence description
    if confidence >= 0.9:
        conf_desc = "with high confidence"
    elif confidence >= 0.7:
        conf_desc = "with good confidence"
    elif confidence >= 0.5:
        conf_desc = "with moderate confidence"
    else:
        conf_desc = "with low confidence - specialist review recommended"
    
    # Build explanation
    explanation = f"Based on your {scan_type} scan, our AI analysis indicates {base_info['condition']} {conf_desc}.\n\n"
    
    if disease.lower() != 'normal':
        explanation += f"What this means: This condition involves {base_info['description']}. "
        
        # Add severity information if available
        if heatmap_analysis.get('severity'):
            severity = heatmap_analysis['severity']
            explanation += f"The analysis suggests {severity} findings. "
        
        explanation += f"\n\nCommon symptoms may include: {base_info['symptoms']}.\n\n"
        explanation += f"Recommended next steps: {base_info['next_steps']}."
        
        # Add heatmap insights
        if heatmap_analysis.get('doctor_note'):
            explanation += f"\n\nTechnical details: {heatmap_analysis['doctor_note']}"
    
    else:
        explanation += f"This is positive news - {base_info['description']}. "
        explanation += f"However, {base_info['symptoms']}, so it's important to {base_info['next_steps']}."
    
    return explanation

def generate_uncertainty_explanation(primary_results: dict, gemma_results: dict = None) -> str:
    """
    Generate explanation when models are uncertain, with optional Gemma insight.
    
    Parameters:
    - primary_results: Results from primary models that were uncertain
    - gemma_results: Optional results from Gemma fallback analysis
    
    Returns:
    - Explanation string for uncertain cases
    """
    explanation = "Our primary AI models found some uncertainty in analyzing your scan. "
    
    # Explain primary model uncertainty
    if 'covid_confidence' in primary_results:
        covid_conf = primary_results['covid_confidence']
        normal_conf = primary_results['normal_confidence']
        explanation += f"The COVID-19 analysis showed {covid_conf:.1%} probability of COVID-19 and {normal_conf:.1%} probability of normal findings. "
        explanation += "For confident diagnosis, we typically require ≥90% confidence for COVID-19 or ≥80% confidence for normal findings.\n\n"
    
    # Add Gemma insight if available
    if gemma_results:
        explanation += f"Our secondary AI analysis suggests: {gemma_results.get('reasoning', 'Additional evaluation needed')}. "
        explanation += f"Recommendation: {gemma_results.get('recommendation', 'Please consult with a healthcare professional')}.\n\n"
        
        if gemma_results.get('confidence', 0) > 0.6:
            explanation += f"The secondary analysis has {gemma_results['confidence']:.1%} confidence in this assessment.\n\n"
    
    explanation += ("This uncertainty doesn't mean there's necessarily a problem - it often occurs with "
                   "borderline cases, image quality issues, or atypical presentations. A radiologist or "
                   "healthcare provider can provide definitive interpretation by considering your clinical "
                   "history, symptoms, and additional testing if needed.")
    
    return explanation

def format_technical_summary(prediction_data: dict, include_probabilities: bool = False) -> str:
    """
    Generate technical summary for healthcare providers.
    
    Parameters:
    - prediction_data: Full prediction results
    - include_probabilities: Whether to include raw probability scores
    
    Returns:
    - Technical summary string
    """
    summary = "=== AI ANALYSIS TECHNICAL SUMMARY ===\n"
    
    # Scan information
    summary += f"Scan Type: {prediction_data.get('scan_type', 'Unknown')} "
    summary += f"(confidence: {prediction_data.get('scan_type_confidence', 0):.3f})\n"
    
    summary += f"Anatomy: {prediction_data.get('anatomy', 'Unknown')} "
    summary += f"(confidence: {prediction_data.get('anatomy_confidence', 0):.3f})\n"
    
    # Primary diagnosis
    summary += f"Primary Finding: {prediction_data.get('disease', 'Unknown')} "
    summary += f"(confidence: {prediction_data.get('disease_confidence', 0):.3f})\n"
    
    # Model information
    primary_result = prediction_data.get('primary_model_result', 'unknown')
    summary += f"Primary Model Result: {primary_result}\n"
    
    fallback_used = prediction_data.get('fallback_used')
    if fallback_used:
        summary += f"Fallback System Used: {fallback_used}\n"
        
        if fallback_used == 'gemma':
            summary += f"Gemma Reasoning: {prediction_data.get('gemma_reasoning', 'N/A')}\n"
            summary += f"Gemma Recommendation: {prediction_data.get('gemma_recommendation', 'N/A')}\n"
    
    # Raw probabilities if requested
    if include_probabilities:
        summary += "\n=== RAW PROBABILITIES ===\n"
        if 'original_covid_confidence' in prediction_data:
            summary += f"COVID-19 Probability: {prediction_data['original_covid_confidence']:.3f}\n"
            summary += f"Normal Probability: {prediction_data['original_normal_confidence']:.3f}\n"
    
    summary += "\n=== CLINICAL NOTES ===\n"
    summary += ("This analysis is generated by AI models and should be used as a screening tool only. "
               "Clinical correlation and professional radiological interpretation are essential for "
               "definitive diagnosis and treatment planning.")
    
    return summary