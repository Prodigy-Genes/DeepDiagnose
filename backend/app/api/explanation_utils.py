import numpy as np


def analyze_activation_patterns(
    heatmap: np.ndarray,
    disease: str
) -> dict:
    """
    Simplified analysis returning a doctor's note based on activation heatmap and disease.

    Parameters:
    - heatmap: 2D numpy array of activation intensities (normalized 0–1).
    - disease: predicted disease label (e.g., "pneumonia", "covid-19").

    Returns:
    - dict with key "doctor_note" containing patient-facing comment.
    """
    # Ensure heatmap is numeric and 2D
    try:
        hm = np.asarray(heatmap, dtype=np.float32)
    except Exception:
        return {"doctor_note": "Invalid heatmap data provided."}

    if hm.ndim != 2:
        return {"doctor_note": "Heatmap must be a 2D array."}

    # Flatten and consider only positive activations
    positives = hm[hm > 0]

    # Default notes for unknown diseases
    default_notes = {
        "strong": f"I found pronounced signs of {disease}.",
        "moderate": f"I see moderate signs of {disease}.",
        "mild": f"I notice subtle indications of {disease}, early follow-up may be recommended."
    }

    # Handle no-activation cases
    if positives.size == 0:
        if disease.lower() == "normal":
            return {"doctor_note": "I reviewed your scan and I don't see any abnormal findings."}
        severity = "mild"
    else:
        max_intensity = float(np.max(positives))
        area_fraction = positives.size / hm.size
        mean_intensity = float(np.mean(positives))

        # Enhanced severity classification for COVID-19
        if disease.lower() == "covid-19":
            # COVID-19 often shows bilateral, ground-glass patterns
            if max_intensity > 0.6 and area_fraction > 0.15 and mean_intensity > 0.4:
                severity = "strong"
            elif max_intensity > 0.4 and area_fraction > 0.08 and mean_intensity > 0.25:
                severity = "moderate"
            else:
                severity = "mild"
        else:
            # Standard severity classification for other diseases
            if max_intensity > 0.7 and area_fraction > 0.1:
                severity = "strong"
            elif max_intensity > 0.5 and area_fraction > 0.05:
                severity = "moderate"
            else:
                severity = "mild"

    # Predefined notes for known diseases
    notes = {
        "pneumonia": {
            "strong": "I found clear signs of pneumonia spread across your lungs.",
            "moderate": "I see signs of pneumonia in a few areas of your lungs.",
            "mild": "I notice some subtle changes that may indicate early pneumonia."
        },
        "osteoarthritis": {
            "strong": "I found clear signs of osteoarthritis with wear around your joint.",
            "moderate": "I see moderate osteoarthritis changes in your joint space.",
            "mild": "I notice early signs of osteoarthritis around your joint."
        },
        "covid-19": {
            "strong": "I found clear signs of COVID-19 with characteristic patterns in both lungs, showing the ground-glass appearance typical of this infection.",
            "moderate": "I see moderate signs of COVID-19 in your lungs, with some of the typical imaging features we associate with this viral infection.",
            "mild": "I notice subtle changes in your lungs that may indicate early COVID-19, though these findings are mild and would benefit from clinical correlation."
        },
    }

    key = disease.lower()
    if key in notes:
        note = notes[key].get(severity)
    else:
        note = default_notes.get(severity)

    return {"doctor_note": note}


def generate_patient_explanation(
    model_output: dict,
    heatmap: np.ndarray
) -> str:
    """
    Generate a patient-facing explanation of scan findings as a single paragraph.

    Parameters:
    - model_output: dict containing at least "disease" and optionally "anatomy", "scan_type".
    - heatmap: 2D numpy array for activation patterns.

    Returns:
    - String with the doctor's explanation.
    """
    disease = str(model_output.get("disease", "Unknown"))
    anatomy = str(model_output.get("anatomy", "")).lower()
    scan_type = str(model_output.get("scan_type", "scan")).lower()

    # Handle normal cases
    if disease.lower() == "normal":
        if "chest" in anatomy or "ct" in scan_type:
            if "ct" in scan_type:
                return (
                    "I've reviewed your CT scan and it looks normal; "
                    "I don't see any signs of COVID-19."
                )
            else:
                return (
                    "I've reviewed your chest X-ray and it looks normal; "
                    "I don't see any signs of pneumonia."
                )
        elif "joint" in anatomy:
            return (
                "I've reviewed your joint X-ray and it looks normal; "
                "I don't see any significant osteoarthritis or joint damage."
            )
        else:
            return (
                f"I've reviewed your {scan_type} and it looks normal; "
                "I don't see any abnormal findings."
            )

    # Handle COVID-19 specific explanations
    if disease.lower() == "covid-19":
        # For detected COVID-19, analyze heatmap patterns
        note_dict = analyze_activation_patterns(heatmap, disease)
        base_note = note_dict.get("doctor_note", "I found signs of COVID-19 pneumonia in your CT scan.")
        
        # Add COVID-specific context
        additional_context = (
            " This finding is based on the imaging patterns characteristic of COVID-19, "
            "but should be correlated with your symptoms, exposure history, and possibly "
            "confirmed with additional testing as recommended by your healthcare provider."
        )
        
        return base_note + additional_context

    # For other detected diseases, analyze heatmap patterns
    note_dict = analyze_activation_patterns(heatmap, disease)
    base_note = note_dict.get("doctor_note", 
        f"I reviewed your {scan_type} but couldn't generate a clear explanation.")
    
    # Add general medical advice for positive findings
    if disease.lower() in ["pneumonia", "osteoarthritis"]:
        additional_context = (
            " I recommend discussing these findings with your healthcare provider "
            "to determine the best treatment approach for your specific situation."
        )
        return base_note + additional_context
    
    return base_note


def get_scan_type_description(scan_type: str) -> str:
    """
    Get a patient-friendly description of the scan type.
    
    Parameters:
    - scan_type: Type of medical scan (e.g., "X-ray", "CT")
    
    Returns:
    - String with patient-friendly description
    """
    descriptions = {
        "x-ray": "X-ray",
        "ct": "CT scan",
        "ct": "CT scan",
        "chest-scan": "chest imaging",
        "joint-scan": "joint imaging"
    }
    
    return descriptions.get(scan_type.lower(), scan_type)


def get_confidence_qualifier(confidence: float) -> str:
    """
    Get a qualifier based on confidence level for patient communication.
    
    Parameters:
    - confidence: Confidence score (0-1)
    
    Returns:
    - String qualifier for confidence level
    """
    if confidence >= 0.95:
        return "with high confidence"
    elif confidence >= 0.85:
        return "with good confidence"
    elif confidence >= 0.75:
        return "with moderate confidence"
    else:
        return "though this finding would benefit from additional review"