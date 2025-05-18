import numpy as np


def analyze_activation_patterns(
    heatmap: np.ndarray,
    disease: str
) -> dict:
    """
    Simplified analysis returning a doctor's note based on activation heatmap and disease.

    Parameters:
    - heatmap: 2D numpy array of activation intensities (normalized 0–1).
    - disease: predicted disease label (e.g., "pneumonia").

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
            return {"doctor_note": "I reviewed your X-ray and I don’t see any abnormal findings."}
        severity = "mild"
    else:
        max_intensity = float(np.max(positives))
        area_fraction = positives.size / hm.size

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
            "moderate": "I see moderate arthritis changes in your joint space.",
            "mild": "I notice early signs of osteoarthritis around your joint."
        }
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
    Generate a patient-facing explanation of X-ray findings as a single paragraph.

    Parameters:
    - model_output: dict containing at least "disease" and optionally "anatomy".
    - heatmap: 2D numpy array for activation patterns.

    Returns:
    - String with the doctor's explanation.
    """
    disease = str(model_output.get("disease", "Unknown"))
    anatomy = str(model_output.get("anatomy", "")).lower()

    # Handle normal cases
    if disease.lower() == "normal":
        if "chest" in anatomy:
            return (
                "I’ve reviewed your chest X-ray and it looks normal; "
                "I don’t see any signs of pneumonia or other lung issues."
            )
        else:
            return (
                "I’ve reviewed your joint X-ray and it looks normal; "
                "I don’t see any significant arthritis or joint damage."
            )

    # For detected diseases, analyze heatmap patterns
    note_dict = analyze_activation_patterns(heatmap, disease)
    return note_dict.get("doctor_note", 
        "I reviewed your X-ray but couldn't generate a clear explanation.")
