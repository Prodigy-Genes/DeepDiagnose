import numpy as np

def analyze_activation_patterns(heatmap: np.ndarray, disease: str) -> dict:
    """
    Simplified analysis returning only the doctor's note for the patient.
    """
    positives = heatmap[heatmap > 0]
    if positives.size == 0:
        return {"doctor_note": f"I reviewed your X-ray and I don’t see any signs of {disease}."}

    max_intensity = float(np.max(positives))
    area_fraction = positives.size / heatmap.size

    if max_intensity > 0.7 and area_fraction > 0.1:
        severity = "strong"
    elif max_intensity > 0.5 and area_fraction > 0.05:
        severity = "moderate"
    else:
        severity = "mild"

    notes = {
        "pneumonia": {
            "strong": "I found clear signs of pneumonia spread across your lungs.",
            "moderate": "I see signs of pneumonia in a few areas of your lungs.",
            "mild": "I notice some subtle changes that may indicate an early pneumonia."
        },
        "osteoarthritis": {
            "strong": "I found clear signs of osteoarthritis with wear around your joint.",
            "moderate": "I see moderate arthritis changes in your joint space.",
            "mild": "I notice early signs of osteoarthritis around your joint."
        }
    }

    default_notes = {
        "strong": f"I found clear signs of {disease}.",
        "moderate": f"I see signs of {disease}.",
        "mild": f"I notice some subtle changes that may relate to {disease}."
    }

    key = disease.lower()
    if key in notes:
        return {"doctor_note": notes[key][severity]}
    return {"doctor_note": default_notes[severity]}


def generate_patient_explanation(model_output: dict, heatmap: np.ndarray) -> str:
    """
    Generate a simple patient-facing explanation of X-ray findings as a single paragraph.
    """
    disease = model_output.get("disease", "Unknown")
    anatomy = model_output.get("anatomy", "")

    if disease.lower() == "normal":
        if anatomy == "chest":
            return "I’ve reviewed your chest X-ray and it looks normal; I don’t see any sign of pneumonia or other issues in your lungs."
        return "I’ve reviewed your joint X-ray and it looks normal; I don’t see any significant arthritis or joint damage."

    note = analyze_activation_patterns(heatmap, disease)["doctor_note"]
    return note
