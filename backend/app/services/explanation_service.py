from typing import Dict

def generate_patient_explanation(model_output: Dict, heatmap=None) -> str:
    """
    Create a simple textual explanation from the prediction output.
    This is intended for patient-friendly interpretation.
    """

    disease = model_output.get("disease", "Unknown")
    confidence = model_output.get("disease_confidence", 0)
    anatomy = model_output.get("anatomy", "Unknown")

    explanation = f"The system has detected a {disease} from your {anatomy.lower()} image with {round(confidence*100)}% confidence."

    if disease.lower() in ["normal", "no disease"]:
        explanation += " This means no signs of abnormality were found in the scan. However, always consult a healthcare professional."
    elif disease == "Uncertain - Consult Specialist":
        explanation = (
            "The system could not confidently classify the condition. "
            "Please consult a medical specialist for further evaluation."
        )
    else:
        explanation += " Please consult a healthcare professional for diagnosis confirmation and next steps."

    return explanation
