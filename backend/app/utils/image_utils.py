import base64
from io import BytesIO
from PIL import Image
import numpy as np

def encode_image_to_base64(overlay: np.ndarray) -> str | None:
    try:
        buf = BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_b64}"
    except Exception as e:
        print(f"Warning: Could not encode overlay image: {e}")
        return None
