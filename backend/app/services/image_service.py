import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import base64
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from app.services.model_service import ModelService

model_service = ModelService()

# Shortcut variables for easier use
med_scan_model     = model_service.models['scan_type']
covid_model        = model_service.models['covid']
pneu_model         = model_service.models['pneumonia']
anat_model         = model_service.models['anatomy']
osteo_model        = model_service.models['osteoarthritis']

med_scan_size      = model_service.metrics['scan_type']['size']
med_scan_norm      = model_service.metrics['scan_type']['norm']
med_scan_thresh    = model_service.metrics['scan_type']['threshold']

covid_size         = model_service.metrics['covid']['size']
covid_thresh       = model_service.metrics['covid']['threshold']
covid_last_conv    = model_service.metrics['covid']['last_conv']

pneu_size          = model_service.metrics['pneumonia']['size']
pneu_norm          = model_service.metrics['pneumonia']['norm']
pneu_thresh        = model_service.metrics['pneumonia']['threshold']
pneu_last_conv     = model_service.metrics['pneumonia']['last_conv']

anat_size          = model_service.metrics['anatomy']['size']
anat_norm          = model_service.metrics['anatomy']['norm']
anat_thresh        = model_service.metrics['anatomy']['threshold']



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


    




