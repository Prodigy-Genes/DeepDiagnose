import httpx
import uvicorn  # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body, status # type: ignore
from uuid import UUID
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
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sys
import os
import logging
from sqlalchemy import select, update
from uuid import UUID, uuid4


# Set up proper logger
logger = logging.getLogger(__name__)

# Add the parent directories to Python path
current_dir = Path(__file__).resolve().parent
app_dir = current_dir.parent
backend_dir = app_dir.parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(backend_dir))
from fastapi.security import OAuth2PasswordBearer
from app.schemas.user import UserOut
from app.services.medical_service import MedicalPredictionService
from app.db.repository.medical_repo import log_system_action
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# Now import your modules
try:
    from database import get_db
    from api.routes.auth import get_current_user
    from db.models.user_models import User
    from services.medical_service import MedicalPredictionService
    from db.repository.medical_repo import log_system_action
    from utils import convert_numpy_types
    from db.models.medical_models import MedicalImage
    from db.repository.medical_repo import log_system_action

except ImportError as e:
    print(f"Import error: {e}")
    # Alternative import paths
    try:
        from app.database import get_db
        from app.api.routes.auth import get_current_user
        from app.db.models.user_models import User
        from app.services.medical_service import MedicalPredictionService
        from app.db.repository.medical_repo import log_system_action
        from app.utils import convert_numpy_types
        from app.db.models.medical_models import MedicalImage
        from app.db.repository.medical_repo import log_system_action

    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        # Create mock dependencies for development
        def get_db():
            return None
        def get_current_user():
            return None
        class User:
            def __init__(self):
                self.user_id = "test_user"
        
    class MedicalPredictionService:
        def __init__(self, db: AsyncSession):
            self.db = db

        async def create_medical_image_record(
            self,
            user_id: str,
            original_filename: str,
            image_url: str
        ) -> MedicalImage:
            """Create a new medical image record in the database"""
            try:
                # Create new medical image record
                medical_image = MedicalImage(
                    image_id=uuid4(),
                    user_id=UUID(user_id) if isinstance(user_id, str) else user_id,
                    original_filename=original_filename,
                    image_url=image_url,
                    uploaded_at=datetime.now(),
                    processed=False
                )
                
                self.db.add(medical_image)
                await self.db.commit()
                await self.db.refresh(medical_image)
                
                return medical_image
                
            except Exception as e:
                await self.db.rollback()
                raise Exception(f"Failed to create medical image record: {str(e)}")

        async def store_prediction_results(
            self,
            image_id: UUID,
            prediction_results: Dict[str, Any],
            overlay_image_base64: Optional[str] = None
        ) -> MedicalImage:
            """Store prediction results in the database with proper numpy type conversion"""
            try:
                # Convert numpy types to native Python types
                clean_results = convert_numpy_types(prediction_results)
                
                # Get the medical image record
                stmt = select(MedicalImage).where(MedicalImage.image_id == image_id)
                result = await self.db.execute(stmt)
                medical_image = result.scalar_one_or_none()
                
                if not medical_image:
                    raise Exception(f"Medical image with ID {image_id} not found")

                # Extract individual fields from prediction results
                scan_type = clean_results.get('scan_type')
                scan_type_confidence = clean_results.get('scan_type_confidence')
                anatomy = clean_results.get('anatomy')
                anatomy_confidence = clean_results.get('anatomy_confidence')
                disease = clean_results.get('disease')
                disease_confidence = clean_results.get('disease_confidence')
                explanation = clean_results.get('explanation')

                # Update the medical image record
                update_stmt = update(MedicalImage).where(
                    MedicalImage.image_id == image_id
                ).values(
                    processed=True,
                    scan_type=scan_type,
                    scan_type_confidence=scan_type_confidence,
                    anatomy=anatomy,
                    anatomy_confidence=anatomy_confidence,
                    disease=disease,
                    disease_confidence=disease_confidence,
                    overlay_image_url=overlay_image_base64,
                    explanation=explanation,
                    prediction_results=clean_results,  # Now properly converted
                    processed_at=datetime.now()
                )
                
                await self.db.execute(update_stmt)
                await self.db.commit()
                
                # Refresh the medical image to get updated data
                await self.db.refresh(medical_image)
                
                return medical_image
                
            except Exception as e:
                await self.db.rollback()
                # Log the error for debugging
                await log_system_action(
                    self.db,
                    user_id=medical_image.user_id,
                    action="prediction_storage_error",
                    details=f"Failed to store prediction results: {str(e)}",
                    resource_id=str(image_id),
                    resource_type="medical_image",
                    status="error"
                )
                raise Exception(f"Failed to store prediction results: {str(e)}")

        async def get_medical_image(self, image_id: UUID) -> Optional[MedicalImage]:
            """Get a medical image by ID"""
            try:
                stmt = select(MedicalImage).where(MedicalImage.image_id == image_id)
                result = await self.db.execute(stmt)
                return result.scalar_one_or_none()
            except Exception as e:
                raise Exception(f"Failed to get medical image: {str(e)}")

        async def get_user_medical_images(
            self,
            user_id: UUID,
            limit: int = 50,
            offset: int = 0,
            processed_only: bool = False
        ) -> list[MedicalImage]:
            """Get medical images for a user with pagination"""
            try:
                stmt = select(MedicalImage).where(MedicalImage.user_id == user_id)
                
                if processed_only:
                    stmt = stmt.where(MedicalImage.processed == True)
                
                stmt = stmt.order_by(MedicalImage.uploaded_at.desc()).limit(limit).offset(offset)
                
                result = await self.db.execute(stmt)
                return result.scalars().all()
                
            except Exception as e:
                raise Exception(f"Failed to get user medical images: {str(e)}")
        async def log_system_action(*args, **kwargs):
            pass

# Grad-CAM utilities
try:
    from ml.grad_cam_utils import (
        make_gradcam_heatmap,
        create_contoured_spot_heatmap,
        overlay_heatmap
    )
except ImportError:
    print("Warning: Grad-CAM utilities not found, creating mock functions")
    def make_gradcam_heatmap(x, model, last_conv):
        return np.random.random((224, 224))
    def create_contoured_spot_heatmap(img, heat, **kwargs):
        return img
    def overlay_heatmap(img, heat, **kwargs):
        return img

# Import explanation utilities
try:
    from api.explanation_utils import generate_patient_explanation
except ImportError:
    print("Warning: Explanation utils not found, creating mock function")
    def generate_patient_explanation(**kwargs):
        return "Analysis completed. Please consult with a healthcare professional for detailed interpretation."
    
# Configuration for auth server
AUTH_SERVER_URL = os.getenv("AUTH_SERVER_URL", "http://localhost:8000")

# Pydantic models for user data
class UserOut(BaseModel):
    user_id: str
    username: str
    email: str

# OAuth2 scheme for extracting tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{AUTH_SERVER_URL}/auth/login")

# Enhanced debugging version of get_current_user
async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    """
    Validate token with the auth server and get current user - with enhanced debugging
    """
    print(f"🔍 [AUTH DEBUG] Starting token validation")
    print(f"🔑 [AUTH DEBUG] Received token: {token[:20]}..." if token else "❌ [AUTH DEBUG] No token received")
    print(f"🌐 [AUTH DEBUG] Auth server URL: {AUTH_SERVER_URL}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            auth_url = f"{AUTH_SERVER_URL}/auth/me"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            print(f"📡 [AUTH DEBUG] Making request to: {auth_url}")
            print(f"📋 [AUTH DEBUG] Headers: Authorization: Bearer {token[:20]}...")
            
            response = await client.get(auth_url, headers=headers)
            
            print(f"📝 [AUTH DEBUG] Response status: {response.status_code}")
            print(f"📄 [AUTH DEBUG] Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                user_data = response.json()
                print(f"✅ [AUTH DEBUG] Auth successful for user: {user_data.get('username', 'Unknown')}")
                print(f"👤 [AUTH DEBUG] User data: {user_data}")
                
                return UserOut(
                    user_id=user_data["user_id"],
                    username=user_data["username"],
                    email=user_data["email"]
                )
            elif response.status_code == 401:
                print(f"❌ [AUTH DEBUG] 401 Unauthorized - Token invalid/expired")
                try:
                    error_detail = response.json()
                    print(f"📄 [AUTH DEBUG] Error response: {error_detail}")
                except:
                    print(f"📄 [AUTH DEBUG] Error response text: {response.text}")
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                print(f"❌ [AUTH DEBUG] Unexpected status code: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"📄 [AUTH DEBUG] Error response: {error_detail}")
                except:
                    print(f"📄 [AUTH DEBUG] Error response text: {response.text}")
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication failed",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
    except httpx.TimeoutException as e:
        print(f"⏰ [AUTH DEBUG] Timeout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable - timeout"
        )
    except httpx.RequestError as e:
        print(f"🌐 [AUTH DEBUG] Request error: {e}")
        print(f"🔍 [AUTH DEBUG] Error type: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to authentication service: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"🚨 [AUTH DEBUG] Unexpected error: {e}")
        print(f"🔍 [AUTH DEBUG] Error type: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}"
        )



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

from app.api.routes.auth import router as auth_router
app.include_router(auth_router, prefix="/auth", tags=["auth"])



# ----------------------
# DEBUG ENDPOINT (for token inspection)
# ----------------------
# Enhanced debug endpoint
@app.get("/debug-token")
async def debug_token(request: Request):
    headers = request.headers
    print("🔍 [DEBUG] Inspecting request headers:")
    
    # Print all headers for debugging
    for key, value in headers.items():
        print(f"📋 [DEBUG] {key}: {value}")
    
    # Check for Authorization header
    if "Authorization" not in headers:
        print("❌ [DEBUG] No Authorization header found")
        return {"status": "no_auth_header"}
    
    token = headers["Authorization"].split("Bearer ")[-1]
    print(f"🔑 [DEBUG] Token received: {token[:20]}...")
    
    # Validate token with auth server - FIX THIS SECTION
    auth_response = httpx.get(
        "http://localhost:8000/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5.0
    )
    
    # Handle auth server response
    if auth_response.status_code == 200:
        user_data = auth_response.json()
        print(f"✅ [DEBUG] Valid token for user: {user_data['username']}")
        return {"status": "valid", "user": user_data}
    else:
        print(f"❌ [DEBUG] Token validation failed: {auth_response.status_code}")
        return {"status": "invalid", "error": auth_response.text}

# Test endpoint with auth
@app.get("/test-auth")
async def test_auth(current_user: UserOut = Depends(get_current_user)):
    """
    Simple test endpoint to verify auth is working
    """
    return {
        "message": "Authentication successful!",
        "user": {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "email": current_user.email
        }
    }

# Test endpoint that manually checks auth server connectivity
@app.get("/test-auth-server")
async def test_auth_server():
    """
    Test direct connectivity to auth server
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test basic connectivity
            response = await client.get(f"{AUTH_SERVER_URL}/")
            print(f"🧪 [TEST] Auth server root - Status: {response.status_code}")
            
            return {
                "auth_server_url": AUTH_SERVER_URL,
                "status": "reachable",
                "root_status": response.status_code,
                "message": "Auth server is reachable"
            }
    except Exception as e:
        print(f"🚨 [TEST] Auth server unreachable: {e}")
        return {
            "auth_server_url": AUTH_SERVER_URL,
            "status": "unreachable",
            "error": str(e),
            "message": "Cannot reach auth server"
        }

# ----------------------
# FASTAPI APP MIDDLEWARE
# ----------------------
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
        
        return overlay, heat.tolist() if heat is not None else None
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
async def predict(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
    db = Depends(get_db),
    request: Request = None,
):
    # Log user making the request
    logger.info(f"📸 Prediction request from user: {current_user.username}")
    
     # Log headers for debugging
    headers = request.headers
    logger.debug("📋 Prediction request headers:")
    for key, value in headers.items():
        logger.debug(f"  {key}: {value}")
    
    # Check token exists
    token = headers.get("authorization", "").replace("Bearer ", "")
    logger.debug(f"🔑 Token received: {token[:10]}...")
    """Main prediction endpoint with JWT authentication"""
    
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/dicom"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. JPEG, PNG, JPG, or DICOM only.",
        )

    # Initialize service and get request info
    service = MedicalPredictionService(db)
    ip = request.client.host if request else None
    ua = request.headers.get("user-agent") if request else None

    # Log successful authentication
    await log_system_action(
        db, current_user.user_id, "authentication",
        "User authenticated for prediction", None, "user", "success", ip, ua
    )

    # Create image record
    record = await service.create_medical_image_record(
        user_id=current_user.user_id,
        original_filename=file.filename,
        image_url=f"temp/{uuid.uuid4()}_{file.filename}"
    )

    # Load and validate image
    try:
        data = await file.read()
        img = Image.open(BytesIO(data)).convert('RGB')
    except Exception as e:
        await log_system_action(
            db, current_user.user_id, "prediction",
            f"Invalid image file: {e}", str(record.image_id), "medical_image", "error", ip, ua
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file format"
        )

    # Validate medical image
    valid, msg = validate_medical_image(img)
    if not valid:
        await log_system_action(
            db, current_user.user_id, "prediction",
            f"Invalid medical image: {msg}", str(record.image_id), "medical_image", "error", ip, ua
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )

    # Run prediction in thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, run_medical_prediction, img)

    # Store results
    stored = await service.store_prediction_results(
        image_id=record.image_id,
        prediction_results=result,
        overlay_image_base64=result.get("overlay_image")
    )
    
    await log_system_action(
        db, current_user.user_id, "prediction",
        "Prediction completed successfully", str(record.image_id), "medical_image", "success", ip, ua
    )

    # Build response
    output = convert_numpy_types(result)
    output.update({
        "image_id": str(stored.image_id),
        "processed_at": stored.processed_at.isoformat(),
        "user_info": {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "email": current_user.email,
        },
        "status": "success",
        "message": "Image processed successfully",
    })
    
    return output

# Helper endpoint to get user_id after login (for frontend integration)
@app.post("/auth/get-user-id")
async def get_user_id_by_credentials(
    credentials: dict = Body(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user_id by providing login credentials
    Useful for clients that need user_id for the predict endpoint
    """
    try:
        identifier = credentials.get("username") or credentials.get("email")
        password = credentials.get("password")
        
        if not identifier or not password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Missing credentials",
                    "message": "Both username/email and password are required",
                    "action_required": "Please provide valid login credentials"
                }
            )
        
        # Import the authenticate_user function from your auth service
        from app.services.auth_service import authenticate_user
        user = await authenticate_user(db, identifier, password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "Invalid credentials",
                    "message": "Username/email or password is incorrect",
                    "action_required": "Please check your credentials or register if you don't have an account",
                    "register_endpoint": "/auth/signup"
                }
            )
        
        return {
            "user_id": str(user.user_id),
            "username": user.username,
            "email": user.email,
            "message": "Use this user_id for the predict endpoint",
            "predict_endpoint": "/predict"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Authentication error",
                "message": f"Could not authenticate user: {str(e)}",
                "action_required": "Please try again or contact support"
            }
        )


# Keep the list users endpoint for development
@app.get("/users/list")
async def list_users(db: AsyncSession = Depends(get_db), limit: int = 10):
    """
    Development endpoint to list existing users with their user_ids
    Remove this in production for security!
    """
    try:
        from sqlalchemy import text
        
        query = text("""
            SELECT user_id, username, email, created_at 
            FROM users 
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        
        result = await db.execute(query, {"limit": limit})
        users = result.fetchall()
        
        return {
            "users": [
                {
                    "user_id": str(user.user_id),
                    "username": user.username,
                    "email": user.email,
                    "created_at": user.created_at.isoformat() if user.created_at else None
                }
                for user in users
            ],
            "total": len(users),
            "message": "Use any user_id from this list for the predict endpoint",
            "predict_endpoint": "/predict"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Database error",
                "message": f"Could not fetch users: {str(e)}",
                "action_required": "Please check database connection"
            }
        )


def run_medical_prediction(img: Image.Image) -> dict:
    """
    Run the medical prediction pipeline - extracted from your original code
    This runs in a separate thread to avoid blocking the async event loop
    """
    try:
        # Step 1: Medical Scan Type Classification (X-ray vs CT)
        x_scan_type = preprocess_medical_scan_type(img)
        scan_pred = med_scan_model.predict(x_scan_type)[0, 0].item()
        
        # Determine scan type based on threshold
        if scan_pred >= med_scan_thresh:
            scan_type, scan_conf = 'X-ray', scan_pred
        else:
            scan_type, scan_conf = 'CT', 1 - scan_pred
        
        # Check confidence threshold
        if scan_conf < 0.8:
            raise HTTPException(400, "Unable to determine scan type with sufficient confidence. Please upload a clearer medical image.")

        # Step 2: Route based on scan type
        if scan_type == 'CT':
            # Preprocess image for COVID-19 model
            x_covid = preprocess_covid(img)
            
            # Use EXACT Streamlit prediction logic
            covid_result = make_enhanced_prediction(
                covid_model, 
                x_covid, 
                covid_thresh, 
                COVID_CONFIDENCE_THRESHOLD, 
                NORMAL_CONFIDENCE_THRESHOLD
            )
            
            # Handle uncertain predictions
            if covid_result['final_prediction'] == "Uncertain - Consult Specialist":
                raise HTTPException(400, f"Inconclusive result - specialist consultation required. COVID probability: {covid_result['covid_confidence']:.1%} (needs ≥90% for confident COVID diagnosis), Normal probability: {covid_result['normal_confidence']:.1%} (needs ≥80% for confident normal diagnosis). Please consult a healthcare professional.")
            
            disease = covid_result['final_prediction']
            disease_conf = covid_result['confidence']
            
            # Generate Grad-CAM overlay
            overlay, heat = generate_gradcam_overlay(img, x_covid, covid_model, covid_last_conv, 'covid-19')
            
            output = {
                "scan_type": scan_type,
                "scan_type_confidence": round(scan_conf, 3),
                "anatomy": "CT Scan",
                "anatomy_confidence": round(scan_conf, 3),
                "disease": disease,
                "disease_confidence": round(disease_conf, 3),
            }
            
        else:  # X-ray
            # Route to anatomy classification first
            x_anat = preprocess_anatomy(img)
            anat_pred = anat_model.predict(x_anat)[0, 0].item()
            
            if anat_pred >= anat_thresh:
                anatomy, anat_conf = 'Joint-scan', anat_pred
            else:
                anatomy, anat_conf = 'Chest-scan', 1 - anat_pred
            
            # Check confidence threshold
            if anat_conf < 0.8:
                raise HTTPException(400, "Unable to classify X-ray anatomy with sufficient confidence. Please upload a clearer X-ray image.")
            
            # Route to appropriate disease classification
            if anatomy == 'Joint-scan':
                # Route to Osteoarthritis detection
                x_osteo = preprocess_osteo(img)
                osteo_pred = osteo_model.predict(x_osteo)[0, 0].item()
                
                if osteo_pred >= anat_thresh:  # Using anatomy threshold for osteo
                    disease, disease_conf = 'Osteoarthritis', osteo_pred
                else:
                    disease, disease_conf = 'Normal', 1 - osteo_pred
                
                # Check confidence threshold
                if disease_conf < 0.8:
                    raise HTTPException(400, "Unable to classify osteoarthritis with sufficient confidence. Please consult a medical professional.")
                
                # Generate Grad-CAM overlay
                overlay, heat = generate_gradcam_overlay(img, x_osteo, osteo_model, None, 'osteoarthritis')
                
            else:  # Chest-scan
                # Route to Pneumonia detection
                x_pneu = preprocess_pneumonia(img)
                pneu_pred = pneu_model.predict(x_pneu)[0, 0].item()
                
                if pneu_pred >= pneu_thresh:
                    disease, disease_conf = 'Pneumonia', pneu_pred
                else:
                    disease, disease_conf = 'Normal', 1 - pneu_pred
                
                # Check confidence threshold
                if disease_conf < 0.8:
                    raise HTTPException(400, "Unable to classify pneumonia with sufficient confidence. Please consult a medical professional.")
                
                # Generate Grad-CAM overlay
                overlay, heat = generate_gradcam_overlay(img, x_pneu, pneu_model, pneu_last_conv, 'pneumonia')
            
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
        
    except Exception as e:
        raise e


# NEW ENDPOINTS FOR MEDICAL HISTORY

@app.get("/medical-images")
async def get_user_medical_images(
    limit: int = 50,
    offset: int = 0,
    processed_only: bool = False,
    disease_filter: str = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's medical images with optional filtering"""
    
    from app.db.repository.medical_repo import get_user_medical_images
    
    images = await get_user_medical_images(
        db, 
        current_user.user_id, 
        limit, 
        offset, 
        processed_only, 
        disease_filter
    )
    
    return {
        "images": [
            {
                "image_id": str(img.image_id),
                "original_filename": img.original_filename,
                "uploaded_at": img.uploaded_at.isoformat(),
                "processed": img.processed,
                "scan_type": img.scan_type,
                "anatomy": img.anatomy,
                "disease": img.disease,
                "disease_confidence": img.disease_confidence,
                "processed_at": img.processed_at.isoformat() if img.processed_at else None
            }
            for img in images
        ],
        "total": len(images),
        "limit": limit,
        "offset": offset
    }


@app.get("/medical-images/{image_id}")
async def get_medical_image_details(
    image_id: str,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific medical image"""
    
    from app.db.repository.medical_repo import get_medical_image_with_report
    
    try:
        image_uuid = uuid.UUID(image_id)
    except ValueError:
        raise HTTPException(400, "Invalid image ID format")
    
    image = await get_medical_image_with_report(db, image_uuid, current_user.user_id)
    
    if not image:
        raise HTTPException(404, "Medical image not found")
    
    result = {
        "image_id": str(image.image_id),
        "original_filename": image.original_filename,
        "uploaded_at": image.uploaded_at.isoformat(),
        "processed": image.processed,
        "scan_type": image.scan_type,
        "scan_type_confidence": image.scan_type_confidence,
        "anatomy": image.anatomy,
        "anatomy_confidence": image.anatomy_confidence,
        "disease": image.disease,
        "disease_confidence": image.disease_confidence,
        "explanation": image.explanation,
        "overlay_image_url": image.overlay_image_url,
        "prediction_results": image.prediction_results,
        "processed_at": image.processed_at.isoformat() if image.processed_at else None,
        "processing_error": image.processing_error
    }
    
    # Include diagnosis report if available
    if hasattr(image, 'report') and image.report:
        result["diagnosis_report"] = {
            "report_id": str(image.report.report_id),
            "diagnosis_summary": image.report.diagnosis_summary,
            "findings": image.report.findings,
            "overall_confidence": image.report.overall_confidence,
            "confidence_breakdown": image.report.confidence_breakdown,
            "recommendations": image.report.recommendations,
            "generated_at": image.report.generated_at.isoformat(),
            "reviewed": image.report.reviewed,
            "reviewed_by": image.report.reviewed_by
        }
    
    return result


@app.get("/medical-statistics")
async def get_medical_statistics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get medical image statistics for the current user"""
    
    from app.db.repository.medical_repo import get_user_medical_statistics
    
    stats = await get_user_medical_statistics(db, current_user.user_id)
    return stats


@app.delete("/medical-images/{image_id}")
async def delete_medical_image(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a medical image and its associated data"""
    
    from app.db.repository.medical_repo import delete_medical_image as delete_image_repo
    
    try:
        image_uuid = uuid.UUID(image_id)
    except ValueError:
        raise HTTPException(400, "Invalid image ID format")
    
    deleted = await delete_image_repo(db, image_uuid, current_user.user_id)
    
    if not deleted:
        raise HTTPException(404, "Medical image not found")
    
    return {"message": "Medical image deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)