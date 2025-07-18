from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import ClassVar


class Settings(BaseSettings):
    DATABASE_URL: str 
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = 'HS256'
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  
    )


#Path Configuration
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
APP_DIR      = BASE_DIR / "app"
ML_DIR       = APP_DIR / "ml"

class MLPaths:
    MODELS_DIR   = ML_DIR / "models"
    PNEU_METRICS = ML_DIR / "pneu_metrics"
    ANAT_METRICS = ML_DIR / "ana_metrics"
    MEDICAL_SCAN_TYPE_METRICS = ML_DIR/"medical_scan_type_metrics"
    COVID_METRICS = ML_DIR / "covid_metrics"


# COVID confidence thresholds - match Streamlit exactly
class ConfidenceThresholds:
    COVID_CONFIDENCE_THRESHOLD = 0.90  # 90% confidence for COVID prediction
    NORMAL_CONFIDENCE_THRESHOLD = 0.80  # 80% confidence for Normal prediction

settings = Settings()

