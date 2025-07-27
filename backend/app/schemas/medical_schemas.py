# app/schemas/medical_schemas.py
from pydantic import BaseModel, UUID4, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class MedicalImageBase(BaseModel):
    original_filename: Optional[str] = None
    scan_type: Optional[str] = None
    scan_type_confidence: Optional[float] = None
    anatomy: Optional[str] = None
    anatomy_confidence: Optional[float] = None
    disease: Optional[str] = None
    disease_confidence: Optional[float] = None
    explanation: Optional[str] = None

class MedicalImageCreate(MedicalImageBase):
    image_url: str

class MedicalImageUpdate(BaseModel):
    scan_type: Optional[str] = None
    scan_type_confidence: Optional[float] = None
    anatomy: Optional[str] = None
    anatomy_confidence: Optional[float] = None
    disease: Optional[str] = None
    disease_confidence: Optional[float] = None
    overlay_image_url: Optional[str] = None
    explanation: Optional[str] = None
    prediction_results: Optional[Dict[str, Any]] = None
    processed: bool = True

class MedicalImageOut(MedicalImageBase):
    image_id: UUID4
    user_id: UUID4
    image_url: str
    uploaded_at: datetime
    processed: bool
    overlay_image_url: Optional[str] = None
    processed_at: Optional[datetime] = None
    processing_error: Optional[str] = None

    class Config:
        from_attributes = True

class MedicalImageSummary(BaseModel):
    """Lightweight summary for listing views"""
    image_id: UUID4
    original_filename: Optional[str]
    uploaded_at: datetime
    processed: bool
    scan_type: Optional[str]
    anatomy: Optional[str]
    disease: Optional[str]
    disease_confidence: Optional[float]
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True

class DiagnosisReportBase(BaseModel):
    diagnosis_summary: str
    findings: Optional[str] = None
    overall_confidence: float
    confidence_breakdown: Optional[Dict[str, float]] = None
    recommendations: Optional[str] = None

class DiagnosisReportCreate(DiagnosisReportBase):
    image_id: UUID4

class DiagnosisReportOut(DiagnosisReportBase):
    report_id: UUID4
    image_id: UUID4
    generated_at: datetime
    reviewed: bool = False
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None

    class Config:
        from_attributes = True

class MedicalImageDetailed(MedicalImageOut):
    """Detailed view including diagnosis report"""
    diagnosis_report: Optional[DiagnosisReportOut] = None
    prediction_results: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """Response from the prediction endpoint"""
    image_id: UUID4
    scan_type: str
    scan_type_confidence: float
    anatomy: str
    anatomy_confidence: float
    disease: str
    disease_confidence: float
    explanation: str
    overlay_image: Optional[str] = None
    processed_at: datetime

class MedicalStatistics(BaseModel):
    """User's medical image statistics"""
    total_images: int
    processed_images: int
    pending_images: int
    disease_distribution: Dict[str, int]
    scan_type_distribution: Dict[str, int]

class MedicalImageList(BaseModel):
    """Paginated list of medical images"""
    images: List[MedicalImageSummary]
    total: int
    limit: int
    offset: int

class SystemLogOut(BaseModel):
    """System log entry"""
    log_id: UUID4
    user_id: Optional[UUID4]
    action: str
    details: Optional[str]
    ip_address: Optional[str]
    resource_id: Optional[str]
    resource_type: Optional[str]
    timestamp: datetime
    status: str

    class Config:
        from_attributes = True

# Request/Response models for specific endpoints
class PredictionRequest(BaseModel):
    """Not used directly since we use UploadFile, but good for documentation"""
    pass

class ImageUploadResponse(BaseModel):
    """Response after successful image upload"""
    message: str
    image_id: UUID4
    upload_url: Optional[str] = None

class DeleteImageResponse(BaseModel):
    message: str = "Medical image deleted successfully"

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Validation schemas
class ImageValidationResult(BaseModel):
    is_valid: bool
    message: str
    confidence: Optional[float] = None

# Medical analysis results
class AnalysisResult(BaseModel):
    scan_type: str = Field(..., description="Type of medical scan (X-ray, CT)")
    scan_type_confidence: float = Field(..., ge=0, le=1, description="Confidence in scan type classification")
    anatomy: str = Field(..., description="Anatomical region (Chest-scan, Joint-scan, CT Scan)")
    anatomy_confidence: float = Field(..., ge=0, le=1, description="Confidence in anatomy classification")
    disease: str = Field(..., description="Disease prediction (Normal, Pneumonia, COVID-19, Osteoarthritis)")
    disease_confidence: float = Field(..., ge=0, le=1, description="Confidence in disease prediction")
    explanation: str = Field(..., description="Patient-friendly explanation of results")
    
    class Config:
        schema_extra = {
            "example": {
                "scan_type": "X-ray",
                "scan_type_confidence": 0.95,
                "anatomy": "Chest-scan",
                "anatomy_confidence": 0.92,
                "disease": "Normal",
                "disease_confidence": 0.88,
                "explanation": "The X-ray of your chest appears normal with no signs of pneumonia or other abnormalities."
            }
        }

# Bulk operations
class BulkDeleteRequest(BaseModel):
    image_ids: List[UUID4] = Field(..., min_items=1, max_items=50)

class BulkDeleteResponse(BaseModel):
    deleted_count: int
    failed_deletions: List[UUID4] = []
    message: str

# Filter and search models
class MedicalImageFilter(BaseModel):
    disease: Optional[str] = None
    scan_type: Optional[str] = None
    anatomy: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    processed_only: bool = False
    min_confidence: Optional[float] = Field(None, ge=0, le=1)

class SearchMedicalImages(BaseModel):
    filters: MedicalImageFilter
    limit: int = Field(50, ge=1, le=100)
    offset: int = Field(0, ge=0)
    sort_by: str = Field("uploaded_at", regex="^(uploaded_at|disease_confidence|processed_at)$")
    sort_order: str = Field("desc", regex="^(asc|desc)$")

# Export and reporting schemas
class MedicalReportExport(BaseModel):
    """Schema for exporting medical reports"""
    user_id: UUID4
    export_format: str = Field("json", regex="^(json|csv|pdf)$")
    date_range: Optional[tuple[datetime, datetime]] = None
    include_images: bool = False
    include_overlays: bool = False

class ExportResponse(BaseModel):
    """Response for export requests"""
    export_id: UUID4
    status: str = Field(..., regex="^(pending|processing|completed|failed)$")
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    file_size: Optional[int] = None

# Medical professional review schemas
class ReviewRequest(BaseModel):
    """Request to mark a diagnosis as reviewed"""
    reviewed_by: str = Field(..., min_length=1, max_length=100)
    review_notes: Optional[str] = Field(None, max_length=1000)
    confirmed: bool = Field(True, description="Whether the AI diagnosis is confirmed")

class ReviewResponse(BaseModel):
    """Response after professional review"""
    message: str = "Diagnosis review recorded successfully"
    reviewed_at: datetime
    reviewed_by: str

# Advanced analytics schemas
class ConfidenceDistribution(BaseModel):
    """Distribution of confidence scores"""
    high_confidence: int = Field(..., description="Count of predictions with >90% confidence")
    medium_confidence: int = Field(..., description="Count of predictions with 70-90% confidence")
    low_confidence: int = Field(..., description="Count of predictions with <70% confidence")

class AdvancedStatistics(MedicalStatistics):
    """Extended statistics with more detailed analytics"""
    confidence_distribution: ConfidenceDistribution
    monthly_upload_trend: Dict[str, int]  # month -> count
    accuracy_by_scan_type: Dict[str, float]
    most_common_findings: List[Dict[str, Any]]
    average_processing_time: Optional[float] = None

# Health insights and trends
class HealthInsight(BaseModel):
    """Individual health insight"""
    insight_type: str = Field(..., regex="^(trend|recommendation|alert|info)$")
    title: str
    description: str
    severity: str = Field(..., regex="^(low|medium|high|critical)$")
    related_images: List[UUID4] = []
    created_at: datetime

class HealthInsights(BaseModel):
    """Collection of health insights for a user"""
    insights: List[HealthInsight]
    summary: str
    last_updated: datetime

# Image comparison schemas
class CompareImagesRequest(BaseModel):
    """Request to compare two medical images"""
    image_id_1: UUID4
    image_id_2: UUID4
    comparison_type: str = Field("basic", regex="^(basic|detailed|progression)$")

class ImageComparison(BaseModel):
    """Result of comparing two medical images"""
    image_1: MedicalImageSummary
    image_2: MedicalImageSummary
    similarity_score: Optional[float] = Field(None, ge=0, le=1)
    differences: List[str] = []
    progression_analysis: Optional[str] = None
    recommendations: Optional[str] = None

# Batch processing schemas
class BatchProcessingRequest(BaseModel):
    """Request for batch processing multiple images"""
    image_ids: List[UUID4] = Field(..., min_items=1, max_items=20)
    priority: str = Field("normal", regex="^(low|normal|high)$")
    notification_email: Optional[str] = None

class BatchProcessingStatus(BaseModel):
    """Status of batch processing job"""
    job_id: UUID4
    status: str = Field(..., regex="^(pending|processing|completed|failed|cancelled)$")
    total_images: int
    processed_images: int
    failed_images: int
    estimated_completion: Optional[datetime] = None
    results: Optional[List[UUID4]] = None

# API response wrappers
class ApiResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    items: List[Any]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool

# Notification schemas
class NotificationPreferences(BaseModel):
    """User notification preferences"""
    email_enabled: bool = True
    processing_complete: bool = True
    high_confidence_alerts: bool = True
    low_confidence_warnings: bool = True
    weekly_summary: bool = False
    monthly_report: bool = False

class Notification(BaseModel):
    """Individual notification"""
    notification_id: UUID4
    user_id: UUID4
    type: str = Field(..., regex="^(info|warning|alert|success)$")
    title: str
    message: str
    read: bool = False
    related_resource_id: Optional[UUID4] = None
    related_resource_type: Optional[str] = None
    created_at: datetime
    expires_at: Optional[datetime] = None

# Data validation and quality schemas
class ImageQualityMetrics(BaseModel):
    """Metrics for assessing image quality"""
    resolution_score: float = Field(..., ge=0, le=10)
    contrast_score: float = Field(..., ge=0, le=10)
    clarity_score: float = Field(..., ge=0, le=10)
    noise_level: float = Field(..., ge=0, le=10)
    overall_quality: str = Field(..., regex="^(poor|fair|good|excellent)$")
    suitable_for_analysis: bool

class ValidationReport(BaseModel):
    """Comprehensive validation report for uploaded images"""
    image_id: UUID4
    validation_passed: bool
    quality_metrics: ImageQualityMetrics
    validation_errors: List[str] = []
    validation_warnings: List[str] = []
    recommendations: List[str] = []

# Integration schemas for external systems
class ExternalSystemConfig(BaseModel):
    """Configuration for external system integration"""
    system_name: str
    endpoint_url: str
    api_key: Optional[str] = None
    enabled: bool = True
    sync_predictions: bool = False
    sync_reports: bool = False

class SyncStatus(BaseModel):
    """Status of data synchronization with external systems"""
    system_name: str
    last_sync: Optional[datetime] = None
    sync_status: str = Field(..., regex="^(success|failed|pending|in_progress)$")
    records_synced: int = 0
    errors: List[str] = []