import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Boolean, Float, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base
from typing import TYPE_CHECKING

# Add conditional import for type checking
if TYPE_CHECKING:
    from .user_models import User  # Import the User model for type hints

class MedicalImage(Base):
    __tablename__ = "medical_images"

    image_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.user_id")
    )

    # Original image information
    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=True
    )
    
    image_url: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    # FIX: Use timezone-naive UTC datetime
    uploaded_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow  # Changed from lambda: datetime.now(timezone.utc)
    )

    processed: Mapped[bool] = mapped_column(
        default=False
    )

    # Scan type information from prediction
    scan_type: Mapped[str] = mapped_column(
        String(50),  # 'X-ray' or 'CT'
        nullable=True
    )
    
    scan_type_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=True
    )

    # Anatomy classification
    anatomy: Mapped[str] = mapped_column(
        String(100),  # 'Chest-scan', 'Joint-scan', 'CT Scan'
        nullable=True
    )
    
    anatomy_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=True
    )

    # Disease prediction
    disease: Mapped[str] = mapped_column(
        String(100),  # 'Pneumonia', 'COVID-19', 'Osteoarthritis', 'Normal'
        nullable=True
    )
    
    disease_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=True
    )

    # Overlay/heatmap image
    overlay_image_url: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    # Patient explanation
    explanation: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    # Store full prediction results as JSON for reference
    prediction_results: Mapped[dict] = mapped_column(
        JSON,
        nullable=True
    )

    # Processing metadata
    processed_at: Mapped[datetime] = mapped_column(
        nullable=True
    )
    
    processing_error: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship(
        back_populates="images"
    )

    # Optional: Keep diagnosis report relationship if you want detailed reports
    report: Mapped["DiagnosisReport"] = relationship(
        back_populates="image",
        uselist=False,
        cascade="all, delete-orphan"
    )

class DiagnosisReport(Base):
    """
    Optional detailed report - can store additional analysis
    """
    __tablename__ = "diagnosis_reports"

    report_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )

    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("medical_images.image_id", ondelete="CASCADE")
    )

    # Summary of diagnosis
    diagnosis_summary: Mapped[str] = mapped_column(
        Text
    )

    # Key findings
    findings: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    # Overall confidence
    overall_confidence: Mapped[float] = mapped_column(
        Float
    )

    # Detailed confidence breakdown
    confidence_breakdown: Mapped[dict] = mapped_column(
        JSON,
        nullable=True
    )

    # Recommendations
    recommendations: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    # FIX: Use timezone-naive UTC datetime
    generated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow  # Changed from lambda: datetime.now(timezone.utc)
    )

    # Optional: Medical professional review
    reviewed: Mapped[bool] = mapped_column(
        default=False
    )
    
    reviewed_by: Mapped[str] = mapped_column(
        String(100),
        nullable=True
    )
    
    review_notes: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )
    heatmap_url = mapped_column(String, nullable=True)

    # Relationship
    image: Mapped["MedicalImage"] = relationship(
        back_populates="report"
    )

class SystemLog(Base):
    __tablename__ = "system_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.user_id"),
        nullable=True
    )
    
    # Action performed
    action: Mapped[str] = mapped_column(
        String(100)  # 'image_upload', 'prediction', 'login', etc.
    )
    
    # Additional details
    details: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    # Request metadata
    ip_address: Mapped[str] = mapped_column(
        String(45),
        nullable=True
    )
    
    user_agent: Mapped[str] = mapped_column(
        String(500),
        nullable=True
    )

    # Related resource (e.g., image_id for predictions)
    resource_id: Mapped[str] = mapped_column(
        String(100),
        nullable=True
    )
    
    resource_type: Mapped[str] = mapped_column(
        String(50),  # 'medical_image', 'user', etc.
        nullable=True
    )

    # FIX: Use timezone-naive UTC datetime
    timestamp: Mapped[datetime] = mapped_column(
        default=datetime.utcnow  # Changed from lambda: datetime.now(timezone.utc)
    )

    # Status/result of the action
    status: Mapped[str] = mapped_column(
        String(20),  # 'success', 'error', 'warning'
        default='success'
    )

    # Relationship
    user: Mapped["User"] = relationship(
        back_populates="logs"
    )