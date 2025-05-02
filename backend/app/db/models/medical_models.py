import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Boolean, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base


class MedicalImage(Base):
    __tablename__ = "medical_images"


    image_id:Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.user_id")
    )

    image_url: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    thumbnail_url: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        default=lambda : datetime.now(timezone.utc)
    )

    processed: Mapped[bool] = mapped_column(
        default=False
    )

    modality: Mapped[str] = mapped_column(
        String(20)
    )

    #Relationships
    user: Mapped["User"] = relationship(
        back_populates="images"
    )

    report: Mapped["DiagnosisReport"] = relationship(
        back_populates="image",
        uselist=False
    )

class DiagnosisReport(Base):
    __tablename__ = "diagnosis_reports"

    report_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4
    )

    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("medical_images.image_id")
    )

    diagnosis_result: Mapped[str] = mapped_column(
        Text
    )

    confidence_score: Mapped[float] = mapped_column(
        Float
    )

    findings: Mapped[str] = mapped_column(
        Text
    )

    heatmap_url: Mapped[str] = mapped_column(
        Text
    )

    generated_at: Mapped[datetime] = mapped_column(
        default= lambda : datetime.now(timezone.utc)
    )

    reviewed_by: Mapped[str] = mapped_column(
        String(100),
        nullable=True
    )

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
    
    action: Mapped[str] = mapped_column(
        String(100)
    )
    
    details: Mapped[str] = mapped_column(
        Text,
        nullable=True
    )

    ip_address: Mapped[str] = mapped_column(
        String(45),
        nullable=True
    )

    timestamp: Mapped[datetime] = mapped_column(
        default= lambda: datetime.now(timezone.utc)
    )

    user: Mapped["User"] = relationship(
        back_populates="logs"
    )