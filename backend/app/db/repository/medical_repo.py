import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload
from typing import Optional, List, Dict, Any

from app.db.models.medical_models import MedicalImage, DiagnosisReport, SystemLog

async def create_medical_image(
    db: AsyncSession,
    user_id: uuid.UUID,
    original_filename: str,
    image_url: str
) -> MedicalImage:
    """Create a new medical image record"""
    
    medical_image = MedicalImage(
        user_id=user_id,
        original_filename=original_filename,
        image_url=image_url
    )
    
    db.add(medical_image)
    await db.commit()
    await db.refresh(medical_image)
    
    return medical_image

async def get_medical_image_by_id(
    db: AsyncSession,
    image_id: uuid.UUID,
    user_id: Optional[uuid.UUID] = None
) -> Optional[MedicalImage]:
    """Get medical image by ID, optionally filtered by user"""
    
    stmt = select(MedicalImage).where(MedicalImage.image_id == image_id)
    
    if user_id:
        stmt = stmt.where(MedicalImage.user_id == user_id)
    
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def get_medical_image_with_report(
    db: AsyncSession,
    image_id: uuid.UUID,
    user_id: Optional[uuid.UUID] = None
) -> Optional[MedicalImage]:
    """Get medical image with diagnosis report"""
    
    stmt = (
        select(MedicalImage)
        .options(selectinload(MedicalImage.report))
        .where(MedicalImage.image_id == image_id)
    )
    
    if user_id:
        stmt = stmt.where(MedicalImage.user_id == user_id)
    
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def get_user_medical_images(
    db: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
    processed_only: bool = False,
    disease_filter: Optional[str] = None
) -> List[MedicalImage]:
    """Get medical images for a user with optional filtering"""
    
    stmt = (
        select(MedicalImage)
        .where(MedicalImage.user_id == user_id)
        .order_by(MedicalImage.uploaded_at.desc())
        .limit(limit)
        .offset(offset)
    )
    
    if processed_only:
        stmt = stmt.where(MedicalImage.processed == True)
    
    if disease_filter:
        stmt = stmt.where(MedicalImage.disease.ilike(f"%{disease_filter}%"))
    
    result = await db.execute(stmt)
    return result.scalars().all()

async def update_medical_image_prediction(
    db: AsyncSession,
    image_id: uuid.UUID,
    prediction_results: Dict[str, Any],
    overlay_image_url: Optional[str] = None
) -> MedicalImage:
    """Update medical image with prediction results"""
    
    stmt = select(MedicalImage).where(MedicalImage.image_id == image_id)
    result = await db.execute(stmt)
    medical_image = result.scalar_one_or_none()
    
    if not medical_image:
        raise ValueError(f"Medical image with ID {image_id} not found")
    
    # Update fields
    medical_image.scan_type = prediction_results.get("scan_type")
    medical_image.scan_type_confidence = prediction_results.get("scan_type_confidence")
    medical_image.anatomy = prediction_results.get("anatomy")
    medical_image.anatomy_confidence = prediction_results.get("anatomy_confidence")
    medical_image.disease = prediction_results.get("disease")
    medical_image.disease_confidence = prediction_results.get("disease_confidence")
    medical_image.explanation = prediction_results.get("explanation")
    medical_image.prediction_results = prediction_results
    medical_image.overlay_image_url = overlay_image_url
    medical_image.processed = True
    medical_image.processed_at = datetime.now(timezone.utc)
    
    await db.commit()
    await db.refresh(medical_image)
    
    return medical_image

async def create_diagnosis_report(
    db: AsyncSession,
    image_id: uuid.UUID,
    diagnosis_summary: str,
    findings: str,
    overall_confidence: float,
    confidence_breakdown: Dict[str, float],
    recommendations: str
) -> DiagnosisReport:
    """Create a diagnosis report for a medical image"""
    
    report = DiagnosisReport(
        image_id=image_id,
        diagnosis_summary=diagnosis_summary,
        findings=findings,
        overall_confidence=overall_confidence,
        confidence_breakdown=confidence_breakdown,
        recommendations=recommendations
    )
    
    db.add(report)
    await db.commit()
    await db.refresh(report)
    
    return report

async def get_user_medical_statistics(
    db: AsyncSession,
    user_id: uuid.UUID
) -> Dict[str, Any]:
    """Get medical image statistics for a user"""
    
    # Total images
    total_stmt = select(func.count(MedicalImage.image_id)).where(
        MedicalImage.user_id == user_id
    )
    total_result = await db.execute(total_stmt)
    total_images = total_result.scalar()
    
    # Processed images
    processed_stmt = select(func.count(MedicalImage.image_id)).where(
        and_(MedicalImage.user_id == user_id, MedicalImage.processed == True)
    )
    processed_result = await db.execute(processed_stmt)
    processed_images = processed_result.scalar()
    
    # Disease distribution
    disease_stmt = (
        select(MedicalImage.disease, func.count(MedicalImage.image_id))
        .where(and_(MedicalImage.user_id == user_id, MedicalImage.processed == True))
        .group_by(MedicalImage.disease)
    )
    disease_result = await db.execute(disease_stmt)
    disease_distribution = dict(disease_result.fetchall())
    
    # Scan type distribution
    scan_type_stmt = (
        select(MedicalImage.scan_type, func.count(MedicalImage.image_id))
        .where(and_(MedicalImage.user_id == user_id, MedicalImage.processed == True))
        .group_by(MedicalImage.scan_type)
    )
    scan_type_result = await db.execute(scan_type_stmt)
    scan_type_distribution = dict(scan_type_result.fetchall())
    
    return {
        "total_images": total_images,
        "processed_images": processed_images,
        "pending_images": total_images - processed_images,
        "disease_distribution": disease_distribution,
        "scan_type_distribution": scan_type_distribution
    }

async def delete_medical_image(
    db: AsyncSession,
    image_id: uuid.UUID,
    user_id: uuid.UUID
) -> bool:
    """Delete a medical image and associated data"""
    
    stmt = select(MedicalImage).where(
        and_(MedicalImage.image_id == image_id, MedicalImage.user_id == user_id)
    )
    result = await db.execute(stmt)
    medical_image = result.scalar_one_or_none()
    
    if not medical_image:
        return False
    
    await db.delete(medical_image)
    await db.commit()
    
    return True

async def log_system_action(
    db: AsyncSession,
    user_id: Optional[uuid.UUID],
    action: str,
    details: str,
    resource_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    status: str = "success",
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> SystemLog:
    """Log a system action"""
    
    log_entry = SystemLog(
        user_id=user_id,
        action=action,
        details=details,
        resource_id=resource_id,
        resource_type=resource_type,
        status=status,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    db.add(log_entry)
    await db.commit()
    await db.refresh(log_entry)
    
    return log_entry

async def get_user_activity_logs(
    db: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 100,
    offset: int = 0,
    action_filter: Optional[str] = None
) -> List[SystemLog]:
    """Get activity logs for a user"""
    
    stmt = (
        select(SystemLog)
        .where(SystemLog.user_id == user_id)
        .order_by(SystemLog.timestamp.desc())
        .limit(limit)
        .offset(offset)
    )
    
    if action_filter:
        stmt = stmt.where(SystemLog.action == action_filter)
    
    result = await db.execute(stmt)
    return result.scalars().all()