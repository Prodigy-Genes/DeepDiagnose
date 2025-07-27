import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, Dict, Any
import base64
import os
from pathlib import Path

from app.db.models.medical_models import MedicalImage, DiagnosisReport, SystemLog
from app.db.models.user_models import User

class MedicalPredictionService:
    """Service for handling medical image predictions and database storage"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_medical_image_record(
        self,
        user_id: uuid.UUID,
        original_filename: str,
        image_url: str
    ) -> MedicalImage:
        """Create initial medical image record before processing"""
        
        medical_image = MedicalImage(
            user_id=user_id,
            original_filename=original_filename,
            image_url=image_url,
            processed=False
        )
        
        self.db.add(medical_image)
        await self.db.commit()
        await self.db.refresh(medical_image)
        
        # Log the upload
        await self._log_action(
            user_id=user_id,
            action="image_upload",
            details=f"Uploaded image: {original_filename}",
            resource_id=str(medical_image.image_id),
            resource_type="medical_image",
            status="success"
        )
        
        return medical_image
    
    async def store_prediction_results(
        self,
        image_id: uuid.UUID,
        prediction_results: Dict[str, Any],
        overlay_image_base64: Optional[str] = None
    ) -> MedicalImage:
        """Store prediction results in the database"""
        
        try:
            # Get the medical image record
            stmt = select(MedicalImage).where(MedicalImage.image_id == image_id)
            result = await self.db.execute(stmt)
            medical_image = result.scalar_one_or_none()
            
            if not medical_image:
                raise ValueError(f"Medical image with ID {image_id} not found")
            
            # Store overlay image if provided
            overlay_url = None
            if overlay_image_base64:
                overlay_url = await self._save_overlay_image(
                    image_id, 
                    overlay_image_base64
                )
            
            # Update medical image with prediction results
            medical_image.scan_type = prediction_results.get("scan_type")
            medical_image.scan_type_confidence = prediction_results.get("scan_type_confidence")
            medical_image.anatomy = prediction_results.get("anatomy")
            medical_image.anatomy_confidence = prediction_results.get("anatomy_confidence")
            medical_image.disease = prediction_results.get("disease")
            medical_image.disease_confidence = prediction_results.get("disease_confidence")
            medical_image.explanation = prediction_results.get("explanation")
            medical_image.prediction_results = prediction_results
            medical_image.overlay_image_url = overlay_url
            medical_image.processed = True
            medical_image.processed_at = datetime.now(timezone.utc)
            
            # Create diagnosis report
            report = DiagnosisReport(
                image_id=image_id,
                diagnosis_summary=self._create_diagnosis_summary(prediction_results),
                findings=self._extract_findings(prediction_results),
                overall_confidence=prediction_results.get("disease_confidence", 0.0),
                confidence_breakdown={
                    "scan_type_confidence": prediction_results.get("scan_type_confidence"),
                    "anatomy_confidence": prediction_results.get("anatomy_confidence"),
                    "disease_confidence": prediction_results.get("disease_confidence")
                },
                recommendations=self._generate_recommendations(prediction_results)
            )
            
            self.db.add(report)
            
            await self.db.commit()
            await self.db.refresh(medical_image)
            
            # Log successful prediction
            await self._log_action(
                user_id=medical_image.user_id,
                action="prediction",
                details=f"Successfully processed image with {prediction_results.get('disease', 'unknown')} prediction",
                resource_id=str(image_id),
                resource_type="medical_image",
                status="success"
            )
            
            return medical_image
            
        except Exception as e:
            # Log error
            if 'medical_image' in locals():
                await self._log_action(
                    user_id=medical_image.user_id,
                    action="prediction",
                    details=f"Error processing image: {str(e)}",
                    resource_id=str(image_id),
                    resource_type="medical_image",
                    status="error"
                )
                
                # Update image with error
                medical_image.processing_error = str(e)
                medical_image.processed_at = datetime.now(timezone.utc)
                await self.db.commit()
            
            raise e
    
    async def get_user_medical_images(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0
    ) -> list[MedicalImage]:
        """Get all medical images for a user"""
        
        stmt = (
            select(MedicalImage)
            .where(MedicalImage.user_id == user_id)
            .order_by(MedicalImage.uploaded_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_medical_image_with_report(
        self,
        image_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> Optional[MedicalImage]:
        """Get medical image with diagnosis report"""
        
        stmt = (
            select(MedicalImage)
            .where(
                MedicalImage.image_id == image_id,
                MedicalImage.user_id == user_id
            )
        )
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _save_overlay_image(
        self,
        image_id: uuid.UUID,
        base64_data: str
    ) -> str:
        """Save overlay image to storage and return URL"""
        
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_data)
            
            # Create storage directory if it doesn't exist
            storage_dir = Path("storage/overlays")
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image
            filename = f"{image_id}_overlay.png"
            file_path = storage_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            # Return relative URL (adjust based on your setup)
            return f"/storage/overlays/{filename}"
            
        except Exception as e:
            print(f"Error saving overlay image: {e}")
            return None
    
    def _create_diagnosis_summary(self, prediction_results: Dict[str, Any]) -> str:
        """Create a diagnosis summary from prediction results"""
        
        scan_type = prediction_results.get("scan_type", "Unknown")
        anatomy = prediction_results.get("anatomy", "Unknown")
        disease = prediction_results.get("disease", "Unknown")
        confidence = prediction_results.get("disease_confidence", 0)
        
        if disease == "Normal":
            return f"{scan_type} of {anatomy} shows normal findings with {confidence:.1%} confidence."
        else:
            return f"{scan_type} of {anatomy} suggests {disease} with {confidence:.1%} confidence."
    
    def _extract_findings(self, prediction_results: Dict[str, Any]) -> str:
        """Extract key findings from prediction results"""
        
        findings = []
        
        scan_type = prediction_results.get("scan_type")
        if scan_type:
            findings.append(f"Scan Type: {scan_type}")
        
        anatomy = prediction_results.get("anatomy")
        if anatomy:
            findings.append(f"Anatomical Region: {anatomy}")
        
        disease = prediction_results.get("disease")
        disease_conf = prediction_results.get("disease_confidence")
        if disease and disease_conf:
            findings.append(f"Primary Finding: {disease} ({disease_conf:.1%} confidence)")
        
        return " | ".join(findings)
    
    def _generate_recommendations(self, prediction_results: Dict[str, Any]) -> str:
        """Generate recommendations based on prediction results"""
        
        disease = prediction_results.get("disease", "").lower()
        confidence = prediction_results.get("disease_confidence", 0)
        
        if disease == "normal":
            return "No immediate medical intervention required. Continue routine check-ups as recommended by your healthcare provider."
        
        recommendations = []
        
        if confidence < 0.8:
            recommendations.append("Low confidence prediction - recommend consultation with a medical professional for confirmation.")
        
        if "covid" in disease:
            recommendations.append("If COVID-19 is suspected, isolate immediately and consult with healthcare provider for testing and treatment options.")
        elif "pneumonia" in disease:
            recommendations.append("Pneumonia suspected - seek immediate medical attention for proper diagnosis and treatment.")
        elif "osteoarthritis" in disease:
            recommendations.append("Osteoarthritis findings detected - consult with orthopedic specialist for comprehensive evaluation and treatment plan.")
        else:
            recommendations.append("Consult with appropriate medical specialist for further evaluation and treatment planning.")
        
        recommendations.append("This AI analysis is for screening purposes only and should not replace professional medical diagnosis.")
        
        return " ".join(recommendations)
    
    async def _log_action(
        self,
        user_id: uuid.UUID,
        action: str,
        details: str,
        resource_id: str = None,
        resource_type: str = None,
        status: str = "success",
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log system action"""
        
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
        
        self.db.add(log_entry)
        # Don't commit here - let the calling method handle commits