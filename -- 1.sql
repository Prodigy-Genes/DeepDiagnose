-- 1. Verify new columns in medical_images
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'medical_images'
AND column_name IN (
    'original_filename', 'scan_type', 'scan_type_confidence',
    'anatomy', 'anatomy_confidence', 'disease', 'disease_confidence',
    'overlay_image_url', 'explanation', 'prediction_results',
    'processed_at', 'processing_error'
);

-- 2. Check removed modality column
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'medical_images' 
AND column_name = 'modality';

-- 3. Verify diagnosis_reports changes
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'diagnosis_reports'
AND column_name IN (
    'diagnosis_summary', 'overall_confidence', 
    'confidence_breakdown', 'recommendations',
    'reviewed', 'review_notes'
);

-- 4. Check system_logs additions
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'system_logs'
AND column_name IN (
    'user_agent', 'resource_id', 
    'resource_type', 'status'
);

-- 5. Verify reset_codes table
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'reset_codes';

-- 6. Check foreign key cascade
SELECT conname AS constraint_name,
       confdeltype AS delete_rule
FROM pg_constraint
WHERE conname = 'diagnosis_reports_image_id_fkey';

-- 7. Verify indexes
SELECT indexname, indexdef 
FROM pg_indexes
WHERE tablename IN ('medical_images', 'system_logs')
AND indexname LIKE 'idx_%';