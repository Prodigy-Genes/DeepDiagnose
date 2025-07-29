SELECT 
    image_id, 
    original_filename,
    uploaded_at,
    processed,
    scan_type,
    disease,
    disease_confidence
FROM medical_images
WHERE user_id = '8b8888d1-06bd-45b8-a5c9-3ac8dc82099e';

SELECT 
    r.report_id,
    r.diagnosis_summary,
    r.overall_confidence,
    r.generated_at,
    i.original_filename,
    i.overlay_image_url
FROM diagnosis_reports r
JOIN medical_images i ON r.image_id = i.image_id
WHERE i.user_id = '8b8888d1-06bd-45b8-a5c9-3ac8dc82099e';

SELECT 
    log_id,
    action,
    timestamp,
    resource_type,
    resource_id,
    status
FROM system_logs
WHERE user_id = '8b8888d1-06bd-45b8-a5c9-3ac8dc82099e'
ORDER BY timestamp DESC
LIMIT 10;