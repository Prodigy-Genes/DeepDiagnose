 -- This SQL query retrieves user information from the 'users' table, including user ID, username, email, and the date the account was created. The results are ordered by the creation date in descending order.
SELECT user_id, username, email, created_at FROM users ORDER BY created_at DESC;


-- See your 18 medical images
-- See your medical images with key details
SELECT 
    image_id,
    user_id,
    original_filename,
    scan_type,
    anatomy,
    disease,
    disease_confidence,
    processed,
    uploaded_at
FROM medical_images 
ORDER BY uploaded_at DESC 
LIMIT 18;

-- See your diagnosis reports
SELECT 
    report_id,
    image_id,
    diagnosis_summary,
    overall_confidence,
    findings,
    recommendations,
    generated_at,
    reviewed
FROM diagnosis_reports 
ORDER BY generated_at DESC 
LIMIT 12;

-- Recent system activity
SELECT 
    log_id,
    user_id,
    action,
    details,
    resource_type,
    status,
    timestamp
FROM system_logs 
ORDER BY timestamp DESC 
LIMIT 15;

-- Summary of your medical analysis results
SELECT 
    scan_type,
    anatomy,
    disease,
    COUNT(*) as count,
    AVG(disease_confidence) as avg_confidence,
    COUNT(CASE WHEN processed = true THEN 1 END) as processed_count
FROM medical_images 
WHERE scan_type IS NOT NULL
GROUP BY scan_type, anatomy, disease
ORDER BY count DESC;

-- Which users are uploading and getting diagnosed
SELECT 
    u.username,
    u.email,
    COUNT(mi.image_id) as images_uploaded,
    COUNT(dr.report_id) as reports_generated,
    MAX(mi.uploaded_at) as last_upload,
    AVG(mi.disease_confidence) as avg_confidence
FROM users u
LEFT JOIN medical_images mi ON u.user_id = mi.user_id
LEFT JOIN diagnosis_reports dr ON mi.image_id = dr.image_id
GROUP BY u.user_id, u.username, u.email
HAVING COUNT(mi.image_id) > 0
ORDER BY images_uploaded DESC;

-- Check processing success rate
SELECT 
    processed,
    COUNT(*) as count,
    COUNT(CASE WHEN processing_error IS NOT NULL THEN 1 END) as error_count,
    COUNT(CASE WHEN disease IS NOT NULL THEN 1 END) as diagnosed_count
FROM medical_images
GROUP BY processed;

-- Latest AI diagnoses with confidence
SELECT 
    mi.original_filename,
    mi.scan_type,
    mi.anatomy,
    mi.disease,
    mi.disease_confidence,
    dr.diagnosis_summary,
    dr.overall_confidence,
    mi.uploaded_at
FROM medical_images mi
LEFT JOIN diagnosis_reports dr ON mi.image_id = dr.image_id
WHERE mi.disease IS NOT NULL
ORDER BY mi.uploaded_at DESC
LIMIT 10;
