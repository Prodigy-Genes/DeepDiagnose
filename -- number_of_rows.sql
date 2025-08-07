-- See how many records are in each table
SELECT 
    'users' as table_name, COUNT(*) as row_count FROM users
UNION ALL SELECT 'auth_tokens', COUNT(*) FROM auth_tokens
UNION ALL SELECT 'diagnosis_reports', COUNT(*) FROM diagnosis_reports  
UNION ALL SELECT 'medical_images', COUNT(*) FROM medical_images
UNION ALL SELECT 'otp_codes', COUNT(*) FROM otp_codes
UNION ALL SELECT 'reset_codes', COUNT(*) FROM reset_codes
UNION ALL SELECT 'signup_otps', COUNT(*) FROM signup_otps
UNION ALL SELECT 'system_logs', COUNT(*) FROM system_logs
UNION ALL SELECT 'alembic_version', COUNT(*) FROM alembic_version
ORDER BY row_count DESC;