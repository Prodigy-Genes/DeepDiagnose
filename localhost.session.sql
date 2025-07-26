-- View table schema:
SELECT 
  column_name, 
  data_type, 
  character_maximum_length,
  is_nullable,
  column_default
FROM information_schema.columns
WHERE table_name = 'users';

-- Query to retrieve all user data
SELECT * FROM users;
SELECT * FROM reset_codes;
-- Query to retrieve all reset codes with their status

SELECT 
    email,
    code,
    expires_at,
    used,
    created_at,
    CASE 
        WHEN expires_at < NOW() THEN 'EXPIRED' 
        WHEN used = true THEN 'USED'
        ELSE 'ACTIVE'
    END AS status
FROM reset_codes
ORDER BY created_at DESC;