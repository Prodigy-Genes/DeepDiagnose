-- View table schema:
SELECT 
  column_name, 
  data_type, 
  character_maximum_length,
  is_nullable,
  column_default
FROM information_schema.columns
WHERE table_name = 'users';

-- View sample data from the users table:
SELECT 
  user_id::text AS uuid_string,  -- Convert UUID to readable string
  username,
  email,
  created_at AT TIME ZONE 'UTC' AS created_utc,
  LENGTH(password_hash) AS hash_length  -- Verify BCrypt hash format
FROM users
LIMIT 10;

-- Check for username/email uniqueness:
SELECT 
  username, 
  COUNT(*) AS duplicates
FROM users
GROUP BY username
HAVING COUNT(*) > 1;


