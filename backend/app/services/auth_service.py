from fastapi import HTTPException, status
from app.core.security import hash_password, verify_password, create_access_token
from app.db.repository.user_repo import get_user_by_email, get_user_by_username, create_user

async def register_user(session, user_data):
    """
    Register a new user - used after OTP verification
    This function assumes the user data has already been validated
    """
    # Create user directly (create_user already handles validation and password hashing)
    return await create_user(session, user_data)

async def authenticate_user(session, identifier: str, password: str):
    """
    Authenticate user for login
    """
    # First try by email
    user = await get_user_by_email(session, identifier)
    # If not found, try by username
    if not user:
        user = await get_user_by_username(session, identifier)
    
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user

# Optional: Helper function to validate user data before sending OTP
async def validate_signup_data(session, email: str, username: str, password: str):
    """
    Validate signup data before sending OTP
    Returns tuple (is_valid, error_message)
    """
    # Check email format (basic check - FastAPI's EmailStr handles most of this)
    if not email or '@' not in email:
        return False, "Invalid email format"
    
    # Check username requirements
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(username) > 50:
        return False, "Username must be less than 50 characters"
    
    # Check if username contains only allowed characters
    import re
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    
    # Check password strength
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    # Check for existing email
    if await get_user_by_email(session, email):
        return False, "Email already registered"
    
    # Check for existing username
    if await get_user_by_username(session, username):
        return False, "Username already taken"
    
    return True, None