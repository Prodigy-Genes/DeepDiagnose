from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr
from app.schemas.user import UserCreate, UserOut, UserMe
from app.services.auth_service import register_user, authenticate_user
from app.core.security import create_access_token, decode_token, hash_password
from app.database import get_db
from app.db.repository.user_repo import get_user_by_email, get_user_by_username
import random
import string
from datetime import datetime, timedelta
from app.services.email_service import send_reset_email  # Add this import
import os
import asyncio


router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# New Pydantic models for forgot password
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class VerifyCodeRequest(BaseModel):
    email: EmailStr
    code: str

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: str

# Define get_current_user FIRST so it's available for the /me endpoint
async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
) -> UserOut:
    """
    Dependency to extract current user from JWT token
    """
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # The subject could be either username or email - check both
    identifier = payload["sub"]
    
    # Try username first, then email
    user = await get_user_by_username(db, identifier)
    if not user:
        user = await get_user_by_email(db, identifier)
    
    if not user:
        # If we have user_id in the token, try that as last resort
        if "user_id" in payload:
            from app.db.repository.user_repo import get_user_by_id
            user = await get_user_by_id(db, payload["user_id"])
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Helper functions for forgot password functionality
async def store_reset_code(db: AsyncSession, email: str, code: str, expires_at: datetime):
    """Store reset code in database"""
    from sqlalchemy import text
    
    # Insert or update reset code
    query = text("""
        INSERT INTO reset_codes (email, code, expires_at, used, created_at) 
        VALUES (:email, :code, :expires_at, false, :created_at)
        ON CONFLICT (email) DO UPDATE SET 
            code = :code, 
            expires_at = :expires_at, 
            used = false,
            created_at = :created_at
    """)
    
    await db.execute(query, {
        "email": email,
        "code": code,
        "expires_at": expires_at,
        "created_at": datetime.utcnow()
    })
    await db.commit()
    
    

async def get_reset_code(db: AsyncSession, email: str, code: str):
    """Get reset code details from database"""
    from sqlalchemy import text
    
    query = text("""
        SELECT code, expires_at, used FROM reset_codes 
        WHERE email = :email AND code = :code
    """)
    
    result = await db.execute(query, {"email": email, "code": code})
    return result.fetchone()

async def mark_code_as_used(db: AsyncSession, email: str, code: str):
    """Mark reset code as used"""
    from sqlalchemy import text
    
    query = text("""
        UPDATE reset_codes SET used = true 
        WHERE email = :email AND code = :code
    """)
    
    await db.execute(query, {"email": email, "code": code})
    await db.commit()

async def update_user_password(db: AsyncSession, email: str, new_password_hash: str):
    """Update user password in database"""
    from sqlalchemy import text
    
    query = text("""
        UPDATE users SET password_hash = :password_hash 
        WHERE email = :email
    """)
    
    await db.execute(query, {"password_hash": new_password_hash, "email": email})
    await db.commit()

@router.post("/signup")
async def signup(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    try:
        new_user = await register_user(db, user)
        return {"message": "User created successfully", "user_id": str(new_user.user_id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
async def login(
    credentials: dict = Body(...),
    db: AsyncSession = Depends(get_db)
):
    """Login user and return access token"""
    identifier = credentials.get("username") or credentials.get("email")
    password = credentials.get("password")
    
    if not identifier or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"msg": "Missing username/email or password"}]
        )
    
    user = await authenticate_user(db, identifier, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create token with email as subject (more reliable since it's unique)
    # Include both username and user_id for flexibility
    access_token = create_access_token(
        data={
            "sub": user.email,  # Use email as primary identifier
            "username": user.username,
            "user_id": str(user.user_id)
        }
    )
    
    # Return both token and user data for frontend
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": {
            "user_id": str(user.user_id),
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat() if hasattr(user, 'created_at') else None
        }
    }

@router.get("/me", response_model=UserMe)
async def get_current_user_info(
    current_user: UserOut = Depends(get_current_user)
):
    """
    Get current user information
    Requires valid JWT token in Authorization header
    """
    return current_user

# NEW FORGOT PASSWORD ENDPOINTS
@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Send password reset code to user's email
    Always returns success for security (doesn't reveal if email exists)
    """
    try:
        user = await get_user_by_email(db, request.email)
        
        if user:
            # Generate 6-digit code
            code = ''.join(random.choices(string.digits, k=6))
            
            # Set expiration (use setting from environment)
            expiry_minutes = int(os.getenv("RESET_CODE_EXPIRY_MINUTES", 15))
            expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)
            
            # Store in database
            await store_reset_code(db, request.email, code, expires_at)
            
            # Run in thread to avoid blocking
            await asyncio.to_thread(
                send_reset_email, 
                request.email, 
                code
            )
        
        return {"message": "If the email exists in our system, a reset code has been sent"}
        
    except Exception as e:
        print(f"Error in forgot_password: {e}")
        return {"message": "If the email exists in our system, a reset code has been sent"}
    

@router.post("/verify-reset-code")
async def verify_reset_code(
    request: VerifyCodeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Verify the reset code sent to user's email
    """
    try:
        # Check if code exists and is valid
        result = await get_reset_code(db, request.email, request.code)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code"
            )
        
        if result.used:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Code has already been used"
            )
        
        if datetime.utcnow() > result.expires_at:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Code has expired"
            )
        
        # Code is valid
        return {"message": "Code verified successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in verify_reset_code: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while verifying the code"
        )

@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Reset user's password using verified code
    """
    try:
        # Re-verify the code (security)
        code_result = await get_reset_code(db, request.email, request.code)
        
        if not code_result or code_result.used or datetime.utcnow() > code_result.expires_at:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired code"
            )
        
        # Check if user exists
        user = await get_user_by_email(db, request.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User not found"
            )
        
        # Validate password strength (you can add more validation here)
        if len(request.new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Hash new password
        hashed_password = hash_password(request.new_password)
        
        # Update password and mark code as used (transaction)
        try:
            await update_user_password(db, request.email, hashed_password)
            await mark_code_as_used(db, request.email, request.code)
        except Exception as e:
            await db.rollback()
            raise e
        
        return {"message": "Password reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in reset_password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while resetting the password"
        )