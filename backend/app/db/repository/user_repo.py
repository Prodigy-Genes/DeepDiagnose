from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import Session
from uuid import UUID
from app.db.models.user_models import User  # Unified model import
from app.schemas.user import UserCreate  # Added for type safety
from app.core.security import hash_password # Added for password hashing
from fastapi import HTTPException
from typing import Optional, Dict

async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    """Get user by email address"""
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

async def get_user_by_username(session: AsyncSession, username: str) -> Optional[User]:
    """Get user by username"""
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()

async def get_user_by_id(session: AsyncSession, user_id: UUID) -> Optional[User]:
    """Get user by UUID"""
    result = await session.execute(select(User).where(User.user_id == user_id))
    return result.scalar_one_or_none()

async def create_user(
    session: AsyncSession, 
    user_data: UserCreate  # Use schema instead of dict
) -> User:
    """Create a new user with password hashing and proper error handling"""
    # Check for existing email or username
    if await get_user_by_email(session, user_data.email):
        raise HTTPException(
            status_code=400, 
            detail="Email already registered"
        )
    
    if await get_user_by_username(session, user_data.username):
        raise HTTPException(
            status_code=400, 
            detail="Username already taken"
        )
    
    # Hash password before storage
    hashed_password = hash_password(user_data.password)
    user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password  # FIXED: Changed from hashed_password to password_hash
    )
    
    session.add(user)
    try:
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Database error: {str(e)}"
        ) from e
    
    await session.refresh(user)
    return user

# Optional synchronous versions for legacy/compatibility
def get_user_by_email_sync(db: Session, email: str) -> Optional[User]:
    """Synchronous version - get user by email"""
    return db.query(User).filter(User.email == email).first()

def get_user_by_username_sync(db: Session, username: str) -> Optional[User]:
    """Synchronous version - get user by username"""
    return db.query(User).filter(User.username == username).first()