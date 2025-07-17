from fastapi import HTTPException, status
from app.core.security import hash_password, verify_password, create_access_token
from app.db.repository.user_repo import get_user_by_email, get_user_by_username, create_user

async def register_user(session, user_data):
    # Check email uniqueness
    if await get_user_by_email(session, user_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check username uniqueness - NEW CHECK
    if await get_user_by_username(session, user_data.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_pw = hash_password(user_data.password)
    return await create_user(session, {
        "email": user_data.email,
        "username": user_data.username,
        "password_hash": hashed_pw
    })

async def authenticate_user(session, identifier: str, password: str):
    # First try by email
    user = await get_user_by_email(session, identifier)
    # If not found, try by username
    if not user:
        user = await get_user_by_username(session, identifier)
    
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user
