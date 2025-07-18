from fastapi import HTTPException, status
from app.core.security import hash_password, verify_password, create_access_token
from app.db.repository.user_repo import get_user_by_email, create_user

async def register_user(session, user_data):
    existing = await get_user_by_email(session, user_data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = hash_password(user_data.password)
    return await create_user(session, {
        # "user_id":str(user_data.user_id),
        "email": user_data.email,
        "username": user_data.username,
        "password_hash": hashed_pw
    })

async def authenticate_user(session, email:str, password: str):
    user = await get_user_by_email(session, email)
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user

