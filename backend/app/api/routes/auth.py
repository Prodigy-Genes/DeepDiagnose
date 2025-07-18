from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.user import UserCreate, UserOut
from app.services.auth_service import register_user, authenticate_user
from app.core.security import create_access_token, decode_token
from app.database import get_db
from app.db.repository.user_repo import get_user_by_email

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.post("/signup", response_model=UserOut)
async def signup(user: UserCreate, db: AsyncSession = Depends(get_db)):
    new_user = await register_user(db, user)
    return {
        "user_id": str(new_user.user_id),  # Convert UUID to string
        "username": new_user.username,
        "email": new_user.email
        }
@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db:AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form_data.username, form_data.password)
    print(user)
    print("User authenticated:", user.username)
    print("Proceeding to create access token")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await get_user_by_email(db, payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
