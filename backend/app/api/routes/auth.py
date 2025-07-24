from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.user import UserCreate, UserOut, UserMe
from app.services.auth_service import register_user, authenticate_user
from app.core.security import create_access_token, decode_token
from app.database import get_db
from app.db.repository.user_repo import get_user_by_email, get_user_by_username

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

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