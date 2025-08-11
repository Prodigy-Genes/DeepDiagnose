from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    user_id: UUID  # Changed from str to UUID
    email: EmailStr
    username: str
    created_at: datetime  
    
    model_config = {
        "from_attributes": True,  # Replaces orm_mode for v2
        "json_encoders": {
            UUID: str  # Converts UUID to string in JSON
        }
    }

class UserMe(BaseModel):
    """Schema for /auth/me endpoint response"""
    user_id: UUID
    email: EmailStr
    username: str
    created_at: datetime
    
    model_config = {
        "from_attributes": True,  # Replaces orm_mode for v2
        "json_encoders": {
            UUID: str  # Converts UUID to string in JSON
        }
    }