from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Add this import



app = FastAPI()

# Add CORS configuration - Place this BEFORE your route definitions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production!)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (including OPTIONS)
    allow_headers=["*"],
)
from app.api.routes import auth
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
