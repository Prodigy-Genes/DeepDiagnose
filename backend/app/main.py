from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import auth
from app.api.routes import image

app = FastAPI(
    title="DeepDiagnose API",
    description="AI-powered diagnostic system for analyzing X-ray and CT scans.",
    version="1.0.0"
)

# Add CORS middleware (so frontend can make requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(image.router, prefix="/image", tags=["Image Prediction"])
