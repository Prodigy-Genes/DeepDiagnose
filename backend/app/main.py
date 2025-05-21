from fastapi import FastAPI
from app.api.routes import auth

app = FastAPI()
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
