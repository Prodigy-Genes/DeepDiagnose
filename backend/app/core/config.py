from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # Add the missing fields from your .env file
    app_name: str
    debug: bool
    
    # Database configuration
    DATABASE_URL: str 
    
    # Security configuration
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = 'HS256'
    
     # Email configuration fields
    SMTP_SERVER: str
    SMTP_PORT: int
    SMTP_USERNAME: str
    SMTP_PASSWORD: str
    SMTP_FROM_EMAIL: str
    RESET_CODE_EXPIRY_MINUTES: int = 15  # Default value
    
    # model configuration
    model_config = ConfigDict(env_file='.env', env_file_encoding='utf-8')
    
    database_url_sync: str
settings = Settings()

