from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = 'HS256'
    MODEL_CONFIG =  SettingsConfigDict(env_file=".env")

settings = Settings()
