from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / '.env'

class Settings(BaseSettings):
    port: int = 8000
    project_name: str = "PortfolioLens"
    project_description: str = "Suggesting the ability to look deeply into my work experience"
    version: str = "1.0.0"
    
    # Environment configuration
    environment: str = "development"  # development, staging, production

    db_name: str = "portfolio.db"
    
    allowed_origins: str = "*"

    openai_api_key: str = ""
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

settings = Settings()