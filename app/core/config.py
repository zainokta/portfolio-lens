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

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "portfolio"
    db_user: str = "postgres"
    db_password: str = ""
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}?sslmode=require"
    
    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL database URL."""
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}?sslmode=require"
    
    allowed_origins: str = "*"

    openai_api_key: str = ""
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse allowed_origins string into a list for CORS middleware."""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

settings = Settings()