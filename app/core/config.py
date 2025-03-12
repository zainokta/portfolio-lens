from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    port: int
    project_name: str
    project_description: str
    version: str

    db_name: str
    
    allowed_origins: str

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

settings = Settings()