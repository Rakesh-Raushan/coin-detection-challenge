"""
Centralized configuration management using Pydantic Settings.
All environment variables and paths are managed here for consistency.
"""
import os
from pathlib import Path
from functools import lru_cache


class Settings:
    """Application settings loaded from environment variables with sensible defaults."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    APP_DIR: Path = BASE_DIR / "app"
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    
    # Database
    DB_PATH: Path = DATA_DIR / "database.db"
    
    # Model configuration
    MODEL_PATH: Path
    
    # API configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Coin Detection API"
    VERSION: str = "1.0.0"
    
    # Detection settings
    CONFIDENCE_THRESHOLD: float = 0.25
    SLANT_THRESHOLD_LOW: float = 0.8
    SLANT_THRESHOLD_HIGH: float = 1.2
    
    def __init__(self):
        # Resolve model path from environment or default
        default_model_path = self.ARTIFACTS_DIR / "models" / "yolov8n-coin-finetuned.pt"
        self.MODEL_PATH = Path(os.getenv("MODEL_PATH", str(default_model_path))).resolve()
        
        # Allow override of data directory (useful for Docker)
        if os.getenv("DATA_DIR"):
            self.DATA_DIR = Path(os.getenv("DATA_DIR"))
            self.UPLOAD_DIR = self.DATA_DIR / "uploads"
            self.DB_PATH = self.DATA_DIR / "database.db"
        
        # Ensure directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def database_url(self) -> str:
        """SQLite database URL."""
        return f"sqlite:///{self.DB_PATH}"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Use this function to get settings throughout the application.
    """
    return Settings()


# Convenience export
settings = get_settings()
