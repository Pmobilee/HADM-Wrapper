"""
HADM Server Configuration
"""
import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Configuration
    api_v1_prefix: str = Field(default="/api/v1", env="API_V1_PREFIX")
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Model Configuration
    model_path: str = Field(default="./pretrained_models", env="MODEL_PATH")
    hadm_l_model: str = Field(default="HADM-L_0249999.pth", env="HADM_L_MODEL")
    hadm_g_model: str = Field(default="HADM-G_0249999.pth", env="HADM_G_MODEL")
    eva02_model: str = Field(default="eva02_L_coco_det_sys_o365.pth", env="EVA02_MODEL")
    
    # Device Configuration
    device: str = Field(default="cuda", env="DEVICE")
    gpu_id: int = Field(default=0, env="GPU_ID")
    
    # Detectron2 Configuration
    detectron2_datasets: str = Field(default="./datasets", env="DETECTRON2_DATASETS")
    
    # Image Processing
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    supported_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "webp"], 
        env="SUPPORTED_FORMATS"
    )
    image_size: int = Field(default=1024, env="IMAGE_SIZE")
    
    # Security
    rate_limit: int = Field(default=100, env="RATE_LIMIT")  # requests per minute
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/hadm_server.log", env="LOG_FILE")
    
    # Model Inference
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    nms_threshold: float = Field(default=0.5, env="NMS_THRESHOLD")
    max_detections: int = Field(default=100, env="MAX_DETECTIONS")
    
    # Performance
    model_cache_size: int = Field(default=1, env="MODEL_CACHE_SIZE")
    preload_models: bool = Field(default=True, env="PRELOAD_MODELS")
    enable_model_ema: bool = Field(default=True, env="ENABLE_MODEL_EMA")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def hadm_l_model_path(self) -> str:
        """Full path to HADM-L model."""
        return os.path.join(self.model_path, self.hadm_l_model)
    
    @property
    def hadm_g_model_path(self) -> str:
        """Full path to HADM-G model."""
        return os.path.join(self.model_path, self.hadm_g_model)
    
    @property
    def eva02_model_path(self) -> str:
        """Full path to EVA-02 base model."""
        return os.path.join(self.model_path, self.eva02_model)


# Global settings instance
settings = Settings() 