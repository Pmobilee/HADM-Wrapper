"""
HADM Server Main Application
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add HADM to Python path
from pathlib import Path
HADM_PATH = Path(__file__).parent.parent / "HADM"
sys.path.insert(0, str(HADM_PATH))

from app.core.config import settings
from app.core.hadm_models import model_manager
from app.api.endpoints import router
from app.models.schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.log_file) if settings.log_file else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting HADM Server...")
    
    # Create necessary directories
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    os.makedirs(settings.model_path, exist_ok=True)
    
    # Set environment variables
    os.environ["DETECTRON2_DATASETS"] = settings.detectron2_datasets
    
    # Load models if preloading is enabled
    if settings.preload_models:
        logger.info("Preloading HADM models...")
        model_status = model_manager.load_models()
        logger.info(f"Model loading status: {model_status}")
    
    logger.info("HADM Server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HADM Server...")
    logger.info("HADM Server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="HADM Server",
    description="FastAPI server for Human Artifact Detection in Machine-generated images",
    version="1.0.0",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this properly in production
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response


# Include API routes
app.include_router(
    router,
    prefix=settings.api_v1_prefix,
    tags=["HADM Detection"]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "HADM Server is running",
        "version": "1.0.0",
        "docs": f"{settings.api_v1_prefix}/docs" if settings.enable_docs else None,
        "health": f"{settings.api_v1_prefix}/health"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error",
            detail=str(exc) if settings.debug else "An unexpected error occurred",
            error_code="INTERNAL_ERROR"
        ).dict()
    )


# Application startup check
def check_requirements():
    """Check if all requirements are met."""
    errors = []
    
    # Check if model directory exists
    if not os.path.exists(settings.model_path):
        errors.append(f"Model directory not found: {settings.model_path}")
    
    # Check if model files exist
    required_models = [
        settings.hadm_l_model_path,
        settings.hadm_g_model_path,
        settings.eva02_model_path
    ]
    
    for model_path in required_models:
        if not os.path.exists(model_path):
            errors.append(f"Model file not found: {model_path}")
    
    if errors:
        logger.warning("Some requirements not met:")
        for error in errors:
            logger.warning(f"  - {error}")
        logger.warning("Server will start but some features may not work")
    
    return len(errors) == 0


if __name__ == "__main__":
    import time
    
    # Check requirements
    check_requirements()
    
    # Run the server
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # Use 1 worker for development
        log_level=settings.log_level.lower(),
        reload=settings.debug
    ) 