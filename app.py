"""
VoxProof - AI Voice Detection API
FastAPI entry point for detecting AI-generated vs human voice samples.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from enum import Enum
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # Changed to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)  # Changed to ERROR
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# Import modules
from audio.processing import get_processor
from model.model import get_model, create_dummy_weights
from utils.explain import get_explainer


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration loaded from environment."""
    API_KEY: str = os.getenv("API_KEY", "voxproof-secret-key-2024")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/classifier.pth")
    WAV2VEC_MODEL: str = os.getenv("WAV2VEC_MODEL", "facebook/wav2vec2-base-960h")
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))


config = Config()


# ============================================================================
# Pydantic Models
# ============================================================================

class SupportedLanguage(str, Enum):
    """Supported input languages."""
    TAMIL = "Tamil"
    ENGLISH = "English"
    HINDI = "Hindi"
    MALAYALAM = "Malayalam"
    TELUGU = "Telugu"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    MP3 = "mp3"


class VoiceDetectionRequest(BaseModel):
    """Request body for voice detection endpoint."""
    language: SupportedLanguage = Field(
        ..., 
        description="Language of the audio sample"
    )
    audioFormat: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Format of the audio file"
    )
    audioBase64: str = Field(
        ..., 
        description="Base64 encoded audio data",
        min_length=100  # Minimum reasonable audio size
    )
    
    @field_validator('audioBase64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that the string looks like valid Base64."""
        # Strip data URL prefix if present
        if "base64," in v:
            v = v.split("base64,")[1]
        
        # Basic validation - should be alphanumeric with +/= characters
        import re
        if not re.match(r'^[A-Za-z0-9+/=]+$', v.replace('\n', '').replace('\r', '')):
            raise ValueError("Invalid Base64 encoding")
        
        return v


class Classification(str, Enum):
    """Classification result types."""
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"


class VoiceDetectionResponse(BaseModel):
    """Response body for voice detection endpoint."""
    status: Literal["success"] = "success"
    language: str = Field(..., description="Input language (echoed back)")
    classification: Classification = Field(
        ..., 
        description="Classification result: AI_GENERATED or HUMAN"
    )
    confidenceScore: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score from 0.0 to 1.0"
    )
    explanation: str = Field(
        ..., 
        description="Human-readable explanation for the decision"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    status: Literal["error"] = "error"
    message: str
    detail: Optional[str] = None


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Load models at startup, cleanup at shutdown.
    """
    logger.info("\n" + "=" * 70)
    logger.info("🎙️  VoxProof API - AI Voice Detection System")
    logger.info("=" * 70)
    logger.info("Starting initialization...")
    
    # Create dummy weights if not exists (for demo purposes)
    if not os.path.exists(config.MODEL_PATH):
        logger.warning("⚠️  No classifier weights found. Creating dummy weights for demo...")
        logger.warning("⚠️  Run 'python train.py' to train with real data!")
        create_dummy_weights(config.MODEL_PATH)
    
    # Pre-load models at startup
    logger.info("📦 Loading models...")
    import time
    start_time = time.time()
    
    try:
        # Initialize audio processor
        processor = get_processor(sample_rate=config.SAMPLE_RATE)
        logger.info("  ✓ Audio processor initialized")
        
        # Initialize and load voice detection model
        model = get_model(
            classifier_path=config.MODEL_PATH,
            wav2vec_model_name=config.WAV2VEC_MODEL
        )
        model.load()
        logger.info("  ✓ Voice detection model loaded")
        
        # Initialize explanation generator
        explainer = get_explainer()
        logger.info("  ✓ Explanation generator initialized")
        
        load_time = time.time() - start_time
        logger.info(f"⏱️  Models loaded in {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")
    
    logger.info("=" * 70)
    logger.info("✅ VoxProof API Ready!")
    logger.info(f"🔑 API Key: {'Configured ✓' if config.API_KEY else 'NOT SET ✗'}")
    logger.info(f"🤖 Model: {config.MODEL_PATH}")
    logger.info(f"🎵 Sample Rate: {config.SAMPLE_RATE} Hz")
    logger.info(f"📡 Access the API at: http://localhost:8000")
    logger.info(f"📚 API Docs: http://localhost:8000/docs")
    logger.info("=" * 70 + "\n")
    
    yield  # Application runs here
    
    # Cleanup
    logger.info("VoxProof API Shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="VoxProof API",
    description="AI Voice Detection API - Detect whether a voice sample is AI-generated or spoken by a human",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependencies
# ============================================================================

async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> str:
    """Verify the API key from request header."""
    if x_api_key != config.API_KEY:
        logger.warning(f"🔒 Invalid API key attempt: {x_api_key[:8]}***")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "service": "VoxProof API",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": True,
        "sample_rate": config.SAMPLE_RATE
    }


@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API Key"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Voice Detection"],
    summary="Detect AI-generated vs Human voice",
    description="Analyze an audio sample to determine if it's AI-generated or spoken by a human"
)
async def voice_detection(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main voice detection endpoint.
    
    Analyzes the provided audio sample and returns:
    - Classification: AI_GENERATED or HUMAN
    - Confidence score: 0.0 to 1.0
    - Explanation: Reason for the decision
    """
    import time
    import uuid
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"🎯 [{request_id}] New request - Language: {request.language.value}")
    
    try:
        # Get singletons
        processor = get_processor(sample_rate=config.SAMPLE_RATE)
        model = get_model(
            classifier_path=config.MODEL_PATH,
            wav2vec_model_name=config.WAV2VEC_MODEL
        )
        explainer = get_explainer()
        
        # Step 1: Process audio
        logger.info(f"  [{request_id}] 🎵 Processing audio...")
        waveform = processor.process_audio(request.audioBase64)
        
        # Step 2: Extract acoustic features
        logger.info(f"  [{request_id}] 📊 Extracting features...")
        features = processor.extract_features(waveform)
        
        # Step 3: Run model inference
        logger.info(f"  [{request_id}] 🤖 Running AI detection...")
        prediction = model.predict(
            waveform=waveform,
            acoustic_features=features,
            sample_rate=config.SAMPLE_RATE
        )
        
        # Step 4: Generate explanation
        explanation = explainer.generate_explanation(features, prediction)
        
        # Build response
        response = VoiceDetectionResponse(
            status="success",
            language=request.language.value,
            classification=Classification(prediction.classification),
            confidenceScore=prediction.confidence_score,
            explanation=explanation
        )
        
        elapsed = time.time() - start_time
        logger.info(
            f"✅ [{request_id}] Complete in {elapsed:.2f}s - "
            f"Result: {prediction.classification} ({prediction.confidence_score:.1%} confidence)"
        )
        
        return response
        
    except ValueError as e:
        # Input validation errors
        logger.error(f"❌ [{request_id}] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"❌ [{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during voice analysis. Please try again."
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Railway sets this automatically)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
