"""
VoxProof - AI Voice Detection API
FastAPI entry point for detecting AI-generated vs human voice samples.
"""

import asyncio
import concurrent.futures
import logging
import os
import sys
from contextlib import asynccontextmanager
from enum import Enum
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables
load_dotenv()

# Detect Railway/production environment
IS_PRODUCTION = os.getenv("RAILWAY_ENVIRONMENT") is not None or os.getenv("PRODUCTION") == "true"

# Configure logging based on environment
if IS_PRODUCTION:
    # Structured JSON logging for Railway
    import json
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "service": "voxproof",
                "message": record.getMessage(),
            }
            if record.exc_info:
                log_obj["exception"] = self.formatException(record.exc_info)
            if hasattr(record, "request_id"):
                log_obj["request_id"] = record.request_id
            if hasattr(record, "duration_ms"):
                log_obj["duration_ms"] = record.duration_ms
            return json.dumps(log_obj)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )
else:
    # Human-readable logging for development
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)

# Log startup mode
logger.info(f"Starting VoxProof API in {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'} mode")

# Reduce noise from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# Import modules
from audio.processing import get_processor
from model.model import get_model, create_dummy_weights, load_classifier, predict
from utils.explain import get_explainer


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration loaded from environment."""
    API_KEY: str = os.getenv("API_KEY", "")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/classifier.pth")
    WAV2VEC_MODEL: str = os.getenv("WAV2VEC_MODEL", "facebook/wav2vec2-base-960h")
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))


config = Config()

# Validate required configuration
if not config.API_KEY:
    raise ValueError("ERROR: API_KEY environment variable is required. Please set it in .env file.")


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


# Language aliases for flexible input
LANGUAGE_ALIASES = {
    # English variations
    "en": SupportedLanguage.ENGLISH,
    "eng": SupportedLanguage.ENGLISH,
    "english": SupportedLanguage.ENGLISH,
    # Tamil variations
    "ta": SupportedLanguage.TAMIL,
    "tam": SupportedLanguage.TAMIL,
    "tamil": SupportedLanguage.TAMIL,
    # Hindi variations
    "hi": SupportedLanguage.HINDI,
    "hin": SupportedLanguage.HINDI,
    "hindi": SupportedLanguage.HINDI,
    # Malayalam variations
    "ml": SupportedLanguage.MALAYALAM,
    "mal": SupportedLanguage.MALAYALAM,
    "malayalam": SupportedLanguage.MALAYALAM,
    # Telugu variations
    "te": SupportedLanguage.TELUGU,
    "tel": SupportedLanguage.TELUGU,
    "telugu": SupportedLanguage.TELUGU,
}


class AudioFormat(str, Enum):
    """Supported audio formats."""
    MP3 = "mp3"


class VoiceDetectionRequest(BaseModel):
    """Request body for voice detection endpoint."""
    language: str = Field(
        ..., 
        description="Language of the audio sample (accepts: English, en, eng, Tamil, ta, Hindi, hi, Malayalam, ml, Telugu, te)"
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
    
    # Store normalized language
    _normalized_language: SupportedLanguage = None
    
    @field_validator('language')
    @classmethod
    def normalize_language(cls, v: str) -> str:
        """Normalize language input to supported format."""
        normalized = v.strip().lower()
        
        # Check aliases first
        if normalized in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[normalized].value
        
        # Check if it matches any enum value (case-insensitive)
        for lang in SupportedLanguage:
            if normalized == lang.value.lower():
                return lang.value
        
        # If not found, raise error with helpful message
        valid_options = list(LANGUAGE_ALIASES.keys()) + [l.value for l in SupportedLanguage]
        raise ValueError(
            f"Unsupported language: '{v}'. "
            f"Valid options: {', '.join(sorted(set(valid_options)))}"
        )
    
    @field_validator('audioBase64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate and clean Base64 string."""
        # Strip data URL prefix if present
        if "base64," in v:
            v = v.split("base64,")[1]
        
        # Remove all whitespace and control characters
        import re
        v = re.sub(r'[\s\x00-\x1f\x7f-\x9f]', '', v)
        
        # Basic validation - should be alphanumeric with +/= characters
        if not re.match(r'^[A-Za-z0-9+/=]+$', v):
            raise ValueError("Invalid Base64 encoding - contains invalid characters")
        
        # Check minimum length for reasonable audio
        if len(v) < 100:
            raise ValueError("Audio data too short - please provide a valid audio file")
        
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
    if not IS_PRODUCTION:
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
        logger.info(f"Models loaded in {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")
    
    if IS_PRODUCTION:
        logger.info(f"VoxProof API Ready - Model: {config.MODEL_PATH}, Sample Rate: {config.SAMPLE_RATE}Hz")
    else:
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
# Request Timeout Middleware
# ============================================================================

REQUEST_TIMEOUT = 90  # 90 second timeout for requests


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to add timeout to requests."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Request timeout after {REQUEST_TIMEOUT}s: {request.url.path}")
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "message": "Request processing timeout",
                    "detail": f"Request took longer than {REQUEST_TIMEOUT} seconds. Try with shorter audio."
                }
            )


app.add_middleware(TimeoutMiddleware)


# Thread pool for CPU-intensive operations
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


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
    
    # Create log record with request context
    extra = {"request_id": request_id}
    logger.info(f"Request received - Language: {request.language}, Request ID: {request_id}", extra=extra)
    
    try:
        # Get singletons
        processor = get_processor(sample_rate=config.SAMPLE_RATE)
        model = get_model(
            classifier_path=config.MODEL_PATH,
            wav2vec_model_name=config.WAV2VEC_MODEL
        )
        explainer = get_explainer()
        
        # Define the CPU-intensive work
        def process_and_predict():
            # Step 1: Process audio
            logger.debug(f"[{request_id}] Processing audio...")
            waveform = processor.process_audio(request.audioBase64)
            
            # Step 2: Extract acoustic features  
            logger.debug(f"[{request_id}] Extracting features...")
            features = processor.extract_features(waveform)
            
            # Step 3: Run model inference
            logger.debug(f"[{request_id}] Running AI detection...")
            prediction = model.predict(
                waveform=waveform,
                acoustic_features=features,
                sample_rate=config.SAMPLE_RATE
            )
            
            return features, prediction
        
        # Run CPU-intensive work in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        features, prediction = await loop.run_in_executor(_executor, process_and_predict)
        
        # Step 4: Generate explanation
        explanation = explainer.generate_explanation(features, prediction)
        
        # Build response
        response = VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=Classification(prediction.classification),
            confidenceScore=prediction.confidence_score,
            explanation=explanation
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        extra = {"request_id": request_id, "duration_ms": round(elapsed_ms, 2)}
        logger.info(
            f"Request completed - Result: {prediction.classification}, "
            f"Confidence: {prediction.confidence_score:.1%}, Duration: {elapsed_ms:.0f}ms, "
            f"Request ID: {request_id}",
            extra=extra
        )
        
        return response
        
    except ValueError as e:
        # Input validation errors
        elapsed_ms = (time.time() - start_time) * 1000
        extra = {"request_id": request_id, "duration_ms": round(elapsed_ms, 2)}
        logger.error(f"Validation error - {e}, Request ID: {request_id}", extra=extra)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected errors
        elapsed_ms = (time.time() - start_time) * 1000
        extra = {"request_id": request_id, "duration_ms": round(elapsed_ms, 2)}
        logger.error(f"Unexpected error - {e}, Request ID: {request_id}", extra=extra, exc_info=True)
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
