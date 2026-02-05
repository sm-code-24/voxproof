"""
VoxProof AI Voice Detection Model
=================================

Architecture:
    Audio Input → Decoding → [Acoustic Features + Wav2Vec2 Embeddings] 
    → Feature Fusion → VoiceClassifier → Probability → Classification

Components:
    - Wav2VecEmbedder: Extracts 768-dim embeddings (pretrained, frozen)
    - AudioProcessor: Extracts 18 acoustic features via librosa
    - VoiceClassifier: Custom PyTorch classifier (786-dim → binary)
    - VoiceDetectionModel: Main inference engine

Author: VoxProof Team
License: MIT
"""

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

# Suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Lazy import transformers to speed up startup
Wav2Vec2Model = None
Wav2Vec2Processor = None

def _load_transformers():
    """Lazily load transformers to speed up startup."""
    global Wav2Vec2Model, Wav2Vec2Processor
    if Wav2Vec2Model is None:
        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import Wav2Vec2Model as _Wav2Vec2Model
        from transformers import Wav2Vec2Processor as _Wav2Vec2Processor
        Wav2Vec2Model = _Wav2Vec2Model
        Wav2Vec2Processor = _Wav2Vec2Processor

from audio.processing import AudioFeatures

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PredictionResult:
    """Structured prediction output."""
    classification: str  # "AI_GENERATED" or "HUMAN"
    confidence_score: float  # 0.0 - 1.0
    raw_logit: float  # Raw model output before threshold


# ============================================================================
# Wav2Vec2 Embedding Extractor (Pretrained, Frozen)
# ============================================================================

class Wav2VecEmbedder:
    """
    Extracts 768-dimensional embeddings using pretrained Wav2Vec2.
    
    This is a PURE FEATURE EXTRACTOR - no classification head.
    The model is frozen and used only to transform audio into embeddings.
    """
    
    EMBEDDING_DIM = 768
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None  # Lazy loaded
        self.model = None  # Lazy loaded
        self._loaded = False
        
    def load(self) -> None:
        """Load the pretrained model (lazy loading)."""
        if self._loaded:
            return
        
        # Lazy load transformers
        _load_transformers()
            
        logger.info(f"Loading Wav2Vec2 embedder: {self.model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters - we only extract embeddings
        for param in self.model.parameters():
            param.requires_grad = False
            
        self._loaded = True
        logger.info(f"  ✓ Wav2Vec2 loaded on {self.device}")
        
    @torch.no_grad()
    def extract_embedding(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract embedding from audio waveform with optimized chunked processing.
        
        For long audio, processes in chunks and averages embeddings for speed.
        
        Args:
            waveform: Audio samples as numpy array
            sample_rate: Sample rate (must be 16kHz for Wav2Vec2)
            
        Returns:
            768-dimensional embedding vector
        """
        if not self._loaded:
            self.load()
        
        # Chunk size: 10 seconds of audio at 16kHz for faster processing
        chunk_size = 10 * sample_rate  # 160,000 samples
        
        # If audio is short enough, process directly
        if len(waveform) <= chunk_size:
            return self._extract_single_embedding(waveform, sample_rate)
        
        # For longer audio, process in chunks and average
        embeddings = []
        num_chunks = (len(waveform) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(waveform))
            chunk = waveform[start:end]
            
            # Skip very short chunks
            if len(chunk) < sample_rate:  # Less than 1 second
                continue
                
            emb = self._extract_single_embedding(chunk, sample_rate)
            embeddings.append(emb)
        
        # Average all chunk embeddings
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return self._extract_single_embedding(waveform[:chunk_size], sample_rate)
    
    @torch.no_grad()
    def _extract_single_embedding(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract embedding from a single chunk of audio."""
        # Process audio
        inputs = self.processor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        
        # Extract embeddings (mean pooling over time)
        outputs = self.model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1)
        
        return embedding.cpu().numpy().squeeze()


# ============================================================================
# Custom Voice Classifier (Trainable)
# ============================================================================

# ============================================================================
# Improved Building Blocks
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch norm and dropout for better gradient flow."""
    def __init__(self, in_dim, out_dim, dropout_rate=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        # Skip connection with projection if dimensions differ
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        return self.dropout(self.activation(self.block(x) + self.skip(x)))


class VoiceClassifier(nn.Module):
    """
    Improved PyTorch classifier for AI voice detection.
    
    Uses residual connections and GELU activation for better training.
    
    Input: 798 dimensions
        - 768: Wav2Vec2 embeddings (deep audio features)
        - 30: Acoustic features (pitch, MFCC, spectral, etc.)
        
    Output: Single logit (apply sigmoid for probability)
    """
    
    ACOUSTIC_FEATURES_DIM = 30
    WAV2VEC_EMBEDDING_DIM = 768
    INPUT_DIM = ACOUSTIC_FEATURES_DIM + WAV2VEC_EMBEDDING_DIM  # 798
    
    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        # Residual blocks for better gradient flow
        self.res1 = ResidualBlock(512, 512, dropout_rate)
        self.res2 = ResidualBlock(512, 256, dropout_rate)
        self.res3 = ResidualBlock(256, 128, dropout_rate)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1),
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.head(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid activation."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


# Legacy classifier for backward compatibility
class VoiceClassifierLegacy(nn.Module):
    """Legacy classifier (786 dims). Keep for loading old models."""
    
    def __init__(self, input_dim: int = 786, dropout_rate: float = 0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 1: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer 3: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer: 64 → 1
            nn.Linear(64, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.network(x)


# ============================================================================
# Main Voice Detection Model
# ============================================================================

class VoiceDetectionModel:
    """
    Main inference engine for AI voice detection.
    
    Pipeline:
        1. Extract acoustic features (from AudioProcessor)
        2. Extract Wav2Vec2 embeddings
        3. Concatenate features [786 dims]
        4. Normalize features (if scaler available)
        5. Pass through VoiceClassifier
        6. Apply threshold → AI_GENERATED or HUMAN
    """
    
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        wav2vec_model_name: str = "facebook/wav2vec2-base-960h"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_path = classifier_path
        
        # Initialize components
        self.embedder = Wav2VecEmbedder(model_name=wav2vec_model_name)
        self.classifier: Optional[VoiceClassifier] = None
        self.scaler = None  # Feature scaler for normalization
        
        self._loaded = False
        
    def load(self) -> None:
        """Load all model components."""
        if self._loaded:
            return
            
        logger.info("Loading VoiceDetectionModel...")
        
        # Load Wav2Vec2 embedder
        self.embedder.load()
        
        # Load feature scaler if available
        scaler_path = Path("model/scaler.pkl")
        if scaler_path.exists():
            try:
                import joblib
                self.scaler = joblib.load(scaler_path)
                logger.info("  ✓ Feature scaler loaded")
            except Exception as e:
                logger.warning(f"  ⚠ Could not load scaler: {e}")
                self.scaler = None
        
        # Initialize classifier
        self.classifier = VoiceClassifier()
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # Load trained weights if available
        weights_loaded = False
        
        if self.classifier_path:
            weights_path = Path(self.classifier_path)
            if weights_path.exists():
                logger.info(f"Loading classifier weights: {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.classifier.load_state_dict(state_dict)
                logger.info("  ✓ Trained weights loaded")
                weights_loaded = True
            else:
                logger.warning(f"  ⚠ Weights not found: {weights_path}")
        
        # Try default path if no weights loaded yet
        if not weights_loaded:
            default_path = Path("model/classifier.pth")
            if default_path.exists():
                logger.info(f"Loading classifier weights: {default_path}")
                state_dict = torch.load(default_path, map_location=self.device, weights_only=True)
                self.classifier.load_state_dict(state_dict)
                logger.info("  ✓ Trained weights loaded")
                weights_loaded = True
        
        if not weights_loaded:
            logger.warning("  ⚠ No trained weights found - using random initialization")
            logger.warning("  ⚠ Run train.py to train the classifier for accurate predictions")
        
        self._loaded = True
        logger.info("✓ VoiceDetectionModel ready")
        
    def _prepare_acoustic_features(self, acoustic_features: AudioFeatures) -> np.ndarray:
        """
        Convert AudioFeatures dataclass to numpy array.
        
        Returns: 30-dimensional feature vector
        """
        # Use the new to_vector method which returns all 30 features
        return acoustic_features.to_vector().astype(np.float32)
    
    @torch.no_grad()
    def predict(
        self, 
        waveform: np.ndarray, 
        sample_rate: int,
        acoustic_features: AudioFeatures
    ) -> PredictionResult:
        """
        Run inference on audio.
        
        Args:
            waveform: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            acoustic_features: AudioFeatures from AudioProcessor
            
        Returns:
            PredictionResult with classification and confidence
        """
        if not self._loaded:
            self.load()
            
        try:
            # Step 1: Prepare acoustic features (30 dims)
            acoustic_vector = self._prepare_acoustic_features(acoustic_features)
            
            # Step 2: Extract Wav2Vec2 embeddings (768 dims)
            wav2vec_embedding = self.embedder.extract_embedding(waveform, sample_rate)
            
            # Step 3: Concatenate features (798 dims total)
            # Order must match training: acoustic first, then wav2vec
            combined_features = np.concatenate([
                acoustic_vector,     # 30 dims
                wav2vec_embedding    # 768 dims
            ])
            
            # Step 4: Apply feature normalization if scaler available
            if self.scaler is not None:
                combined_features = self.scaler.transform(combined_features.reshape(1, -1)).flatten()
            
            # Step 5: Convert to tensor and run through classifier
            features_tensor = torch.tensor(
                combined_features, 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Get prediction from VoiceClassifier (the ONLY classifier)
            logits = self.classifier(features_tensor)
            probability = torch.sigmoid(logits).item()
            
            # Step 6: Apply threshold and calculate confidence
            is_ai = probability > 0.5
            classification = "AI_GENERATED" if is_ai else "HUMAN"
            
            # Confidence is the raw probability for the predicted class
            # If predicting AI (prob > 0.5), confidence = probability
            # If predicting HUMAN (prob <= 0.5), confidence = 1 - probability
            if is_ai:
                confidence = probability
            else:
                confidence = 1.0 - probability
            
            # Clamp to reasonable range
            confidence = max(0.50, min(0.99, confidence))
            
            return PredictionResult(
                classification=classification,
                confidence_score=round(confidence, 4),
                raw_logit=round(probability, 4)
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Model inference failed: {e}")


# ============================================================================
# Singleton Pattern
# ============================================================================

_model: Optional[VoiceDetectionModel] = None


def get_model(
    classifier_path: Optional[str] = None,
    wav2vec_model_name: str = "facebook/wav2vec2-base-960h"
) -> VoiceDetectionModel:
    """Get the global model instance (singleton)."""
    global _model
    if _model is None:
        _model = VoiceDetectionModel(
            classifier_path=classifier_path,
            wav2vec_model_name=wav2vec_model_name
        )
    return _model


def reset_model() -> None:
    """Reset the global model instance (useful for testing)."""
    global _model
    _model = None


# ============================================================================
# Utility Functions
# ============================================================================

def create_dummy_weights(output_path: str = "model/classifier.pth") -> str:
    """
    Create placeholder weights for testing.
    
    These weights are randomly initialized and won't produce
    accurate results. Train the model for real predictions.
    """
    classifier = VoiceClassifier()
    
    # Xavier initialization for better starting point
    for module in classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), output_path)
    logger.info(f"  ✓ Created placeholder weights: {output_path}")
    
    return output_path


# ============================================================================
# Standalone Helper Functions
# ============================================================================

def load_classifier(weights_path: str = "model/classifier.pth") -> VoiceClassifier:
    """
    Load the trained VoiceClassifier from disk.
    
    Args:
        weights_path: Path to the trained weights file
        
    Returns:
        VoiceClassifier model ready for inference
    """
    model = VoiceClassifier()
    
    path = Path(weights_path)
    if path.exists():
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded classifier weights from: {weights_path}")
    else:
        logger.warning(f"No weights found at {weights_path}, using random initialization")
    
    model.eval()
    return model


@torch.no_grad()
def predict(model: VoiceClassifier, features: np.ndarray) -> float:
    """
    Run inference on combined features.
    
    Args:
        model: Trained VoiceClassifier
        features: Combined feature vector (798 dims: 30 acoustic + 768 wav2vec)
        
    Returns:
        Probability that the audio is AI-generated (0.0 to 1.0)
    """
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    logits = model(features_tensor)
    probability = torch.sigmoid(logits).item()
    return probability
