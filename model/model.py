"""
Model Module - The Brain of VoxProof
=====================================

This module handles the heavy lifting of AI voice detection:

1. Wav2Vec2 Embeddings: We use Facebook's pre-trained wav2vec2 model to extract 
   deep representations of audio. This model learned from 960 hours of speech 
   and captures nuances that hand-crafted features might miss.

2. VoiceClassifier: A fully-connected neural network that takes both the wav2vec2
   embeddings AND our acoustic features (MFCCs, pitch, etc.) to make the final
   AI vs Human decision.

The key insight is that AI-generated voices often have subtle artifacts that 
humans can't hear but show up in the learned representations. Think of it like
how JPEG compression leaves invisible fingerprints in images.

Author: VoxProof Team
License: MIT
"""

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Suppress transformers load report warnings before import
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import transformers
transformers.logging.set_verbosity_error()

from audio.processing import AudioFeatures

# Set up module logger - helps with debugging in production
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Container for model prediction results.
    
    We keep track of multiple values because they're useful for different purposes:
    - classification: The final verdict (what we show users)
    - confidence_score: How sure we are (helps users decide whether to trust it)
    - raw_logit: The raw neural network output (useful for debugging/analysis)
    """
    classification: str      # "AI_GENERATED" or "HUMAN" - the final answer
    confidence_score: float  # 0.0 to 1.0 - how certain we are
    raw_logit: float        # Raw model output before sigmoid - for debugging
    
    
class VoiceClassifier(nn.Module):
    """
    Neural network classifier for AI vs Human voice detection.
    
    Why this architecture?
    ----------------------
    We use a fairly standard MLP (multi-layer perceptron) because:
    1. Our input is already a fixed-size feature vector (not raw audio)
    2. The wav2vec2 embedding already captures sequential patterns
    3. MLPs are fast and easy to deploy - important for real-time APIs
    
    Architecture Details:
    - Input: 786 dimensions (18 acoustic features + 768 wav2vec2 embedding)
    - 4 hidden layers with decreasing size (512 -> 256 -> 128 -> 64)
    - BatchNorm for faster training and better generalization
    - Dropout (30%) to prevent overfitting
    - Single output neuron with sigmoid for binary classification
    
    The "funnel" shape (wide to narrow) forces the network to learn
    increasingly abstract representations as we go deeper.
    """
    
    # Feature dimensions - these MUST match what we extract!
    ACOUSTIC_FEATURES_DIM = 18  # 13 MFCCs + pitch_mean + pitch_std + rolloff + zcr + duration
    WAV2VEC_EMBEDDING_DIM = 768  # wav2vec2-base always outputs 768-dim vectors
    INPUT_DIM = ACOUSTIC_FEATURES_DIM + WAV2VEC_EMBEDDING_DIM  # 786 total
    
    def __init__(self, dropout_rate: float = 0.3):
        """
        Initialize the classifier network.
        
        Args:
            dropout_rate: Probability of dropping neurons during training.
                         Higher = more regularization, but slower learning.
                         0.3 is a good default for medium-sized datasets.
        """
        super().__init__()
        
        # We use nn.Sequential for cleaner code - it's just a container
        # that passes data through each layer in order
        self.network = nn.Sequential(
            # Layer 1: Input -> 512 neurons
            # This first layer does most of the "heavy lifting"
            nn.Linear(self.INPUT_DIM, 512),
            nn.BatchNorm1d(512),  # Normalizes activations - helps training a lot!
            nn.ReLU(),            # Non-linearity - without this, the network would just be linear
            nn.Dropout(dropout_rate),  # Randomly zero out neurons - prevents overfitting
            
            # Layer 2: 512 -> 256 neurons
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3: 256 -> 128 neurons
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 4: 128 -> 64 neurons
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer: 64 -> 1 neuron
            # Single output for binary classification (AI vs Human)
            # No activation here - we apply sigmoid later for numerical stability
            nn.Linear(64, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Returns raw logits (not probabilities!) because:
        1. BCEWithLogitsLoss expects logits and is more numerically stable
        2. We can apply sigmoid later when we need probabilities
        
        Args:
            x: Input tensor of shape (batch_size, 786)
            
        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions (0 to 1).
        
        This is what you'd use during inference when you want
        actual probabilities, not raw logits.
        
        Args:
            x: Input tensor of shape (batch_size, 786)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
            Values > 0.5 indicate AI-generated, < 0.5 indicate human
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


class Wav2VecEmbedder:
    """
    Extracts deep audio embeddings using Facebook's wav2vec2 model.
    
    Why wav2vec2?
    -------------
    wav2vec2 is a self-supervised model that learned to understand speech
    by predicting masked audio segments (similar to how BERT works for text).
    
    It was trained on 960 hours of unlabeled speech, so it has learned
    rich representations of how speech sounds. These representations
    capture things that hand-crafted features (like MFCCs) might miss.
    
    For AI voice detection, this is gold - the model picks up on subtle
    artifacts and patterns that distinguish synthetic from natural speech.
    
    We use the base model (not large) for faster inference. In production,
    you could try the large model if accuracy matters more than speed.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Set up the embedder (but don't load the model yet).
        
        We delay loading until first use because the model is ~360MB
        and takes a few seconds to load. No point doing that if the
        API never gets any requests.
        
        Args:
            model_name: HuggingFace model ID. Other options include:
                       - facebook/wav2vec2-large-960h (better but slower)
                       - facebook/wav2vec2-base (not fine-tuned on speech)
        """
        self.model_name = model_name
        # Use GPU if available - wav2vec2 is MUCH faster on GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[Wav2Vec2Model] = None
        self._loaded = False  # Track whether we've loaded the model yet
        
    def load(self):
        """
        Actually download and load the wav2vec2 model.
        
        This is called lazily on first use. The model will be cached
        by HuggingFace, so subsequent runs start much faster.
        
        First run: Downloads ~360MB from HuggingFace Hub
        Subsequent runs: Loads from ~/.cache/huggingface/
        """
        # Don't reload if already loaded
        if self._loaded:
            return
            
        logger.info(f"📥 Loading wav2vec2: {self.model_name}")
        if not os.path.exists(os.path.expanduser("~/.cache/huggingface")):
            logger.info("  ⏳ First run - downloading model (~360MB, may take 1-2 minutes)...")
        
        try:
            # The processor handles audio preprocessing (normalization, etc.)
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            
            # Load the actual model
            # We use Wav2Vec2Model (not Wav2Vec2ForCTC) since we only need embeddings
            # lm_head/masked_spec_embed warnings are expected - we're not doing ASR
            self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            
            # Move to GPU if available
            self.model.to(self.device)  # type: ignore
            
            # Set to evaluation mode - disables dropout, uses running stats for batchnorm
            # IMPORTANT: Forgetting this is a common bug that causes inconsistent results!
            self.model.eval()
            
            self._loaded = True
            device_name = "GPU" if self.device.type == "cuda" else "CPU"
            logger.info(f"  ✓ wav2vec2 loaded on {device_name}")
            
        except Exception as e:
            logger.error(f"Failed to load wav2vec2 model: {e}")
            raise RuntimeError(f"Could not load wav2vec2 model: {e}")
    
    @torch.no_grad()  # Disable gradient computation - saves memory during inference
    def extract_embedding(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract a fixed-size embedding from variable-length audio.
        
        The key insight here is MEAN POOLING: wav2vec2 outputs a sequence
        of vectors (one per ~20ms of audio), but we need a single fixed-size
        vector to feed to our classifier. We simply average across time.
        
        This loses some temporal information, but it's simple and works well.
        More sophisticated approaches (attention pooling, etc.) could help.
        
        Args:
            waveform: Raw audio samples, should be float32 in [-1, 1] range
            sample_rate: Must be 16kHz for wav2vec2!
            
        Returns:
            768-dimensional numpy array (the "fingerprint" of this audio)
        """
        # Lazy loading - only load model when first needed
        if not self._loaded:
            self.load()
        
        try:
            # Preprocess: normalize and convert to the format wav2vec2 expects
            inputs = self.processor(
                waveform, 
                sampling_rate=sample_rate, # type: ignore
                return_tensors="pt",  # type: ignore[call-arg]
                padding=True # type: ignore
            )
            
            # Move input to same device as model (CPU or GPU)
            input_values = inputs.input_values.to(self.device)
            
            # Run through wav2vec2 - this is where the magic happens!
            # The model outputs hidden states for each time step
            outputs = self.model(input_values)  # type: ignore
            hidden_states = outputs.last_hidden_state  # Shape: (1, num_frames, 768)
            
            # Mean pooling: average across the time dimension
            # This gives us a single 768-dim vector regardless of audio length
            embedding = hidden_states.mean(dim=1)  # Shape: (1, 768)
            
            # Convert back to numpy for compatibility with rest of pipeline
            embedding_np = embedding.cpu().numpy().squeeze()
            return embedding_np
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise RuntimeError(f"Failed to extract wav2vec2 embedding: {e}")


class VoiceDetectionModel:
    """
    The main inference engine - ties everything together.
    
    This is the class that app.py uses. It:
    1. Manages the wav2vec2 embedder (for deep features)
    2. Manages the classifier (for the final decision)
    3. Handles the prediction pipeline
    
    We use a singleton pattern (see get_model()) to ensure we only
    load these heavy models once, even if multiple requests come in.
    """
    
    def __init__(
        self, 
        classifier_path: Optional[str] = None,
        wav2vec_model_name: str = "facebook/wav2vec2-base-960h"
    ):
        """
        Initialize the voice detection model.
        
        Note: This doesn't actually load the models yet! That happens
        in load() to support lazy loading.
        
        Args:
            classifier_path: Path to trained classifier weights (.pth file)
                           If None or file doesn't exist, uses random weights
            wav2vec_model_name: Which wav2vec2 model to use from HuggingFace
        """
        self.classifier_path = classifier_path
        self.wav2vec_model_name = wav2vec_model_name
        
        # Determine device - prefer GPU for speed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # These get initialized in load()
        self.embedder: Optional[Wav2VecEmbedder] = None
        self.classifier: Optional[VoiceClassifier] = None
        self._loaded = False
        
    def load(self):
        """
        Load all model components into memory.
        
        This is where the heavy initialization happens:
        1. Download/load wav2vec2 (~360MB)
        2. Initialize classifier architecture
        3. Load trained weights (if available)
        
        Called automatically on first prediction, or explicitly at startup.
        """
        if self._loaded:
            return
            
        # Step 1: Load the wav2vec2 embedder
        self.embedder = Wav2VecEmbedder(model_name=self.wav2vec_model_name)
        self.embedder.load()  # This downloads the model if needed
        
        # Step 2: Create the classifier network
        self.classifier = VoiceClassifier()
        device_name = "GPU" if self.device.type == "cuda" else "CPU"
        self.classifier.to(self.device)
        logger.info(f"  ✓ Classifier initialized on {device_name}")
        
        # Step 3: Load trained weights if we have them
        if self.classifier_path and os.path.exists(self.classifier_path):
            try:
                # map_location handles the case where model was saved on GPU but we're on CPU
                state_dict = torch.load(self.classifier_path, map_location=self.device)
                self.classifier.load_state_dict(state_dict)
                logger.info(f"  ✓ Loaded trained weights: {os.path.basename(self.classifier_path)}")
            except Exception as e:
                # Don't crash - fall back to random weights (won't be accurate though)
                logger.warning(f"  ⚠️  Could not load classifier weights: {e}")
                logger.warning("  ⚠️  Using random initialization - predictions will be inaccurate!")
        else:
            # No weights found - this is expected on first run before training
            logger.warning(f"  ⚠️  No classifier weights found at '{self.classifier_path}'")
            logger.warning("  ⚠️  Run 'python train.py' to train the model!")
        
        # IMPORTANT: Set to eval mode for inference
        # This disables dropout and uses running stats for batch normalization
        self.classifier.eval()
        self._loaded = True
    
    @torch.no_grad()  # No gradients needed for inference - saves memory
    def predict(
        self, 
        waveform: np.ndarray, 
        acoustic_features: AudioFeatures,
        sample_rate: int = 16000
    ) -> PredictionResult:
        """
        Run the full inference pipeline on an audio sample.
        
        This is the main entry point for predictions. It:
        1. Extracts wav2vec2 embedding (deep learned features)
        2. Combines with acoustic features (hand-crafted features)
        3. Runs through the classifier
        4. Returns a human-readable result
        
        The combination of learned + hand-crafted features is powerful:
        - wav2vec2 catches subtle patterns humans might miss
        - Acoustic features (pitch variance, etc.) are interpretable
        
        Args:
            waveform: Preprocessed audio (16kHz, mono, normalized)
            acoustic_features: MFCCs, pitch, etc. from AudioProcessor
            sample_rate: Should be 16000 for wav2vec2
            
        Returns:
            PredictionResult with classification, confidence, and raw score
        """
        # Lazy loading - load models on first use
        if not self._loaded:
            self.load()
        
        try:
            # STEP 1: Get the deep embedding from wav2vec2
            # This captures learned representations of speech patterns
            wav2vec_embedding = self.embedder.extract_embedding(waveform, sample_rate)  # type: ignore
            
            # STEP 2: Get the hand-crafted acoustic features
            # These are more interpretable (pitch, MFCCs, etc.)
            acoustic_vector = acoustic_features.to_vector()
            
            # STEP 3: Concatenate both feature types
            # The classifier will learn how to weight each
            combined_features = np.concatenate([acoustic_vector, wav2vec_embedding])
            
            # STEP 4: Convert to PyTorch tensor and add batch dimension
            input_tensor = torch.tensor(
                combined_features, 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)  # Shape: (1, 786)
            
            # STEP 5: Run through the classifier
            logits = self.classifier(input_tensor)  # type: ignore
            probability = torch.sigmoid(logits).item()  # Convert to 0-1 probability
            
            # STEP 6: Make the final decision
            # probability > 0.5 means the model thinks it's AI-generated
            classification = "AI_GENERATED" if probability > 0.5 else "HUMAN"
            
            # STEP 7: Calculate confidence score
            # We want high confidence near 0 or 1, low confidence near 0.5
            # This maps [0, 0.5, 1] -> [1, 0, 1] (distance from decision boundary)
            confidence = abs(probability - 0.5) * 2
            
            result = PredictionResult(
                classification=classification,
                confidence_score=round(confidence, 4),
                raw_logit=logits.item()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Model inference failed: {e}")


# ============================================================================
# Singleton Pattern for Model Management
# ============================================================================
#
# We use a module-level singleton to ensure the model is only loaded once.
# Loading wav2vec2 takes several seconds and uses ~1GB of RAM, so we don't
# want to do it on every request!
# ============================================================================

_model: Optional[VoiceDetectionModel] = None


def get_model(
    classifier_path: Optional[str] = None,
    wav2vec_model_name: str = "facebook/wav2vec2-base-960h"
) -> VoiceDetectionModel:
    """
    Get the global model instance (creates it if needed).
    
    This is the recommended way to get a model instance. It ensures:
    1. The model is only loaded once (singleton pattern)
    2. All parts of the app use the same instance
    3. Memory is used efficiently
    
    Usage:
        model = get_model(classifier_path="model/classifier.pth")
        result = model.predict(waveform, features)
    
    Args:
        classifier_path: Path to the trained .pth weights file
        wav2vec_model_name: HuggingFace model ID for wav2vec2
        
    Returns:
        The global VoiceDetectionModel instance
    """
    global _model
    if _model is None:
        _model = VoiceDetectionModel(
            classifier_path=classifier_path,
            wav2vec_model_name=wav2vec_model_name
        )
    return _model


def create_dummy_weights(output_path: str = "model/classifier.pth"):
    """
    Create placeholder classifier weights for testing the pipeline.
    
    ⚠️  WARNING: These are RANDOM weights! The model will make random
    predictions until you train it properly with real data.
    
    This function exists so you can test the API without training first.
    Run 'python train.py' to generate real trained weights.
    
    Args:
        output_path: Where to save the .pth file
        
    Returns:
        The path where weights were saved
    """
    classifier = VoiceClassifier()
    
    # Xavier initialization gives better starting weights than pure random
    # It helps gradients flow properly during training
    for module in classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    # Make sure the directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the state dict (just the weights, not the whole model)
    torch.save(classifier.state_dict(), output_path)
    logger.info(f"  ✓ Created placeholder weights: {output_path}")
    
    return output_path
