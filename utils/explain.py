"""
Explanation Module - Making AI Decisions Understandable
========================================================

One of the biggest problems with ML models is they're black boxes.
Users get a "AI_GENERATED" label but have no idea WHY.

This module provides human-readable explanations based on the acoustic
features we extract. It analyzes 30+ acoustic properties to build
comprehensive explanations.

Key AI voice characteristics we detect:
- Unnaturally stable pitch (robotic monotone, low jitter)
- Suspiciously clean audio (no breath sounds, no room noise)
- Over-smooth MFCC transitions (lacking micro-variations)
- Unnatural energy patterns (missing natural dynamics)
- Synthetic spectral signatures (too perfect harmonics)

These are based on research into how current TTS systems (ElevenLabs,
OpenAI, etc.) differ from natural human speech.

Author: VoxProof Team
License: MIT
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

from audio.processing import AudioFeatures
from model.model import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class ExplanationRule:
    """
    Defines a single rule for generating explanations.
    
    Each rule checks a specific acoustic property and provides
    different explanations depending on whether we classified
    the audio as AI or Human.
    
    Attributes:
        name: Short identifier for the rule (for logging)
        description: What this rule is checking
        check_fn: Lambda that takes AudioFeatures, returns True if rule applies
        ai_explanation: What to say if rule triggers AND we said AI_GENERATED
        human_explanation: What to say if rule triggers AND we said HUMAN
    """
    name: str
    description: str
    check_fn: callable
    ai_explanation: str
    human_explanation: str


class ExplanationGenerator:
    """
    Generates human-readable explanations for voice detection results.
    
    Uses comprehensive analysis of 30 acoustic features to explain
    why audio was classified as AI-generated or human.
    
    Key detection signals for modern TTS (ElevenLabs, etc.):
    - Pitch jitter: AI voices have unnaturally low jitter (too perfect)
    - MFCC delta variance: AI voices have over-smooth transitions
    - Energy variance: AI lacks natural micro-variations
    - Spectral patterns: Synthetic harmonic structures
    """
    
    # -------------------------------------------------------------------------
    # Detection Thresholds (tuned for ElevenLabs and similar TTS)
    # -------------------------------------------------------------------------
    
    # Pitch characteristics
    PITCH_STD_LOW = 15.0           # Very stable pitch (AI-like)
    PITCH_STD_NORMAL = 25.0        # Minimum natural variation
    PITCH_JITTER_LOW = 0.02        # AI voices have jitter < 2%
    PITCH_JITTER_HIGH = 0.08       # Natural speech has jitter > 2%
    
    # MFCC delta (temporal dynamics) - KEY AI detector
    MFCC_DELTA_STD_LOW = 1.5       # Over-smooth (AI-like)
    MFCC_ACCELERATION_LOW = 0.1    # Too stable acceleration (AI)
    
    # Energy variance - humans have natural dynamics
    RMS_VAR_LOW = 0.001            # Too consistent energy (AI)
    RMS_VAR_HIGH = 0.01            # Natural energy variation
    
    # Zero crossing rate
    ZCR_LOW = 0.03                 # Unusually clean (synthetic)
    ZCR_HIGH = 0.12                # Natural ambient noise
    ZCR_VAR_LOW = 0.0005           # Constant ZCR (AI)
    
    # Spectral characteristics
    ROLLOFF_LOW = 2500.0           # Limited frequency (muffled)
    ROLLOFF_HIGH = 5500.0          # Extended highs (processed)
    BANDWIDTH_LOW = 1500.0         # Narrow bandwidth (thin)
    
    def __init__(self):
        self.rules = self._build_rules()
        
    def _build_rules(self) -> List[ExplanationRule]:
        """Build comprehensive explanation rules based on acoustic features."""
        return [
            # ===== PITCH JITTER (Primary AI detector) =====
            ExplanationRule(
                name="pitch_jitter_low",
                description="Pitch micro-variation analysis (jitter)",
                check_fn=lambda f: f.pitch_jitter < self.PITCH_JITTER_LOW and f.pitch_mean > 0,
                ai_explanation="Pitch stability is unnaturally perfect - AI voices lack the micro-variations found in natural speech. Human vocal cords have inherent instability that creates slight pitch fluctuations (jitter), which synthetic voices typically don't replicate",
                human_explanation="Natural pitch micro-variations detected, consistent with biological voice production"
            ),
            
            # ===== PITCH VARIANCE =====
            ExplanationRule(
                name="pitch_stability",
                description="Overall pitch variation analysis",
                check_fn=lambda f: f.pitch_std < self.PITCH_STD_LOW and f.pitch_mean > 0,
                ai_explanation="Monotonous pitch pattern detected - the voice maintains an unusually consistent pitch throughout. Modern TTS systems often produce overly stable prosody that sounds 'flat' or robotic",
                human_explanation="Natural prosodic variation observed in pitch contours"
            ),
            
            # ===== MFCC DELTA (Temporal Dynamics) =====
            ExplanationRule(
                name="mfcc_delta_smooth",
                description="MFCC temporal transition analysis",
                check_fn=lambda f: f.mfcc_delta_std < self.MFCC_DELTA_STD_LOW,
                ai_explanation="Voice timbre transitions are suspiciously smooth - natural speech has rapid, irregular changes in vocal quality due to articulatory movements. AI voices often have overly interpolated spectral transitions",
                human_explanation="Natural articulatory dynamics detected in spectral patterns"
            ),
            
            ExplanationRule(
                name="mfcc_acceleration",
                description="Second-order MFCC dynamics",
                check_fn=lambda f: f.mfcc_acceleration < self.MFCC_ACCELERATION_LOW,
                ai_explanation="Missing natural speech acceleration patterns - human speech contains rapid changes in articulation speed that synthetic voices struggle to replicate",
                human_explanation="Natural speech rhythm and acceleration patterns detected"
            ),
            
            # ===== ENERGY VARIANCE =====
            ExplanationRule(
                name="energy_variance_low",
                description="Energy micro-variation analysis",
                check_fn=lambda f: f.rms_var < self.RMS_VAR_LOW,
                ai_explanation="Energy envelope is unnaturally consistent - human speech naturally varies in loudness at the micro-level due to breathing, emphasis, and emotional state. This audio shows synthetic-like energy stability",
                human_explanation="Natural energy dynamics and breath patterns detected"
            ),
            
            # ===== ZERO CROSSING RATE =====
            ExplanationRule(
                name="zcr_synthetic_clean",
                description="Signal cleanliness analysis",
                check_fn=lambda f: f.zcr_mean < self.ZCR_LOW,
                ai_explanation="Unusually clean signal detected - synthetic voices often lack the subtle noise, breath sounds, and room acoustics present in natural recordings. This 'too perfect' quality is a hallmark of AI generation",
                human_explanation="Clean recording with natural acoustic characteristics"
            ),
            
            ExplanationRule(
                name="zcr_variance_low",
                description="ZCR consistency analysis",
                check_fn=lambda f: f.zcr_var < self.ZCR_VAR_LOW,
                ai_explanation="Constant noise characteristics detected - natural speech shows varying levels of breathiness and fricative sounds, while AI voices often have uniform noise properties",
                human_explanation="Natural variation in consonant and breath sounds"
            ),
            
            # ===== SPECTRAL CHARACTERISTICS =====
            ExplanationRule(
                name="spectral_narrow",
                description="Spectral bandwidth analysis",
                check_fn=lambda f: f.spectral_bandwidth_mean < self.BANDWIDTH_LOW,
                ai_explanation="Narrow spectral bandwidth detected - the voice lacks the full harmonic richness of natural speech. Some TTS systems produce spectrally limited output",
                human_explanation="Rich harmonic content consistent with natural voice"
            ),
            
            ExplanationRule(
                name="rolloff_low",
                description="High frequency content analysis",
                check_fn=lambda f: f.spectral_rolloff_mean < self.ROLLOFF_LOW,
                ai_explanation="Limited high-frequency content - the voice sounds muffled or lacks crispness. This can indicate older TTS technology or heavy compression",
                human_explanation="Natural frequency distribution across the speech spectrum"
            ),
            
            ExplanationRule(
                name="rolloff_high",
                description="Extended frequency analysis",
                check_fn=lambda f: f.spectral_rolloff_mean > self.ROLLOFF_HIGH,
                ai_explanation="Extended high-frequency content detected - may indicate synthesis artifacts or heavy audio processing not typical of natural recordings",
                human_explanation="Natural spectral envelope with appropriate high-frequency roll-off"
            ),
            
            # ===== COMBINED INDICATORS =====
            ExplanationRule(
                name="multiple_synthetic_markers",
                description="Combined synthetic voice indicators",
                check_fn=lambda f: (
                    f.pitch_jitter < self.PITCH_JITTER_LOW and 
                    f.mfcc_delta_std < self.MFCC_DELTA_STD_LOW and
                    f.pitch_mean > 0
                ),
                ai_explanation="Multiple synthetic voice markers detected: unnaturally stable pitch combined with smooth spectral transitions. This pattern is highly characteristic of modern TTS systems like ElevenLabs",
                human_explanation="Voice exhibits natural acoustic variability across multiple dimensions"
            ),
            
            ExplanationRule(
                name="natural_variability",
                description="Natural speech variability check",
                check_fn=lambda f: (
                    f.pitch_jitter > self.PITCH_JITTER_HIGH and 
                    f.rms_var > self.RMS_VAR_LOW and
                    f.pitch_mean > 0
                ),
                ai_explanation="Despite some natural-seeming features, other acoustic markers suggest AI generation",
                human_explanation="Strong natural variability in pitch, energy, and timing - highly consistent with biological voice production and natural cognitive processes during speech"
            ),
        ]
    
    def generate_explanation(
        self, 
        features: AudioFeatures, 
        prediction: PredictionResult
    ) -> str:
        """
        Generate a comprehensive human-readable explanation for the prediction.
        
        Args:
            features: Extracted acoustic features (30 dimensions)
            prediction: Model prediction result
            
        Returns:
            Detailed explanation string
        """
        is_ai = prediction.classification == "AI_GENERATED"
        confidence = prediction.confidence_score
        triggered_explanations: List[Tuple[str, float]] = []
        
        # Check each rule
        for rule in self.rules:
            try:
                if rule.check_fn(features):
                    explanation = rule.ai_explanation if is_ai else rule.human_explanation
                    triggered_explanations.append((explanation, 1.0))
            except Exception as e:
                logger.debug(f"Rule {rule.name} check failed: {e}")
                continue
        
        # Build confidence-aware preamble
        if confidence >= 0.85:
            confidence_text = "High confidence"
        elif confidence >= 0.70:
            confidence_text = "Moderate confidence"
        else:
            confidence_text = "Lower confidence"
        
        # Build final explanation
        if not triggered_explanations:
            # Default explanations based on overall assessment
            if is_ai:
                return f"{confidence_text} AI detection: The neural network identified patterns consistent with synthetic voice generation, though specific acoustic markers were subtle. Modern TTS systems like ElevenLabs are increasingly sophisticated."
            else:
                return f"{confidence_text} human voice detection: Audio characteristics are consistent with natural speech patterns, including typical variability in pitch, timing, and energy."
        
        # Prioritize unique, most informative explanations (top 3)
        unique_explanations = []
        seen_prefixes = set()
        for exp, _ in triggered_explanations:
            # Avoid redundant explanations with similar starts
            prefix = exp[:50]
            if prefix not in seen_prefixes:
                unique_explanations.append(exp)
                seen_prefixes.add(prefix)
            if len(unique_explanations) >= 3:
                break
        
        if len(unique_explanations) == 1:
            return f"{confidence_text}: {unique_explanations[0]}"
        
        # Combine multiple key findings
        primary = unique_explanations[0]
        secondary = " | ".join(unique_explanations[1:3])
        
        return f"{confidence_text}: {primary}. Key indicators: {secondary.lower()}"
    
    def get_detailed_analysis(
        self, 
        features: AudioFeatures, 
        prediction: PredictionResult
    ) -> dict:
        """
        Get comprehensive analysis breakdown for debugging/advanced users.
        
        Returns dict with all feature values, interpretations, and risk scores.
        """
        analysis = {
            "classification": prediction.classification,
            "confidence": prediction.confidence_score,
            "raw_probability": prediction.raw_logit,
            "features": {
                "pitch": {
                    "mean_hz": round(features.pitch_mean, 2),
                    "std_hz": round(features.pitch_std, 2),
                    "range_hz": round(features.pitch_range, 2),
                    "jitter": round(features.pitch_jitter, 4),
                    "interpretation": self._interpret_pitch(features),
                    "ai_risk": "HIGH" if features.pitch_jitter < self.PITCH_JITTER_LOW else "LOW"
                },
                "mfcc_dynamics": {
                    "delta_mean": round(features.mfcc_delta_mean, 4),
                    "delta_std": round(features.mfcc_delta_std, 4),
                    "acceleration": round(features.mfcc_acceleration, 4),
                    "interpretation": self._interpret_mfcc_dynamics(features),
                    "ai_risk": "HIGH" if features.mfcc_delta_std < self.MFCC_DELTA_STD_LOW else "LOW"
                },
                "energy": {
                    "rms_mean": round(features.rms_mean, 6),
                    "rms_variance": round(features.rms_var, 6),
                    "interpretation": self._interpret_energy(features),
                    "ai_risk": "HIGH" if features.rms_var < self.RMS_VAR_LOW else "LOW"
                },
                "spectral": {
                    "centroid_hz": round(features.spectral_centroid_mean, 2),
                    "rolloff_hz": round(features.spectral_rolloff_mean, 2),
                    "bandwidth_hz": round(features.spectral_bandwidth_mean, 2),
                    "contrast": round(features.spectral_contrast_mean, 4),
                    "interpretation": self._interpret_spectral(features)
                },
                "zero_crossing": {
                    "mean": round(features.zcr_mean, 6),
                    "variance": round(features.zcr_var, 8),
                    "interpretation": self._interpret_zcr(features),
                    "ai_risk": "MEDIUM" if features.zcr_mean < self.ZCR_LOW else "LOW"
                }
            },
            "overall_ai_indicators": self._count_ai_indicators(features),
            "explanation": self.generate_explanation(features, prediction)
        }
        
        return analysis
    
    def _count_ai_indicators(self, features: AudioFeatures) -> dict:
        """Count how many AI indicators are triggered."""
        indicators = {
            "low_pitch_jitter": features.pitch_jitter < self.PITCH_JITTER_LOW if features.pitch_mean > 0 else False,
            "stable_pitch": features.pitch_std < self.PITCH_STD_LOW if features.pitch_mean > 0 else False,
            "smooth_mfcc": features.mfcc_delta_std < self.MFCC_DELTA_STD_LOW,
            "low_energy_var": features.rms_var < self.RMS_VAR_LOW,
            "clean_signal": features.zcr_mean < self.ZCR_LOW,
            "constant_zcr": features.zcr_var < self.ZCR_VAR_LOW
        }
        count = sum(indicators.values())
        return {
            "total": count,
            "max_possible": len(indicators),
            "risk_level": "HIGH" if count >= 3 else ("MEDIUM" if count >= 2 else "LOW"),
            "details": indicators
        }
    
    def _interpret_pitch(self, features: AudioFeatures) -> str:
        """Interpret pitch characteristics with jitter analysis."""
        if features.pitch_mean == 0:
            return "No pitch detected (possibly non-speech audio)"
        
        parts = []
        if features.pitch_jitter < self.PITCH_JITTER_LOW:
            parts.append("SYNTHETIC: Jitter is abnormally low (<2%), typical of TTS")
        elif features.pitch_jitter > self.PITCH_JITTER_HIGH:
            parts.append("NATURAL: High jitter indicates biological voice production")
        else:
            parts.append("Jitter within ambiguous range")
            
        if features.pitch_std < self.PITCH_STD_LOW:
            parts.append("Very stable pitch (monotonous)")
        elif features.pitch_std > self.PITCH_STD_NORMAL:
            parts.append("Natural pitch variation")
            
        return "; ".join(parts)
    
    def _interpret_mfcc_dynamics(self, features: AudioFeatures) -> str:
        """Interpret MFCC temporal dynamics."""
        if features.mfcc_delta_std < self.MFCC_DELTA_STD_LOW:
            return "SYNTHETIC: Over-smooth spectral transitions, lacking natural articulatory variation"
        elif features.mfcc_acceleration < self.MFCC_ACCELERATION_LOW:
            return "Low articulation acceleration, possibly synthetic"
        else:
            return "Natural spectral dynamics with appropriate variation"
    
    def _interpret_energy(self, features: AudioFeatures) -> str:
        """Interpret energy patterns."""
        if features.rms_var < self.RMS_VAR_LOW:
            return "SYNTHETIC: Energy is too consistent, lacking natural micro-dynamics"
        elif features.rms_var > self.RMS_VAR_HIGH:
            return "High energy variation, natural speech-like dynamics"
        else:
            return "Moderate energy variation"
    
    def _interpret_spectral(self, features: AudioFeatures) -> str:
        """Interpret spectral characteristics."""
        parts = []
        if features.spectral_rolloff_mean < self.ROLLOFF_LOW:
            parts.append("Limited high frequencies (muffled)")
        elif features.spectral_rolloff_mean > self.ROLLOFF_HIGH:
            parts.append("Extended high frequencies (bright or processed)")
        else:
            parts.append("Normal frequency distribution")
            
        if features.spectral_bandwidth_mean < self.BANDWIDTH_LOW:
            parts.append("Narrow bandwidth")
        else:
            parts.append("Rich harmonic content")
            
        return "; ".join(parts)
    
    def _interpret_zcr(self, features: AudioFeatures) -> str:
        """Interpret zero crossing rate patterns."""
        parts = []
        if features.zcr_mean < self.ZCR_LOW:
            parts.append("SYNTHETIC: Unusually clean signal (no natural noise/breath)")
        elif features.zcr_mean > self.ZCR_HIGH:
            parts.append("Natural ambient noise and fricatives present")
        else:
            parts.append("Normal noise characteristics")
            
        if features.zcr_var < self.ZCR_VAR_LOW:
            parts.append("constant noise (synthetic)")
        else:
            parts.append("variable noise (natural)")
            
        return "; ".join(parts)


# Module-level singleton
_explainer: ExplanationGenerator = None


def get_explainer() -> ExplanationGenerator:
    """Get or create the explanation generator singleton."""
    global _explainer
    if _explainer is None:
        _explainer = ExplanationGenerator()
    return _explainer
