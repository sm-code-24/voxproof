"""
Explanation Module - Making AI Decisions Understandable
========================================================

Generates clear, professional explanations for voice authentication results.
Uses acoustic analysis to explain WHY audio was classified as AI or human.

Key features:
- Confidence-based explanation depth
- Clear, non-technical language for users
- Technical details available for advanced analysis

Author: VoxProof Team
License: MIT
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

from audio.processing import AudioFeatures
from model.model import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class SignalIndicator:
    """An acoustic indicator with its assessment."""
    name: str
    value: float
    assessment: str  # "ai", "human", or "neutral"
    weight: float  # How important this indicator is (0-1)
    description: str  # Human-readable finding


class ExplanationGenerator:
    """
    Generates professional explanations for voice detection results.
    
    Optimized for clarity and actionability - explains what was detected
    and why it matters, without overwhelming technical jargon.
    """
    
    # Detection thresholds (calibrated for modern TTS like ElevenLabs)
    THRESHOLDS = {
        # Pitch (key AI detector)
        "pitch_jitter_ai": 0.02,       # AI voices have < 2% jitter
        "pitch_jitter_human": 0.05,    # Human voices have > 5% jitter
        "pitch_std_ai": 15.0,          # AI pitch is too stable
        "pitch_std_human": 25.0,       # Humans have more variation
        
        # MFCC dynamics (catches smooth AI transitions)
        "mfcc_delta_std_ai": 1.5,      # AI is too smooth
        "mfcc_accel_ai": 0.1,          # AI lacks acceleration
        
        # Energy (natural breathing/emphasis patterns)
        "rms_var_ai": 0.001,           # AI is too consistent
        "rms_var_human": 0.005,        # Humans vary more
        
        # Signal cleanliness
        "zcr_ai": 0.03,                # AI is too clean
        "zcr_var_ai": 0.0005,          # AI has constant noise floor
        
        # Spectral
        "bandwidth_narrow": 1500.0,    # Thin voice
        "rolloff_low": 2500.0,         # Muffled
        "rolloff_high": 5500.0,        # Over-processed
    }
    
    def __init__(self):
        pass
    
    def _analyze_features(self, features: AudioFeatures) -> List[SignalIndicator]:
        """Analyze all acoustic features and return indicators."""
        indicators = []
        T = self.THRESHOLDS
        
        # 1. PITCH JITTER (strongest AI detector)
        if features.pitch_mean > 0:
            if features.pitch_jitter < T["pitch_jitter_ai"]:
                indicators.append(SignalIndicator(
                    name="pitch_jitter",
                    value=features.pitch_jitter,
                    assessment="ai",
                    weight=0.9,
                    description="Pitch is unnaturally stable - lacks the micro-variations present in human vocal cords"
                ))
            elif features.pitch_jitter > T["pitch_jitter_human"]:
                indicators.append(SignalIndicator(
                    name="pitch_jitter",
                    value=features.pitch_jitter,
                    assessment="human",
                    weight=0.9,
                    description="Natural pitch variation detected, consistent with biological voice production"
                ))
        
        # 2. PITCH STANDARD DEVIATION
        if features.pitch_mean > 0:
            if features.pitch_std < T["pitch_std_ai"]:
                indicators.append(SignalIndicator(
                    name="pitch_stability",
                    value=features.pitch_std,
                    assessment="ai",
                    weight=0.7,
                    description="Monotonous pitch pattern - voice maintains an unusually flat tone"
                ))
            elif features.pitch_std > T["pitch_std_human"]:
                indicators.append(SignalIndicator(
                    name="pitch_stability",
                    value=features.pitch_std,
                    assessment="human",
                    weight=0.6,
                    description="Natural prosodic variation in speech melody"
                ))
        
        # 3. MFCC DYNAMICS (spectral transitions)
        if features.mfcc_delta_std < T["mfcc_delta_std_ai"]:
            indicators.append(SignalIndicator(
                name="spectral_dynamics",
                value=features.mfcc_delta_std,
                assessment="ai",
                weight=0.8,
                description="Voice timbre transitions are too smooth - lacking natural articulatory variation"
            ))
        elif features.mfcc_delta_std > T["mfcc_delta_std_ai"] * 2:
            indicators.append(SignalIndicator(
                name="spectral_dynamics",
                value=features.mfcc_delta_std,
                assessment="human",
                weight=0.6,
                description="Natural articulation dynamics detected"
            ))
        
        # 4. MFCC ACCELERATION
        if features.mfcc_acceleration < T["mfcc_accel_ai"]:
            indicators.append(SignalIndicator(
                name="speech_rhythm",
                value=features.mfcc_acceleration,
                assessment="ai",
                weight=0.5,
                description="Speech rhythm lacks natural acceleration patterns"
            ))
        
        # 5. ENERGY VARIANCE
        if features.rms_var < T["rms_var_ai"]:
            indicators.append(SignalIndicator(
                name="energy_dynamics",
                value=features.rms_var,
                assessment="ai",
                weight=0.7,
                description="Volume is unnaturally consistent - missing natural breathing and emphasis"
            ))
        elif features.rms_var > T["rms_var_human"]:
            indicators.append(SignalIndicator(
                name="energy_dynamics",
                value=features.rms_var,
                assessment="human",
                weight=0.6,
                description="Natural volume dynamics with breathing patterns"
            ))
        
        # 6. ZERO CROSSING (signal cleanliness)
        if features.zcr_mean < T["zcr_ai"]:
            indicators.append(SignalIndicator(
                name="signal_quality",
                value=features.zcr_mean,
                assessment="ai",
                weight=0.5,
                description="Audio is suspiciously clean - lacks natural room acoustics and breath sounds"
            ))
        
        # 7. ZCR VARIANCE  
        if features.zcr_var < T["zcr_var_ai"]:
            indicators.append(SignalIndicator(
                name="noise_consistency",
                value=features.zcr_var,
                assessment="ai",
                weight=0.4,
                description="Background characteristics are too uniform"
            ))
        
        # 8. SPECTRAL BANDWIDTH
        if features.spectral_bandwidth_mean < T["bandwidth_narrow"]:
            indicators.append(SignalIndicator(
                name="harmonic_richness",
                value=features.spectral_bandwidth_mean,
                assessment="ai",
                weight=0.4,
                description="Voice lacks full harmonic richness"
            ))
        
        return indicators
    
    def _count_indicators(self, indicators: List[SignalIndicator]) -> Dict[str, float]:
        """Count weighted AI vs human indicators."""
        ai_score = sum(i.weight for i in indicators if i.assessment == "ai")
        human_score = sum(i.weight for i in indicators if i.assessment == "human")
        return {"ai": ai_score, "human": human_score}
    
    def generate_explanation(
        self, 
        features: AudioFeatures, 
        prediction: PredictionResult
    ) -> str:
        """
        Generate a clear, professional explanation for the prediction.
        
        Structure:
        1. Confidence statement
        2. Primary finding (most important indicator)
        3. Supporting evidence (1-2 additional indicators)
        """
        is_ai = prediction.classification == "AI_GENERATED"
        confidence = prediction.confidence_score
        
        # Analyze acoustic features
        indicators = self._analyze_features(features)
        
        # Filter by classification direction
        relevant = [i for i in indicators if i.assessment == ("ai" if is_ai else "human")]
        
        # Sort by weight (importance)
        relevant.sort(key=lambda x: x.weight, reverse=True)
        
        # Build confidence prefix
        if confidence >= 0.90:
            conf_text = "Very high confidence"
        elif confidence >= 0.80:
            conf_text = "High confidence"
        elif confidence >= 0.70:
            conf_text = "Moderate confidence"
        elif confidence >= 0.60:
            conf_text = "Low confidence"
        else:
            conf_text = "Uncertain"
        
        # Case: Strong indicators found
        if relevant:
            primary = relevant[0].description
            
            if len(relevant) >= 2:
                secondary = relevant[1].description.lower()
                if len(relevant) >= 3:
                    tertiary = relevant[2].description.lower()
                    return f"{conf_text}: {primary}. Additionally, {secondary}, and {tertiary}."
                return f"{conf_text}: {primary}. Additionally, {secondary}."
            return f"{conf_text}: {primary}."
        
        # Case: No clear indicators - use default
        if is_ai:
            ai_count = sum(1 for i in indicators if i.assessment == "ai")
            if ai_count > 0:
                return f"{conf_text}: Neural network detected subtle patterns consistent with AI voice synthesis across {ai_count} acoustic dimensions."
            return f"{conf_text}: The voice exhibits characteristics associated with synthetic generation, though specific markers are subtle. Modern TTS systems like ElevenLabs produce increasingly realistic output."
        else:
            human_count = sum(1 for i in indicators if i.assessment == "human")
            if human_count > 0:
                return f"{conf_text}: Voice characteristics match natural human speech patterns across {human_count} acoustic dimensions."
            return f"{conf_text}: Audio characteristics are consistent with natural human speech, including normal variation in pitch, timing, and vocal quality."
    
    def get_detailed_analysis(
        self, 
        features: AudioFeatures, 
        prediction: PredictionResult
    ) -> dict:
        """
        Get comprehensive analysis for debugging/advanced users.
        
        Returns a structured breakdown of all acoustic findings.
        """
        indicators = self._analyze_features(features)
        scores = self._count_indicators(indicators)
        
        # Categorize indicators
        ai_indicators = [i for i in indicators if i.assessment == "ai"]
        human_indicators = [i for i in indicators if i.assessment == "human"]
        
        return {
            "classification": prediction.classification,
            "confidence": round(prediction.confidence_score, 4),
            "explanation": self.generate_explanation(features, prediction),
            
            "acoustic_analysis": {
                "pitch": {
                    "mean_hz": round(features.pitch_mean, 1),
                    "std_hz": round(features.pitch_std, 2),
                    "jitter_percent": round(features.pitch_jitter * 100, 2),
                    "assessment": self._assess_pitch(features)
                },
                "dynamics": {
                    "mfcc_smoothness": round(features.mfcc_delta_std, 3),
                    "energy_variance": round(features.rms_var * 1000, 4),
                    "assessment": self._assess_dynamics(features)
                },
                "spectral": {
                    "centroid_hz": round(features.spectral_centroid_mean, 1),
                    "bandwidth_hz": round(features.spectral_bandwidth_mean, 1),
                    "rolloff_hz": round(features.spectral_rolloff_mean, 1),
                },
                "signal_quality": {
                    "zcr_mean": round(features.zcr_mean, 5),
                    "zcr_variance": round(features.zcr_var, 7),
                    "assessment": "synthetic" if features.zcr_mean < self.THRESHOLDS["zcr_ai"] else "natural"
                }
            },
            
            "indicator_summary": {
                "ai_signals": len(ai_indicators),
                "human_signals": len(human_indicators),
                "ai_weighted_score": round(scores["ai"], 2),
                "human_weighted_score": round(scores["human"], 2),
                "primary_indicators": [
                    {"name": i.name, "finding": i.description}
                    for i in sorted(indicators, key=lambda x: x.weight, reverse=True)[:3]
                ]
            }
        }
    
    def _assess_pitch(self, features: AudioFeatures) -> str:
        """Quick pitch assessment."""
        if features.pitch_mean == 0:
            return "undetected"
        if features.pitch_jitter < self.THRESHOLDS["pitch_jitter_ai"]:
            return "synthetic (low jitter)"
        if features.pitch_jitter > self.THRESHOLDS["pitch_jitter_human"]:
            return "natural (normal jitter)"
        return "ambiguous"
    
    def _assess_dynamics(self, features: AudioFeatures) -> str:
        """Quick dynamics assessment."""
        if features.mfcc_delta_std < self.THRESHOLDS["mfcc_delta_std_ai"]:
            return "synthetic (over-smooth)"
        if features.rms_var < self.THRESHOLDS["rms_var_ai"]:
            return "synthetic (flat energy)"
        return "natural"


# Module-level singleton
_explainer: Optional[ExplanationGenerator] = None


def get_explainer() -> ExplanationGenerator:
    """Get or create the explanation generator singleton."""
    global _explainer
    if _explainer is None:
        _explainer = ExplanationGenerator()
    return _explainer
