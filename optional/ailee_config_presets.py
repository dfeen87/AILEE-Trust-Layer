"""
AILEE Trust Pipeline - Configuration Presets
Version: 1.0.0

Drop-in configurations optimized for common AI domains.
These presets are battle-tested starting points that can be
fine-tuned for specific use cases.

Usage:
    from ailee_config_presets import LLM_SCORING
    pipeline = AileeTrustPipeline(LLM_SCORING)
"""

from ailee_trust_pipeline_v1 import AileeConfig


# ===========================
# LLM & NLP Applications
# ===========================

LLM_SCORING = AileeConfig(
    # Higher thresholds for LLM outputs (confidence is more reliable)
    accept_threshold=0.92,
    borderline_low=0.75,
    borderline_high=0.92,
    
    # Moderate consensus requirements (3+ models typical)
    consensus_quorum=3,
    consensus_delta=0.15,
    
    # Reasonable peer tolerance for ensemble voting
    grace_peer_delta=0.20,
    
    # Median fallback for robustness against outlier models
    fallback_mode="median",
    
    # Standard weights favor stability
    w_stability=0.45,
    w_agreement=0.30,
    w_likelihood=0.25,
    
    # Reasonable history for language tasks
    history_window=60,
    forecast_window=12,
    
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


LLM_CLASSIFICATION = AileeConfig(
    # Very high confidence required for classification tasks
    accept_threshold=0.95,
    borderline_low=0.85,
    borderline_high=0.95,
    
    # Strict consensus for safety-critical classification
    consensus_quorum=3,
    consensus_delta=0.10,
    
    grace_peer_delta=0.15,
    fallback_mode="last_good",
    
    # Favor agreement for classification stability
    w_stability=0.35,
    w_agreement=0.40,
    w_likelihood=0.25,
    
    history_window=50,
    enable_grace=True,
    enable_consensus=True,
)


LLM_GENERATION_QUALITY = AileeConfig(
    # Moderate thresholds for generative quality scoring
    accept_threshold=0.88,
    borderline_low=0.70,
    borderline_high=0.88,
    
    # Lower consensus requirements (subjective quality)
    consensus_quorum=2,
    consensus_delta=0.20,
    
    grace_peer_delta=0.25,
    fallback_mode="median",
    
    # Balanced weights for quality assessment
    w_stability=0.40,
    w_agreement=0.35,
    w_likelihood=0.25,
    
    history_window=40,
    enable_grace=True,
    enable_consensus=True,
)


# ===========================
# Sensor & IoT Applications
# ===========================

SENSOR_FUSION = AileeConfig(
    # Standard thresholds for physical sensor data
    accept_threshold=0.90,
    borderline_low=0.70,
    borderline_high=0.90,
    
    # Higher quorum for multi-sensor fusion
    consensus_quorum=4,
    consensus_delta=1.5,  # Absolute units (e.g., degrees, meters)
    
    grace_peer_delta=2.0,  # Physical tolerance
    
    # Physical constraints
    hard_min=0.0,
    hard_max=100.0,
    
    # Median for outlier rejection
    fallback_mode="median",
    
    # Heavy stability weight (sensors should be stable)
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,
    
    # Longer history for sensor drift detection
    history_window=100,
    forecast_window=15,
    
    enable_grace=True,
    enable_consensus=True,
)


TEMPERATURE_MONITORING = AileeConfig(
    # Relaxed for stable temperature readings
    accept_threshold=0.85,
    borderline_low=0.65,
    borderline_high=0.85,
    
    consensus_quorum=3,
    consensus_delta=0.5,  # 0.5°C tolerance
    
    grace_peer_delta=1.0,  # 1°C tolerance
    
    # Physical temperature bounds
    hard_min=-50.0,
    hard_max=150.0,
    
    fallback_mode="median",
    
    w_stability=0.55,  # Temperature changes slowly
    w_agreement=0.25,
    w_likelihood=0.20,
    
    history_window=120,
    forecast_window=20,
)


VIBRATION_DETECTION = AileeConfig(
    # Sensitive to anomalies
    accept_threshold=0.93,
    borderline_low=0.75,
    borderline_high=0.93,
    
    consensus_quorum=4,
    consensus_delta=0.2,  # Vibration amplitude units
    
    grace_peer_delta=0.3,
    
    hard_min=0.0,
    hard_max=10.0,
    
    fallback_mode="median",
    
    # Favor likelihood (anomaly detection)
    w_stability=0.30,
    w_agreement=0.30,
    w_likelihood=0.40,
    
    history_window=80,
    forecast_window=10,
)


# ===========================
# Financial Applications
# ===========================

FINANCIAL_SIGNAL = AileeConfig(
    # Very high confidence for financial decisions
    accept_threshold=0.95,
    borderline_low=0.85,
    borderline_high=0.95,
    
    # Strict consensus (multiple models/indicators)
    consensus_quorum=5,
    consensus_delta=0.05,  # Tight tolerance for prices/returns
    
    grace_peer_delta=0.08,
    
    # Last good value critical for continuity
    fallback_mode="last_good",
    
    # High stability weight (avoid false signals)
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,
    
    history_window=100,
    forecast_window=20,
    
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


TRADING_SIGNAL = AileeConfig(
    # Extremely high bar for trading signals
    accept_threshold=0.97,
    borderline_low=0.90,
    borderline_high=0.97,
    
    # Very strict consensus
    consensus_quorum=5,
    consensus_delta=0.03,
    
    grace_peer_delta=0.05,
    
    fallback_mode="last_good",
    
    # Maximum stability emphasis
    w_stability=0.55,
    w_agreement=0.25,
    w_likelihood=0.20,
    
    history_window=150,
    forecast_window=30,
    
    # Disable grace for trading (too risky)
    enable_grace=False,
    enable_consensus=True,
)


RISK_ASSESSMENT = AileeConfig(
    # High confidence for risk scoring
    accept_threshold=0.94,
    borderline_low=0.82,
    borderline_high=0.94,
    
    consensus_quorum=4,
    consensus_delta=0.10,
    
    grace_peer_delta=0.15,
    fallback_mode="median",
    
    # Balanced weights for risk assessment
    w_stability=0.40,
    w_agreement=0.35,
    w_likelihood=0.25,
    
    history_window=80,
    enable_grace=True,
    enable_consensus=True,
)


# ===========================
# Medical & Healthcare
# ===========================

MEDICAL_DIAGNOSIS = AileeConfig(
    # Extremely high bar for medical decisions
    accept_threshold=0.97,
    borderline_low=0.90,
    borderline_high=0.97,
    
    # Very strict consensus (multiple expert models)
    consensus_quorum=5,
    consensus_delta=0.05,
    
    grace_peer_delta=0.08,
    
    # Conservative fallback
    fallback_mode="last_good",
    
    # Favor stability and agreement
    w_stability=0.40,
    w_agreement=0.40,
    w_likelihood=0.20,
    
    history_window=50,
    
    # Grace enabled but strict
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,  # Critical for medical records
)


PATIENT_MONITORING = AileeConfig(
    # Balanced for continuous monitoring
    accept_threshold=0.90,
    borderline_low=0.75,
    borderline_high=0.90,
    
    consensus_quorum=3,
    consensus_delta=2.0,  # e.g., heart rate BPM
    
    grace_peer_delta=3.0,
    
    # Physical bounds (e.g., heart rate)
    hard_min=40.0,
    hard_max=200.0,
    
    fallback_mode="median",
    
    w_stability=0.45,
    w_agreement=0.30,
    w_likelihood=0.25,
    
    history_window=120,
    forecast_window=15,
    
    enable_grace=True,
    enable_consensus=True,
)


# ===========================
# Autonomous Systems
# ===========================

AUTONOMOUS_VEHICLE = AileeConfig(
    # Very high confidence for safety-critical decisions
    accept_threshold=0.96,
    borderline_low=0.88,
    borderline_high=0.96,
    
    # Strong consensus requirement
    consensus_quorum=4,
    consensus_delta=0.5,  # e.g., distance in meters
    
    grace_peer_delta=0.8,
    
    # Physical constraints
    hard_min=0.0,
    hard_max=100.0,
    
    fallback_mode="last_good",
    
    # High stability (safety)
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,
    
    history_window=60,
    forecast_window=10,
    
    enable_grace=True,
    enable_consensus=True,
)


ROBOTICS_CONTROL = AileeConfig(
    # High confidence for control signals
    accept_threshold=0.93,
    borderline_low=0.80,
    borderline_high=0.93,
    
    consensus_quorum=3,
    consensus_delta=1.0,  # Control signal units
    
    grace_peer_delta=1.5,
    
    # Control bounds
    hard_min=-100.0,
    hard_max=100.0,
    
    fallback_mode="median",
    
    w_stability=0.45,
    w_agreement=0.30,
    w_likelihood=0.25,
    
    history_window=80,
    forecast_window=12,
    
    enable_grace=True,
    enable_consensus=True,
)


DRONE_NAVIGATION = AileeConfig(
    # High confidence for navigation
    accept_threshold=0.94,
    borderline_low=0.82,
    borderline_high=0.94,
    
    consensus_quorum=4,
    consensus_delta=0.3,  # Position tolerance (meters)
    
    grace_peer_delta=0.5,
    
    fallback_mode="last_good",
    
    # Favor stability for smooth flight
    w_stability=0.50,
    w_agreement=0.28,
    w_likelihood=0.22,
    
    history_window=100,
    forecast_window=15,
)


# ===========================
# General Purpose
# ===========================

CONSERVATIVE = AileeConfig(
    # Very high bar for acceptance
    accept_threshold=0.95,
    borderline_low=0.85,
    borderline_high=0.95,
    
    consensus_quorum=4,
    consensus_delta=0.10,
    
    grace_peer_delta=0.15,
    fallback_mode="last_good",
    
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,
    
    history_window=100,
    enable_grace=True,
    enable_consensus=True,
)


BALANCED = AileeConfig(
    # Moderate thresholds for general use
    accept_threshold=0.90,
    borderline_low=0.70,
    borderline_high=0.90,
    
    consensus_quorum=3,
    consensus_delta=0.15,
    
    grace_peer_delta=0.20,
    fallback_mode="median",
    
    w_stability=0.45,
    w_agreement=0.30,
    w_likelihood=0.25,
    
    history_window=60,
    enable_grace=True,
    enable_consensus=True,
)


PERMISSIVE = AileeConfig(
    # Lower bar for exploratory use
    accept_threshold=0.85,
    borderline_low=0.60,
    borderline_high=0.85,
    
    consensus_quorum=2,
    consensus_delta=0.25,
    
    grace_peer_delta=0.30,
    fallback_mode="mean",
    
    w_stability=0.40,
    w_agreement=0.30,
    w_likelihood=0.30,
    
    history_window=40,
    enable_grace=True,
    enable_consensus=False,  # More lenient
)


# ===========================
# Configuration Catalog
# ===========================

PRESET_CATALOG = {
    # LLM & NLP
    "llm_scoring": LLM_SCORING,
    "llm_classification": LLM_CLASSIFICATION,
    "llm_generation_quality": LLM_GENERATION_QUALITY,
    
    # Sensors & IoT
    "sensor_fusion": SENSOR_FUSION,
    "temperature_monitoring": TEMPERATURE_MONITORING,
    "vibration_detection": VIBRATION_DETECTION,
    
    # Financial
    "financial_signal": FINANCIAL_SIGNAL,
    "trading_signal": TRADING_SIGNAL,
    "risk_assessment": RISK_ASSESSMENT,
    
    # Medical
    "medical_diagnosis": MEDICAL_DIAGNOSIS,
    "patient_monitoring": PATIENT_MONITORING,
    
    # Autonomous Systems
    "autonomous_vehicle": AUTONOMOUS_VEHICLE,
    "robotics_control": ROBOTICS_CONTROL,
    "drone_navigation": DRONE_NAVIGATION,
    
    # General Purpose
    "conservative": CONSERVATIVE,
    "balanced": BALANCED,
    "permissive": PERMISSIVE,
}


def get_preset(name: str) -> AileeConfig:
    """
    Retrieve a preset configuration by name.
    
    Args:
        name: Preset name (see PRESET_CATALOG)
        
    Returns:
        AileeConfig instance
        
    Raises:
        KeyError: If preset name not found
        
    Example:
        >>> config = get_preset("llm_scoring")
        >>> pipeline = AileeTrustPipeline(config)
    """
    if name not in PRESET_CATALOG:
        available = ", ".join(PRESET_CATALOG.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    
    return PRESET_CATALOG[name]


def list_presets() -> dict[str, str]:
    """
    List all available presets with descriptions.
    
    Returns:
        Dictionary mapping preset names to descriptions
    """
    descriptions = {
        "llm_scoring": "LLM ensemble scoring and ranking",
        "llm_classification": "High-confidence LLM classification tasks",
        "llm_generation_quality": "Generative output quality assessment",
        "sensor_fusion": "Multi-sensor data fusion",
        "temperature_monitoring": "Temperature sensor readings",
        "vibration_detection": "Vibration/anomaly detection",
        "financial_signal": "Financial market signals",
        "trading_signal": "High-stakes trading signals",
        "risk_assessment": "Financial risk scoring",
        "medical_diagnosis": "Medical diagnosis assistance",
        "patient_monitoring": "Continuous patient monitoring",
        "autonomous_vehicle": "Self-driving vehicle decisions",
        "robotics_control": "Robot control signals",
        "drone_navigation": "Drone navigation and positioning",
        "conservative": "High-confidence general purpose",
        "balanced": "Moderate general purpose",
        "permissive": "Exploratory general purpose",
    }
    return descriptions


# Convenience exports
__all__ = [
    # LLM
    'LLM_SCORING',
    'LLM_CLASSIFICATION',
    'LLM_GENERATION_QUALITY',
    # Sensors
    'SENSOR_FUSION',
    'TEMPERATURE_MONITORING',
    'VIBRATION_DETECTION',
    # Financial
    'FINANCIAL_SIGNAL',
    'TRADING_SIGNAL',
    'RISK_ASSESSMENT',
    # Medical
    'MEDICAL_DIAGNOSIS',
    'PATIENT_MONITORING',
    # Autonomous
    'AUTONOMOUS_VEHICLE',
    'ROBOTICS_CONTROL',
    'DRONE_NAVIGATION',
    # General
    'CONSERVATIVE',
    'BALANCED',
    'PERMISSIVE',
    # Utilities
    'PRESET_CATALOG',
    'get_preset',
    'list_presets',
]


# Demo
if __name__ == "__main__":
    print("=== AILEE Configuration Presets ===\n")
    
    print("Available presets:")
    for name, desc in list_presets().items():
        print(f"  • {name:25s} - {desc}")
    
    print("\n=== Example: LLM Scoring Config ===")
    config = get_preset("llm_scoring")
    print(f"Accept threshold: {config.accept_threshold}")
    print(f"Borderline range: [{config.borderline_low}, {config.borderline_high})")
    print(f"Consensus quorum: {config.consensus_quorum}")
    print(f"Fallback mode: {config.fallback_mode}")
    
    print("\n=== Example: Medical Diagnosis Config ===")
    config = get_preset("medical_diagnosis")
    print(f"Accept threshold: {config.accept_threshold} (very strict)")
    print(f"Consensus quorum: {config.consensus_quorum}")
    print(f"Audit metadata: {config.enable_audit_metadata}")
