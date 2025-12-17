"""
AILEE Trust Pipeline
Version: 1.0.0

Adaptive Integrity Layer for AI Decision Systems

A production-ready trust middleware for AI systems that transforms
uncertain, noisy, or distributed AI outputs into deterministic,
auditable, and safe final decisions.

Quick Start:
    >>> from ailee import AileeTrustPipeline, AileeConfig, LLM_SCORING
    >>> 
    >>> pipeline = AileeTrustPipeline(LLM_SCORING)
    >>> result = pipeline.process(
    ...     raw_value=10.5,
    ...     raw_confidence=0.75,
    ...     peer_values=[10.3, 10.6, 10.4]
    ... )
    >>> print(f"Trusted value: {result.value}")

Documentation:
    - GitHub: https://github.com/dfeen87/ailee-trust-layer
    - Paper: https://www.linkedin.com/pulse/navigating-nonlinear-ailees-framework-adaptive-resilient-feeney-bbkfe
"""

# Core pipeline and configuration
from .ailee_trust_pipeline_v1 import (
    AileeTrustPipeline,
    AileeConfig,
    DecisionResult,
    SafetyStatus,
    GraceStatus,
    ConsensusStatus,
)

# Configuration presets (all presets)
from .ailee_config_presets import (
    # LLM & NLP
    LLM_SCORING,
    LLM_CLASSIFICATION,
    LLM_GENERATION_QUALITY,
    # Sensors & IoT
    SENSOR_FUSION,
    TEMPERATURE_MONITORING,
    VIBRATION_DETECTION,
    # Financial
    FINANCIAL_SIGNAL,
    TRADING_SIGNAL,
    RISK_ASSESSMENT,
    # Medical
    MEDICAL_DIAGNOSIS,
    PATIENT_MONITORING,
    # Autonomous Systems
    AUTONOMOUS_VEHICLE,
    ROBOTICS_CONTROL,
    DRONE_NAVIGATION,
    # General Purpose
    CONSERVATIVE,
    BALANCED,
    PERMISSIVE,
    # Utilities
    get_preset,
    list_presets,
    PRESET_CATALOG,
)

# Peer adapters
from .ailee_peer_adapters import (
    PeerAdapter,
    StaticPeerAdapter,
    RollingPeerAdapter,
    CallbackPeerAdapter,
    MultiSourcePeerAdapter,
    WeightedPeerAdapter,
    FilteredPeerAdapter,
    MetadataPeerAdapter,
    create_multi_model_adapter,
)

# Monitoring and observability
from .ailee_monitors import (
    TrustMonitor,
    AlertingMonitor,
    PrometheusExporter,
    MetricSnapshot,
)

# Serialization utilities
from .ailee_serialization import (
    decision_to_dict,
    decision_to_json,
    decision_to_audit_log,
    decision_to_csv_row,
    decision_to_compact_string,
)

# Replay and testing
from .ailee_replay import (
    ReplayBuffer,
    ReplayRecord,
)

# Version info
__version__ = "1.0.0"
__author__ = "Don Michael Feeney Jr."
__license__ = "MIT"

# Public API
__all__ = [
    # Core
    "AileeTrustPipeline",
    "AileeConfig",
    "DecisionResult",
    "SafetyStatus",
    "GraceStatus",
    "ConsensusStatus",
    
    # Configuration Presets - LLM & NLP
    "LLM_SCORING",
    "LLM_CLASSIFICATION",
    "LLM_GENERATION_QUALITY",
    
    # Configuration Presets - Sensors & IoT
    "SENSOR_FUSION",
    "TEMPERATURE_MONITORING",
    "VIBRATION_DETECTION",
    
    # Configuration Presets - Financial
    "FINANCIAL_SIGNAL",
    "TRADING_SIGNAL",
    "RISK_ASSESSMENT",
    
    # Configuration Presets - Medical
    "MEDICAL_DIAGNOSIS",
    "PATIENT_MONITORING",
    
    # Configuration Presets - Autonomous
    "AUTONOMOUS_VEHICLE",
    "ROBOTICS_CONTROL",
    "DRONE_NAVIGATION",
    
    # Configuration Presets - General
    "CONSERVATIVE",
    "BALANCED",
    "PERMISSIVE",
    
    # Configuration Utilities
    "get_preset",
    "list_presets",
    "PRESET_CATALOG",
    
    # Peer Adapters
    "PeerAdapter",
    "StaticPeerAdapter",
    "RollingPeerAdapter",
    "CallbackPeerAdapter",
    "MultiSourcePeerAdapter",
    "WeightedPeerAdapter",
    "FilteredPeerAdapter",
    "MetadataPeerAdapter",
    "create_multi_model_adapter",
    
    # Monitoring
    "TrustMonitor",
    "AlertingMonitor",
    "PrometheusExporter",
    "MetricSnapshot",
    
    # Serialization
    "decision_to_dict",
    "decision_to_json",
    "decision_to_audit_log",
    "decision_to_csv_row",
    "decision_to_compact_string",
    
    # Replay & Testing
    "ReplayBuffer",
    "ReplayRecord",
    
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]


# Convenience function for quick setup
def create_pipeline(preset_name: str = "balanced", **config_overrides) -> AileeTrustPipeline:
    """
    Create a pre-configured AILEE pipeline with optional overrides.
    
    Args:
        preset_name: Name of configuration preset (default: "balanced")
        **config_overrides: Override specific config parameters
        
    Returns:
        Configured AileeTrustPipeline instance
        
    Example:
        >>> # Use preset as-is
        >>> pipeline = create_pipeline("llm_scoring")
        >>> 
        >>> # Use preset with overrides
        >>> pipeline = create_pipeline(
        ...     "sensor_fusion",
        ...     accept_threshold=0.92,
        ...     consensus_quorum=5
        ... )
    """
    config = get_preset(preset_name)
    
    # Apply overrides
    if config_overrides:
        config_dict = config.__dict__.copy()
        config_dict.update(config_overrides)
        config = AileeConfig(**config_dict)
    
    return AileeTrustPipeline(config)


# Add create_pipeline to exports
__all__.append("create_pipeline")


# Package-level docstring helper
def get_info() -> dict:
    """
    Get package information.
    
    Returns:
        Dictionary with version, author, and available presets
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "available_presets": list(list_presets().keys()),
        "github": "https://github.com/dfeen87/ailee-trust-layer",
    }


__all__.append("get_info")
