"""
AILEE Trust Layer v1.1.0
Adaptive Integrity Layer for AI Decision Systems

A production-ready trust middleware for AI systems that transforms
uncertain, noisy, or distributed AI outputs into deterministic,
auditable, and safe final decisions.

Quick Start:
    >>> from ailee import AileeTrustPipeline, LLM_SCORING
    >>> 
    >>> pipeline = AileeTrustPipeline(LLM_SCORING)
    >>> result = pipeline.process(
    ...     raw_value=10.5,
    ...     raw_confidence=0.75,
    ...     peer_values=[10.3, 10.6, 10.4]
    ... )
    >>> print(f"Trusted value: {result.value}")

Core Philosophy:
    Trust is not a probability. Trust is a structure.

Documentation:
    - GitHub: https://github.com/dfeen87/ailee-trust-layer
    - Paper: https://www.linkedin.com/pulse/navigating-nonlinear-ailees-framework-adaptive-resilient-feeney-bbkfe
    - Substack: https://substack.com/home/post/p-165731733

License: MIT
Author: Don Michael Feeney Jr.
"""

# ===========================
# Core Pipeline
# ===========================
from .ailee_trust_pipeline_v1 import (
    AileeTrustPipeline,
    AileeConfig,
    DecisionResult,
    SafetyStatus,
    GraceStatus,
    ConsensusStatus,
)

# ===========================
# Configuration Presets
# ===========================
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
)

# ===========================
# Peer Adapters
# ===========================
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

# ===========================
# Monitoring & Observability
# ===========================
from .ailee_monitors import (
    TrustMonitor,
    AlertingMonitor,
    PrometheusExporter,
    MetricSnapshot,
)

# ===========================
# Serialization
# ===========================
from .ailee_serialization import (
    decision_to_dict,
    decision_to_json,
    decision_to_audit_log,
    decision_to_csv_row,
    decision_to_compact_string,
)

# ===========================
# Replay & Testing
# ===========================
from .ailee_replay import (
    ReplayBuffer,
    ReplayRecord,
)

# ===========================
# Version & Metadata
# ===========================
__version__ = "1.1.0"
__author__ = "Don Michael Feeney Jr."
__license__ = "MIT"
__status__ = "Production/Stable"

# ===========================
# Public API
# ===========================
__all__ = [
    # ----------------
    # Core Pipeline
    # ----------------
    "AileeTrustPipeline",
    "AileeConfig",
    "DecisionResult",
    "SafetyStatus",
    "GraceStatus",
    "ConsensusStatus",
    
    # ----------------
    # Configuration Presets - LLM & NLP
    # ----------------
    "LLM_SCORING",
    "LLM_CLASSIFICATION",
    "LLM_GENERATION_QUALITY",
    
    # ----------------
    # Configuration Presets - Sensors & IoT
    # ----------------
    "SENSOR_FUSION",
    "TEMPERATURE_MONITORING",
    "VIBRATION_DETECTION",
    
    # ----------------
    # Configuration Presets - Financial
    # ----------------
    "FINANCIAL_SIGNAL",
    "TRADING_SIGNAL",
    "RISK_ASSESSMENT",
    
    # ----------------
    # Configuration Presets - Medical
    # ----------------
    "MEDICAL_DIAGNOSIS",
    "PATIENT_MONITORING",
    
    # ----------------
    # Configuration Presets - Autonomous
    # ----------------
    "AUTONOMOUS_VEHICLE",
    "ROBOTICS_CONTROL",
    "DRONE_NAVIGATION",
    
    # ----------------
    # Configuration Presets - General
    # ----------------
    "CONSERVATIVE",
    "BALANCED",
    "PERMISSIVE",
    
    # ----------------
    # Configuration Utilities
    # ----------------
    "get_preset",
    "list_presets",
    
    # ----------------
    # Peer Adapters
    # ----------------
    "PeerAdapter",
    "StaticPeerAdapter",
    "RollingPeerAdapter",
    "CallbackPeerAdapter",
    "MultiSourcePeerAdapter",
    "WeightedPeerAdapter",
    "FilteredPeerAdapter",
    "MetadataPeerAdapter",
    "create_multi_model_adapter",
    
    # ----------------
    # Monitoring & Observability
    # ----------------
    "TrustMonitor",
    "AlertingMonitor",
    "PrometheusExporter",
    "MetricSnapshot",
    
    # ----------------
    # Serialization
    # ----------------
    "decision_to_dict",
    "decision_to_json",
    "decision_to_audit_log",
    "decision_to_csv_row",
    "decision_to_compact_string",
    
    # ----------------
    # Replay & Testing
    # ----------------
    "ReplayBuffer",
    "ReplayRecord",
    
    # ----------------
    # Metadata
    # ----------------
    "__version__",
    "__author__",
    "__license__",
    "__status__",
]


# ===========================
# Convenience Functions
# ===========================

def create_pipeline(preset_name: str = "balanced", **config_overrides):
    """
    Create a pre-configured AILEE pipeline with optional overrides.
    
    This is the recommended way to quickly instantiate production pipelines.
    
    Args:
        preset_name: Name of configuration preset (default: "balanced")
            Available: llm_scoring, sensor_fusion, financial_signal,
            medical_diagnosis, autonomous_vehicle, and 12 more.
            Use list_presets() to see all options.
        **config_overrides: Override specific config parameters
        
    Returns:
        Configured AileeTrustPipeline instance
        
    Examples:
        >>> # Use preset as-is
        >>> pipeline = create_pipeline("llm_scoring")
        >>> 
        >>> # Use preset with custom overrides
        >>> pipeline = create_pipeline(
        ...     "sensor_fusion",
        ...     accept_threshold=0.92,
        ...     consensus_quorum=5
        ... )
        >>> 
        >>> # Quick production setup
        >>> pipeline = create_pipeline("medical_diagnosis")
        >>> result = pipeline.process(raw_value=98.6, raw_confidence=0.91)
    
    Raises:
        KeyError: If preset_name is not found
    """
    config = get_preset(preset_name)
    
    # Apply configuration overrides
    if config_overrides:
        # Create a new config with overrides
        config_dict = {
            key: value for key, value in config.__dict__.items()
            if not key.startswith('_')
        }
        config_dict.update(config_overrides)
        config = AileeConfig(**config_dict)
    
    return AileeTrustPipeline(config)


def get_info():
    """
    Get package information including version and available presets.
    
    Returns:
        Dictionary with version, author, license, and available presets
        
    Example:
        >>> import ailee
        >>> info = ailee.get_info()
        >>> print(f"AILEE v{info['version']}")
        >>> print(f"Available presets: {len(info['available_presets'])}")
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "status": __status__,
        "available_presets": list(list_presets().keys()),
        "preset_count": len(list_presets()),
        "github": "https://github.com/dfeen87/ailee-trust-layer",
        "documentation": "https://www.linkedin.com/pulse/navigating-nonlinear-ailees-framework-adaptive-resilient-feeney-bbkfe",
    }


# Add convenience functions to exports
__all__.extend(["create_pipeline", "get_info"])


# ===========================
# Module-level Validation
# ===========================

def _validate_imports():
    """
    Validate that all critical imports succeeded.
    Run at module import time to catch issues early.
    """
    critical_classes = [
        AileeTrustPipeline,
        AileeConfig,
        DecisionResult,
        TrustMonitor,
        ReplayBuffer,
    ]
    
    for cls in critical_classes:
        if cls is None:
            raise ImportError(f"Critical class {cls.__name__} failed to import")
    
    # Validate at least some presets exist
    if len(list_presets()) < 10:
        raise ImportError("Configuration presets failed to load properly")


# Run validation on import (fails fast if something is broken)
_validate_imports()


# ===========================
# Deprecation Warnings (if needed in future)
# ===========================

# Example for future use:
# import warnings
# def deprecated_function():
#     warnings.warn(
#         "This function is deprecated and will be removed in v2.0. "
#         "Use new_function() instead.",
#         DeprecationWarning,
#         stacklevel=2
#     )
