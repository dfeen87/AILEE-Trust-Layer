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
# Optional Modules (from optional/)
# ===========================
try:
    from .optional.ailee_config_presets import (
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
    _HAS_CONFIG_PRESETS = True
except ImportError:
    _HAS_CONFIG_PRESETS = False

try:
    from .optional.ailee_peer_adapters import (
        PeerAdapter,
        StaticPeerAdapter,
        RollingPeerAdapter,
        CallbackPeerAdapter,
        MultiSourcePeerAdapter,
        WeightedPeerAdapter,
        FilteredPeerAdapter,
        MetadataPeerAdapter,
        PeerMetadata,
        create_multi_model_adapter,
    )
    _HAS_PEER_ADAPTERS = True
except ImportError:
    _HAS_PEER_ADAPTERS = False

try:
    from .optional.ailee_monitors import (
        TrustMonitor,
        AlertingMonitor,
        PrometheusExporter,
        MetricSnapshot,
    )
    _HAS_MONITORS = True
except ImportError:
    _HAS_MONITORS = False

try:
    from .optional.ailee_serialization import (
        decision_to_dict,
        decision_to_json,
        decision_to_audit_log,
        decision_to_csv_row,
        decision_to_compact_string,
    )
    _HAS_SERIALIZATION = True
except ImportError:
    _HAS_SERIALIZATION = False

try:
    from .optional.ailee_replay import (
        ReplayBuffer,
        ReplayRecord,
    )
    _HAS_REPLAY = True
except ImportError:
    _HAS_REPLAY = False

try:
    from .optional.ailee_datacenter_helpers import (
        # Configurations
        COOLING_CONTROL,
        POWER_CAPPING,
        WORKLOAD_PLACEMENT,
        # Telemetry
        SensorReading,
        TelemetryProcessor,
        # Controllers
        CoolingController,
        PowerCapController,
        WorkloadPlacementGovernor,
        # Monitoring
        DataCenterMonitor,
    )
    _HAS_DATACENTER_HELPERS = True
except ImportError:
    _HAS_DATACENTER_HELPERS = False

# ===========================
# Version & Metadata
# ===========================
__version__ = "1.1.0"
__author__ = "Don Michael Feeney Jr."
__license__ = "MIT"
__status__ = "Production/Stable"

# ===========================
# Public API (Dynamic)
# ===========================
__all__ = [
    # Core Pipeline (Always Available)
    "AileeTrustPipeline",
    "AileeConfig",
    "DecisionResult",
    "SafetyStatus",
    "GraceStatus",
    "ConsensusStatus",
    
    # Metadata
    "__version__",
    "__author__",
    "__license__",
    "__status__",
]

# Configuration Presets
if _HAS_CONFIG_PRESETS:
    __all__.extend([
        # LLM & NLP
        "LLM_SCORING",
        "LLM_CLASSIFICATION",
        "LLM_GENERATION_QUALITY",
        # Sensors & IoT
        "SENSOR_FUSION",
        "TEMPERATURE_MONITORING",
        "VIBRATION_DETECTION",
        # Financial
        "FINANCIAL_SIGNAL",
        "TRADING_SIGNAL",
        "RISK_ASSESSMENT",
        # Medical
        "MEDICAL_DIAGNOSIS",
        "PATIENT_MONITORING",
        # Autonomous
        "AUTONOMOUS_VEHICLE",
        "ROBOTICS_CONTROL",
        "DRONE_NAVIGATION",
        # General
        "CONSERVATIVE",
        "BALANCED",
        "PERMISSIVE",
        # Utilities
        "get_preset",
        "list_presets",
        "PRESET_CATALOG",
    ])

# Peer Adapters
if _HAS_PEER_ADAPTERS:
    __all__.extend([
        "PeerAdapter",
        "StaticPeerAdapter",
        "RollingPeerAdapter",
        "CallbackPeerAdapter",
        "MultiSourcePeerAdapter",
        "WeightedPeerAdapter",
        "FilteredPeerAdapter",
        "MetadataPeerAdapter",
        "PeerMetadata",
        "create_multi_model_adapter",
    ])

# Monitoring
if _HAS_MONITORS:
    __all__.extend([
        "TrustMonitor",
        "AlertingMonitor",
        "PrometheusExporter",
        "MetricSnapshot",
    ])

# Serialization
if _HAS_SERIALIZATION:
    __all__.extend([
        "decision_to_dict",
        "decision_to_json",
        "decision_to_audit_log",
        "decision_to_csv_row",
        "decision_to_compact_string",
    ])

# Replay & Testing
if _HAS_REPLAY:
    __all__.extend([
        "ReplayBuffer",
        "ReplayRecord",
    ])

# Data Center Helpers
if _HAS_DATACENTER_HELPERS:
    __all__.extend([
        # Configurations
        "COOLING_CONTROL",
        "POWER_CAPPING",
        "WORKLOAD_PLACEMENT",
        # Telemetry
        "SensorReading",
        "TelemetryProcessor",
        # Controllers
        "CoolingController",
        "PowerCapController",
        "WorkloadPlacementGovernor",
        # Monitoring
        "DataCenterMonitor",
    ])


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
        ImportError: If config presets module not available
        KeyError: If preset_name is not found
    """
    if not _HAS_CONFIG_PRESETS:
        raise ImportError(
            "Configuration presets not available. "
            "Ensure optional/ailee_config_presets.py is present."
        )
    
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


def create_datacenter_pipeline(
    system_type: str = "cooling",
    **config_overrides
):
    """
    Create a pre-configured pipeline for data center systems.
    
    Args:
        system_type: Type of system ("cooling", "power", "workload")
        **config_overrides: Override specific config parameters
        
    Returns:
        Configured AileeTrustPipeline instance
        
    Examples:
        >>> # Cooling control
        >>> pipeline = create_datacenter_pipeline("cooling", hard_min=18.0, hard_max=26.0)
        >>> 
        >>> # Power capping
        >>> pipeline = create_datacenter_pipeline("power")
        >>> 
        >>> # Workload placement
        >>> pipeline = create_datacenter_pipeline("workload", consensus_quorum=4)
    
    Raises:
        ImportError: If datacenter helpers not available
        ValueError: If system_type is invalid
    """
    if not _HAS_DATACENTER_HELPERS:
        raise ImportError(
            "Data center helpers not available. "
            "Ensure optional/ailee_datacenter_helpers.py is present."
        )
    
    # Map system types to configs
    config_map = {
        "cooling": COOLING_CONTROL,
        "power": POWER_CAPPING,
        "workload": WORKLOAD_PLACEMENT,
    }
    
    if system_type not in config_map:
        raise ValueError(
            f"Unknown system_type: {system_type}. "
            f"Choose from: {list(config_map.keys())}"
        )
    
    config = config_map[system_type]
    
    # Apply overrides
    if config_overrides:
        config_dict = {
            key: value for key, value in config.__dict__.items()
            if not key.startswith('_')
        }
        config_dict.update(config_overrides)
        config = AileeConfig(**config_dict)
    
    return AileeTrustPipeline(config)


def get_info():
    """
    Get package information including version and available modules.
    
    Returns:
        Dictionary with version, author, license, and module availability
        
    Example:
        >>> import ailee
        >>> info = ailee.get_info()
        >>> print(f"AILEE v{info['version']}")
        >>> print(f"Available presets: {len(info.get('available_presets', []))}")
        >>> print(f"Data center helpers: {info['modules']['datacenter_helpers']}")
    """
    info = {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "status": __status__,
        "github": "https://github.com/dfeen87/ailee-trust-layer",
        "documentation": "https://www.linkedin.com/pulse/navigating-nonlinear-ailees-framework-adaptive-resilient-feeney-bbkfe",
        "modules": {
            "config_presets": _HAS_CONFIG_PRESETS,
            "peer_adapters": _HAS_PEER_ADAPTERS,
            "monitors": _HAS_MONITORS,
            "serialization": _HAS_SERIALIZATION,
            "replay": _HAS_REPLAY,
            "datacenter_helpers": _HAS_DATACENTER_HELPERS,
        },
        "total_exports": len(__all__),
    }
    
    # Add preset list if available
    if _HAS_CONFIG_PRESETS:
        info["available_presets"] = list(list_presets().keys())
        info["preset_count"] = len(list_presets())
    
    return info


def print_status():
    """
    Print a detailed status report of the AILEE installation.
    
    Example:
        >>> import ailee
        >>> ailee.print_status()
    """
    print("=" * 70)
    print(f"AILEE Trust Layer v{__version__}")
    print("=" * 70)
    print(f"Status: {__status__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print()
    
    print("Module Availability:")
    print("-" * 70)
    modules = {
        "Configuration Presets": _HAS_CONFIG_PRESETS,
        "Peer Adapters": _HAS_PEER_ADAPTERS,
        "Monitoring & Observability": _HAS_MONITORS,
        "Serialization Utilities": _HAS_SERIALIZATION,
        "Replay & Testing": _HAS_REPLAY,
        "Data Center Helpers": _HAS_DATACENTER_HELPERS,
    }
    
    for name, available in modules.items():
        status = "✓ AVAILABLE" if available else "✗ NOT FOUND"
        print(f"  {name:30s} {status}")
    
    print()
    print(f"Total exports: {len(__all__)}")
    
    if _HAS_CONFIG_PRESETS:
        print(f"Configuration presets: {len(list_presets())}")
    
    print("=" * 70)


# Add convenience functions to exports
__all__.extend(["create_pipeline", "create_datacenter_pipeline", "get_info", "print_status"])


# ===========================
# Module-level Validation
# ===========================

def _validate_imports():
    """
    Validate that all critical imports succeeded.
    Run at module import time to catch issues early.
    """
    # Core pipeline must always be available
    critical_classes = [
        AileeTrustPipeline,
        AileeConfig,
        DecisionResult,
        SafetyStatus,
        GraceStatus,
        ConsensusStatus,
    ]
    
    for cls in critical_classes:
        if cls is None:
            raise ImportError(
                f"Critical class {cls.__name__} failed to import. "
                "Core pipeline is required."
            )
    
    # Warn if optional modules missing (but don't fail)
    if not _HAS_CONFIG_PRESETS:
        import warnings
        warnings.warn(
            "Configuration presets not available. Some features will be limited.",
            ImportWarning
        )
    
    # Validate presets if available
    if _HAS_CONFIG_PRESETS and len(list_presets()) < 10:
        raise ImportError(
            "Configuration presets loaded but incomplete. "
            f"Found {len(list_presets())} presets, expected at least 10."
        )


# Run validation on import (fails fast if core is broken)
try:
    _validate_imports()
except ImportError as e:
    # Re-raise with helpful context
    raise ImportError(
        f"AILEE Trust Layer failed to initialize: {e}\n"
        "Please ensure all required files are present."
    ) from e


# ===========================
# Deprecation Warnings (Future Use)
# ===========================

# Example for future use in v2.0.0:
# import warnings
#
# def deprecated_function():
#     """
#     Example deprecated function.
#     This will be removed in v2.0.0.
#     """
#     warnings.warn(
#         "This function is deprecated and will be removed in v2.0.0. "
#         "Use new_function() instead.",
#         DeprecationWarning,
#         stacklevel=2
#     )
#     # ... implementation


# ===========================
# Optional: Access to Optional Module
# ===========================

# Users can also access the optional module directly
# This allows: from ailee import optional
# Then: optional.get_available_modules()

try:
    from . import optional
    __all__.append("optional")
except ImportError:
    # Optional module not available
    pass


# ===========================
# Package Initialization Complete
# ===========================

# Silent initialization - ready for import
# No print statements or side effects by default
