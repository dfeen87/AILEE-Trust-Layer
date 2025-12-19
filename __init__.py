"""
AILEE Trust Layer v1.1.1
Adaptive Integrity Layer for AI Decision Systems

A production-ready trust middleware for AI systems that transforms
uncertain, noisy, or distributed AI outputs into deterministic,
auditable, and safe final decisions.

Core Philosophy:
    Trust is not a probability. Trust is a structure.
"""

# =============================================================================
# Core Pipeline (Always Available)
# =============================================================================
from .ailee_trust_pipeline_v1 import (
    AileeTrustPipeline,
    AileeConfig,
    DecisionResult,
    SafetyStatus,
    GraceStatus,
    ConsensusStatus,
)

# =============================================================================
# Optional Helper Modules (domain-agnostic)
# =============================================================================
try:
    from .optional.ailee_config_presets import (
        LLM_SCORING,
        LLM_CLASSIFICATION,
        LLM_GENERATION_QUALITY,
        SENSOR_FUSION,
        TEMPERATURE_MONITORING,
        VIBRATION_DETECTION,
        FINANCIAL_SIGNAL,
        TRADING_SIGNAL,
        RISK_ASSESSMENT,
        MEDICAL_DIAGNOSIS,
        PATIENT_MONITORING,
        AUTONOMOUS_VEHICLE,
        ROBOTICS_CONTROL,
        DRONE_NAVIGATION,
        CONSERVATIVE,
        BALANCED,
        PERMISSIVE,
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

# =============================================================================
# Domains (optional, governance layers)
# =============================================================================
try:
    from .domains.imaging import ImagingDomain
    _HAS_IMAGING_DOMAIN = True
except ImportError:
    _HAS_IMAGING_DOMAIN = False

# =============================================================================
# Metadata
# =============================================================================
__version__ = "1.1.1"
__author__ = "Don Michael Feeney Jr."
__license__ = "MIT"
__status__ = "Production/Stable"

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    "AileeTrustPipeline",
    "AileeConfig",
    "DecisionResult",
    "SafetyStatus",
    "GraceStatus",
    "ConsensusStatus",
    "__version__",
    "__author__",
    "__license__",
    "__status__",
]

if _HAS_CONFIG_PRESETS:
    __all__.extend([
        "LLM_SCORING",
        "LLM_CLASSIFICATION",
        "LLM_GENERATION_QUALITY",
        "SENSOR_FUSION",
        "TEMPERATURE_MONITORING",
        "VIBRATION_DETECTION",
        "FINANCIAL_SIGNAL",
        "TRADING_SIGNAL",
        "RISK_ASSESSMENT",
        "MEDICAL_DIAGNOSIS",
        "PATIENT_MONITORING",
        "AUTONOMOUS_VEHICLE",
        "ROBOTICS_CONTROL",
        "DRONE_NAVIGATION",
        "CONSERVATIVE",
        "BALANCED",
        "PERMISSIVE",
        "get_preset",
        "list_presets",
        "PRESET_CATALOG",
    ])

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

if _HAS_MONITORS:
    __all__.extend([
        "TrustMonitor",
        "AlertingMonitor",
        "PrometheusExporter",
        "MetricSnapshot",
    ])

if _HAS_SERIALIZATION:
    __all__.extend([
        "decision_to_dict",
        "decision_to_json",
        "decision_to_audit_log",
        "decision_to_csv_row",
        "decision_to_compact_string",
    ])

if _HAS_REPLAY:
    __all__.extend([
        "ReplayBuffer",
        "ReplayRecord",
    ])

if _HAS_IMAGING_DOMAIN:
    __all__.append("ImagingDomain")

# =============================================================================
# Convenience
# =============================================================================

def create_pipeline(preset_name: str = "balanced", **overrides):
    if not _HAS_CONFIG_PRESETS:
        raise ImportError("Configuration presets not available")
    config = get_preset(preset_name)
    if overrides:
        cfg = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        cfg.update(overrides)
        config = AileeConfig(**cfg)
    return AileeTrustPipeline(config)

__all__.append("create_pipeline")

# Allow explicit access to optional modules and domains namespace
from . import optional
from . import domains

__all__.extend(["optional", "domains"])
