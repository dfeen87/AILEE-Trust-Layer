"""
AILEE Trust Layer — Optional Modules
Version: 1.1.2

Extended functionality for specialized and non-core use cases.

These modules are OPTIONAL helpers that extend the core AILEE Trust Pipeline.
They do NOT define governance domains and do NOT perform actuation.

Typical use cases:
- Configuration presets
- Peer / multi-model adapters
- Monitoring & observability
- Serialization & audit logging
- Replay & testing utilities

Governance domains (e.g., grids, automotive, datacenters) live under
the `domains/` package and are intentionally NOT imported here.
"""

# =============================================================================
# Configuration Presets
# =============================================================================
try:
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

        # Autonomous (generic, non-domain)
        AUTONOMOUS_VEHICLE,
        ROBOTICS_CONTROL,
        DRONE_NAVIGATION,

        # General
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

# =============================================================================
# Peer Adapters
# =============================================================================
try:
    from .ailee_peer_adapters import (
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

# =============================================================================
# Monitoring & Observability
# =============================================================================
try:
    from .ailee_monitors import (
        TrustMonitor,
        AlertingMonitor,
        PrometheusExporter,
        MetricSnapshot,
    )
    _HAS_MONITORS = True
except ImportError:
    _HAS_MONITORS = False

# =============================================================================
# Serialization
# =============================================================================
try:
    from .ailee_serialization import (
        decision_to_dict,
        decision_to_json,
        decision_to_audit_log,
        decision_to_csv_row,
        decision_to_compact_string,
    )
    _HAS_SERIALIZATION = True
except ImportError:
    _HAS_SERIALIZATION = False

# =============================================================================
# Replay & Testing
# =============================================================================
try:
    from .ailee_replay import (
        ReplayBuffer,
        ReplayRecord,
    )
    _HAS_REPLAY = True
except ImportError:
    _HAS_REPLAY = False

# =============================================================================
# Version & Metadata
# =============================================================================
__version__ = "1.1.2"
__author__ = "Don Michael Feeney Jr."
__license__ = "MIT"

# =============================================================================
# Public API (Dynamic Export)
# =============================================================================
__all__ = []

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

# =============================================================================
# Utility Introspection
# =============================================================================

def get_available_modules() -> dict:
    """Return availability of optional submodules."""
    return {
        "config_presets": _HAS_CONFIG_PRESETS,
        "peer_adapters": _HAS_PEER_ADAPTERS,
        "monitors": _HAS_MONITORS,
        "serialization": _HAS_SERIALIZATION,
        "replay": _HAS_REPLAY,
    }


def get_info() -> dict:
    """Return optional package metadata and status."""
    available = get_available_modules()
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "available_modules": available,
        "modules_loaded": sum(available.values()),
        "total_exports": len(__all__),
        "github": "https://github.com/dfeen87/ailee-trust-layer",
    }


__all__.extend([
    "get_available_modules",
    "get_info",
    "__version__",
    "__author__",
    "__license__",
])
