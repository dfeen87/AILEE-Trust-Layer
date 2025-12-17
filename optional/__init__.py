"""
AILEE Trust Layer - Optional Modules
Version: 1.1.0

Extended functionality for specialized use cases.
These modules are optional and can be imported individually as needed.

Quick Start:
    # Configuration presets
    from ailee.optional import LLM_SCORING, MEDICAL_DIAGNOSIS
    
    # Peer adapters
    from ailee.optional import create_multi_model_adapter
    
    # Monitoring
    from ailee.optional import TrustMonitor, AlertingMonitor
    
    # Data center helpers
    from ailee.optional import CoolingController, PowerCapController
    
    # Serialization
    from ailee.optional import decision_to_json, decision_to_audit_log
    
    # Replay/Testing
    from ailee.optional import ReplayBuffer

Documentation:
    - GitHub: https://github.com/dfeen87/ailee-trust-layer
    - Paper: https://www.linkedin.com/pulse/navigating-nonlinear-ailees-framework-adaptive-resilient-feeney-bbkfe
"""

# ===========================
# Configuration Presets
# ===========================
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
        # Autonomous
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

# ===========================
# Peer Adapters
# ===========================
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

# ===========================
# Monitoring & Observability
# ===========================
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

# ===========================
# Serialization
# ===========================
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

# ===========================
# Replay & Testing
# ===========================
try:
    from .ailee_replay import (
        ReplayBuffer,
        ReplayRecord,
    )
    _HAS_REPLAY = True
except ImportError:
    _HAS_REPLAY = False

# ===========================
# Data Center Helpers
# ===========================
try:
    from .ailee_datacenter_helpers import (
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


# ===========================
# Public API (Dynamic Export)
# ===========================
__all__ = []

# Configuration Presets
if _HAS_CONFIG_PRESETS:
    __all__.extend([
        # LLM & NLP
        'LLM_SCORING',
        'LLM_CLASSIFICATION',
        'LLM_GENERATION_QUALITY',
        # Sensors & IoT
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
        'get_preset',
        'list_presets',
        'PRESET_CATALOG',
    ])

# Peer Adapters
if _HAS_PEER_ADAPTERS:
    __all__.extend([
        'PeerAdapter',
        'StaticPeerAdapter',
        'RollingPeerAdapter',
        'CallbackPeerAdapter',
        'MultiSourcePeerAdapter',
        'WeightedPeerAdapter',
        'FilteredPeerAdapter',
        'MetadataPeerAdapter',
        'PeerMetadata',
        'create_multi_model_adapter',
    ])

# Monitoring
if _HAS_MONITORS:
    __all__.extend([
        'TrustMonitor',
        'AlertingMonitor',
        'PrometheusExporter',
        'MetricSnapshot',
    ])

# Serialization
if _HAS_SERIALIZATION:
    __all__.extend([
        'decision_to_dict',
        'decision_to_json',
        'decision_to_audit_log',
        'decision_to_csv_row',
        'decision_to_compact_string',
    ])

# Replay & Testing
if _HAS_REPLAY:
    __all__.extend([
        'ReplayBuffer',
        'ReplayRecord',
    ])

# Data Center Helpers
if _HAS_DATACENTER_HELPERS:
    __all__.extend([
        # Configurations
        'COOLING_CONTROL',
        'POWER_CAPPING',
        'WORKLOAD_PLACEMENT',
        # Telemetry
        'SensorReading',
        'TelemetryProcessor',
        # Controllers
        'CoolingController',
        'PowerCapController',
        'WorkloadPlacementGovernor',
        # Monitoring
        'DataCenterMonitor',
    ])


# ===========================
# Utility Functions
# ===========================

def get_available_modules():
    """
    Return which optional modules are available.
    
    Returns:
        Dict mapping module names to availability status
        
    Example:
        >>> from ailee.optional import get_available_modules
        >>> available = get_available_modules()
        >>> print(f"Data center helpers: {available['datacenter_helpers']}")
    """
    return {
        'config_presets': _HAS_CONFIG_PRESETS,
        'peer_adapters': _HAS_PEER_ADAPTERS,
        'monitors': _HAS_MONITORS,
        'serialization': _HAS_SERIALIZATION,
        'replay': _HAS_REPLAY,
        'datacenter_helpers': _HAS_DATACENTER_HELPERS,
    }


def get_info():
    """
    Get optional modules package information.
    
    Returns:
        Dictionary with version, available modules, and counts
        
    Example:
        >>> from ailee.optional import get_info
        >>> info = get_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Total exports: {info['total_exports']}")
    """
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


# Add utility functions to exports
__all__.extend(['get_available_modules', 'get_info'])


# Add metadata to exports
__all__.extend(['__version__', '__author__', '__license__'])


# ===========================
# Convenience Functions
# ===========================

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
        
    Example:
        >>> from ailee.optional import create_datacenter_pipeline
        >>> pipeline = create_datacenter_pipeline("cooling", hard_min=18.0, hard_max=26.0)
    """
    if not _HAS_CONFIG_PRESETS:
        raise ImportError("ailee_config_presets module not available")
    
    # Import locally to avoid circular dependency
    from ..ailee_trust_pipeline_v1 import AileeTrustPipeline
    
    # Select appropriate config
    config_map = {
        "cooling": COOLING_CONTROL if _HAS_DATACENTER_HELPERS else TEMPERATURE_MONITORING,
        "power": POWER_CAPPING if _HAS_DATACENTER_HELPERS else AUTONOMOUS_VEHICLE,
        "workload": WORKLOAD_PLACEMENT if _HAS_DATACENTER_HELPERS else BALANCED,
    }
    
    if system_type not in config_map:
        raise ValueError(f"Unknown system_type: {system_type}. Choose from: {list(config_map.keys())}")
    
    config = config_map[system_type]
    
    # Apply overrides
    if config_overrides:
        from ..ailee_trust_pipeline_v1 import AileeConfig
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        config_dict.update(config_overrides)
        config = AileeConfig(**config_dict)
    
    return AileeTrustPipeline(config)


# Only add if datacenter helpers are available
if _HAS_DATACENTER_HELPERS:
    __all__.append('create_datacenter_pipeline')


# ===========================
# Module Status Report
# ===========================

def print_module_status():
    """
    Print a status report of available optional modules.
    
    Example:
        >>> from ailee.optional import print_module_status
        >>> print_module_status()
    """
    print("=" * 60)
    print("AILEE Optional Modules Status")
    print("=" * 60)
    print(f"Version: {__version__}")
    print()
    
    modules = get_available_modules()
    for name, available in modules.items():
        status = "✓ AVAILABLE" if available else "✗ NOT FOUND"
        print(f"{name:25s} {status}")
    
    print()
    print(f"Total modules loaded: {sum(modules.values())}/{len(modules)}")
    print(f"Total exports: {len(__all__)}")
    print("=" * 60)


if _HAS_DATACENTER_HELPERS:
    __all__.append('print_module_status')
