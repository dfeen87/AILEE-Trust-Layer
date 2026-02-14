"""
AILEE Trust Layer v2.0.0
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

try:
    from .optional.ailee_ai_integrations import (
        AIResponse,
        AIAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        HuggingFaceAdapter,
        LangChainAdapter,
        MultiModelEnsemble,
        create_openai_adapter,
        create_anthropic_adapter,
        create_huggingface_adapter,
        create_langchain_adapter,
        create_multi_model_ensemble,
    )
    _HAS_AI_INTEGRATIONS = True
except ImportError:
    _HAS_AI_INTEGRATIONS = False

# =============================================================================
# Domains (optional, production-grade governance layers)
# =============================================================================
try:
    from .domains.imaging import ImagingDomain
    _HAS_IMAGING_DOMAIN = True
except ImportError:
    _HAS_IMAGING_DOMAIN = False

try:
    from .domains.robotics import RoboticsDomain
    _HAS_ROBOTICS_DOMAIN = True
except ImportError:
    _HAS_ROBOTICS_DOMAIN = False

try:
    from .domains.telecommunications import TelecomGovernor
    _HAS_TELECOMMUNICATIONS_DOMAIN = True
except ImportError:
    _HAS_TELECOMMUNICATIONS_DOMAIN = False

try:
    from .domains.ocean import OceanGovernor
    _HAS_OCEAN_DOMAIN = True
except ImportError:
    _HAS_OCEAN_DOMAIN = False

try:
    from .domains.grids import GridsGovernor
    _HAS_GRIDS_DOMAIN = True
except ImportError:
    _HAS_GRIDS_DOMAIN = False

try:
    from .domains.datacenter import DataCenterMonitor as DatacenterGovernor
    _HAS_DATACENTERS_DOMAIN = True
except ImportError:
    _HAS_DATACENTERS_DOMAIN = False

try:
    from .domains.automotive import AutonomyGovernor as AutomobilesGovernor
    _HAS_AUTOMOBILES_DOMAIN = True
except ImportError:
    _HAS_AUTOMOBILES_DOMAIN = False

try:
    from .domains.governance import (
        GovernanceGovernor,
        GovernanceConfig,
        GovernanceSignal,
        GovernanceDecision,
        GovernanceTrustLevel,
        AuthorityStatus,
        ScopeStatus,
        TemporalStatus,
        create_default_governor as create_governance_governor,
        create_strict_governor as create_strict_governance_governor,
        create_permissive_governor as create_permissive_governance_governor,
    )
    _HAS_GOVERNANCE_DOMAIN = True
except ImportError:
    _HAS_GOVERNANCE_DOMAIN = False

try:
    from .domains.cross_ecosystem import (
        CrossEcosystemGovernor,
        CrossEcosystemPolicy,
        CrossEcosystemSignals,
        TranslationTrustLevel,
        ConsentStatus,
        PrivacyBoundaries,
        ContextPreservation,
        EcosystemCapabilities,
        TranslationPath,
        GovernanceEvent,
        KNOWN_ECOSYSTEMS,
        create_default_governor as create_cross_ecosystem_governor,
        create_strict_governor as create_strict_cross_ecosystem_governor,
        create_permissive_governor as create_permissive_cross_ecosystem_governor,
        create_health_data_signals,
        create_wearable_continuity_signals,
        check_ecosystem_compatibility,
    )
    _HAS_CROSS_ECOSYSTEM_DOMAIN = True
except ImportError:
    _HAS_CROSS_ECOSYSTEM_DOMAIN = False

try:
    from .domains.neuro_assistive import (
        NeuroGovernor,
        NeuroAssistivePolicy,
        NeuroSignals,
        NeuroDecisionResult,
        NeuroEvent,
        CognitiveState,
        AssistanceLevel,
        AssistanceOutcome,
        ImpairmentCategory,
        ConsentStatus as NeuroConsentStatus,
        ConsentRecord,
        InterpretationResult,
        CognitiveLoadMetrics,
        SessionMetrics,
        TemporalSafeguards,
        AssistanceConstraints,
        PolicyEvaluator,
        CognitiveStateTracker,
        create_neuro_governor,
        validate_neuro_signals,
    )
    _HAS_NEURO_ASSISTIVE_DOMAIN = True
except ImportError:
    _HAS_NEURO_ASSISTIVE_DOMAIN = False

try:
    from .domains.auditory import (
        AuditoryGovernor,
        AuditoryGovernancePolicy,
        AuditorySignals,
        AuditoryDecision,
        AuditoryEvent,
        OutputAuthorizationLevel,
        ListeningMode,
        UserSafetyProfile,
        DecisionOutcome,
        RegulatoryGateResult,
        HearingProfile,
        EnvironmentMetrics,
        EnhancementMetrics,
        ComfortMetrics,
        DeviceHealth,
        AuditoryUncertainty,
        AuditoryDecisionDelta,
        AuditoryPolicyEvaluator,
        AuditoryUncertaintyCalculator,
        default_auditory_config,
        create_auditory_governor,
        validate_auditory_signals,
        AUDITORY_FLAG_SEVERITY,
    )
    _HAS_AUDITORY_DOMAIN = True
except ImportError:
    _HAS_AUDITORY_DOMAIN = False

# =============================================================================
# Metadata
# =============================================================================
__version__ = "2.0.0"
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

if _HAS_AI_INTEGRATIONS:
    __all__.extend([
        "AIResponse",
        "AIAdapter",
        "OpenAIAdapter",
        "AnthropicAdapter",
        "HuggingFaceAdapter",
        "LangChainAdapter",
        "MultiModelEnsemble",
        "create_openai_adapter",
        "create_anthropic_adapter",
        "create_huggingface_adapter",
        "create_langchain_adapter",
        "create_multi_model_ensemble",
    ])

# ---- Domains ----
if _HAS_IMAGING_DOMAIN:
    __all__.append("ImagingDomain")
if _HAS_ROBOTICS_DOMAIN:
    __all__.append("RoboticsDomain")
if _HAS_TELECOMMUNICATIONS_DOMAIN:
    __all__.append("TelecomGovernor")
if _HAS_OCEAN_DOMAIN:
    __all__.append("OceanGovernor")
if _HAS_GRIDS_DOMAIN:
    __all__.append("GridsGovernor")
if _HAS_DATACENTERS_DOMAIN:
    __all__.append("DatacenterGovernor")
if _HAS_AUTOMOBILES_DOMAIN:
    __all__.append("AutomobilesGovernor")
if _HAS_GOVERNANCE_DOMAIN:
    __all__.extend([
        "GovernanceGovernor",
        "GovernanceConfig",
        "GovernanceSignal",
        "GovernanceDecision",
        "GovernanceTrustLevel",
        "AuthorityStatus",
        "ScopeStatus",
        "TemporalStatus",
        "create_governance_governor",
        "create_strict_governance_governor",
        "create_permissive_governance_governor",
    ])
if _HAS_CROSS_ECOSYSTEM_DOMAIN:
    __all__.extend([
        "CrossEcosystemGovernor",
        "CrossEcosystemPolicy",
        "CrossEcosystemSignals",
        "TranslationTrustLevel",
        "ConsentStatus",
        "PrivacyBoundaries",
        "ContextPreservation",
        "EcosystemCapabilities",
        "TranslationPath",
        "GovernanceEvent",
        "KNOWN_ECOSYSTEMS",
        "create_cross_ecosystem_governor",
        "create_strict_cross_ecosystem_governor",
        "create_permissive_cross_ecosystem_governor",
        "create_health_data_signals",
        "create_wearable_continuity_signals",
        "check_ecosystem_compatibility",
    ])
if _HAS_NEURO_ASSISTIVE_DOMAIN:
    __all__.extend([
        "NeuroGovernor",
        "NeuroAssistivePolicy",
        "NeuroSignals",
        "NeuroDecisionResult",
        "NeuroEvent",
        "CognitiveState",
        "AssistanceLevel",
        "AssistanceOutcome",
        "ImpairmentCategory",
        "NeuroConsentStatus",
        "ConsentRecord",
        "InterpretationResult",
        "CognitiveLoadMetrics",
        "SessionMetrics",
        "TemporalSafeguards",
        "AssistanceConstraints",
        "PolicyEvaluator",
        "CognitiveStateTracker",
        "create_neuro_governor",
        "validate_neuro_signals",
    ])
if _HAS_AUDITORY_DOMAIN:
    __all__.extend([
        "AuditoryGovernor",
        "AuditoryGovernancePolicy",
        "AuditorySignals",
        "AuditoryDecision",
        "AuditoryEvent",
        "OutputAuthorizationLevel",
        "ListeningMode",
        "UserSafetyProfile",
        "DecisionOutcome",
        "RegulatoryGateResult",
        "HearingProfile",
        "EnvironmentMetrics",
        "EnhancementMetrics",
        "ComfortMetrics",
        "DeviceHealth",
        "AuditoryUncertainty",
        "AuditoryDecisionDelta",
        "AuditoryPolicyEvaluator",
        "AuditoryUncertaintyCalculator",
        "default_auditory_config",
        "create_auditory_governor",
        "validate_auditory_signals",
        "AUDITORY_FLAG_SEVERITY",
    ])

# =============================================================================
# Convenience
# =============================================================================
def create_pipeline(preset_name: str = "balanced", **overrides):
    """
    Create an AILEE Trust Pipeline with a configuration preset.
    
    Args:
        preset_name: Name of the configuration preset (e.g., "balanced", "conservative")
        **overrides: Optional configuration parameters to override
        
    Returns:
        AileeTrustPipeline: Configured pipeline instance
        
    Example:
        >>> pipeline = create_pipeline("conservative", accept_threshold=0.9)
        >>> result = pipeline.process(raw_value=0.85, raw_confidence=0.92)
    """
    if not _HAS_CONFIG_PRESETS:
        raise ImportError("Configuration presets not available")
    config = get_preset(preset_name)
    if overrides:
        cfg = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        cfg.update(overrides)
        config = AileeConfig(**cfg)
    return AileeTrustPipeline(config)


def get_available_domains():
    """
    Get list of available domain modules.
    
    Returns:
        dict: Dictionary mapping domain names to availability status
        
    Example:
        >>> domains = get_available_domains()
        >>> if domains['governance']:
        ...     from ailee import GovernanceGovernor
    """
    return {
        "imaging": _HAS_IMAGING_DOMAIN,
        "robotics": _HAS_ROBOTICS_DOMAIN,
        "telecommunications": _HAS_TELECOMMUNICATIONS_DOMAIN,
        "ocean": _HAS_OCEAN_DOMAIN,
        "grids": _HAS_GRIDS_DOMAIN,
        "datacenters": _HAS_DATACENTERS_DOMAIN,
        "automobiles": _HAS_AUTOMOBILES_DOMAIN,
        "governance": _HAS_GOVERNANCE_DOMAIN,
        "cross_ecosystem": _HAS_CROSS_ECOSYSTEM_DOMAIN,
        "neuro_assistive": _HAS_NEURO_ASSISTIVE_DOMAIN,
        "auditory": _HAS_AUDITORY_DOMAIN,
    }


def get_available_helpers():
    """
    Get list of available helper modules.
    
    Returns:
        dict: Dictionary mapping helper module names to availability status
        
    Example:
        >>> helpers = get_available_helpers()
        >>> if helpers['monitors']:
        ...     from ailee import TrustMonitor
    """
    return {
        "config_presets": _HAS_CONFIG_PRESETS,
        "peer_adapters": _HAS_PEER_ADAPTERS,
        "monitors": _HAS_MONITORS,
        "serialization": _HAS_SERIALIZATION,
        "replay": _HAS_REPLAY,
        "ai_integrations": _HAS_AI_INTEGRATIONS,
    }


def print_available_modules():
    """
    Print a summary of available AILEE modules.
    
    Example:
        >>> print_available_modules()
        AILEE Trust Layer v2.0.0
        
        Core Modules:
          ✓ Trust Pipeline
          ✓ Configuration
        ...
    """
    print(f"AILEE Trust Layer v{__version__}")
    print()
    print("Core Modules:")
    print("  ✓ Trust Pipeline")
    print("  ✓ Configuration")
    print()
    
    helpers = get_available_helpers()
    print("Helper Modules:")
    for name, available in helpers.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    print()
    
    domains = get_available_domains()
    print("Domain Modules:")
    for name, available in domains.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    print()


__all__.extend(["create_pipeline", "get_available_domains", "get_available_helpers", "print_available_modules"])

# Allow explicit access to optional modules and domains namespace
from . import optional
from . import domains

__all__.extend(["optional", "domains"])

from .ailee_client import AileeClient
__all__.append("AileeClient")

