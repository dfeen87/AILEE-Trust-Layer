"""
AILEE Trust Layer — IMAGING Domain

This package provides governance and validation utilities for
AI-assisted and computational imaging systems.

The IMAGING domain governs trust, quality, safety, and efficiency
of imaging outputs. It does NOT implement imaging physics,
acquisition hardware, or reconstruction algorithms.
"""


from .imaging import (
    get_health,
    get_subsystem_health,
    get_metrics,
    get_events,
    get_decision_history,
    create_strict_governor,
    create_permissive_governor,
    create_default_governor,
    validate_imaging_signals,
    ImagingTrustLevel,
    ImagingHealthStatus,
    ImagingControlDomain,
    ImagingControlAction,
)
from .imaging import (
    ImagingDomain,
    ImagingGovernor,
    ImagingGovernancePolicy,
    ImagingDecisionResult,
    ImagingSignals,
    ImagingEvent,
    ImagingModality,
    ReconstructionMethod,
    QualityDecision,
    AcquisitionParams,
    NoiseModel,
    ArtifactAssessment,
    ReconstructionResult,
    AdaptiveStrategy,
    default_imaging_config,
)

__all__ = [
    "ImagingDomain",
    "ImagingGovernor",
    "ImagingGovernancePolicy",
    "ImagingDecisionResult",
    "ImagingSignals",
    "ImagingEvent",
    "ImagingModality",
    "ReconstructionMethod",
    "QualityDecision",
    "AcquisitionParams",
    "NoiseModel",
    "ArtifactAssessment",
    "ReconstructionResult",
    "AdaptiveStrategy",
    "default_imaging_config",
    "get_health",
    "get_subsystem_health",
    "get_metrics",
    "get_events",
    "get_decision_history",
    "create_strict_governor",
    "create_permissive_governor",
    "create_default_governor",
    "validate_imaging_signals",
    "ImagingTrustLevel",
    "ImagingHealthStatus",
    "ImagingControlDomain",
    "ImagingControlAction",
]

__version__ = "4.2.0"
__doc_url__ = "https://github.com/dfeen87/ailee-trust-layer"
__source_url__ = "https://github.com/dfeen87/ailee-trust-layer"
__bug_tracker_url__ = "https://github.com/dfeen87/ailee-trust-layer/issues"
