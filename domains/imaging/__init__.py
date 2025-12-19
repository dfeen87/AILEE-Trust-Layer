"""
AILEE Trust Layer — IMAGING Domain

This package provides governance and validation utilities for
AI-assisted and computational imaging systems.

The IMAGING domain governs trust, quality, safety, and efficiency
of imaging outputs. It does NOT implement imaging physics,
acquisition hardware, or reconstruction algorithms.
"""

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
]
