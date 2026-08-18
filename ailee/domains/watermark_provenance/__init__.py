# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
AILEE Trust Layer — Watermark-Provenance-Governance Domain
Version: 5.0.0

Governance domain evaluating, interpreting, and contextualizing AI watermark signals
(e.g., SynthID-Text, Claude watermarking) across real workflows. Treats watermark detection
as one fragment of provenance, not a verdict on authorship.
"""

from .watermark_provenance import (
    ProvenanceQualifier,
    WatermarkProvenanceTrustLevel,
    WatermarkProvenanceHealthStatus,
    WatermarkProvenanceControlDomain,
    WatermarkProvenanceControlAction,
    WatermarkProvenancePolicy,
    ProvenanceEventNode,
    WatermarkSignalData,
    WatermarkProvenanceSignals,
    WatermarkProvenanceDecision,
    WatermarkProvenanceEvent,
    WatermarkProvenanceGovernor,
    get_health,
    get_subsystem_health,
    get_metrics,
    get_events,
    get_decision_history,
    create_watermark_provenance_governor,
    create_default_governor,
    create_strict_governor,
    create_permissive_governor,
    validate_watermark_provenance_signals,
)

from .wrapper import WatermarkProvenanceWrapper, create_watermark_provenance_wrapper

__all__ = [
    "ProvenanceQualifier",
    "WatermarkProvenanceTrustLevel",
    "WatermarkProvenanceHealthStatus",
    "WatermarkProvenanceControlDomain",
    "WatermarkProvenanceControlAction",
    "WatermarkProvenancePolicy",
    "ProvenanceEventNode",
    "WatermarkSignalData",
    "WatermarkProvenanceSignals",
    "WatermarkProvenanceDecision",
    "WatermarkProvenanceEvent",
    "WatermarkProvenanceGovernor",
    "WatermarkProvenanceWrapper",
    "get_health",
    "get_subsystem_health",
    "get_metrics",
    "get_events",
    "get_decision_history",
    "create_watermark_provenance_governor",
    "create_default_governor",
    "create_strict_governor",
    "create_permissive_governor",
    "create_watermark_provenance_wrapper",
    "validate_watermark_provenance_signals",
]
