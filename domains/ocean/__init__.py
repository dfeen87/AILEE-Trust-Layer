"""
OCEAN Domain — AILEE Trust Layer
Version: 1.0.1 - Production Grade

Governance and restraint framework for marine ecosystem monitoring
and environmental intervention decisions.

Primary entry point: OceanGovernor
Quick start with: create_ocean_governor()
"""

from .ocean import (
    # === Primary API ===
    OceanGovernor,
    create_ocean_governor,
    validate_ocean_signals,
    default_ocean_config,
    
    # === Core Data Structures ===
    OceanSignals,
    OceanDecisionResult,
    OceanEvent,
    
    # === Configuration ===
    OceanGovernancePolicy,
    
    # === Enumerations ===
    EcosystemType,
    EcosystemRegime,
    InterventionCategory,
    AuthorityLevel,
    DecisionOutcome,
    RegulatoryGateResult,
    
    # === Assessment Components ===
    EcosystemHealth,
    RiskAssessment,
    RegulatoryStatus,
    TemporalContext,
    AggregateUncertainty,
    
    # === Supporting Types ===
    OceanSensorReading,
    ModelPrediction,
    StagingRequirements,
    DecisionDelta,
    
    # === Analysis Components (Advanced) ===
    PolicyEvaluator,
    UncertaintyCalculator,
    RegimeTracker,
    
    # === Constants ===
    FLAG_SEVERITY,
)

__version__ = "1.0.1"

__all__ = [
    # Primary API
    "OceanGovernor",
    "create_ocean_governor",
    "validate_ocean_signals",
    "default_ocean_config",
    
    # Core structures
    "OceanSignals",
    "OceanDecisionResult",
    "OceanEvent",
    "OceanGovernancePolicy",
    
    # Enums
    "EcosystemType",
    "EcosystemRegime",
    "InterventionCategory",
    "AuthorityLevel",
    "DecisionOutcome",
    "RegulatoryGateResult",
    
    # Assessment components
    "EcosystemHealth",
    "RiskAssessment",
    "RegulatoryStatus",
    "TemporalContext",
    "AggregateUncertainty",
    
    # Supporting types
    "OceanSensorReading",
    "ModelPrediction",
    "StagingRequirements",
    "DecisionDelta",
    
    # Advanced (optional)
    "PolicyEvaluator",
    "UncertaintyCalculator",
    "RegimeTracker",
    
    # Constants
    "FLAG_SEVERITY",
]
