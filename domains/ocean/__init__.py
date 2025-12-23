"""
OCEAN Domain — AILEE Trust Layer
Version: 1.0.1 - Production Grade

Governance and restraint framework for marine ecosystem monitoring
and environmental intervention decisions.

This domain provides:
- Staged authority levels for marine interventions
- Precautionary principle enforcement
- Uncertainty-aware decision ceilings
- Regulatory compliance validation
- Ecosystem regime tracking with hysteresis
- Complete audit trails for environmental compliance

Quick Start:
    >>> from ailee.domains.ocean import create_ocean_governor, OceanSignals, EcosystemType
    >>> 
    >>> governor = create_ocean_governor(
    ...     ecosystem_type=EcosystemType.CORAL_REEF,
    ...     precautionary_bias=0.85
    ... )
    >>> 
    >>> signals = OceanSignals(
    ...     proposed_action_trust_score=0.82,
    ...     ecosystem_health_index=0.75,
    ...     measurement_reliability=0.88,
    ...     ecosystem_type=EcosystemType.CORAL_REEF,
    ...     intervention_category=InterventionCategory.REEF_RESTORATION,
    ... )
    >>> 
    >>> decision = governor.evaluate(signals)
    >>> 
    >>> if decision.intervention_authorized:
    ...     print(f"Authorized: {decision.authority_level.value}")
    ...     print(f"Constraints: {decision.intervention_constraints}")

Primary entry point: OceanGovernor
Quick start with: create_ocean_governor()

For detailed documentation, see: https://github.com/your-org/ailee/docs/ocean
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
__author__ = "AILEE Project"
__license__ = "MIT"

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
    
    # Convenience functions (added below)
    "create_coral_reef_governor",
    "create_deep_sea_governor",
    "create_coastal_governor",
    "create_protected_area_governor",
    "is_intervention_safe",
    "get_authority_description",
    "export_compliance_report",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Factory Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_coral_reef_governor(**policy_overrides) -> OceanGovernor:
    """
    Create an OceanGovernor optimized for coral reef ecosystems.
    
    Coral reefs are highly sensitive ecosystems requiring:
    - Higher precautionary bias (0.90)
    - Extended observation periods (60 days)
    - Stricter risk thresholds
    - Enhanced regulatory compliance
    
    Args:
        **policy_overrides: Optional policy parameters to override defaults
        
    Returns:
        OceanGovernor: Configured for coral reef protection
        
    Example:
        >>> governor = create_coral_reef_governor()
        >>> signals = OceanSignals(
        ...     proposed_action_trust_score=0.88,
        ...     ecosystem_health_index=0.82,
        ...     measurement_reliability=0.90,
        ...     ecosystem_type=EcosystemType.CORAL_REEF,
        ... )
        >>> decision = governor.evaluate(signals)
    """
    defaults = {
        "min_intervention_safety_score": 0.85,
        "min_ecosystem_health_for_intervention": 0.50,
        "max_ecological_risk": 0.20,
        "min_reversibility_score": 0.70,
        "min_observation_period_days": 60.0,
        "min_trend_stability_days": 21.0,
        "require_regulatory_approval": True,
        "min_compliance_score": 0.98,
        "precautionary_bias": 0.90,
    }
    defaults.update(policy_overrides)
    
    return create_ocean_governor(
        ecosystem_type=EcosystemType.CORAL_REEF,
        **defaults
    )


def create_deep_sea_governor(**policy_overrides) -> OceanGovernor:
    """
    Create an OceanGovernor optimized for deep sea ecosystems.
    
    Deep sea ecosystems are slow-recovering and poorly understood:
    - Maximum precautionary bias (0.95)
    - Extended observation periods (90 days)
    - Very low risk tolerance
    - Long-term recovery assessment required
    
    Args:
        **policy_overrides: Optional policy parameters to override defaults
        
    Returns:
        OceanGovernor: Configured for deep sea protection
        
    Example:
        >>> governor = create_deep_sea_governor()
        >>> # Deep sea interventions require exceptional justification
    """
    defaults = {
        "min_intervention_safety_score": 0.90,
        "min_ecosystem_health_for_intervention": 0.60,
        "max_ecological_risk": 0.15,
        "min_reversibility_score": 0.80,
        "min_observation_period_days": 90.0,
        "min_trend_stability_days": 30.0,
        "require_regulatory_approval": True,
        "min_compliance_score": 0.99,
        "precautionary_bias": 0.95,
    }
    defaults.update(policy_overrides)
    
    return create_ocean_governor(
        ecosystem_type=EcosystemType.DEEP_SEA,
        **defaults
    )


def create_coastal_governor(**policy_overrides) -> OceanGovernor:
    """
    Create an OceanGovernor optimized for coastal zone ecosystems.
    
    Coastal zones balance human activity with ecosystem health:
    - Moderate precautionary bias (0.75)
    - Standard observation periods (30 days)
    - Balanced risk tolerance
    
    Args:
        **policy_overrides: Optional policy parameters to override defaults
        
    Returns:
        OceanGovernor: Configured for coastal zone management
        
    Example:
        >>> governor = create_coastal_governor()
        >>> # Suitable for estuaries, coastal waters, harbors
    """
    defaults = {
        "min_intervention_safety_score": 0.75,
        "min_ecosystem_health_for_intervention": 0.40,
        "max_ecological_risk": 0.30,
        "min_reversibility_score": 0.50,
        "min_observation_period_days": 30.0,
        "min_trend_stability_days": 14.0,
        "precautionary_bias": 0.75,
    }
    defaults.update(policy_overrides)
    
    return create_ocean_governor(
        ecosystem_type=EcosystemType.COASTAL_ZONE,
        **defaults
    )


def create_protected_area_governor(**policy_overrides) -> OceanGovernor:
    """
    Create an OceanGovernor optimized for Marine Protected Areas (MPAs).
    
    MPAs require strict governance regardless of ecosystem type:
    - High precautionary bias (0.88)
    - Mandatory regulatory approval
    - Extended observation periods
    - Enhanced compliance requirements
    
    Args:
        **policy_overrides: Optional policy parameters to override defaults
        
    Returns:
        OceanGovernor: Configured for MPA protection
        
    Example:
        >>> governor = create_protected_area_governor()
        >>> # Applies strict protections for designated MPAs
    """
    defaults = {
        "min_intervention_safety_score": 0.85,
        "min_ecosystem_health_for_intervention": 0.50,
        "max_ecological_risk": 0.25,
        "min_reversibility_score": 0.65,
        "min_observation_period_days": 45.0,
        "min_trend_stability_days": 21.0,
        "require_regulatory_approval": True,
        "min_compliance_score": 0.98,
        "precautionary_bias": 0.88,
    }
    defaults.update(policy_overrides)
    
    return create_ocean_governor(
        ecosystem_type=EcosystemType.MARINE_PROTECTED_AREA,
        **defaults
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def is_intervention_safe(
    decision: OceanDecisionResult,
    min_authority: AuthorityLevel = AuthorityLevel.CONTROLLED_INTERVENTION
) -> bool:
    """
    Check if intervention is authorized at the required authority level.
    
    Args:
        decision: Ocean governance decision to check
        min_authority: Minimum required authority level
        
    Returns:
        bool: True if intervention meets or exceeds required authority
        
    Example:
        >>> decision = governor.evaluate(signals)
        >>> if is_intervention_safe(decision, AuthorityLevel.FULL_INTERVENTION):
        ...     execute_unrestricted_intervention()
        >>> elif is_intervention_safe(decision):
        ...     execute_controlled_intervention()
    """
    if not decision.intervention_authorized:
        return False
    
    authority_hierarchy = {
        AuthorityLevel.NO_ACTION: 0,
        AuthorityLevel.OBSERVE_ONLY: 1,
        AuthorityLevel.STAGE_INTERVENTION: 2,
        AuthorityLevel.CONTROLLED_INTERVENTION: 3,
        AuthorityLevel.FULL_INTERVENTION: 4,
        AuthorityLevel.EMERGENCY_RESPONSE: 5,
    }
    
    return authority_hierarchy[decision.authority_level] >= authority_hierarchy[min_authority]


def get_authority_description(level: AuthorityLevel) -> str:
    """
    Get human-readable description of an authority level.
    
    Args:
        level: Authority level to describe
        
    Returns:
        str: Plain-language description
        
    Example:
        >>> desc = get_authority_description(AuthorityLevel.CONTROLLED_INTERVENTION)
        >>> print(desc)
    """
    descriptions = {
        AuthorityLevel.NO_ACTION: (
            "No action permitted. Intervention is prohibited due to unacceptable "
            "risks or policy violations."
        ),
        AuthorityLevel.OBSERVE_ONLY: (
            "Observation only. Continue monitoring without any intervention actions. "
            "Collect data to improve future assessments."
        ),
        AuthorityLevel.STAGE_INTERVENTION: (
            "Staging approved. Prepare intervention resources and complete pre-deployment "
            "requirements, but do not execute intervention until further authorization."
        ),
        AuthorityLevel.CONTROLLED_INTERVENTION: (
            "Controlled intervention authorized. Execute intervention with continuous "
            "monitoring, defined constraints, and readiness for immediate abort if needed."
        ),
        AuthorityLevel.FULL_INTERVENTION: (
            "Full intervention authorized. Execute intervention within approved parameters "
            "with standard monitoring and compliance requirements."
        ),
        AuthorityLevel.EMERGENCY_RESPONSE: (
            "Emergency intervention authorized. Ecosystem crisis detected - execute "
            "immediate response with enhanced monitoring and mandatory post-action review."
        ),
    }
    return descriptions.get(level, "Unknown authority level")


def get_ecosystem_description(ecosystem_type: EcosystemType) -> str:
    """
    Get description of ecosystem characteristics and sensitivities.
    
    Args:
        ecosystem_type: Type of ecosystem
        
    Returns:
        str: Description of ecosystem characteristics
        
    Example:
        >>> desc = get_ecosystem_description(EcosystemType.CORAL_REEF)
        >>> print(desc)
    """
    descriptions = {
        EcosystemType.CORAL_REEF: (
            "Coral reefs are highly diverse but fragile ecosystems. Extremely sensitive "
            "to temperature, pH, and water quality changes. Slow recovery from damage."
        ),
        EcosystemType.DEEP_SEA: (
            "Deep sea ecosystems are poorly understood and extremely slow to recover. "
            "Organisms adapted to stable conditions with minimal disturbance tolerance."
        ),
        EcosystemType.COASTAL_ZONE: (
            "Coastal zones experience natural variability and human activity. More "
            "resilient than reefs but still sensitive to cumulative impacts."
        ),
        EcosystemType.ESTUARY: (
            "Estuaries are transition zones with variable salinity and high productivity. "
            "Critical nursery habitat for many species."
        ),
        EcosystemType.KELP_FOREST: (
            "Kelp forests are productive ecosystems providing habitat and food. Sensitive "
            "to temperature and nutrient changes."
        ),
        EcosystemType.MANGROVE: (
            "Mangroves protect coastlines and provide nursery habitat. Sensitive to "
            "salinity changes and physical disturbance."
        ),
        EcosystemType.MARINE_PROTECTED_AREA: (
            "Marine Protected Areas have special conservation status requiring strict "
            "governance and precautionary approach to any interventions."
        ),
        EcosystemType.POLAR: (
            "Polar ecosystems are highly sensitive to climate change with limited "
            "resilience. Changes propagate through entire food webs."
        ),
    }
    return descriptions.get(ecosystem_type, "Unknown ecosystem type")


def export_compliance_report(
    events: list[OceanEvent],
    format: str = "text"
) -> str:
    """
    Export ocean governance events as compliance report.
    
    Args:
        events: List of ocean governance events
        format: Output format ("text" or "csv")
        
    Returns:
        str: Formatted compliance report
        
    Example:
        >>> events = governor.export_events(since_ts=start_time)
        >>> report = export_compliance_report(events, format="text")
        >>> with open("compliance_report.txt", "w") as f:
        ...     f.write(report)
    """
    if format == "csv":
        lines = ["timestamp,event_type,ecosystem_type,intervention_category,authority_level,"
                "decision_outcome,trust_score,health_index,measurement_reliability"]
        
        for e in events:
            lines.append(
                f"{e.timestamp},{e.event_type},{e.ecosystem_type.value},"
                f"{e.intervention_category.value},{e.authority_level.value},"
                f"{e.decision_outcome.value},{e.proposed_action_trust_score:.3f},"
                f"{e.ecosystem_health_index:.3f},{e.measurement_reliability:.3f}"
            )
        
        return "\n".join(lines)
    
    else:  # text format
        lines = ["=" * 80]
        lines.append("OCEAN GOVERNANCE COMPLIANCE REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total Events: {len(events)}")
        lines.append("")
        
        # Summary statistics
        interventions = [e for e in events if e.decision_outcome == DecisionOutcome.INTERVENTION_AUTHORIZED]
        emergencies = [e for e in events if e.decision_outcome == DecisionOutcome.EMERGENCY_APPROVED]
        prohibitions = [e for e in events if e.decision_outcome == DecisionOutcome.PROHIBIT]
        
        lines.append("EVENT SUMMARY:")
        lines.append(f"  Interventions Authorized: {len(interventions)}")
        lines.append(f"  Emergency Responses: {len(emergencies)}")
        lines.append(f"  Interventions Prohibited: {len(prohibitions)}")
        lines.append(f"  Observations Only: {len([e for e in events if e.decision_outcome == DecisionOutcome.OBSERVE_CONTINUE])}")
        lines.append("")
        
        # Detailed events
        lines.append("DETAILED EVENT LOG:")
        lines.append("-" * 80)
        
        for e in events:
            lines.append(f"Timestamp: {e.timestamp}")
            lines.append(f"  Event: {e.event_type}")
            lines.append(f"  Ecosystem: {e.ecosystem_type.value}")
            lines.append(f"  Intervention: {e.intervention_category.value}")
            lines.append(f"  Authority Level: {e.authority_level.value}")
            lines.append(f"  Decision: {e.decision_outcome.value}")
            lines.append(f"  Trust Score: {e.proposed_action_trust_score:.3f}")
            lines.append(f"  Ecosystem Health: {e.ecosystem_health_index:.3f}")
            
            if e.reasons:
                lines.append(f"  Reasons:")
                for reason in e.reasons[:3]:
                    lines.append(f"    - {reason}")
            
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


ECOSYSTEM_SENSITIVITY_RANKING = [
    EcosystemType.DEEP_SEA,           # Highest sensitivity
    EcosystemType.CORAL_REEF,
    EcosystemType.POLAR,
    EcosystemType.MARINE_PROTECTED_AREA,
    EcosystemType.KELP_FOREST,
    EcosystemType.MANGROVE,
    EcosystemType.SEAGRASS_MEADOW,
    EcosystemType.ESTUARY,
    EcosystemType.COASTAL_ZONE,
    EcosystemType.OPEN_OCEAN,
    EcosystemType.AQUACULTURE_ZONE,  # Lowest sensitivity
]

CRITICAL_THRESHOLDS = {
    "dissolved_oxygen_critical_mg_l": 2.0,
    "dissolved_oxygen_warning_mg_l": 4.0,
    "ph_acidification_critical": 7.5,
    "ph_alkalinity_high": 8.5,
    "chlorophyll_bloom_threshold_ug_l": 50.0,
    "health_index_critical": 0.30,
    "health_index_degraded": 0.50,
}

INTERVENTION_RISK_CATEGORIES = {
    "low_risk": ["OBSERVATION_ONLY", "NUTRIENT_MANAGEMENT"],
    "moderate_risk": ["OXYGENATION", "SEDIMENT_MANAGEMENT", "POLLUTION_REMEDIATION"],
    "high_risk": ["ALKALINITY_ENHANCEMENT", "TEMPERATURE_MODIFICATION", "REEF_RESTORATION"],
    "very_high_risk": ["SPECIES_INTRODUCTION", "HARMFUL_ALGAE_MITIGATION"],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Package Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__doc_url__ = "https://github.com/your-org/ailee/docs/domains/ocean"
__source_url__ = "https://github.com/your-org/ailee"
__bug_tracker_url__ = "https://github.com/your-org/ailee/issues"

__description__ = (
    "AILEE Ocean Domain: Governance and restraint framework for marine ecosystem "
    "monitoring and environmental intervention decisions with precautionary principle "
    "enforcement and uncertainty-aware authority ceilings."
)
