"""
AILEE Trust Layer — OCEAN Domain
Version: 1.0.1 - Production Grade

Enhancements in v1.0.1:
- Fixed: OceanEvent field name mismatch (intervention_safety_score → proposed_action_trust_score)
- Added: Uncertainty-aware authority ceilings
- Added: Severity-weighted precautionary flags
- Added: Regime transition hysteresis
- Added: Explicit regulatory HOLD vs FAIL distinction
- Added: DecisionDelta tracking for monitoring
- Renamed: validated_safety_score → validated_trust_score for consistency

Ocean governance domain for marine ecosystem interventions, monitoring decisions,
and environmental management actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time
import statistics
import math


# ---- Core imports ----
try:
    from ailee_trust_pipeline_v1 import (
        AileeTrustPipeline,
        AileeConfig,
        DecisionResult,
        SafetyStatus
    )
except Exception:
    AileeTrustPipeline = None
    AileeConfig = None
    DecisionResult = None
    SafetyStatus = None


# ===== SEVERITY WEIGHTING FOR FLAGS =====

FLAG_SEVERITY: Dict[str, float] = {
    "regulatory_noncompliance": 0.10,
    "near_irreversible_intervention": 0.08,
    "aggregate_uncertainty_excessive": 0.07,
    "ecosystem_crisis_emergency_authorized": 0.06,
    "critical_thresholds_breached": 0.06,
    "model_disagreement": 0.05,
    "measurement_uncertainty_high": 0.04,
    "risk_profile_unacceptable": 0.04,
    "observation_period_insufficient": 0.03,
    "trend_instability": 0.03,
    "intervention_frequency_excessive": 0.03,
}


# -----------------------------
# Ecosystem and Intervention Types
# -----------------------------

class EcosystemRegime(str, Enum):
    """Ecosystem regime states for temporal memory tracking"""
    STABLE_HEALTHY = "STABLE_HEALTHY"
    STABLE_DEGRADED = "STABLE_DEGRADED"
    IMPROVING = "IMPROVING"
    DEGRADING = "DEGRADING"
    POST_INTERVENTION = "POST_INTERVENTION"
    DISTURBANCE = "DISTURBANCE"
    SEASONAL_VARIATION = "SEASONAL_VARIATION"
    RECOVERY = "RECOVERY"
    COLLAPSE = "COLLAPSE"
    UNKNOWN = "UNKNOWN"


class EcosystemType(str, Enum):
    """Marine ecosystem categories with different sensitivities"""
    OPEN_OCEAN = "OPEN_OCEAN"
    COASTAL_ZONE = "COASTAL_ZONE"
    ESTUARY = "ESTUARY"
    CORAL_REEF = "CORAL_REEF"
    KELP_FOREST = "KELP_FOREST"
    MANGROVE = "MANGROVE"
    SEAGRASS_MEADOW = "SEAGRASS_MEADOW"
    DEEP_SEA = "DEEP_SEA"
    POLAR = "POLAR"
    MARINE_PROTECTED_AREA = "MARINE_PROTECTED_AREA"
    AQUACULTURE_ZONE = "AQUACULTURE_ZONE"
    UNKNOWN = "UNKNOWN"


class InterventionCategory(str, Enum):
    """Types of ocean interventions"""
    NUTRIENT_MANAGEMENT = "NUTRIENT_MANAGEMENT"
    OXYGENATION = "OXYGENATION"
    ALKALINITY_ENHANCEMENT = "ALKALINITY_ENHANCEMENT"
    SPECIES_INTRODUCTION = "SPECIES_INTRODUCTION"
    HARMFUL_ALGAE_MITIGATION = "HARMFUL_ALGAE_MITIGATION"
    SEDIMENT_MANAGEMENT = "SEDIMENT_MANAGEMENT"
    TEMPERATURE_MODIFICATION = "TEMPERATURE_MODIFICATION"
    REEF_RESTORATION = "REEF_RESTORATION"
    POLLUTION_REMEDIATION = "POLLUTION_REMEDIATION"
    OBSERVATION_ONLY = "OBSERVATION_ONLY"
    UNKNOWN = "UNKNOWN"


class AuthorityLevel(str, Enum):
    """Escalating intervention authority levels (staged decision ladder)"""
    NO_ACTION = "NO_ACTION"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    STAGE_INTERVENTION = "STAGE_INTERVENTION"
    CONTROLLED_INTERVENTION = "CONTROLLED_INTERVENTION"
    FULL_INTERVENTION = "FULL_INTERVENTION"
    EMERGENCY_RESPONSE = "EMERGENCY_RESPONSE"


class DecisionOutcome(str, Enum):
    """Ocean governance decision outcomes"""
    PROHIBIT = "PROHIBIT"
    OBSERVE_CONTINUE = "OBSERVE_CONTINUE"
    STAGE_APPROVED = "STAGE_APPROVED"
    INTERVENTION_AUTHORIZED = "INTERVENTION_AUTHORIZED"
    ESCALATE_HUMAN_REVIEW = "ESCALATE_HUMAN_REVIEW"
    EMERGENCY_APPROVED = "EMERGENCY_APPROVED"
    REGULATORY_HOLD = "REGULATORY_HOLD"


class RegulatoryGateResult(str, Enum):
    """Distinct regulatory outcomes"""
    PASS = "PASS"
    HOLD = "HOLD"     # Not yet allowed (pending)
    FAIL = "FAIL"     # Not allowed (denied)


# -----------------------------
# Ecosystem Health State
# -----------------------------

@dataclass(frozen=True)
class EcosystemHealth:
    """Current health indicators for the target ecosystem"""
    overall_health_index: float
    
    dissolved_oxygen_mg_l: Optional[float] = None
    ph: Optional[float] = None
    temperature_c: Optional[float] = None
    salinity_ppt: Optional[float] = None
    turbidity_ntu: Optional[float] = None
    chlorophyll_a_ug_l: Optional[float] = None
    nutrient_concentration_umol: Optional[float] = None
    
    species_richness_score: Optional[float] = None
    keystone_species_present: bool = True
    
    trend_direction: Optional[str] = None
    trend_confidence: Optional[float] = None
    
    measurement_staleness_hours: Optional[float] = None
    spatial_coverage_score: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_critical_threshold_breached(self) -> Tuple[bool, List[str]]:
        """Check if any critical ecological thresholds are breached"""
        issues: List[str] = []
        
        if self.dissolved_oxygen_mg_l is not None:
            if self.dissolved_oxygen_mg_l < 2.0:
                issues.append(f"hypoxia_critical (DO={self.dissolved_oxygen_mg_l:.1f} mg/L < 2.0)")
            elif self.dissolved_oxygen_mg_l < 4.0:
                issues.append(f"hypoxia_warning (DO={self.dissolved_oxygen_mg_l:.1f} mg/L < 4.0)")
        
        if self.ph is not None:
            if self.ph < 7.5:
                issues.append(f"acidification_critical (pH={self.ph:.2f} < 7.5)")
            elif self.ph > 8.5:
                issues.append(f"alkalinity_high (pH={self.ph:.2f} > 8.5)")
        
        if self.chlorophyll_a_ug_l is not None:
            if self.chlorophyll_a_ug_l > 50.0:
                issues.append(f"algal_bloom_likely (Chl-a={self.chlorophyll_a_ug_l:.1f} > 50 ug/L)")
        
        if self.overall_health_index < 0.30:
            issues.append(f"ecosystem_health_critical (index={self.overall_health_index:.2f})")
        
        return len(issues) > 0, issues


# -----------------------------
# Risk Assessment
# -----------------------------

@dataclass(frozen=True)
class RiskAssessment:
    """Comprehensive risk evaluation for proposed intervention"""
    reversibility_score: float
    ecological_risk_score: float
    
    cascade_risk_score: Optional[float] = None
    tipping_point_proximity: Optional[float] = None
    
    off_target_effects_score: Optional[float] = None
    temporal_persistence_years: Optional[float] = None
    spatial_extent_km2: Optional[float] = None
    
    model_uncertainty: Optional[float] = None
    measurement_uncertainty: Optional[float] = None
    
    similar_interventions_success_rate: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_acceptable_risk(
        self,
        max_ecological_risk: float = 0.30,
        min_reversibility: float = 0.50,
    ) -> Tuple[bool, str]:
        """Evaluate if risk profile is acceptable for intervention"""
        
        if self.ecological_risk_score > max_ecological_risk:
            return False, f"ecological_risk={self.ecological_risk_score:.2f} exceeds {max_ecological_risk}"
        
        if self.reversibility_score < min_reversibility:
            return False, f"reversibility={self.reversibility_score:.2f} below {min_reversibility}"
        
        if self.cascade_risk_score is not None and self.cascade_risk_score > 0.60:
            return False, f"cascade_risk={self.cascade_risk_score:.2f} too high"
        
        if self.tipping_point_proximity is not None and self.tipping_point_proximity > 0.80:
            return False, f"tipping_point_proximity={self.tipping_point_proximity:.2f} dangerously close"
        
        return True, "risk_acceptable"


# -----------------------------
# Regulatory Status
# -----------------------------

@dataclass(frozen=True)
class RegulatoryStatus:
    """Regulatory compliance and permit status"""
    permit_status: str
    compliance_score: float
    
    environmental_impact_assessment_complete: bool = False
    public_consultation_complete: bool = False
    endangered_species_clearance: bool = True
    marine_protected_area_clearance: bool = True
    
    national_agency_approval: Optional[str] = None
    international_treaty_compliance: bool = True
    
    monitoring_requirements: Optional[List[str]] = None
    reporting_frequency_days: Optional[int] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_regulatory_compliant(self) -> Tuple[bool, List[str]]:
        """Check if regulatory requirements are met"""
        issues: List[str] = []
        
        if self.permit_status == "denied":
            issues.append("permit_denied")
        elif self.permit_status == "pending_review":
            issues.append("permit_pending")
        
        if self.compliance_score < 0.90:
            issues.append(f"compliance_score={self.compliance_score:.2f} below 0.90")
        
        if not self.environmental_impact_assessment_complete:
            issues.append("eia_incomplete")
        
        if not self.endangered_species_clearance:
            issues.append("endangered_species_conflict")
        
        if not self.marine_protected_area_clearance:
            issues.append("mpa_violation")
        
        return len(issues) == 0, issues


# -----------------------------
# Temporal Context
# -----------------------------

@dataclass(frozen=True)
class TemporalContext:
    """Time-related context for decision staging"""
    observation_period_days: float
    trend_stability_days: Optional[float] = None
    trend_confidence: Optional[float] = None  # ADDED: Confidence in trend direction [0-1]
    
    last_intervention_date: Optional[float] = None
    time_since_last_intervention_days: Optional[float] = None
    
    seasonal_appropriateness: Optional[str] = None
    
    forecast_horizon_days: Optional[int] = None
    forecast_confidence: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Sensor Reading
# -----------------------------

@dataclass(frozen=True)
class OceanSensorReading:
    """Individual oceanographic sensor reading"""
    parameter: str
    value: float
    unit: str
    confidence: float
    
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_m: Optional[float] = None
    
    sensor_type: Optional[str] = None
    timestamp: Optional[float] = None
    quality_flags: Optional[List[str]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Multi-Model Validation
# -----------------------------

@dataclass(frozen=True)
class ModelPrediction:
    """Prediction from oceanographic model or expert system"""
    model_name: str
    proposed_action_trust_score: float
    confidence: float
    
    forecast_outcome: Optional[str] = None
    expected_impact_magnitude: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Domain Inputs
# -----------------------------

@dataclass(frozen=True)
class OceanSignals:
    """Governance signals for ocean intervention assessment"""
    proposed_action_trust_score: float
    
    ecosystem_health_index: float
    ecosystem_health: Optional[EcosystemHealth] = None
    
    measurement_reliability: float
    
    sensor_readings: Tuple[OceanSensorReading, ...] = ()
    
    model_predictions: Tuple[ModelPrediction, ...] = ()
    
    risk_assessment: Optional[RiskAssessment] = None
    
    regulatory_status: Optional[RegulatoryStatus] = None
    
    temporal_context: Optional[TemporalContext] = None
    
    ecosystem_type: EcosystemType = EcosystemType.UNKNOWN
    intervention_category: InterventionCategory = InterventionCategory.UNKNOWN
    intervention_id: Optional[str] = None
    
    current_regime: EcosystemRegime = EcosystemRegime.UNKNOWN
    
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())


# -----------------------------
# Uncertainty Aggregation
# -----------------------------

@dataclass(frozen=True)
class AggregateUncertainty:
    """Explicit aggregation of all uncertainty sources"""
    aggregate_uncertainty_score: float
    
    measurement_uncertainty: float
    model_disagreement: float
    temporal_instability: float
    risk_uncertainty: float
    
    dominant_uncertainty_source: str
    uncertainty_sources: Dict[str, float] = field(default_factory=dict)
    
    def is_uncertainty_acceptable(self, max_aggregate: float = 0.40) -> Tuple[bool, str]:
        """Check if aggregate uncertainty is within acceptable bounds"""
        if self.aggregate_uncertainty_score > max_aggregate:
            return False, (
                f"aggregate_uncertainty={self.aggregate_uncertainty_score:.2f} "
                f"exceeds {max_aggregate:.2f} (source: {self.dominant_uncertainty_source})"
            )
        return True, "uncertainty_acceptable"


# ===== DECISION DELTA TRACKING =====

@dataclass(frozen=True)
class DecisionDelta:
    """
    Change tracking since last decision.
    
    Critical for:
    - Real-time monitoring dashboards
    - Post-hoc audits
    - Trend detection
    - System introspection
    """
    trust_score_delta: float
    ecosystem_health_delta: float
    uncertainty_delta: float
    measurement_reliability_delta: float
    
    regime_changed: bool
    authority_level_changed: bool
    
    time_since_last_decision_seconds: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Staging Requirements
# -----------------------------

@dataclass
class StagingRequirements:
    """Requirements before intervention can proceed to next stage"""
    required_observation_days: Optional[int] = None
    required_sensor_types: Optional[List[str]] = None
    required_model_agreement: Optional[float] = None
    required_regulatory_approvals: Optional[List[str]] = None
    required_stakeholder_consultations: Optional[List[str]] = None
    
    environmental_conditions: Optional[List[str]] = None
    seasonal_windows: Optional[List[str]] = None
    
    reason: str = ""


# -----------------------------
# Ocean Events (Compliance Logging)
# -----------------------------

@dataclass(frozen=True)
class OceanEvent:
    """Structured event for environmental compliance and scientific logging"""
    timestamp: float
    event_type: str
    
    ecosystem_type: EcosystemType
    intervention_category: InterventionCategory
    authority_level: AuthorityLevel
    decision_outcome: DecisionOutcome
    
    proposed_action_trust_score: float  # FIXED: was intervention_safety_score
    ecosystem_health_index: float
    measurement_reliability: float
    
    reasons: List[str]
    
    ecosystem_regime: EcosystemRegime = EcosystemRegime.UNKNOWN
    
    ecosystem_health: Optional[EcosystemHealth] = None
    risk_assessment: Optional[RiskAssessment] = None
    regulatory_status: Optional[RegulatoryStatus] = None
    temporal_context: Optional[TemporalContext] = None
    aggregate_uncertainty: Optional[AggregateUncertainty] = None
    
    ailee_decision: Optional[DecisionResult] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Domain Configuration
# -----------------------------

@dataclass(frozen=True)
class OceanGovernancePolicy:
    """Domain policy for ocean intervention governance"""
    ecosystem_type: EcosystemType = EcosystemType.UNKNOWN
    intervention_category: InterventionCategory = InterventionCategory.UNKNOWN
    
    precautionary_bias: float = 0.80
    
    min_intervention_safety_score: float = 0.75
    min_ecosystem_health_for_intervention: float = 0.40
    min_measurement_reliability: float = 0.70
    
    max_ecological_risk: float = 0.30
    min_reversibility_score: float = 0.50
    max_cascade_risk: float = 0.60
    
    min_observation_period_days: float = 30.0
    min_trend_stability_days: float = 14.0
    min_time_between_interventions_days: float = 90.0
    
    require_model_consensus: bool = True
    min_model_agreement_ratio: float = 0.75
    
    require_regulatory_approval: bool = True
    min_compliance_score: float = 0.95
    
    observe_only_threshold: float = 0.60
    stage_intervention_threshold: float = 0.75
    controlled_intervention_threshold: float = 0.85
    full_intervention_threshold: float = 0.95
    
    enable_emergency_override: bool = True
    emergency_health_threshold: float = 0.20
    
    max_event_log_size: int = 10000
    
    metadata: Dict[str, Any] = field(default_factory=dict)


def default_ocean_config(ecosystem_type: EcosystemType) -> "AileeConfig":
    """Safe defaults for ocean governance pipeline configuration"""
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    cfg = AileeConfig(
        accept_threshold=0.85,
        borderline_low=0.70,
        borderline_high=0.85,
        
        w_stability=0.50,
        w_agreement=0.35,
        w_likelihood=0.15,
        
        history_window=200,
        forecast_window=30,
        
        grace_peer_delta=0.12,
        grace_min_peer_agreement_ratio=0.70,
        grace_forecast_epsilon=0.15,
        grace_max_abs_z=2.0,
        
        consensus_quorum=3,
        consensus_delta=0.15,
        consensus_pass_ratio=0.75,
        
        fallback_mode="last_good",
        
        enable_grace=True,
        enable_consensus=True,
        enable_audit_metadata=True,
    )
    
    if ecosystem_type in (EcosystemType.CORAL_REEF, EcosystemType.DEEP_SEA):
        cfg.accept_threshold = 0.90
        cfg.w_stability = 0.55
        cfg.history_window = 300
        cfg.grace_peer_delta = 0.10
    
    elif ecosystem_type in (EcosystemType.POLAR, EcosystemType.MARINE_PROTECTED_AREA):
        cfg.accept_threshold = 0.88
        cfg.w_stability = 0.52
        cfg.consensus_quorum = 4
    
    elif ecosystem_type in (EcosystemType.COASTAL_ZONE, EcosystemType.ESTUARY):
        cfg.accept_threshold = 0.82
        cfg.w_stability = 0.48
        cfg.history_window = 150
    
    elif ecosystem_type == EcosystemType.AQUACULTURE_ZONE:
        cfg.accept_threshold = 0.80
        cfg.w_stability = 0.45
    
    return cfg


# -----------------------------
# Result Structure
# -----------------------------

@dataclass(frozen=True)
class OceanDecisionResult:
    """Ocean governance decision result"""
    intervention_authorized: bool
    authority_level: AuthorityLevel
    decision_outcome: DecisionOutcome
    
    validated_trust_score: float  # RENAMED: was validated_safety_score
    confidence_score: float
    
    recommendation: str
    reasons: List[str]
    
    staging_requirements: Optional[StagingRequirements] = None
    intervention_constraints: Optional[Dict[str, Any]] = None
    
    ailee_result: Optional[DecisionResult] = None
    
    risk_level: Optional[str] = None
    precautionary_flags: Optional[List[str]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== ENHANCED POLICY EVALUATOR =====

class PolicyEvaluator:
    """
    Enhanced policy evaluator with explicit regulatory distinction
    and severity-weighted precautionary flags.
    """
    
    def __init__(self, policy: OceanGovernancePolicy):
        self.policy = policy
    
    def check_health_gates(
        self,
        signals: OceanSignals
    ) -> Tuple[bool, List[str], bool]:
        """Check ecosystem health gates"""
        issues = []
        is_emergency = False
        
        if not signals.ecosystem_health:
            return True, [], False
        
        health = signals.ecosystem_health
        
        critical, health_issues = health.is_critical_threshold_breached()
        if critical:
            issues.extend(health_issues)
            
            if (self.policy.enable_emergency_override and
                health.overall_health_index < self.policy.emergency_health_threshold):
                is_emergency = True
        
        if health.overall_health_index < self.policy.min_ecosystem_health_for_intervention:
            issues.append(
                f"Ecosystem too degraded (health={health.overall_health_index:.2f} "
                f"< {self.policy.min_ecosystem_health_for_intervention})"
            )
        
        passed = len(issues) == 0
        return passed, issues, is_emergency
    
    def check_risk_gates(
        self,
        signals: OceanSignals
    ) -> Tuple[bool, List[str], float]:
        """Check risk assessment gates"""
        issues = []
        irreversibility_penalty = 0.0
        
        if not signals.risk_assessment:
            return True, [], 0.0
        
        risk = signals.risk_assessment
        
        risk_ok, risk_reason = risk.is_acceptable_risk(
            max_ecological_risk=self.policy.max_ecological_risk,
            min_reversibility=self.policy.min_reversibility_score,
        )
        if not risk_ok:
            issues.append(f"Risk: {risk_reason}")
        
        if risk.reversibility_score < 0.30:
            irreversibility_penalty = 0.15
            issues.append(
                f"Near-irreversible intervention (reversibility={risk.reversibility_score:.2f}) "
                f"requires elevated trust score"
            )
        elif risk.reversibility_score < 0.50:
            irreversibility_penalty = 0.10
        
        passed = len(issues) == 0 or irreversibility_penalty > 0
        return passed, issues, irreversibility_penalty
    
    def check_temporal_gates(
        self,
        signals: OceanSignals
    ) -> Tuple[bool, List[str]]:
        """Check temporal staging requirements"""
        issues = []
        
        if not signals.temporal_context:
            return True, []
        
        tc = signals.temporal_context
        
        if tc.observation_period_days < self.policy.min_observation_period_days:
            issues.append(
                f"Insufficient observation ({tc.observation_period_days:.1f} days "
                f"< {self.policy.min_observation_period_days} required)"
            )
        
        if tc.trend_stability_days is not None:
            if tc.trend_stability_days < self.policy.min_trend_stability_days:
                issues.append(
                    f"Trend unstable ({tc.trend_stability_days:.1f} days "
                    f"< {self.policy.min_trend_stability_days} required)"
                )
        
        if tc.time_since_last_intervention_days is not None:
            if tc.time_since_last_intervention_days < self.policy.min_time_between_interventions_days:
                issues.append(
                    f"Too soon after last intervention ({tc.time_since_last_intervention_days:.1f} days "
                    f"< {self.policy.min_time_between_interventions_days} required)"
                )
        
        if tc.seasonal_appropriateness == "inappropriate":
            issues.append("Seasonal timing inappropriate")
        
        passed = len(issues) == 0
        return passed, issues
    
    def check_regulatory_gates(
        self,
        signals: OceanSignals
    ) -> Tuple[RegulatoryGateResult, List[str]]:
        """
        Check regulatory compliance with explicit HOLD vs FAIL distinction.
        
        Returns:
            (gate_result, issues)
        """
        if not self.policy.require_regulatory_approval:
            return RegulatoryGateResult.PASS, []
        
        if not signals.regulatory_status:
            return RegulatoryGateResult.FAIL, ["Regulatory status required but not provided"]
        
        reg_status = signals.regulatory_status
        issues = []
        
        # Explicit permit status check
        if reg_status.permit_status == "denied":
            return RegulatoryGateResult.FAIL, ["Permit explicitly denied"]
        
        if reg_status.permit_status == "pending_review":
            return RegulatoryGateResult.HOLD, ["Permit pending regulatory review"]
        
        # Compliance checks (only if not denied/pending)
        if reg_status.compliance_score < self.policy.min_compliance_score:
            issues.append(
                f"compliance_score={reg_status.compliance_score:.2f} "
                f"below {self.policy.min_compliance_score}"
            )
        
        if not reg_status.environmental_impact_assessment_complete:
            issues.append("eia_incomplete")
        
        if not reg_status.endangered_species_clearance:
            issues.append("endangered_species_conflict")
        
        if not reg_status.marine_protected_area_clearance:
            issues.append("mpa_violation")
        
        if issues:
            return RegulatoryGateResult.FAIL, issues
        
        return RegulatoryGateResult.PASS, []
    
    def check_measurement_gates(
        self,
        signals: OceanSignals
    ) -> Tuple[bool, List[str]]:
        """Check measurement quality gates"""
        issues = []
        
        if signals.measurement_reliability < self.policy.min_measurement_reliability:
            issues.append(
                f"Insufficient measurement reliability ({signals.measurement_reliability:.2f} "
                f"< {self.policy.min_measurement_reliability})"
            )
        
        passed = len(issues) == 0
        return passed, issues
    
    def compute_precautionary_penalty(
        self,
        precautionary_flags: List[str]
    ) -> Tuple[float, str]:
        """
        Compute severity-weighted penalty from precautionary flags.
        
        Returns:
            (total_penalty, explanation)
        """
        if not precautionary_flags:
            return 0.0, "no_flags"
        
        # Sum weighted severities
        total_severity = sum(
            FLAG_SEVERITY.get(flag, 0.03)  # Default 3% for unknown flags
            for flag in precautionary_flags
        )
        
        # Cap at 25% total penalty
        capped_penalty = min(0.25, total_severity)
        
        # Find most severe flag
        severities = [(FLAG_SEVERITY.get(f, 0.03), f) for f in precautionary_flags]
        most_severe_score, most_severe_flag = max(severities, key=lambda x: x[0])
        
        explanation = (
            f"{len(precautionary_flags)} flags, "
            f"most_severe={most_severe_flag} ({most_severe_score:.2f}), "
            f"total_penalty={capped_penalty:.2f}"
        )
        
        return capped_penalty, explanation


# -----------------------------
# Uncertainty Calculator
# -----------------------------

class UncertaintyCalculator:
    """Explicit uncertainty aggregation from multiple sources"""
    
    @staticmethod
    def compute_aggregate_uncertainty(
        signals: OceanSignals,
        policy: OceanGovernancePolicy
    ) -> AggregateUncertainty:
        """Aggregate uncertainty from all sources"""
        components = {}
        
        # 1. Measurement uncertainty
        measurement_unc = 1.0 - signals.measurement_reliability
        components["measurement"] = measurement_unc
        
        # 2. Model disagreement
        model_unc = 0.0
        if len(signals.model_predictions) >= 2:
            scores = [m.proposed_action_trust_score for m in signals.model_predictions]
            confidences = [m.confidence for m in signals.model_predictions]
            
            weighted_mean = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
            weighted_var = sum(c * (s - weighted_mean)**2 for s, c in zip(scores, confidences)) / sum(confidences)
            model_unc = min(1.0, math.sqrt(weighted_var) * 3)
        components["model_disagreement"] = model_unc
        
        # 3. Temporal instability
        temporal_unc = 0.0
        if signals.temporal_context:
            if signals.temporal_context.trend_confidence is not None:
                temporal_unc = 1.0 - signals.temporal_context.trend_confidence
            elif signals.temporal_context.observation_period_days < policy.min_observation_period_days:
                temporal_unc = 0.5
        components["temporal"] = temporal_unc
        
        # 4. Risk uncertainty
        risk_unc = 0.0
        if signals.risk_assessment:
            if signals.risk_assessment.model_uncertainty is not None:
                risk_unc = signals.risk_assessment.model_uncertainty
        components["risk"] = risk_unc
        
        # Aggregate using weighted average
        weights = {
            "measurement": 0.35,
            "model_disagreement": 0.30,
            "temporal": 0.20,
            "risk": 0.15,
        }
        
        aggregate = sum(components[k] * weights[k] for k in components.keys())
        
        dominant_source = max(components.items(), key=lambda x: x[1])[0]
        
        return AggregateUncertainty(
            aggregate_uncertainty_score=aggregate,
            measurement_uncertainty=measurement_unc,
            model_disagreement=model_unc,
            temporal_instability=temporal_unc,
            risk_uncertainty=risk_unc,
            dominant_uncertainty_source=dominant_source,
            uncertainty_sources=components,
        )


# ===== REGIME TRACKER WITH HYSTERESIS =====

class RegimeTracker:
    """
    Ecosystem regime tracking with hysteresis to prevent oscillation.
    
    Philosophy: Real ecosystems don't snap cleanly between states.
    Require sustained evidence before declaring regime shift.
    """
    
    def __init__(self):
        self._regime_history: List[Tuple[float, EcosystemRegime]] = []
        self._current_regime: EcosystemRegime = EcosystemRegime.UNKNOWN
        self._last_transition_time: Optional[float] = None
        self._min_regime_duration_seconds: float = 86400.0 * 7  # 7 days minimum
    
    def get_current_regime(self) -> EcosystemRegime:
        """Get current regime (clean API)"""
        return self._current_regime
    
    def infer_regime_with_hysteresis(
        self,
        health_history: List[Tuple[float, float, EcosystemRegime]],
        current_health: float,
        signals: OceanSignals
    ) -> EcosystemRegime:
        """Infer regime with hysteresis to prevent thrashing"""
        
        if signals.current_regime != EcosystemRegime.UNKNOWN:
            return signals.current_regime
        
        if len(health_history) < 10:
            return self._current_regime
        
        # Calculate metrics
        recent_health = [h for _, h, _ in health_history[-10:]]
        health_trend = recent_health[-1] - recent_health[0]
        health_volatility = statistics.stdev(recent_health) if len(recent_health) > 1 else 0.0
        
        # Infer candidate regime
        candidate_regime = self._classify_regime(
            current_health, health_trend, health_volatility, signals
        )
        
        # Apply hysteresis
        if self._last_transition_time is None:
            self._current_regime = candidate_regime
            self._last_transition_time = time.time()
            return candidate_regime
        
        if candidate_regime == self._current_regime:
            return self._current_regime
        
        # Check if regime wants to change
        time_in_current_regime = time.time() - self._last_transition_time
        
        # Fast transitions for emergencies
        if candidate_regime in (EcosystemRegime.COLLAPSE, EcosystemRegime.DISTURBANCE):
            self._current_regime = candidate_regime
            self._last_transition_time = time.time()
            return candidate_regime
        
        # For non-emergency transitions, require minimum duration
        if time_in_current_regime < self._min_regime_duration_seconds:
            return self._current_regime
        
        # Additional stability check
        if self._is_transition_stable(candidate_regime, health_history):
            self._current_regime = candidate_regime
            self._last_transition_time = time.time()
            return candidate_regime
        
        return self._current_regime
    
    def _classify_regime(
        self,
        current_health: float,
        health_trend: float,
        health_volatility: float,
        signals: OceanSignals
    ) -> EcosystemRegime:
        """Classify regime based on metrics (no hysteresis)"""
        
        if current_health < 0.30:
            return EcosystemRegime.COLLAPSE
        
        if health_volatility > 0.15:
            return EcosystemRegime.DISTURBANCE
        
        if current_health >= 0.80 and abs(health_trend) < 0.05:
            return EcosystemRegime.STABLE_HEALTHY
        
        if current_health < 0.50 and abs(health_trend) < 0.05:
            return EcosystemRegime.STABLE_DEGRADED
        
        if self._current_regime == EcosystemRegime.POST_INTERVENTION:
            if health_trend > 0.05:
                return EcosystemRegime.RECOVERY
            elif abs(health_trend) < 0.05 and current_health > 0.60:
                return EcosystemRegime.STABLE_HEALTHY
            else:
                return EcosystemRegime.POST_INTERVENTION
        
        if health_trend > 0.10:
            return EcosystemRegime.IMPROVING
        
        if health_trend < -0.10:
            return EcosystemRegime.DEGRADING
        
        if signals.temporal_context and signals.temporal_context.seasonal_appropriateness:
            return EcosystemRegime.SEASONAL_VARIATION
        
        return EcosystemRegime.UNKNOWN
    
    def _is_transition_stable(
        self,
        candidate_regime: EcosystemRegime,
        health_history: List[Tuple[float, float, EcosystemRegime]]
    ) -> bool:
        """Check if regime transition has stable evidence"""
        if len(health_history) < 5:
            return False
        
        recent_health = [h for _, h, _ in health_history[-5:]]
        
        if candidate_regime == EcosystemRegime.IMPROVING:
            diffs = [recent_health[i+1] - recent_health[i] for i in range(len(recent_health)-1)]
            return sum(1 for d in diffs if d > 0) >= 3
        
        if candidate_regime == EcosystemRegime.DEGRADING:
            diffs = [recent_health[i+1] - recent_health[i] for i in range(len(recent_health)-1)]
            return sum(1 for d in diffs if d < 0) >= 3
        
        return True


# ===== ENHANCED OCEAN GOVERNOR =====

class OceanGovernor:
    """
    Enhanced ocean governor with all improvements:
    - Uncertainty-aware authority ceilings
    - Severity-weighted precautionary flags  
    - Regime hysteresis
    - Explicit regulatory HOLD/FAIL
    - Decision delta tracking
    """
    
    def __init__(
        self,
        cfg: Optional[AileeConfig] = None,
        policy: Optional[OceanGovernancePolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable")
        
        self.policy = policy or OceanGovernancePolicy()
        self.cfg = cfg or default_ocean_config(self.policy.ecosystem_type)
        
        self.cfg.hard_min = 0.0
        self.cfg.hard_max = 1.0
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        self.policy_evaluator = PolicyEvaluator(self.policy)
        self.regime_tracker = RegimeTracker()
        
        # State tracking for delta computation
        self._last_decision: Optional[AuthorityLevel] = None
        self._last_trust_score: Optional[float] = None
        self._last_health_index: Optional[float] = None
        self._last_uncertainty: Optional[float] = None
        self._last_measurement_reliability: Optional[float] = None
        self._last_decision_time: Optional[float] = None
        self._last_intervention_time: Optional[float] = None
        self._observation_start_time: Optional[float] = time.time()
        
        self._health_history: List[Tuple[float, float, EcosystemRegime]] = []
        self._safety_history: List[Tuple[float, float, EcosystemRegime]] = []
        
        self._event_log: List[OceanEvent] = []
        self._last_event: Optional[OceanEvent] = None
    
    def evaluate(self, signals: OceanSignals) -> OceanDecisionResult:
        """Enhanced evaluation with all improvements"""
        ts = float(signals.timestamp)
        reasons: List[str] = []
        precautionary_flags: List[str] = []
        
        # 0) Compute aggregate uncertainty
        aggregate_unc = UncertaintyCalculator.compute_aggregate_uncertainty(signals, self.policy)
        
        unc_ok, unc_reason = aggregate_unc.is_uncertainty_acceptable(max_aggregate=0.40)
        if not unc_ok:
            reasons.append(unc_reason)
            precautionary_flags.append("aggregate_uncertainty_excessive")
        
        # 1) Regulatory gate with HOLD/FAIL distinction
        reg_result, reg_issues = self.policy_evaluator.check_regulatory_gates(signals)
        
        if reg_result == RegulatoryGateResult.FAIL:
            reasons.extend([f"Regulatory: {issue}" for issue in reg_issues])
            decision = self._create_prohibition_decision(
                signals, reasons, ts, "regulatory_failure", aggregate_unc
            )
            self._log_and_track(ts, signals, decision, reasons, aggregate_unc)
            return decision
        
        if reg_result == RegulatoryGateResult.HOLD:
            reasons.extend([f"Regulatory: {issue}" for issue in reg_issues])
            precautionary_flags.append("regulatory_hold")
        
        # 2) Health gates
        health_ok, health_issues, is_emergency = self.policy_evaluator.check_health_gates(signals)
        if not health_ok:
            reasons.extend([f"Ecosystem: {issue}" for issue in health_issues])
            if is_emergency:
                precautionary_flags.append("ecosystem_crisis_emergency_authorized")
            else:
                precautionary_flags.append("critical_thresholds_breached")
        
        # 3) Measurement gates
        meas_ok, meas_issues = self.policy_evaluator.check_measurement_gates(signals)
        if not meas_ok:
            reasons.extend(meas_issues)
            precautionary_flags.append("measurement_uncertainty_high")
        
        # 4) Risk gates
        risk_ok, risk_issues, irreversibility_penalty = self.policy_evaluator.check_risk_gates(signals)
        if not risk_ok:
            reasons.extend(risk_issues)
            precautionary_flags.append("risk_profile_unacceptable")
        
        if irreversibility_penalty > 0:
            precautionary_flags.append("near_irreversible_intervention")
        
        # 5) Temporal gates
        temporal_ok, temporal_issues = self.policy_evaluator.check_temporal_gates(signals)
        if not temporal_ok:
            reasons.extend(temporal_issues)
            precautionary_flags.extend([
                "observation_period_insufficient",
                "trend_instability",
                "intervention_frequency_excessive"
            ])
        
        # 6) Model consensus
        if self.policy.require_model_consensus and len(signals.model_predictions) >= 2:
            consensus_ok, consensus_reason = self._check_model_consensus_weighted(signals)
            if not consensus_ok:
                reasons.append(f"Model consensus: {consensus_reason}")
                precautionary_flags.append("model_disagreement")
        
        # 7) Compute severity-weighted precautionary penalty
        precautionary_penalty, penalty_explanation = \
            self.policy_evaluator.compute_precautionary_penalty(precautionary_flags)
        
        # 8) Extract peer values
        peer_values = []
        for sensor in signals.sensor_readings:
            if sensor.confidence > 0.60:
                peer_values.append(sensor.confidence)
        for model_pred in signals.model_predictions:
            peer_values.append(model_pred.proposed_action_trust_score)
        
        # 9) Build context
        ctx = self._build_context(signals, reasons, precautionary_flags, aggregate_unc)
        
        # 10) Apply penalties
        base_score = signals.proposed_action_trust_score
        score_after_precaution = base_score * (1.0 - precautionary_penalty)
        uncertainty_penalty = aggregate_unc.aggregate_uncertainty_score * 0.20
        final_adjusted_score = score_after_precaution * (1.0 - uncertainty_penalty)
        
        ctx["precautionary_penalty"] = precautionary_penalty
        ctx["penalty_explanation"] = penalty_explanation
        ctx["uncertainty_penalty"] = uncertainty_penalty
        ctx["final_adjustment"] = base_score - final_adjusted_score
        
        # 11) AILEE pipeline
        ailee_result = self.pipeline.process(
            raw_value=float(final_adjusted_score),
            raw_confidence=float(signals.measurement_reliability * (1.0 - aggregate_unc.aggregate_uncertainty_score)),
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        # 12) Determine authority ceiling from uncertainty
        max_authority = self._compute_authority_ceiling(aggregate_unc)
        
        # 13) Make decision with ceiling applied
        decision = self._make_ocean_decision(
            signals, ailee_result, reasons, precautionary_flags,
            aggregate_unc, is_emergency, ts, max_authority
        )
        
        # 14) Compute decision delta
        decision_delta = self._compute_decision_delta(signals, decision, aggregate_unc, ts)
        if decision_delta:
            decision.metadata["decision_delta"] = decision_delta
        
        # 15) Track regime with hysteresis
        inferred_regime = self.regime_tracker.infer_regime_with_hysteresis(
            self._health_history, signals.ecosystem_health_index, signals
        )
        
        # 16) Log and track
        self._log_and_track(ts, signals, decision, reasons, aggregate_unc)
        
        return decision
    
    def _compute_authority_ceiling(
        self,
        aggregate_unc: AggregateUncertainty
    ) -> AuthorityLevel:
        """
        Compute maximum allowed authority based on uncertainty.
        
        Philosophy: High trust + high uncertainty ≠ permission to act.
        """
        unc_score = aggregate_unc.aggregate_uncertainty_score
        
        if unc_score > 0.50:
            return AuthorityLevel.OBSERVE_ONLY
        if unc_score > 0.35:
            return AuthorityLevel.STAGE_INTERVENTION
        if unc_score > 0.25:
            return AuthorityLevel.CONTROLLED_INTERVENTION
        
        return AuthorityLevel.FULL_INTERVENTION
    
    def _check_model_consensus_weighted(self, signals: OceanSignals) -> Tuple[bool, str]:
        """Validate multi-model agreement with confidence weighting"""
        if not self.policy.require_model_consensus:
            return True, "consensus_not_required"
        
        if len(signals.model_predictions) < 2:
            return True, "insufficient_models_for_consensus"
        
        scores = [m.proposed_action_trust_score for m in signals.model_predictions]
        confidences = [m.confidence for m in signals.model_predictions]
        
        weighted_avg = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
        
        weighted_deviations = [
            c * abs(s - weighted_avg) 
            for s, c in zip(scores, confidences)
        ]
        max_weighted_dev = max(weighted_deviations) if weighted_deviations else 0.0
        
        if max_weighted_dev > 0.15:
            return False, f"model_disagreement (max_weighted_deviation={max_weighted_dev:.3f})"
        
        total_conf = sum(confidences)
        conf_in_agreement = sum(
            c for s, c in zip(scores, confidences) 
            if abs(s - weighted_avg) < 0.15
        )
        agreement_ratio = conf_in_agreement / total_conf
        
        if agreement_ratio < self.policy.min_model_agreement_ratio:
            return False, f"insufficient_weighted_agreement (ratio={agreement_ratio:.2f})"
        
        return True, "model_consensus_achieved"
    
    def _build_context(
        self,
        signals: OceanSignals,
        reasons: List[str],
        precautionary_flags: List[str],
        aggregate_unc: AggregateUncertainty
    ) -> Dict[str, Any]:
        """Build context dictionary for AILEE pipeline"""
        ctx = {
            "ecosystem_type": signals.ecosystem_type.value,
            "intervention_category": signals.intervention_category.value,
            "ecosystem_health_index": signals.ecosystem_health_index,
            "measurement_reliability": signals.measurement_reliability,
            "aggregate_uncertainty": aggregate_unc.aggregate_uncertainty_score,
            "dominant_uncertainty_source": aggregate_unc.dominant_uncertainty_source,
            "reasons": reasons[:],
            "precautionary_flags": precautionary_flags[:],
            "current_regime": signals.current_regime.value,
        }
        
        if signals.intervention_id:
            ctx["intervention_id"] = signals.intervention_id
        
        if signals.risk_assessment:
            ctx["ecological_risk"] = signals.risk_assessment.ecological_risk_score
            ctx["reversibility"] = signals.risk_assessment.reversibility_score
        
        if signals.temporal_context:
            ctx["observation_days"] = signals.temporal_context.observation_period_days
        
        ctx.update(signals.context)
        return ctx
    
    def _make_ocean_decision(
        self,
        signals: OceanSignals,
        ailee_result: DecisionResult,
        reasons: List[str],
        precautionary_flags: List[str],
        aggregate_unc: AggregateUncertainty,
        is_emergency: bool,
        ts: float,
        max_authority: AuthorityLevel
    ) -> OceanDecisionResult:
        """Convert AILEE result to ocean-specific staged decision with authority ceiling"""
        
        validated_score = ailee_result.validated_value
        
        # Determine base authority level
        if validated_score >= self.policy.full_intervention_threshold and len(precautionary_flags) == 0:
            authority_level = AuthorityLevel.FULL_INTERVENTION
            outcome = DecisionOutcome.INTERVENTION_AUTHORIZED
            authorized = True
            recommendation = "full_intervention_authorized"
        
        elif validated_score >= self.policy.controlled_intervention_threshold:
            if len(precautionary_flags) > 2:
                authority_level = AuthorityLevel.STAGE_INTERVENTION
                outcome = DecisionOutcome.STAGE_APPROVED
                authorized = True
                recommendation = "stage_intervention_multiple_precautions"
            else:
                authority_level = AuthorityLevel.CONTROLLED_INTERVENTION
                outcome = DecisionOutcome.INTERVENTION_AUTHORIZED
                authorized = True
                recommendation = "controlled_intervention_with_monitoring"
        
        elif validated_score >= self.policy.stage_intervention_threshold:
            authority_level = AuthorityLevel.STAGE_INTERVENTION
            outcome = DecisionOutcome.STAGE_APPROVED
            authorized = True
            recommendation = "prepare_intervention_continue_monitoring"
        
        elif validated_score >= self.policy.observe_only_threshold:
            authority_level = AuthorityLevel.OBSERVE_ONLY
            outcome = DecisionOutcome.OBSERVE_CONTINUE
            authorized = False
            recommendation = "continue_observation_no_intervention"
        
        else:
            if ailee_result.status == SafetyStatus.UNSAFE:
                authority_level = AuthorityLevel.NO_ACTION
                outcome = DecisionOutcome.PROHIBIT
                authorized = False
                recommendation = "intervention_prohibited"
            else:
                authority_level = AuthorityLevel.OBSERVE_ONLY
                outcome = DecisionOutcome.OBSERVE_CONTINUE
                authorized = False
                recommendation = "insufficient_evidence_observe_only"
        
        # Apply authority ceiling
        authority_index = {
            AuthorityLevel.NO_ACTION: 0,
            AuthorityLevel.OBSERVE_ONLY: 1,
            AuthorityLevel.STAGE_INTERVENTION: 2,
            AuthorityLevel.CONTROLLED_INTERVENTION: 3,
            AuthorityLevel.FULL_INTERVENTION: 4,
            AuthorityLevel.EMERGENCY_RESPONSE: 5,
        }
        
        if authority_index[authority_level] > authority_index[max_authority]:
            authority_level = max_authority
            recommendation = f"{recommendation}_uncertainty_capped"
        
        # Emergency override
        if is_emergency and validated_score >= 0.70:
            # Check if ceiling was bypassed
            if authority_index[max_authority] < authority_index[AuthorityLevel.EMERGENCY_RESPONSE]:
                reasons.append(f"Emergency override bypassed uncertainty ceiling (was: {max_authority.value})")
                # Metadata flag for audit traceability
                metadata_flags = {"emergency_override_bypassed_ceiling": True, "original_ceiling": max_authority.value}
            else:
                metadata_flags = {}
            
            authority_level = AuthorityLevel.EMERGENCY_RESPONSE
            outcome = DecisionOutcome.EMERGENCY_APPROVED
            authorized = True
            recommendation = "emergency_intervention_ecosystem_crisis"
            reasons.append("Emergency response authorized due to ecosystem crisis")
            reasons.append("MANDATORY: Post-emergency after-action review required")
        else:
            metadata_flags = {}
        
        # Regulatory hold check
        if signals.regulatory_status and signals.regulatory_status.permit_status == "pending_review":
            if authority_level not in (AuthorityLevel.OBSERVE_ONLY, AuthorityLevel.NO_ACTION):
                authority_level = AuthorityLevel.STAGE_INTERVENTION
                outcome = DecisionOutcome.REGULATORY_HOLD
                recommendation = "staging_approved_pending_regulatory_approval"
        
        # Determine risk level
        if validated_score >= 0.85:
            risk_level = "low"
        elif validated_score >= 0.75:
            risk_level = "moderate"
        elif validated_score >= 0.60:
            risk_level = "high"
        else:
            risk_level = "severe"
        
        # Generate staging requirements
        staging_reqs = None
        if authority_level == AuthorityLevel.STAGE_INTERVENTION:
            staging_reqs = self._generate_staging_requirements(signals, precautionary_flags)
        
        # Generate intervention constraints
        intervention_constraints = None
        if authorized and authority_level in (
            AuthorityLevel.CONTROLLED_INTERVENTION,
            AuthorityLevel.FULL_INTERVENTION,
            AuthorityLevel.EMERGENCY_RESPONSE
        ):
            intervention_constraints = self._generate_intervention_constraints(
                signals, precautionary_flags, authority_level
            )
        
        return OceanDecisionResult(
            intervention_authorized=authorized,
            authority_level=authority_level,
            decision_outcome=outcome,
            validated_trust_score=validated_score,  # RENAMED
            confidence_score=ailee_result.confidence_score,
            recommendation=recommendation,
            reasons=reasons[:],
            staging_requirements=staging_reqs,
            intervention_constraints=intervention_constraints,
            ailee_result=ailee_result,
            risk_level=risk_level,
            precautionary_flags=precautionary_flags[:],
            metadata={
                "timestamp": ts,
                "ecosystem_type": signals.ecosystem_type.value,
                "intervention_category": signals.intervention_category.value,
                "precautionary_bias": self.policy.precautionary_bias,
                "aggregate_uncertainty": aggregate_unc.aggregate_uncertainty_score,
                "dominant_uncertainty_source": aggregate_unc.dominant_uncertainty_source,
                "authority_ceiling": max_authority.value,
                **metadata_flags,  # Include emergency override flags if present
            }
        )
    
    def _compute_decision_delta(
        self,
        signals: OceanSignals,
        decision: OceanDecisionResult,
        aggregate_unc: AggregateUncertainty,
        ts: float
    ) -> Optional[DecisionDelta]:
        """Compute change since last decision with null safety"""
        
        # Null safety: check all required previous values exist
        if None in (
            self._last_trust_score,
            self._last_health_index,
            self._last_uncertainty,
            self._last_measurement_reliability,
        ):
            return None
        
        trust_delta = decision.validated_trust_score - self._last_trust_score
        health_delta = signals.ecosystem_health_index - self._last_health_index
        unc_delta = aggregate_unc.aggregate_uncertainty_score - self._last_uncertainty
        reliability_delta = signals.measurement_reliability - self._last_measurement_reliability
        
        regime_changed = (
            signals.current_regime != self.regime_tracker.get_current_regime()
        )
        
        authority_changed = (
            decision.authority_level != self._last_decision
        )
        
        time_delta = ts - (self._last_decision_time or ts)
        
        return DecisionDelta(
            trust_score_delta=trust_delta,
            ecosystem_health_delta=health_delta,
            uncertainty_delta=unc_delta,
            measurement_reliability_delta=reliability_delta,
            regime_changed=regime_changed,
            authority_level_changed=authority_changed,
            time_since_last_decision_seconds=time_delta,
        )
    
    def _create_prohibition_decision(
        self,
        signals: OceanSignals,
        reasons: List[str],
        ts: float,
        prohibition_type: str,
        aggregate_unc: AggregateUncertainty
    ) -> OceanDecisionResult:
        """Create absolute prohibition decision"""
        return OceanDecisionResult(
            intervention_authorized=False,
            authority_level=AuthorityLevel.NO_ACTION,
            decision_outcome=DecisionOutcome.PROHIBIT,
            validated_trust_score=0.0,  # RENAMED
            confidence_score=1.0,
            recommendation=f"intervention_prohibited_{prohibition_type}",
            reasons=reasons[:],
            ailee_result=None,
            risk_level="severe",
            precautionary_flags=[prohibition_type],
            metadata={
                "timestamp": ts,
                "ecosystem_type": signals.ecosystem_type.value,
                "prohibition_type": prohibition_type,
            }
        )
    
    def _generate_staging_requirements(
        self,
        signals: OceanSignals,
        precautionary_flags: List[str]
    ) -> StagingRequirements:
        """Generate requirements before intervention can proceed"""
        reqs = StagingRequirements()
        
        if signals.temporal_context:
            current_obs = signals.temporal_context.observation_period_days
            if current_obs < self.policy.min_observation_period_days:
                reqs.required_observation_days = int(self.policy.min_observation_period_days - current_obs)
        
        if "measurement_uncertainty_high" in precautionary_flags:
            reqs.required_sensor_types = ["buoy", "satellite", "lab_sample"]
        
        if "model_disagreement" in precautionary_flags:
            reqs.required_model_agreement = 0.85
        
        if signals.regulatory_status and signals.regulatory_status.permit_status != "approved":
            reqs.required_regulatory_approvals = ["environmental_impact_assessment", "permit_approval"]
        
        if signals.ecosystem_type in (EcosystemType.CORAL_REEF, EcosystemType.POLAR):
            reqs.environmental_conditions = ["no_storm", "appropriate_temperature", "daylight"]
        
        reqs.reason = f"Staging requirements based on {len(precautionary_flags)} precautionary flags"
        
        return reqs
    
    def _generate_intervention_constraints(
        self,
        signals: OceanSignals,
        precautionary_flags: List[str],
        authority_level: AuthorityLevel
    ) -> Dict[str, Any]:
        """Generate constraints for intervention execution"""
        constraints = {
            "monitoring_required": True,
            "reporting_frequency_days": 7,
        }
        
        if signals.risk_assessment:
            if signals.risk_assessment.reversibility_score < 0.50:
                constraints["reversibility_monitoring"] = "continuous"
                constraints["abort_conditions"] = ["unexpected_cascade", "off_target_effects"]
            
            if signals.risk_assessment.spatial_extent_km2:
                if signals.risk_assessment.spatial_extent_km2 > 100:
                    constraints["max_spatial_extent_km2"] = min(
                        signals.risk_assessment.spatial_extent_km2,
                        50.0
                    )
        
        if len(precautionary_flags) > 0:
            constraints["precautionary_measures"] = precautionary_flags[:]
            constraints["enhanced_monitoring"] = True
            constraints["stakeholder_notification"] = True
        
        if signals.ecosystem_type == EcosystemType.CORAL_REEF:
            constraints["max_temperature_change_c"] = 0.5
            constraints["prohibited_chemicals"] = ["copper_based", "oxybenzone"]
        elif signals.ecosystem_type == EcosystemType.DEEP_SEA:
            constraints["minimum_recovery_assessment_years"] = 10
        
        return constraints
    
    def _log_and_track(
        self,
        ts: float,
        signals: OceanSignals,
        decision: OceanDecisionResult,
        reasons: List[str],
        aggregate_unc: AggregateUncertainty
    ):
        """Combined logging and state tracking"""
        
        event_type_map = {
            DecisionOutcome.PROHIBIT: "intervention_prohibited",
            DecisionOutcome.OBSERVE_CONTINUE: "observation_recommended",
            DecisionOutcome.STAGE_APPROVED: "staging_approved",
            DecisionOutcome.INTERVENTION_AUTHORIZED: "intervention_authorized",
            DecisionOutcome.EMERGENCY_APPROVED: "emergency_intervention",
            DecisionOutcome.REGULATORY_HOLD: "regulatory_hold",
            DecisionOutcome.ESCALATE_HUMAN_REVIEW: "escalated_to_human",
        }
        
        event = OceanEvent(
            timestamp=ts,
            event_type=event_type_map.get(decision.decision_outcome, "unknown"),
            ecosystem_type=signals.ecosystem_type,
            intervention_category=signals.intervention_category,
            authority_level=decision.authority_level,
            decision_outcome=decision.decision_outcome,
            proposed_action_trust_score=decision.validated_trust_score,  # FIXED
            ecosystem_health_index=signals.ecosystem_health_index,
            measurement_reliability=signals.measurement_reliability,
            reasons=reasons[:],
            ecosystem_regime=self.regime_tracker.get_current_regime(),  # Use clean API
            ecosystem_health=signals.ecosystem_health,
            risk_assessment=signals.risk_assessment,
            regulatory_status=signals.regulatory_status,
            temporal_context=signals.temporal_context,
            aggregate_uncertainty=aggregate_unc,
            ailee_decision=decision.ailee_result,
            metadata=decision.metadata,
        )
        
        self._event_log.append(event)
        
        if len(self._event_log) > self.policy.max_event_log_size:
            self._event_log = self._event_log[-self.policy.max_event_log_size:]
        
        self._last_event = event
        
        # Update state tracking
        self._last_decision = decision.authority_level
        self._last_trust_score = decision.validated_trust_score
        self._last_health_index = signals.ecosystem_health_index
        self._last_uncertainty = aggregate_unc.aggregate_uncertainty_score
        self._last_measurement_reliability = signals.measurement_reliability
        self._last_decision_time = ts
        
        if decision.intervention_authorized and decision.authority_level not in (
            AuthorityLevel.OBSERVE_ONLY, AuthorityLevel.NO_ACTION
        ):
            self._last_intervention_time = ts
        
        # Track regime
        self._health_history.append((
            ts,
            signals.ecosystem_health_index,
            self.regime_tracker.get_current_regime()  # Use clean API
        ))
        if len(self._health_history) > 1000:
            self._health_history = self._health_history[-1000:]
        
        self._safety_history.append((
            ts,
            decision.validated_trust_score,
            self.regime_tracker.get_current_regime()  # Use clean API
        ))
        if len(self._safety_history) > 1000:
            self._safety_history = self._safety_history[-1000:]
    
    # -------------------------
    # Public API - Explainability
    # -------------------------
    
    def explain_decision(self, decision: OceanDecisionResult) -> str:
        """Generate plain-language explanation of decision"""
        lines = []
        
        lines.append("=" * 70)
        lines.append("OCEAN GOVERNANCE DECISION EXPLANATION")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"DECISION: {decision.decision_outcome.value}")
        lines.append(f"Authority Level: {decision.authority_level.value}")
        lines.append(f"Intervention Authorized: {'YES' if decision.intervention_authorized else 'NO'}")
        lines.append(f"Risk Level: {decision.risk_level.upper()}")
        lines.append("")
        
        lines.append(f"Validated Trust Score: {decision.validated_trust_score:.2f} / 1.00")
        lines.append(f"Decision Confidence: {decision.confidence_score:.2f} / 1.00")
        lines.append("")
        
        lines.append("WHAT MATTERED MOST:")
        lines.append("-" * 70)
        
        if decision.ailee_result:
            lines.append(f"• AILEE Pipeline Status: {decision.ailee_result.status.value}")
            if decision.ailee_result.grace_applied:
                lines.append("  → Grace conditions applied (leniency given)")
            if decision.ailee_result.consensus_status:
                lines.append(f"  → Consensus: {decision.ailee_result.consensus_status}")
        
        if decision.metadata.get("aggregate_uncertainty"):
            unc = decision.metadata["aggregate_uncertainty"]
            source = decision.metadata.get("dominant_uncertainty_source", "unknown")
            lines.append(f"• Aggregate Uncertainty: {unc:.2f} (primary source: {source})")
        
        if decision.metadata.get("authority_ceiling"):
            ceiling = decision.metadata["authority_ceiling"]
            lines.append(f"• Authority Ceiling from Uncertainty: {ceiling}")
            
            # Show if emergency override bypassed ceiling
            if decision.metadata.get("emergency_override_bypassed_ceiling"):
                original = decision.metadata.get("original_ceiling", "unknown")
                lines.append(f"  ⚠️ EMERGENCY OVERRIDE: Ceiling ({original}) bypassed for crisis response")
        
        if decision.metadata.get("precautionary_bias"):
            bias = decision.metadata["precautionary_bias"]
            lines.append(f"• Precautionary Bias Applied: {bias:.2f} (higher = more caution)")
        
        lines.append("")
        
        if not decision.intervention_authorized or decision.authority_level == AuthorityLevel.OBSERVE_ONLY:
            lines.append("WHAT BLOCKED INTERVENTION:")
            lines.append("-" * 70)
            
            if decision.reasons:
                for reason in decision.reasons:
                    lines.append(f"• {reason}")
            else:
                lines.append("• Insufficient trust score for authorization")
            
            if decision.precautionary_flags:
                lines.append("")
                lines.append("Precautionary Flags Raised:")
                for flag in decision.precautionary_flags:
                    lines.append(f"  - {flag}")
            
            lines.append("")
        
        lines.append("WHAT WOULD CHANGE THIS DECISION:")
        lines.append("-" * 70)
        
        if decision.validated_trust_score < 0.75:
            gap = 0.75 - decision.validated_trust_score
            lines.append(f"• Increase trust score by {gap:.2f} to reach staging threshold (0.75)")
        
        if decision.precautionary_flags:
            lines.append("• Address precautionary concerns:")
            for flag in decision.precautionary_flags:
                if "observation" in flag.lower():
                    lines.append("  → Extend observation period")
                elif "uncertainty" in flag.lower():
                    lines.append("  → Improve measurement quality / reduce uncertainty")
                elif "risk" in flag.lower():
                    lines.append("  → Demonstrate intervention reversibility")
                elif "regulatory" in flag.lower():
                    lines.append("  → Obtain required permits and approvals")
        
        if decision.metadata.get("aggregate_uncertainty", 0) > 0.30:
            lines.append("• Reduce aggregate uncertainty through:")
            lines.append("  → Additional sensor deployments")
            lines.append("  → Longer observation period")
            lines.append("  → Multiple independent models showing agreement")
        
        lines.append("")
        
        if decision.staging_requirements:
            lines.append("STAGING REQUIREMENTS:")
            lines.append("-" * 70)
            sr = decision.staging_requirements
            
            if sr.required_observation_days:
                lines.append(f"• Observe for {sr.required_observation_days} more days")
            if sr.required_sensor_types:
                lines.append(f"• Deploy sensors: {', '.join(sr.required_sensor_types)}")
            if sr.required_model_agreement:
                lines.append(f"• Achieve model agreement ≥ {sr.required_model_agreement:.2f}")
            if sr.required_regulatory_approvals:
                lines.append(f"• Obtain approvals: {', '.join(sr.required_regulatory_approvals)}")
            
            lines.append("")
        
        if decision.intervention_constraints:
            lines.append("INTERVENTION CONSTRAINTS:")
            lines.append("-" * 70)
            for key, value in decision.intervention_constraints.items():
                lines.append(f"• {key}: {value}")
            lines.append("")
        
        lines.append("RECOMMENDED ACTION:")
        lines.append("-" * 70)
        lines.append(f"→ {decision.recommendation.replace('_', ' ').title()}")
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    # -------------------------
    # Public API - State Queries
    # -------------------------
    
    def get_last_event(self) -> Optional[OceanEvent]:
        """Get most recent ocean governance event"""
        return self._last_event
    
    def export_events(self, since_ts: Optional[float] = None) -> List[OceanEvent]:
        """Export events for environmental compliance reporting"""
        if since_ts is None:
            return self._event_log[:]
        return [e for e in self._event_log if e.timestamp >= since_ts]
    
    def get_ecosystem_trend(self) -> str:
        """Get ecosystem health trend: improving, stable, degrading"""
        if len(self._health_history) < 20:
            return "insufficient_data"
        
        recent = [h for _, h, _ in self._health_history[-10:]]
        older = [h for _, h, _ in self._health_history[-20:-10]]
        
        recent_avg = statistics.fmean(recent)
        older_avg = statistics.fmean(older)
        
        diff = recent_avg - older_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        return "stable"
    
    def get_regime_history(self) -> Dict[str, Any]:
        """Get regime transition history"""
        if not self._health_history:
            return {"status": "no_history"}
        
        regimes = [r for _, _, r in self._health_history]
        regime_counts = {}
        for r in regimes:
            regime_counts[r.value] = regime_counts.get(r.value, 0) + 1
        
        current_regime = self.regime_tracker.get_current_regime().value  # Use clean API
        
        return {
            "current_regime": current_regime,
            "regime_distribution": regime_counts,
            "total_observations": len(regimes),
        }
    
    def get_intervention_history(self) -> Dict[str, Any]:
        """Get history of intervention decisions"""
        interventions = [
            e for e in self._event_log
            if e.decision_outcome in (
                DecisionOutcome.INTERVENTION_AUTHORIZED,
                DecisionOutcome.EMERGENCY_APPROVED
            )
        ]
        
        return {
            "total_interventions": len(interventions),
            "last_intervention_time": self._last_intervention_time,
            "intervention_types": [e.intervention_category.value for e in interventions],
            "average_trust_score": statistics.fmean([e.proposed_action_trust_score for e in interventions]) if interventions else 0.0,
        }
    
    def check_temporal_readiness(self, signals: OceanSignals) -> Tuple[bool, List[str]]:
        """Check if temporal requirements are met for intervention"""
        issues = []
        
        if signals.temporal_context:
            tc = signals.temporal_context
            
            if tc.observation_period_days < self.policy.min_observation_period_days:
                issues.append(
                    f"Need {self.policy.min_observation_period_days - tc.observation_period_days:.1f} "
                    f"more days of observation"
                )
            
            if tc.time_since_last_intervention_days is not None:
                if tc.time_since_last_intervention_days < self.policy.min_time_between_interventions_days:
                    issues.append(
                        f"Need {self.policy.min_time_between_interventions_days - tc.time_since_last_intervention_days:.1f} "
                        f"more days since last intervention"
                    )
        
        return len(issues) == 0, issues


# -----------------------------
# Convenience Functions
# -----------------------------

def create_ocean_governor(
    ecosystem_type: EcosystemType = EcosystemType.UNKNOWN,
    intervention_category: InterventionCategory = InterventionCategory.UNKNOWN,
    precautionary_bias: float = 0.80,
    **policy_overrides
) -> OceanGovernor:
    """Convenience factory for creating ocean governor with common configurations"""
    policy_kwargs = {
        "ecosystem_type": ecosystem_type,
        "intervention_category": intervention_category,
        "precautionary_bias": precautionary_bias,
    }
    
    if ecosystem_type in (EcosystemType.CORAL_REEF, EcosystemType.DEEP_SEA):
        policy_kwargs.update({
            "min_intervention_safety_score": 0.85,
            "max_ecological_risk": 0.20,
            "min_observation_period_days": 60.0,
            "precautionary_bias": max(precautionary_bias, 0.85),
        })
    elif ecosystem_type == EcosystemType.MARINE_PROTECTED_AREA:
        policy_kwargs.update({
            "require_regulatory_approval": True,
            "min_compliance_score": 0.98,
            "min_observation_period_days": 45.0,
        })
    
    policy_kwargs.update(policy_overrides)
    policy = OceanGovernancePolicy(**policy_kwargs)
    
    cfg = default_ocean_config(ecosystem_type)
    
    return OceanGovernor(cfg=cfg, policy=policy)


def validate_ocean_signals(signals: OceanSignals) -> Tuple[bool, List[str]]:
    """Pre-flight validation of ocean signals structure"""
    issues: List[str] = []
    
    if not (0.0 <= signals.proposed_action_trust_score <= 1.0):
        issues.append(f"proposed_action_trust_score={signals.proposed_action_trust_score} outside [0.0, 1.0]")
    
    if not (0.0 <= signals.ecosystem_health_index <= 1.0):
        issues.append(f"ecosystem_health_index={signals.ecosystem_health_index} outside [0.0, 1.0]")
    
    if not (0.0 <= signals.measurement_reliability <= 1.0):
        issues.append(f"measurement_reliability={signals.measurement_reliability} outside [0.0, 1.0]")
    
    for i, sensor in enumerate(signals.sensor_readings):
        if not (0.0 <= sensor.confidence <= 1.0):
            issues.append(f"sensor_readings[{i}].confidence={sensor.confidence} outside [0.0, 1.0]")
    
    for i, model in enumerate(signals.model_predictions):
        if not (0.0 <= model.proposed_action_trust_score <= 1.0):
            issues.append(f"model_predictions[{i}].trust_score={model.proposed_action_trust_score} outside [0.0, 1.0]")
        if not (0.0 <= model.confidence <= 1.0):
            issues.append(f"model_predictions[{i}].confidence={model.confidence} outside [0.0, 1.0]")
    
    if signals.risk_assessment:
        ra = signals.risk_assessment
        if not (0.0 <= ra.reversibility_score <= 1.0):
            issues.append(f"risk_assessment.reversibility={ra.reversibility_score} outside [0.0, 1.0]")
        if not (0.0 <= ra.ecological_risk_score <= 1.0):
            issues.append(f"risk_assessment.ecological_risk={ra.ecological_risk_score} outside [0.0, 1.0]")
    
    if signals.regulatory_status:
        if not (0.0 <= signals.regulatory_status.compliance_score <= 1.0):
            issues.append(f"regulatory_status.compliance_score outside [0.0, 1.0]")
    
    return len(issues) == 0, issues


# -----------------------------
# Module Exports
# -----------------------------

__all__ = [
    # Enums
    "EcosystemType",
    "EcosystemRegime",
    "InterventionCategory",
    "AuthorityLevel",
    "DecisionOutcome",
    "RegulatoryGateResult",
    
    # Data structures
    "EcosystemHealth",
    "RiskAssessment",
    "RegulatoryStatus",
    "TemporalContext",
    "OceanSensorReading",
    "ModelPrediction",
    "OceanSignals",
    "AggregateUncertainty",
    "DecisionDelta",
    "StagingRequirements",
    "OceanEvent",
    
    # Configuration
    "OceanGovernancePolicy",
    "OceanDecisionResult",
    
    # Governance components
    "PolicyEvaluator",
    "UncertaintyCalculator",
    "RegimeTracker",
    "OceanGovernor",
    
    # Utilities
    "default_ocean_config",
    "create_ocean_governor",
    "validate_ocean_signals",
    
    # Constants
    "FLAG_SEVERITY",
]
