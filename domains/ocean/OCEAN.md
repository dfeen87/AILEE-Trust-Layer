# AILEE Ocean Governance Domain

**Version 1.0.1 - Production Grade**

A specialized trust layer for governing marine ecosystem interventions with uncertainty-aware decision staging, regulatory compliance tracking, and precautionary risk management.

---

## Table of Contents

- [Overview](#overview)
- [Core Philosophy](#core-philosophy)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Ecosystem Types & Intervention Categories](#ecosystem-types--intervention-categories)
- [Decision Staging & Authority Levels](#decision-staging--authority-levels)
- [Uncertainty Management](#uncertainty-management)
- [Regulatory Compliance](#regulatory-compliance)
- [Risk Assessment](#risk-assessment)
- [Temporal Context & Regime Tracking](#temporal-context--regime-tracking)
- [Precautionary Flags](#precautionary-flags)
- [API Reference](#api-reference)
- [Configuration Guide](#configuration-guide)
- [Explainability & Auditing](#explainability--auditing)
- [Best Practices](#best-practices)
- [Version History](#version-history)

---

## Overview

The AILEE Ocean Governance Domain provides a **production-grade trust evaluation pipeline** for marine ecosystem intervention decisions. It combines multi-model validation, regulatory gate checks, risk profiling, and temporal staging requirements to determine when and how ocean interventions should proceed.

**Use Cases:**
- Marine restoration projects (coral reefs, kelp forests, seagrass meadows)
- Water quality management (nutrient control, oxygenation, alkalinity enhancement)
- Harmful algal bloom mitigation
- Species introduction decisions
- Pollution remediation
- Climate intervention experiments (ocean alkalinity enhancement, iron fertilization)
- Marine protected area management

---

## Core Philosophy

### 🌊 Precautionary Principle First
High trust + high uncertainty ≠ permission to act. The system defaults to caution when dealing with complex marine ecosystems.

### 📊 Uncertainty as a First-Class Citizen
Uncertainty isn't just tracked—it actively shapes decision authority through **uncertainty-aware authority ceilings**.

### 🎯 Staged Decision Ladder
Not all interventions are equal. Decisions progress through graduated authority levels from observation → staging → controlled intervention → full intervention.

### 🔄 Regime-Aware Governance
Ecosystems exist in different states (stable, degrading, recovering). Decisions adapt based on current regime with **hysteresis** to prevent oscillation.

### ⚖️ Explicit Regulatory Separation
Clear distinction between **HOLD** (pending approval) vs **FAIL** (denied) regulatory outcomes.

### 🔍 Explainability by Design
Every decision includes plain-language explanations of what mattered, what blocked intervention, and what would change the outcome.

---

## Key Features

### ✅ v1.0.1 Enhancements

- **Fixed:** Field name consistency (`intervention_safety_score` → `proposed_action_trust_score`)
- **Added:** Uncertainty-aware authority ceilings that cap decision authority based on aggregate uncertainty
- **Added:** Severity-weighted precautionary flags with configurable penalties
- **Added:** Regime transition hysteresis to prevent thrashing between ecosystem states
- **Added:** Explicit regulatory HOLD vs FAIL distinction
- **Added:** DecisionDelta tracking for real-time monitoring and trend detection
- **Renamed:** `validated_safety_score` → `validated_trust_score` for consistency

### 🎯 Core Capabilities

- **Multi-Gate Validation**: Health, risk, temporal, measurement, and regulatory gates
- **Multi-Model Consensus**: Confidence-weighted agreement across oceanographic models
- **Aggregate Uncertainty Calculation**: Unified uncertainty score from measurement, model, temporal, and risk sources
- **Emergency Override**: Bypass normal constraints during ecosystem crises (with audit trail)
- **Intervention Constraints**: Dynamic safety limits based on ecosystem type and risk profile
- **Compliance Logging**: Structured event stream for environmental reporting
- **Decision Deltas**: Track how key metrics change between decisions

---

## Quick Start

### Basic Usage

```python
from ailee_ocean_domain_v1 import (
    create_ocean_governor,
    OceanSignals,
    EcosystemType,
    InterventionCategory,
    EcosystemHealth,
    RiskAssessment,
    RegulatoryStatus,
    TemporalContext,
    validate_ocean_signals
)

# 1. Create a governor for your ecosystem
governor = create_ocean_governor(
    ecosystem_type=EcosystemType.CORAL_REEF,
    intervention_category=InterventionCategory.REEF_RESTORATION,
    precautionary_bias=0.85  # Higher = more cautious
)

# 2. Assemble your signals
signals = OceanSignals(
    proposed_action_trust_score=0.82,  # From your intervention model
    ecosystem_health_index=0.65,
    measurement_reliability=0.88,
    
    ecosystem_health=EcosystemHealth(
        overall_health_index=0.65,
        dissolved_oxygen_mg_l=6.2,
        ph=8.05,
        temperature_c=26.5,
        chlorophyll_a_ug_l=12.3,
        species_richness_score=0.72,
        trend_direction="stable"
    ),
    
    risk_assessment=RiskAssessment(
        reversibility_score=0.75,
        ecological_risk_score=0.22,
        cascade_risk_score=0.15,
        model_uncertainty=0.18
    ),
    
    regulatory_status=RegulatoryStatus(
        permit_status="approved",
        compliance_score=0.96,
        environmental_impact_assessment_complete=True
    ),
    
    temporal_context=TemporalContext(
        observation_period_days=45.0,
        trend_stability_days=20.0,
        time_since_last_intervention_days=120.0
    ),
    
    ecosystem_type=EcosystemType.CORAL_REEF,
    intervention_category=InterventionCategory.REEF_RESTORATION
)

# 3. Validate inputs
valid, issues = validate_ocean_signals(signals)
if not valid:
    print(f"Validation errors: {issues}")

# 4. Get decision
decision = governor.evaluate(signals)

# 5. Interpret results
print(f"Intervention Authorized: {decision.intervention_authorized}")
print(f"Authority Level: {decision.authority_level.value}")
print(f"Decision: {decision.decision_outcome.value}")
print(f"Risk Level: {decision.risk_level}")
print(f"\nValidated Trust Score: {decision.validated_trust_score:.2f}")
print(f"Confidence: {decision.confidence_score:.2f}")

# 6. Get human-readable explanation
explanation = governor.explain_decision(decision)
print(explanation)
```

---

## Architecture

### Component Hierarchy

```
OceanGovernor (Orchestrator)
├── PolicyEvaluator (Gate Checks)
│   ├── Health Gates
│   ├── Risk Gates
│   ├── Temporal Gates
│   ├── Measurement Gates
│   └── Regulatory Gates (HOLD/FAIL)
├── UncertaintyCalculator (Aggregation)
│   ├── Measurement Uncertainty
│   ├── Model Disagreement
│   ├── Temporal Instability
│   └── Risk Uncertainty
├── RegimeTracker (Hysteresis)
│   └── Ecosystem State Monitoring
└── AileeTrustPipeline (Core Trust Validation)
    ├── Stability Analysis
    ├── Peer Agreement
    ├── Forecast Validation
    ├── Grace Conditions
    └── Consensus Checks
```

### Data Flow

```
Input: OceanSignals
    ↓
[Gate Validation] → Pass/Fail per gate
    ↓
[Uncertainty Aggregation] → Single uncertainty score
    ↓
[Authority Ceiling Computation] → Max allowed authority from uncertainty
    ↓
[Precautionary Flag Collection] → Severity-weighted penalties
    ↓
[Score Adjustment] → Base score × (1 - penalties)
    ↓
[AILEE Trust Pipeline] → Validated score + confidence
    ↓
[Decision Mapping] → Authority level + Outcome
    ↓
[Ceiling Application] → Cap authority if uncertainty too high
    ↓
[Emergency Override Check] → Bypass ceiling if crisis + adequate trust
    ↓
[Decision Delta Computation] → Track changes since last decision
    ↓
Output: OceanDecisionResult
```

---

## Ecosystem Types & Intervention Categories

### Ecosystem Types

Each ecosystem has unique sensitivity profiles that affect thresholds:

| Ecosystem Type | Accept Threshold | Observation Period | Characteristics |
|----------------|------------------|-------------------|-----------------|
| `CORAL_REEF` | 0.90 | 60 days | Highly sensitive, slow recovery |
| `DEEP_SEA` | 0.90 | 60 days | Unknown dynamics, extreme caution |
| `POLAR` | 0.88 | 45 days | Climate-critical, fragile |
| `MARINE_PROTECTED_AREA` | 0.88 | 45 days | Legal protections, high compliance |
| `COASTAL_ZONE` | 0.82 | 30 days | High human activity, resilient |
| `ESTUARY` | 0.82 | 30 days | Dynamic, variable conditions |
| `AQUACULTURE_ZONE` | 0.80 | 30 days | Managed system, faster decisions |
| `KELP_FOREST` | 0.85 | 30 days | Moderate sensitivity |
| `MANGROVE` | 0.85 | 30 days | Important carbon sink |
| `SEAGRASS_MEADOW` | 0.85 | 30 days | Nursery habitat |
| `OPEN_OCEAN` | 0.85 | 30 days | Large scale, slow change |

### Intervention Categories

- `NUTRIENT_MANAGEMENT` - Control nitrogen/phosphorus inputs
- `OXYGENATION` - Address hypoxia/anoxia
- `ALKALINITY_ENHANCEMENT` - Ocean acidification mitigation
- `SPECIES_INTRODUCTION` - Restoration or control species
- `HARMFUL_ALGAE_MITIGATION` - HAB response
- `SEDIMENT_MANAGEMENT` - Dredging, deposition control
- `TEMPERATURE_MODIFICATION` - Localized cooling/warming
- `REEF_RESTORATION` - Coral/oyster/mussel reef rebuilding
- `POLLUTION_REMEDIATION` - Cleanup operations
- `OBSERVATION_ONLY` - Monitoring without intervention

---

## Decision Staging & Authority Levels

### Authority Ladder

Decisions progress through six graduated levels:

```
NO_ACTION (0) 
    ↓ Trust increases, gates pass
OBSERVE_ONLY (1)
    ↓ Observation period complete
STAGE_INTERVENTION (2)
    ↓ Staging requirements met
CONTROLLED_INTERVENTION (3)
    ↓ All constraints satisfied
FULL_INTERVENTION (4)
    ↓ Emergency override possible
EMERGENCY_RESPONSE (5)
```

### Authority Level Details

#### 1. NO_ACTION
- **When**: Absolute prohibition
- **Triggers**: Regulatory FAIL, critical safety violations
- **Characteristics**: Cannot be overridden (except emergency)

#### 2. OBSERVE_ONLY
- **When**: Trust score 0.60-0.75, or insufficient data
- **Actions**: Monitoring, data collection, no intervention
- **Requirements**: Continue observation until staging threshold met

#### 3. STAGE_INTERVENTION
- **When**: Trust score 0.75-0.85
- **Actions**: Preparation, equipment deployment, final checks
- **Requirements**: Complete staging requirements before proceeding

#### 4. CONTROLLED_INTERVENTION
- **When**: Trust score 0.85-0.95, some precautionary flags
- **Actions**: Limited intervention with enhanced monitoring
- **Constraints**: Spatial limits, reporting requirements, abort conditions

#### 5. FULL_INTERVENTION
- **When**: Trust score ≥ 0.95, no precautionary flags
- **Actions**: Unrestricted intervention within plan
- **Characteristics**: Highest confidence, minimal restrictions

#### 6. EMERGENCY_RESPONSE
- **When**: Ecosystem health < 0.20 AND trust ≥ 0.70
- **Actions**: Crisis intervention, bypass normal gates
- **Requirements**: Post-emergency after-action review mandatory

### Authority Ceiling from Uncertainty

The system caps authority based on aggregate uncertainty:

| Aggregate Uncertainty | Maximum Authority Level |
|-----------------------|-------------------------|
| > 0.50 | OBSERVE_ONLY |
| 0.36 - 0.50 | STAGE_INTERVENTION |
| 0.26 - 0.35 | CONTROLLED_INTERVENTION |
| ≤ 0.25 | FULL_INTERVENTION |

**Philosophy**: Even with high trust scores, excessive uncertainty prevents action. This prevents overconfident decisions when measurement quality is poor, models disagree, or temporal data is insufficient.

---

## Uncertainty Management

### Aggregate Uncertainty Calculation

Four sources combine into a single score:

```python
# Component weights
measurement_uncertainty:  35%  # 1 - measurement_reliability
model_disagreement:       30%  # Weighted variance across models
temporal_instability:     20%  # 1 - trend_confidence
risk_uncertainty:         15%  # From risk assessment
```

### AggregateUncertainty Object

```python
@dataclass
class AggregateUncertainty:
    aggregate_uncertainty_score: float     # Combined [0-1]
    measurement_uncertainty: float         # From sensors
    model_disagreement: float              # From predictions
    temporal_instability: float            # From trends
    risk_uncertainty: float                # From risk model
    dominant_uncertainty_source: str       # Which is highest
    uncertainty_sources: Dict[str, float]  # All components
```

### Model Disagreement Calculation

When multiple models provide predictions:

```python
# Confidence-weighted mean
weighted_mean = Σ(score_i × confidence_i) / Σ(confidence_i)

# Confidence-weighted variance
weighted_variance = Σ(confidence_i × (score_i - mean)²) / Σ(confidence_i)

# Disagreement score (3-sigma scaled)
model_disagreement = min(1.0, √variance × 3)
```

### Uncertainty Impact

Uncertainty affects decisions in two ways:

1. **Score Adjustment**: `adjusted_score = base_score × (1 - uncertainty × 0.20)`
2. **Authority Ceiling**: Limits maximum authority regardless of trust score

---

## Regulatory Compliance

### Regulatory Gate Results

Three explicit outcomes:

#### PASS ✅
- Permit status: "approved"
- Compliance score ≥ threshold
- All clearances obtained
- **Effect**: No restrictions

#### HOLD ⏸️
- Permit status: "pending_review"
- **Effect**: Can stage intervention, cannot execute
- **Decision Outcome**: `REGULATORY_HOLD`
- **Interpretation**: "Not yet allowed" vs "not allowed"

#### FAIL ❌
- Permit status: "denied"
- OR compliance violations
- OR endangered species conflict
- OR MPA violations
- **Effect**: Absolute prohibition (`NO_ACTION`)

### RegulatoryStatus Fields

```python
@dataclass
class RegulatoryStatus:
    permit_status: str  # "approved", "pending_review", "denied"
    compliance_score: float  # [0-1]
    
    environmental_impact_assessment_complete: bool
    public_consultation_complete: bool
    endangered_species_clearance: bool
    marine_protected_area_clearance: bool
    
    national_agency_approval: Optional[str]
    international_treaty_compliance: bool
    
    monitoring_requirements: Optional[List[str]]
    reporting_frequency_days: Optional[int]
```

### Compliance Checking

```python
# Automatic checks
if not eia_complete:
    FAIL: "eia_incomplete"

if not endangered_species_clearance:
    FAIL: "endangered_species_conflict"

if not mpa_clearance:
    FAIL: "mpa_violation"

if compliance_score < policy.min_compliance_score:
    FAIL: f"compliance_score={score} below {threshold}"
```

---

## Risk Assessment

### RiskAssessment Structure

```python
@dataclass
class RiskAssessment:
    reversibility_score: float       # [0-1], 1=fully reversible
    ecological_risk_score: float     # [0-1], 0=no risk
    
    cascade_risk_score: Optional[float]      # Risk of trophic cascades
    tipping_point_proximity: Optional[float] # How close to regime shift
    
    off_target_effects_score: Optional[float]
    temporal_persistence_years: Optional[float]
    spatial_extent_km2: Optional[float]
    
    model_uncertainty: Optional[float]
    measurement_uncertainty: Optional[float]
    
    similar_interventions_success_rate: Optional[float]
```

### Risk Gates

```python
# Default policy thresholds
max_ecological_risk = 0.30      # Stricter for sensitive ecosystems
min_reversibility = 0.50        # Higher for irreversible actions
max_cascade_risk = 0.60
max_tipping_point_proximity = 0.80

# Automatic checks
if ecological_risk > max_ecological_risk:
    FAIL: "ecological_risk_unacceptable"

if reversibility < min_reversibility:
    FAIL: "intervention_not_sufficiently_reversible"

if cascade_risk > 0.60:
    FAIL: "cascade_risk_too_high"

if tipping_point_proximity > 0.80:
    FAIL: "dangerously_close_to_tipping_point"
```

### Irreversibility Penalty

Low reversibility triggers elevated trust requirements:

```python
if reversibility_score < 0.30:
    irreversibility_penalty = 0.15  # Require 15% higher trust
    flag: "near_irreversible_intervention"

elif reversibility_score < 0.50:
    irreversibility_penalty = 0.10  # Require 10% higher trust
```

---

## Temporal Context & Regime Tracking

### TemporalContext Structure

```python
@dataclass
class TemporalContext:
    observation_period_days: float
    trend_stability_days: Optional[float]
    trend_confidence: Optional[float]  # [0-1]
    
    last_intervention_date: Optional[float]
    time_since_last_intervention_days: Optional[float]
    
    seasonal_appropriateness: Optional[str]  # "appropriate", "inappropriate"
    
    forecast_horizon_days: Optional[int]
    forecast_confidence: Optional[float]
```

### Temporal Gates

```python
# Policy defaults (vary by ecosystem)
min_observation_period_days = 30.0
min_trend_stability_days = 14.0
min_time_between_interventions_days = 90.0

# Automatic checks
if observation_period < min_observation_period:
    FAIL: "observation_period_insufficient"

if trend_stability < min_trend_stability:
    FAIL: "trend_unstable"

if time_since_last_intervention < min_time_between:
    FAIL: "too_soon_after_last_intervention"

if seasonal_appropriateness == "inappropriate":
    FAIL: "seasonal_timing_inappropriate"
```

### Ecosystem Regime Tracking

#### Regime States

```python
class EcosystemRegime(Enum):
    STABLE_HEALTHY           # Baseline good condition
    STABLE_DEGRADED          # Persistent poor condition
    IMPROVING                # Recovering
    DEGRADING                # Declining
    POST_INTERVENTION        # Recently intervened
    DISTURBANCE              # External shock
    SEASONAL_VARIATION       # Normal fluctuation
    RECOVERY                 # Post-intervention improvement
    COLLAPSE                 # Critical failure
    UNKNOWN                  # Insufficient data
```

#### Hysteresis Mechanism

**Problem**: Ecosystems don't snap between states. Noise can cause false regime shifts.

**Solution**: Require sustained evidence before declaring transition.

```python
# Minimum 7 days in current regime before transition
min_regime_duration = 86400 × 7 seconds

# Fast transitions for emergencies
if new_regime in (COLLAPSE, DISTURBANCE):
    transition_immediately()

# Otherwise require stability confirmation
if time_in_regime < min_duration:
    stay_in_current_regime()
else:
    check_transition_stability()  # Require 3/5 confirming signals
```

#### Regime Inference

```python
def classify_regime(health, trend, volatility):
    if health < 0.30:
        return COLLAPSE
    
    if volatility > 0.15:
        return DISTURBANCE
    
    if health >= 0.80 and |trend| < 0.05:
        return STABLE_HEALTHY
    
    if health < 0.50 and |trend| < 0.05:
        return STABLE_DEGRADED
    
    if trend > 0.10:
        return IMPROVING
    
    if trend < -0.10:
        return DEGRADING
    
    # ... additional logic
```

---

## Precautionary Flags

### Severity-Weighted System

Each flag has a defined severity that affects score penalty:

| Flag | Severity | Description |
|------|----------|-------------|
| `regulatory_noncompliance` | 0.10 | Highest - blocks intervention |
| `near_irreversible_intervention` | 0.08 | Cannot easily undo |
| `aggregate_uncertainty_excessive` | 0.07 | Too much unknown |
| `ecosystem_crisis_emergency_authorized` | 0.06 | Emergency override used |
| `critical_thresholds_breached` | 0.06 | Health metrics alarming |
| `model_disagreement` | 0.05 | Models don't agree |
| `measurement_uncertainty_high` | 0.04 | Poor sensor quality |
| `risk_profile_unacceptable` | 0.04 | Risk assessment failed |
| `observation_period_insufficient` | 0.03 | Need more time |
| `trend_instability` | 0.03 | Conditions not stable |
| `intervention_frequency_excessive` | 0.03 | Too many interventions |

### Penalty Calculation

```python
# Sum weighted severities
total_severity = sum(FLAG_SEVERITY[flag] for flag in flags)

# Cap at 25% maximum penalty
penalty = min(0.25, total_severity)

# Apply to base score
adjusted_score = base_score × (1 - penalty)
```

**Example:**
- Base trust score: 0.85
- Flags: `model_disagreement` (0.05), `trend_instability` (0.03), `measurement_uncertainty_high` (0.04)
- Total severity: 0.12
- Adjusted score: 0.85 × (1 - 0.12) = 0.748

---

## API Reference

### Primary Classes

#### OceanGovernor

Main orchestrator for ocean governance decisions.

```python
class OceanGovernor:
    def __init__(
        cfg: Optional[AileeConfig] = None,
        policy: Optional[OceanGovernancePolicy] = None
    )
    
    def evaluate(signals: OceanSignals) -> OceanDecisionResult
    
    def explain_decision(decision: OceanDecisionResult) -> str
    
    def get_last_event() -> Optional[OceanEvent]
    
    def export_events(since_ts: Optional[float] = None) -> List[OceanEvent]
    
    def get_ecosystem_trend() -> str  # "improving", "stable", "degrading"
    
    def get_regime_history() -> Dict[str, Any]
    
    def get_intervention_history() -> Dict[str, Any]
    
    def check_temporal_readiness(signals: OceanSignals) -> Tuple[bool, List[str]]
```

#### OceanSignals

Primary input structure for evaluation.

```python
@dataclass
class OceanSignals:
    # Core scores
    proposed_action_trust_score: float        # [0-1] From intervention model
    ecosystem_health_index: float             # [0-1] Current health
    measurement_reliability: float            # [0-1] Sensor quality
    
    # Detailed assessments
    ecosystem_health: Optional[EcosystemHealth]
    risk_assessment: Optional[RiskAssessment]
    regulatory_status: Optional[RegulatoryStatus]
    temporal_context: Optional[TemporalContext]
    
    # Multi-source validation
    sensor_readings: Tuple[OceanSensorReading, ...]
    model_predictions: Tuple[ModelPrediction, ...]
    
    # Classification
    ecosystem_type: EcosystemType
    intervention_category: InterventionCategory
    current_regime: EcosystemRegime
    
    # Metadata
    intervention_id: Optional[str]
    context: Dict[str, Any]
    timestamp: Optional[float]
```

#### OceanDecisionResult

Output structure from evaluation.

```python
@dataclass
class OceanDecisionResult:
    # Core decision
    intervention_authorized: bool
    authority_level: AuthorityLevel
    decision_outcome: DecisionOutcome
    
    # Scores
    validated_trust_score: float
    confidence_score: float
    
    # Explanation
    recommendation: str
    reasons: List[str]
    
    # Requirements
    staging_requirements: Optional[StagingRequirements]
    intervention_constraints: Optional[Dict[str, Any]]
    
    # Risk & flags
    risk_level: Optional[str]  # "low", "moderate", "high", "severe"
    precautionary_flags: Optional[List[str]]
    
    # Pipeline result
    ailee_result: Optional[DecisionResult]
    
    # Metadata
    metadata: Dict[str, Any]
```

### Convenience Functions

#### create_ocean_governor()

Factory for creating governors with ecosystem-specific defaults.

```python
def create_ocean_governor(
    ecosystem_type: EcosystemType = EcosystemType.UNKNOWN,
    intervention_category: InterventionCategory = InterventionCategory.UNKNOWN,
    precautionary_bias: float = 0.80,
    **policy_overrides
) -> OceanGovernor
```

**Example:**
```python
governor = create_ocean_governor(
    ecosystem_type=EcosystemType.CORAL_REEF,
    intervention_category=InterventionCategory.REEF_RESTORATION,
    precautionary_bias=0.90,  # Extra cautious
    min_observation_period_days=90.0  # Override default
)
```

#### validate_ocean_signals()

Pre-flight validation of signal structure.

```python
def validate_ocean_signals(
    signals: OceanSignals
) -> Tuple[bool, List[str]]

# Returns (is_valid, list_of_issues)
```

**Example:**
```python
valid, issues = validate_ocean_signals(signals)
if not valid:
    print(f"Validation failed: {issues}")
    # Handle errors before calling evaluate()
```

---

## Configuration Guide

### OceanGovernancePolicy

Domain-specific policy configuration.

```python
@dataclass
class OceanGovernancePolicy:
    # Classification
    ecosystem_type: EcosystemType = UNKNOWN
    intervention_category: InterventionCategory = UNKNOWN
    
    # Precautionary stance
    precautionary_bias: float = 0.80  # [0-1], higher = more cautious
    
    # Trust thresholds
    min_intervention_safety_score: float = 0.75
    min_ecosystem_health_for_intervention: float = 0.40
    min_measurement_reliability: float = 0.70
    
    # Risk limits
    max_ecological_risk: float = 0.30
    min_reversibility_score: float = 0.50
    max_cascade_risk: float = 0.60
    
    # Temporal requirements
    min_observation_period_days: float = 30.0
    min_trend_stability_days: float = 14.0
    min_time_between_interventions_days: float = 90.0
    
    # Model consensus
    require_model_consensus: bool = True
    min_model_agreement_ratio: float = 0.75
    
    # Regulatory
    require_regulatory_approval: bool = True
    min_compliance_score: float = 0.95
    
    # Authority thresholds
    observe_only_threshold: float = 0.60
    stage_intervention_threshold: float = 0.75
    controlled_intervention_threshold: float = 0.85
    full_intervention_threshold: float = 0.95
    
    # Emergency override
    enable_emergency_override: bool = True
    emergency_health_threshold: float = 0.20
    
    # Logging
    max_event_log_size: int = 10000
```

### AileeConfig

Core trust pipeline configuration (auto-generated from ecosystem type).

```python
@dataclass
class AileeConfig:
    # Thresholds
    accept_threshold: float = 0.85  # Trust score required for SAFE
    borderline_low: float = 0.70
    borderline_high: float = 0.85
    
    # Component weights
    w_stability: float = 0.50    # Historical consistency
    w_agreement: float = 0.35    # Peer consensus
    w_likelihood: float = 0.15   # Statistical plausibility
    
    # History windows
    history_window: int = 200    # Samples for stability
    forecast_window: int = 30    # Samples for prediction
    
    # Grace conditions (leniency)
    grace_peer_delta: float = 0.12
    grace_min_peer_agreement_ratio: float = 0.70
    grace_forecast_epsilon: float = 0.15
    grace_max_abs_z: float = 2.0
    
    # Consensus requirements
    consensus_quorum: int = 3
    consensus_delta: float = 0.15
    consensus_pass_ratio: float = 0.75
    
    # Fallback behavior
    fallback_mode: str = "last_good"
    
    # Features
    enable_grace: bool = True
    enable_consensus: bool = True
    enable_audit_metadata: bool = True
```

### Ecosystem-Specific Defaults

```python
# Coral Reef (highly sensitive)
cfg = default_ocean_config(EcosystemType.CORAL_REEF)
# → accept_threshold = 0.90
# → w_stability = 0.55
# → history_window = 300
# → grace_peer_delta = 0.10 (stricter)

# Deep Sea (unknown dynamics)
cfg = default_ocean_config(EcosystemType.DEEP_SEA)
# → accept_threshold = 0.90
# → Same conservative settings as coral

# Coastal Zone (more resilient)
cfg = default_ocean_config(EcosystemType.COASTAL_ZONE)
# → accept_threshold = 0.82
# → w_stability = 0.48
# → history_window = 150

# Aquaculture (managed system)
cfg = default_ocean_config(EcosystemType.AQUACULTURE_ZONE)
# → accept_threshold = 0.80
# → w_stability = 0.45
# → Fastest decisions
```

---

## Explainability & Auditing

### Decision Explanation

Every decision includes a comprehensive plain-language explanation:

```python
explanation = governor.explain_decision(decision)
print(explanation)
```

**Output Structure:**

```
======================================================================
OCEAN GOVERNANCE DECISION EXPLANATION
======================================================================

DECISION: INTERVENTION_AUTHORIZED
Authority Level: CONTROLLED_INTERVENTION
Intervention Authorized: YES
Risk Level: MODERATE

Validated Trust Score: 0.82 / 1.00
Decision Confidence: 0.88 / 1.00

WHAT MATTERED MOST:
----------------------------------------------------------------------
• AILEE Pipeline Status: SAFE
• Aggregate Uncertainty: 0.28 (primary source: model_disagreement)
• Authority Ceiling from Uncertainty: CONTROLLED_INTERVENTION
• Precautionary Bias Applied: 0.80 (higher = more caution)

WHAT WOULD CHANGE THIS DECISION:
----------------------------------------------------------------------
• Increase trust score by 0.13
to reach full intervention (0.95)
• Address precautionary concerns:
  → Additional sensor deployments
  → Longer observation period
  → Multiple independent models showing agreement

INTERVENTION CONSTRAINTS:
----------------------------------------------------------------------
• monitoring_required: True
• reporting_frequency_days: 7
• enhanced_monitoring: True
• stakeholder_notification: True

RECOMMENDED ACTION:
----------------------------------------------------------------------
→ Controlled Intervention With Monitoring
```

### Event Logging

Every decision creates a structured `OceanEvent` for compliance reporting:

```python
# Get last event
event = governor.get_last_event()

# Export events since timestamp
events = governor.export_events(since_ts=start_time)

# Event structure
@dataclass
class OceanEvent:
    timestamp: float
    event_type: str  # "intervention_authorized", "staging_approved", etc.
    
    ecosystem_type: EcosystemType
    intervention_category: InterventionCategory
    authority_level: AuthorityLevel
    decision_outcome: DecisionOutcome
    
    proposed_action_trust_score: float
    ecosystem_health_index: float
    measurement_reliability: float
    
    reasons: List[str]
    ecosystem_regime: EcosystemRegime
    
    # Full context preserved
    ecosystem_health: Optional[EcosystemHealth]
    risk_assessment: Optional[RiskAssessment]
    regulatory_status: Optional[RegulatoryStatus]
    temporal_context: Optional[TemporalContext]
    aggregate_uncertainty: Optional[AggregateUncertainty]
    
    ailee_decision: Optional[DecisionResult]
    metadata: Dict[str, Any]
```

### Decision Delta Tracking

Track how metrics change between decisions:

```python
decision = governor.evaluate(signals)

if "decision_delta" in decision.metadata:
    delta = decision.metadata["decision_delta"]
    
    print(f"Trust Score Change: {delta.trust_score_delta:+.3f}")
    print(f"Health Change: {delta.ecosystem_health_delta:+.3f}")
    print(f"Uncertainty Change: {delta.uncertainty_delta:+.3f}")
    print(f"Regime Changed: {delta.regime_changed}")
    print(f"Authority Changed: {delta.authority_level_changed}")
    print(f"Time Since Last Decision: {delta.time_since_last_decision_seconds:.1f}s")
```

### Ecosystem Trend Analysis

```python
# Get overall trend
trend = governor.get_ecosystem_trend()
# → "improving", "stable", "degrading", "insufficient_data"

# Get regime distribution
regime_history = governor.get_regime_history()
print(regime_history)
# {
#   "current_regime": "STABLE_HEALTHY",
#   "regime_distribution": {
#     "STABLE_HEALTHY": 45,
#     "IMPROVING": 12,
#     "SEASONAL_VARIATION": 8
#   },
#   "total_observations": 65
# }

# Get intervention history
intervention_history = governor.get_intervention_history()
print(intervention_history)
# {
#   "total_interventions": 3,
#   "last_intervention_time": 1234567890.0,
#   "intervention_types": ["REEF_RESTORATION", "NUTRIENT_MANAGEMENT"],
#   "average_trust_score": 0.87
# }
```

---

## Best Practices

### 1. Start Conservative

```python
# Begin with high precautionary bias
governor = create_ocean_governor(
    ecosystem_type=EcosystemType.CORAL_REEF,
    precautionary_bias=0.90  # Very cautious
)

# Relax over time as confidence builds
governor_later = create_ocean_governor(
    ecosystem_type=EcosystemType.CORAL_REEF,
    precautionary_bias=0.80  # Normal caution
)
```

### 2. Always Validate Signals

```python
valid, issues = validate_ocean_signals(signals)
if not valid:
    log.error(f"Signal validation failed: {issues}")
    # Don't proceed with evaluate()
```

### 3. Use Multiple Models

```python
signals = OceanSignals(
    proposed_action_trust_score=0.85,
    model_predictions=(
        ModelPrediction("ensemble_model_v1", 0.87, 0.92),
        ModelPrediction("physics_model", 0.83, 0.88),
        ModelPrediction("ml_model", 0.86, 0.85),
    ),
    # ... other fields
)
```

### 4. Provide Rich Temporal Context

```python
temporal = TemporalContext(
    observation_period_days=60.0,  # Longer is better
    trend_stability_days=25.0,
    trend_confidence=0.85,  # Helps reduce temporal uncertainty
    time_since_last_intervention_days=120.0,
    seasonal_appropriateness="appropriate"
)
```

### 5. Track Critical Thresholds

```python
health = EcosystemHealth(
    overall_health_index=0.65,
    dissolved_oxygen_mg_l=5.8,  # Track hypoxia
    ph=7.95,  # Track acidification
    chlorophyll_a_ug_l=15.2,  # Track blooms
    species_richness_score=0.72
)

critical, issues = health.is_critical_threshold_breached()
if critical:
    log.warning(f"Critical thresholds breached: {issues}")
```

### 6. Use Staging for Unknown Interventions

```python
# First time trying this intervention type
if decision.authority_level == AuthorityLevel.STAGE_INTERVENTION:
    # Good! System is being cautious
    # Complete staging requirements
    requirements = decision.staging_requirements
    
    if requirements.required_observation_days:
        wait_days = requirements.required_observation_days
        # Continue monitoring
```

### 7. Respect Emergency Overrides

```python
if decision.authority_level == AuthorityLevel.EMERGENCY_RESPONSE:
    log.critical("Emergency intervention authorized")
    log.critical("Post-emergency review MANDATORY")
    
    # Execute intervention
    # ...
    
    # Schedule after-action review
    schedule_emergency_review(intervention_id)
```

### 8. Monitor Decision Deltas

```python
# Set up alerts for rapid changes
if "decision_delta" in decision.metadata:
    delta = decision.metadata["decision_delta"]
    
    if abs(delta.trust_score_delta) > 0.15:
        alert("Large trust score change detected")
    
    if delta.regime_changed:
        alert("Ecosystem regime transition detected")
    
    if abs(delta.ecosystem_health_delta) > 0.10:
        alert("Rapid health change detected")
```

### 9. Export Events for Compliance

```python
# Daily compliance export
daily_events = governor.export_events(
    since_ts=time.time() - 86400  # Last 24 hours
)

# Generate compliance report
for event in daily_events:
    compliance_log.write({
        "timestamp": event.timestamp,
        "ecosystem": event.ecosystem_type.value,
        "decision": event.decision_outcome.value,
        "trust_score": event.proposed_action_trust_score,
        "reasons": event.reasons,
        "regulatory_status": event.regulatory_status
    })
```

### 10. Understand Uncertainty Ceilings

```python
decision = governor.evaluate(signals)

# Check if uncertainty limited authority
if decision.metadata.get("authority_ceiling"):
    ceiling = decision.metadata["authority_ceiling"]
    actual = decision.authority_level.value
    
    if ceiling != actual:
        log.info(f"Authority capped at {ceiling} due to uncertainty")
        log.info(f"Reduce uncertainty to reach {actual}")
```

---

## Version History

### v1.0.1 - Production Grade (Current)

**Fixed:**
- Field name consistency: `intervention_safety_score` → `proposed_action_trust_score`
- `validated_safety_score` → `validated_trust_score`

**Added:**
- **Uncertainty-aware authority ceilings**: Aggregate uncertainty now caps maximum authority level
- **Severity-weighted precautionary flags**: Each flag has defined penalty weight
- **Regime transition hysteresis**: Prevents thrashing between ecosystem states (7-day minimum)
- **Explicit regulatory HOLD vs FAIL**: Clear distinction between pending vs denied
- **DecisionDelta tracking**: Monitor changes in trust, health, uncertainty between decisions
- Emergency override ceiling bypass with audit trail

**Improved:**
- UncertaintyCalculator: Added trend_confidence support
- PolicyEvaluator: Severity-weighted penalty computation
- RegimeTracker: Stability checks for non-emergency transitions
- Documentation: Comprehensive README with all features

### v1.0.0 - Initial Release

- Core ocean governance pipeline
- Multi-gate validation (health, risk, temporal, regulatory, measurement)
- Multi-model consensus with confidence weighting
- Staged authority levels (NO_ACTION → EMERGENCY_RESPONSE)
- Ecosystem regime tracking
- Precautionary flag system
- AILEE trust pipeline integration
- Event logging for compliance
- Explainability utilities

---

## Support & Contributing

### Issues & Questions

For production use, ensure:
- AILEE core pipeline (`ailee_trust_pipeline_v1`) is available
- All signal validations pass before evaluation
- Compliance logging is configured for your jurisdiction
- Emergency override procedures are documented

### Future Enhancements

Potential v1.1.0 features:
- Adaptive thresholds based on intervention success rate
- Multi-ecosystem interaction modeling
- Predictive regime transition warnings
- Automated staging requirement generation
- Integration with real-time sensor networks

---

**License**: MIT

---

**Last Updated**: December 2025   
**Status**: Production Ready ✅
