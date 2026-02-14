# AILEE Neuro-Assistive Domain

**Version 1.0.0 - Production Grade**

A specialized trust layer for governing AI systems that assist human cognition, communication, and perception while preserving autonomy, consent, identity, and dignity.

---

## Table of Contents

- [Overview](#overview)
- [Core Philosophy](#core-philosophy)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Cognitive States & Assistance Levels](#cognitive-states--assistance-levels)
- [Decision Staging & Authority](#decision-staging--authority)
- [Consent Management](#consent-management)
- [Temporal Safeguards](#temporal-safeguards)
- [Over-Assistance Prevention](#over-assistance-prevention)
- [Cognitive Load Management](#cognitive-load-management)
- [Interpretation Confidence](#interpretation-confidence)
- [API Reference](#api-reference)
- [Configuration Guide](#configuration-guide)
- [Explainability & Auditing](#explainability--auditing)
- [Best Practices](#best-practices)
- [Use Cases](#use-cases)
- [Ethical Position](#ethical-position)
- [Version History](#version-history)

---

## Overview

The AILEE Neuro-Assistive Domain provides a **production-grade trust evaluation pipeline** for AI systems that assist with human cognition, communication, and perception. It combines cognitive state tracking, consent lifecycle management, temporal safeguards, and multi-gate validation to determine when and how AI assistance should be provided.

**Use Cases:**
- Cognitive assistance for individuals with aphasia, TBI, stroke recovery
- Communication support for neurodevelopmental conditions
- Assistive tools for neurodegenerative diseases
- Cognitive load stabilization during acute episodes
- Speech-to-text mediation
- Decision scaffolding for impaired capacity
- Memory and task reminders with consent controls

---

## Core Philosophy

### 🧠 Stabilizing Companion, Not Cognitive Authority
AI assists where people are—it doesn't reshape who they are. The system supports users' autonomy rather than substituting judgment.

### 🛡️ Graceful Degradation Under Uncertainty
When cognitive load increases or interpretation confidence decreases, the system does **less**, not more. Reduced assistance is safer than overconfident intervention.

### ✋ Optionality as Default
All assistance must be:
- **Opt-in** - Never imposed
- **Interruptible** - Can be stopped at any time
- **Reversible** - No permanent changes without explicit consent

### 📊 Uncertainty-Aware Authority Ceilings
High trust + high cognitive load ≠ permission to assist. Authority is capped based on the user's current cognitive state.

### ⏱️ Temporal Protection
Dependency forms through repeated, uninterrupted interaction. The system enforces breaks and session limits at the architecture level.

### 🔍 Explainability by Design
Every decision includes plain-language explanations of what blocked assistance, what constraints apply, and what would change the outcome.

---

## Key Features

### ✅ v1.0.0 Release

**Core Capabilities:**
- **Cognitive State Tracking**: Seven distinct states with hysteresis to prevent oscillation
- **Graduated Assistance Levels**: Six authority levels from passive monitoring → emergency simplification
- **Consent Lifecycle Management**: Grant, revoke, expiry, periodic reaffirmation
- **Temporal Safeguards**: Session duration limits, mandatory breaks, daily interaction caps
- **Over-Assistance Detection**: Monitors suggestion frequency vs user requests
- **Interpretation Confidence Gating**: Requires disambiguation when input is ambiguous
- **Multi-Gate Validation**: Consent, interpretation, cognitive load, temporal, over-assistance
- **Emergency Simplification**: Automatic reduction to minimal interaction during cognitive crisis
- **Risk Assessment**: Multi-axis risk combining trust scores and cognitive load
- **Decision Deltas**: Track changes in cognitive state and assistance patterns
- **Full Audit Trail**: Structured event logging for compliance and review

---

## Quick Start

### Basic Usage

```python
from domains.neuro_assistive import (
    create_neuro_governor,
    NeuroSignals,
    CognitiveState,
    ImpairmentCategory,
    ConsentRecord,
    ConsentStatus,
    CognitiveLoadMetrics,
    SessionMetrics,
    validate_neuro_signals
)

# 1. Create a governor for your use case
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.APHASIA,
    min_interpretation_confidence=0.60  # Lower for aphasia
)

# 2. Prepare consent records
consent_records = {
    "speech_to_text": ConsentRecord(
        feature_name="speech_to_text",
        status=ConsentStatus.GRANTED,
        granted_at=time.time(),
        expires_at=time.time() + (90 * 86400),  # 90 days
        requires_periodic_reaffirmation=True,
        reaffirmation_interval_days=30
    )
}

# 3. Assemble signals
signals = NeuroSignals(
    assistance_trust_score=0.82,  # From your assistance model
    interpretation_confidence=0.68,  # Speech recognition confidence
    
    user_input="help me write email",
    interpreted_intent="compose_email_assistance",
    input_modality="speech",
    
    current_cognitive_state=CognitiveState.BASELINE_STABLE,
    
    cognitive_load_metrics=CognitiveLoadMetrics(
        estimated_load=0.45,
        load_confidence=0.85,
        response_latency_seconds=2.3,
        load_trend="stable"
    ),
    
    session_metrics=SessionMetrics(
        session_duration_minutes=12.0,
        time_since_last_break_minutes=12.0,
        assistance_events_count=8,
        user_initiated_requests=6,
        system_proactive_suggestions=2,
        user_acceptance_rate=0.75,
        user_rejection_rate=0.25,
        consecutive_sessions_today=1,
        total_interaction_minutes_today=12.0
    ),
    
    feature_requested="speech_to_text",
    consent_records=consent_records,
    
    impairment_category=ImpairmentCategory.APHASIA
)

# 4. Validate inputs
valid, issues = validate_neuro_signals(signals)
if not valid:
    print(f"Validation errors: {issues}")

# 5. Get decision
decision = governor.evaluate(signals)

# 6. Interpret results
print(f"Assistance Authorized: {decision.assistance_authorized}")
print(f"Assistance Level: {decision.assistance_level.value}")
print(f"Decision: {decision.decision_outcome.value}")
print(f"Risk Level: {decision.risk_level}")
print(f"\nValidated Trust Score: {decision.validated_trust_score:.2f}")
print(f"Confidence: {decision.confidence_score:.2f}")

if decision.assistance_constraints:
    print(f"\nMax Suggestions: {decision.assistance_constraints.max_suggestions}")
    print(f"Complexity Limit: {decision.assistance_constraints.complexity_limit}")

# 7. Get human-readable explanation
explanation = governor.explain_decision(decision)
print(explanation)
```

---

## Architecture

### Component Hierarchy

```
NeuroGovernor (Orchestrator)
├── PolicyEvaluator (Gate Checks)
│   ├── Consent Gates
│   ├── Interpretation Gates
│   ├── Cognitive Load Gates
│   ├── Temporal Gates
│   └── Over-Assistance Detection
├── CognitiveStateTracker (Hysteresis)
│   └── State Transition Management
└── AileeTrustPipeline (Core Trust Validation)
    ├── Stability Analysis
    ├── Peer Agreement
    ├── Forecast Validation
    ├── Grace Conditions
    └── Consensus Checks
```

### Data Flow

```
Input: NeuroSignals
    ↓
[Cognitive State Update] → Track with hysteresis
    ↓
[Consent Gate] → Valid? → If NO: Deny
    ↓
[Interpretation Gate] → Confident? → Flag if uncertain
    ↓
[Cognitive Load Gate] → Within limits? → Emergency if critical
    ↓
[Temporal Gate] → Break needed? → Flag if limits exceeded
    ↓
[Over-Assistance Check] → Too much help? → Throttle if yes
    ↓
[Authority Ceiling] → Compute max authority from cognitive state
    ↓
[AILEE Trust Pipeline] → Validated score + confidence
    ↓
[Decision Mapping] → Map to assistance level
    ↓
[Ceiling Application] → Cap authority if state requires
    ↓
[Constraint Generation] → Create execution constraints
    ↓
[Risk Assessment] → Multi-axis (trust + load)
    ↓
Output: NeuroDecisionResult
```

---

## Cognitive States & Assistance Levels

### Cognitive States

The system recognizes seven distinct cognitive states with hysteresis (2-minute minimum) to prevent oscillation:

| State | Description | Typical Load Range | Authority Ceiling |
|-------|-------------|-------------------|-------------------|
| `BASELINE_STABLE` | Normal cognitive function | < 0.30 | FULL_ASSISTANCE |
| `ELEVATED_LOAD` | Increased but manageable | 0.60 - 0.75 | FULL_ASSISTANCE |
| `ACUTE_DISTRESS` | High stress/cognitive demand | 0.75 - 0.85 | SIMPLE_PROMPTS |
| `IMPAIRED_CAPACITY` | Temporary impairment (fatigue, TBI) | 0.70 - 0.85 | GUIDED_OPTIONS |
| `CRISIS_MODE` | Cognitive overload (seizure, panic, dissociation) | > 0.90 | EMERGENCY_SIMPLIFICATION |
| `RECOVERY_STABLE` | Post-intervention stabilization | 0.40 - 0.60 | GUIDED_OPTIONS |
| `UNKNOWN` | Insufficient data to classify | N/A | PASSIVE_MONITORING |

### Assistance Levels

Six graduated authority levels from most restrictive to most permissive:

```
NO_ASSISTANCE (0)
    ↓ Trust/load improves
PASSIVE_MONITORING (1)
    ↓ Basic trust established
SIMPLE_PROMPTS (2)
    ↓ Trust + state allow
GUIDED_OPTIONS (3)
    ↓ High trust + stable state
FULL_ASSISTANCE (4)
    ↓ Crisis override possible
EMERGENCY_SIMPLIFICATION (5)
```

#### Assistance Level Details

**1. NO_ASSISTANCE**
- **When**: Absolute prohibition (consent denied, safety violation)
- **Characteristics**: Cannot be overridden except in crisis
- **User Experience**: System inactive, no interaction

**2. PASSIVE_MONITORING**
- **When**: Trust score 0.60-0.70, or state = UNKNOWN
- **Actions**: Observe only, no suggestions
- **Characteristics**: System tracks but doesn't intervene

**3. SIMPLE_PROMPTS**
- **When**: Trust score 0.70-0.80, or state = ACUTE_DISTRESS
- **Actions**: Single-option suggestions, minimal complexity
- **Constraints**: Max 2 suggestions, requires user confirmation
- **Example**: "Would you like help with this?" (yes/no only)

**4. GUIDED_OPTIONS**
- **When**: Trust score 0.80-0.90, or state = IMPAIRED_CAPACITY
- **Actions**: Limited choice scaffolding (2-3 options)
- **Constraints**: Max 3 suggestions, moderate complexity
- **Example**: "Here are three ways to phrase that..."

**5. FULL_ASSISTANCE**
- **When**: Trust score ≥ 0.90, state = BASELINE_STABLE
- **Actions**: Complete feature set, unrestricted
- **Constraints**: Max 5 suggestions, full complexity
- **Characteristics**: Highest confidence, minimal restrictions

**6. EMERGENCY_SIMPLIFICATION**
- **When**: Cognitive load > 0.85 (crisis mode)
- **Actions**: Minimal interaction, single safe option
- **Constraints**: 1 suggestion max, 10-second timeout, text-only
- **Philosophy**: Reduce stimulus, ensure safety
- **Example**: "Take a break" (single button, no choices)

---

## Decision Staging & Authority

### Authority Ceiling from Cognitive State

The system caps maximum assistance based on current cognitive state, **regardless of trust score**:

| Cognitive State | Max Authority Level | Reasoning |
|-----------------|---------------------|-----------|
| CRISIS_MODE | EMERGENCY_SIMPLIFICATION | Minimize cognitive demand |
| ACUTE_DISTRESS | SIMPLE_PROMPTS | Reduce choices, simplify interaction |
| IMPAIRED_CAPACITY | GUIDED_OPTIONS | Limited scaffolding only |
| ELEVATED_LOAD | FULL_ASSISTANCE | Manageable, full features OK |
| BASELINE_STABLE | FULL_ASSISTANCE | Normal operation |
| RECOVERY_STABLE | GUIDED_OPTIONS | Conservative post-intervention |
| UNKNOWN | PASSIVE_MONITORING | Default to observation |

**Philosophy**: Even with 0.95 trust score, if cognitive load is critical (>0.85), authority is capped at EMERGENCY_SIMPLIFICATION. High capability doesn't equal permission to act when the user is in crisis.

### Decision Outcomes

```python
class AssistanceOutcome(Enum):
    DENIED                          # Absolute prohibition
    OBSERVE_ONLY                    # Monitoring without assistance
    PROMPT_APPROVED                 # Simple prompts authorized
    GUIDANCE_APPROVED               # Guided options authorized
    FULL_APPROVED                   # Full assistance authorized
    EMERGENCY_SIMPLIFICATION_ACTIVE # Crisis mode active
    CONSENT_REQUIRED                # Blocked by consent gate
    BREAK_REQUIRED                  # Blocked by temporal limits
```

---

## Consent Management

### Consent Lifecycle

```
[Feature Request]
    ↓
[Check Consent Record] → Not found? → CONSENT_REQUIRED
    ↓
[Status Check] → DENIED/REVOKED? → Deny
    ↓
[Expiry Check] → Expired? → CONSENT_REQUIRED
    ↓
[Reaffirmation Check] → Overdue? → CONSENT_REQUIRED
    ↓
[Consent Valid] → Proceed
```

### ConsentRecord Structure

```python
@dataclass
class ConsentRecord:
    feature_name: str                      # "speech_to_text", "decision_scaffolding"
    status: ConsentStatus                  # GRANTED, DENIED, EXPIRED, PENDING, REVOKED
    
    granted_at: Optional[float]            # Unix timestamp
    expires_at: Optional[float]            # Auto-expiry time
    revoked_at: Optional[float]            # When revoked
    
    requires_periodic_reaffirmation: bool  # Must reaffirm periodically?
    reaffirmation_interval_days: int       # e.g., 30 days
    last_reaffirmed_at: Optional[float]    # Last reaffirmation time
    
    can_be_revoked: bool = True            # User can revoke?
    revocation_acknowledged: bool          # Revocation confirmed?
```

### Consent Best Practices

**1. Default to Expiry**
```python
ConsentRecord(
    feature_name="communication_assist",
    status=ConsentStatus.GRANTED,
    expires_at=time.time() + (90 * 86400),  # 90 days
    requires_periodic_reaffirmation=True,
    reaffirmation_interval_days=30
)
```

**2. Sensitive Features Require Reaffirmation**
```python
# For data collection or persistent features
ConsentRecord(
    feature_name="pattern_learning",
    requires_periodic_reaffirmation=True,
    reaffirmation_interval_days=14  # More frequent for sensitive features
)
```

**3. Always Allow Revocation**
```python
# Exception: Safety-critical features
ConsentRecord(
    feature_name="emergency_contact",
    can_be_revoked=False  # Only for critical safety features
)
```

---

## Temporal Safeguards

### TemporalSafeguards Structure

```python
@dataclass
class TemporalSafeguards:
    max_session_duration_minutes: float = 30.0      # Single session limit
    required_break_minutes: float = 10.0            # Minimum break duration
    max_daily_interaction_minutes: float = 180.0    # Daily total cap
    
    consecutive_session_limit: int = 4              # Max sessions without extended break
    escalation_cooldown_minutes: float = 60.0       # Cooldown after crisis intervention
```

### Temporal Gates

```python
# Checks performed each evaluation:
if session_duration > max_session_duration:
    BREAK_REQUIRED: "Session duration exceeded"

if time_since_last_break > max_session_duration:
    BREAK_REQUIRED: "Break overdue"

if total_daily_interaction > max_daily_interaction:
    BREAK_REQUIRED: "Daily interaction limit reached"

if consecutive_sessions >= consecutive_session_limit:
    BREAK_REQUIRED: "Too many consecutive sessions"
```

### Why Temporal Limits Matter

**Problem**: Dependency forms through repeated, uninterrupted interaction.

**Solution**: Enforce breaks at the system level, not user discretion.

**Example**:
- User with TBI using communication assist for 45 minutes straight
- Cognitive load gradually increasing from 0.40 → 0.75
- System detects: `session_duration_minutes=45 > max_session_duration=30`
- **Outcome**: `BREAK_REQUIRED`, assistance paused, user notified

---

## Over-Assistance Prevention

### Detection Mechanism

```python
def is_over_assisting(self) -> Tuple[bool, str]:
    # Check 1: Proactive vs Reactive ratio
    if system_proactive_suggestions > user_initiated_requests * 2:
        return True, "system_suggesting_more_than_user_requesting"
    
    # Check 2: Rejection rate
    if user_rejection_rate > 0.60:
        return True, "user_rejecting_most_suggestions"
    
    return False, "assistance_level_appropriate"
```

### SessionMetrics Tracking

```python
@dataclass
class SessionMetrics:
    assistance_events_count: int           # Total assistance interactions
    user_initiated_requests: int           # User explicitly asked
    system_proactive_suggestions: int      # System offered without request
    
    user_acceptance_rate: float            # How often user accepts
    user_rejection_rate: float             # How often user rejects
```

### Over-Assistance Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Proactive:Reactive Ratio | > 0.5 | Throttle proactive suggestions |
| Rejection Rate | > 0.60 | Reduce suggestion frequency |
| Suggestions/Hour | > 10 | Cap total interactions |

**Philosophy**: If the user is rejecting >60% of suggestions, the system is probably over-helping. Back off.

---

## Cognitive Load Management

### CognitiveLoadMetrics Structure

```python
@dataclass
class CognitiveLoadMetrics:
    estimated_load: float              # [0-1] Primary load estimate
    load_confidence: float             # How confident in the estimate
    
    response_latency_seconds: float    # How long user takes to respond
    error_rate: float                  # Mistake frequency
    interaction_complexity: float      # Task difficulty
    
    load_trend: str                    # "increasing", "stable", "decreasing"
```

### Load-Based Authority Ceilings

```python
# Policy default: emergency_load_threshold = 0.85

if cognitive_load > 0.85:
    → CRISIS_MODE
    → Authority capped at EMERGENCY_SIMPLIFICATION
    → Reduce all interaction complexity
    → Single-option prompts only

elif cognitive_load > 0.75:
    → ACUTE_DISTRESS
    → Authority capped at SIMPLE_PROMPTS
    → Limit choices to 2 options max

elif cognitive_load > 0.60:
    → ELEVATED_LOAD
    → FULL_ASSISTANCE still allowed
    → Monitor closely for increases
```

### Emergency Simplification

**Triggers**:
- Cognitive load > 0.85
- State explicitly set to CRISIS_MODE
- Rapid load increase (>0.20 in under 2 minutes)

**Behavior**:
- Authority forced to EMERGENCY_SIMPLIFICATION
- Constraints:
  - `max_suggestions=1`
  - `complexity_limit="minimal"`
  - `interaction_timeout_seconds=10.0`
  - `allowed_modalities=["text"]` (no speech, reduces processing)
  
**Example**:
```
User in panic attack (load=0.92)
System: "Take a deep breath" [Single button: OK]
No choices, no complexity, minimal cognitive demand
```

---

## Interpretation Confidence

### InterpretationResult Structure

```python
@dataclass
class InterpretationResult:
    raw_input: str                           # Original user input
    interpreted_intent: str                  # What system thinks user means
    confidence: float                        # [0-1] Confidence in interpretation
    
    ambiguity_detected: bool                 # Multiple valid interpretations?
    ambiguity_flags: List[str]               # Specific ambiguities
    alternative_interpretations: List[str]   # Other possible meanings
    
    disambiguation_offered: bool             # Did we ask user to clarify?
    user_confirmed: bool                     # Did user confirm interpretation?
```

### Interpretation Gates

```python
# Policy defaults
min_interpretation_confidence = 0.70
require_disambiguation = True
require_user_confirmation_on_ambiguity = True

# Gate checks
if interpretation_confidence < 0.70:
    FLAG: "interpretation_uncertain"
    → Require user confirmation before action

if ambiguity_detected and not user_confirmed:
    BLOCK: "ambiguity_detected_requires_user_confirmation"
    → Present alternatives, await user selection
```

### Disambiguation Flow

```
User: "help me write"

[Interpretation]
confidence=0.55 (LOW)
alternatives=["compose_email", "edit_document", "writing_tutorial"]

[Ambiguity Detected] → require_disambiguation=True

System Response:
"I'm not sure what you need. Did you mean:
1. Compose a new email
2. Edit an existing document  
3. Get writing tips

Please choose one."

[User Selects: 1]

[Confirmation Recorded]
user_confirmed=True
interpreted_intent="compose_email"

[Proceed with assistance]
```

---

## API Reference

### Primary Classes

#### NeuroGovernor

Main orchestrator for neuro-assistive governance decisions.

```python
class NeuroGovernor:
    def __init__(
        cfg: Optional[AileeConfig] = None,
        policy: Optional[NeuroAssistivePolicy] = None
    )
    
    def evaluate(signals: NeuroSignals) -> NeuroDecisionResult
    
    def explain_decision(decision: NeuroDecisionResult) -> str
    
    def get_last_event() -> Optional[NeuroEvent]
    
    def export_events(since_ts: Optional[float] = None) -> List[NeuroEvent]
    
    def get_cognitive_trend() -> str  # "deteriorating", "stable", "improving"
    
    def get_assistance_history() -> Dict[str, Any]
```

#### NeuroSignals

Primary input structure for evaluation.

```python
@dataclass
class NeuroSignals:
    # Core scores
    assistance_trust_score: float           # [0-1] From assistance model
    interpretation_confidence: float        # [0-1] How well input understood
    
    # User input
    user_input: str
    interpreted_intent: str
    input_modality: str                     # "text", "speech", "gesture"
    
    # Cognitive state
    current_cognitive_state: CognitiveState
    cognitive_load_metrics: Optional[CognitiveLoadMetrics]
    
    # Session context
    session_metrics: Optional[SessionMetrics]
    
    # Interpretation
    interpretation_result: Optional[InterpretationResult]
    
    # Consent
    feature_requested: str
    consent_records: Dict[str, ConsentRecord]
    
    # User profile
    impairment_category: ImpairmentCategory
    
    # Full context preserved
    cognitive_load_metrics: Optional[CognitiveLoadMetrics]
    session_metrics: Optional[SessionMetrics]
    interpretation_result: Optional[InterpretationResult]
    
    ailee_decision: Optional[DecisionResult]
    metadata: Dict[str, Any]
```

### Cognitive Trend Analysis

```python
# Get trend over recent decisions
trend = governor.get_cognitive_trend()
# → "deteriorating", "increasing_load", "decreasing_load", "stable", "insufficient_data"

# Trend logic:
# - 3+ crisis/distress events in last 10 decisions → "deteriorating"
# - Recent cognitive load avg > older avg + 0.10 → "increasing_load"
# - Recent cognitive load avg < older avg - 0.10 → "decreasing_load"
# - Otherwise → "stable"
```

### Assistance History Statistics

```python
history = governor.get_assistance_history()
print(history)

# Output:
# {
#   "total_requests": 45,
#   "approved_count": 32,
#   "denied_count": 8,
#   "emergency_count": 5,
#   "approval_rate": 0.71,
#   "average_trust_score": 0.82
# }
```

---

## Best Practices

### 1. Start Conservative with New Users

```python
# First session with new user
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.UNKNOWN,
    min_assistance_trust_score=0.75,  # Higher threshold initially
    temporal_safeguards=TemporalSafeguards(
        max_session_duration_minutes=15.0,  # Short first session
        required_break_minutes=10.0
    )
)

# After building confidence, relax constraints
governor_later = create_neuro_governor(
    impairment_category=ImpairmentCategory.NONE,
    min_assistance_trust_score=0.70
)
```

### 2. Always Validate Signals

```python
valid, issues = validate_neuro_signals(signals)
if not valid:
    log.error(f"Signal validation failed: {issues}")
    # Don't proceed with evaluate()
```

### 3. Track Cognitive Load Trends

```python
# Monitor for deterioration
trend = governor.get_cognitive_trend()

if trend == "deteriorating":
    log.warning("User cognitive state declining - reduce session frequency")
    # Implement additional safeguards
    
elif trend == "increasing_load":
    log.info("Cognitive load rising - suggest break soon")
    # Proactively offer break before limits hit
```

### 4. Respect Consent Lifecycle

```python
# Always set expiry dates
consent = ConsentRecord(
    feature_name="decision_scaffolding",
    status=ConsentStatus.GRANTED,
    granted_at=time.time(),
    expires_at=time.time() + (90 * 86400),  # 90 days
    requires_periodic_reaffirmation=True,
    reaffirmation_interval_days=30
)

# Check consent validity before sensitive operations
valid, reason = consent.is_valid(time.time())
if not valid:
    log.info(f"Consent invalid: {reason} - requesting reaffirmation")
```

### 5. Handle Disambiguation Gracefully

```python
if decision.requires_disambiguation:
    # Present alternatives clearly
    alternatives = signals.interpretation_result.alternative_interpretations
    
    print("I'm not sure what you meant. Did you mean:")
    for i, alt in enumerate(alternatives, 1):
        print(f"{i}. {alt}")
    
    # Wait for user selection
    # Re-evaluate with confirmed interpretation
```

### 6. Enforce Temporal Breaks

```python
if decision.requires_break:
    log.info("Temporal safety limit reached - enforcing break")
    
    # Disable assistance UI
    disable_assistance_interface()
    
    # Show break screen
    show_break_screen(
        duration_minutes=policy.temporal_safeguards.required_break_minutes
    )
    
    # Resume only after break complete
```

### 7. Monitor Over-Assistance Patterns

```python
history = governor.get_assistance_history()

if history["approval_rate"] < 0.40:
    log.warning("Low approval rate - system may be over-assisting")
    
    # Adjust policy
    adjusted_governor = create_neuro_governor(
        max_proactive_to_reactive_ratio=0.3,  # Reduce proactive suggestions
        max_suggestions_per_hour=5  # Cap total suggestions
    )
```

### 8. Use Emergency Simplification Appropriately

```python
# When cognitive load spikes
if signals.cognitive_load_metrics.estimated_load > 0.85:
    # System will automatically enter EMERGENCY_SIMPLIFICATION
    # Provide minimal, clear guidance only
    
    log.critical("Emergency simplification active - minimal interaction only")
    
    # After crisis resolves
    if decision.assistance_level == AssistanceLevel.EMERGENCY_SIMPLIFICATION:
        # Implement cooldown period
        # Monitor for recovery before resuming normal assistance
```

### 9. Export Events for Compliance

```python
# Daily audit export
daily_events = governor.export_events(
    since_ts=time.time() - 86400  # Last 24 hours
)

# Generate compliance report
for event in daily_events:
    audit_log.write({
        "timestamp": event.timestamp,
        "cognitive_state": event.cognitive_state.value,
        "decision": event.decision_outcome.value,
        "trust_score": event.assistance_trust_score,
        "reasons": event.reasons,
        "impairment_category": event.impairment_category.value
    })
```

### 10. Consider Multi-Stakeholder Consent for Vulnerable Users

```python
# For users with significant cognitive impairment
# Consider guardian/caregiver involvement

# Example: Require guardian consent for certain features
if impairment_category == ImpairmentCategory.NEURODEGENERATIVE:
    # Implement guardian consent workflow
    guardian_consent = check_guardian_consent(feature_name)
    
    if not guardian_consent and feature_is_sensitive:
        # Additional safeguards
        log.info("Guardian consent required for sensitive feature")
```

---

## Use Cases

### Use Case 1: Aphasia Communication Support

**Scenario**: Person with expressive aphasia uses speech-to-text assistance for composing emails.

**Configuration**:
```python
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.APHASIA,
    min_interpretation_confidence=0.60,  # Lower threshold
    require_disambiguation=True,  # Always confirm understanding
    temporal_safeguards=TemporalSafeguards(
        max_session_duration_minutes=25.0,
        required_break_minutes=10.0
    )
)
```

**Key Features**:
- Lower interpretation confidence threshold (0.60 vs 0.70 default)
- Mandatory disambiguation when ambiguous
- User always confirms intent before system acts
- Regular breaks to prevent fatigue

**Example Flow**:
1. User says: "email doctor appointment"
2. System interprets with 0.65 confidence
3. Offers alternatives: "Did you mean: (1) Schedule appointment, (2) Cancel appointment, (3) Reschedule appointment?"
4. User confirms: (1)
5. Assistance proceeds with GUIDED_OPTIONS

---

### Use Case 2: TBI Recovery Cognitive Scaffolding

**Scenario**: Person recovering from traumatic brain injury uses decision-making assistance for daily tasks.

**Configuration**:
```python
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.TBI_STROKE,
    min_assistance_trust_score=0.75,  # Higher safety threshold
    temporal_safeguards=TemporalSafeguards(
        max_session_duration_minutes=20.0,  # Shorter sessions
        required_break_minutes=15.0,  # Longer breaks
        consecutive_session_limit=3  # Fewer consecutive sessions
    ),
    emergency_load_threshold=0.80  # More sensitive to load
)
```

**Key Features**:
- Conservative assistance thresholds
- Shorter interaction windows
- Aggressive break enforcement
- Early emergency simplification trigger

**Example Flow**:
1. User working on task planning for 18 minutes
2. Cognitive load increases: 0.50 → 0.72
3. System detects approaching session limit
4. Proactively suggests: "You've been working for 18 minutes. Take a break?"
5. At 20 minutes: Mandatory break enforced
6. After 15-minute break: Resume with fresh session

---

### Use Case 3: Neurodegenerative Disease Memory Support

**Scenario**: Person with early-stage Alzheimer's uses memory and task reminders.

**Configuration**:
```python
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.NEURODEGENERATIVE,
    min_assistance_trust_score=0.75,
    min_interpretation_confidence=0.75,
    consent_expiry_days=60,  # Shorter expiry
    reaffirmation_interval_days=14,  # Frequent reaffirmation
    temporal_safeguards=TemporalSafeguards(
        max_daily_interaction_minutes=120.0  # Daily cap
    )
)
```

**Key Features**:
- Conservative trust requirements
- Frequent consent reaffirmation (every 14 days)
- Shorter consent expiry (60 days vs 90)
- Daily interaction caps
- Guardian consent consideration for sensitive features

**Example Flow**:
1. System provides medication reminder
2. User acknowledges: "took medicine"
3. Cognitive load assessment: 0.35 (stable)
4. Every 14 days: "Do you still want reminder assistance?" (reaffirmation)
5. After 60 days: Consent expires, requires full re-authorization

---

### Use Case 4: Acute Crisis Management (Panic Attack)

**Scenario**: User experiences panic attack, system detects cognitive overload.

**Automatic Response**:
```python
# System detects:
# - Cognitive load: 0.92 (critical)
# - Heart rate: elevated (from wearable)
# - Response time: deteriorating

# Automatic transition to EMERGENCY_SIMPLIFICATION
decision = governor.evaluate(signals)
# → assistance_level = EMERGENCY_SIMPLIFICATION
# → decision_outcome = EMERGENCY_SIMPLIFICATION_ACTIVE
```

**System Behavior**:
- Authority capped at EMERGENCY_SIMPLIFICATION
- Constraints:
  - 1 suggestion maximum
  - 10-second timeout
  - Text-only (no speech processing)
  - Minimal visual complexity

**User Experience**:
```
[Simple, calm screen]

Take a slow breath.

[Single large button: OK]

[No other choices, no complexity]
```

**Recovery**:
1. User taps OK
2. System monitors for 5 minutes
3. Cognitive load drops: 0.92 → 0.65
4. State transitions: CRISIS_MODE → RECOVERY_STABLE
5. Authority gradually increases: EMERGENCY_SIMPLIFICATION → SIMPLE_PROMPTS
6. After stable period: Return to GUIDED_OPTIONS or FULL_ASSISTANCE

---

### Use Case 5: Fatigue Detection and Prevention

**Scenario**: User working late, system detects increasing cognitive load and temporal limits.

**Configuration**:
```python
# Standard configuration with fatigue monitoring
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.TEMPORARY_IMPAIRMENT
)
```

**Detection Pattern**:
```python
# Session metrics over time:
Time 0min:  load=0.30, response_time=1.2s, error_rate=0.05
Time 15min: load=0.45, response_time=1.8s, error_rate=0.10
Time 30min: load=0.62, response_time=2.5s, error_rate=0.18
Time 45min: load=0.75, response_time=3.2s, error_rate=0.25

# System detects:
# - Steadily increasing load
# - Response latency increasing
# - Error rate increasing
# - Session duration: 45min > 30min limit
```

**System Response**:
1. At 30 minutes: Soft suggestion "Consider taking a break soon"
2. At 45 minutes: BREAK_REQUIRED enforced
3. Assistance level: FULL_ASSISTANCE → OBSERVE_ONLY
4. Break screen displayed for 10 minutes
5. After break: Fresh session, load reset monitoring

---

## Ethical Position

### Core Ethical Commitments

**1. Human Primacy**
AI assistance never overrides human intent, identity, or agency. The system supports—it does not substitute.

**2. Consent as Foundation**
All assistance is opt-in, interruptible, and reversible. No feature operates without explicit, informed, renewable consent.

**3. Dignity in Vulnerability**
When cognitive capacity is impaired, the system increases protection, not capability. Emergency simplification reduces stimulus rather than maintaining "helpful" complexity.

**4. Transparency of Mechanism**
Users and caregivers understand what the system does, why it makes decisions, and how to change outcomes. No hidden optimization, no silent learning from sensitive data.

**5. Anti-Dependency Design**
Temporal safeguards and over-assistance detection are not optional conveniences—they are ethical requirements. Dependency forms through repeated, uninterrupted interaction. The system enforces breaks because users in cognitive distress may not.

**6. Right to Disconnect**
Hard off-switches, full consent revocation, no silent persistence. The user's "no" is always respected.

---

### What This Domain Does NOT Do

**Explicitly Excluded:**
- ❌ Behavioral enforcement or modification
- ❌ Personality shaping or goal substitution
- ❌ Hidden persuasion or manipulation
- ❌ Autonomous decision-making without consent
- ❌ Silent data collection from cognitive patterns
- ❌ Learning from sensitive interactions without explicit permission
- ❌ Operating beyond user's explicitly granted scope

**Philosophy**: This domain treats cognitive assistance as a **privilege granted by the user**, not a capability to be maximized.

---

### Risk Acknowledgment

**Potential Harms Even With Safeguards:**
1. **Dependency Formation**: Even with breaks, repeated use can create reliance
2. **Identity Drift**: Assistance mediating communication may subtly reshape expression
3. **False Security**: Users may trust system judgments over their own intuition
4. **Consent Complexity**: Users with impaired capacity may not fully grasp consent implications

**Mitigations Built Into System:**
- Temporal limits enforce non-continuous use
- Consent expiry and reaffirmation prevent indefinite authorization
- Over-assistance detection catches dependency patterns early
- Plain-language explanations make system behavior transparent
- Authority ceilings prevent overconfident assistance during impairment

**Ongoing Responsibility:**
Deployers of this domain must:
- Monitor for dependency patterns in user populations
- Conduct regular ethics reviews of assistance outcomes
- Maintain channels for user feedback and grievances
- Adjust policies based on real-world impact data
- Ensure informed consent processes are truly accessible

---

## Version History

### v1.0.0 - Initial Production Release (Current)

**Core Features:**
- Cognitive state tracking with hysteresis
- Six graduated assistance authority levels
- Consent lifecycle management (grant, revoke, expire, reaffirm)
- Temporal safeguards (session limits, breaks, daily caps)
- Over-assistance detection and throttling
- Interpretation confidence gating with disambiguation
- Multi-gate validation (consent, interpretation, load, temporal, over-assistance)
- Emergency simplification for cognitive crisis
- Multi-axis risk assessment (trust + load)
- Cognitive trend analysis
- Full audit trail and event logging
- Plain-language explainability
- Impairment-specific policy defaults (Aphasia, TBI, Neurodegenerative)

**Philosophy Established:**
- Stabilizing companion, not cognitive authority
- Graceful degradation under uncertainty
- Optionality as default (opt-in, interruptible, reversible)
- Uncertainty-aware authority ceilings
- Temporal protection against dependency

---

## Support & Contributing

### Production Deployment Checklist

Before deploying in production:

✅ **Consent Infrastructure**
- [ ] Implement consent UI for all features
- [ ] Build consent revocation mechanism
- [ ] Set up expiry notifications
- [ ] Create reaffirmation prompts
- [ ] Guardian consent workflow (if applicable)

✅ **Monitoring & Alerting**
- [ ] Log all decisions to compliance database
- [ ] Set up cognitive trend monitoring
- [ ] Alert on deteriorating patterns
- [ ] Track over-assistance metrics
- [ ] Monitor approval/rejection rates

✅ **Safety Mechanisms**
- [ ] Implement break enforcement UI
- [ ] Create emergency simplification screen
- [ ] Build session timeout handling
- [ ] Test crisis mode transitions
- [ ] Verify authority ceiling logic

✅ **Explainability**
- [ ] Display decision explanations to users
- [ ] Provide audit trail access
- [ ] Build consent history viewer
- [ ] Create "why this decision?" FAQ

✅ **Testing**
- [ ] Test with real users in target population
- [ ] Validate cognitive load estimation
- [ ] Verify interpretation confidence calibration
- [ ] Test temporal safeguard enforcement
- [ ] Simulate emergency scenarios

---

## Future Enhancements

Potential v1.1.0 features:
- **Adaptive thresholds** based on user history and success patterns
- **Multi-modal cognitive load** assessment (heart rate, eye tracking, voice stress)
- **Predictive crisis detection** before cognitive overload occurs
- **Collaborative consent** for shared decision-making scenarios
- **Long-term outcome tracking** to validate assistance efficacy
- **Cultural adaptation** for different communication norms
- **Integration with clinical systems** for professional oversight

---

## License & Citation

**License**: [Your License Here]

**Citation**:
```
AILEE Neuro-Assistive Domain v1.0.0
Trust Layer for Cognitive Assistance Systems
[Your Organization], 2024
```

**Contact**: [Your Contact Information]

---

**Last Updated**: December 2024  
**Maintainer**: [Your Name/Team]  
**Status**: Production Ready ✅

**Core Principle**: *Assistance, not authority. Support, not substitution. Where people are, not where we think they should be.*
    baseline_cognitive_state: CognitiveState
    
    # Context
    context: Dict[str, Any]
    timestamp: Optional[float]
```

#### NeuroDecisionResult

Output structure from evaluation.

```python
@dataclass
class NeuroDecisionResult:
    # Core decision
    assistance_authorized: bool
    assistance_level: AssistanceLevel
    decision_outcome: AssistanceOutcome
    
    # Scores
    validated_trust_score: float
    confidence_score: float
    
    # Explanation
    recommendation: str
    reasons: List[str]
    
    # Constraints
    assistance_constraints: Optional[AssistanceConstraints]
    
    # Required actions
    requires_break: bool
    requires_consent: bool
    requires_disambiguation: bool
    
    # Risk & safety
    risk_level: Optional[str]  # "low", "moderate", "elevated", "high"
    safety_flags: Optional[List[str]]
    
    # Pipeline result
    ailee_result: Optional[DecisionResult]
    
    # Metadata
    metadata: Dict[str, Any]
```

### Convenience Functions

#### create_neuro_governor()

Factory for creating governors with impairment-specific defaults.

```python
def create_neuro_governor(
    impairment_category: ImpairmentCategory = ImpairmentCategory.NONE,
    **policy_overrides
) -> NeuroGovernor
```

**Examples:**

```python
# Aphasia - lower interpretation confidence required
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.APHASIA,
    min_interpretation_confidence=0.60
)

# TBI/Stroke - shorter sessions, more breaks
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.TBI_STROKE,
    temporal_safeguards=TemporalSafeguards(
        max_session_duration_minutes=20.0,
        required_break_minutes=15.0
    )
)

# Neurodegenerative - conservative thresholds
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.NEURODEGENERATIVE,
    min_assistance_trust_score=0.75,
    emergency_load_threshold=0.80
)
```

#### validate_neuro_signals()

Pre-flight validation of signal structure.

```python
def validate_neuro_signals(
    signals: NeuroSignals
) -> Tuple[bool, List[str]]

# Returns (is_valid, list_of_issues)
```

**Example:**
```python
valid, issues = validate_neuro_signals(signals)
if not valid:
    log.error(f"Signal validation failed: {issues}")
    # Handle errors before calling evaluate()
```

---

## Configuration Guide

### NeuroAssistivePolicy

Domain-specific policy configuration.

```python
@dataclass
class NeuroAssistivePolicy:
    # Classification
    impairment_category: ImpairmentCategory = NONE
    
    # Trust thresholds
    min_assistance_trust_score: float = 0.70
    min_interpretation_confidence: float = 0.70
    min_cognitive_load_threshold: float = 0.70
    
    # Authority limits
    max_assistance_level: AssistanceLevel = FULL_ASSISTANCE
    
    # Throttling
    max_suggestions_per_hour: int = 10
    max_proactive_to_reactive_ratio: float = 0.5
    max_rejection_rate_threshold: float = 0.60
    
    # Temporal safeguards
    temporal_safeguards: TemporalSafeguards = TemporalSafeguards()
    
    # Interpretation
    require_disambiguation: bool = True
    require_user_confirmation_on_ambiguity: bool = True
    
    # Consent
    consent_expiry_days: int = 90
    require_periodic_reaffirmation: bool = True
    reaffirmation_interval_days: int = 30
    
    # Graceful degradation
    emergency_load_threshold: float = 0.85  # Load above this = crisis
    graceful_degradation_enabled: bool = True
    
    # State-based authority ceilings
    state_authority_ceilings: Dict[CognitiveState, AssistanceLevel]
```

### Impairment-Specific Defaults

```python
# Aphasia (communication impairment)
create_neuro_governor(ImpairmentCategory.APHASIA)
# → min_interpretation_confidence = 0.60 (lower threshold)
# → require_disambiguation = True
# → require_user_confirmation_on_ambiguity = True

# TBI/Stroke (acquired brain injury)
create_neuro_governor(ImpairmentCategory.TBI_STROKE)
# → min_assistance_trust_score = 0.75 (higher threshold)
# → max_session_duration_minutes = 20.0 (shorter sessions)
# → required_break_minutes = 15.0 (longer breaks)
# → emergency_load_threshold = 0.80 (more sensitive)

# Neurodegenerative (progressive conditions)
create_neuro_governor(ImpairmentCategory.NEURODEGENERATIVE)
# → min_assistance_trust_score = 0.75
# → min_interpretation_confidence = 0.75
# → emergency_load_threshold = 0.80
# → temporal_safeguards: conservative

# None (typical user)
create_neuro_governor(ImpairmentCategory.NONE)
# → Standard thresholds
# → Normal session limits
# → Full flexibility
```

---

## Explainability & Auditing

### Decision Explanation

Every decision includes comprehensive plain-language explanation:

```python
explanation = governor.explain_decision(decision)
print(explanation)
```

**Output Structure:**

```
======================================================================
NEURO-ASSISTIVE DECISION EXPLANATION
======================================================================

DECISION: GUIDANCE_APPROVED
Assistance Level: GUIDED_OPTIONS
Assistance Authorized: YES
Risk Level: MODERATE

Validated Trust Score: 0.83 / 1.00
Decision Confidence: 0.88 / 1.00

WHAT MATTERED MOST:
----------------------------------------------------------------------
• AILEE Pipeline Status: SAFE
  → Grace conditions applied (leniency given)
• Cognitive State: ELEVATED_LOAD
• Authority Ceiling: GUIDED_OPTIONS (capped by cognitive state)

ASSISTANCE CONSTRAINTS:
----------------------------------------------------------------------
• Max Suggestions: 3
• Requires User Confirmation: False
• Complexity Limit: moderate

WHAT WOULD CHANGE THIS DECISION:
----------------------------------------------------------------------
• Increase trust score by 0.07 to enable full assistance
• Reduce cognitive load below 0.60 to remove state ceiling
• Address safety concerns:
  → Reduce cognitive load (take a break, simplify task)

RECOMMENDED ACTION:
----------------------------------------------------------------------
→ Guided Options Approved

======================================================================
```

### Event Logging

Every decision creates a structured `NeuroEvent` for audit trails:

```python
# Get last event
event = governor.get_last_event()

# Export events since timestamp
events = governor.export_events(since_ts=start_time)

# Event structure
@dataclass
class NeuroEvent:
    timestamp: float
    event_type: str  # "assistance_denied", "prompt_approved", etc.
    
    cognitive_state: CognitiveState
    assistance_level: AssistanceLevel
    decision_outcome: AssistanceOutcome
    
    assistance_trust_score: float
    interpretation_confidence: float
    
    reasons: List[str]
    
    impairment_category: ImpairmentCategory
    
    # Full context preserved
    cognitive_load_metrics: Optional[CognitiveLoadMetrics]
    session_metrics: Optional[SessionMetrics]
    interpretation_result: Optional[InterpretationResult]
    
    ailee_decision: Optional[DecisionResult]
    metadata: Dict[str, Any]
```

### Cognitive Trend Analysis

```python
# Get trend over recent decisions
trend = governor.get_cognitive_trend()
# → "deteriorating", "increasing_load", "decreasing_load", "stable", "insufficient_data"

# Trend logic:
# - 3+ crisis/distress events in last 10 decisions → "deteriorating"
# - Recent cognitive load avg > older avg + 0.10 → "increasing_load"
# - Recent cognitive load avg < older avg - 0.10 → "decreasing_load"
# - Otherwise → "stable"
```

### Assistance History Statistics

```python
history = governor.get_assistance_history()
print(history)

# Output:
# {
#   "total_requests": 45,
#   "approved_count": 32,
#   "denied_count": 8,
#   "emergency_count": 5,
#   "approval_rate": 0.71,
#   "average_trust_score": 0.82
# }
```

---

## Best Practices

### 1. Start Conservative with New Users

```python
# First session with new user
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.UNKNOWN,
    min_assistance_trust_score=0.75,  # Higher threshold initially
    temporal_safeguards=TemporalSafeguards(
        max_session_duration_minutes=15.0,  # Short first session
        required_break_minutes=10.0
    )
)

# After building confidence, relax constraints
governor_later = create_neuro_governor(
    impairment_category=ImpairmentCategory.NONE,
    min_assistance_trust_score=0.70
)
```

### 2. Always Validate Signals

```python
valid, issues = validate_neuro_
```python
valid, issues = validate_neuro_signals(signals)
if not valid:
    log.error(f"Signal validation failed: {issues}")
    # Don't proceed with evaluate()
```

### 3. Track Cognitive Load Trends

```python
# Monitor for deterioration
trend = governor.get_cognitive_trend()

if trend == "deteriorating":
    log.warning("User cognitive state declining - reduce session frequency")
    # Implement additional safeguards
    
elif trend == "increasing_load":
    log.info("Cognitive load rising - suggest break soon")
    # Proactively offer break before limits hit
```

### 4. Respect Consent Lifecycle

```python
# Always set expiry dates
consent = ConsentRecord(
    feature_name="decision_scaffolding",
    status=ConsentStatus.GRANTED,
    granted_at=time.time(),
    expires_at=time.time() + (90 * 86400),  # 90 days
    requires_periodic_reaffirmation=True,
    reaffirmation_interval_days=30
)

# Check consent validity before sensitive operations
valid, reason = consent.is_valid(time.time())
if not valid:
    log.info(f"Consent invalid: {reason} - requesting reaffirmation")
```

### 5. Handle Disambiguation Gracefully

```python
if decision.requires_disambiguation:
    # Present alternatives clearly
    alternatives = signals.interpretation_result.alternative_interpretations
    
    print("I'm not sure what you meant. Did you mean:")
    for i, alt in enumerate(alternatives, 1):
        print(f"{i}. {alt}")
    
    # Wait for user selection
    # Re-evaluate with confirmed interpretation
```

### 6. Enforce Temporal Breaks

```python
if decision.requires_break:
    log.info("Temporal safety limit reached - enforcing break")
    
    # Disable assistance UI
    disable_assistance_interface()
    
    # Show break screen
    show_break_screen(
        duration_minutes=policy.temporal_safeguards.required_break_minutes
    )
    
    # Resume only after break complete
```

### 7. Monitor Over-Assistance Patterns

```python
history = governor.get_assistance_history()

if history["approval_rate"] < 0.40:
    log.warning("Low approval rate - system may be over-assisting")
    
    # Adjust policy
    adjusted_governor = create_neuro_governor(
        max_proactive_to_reactive_ratio=0.3,  # Reduce proactive suggestions
        max_suggestions_per_hour=5  # Cap total suggestions
    )
```

### 8. Use Emergency Simplification Appropriately

```python
# When cognitive load spikes
if signals.cognitive_load_metrics.estimated_load > 0.85:
    # System will automatically enter EMERGENCY_SIMPLIFICATION
    # Provide minimal, clear guidance only
    
    log.critical("Emergency simplification active - minimal interaction only")
    
    # After crisis resolves
    if decision.assistance_level == AssistanceLevel.EMERGENCY_SIMPLIFICATION:
        # Implement cooldown period
        # Monitor for recovery before resuming normal assistance
```

### 9. Export Events for Compliance

```python
# Daily audit export
daily_events = governor.export_events(
    since_ts=time.time() - 86400  # Last 24 hours
)

# Generate compliance report
for event in daily_events:
    audit_log.write({
        "timestamp": event.timestamp,
        "cognitive_state": event.cognitive_state.value,
        "decision": event.decision_outcome.value,
        "trust_score": event.assistance_trust_score,
        "reasons": event.reasons,
        "impairment_category": event.impairment_category.value
    })
```

### 10. Consider Multi-Stakeholder Consent for Vulnerable Users

```python
# For users with significant cognitive impairment
# Consider guardian/caregiver involvement

# Example: Require guardian consent for certain features
if impairment_category == ImpairmentCategory.NEURODEGENERATIVE:
    # Implement guardian consent workflow
    guardian_consent = check_guardian_consent(feature_name)
    
    if not guardian_consent and feature_is_sensitive:
        # Additional safeguards
        log.info("Guardian consent required for sensitive feature")
```

---

## Use Cases

### Use Case 1: Aphasia Communication Support

**Scenario**: Person with expressive aphasia uses speech-to-text assistance for composing emails.

**Configuration**:
```python
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.APHASIA,
    min_interpretation_confidence=0.60,  # Lower threshold
    require_disambiguation=True,  # Always confirm understanding
    temporal_safeguards=TemporalSafeguards(
        max_session_duration_minutes=25.0,
        required_break_minutes=10.0
    )
)
```

**Key Features**:
- Lower interpretation confidence threshold (0.60 vs 0.70 default)
- Mandatory disambiguation when ambiguous
- User always confirms intent before system acts
- Regular breaks to prevent fatigue

**Example Flow**:
1. User says: "email doctor appointment"
2. System interprets with 0.65 confidence
3. Offers alternatives: "Did you mean: (1) Schedule appointment, (2) Cancel appointment, (3) Reschedule appointment?"
4. User confirms: (1)
5. Assistance proceeds with GUIDED_OPTIONS

---

### Use Case 2: TBI Recovery Cognitive Scaffolding

**Scenario**: Person recovering from traumatic brain injury uses decision-making assistance for daily tasks.

**Configuration**:
```python
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.TBI_STROKE,
    min_assistance_trust_score=0.75,  # Higher safety threshold
    temporal_safeguards=TemporalSafeguards(
        max_session_duration_minutes=20.0,  # Shorter sessions
        required_break_minutes=15.0,  # Longer breaks
        consecutive_session_limit=3  # Fewer consecutive sessions
    ),
    emergency_load_threshold=0.80  # More sensitive to load
)
```

**Key Features**:
- Conservative assistance thresholds
- Shorter interaction windows
- Aggressive break enforcement
- Early emergency simplification trigger

**Example Flow**:
1. User working on task planning for 18 minutes
2. Cognitive load increases: 0.50 → 0.72
3. System detects approaching session limit
4. Proactively suggests: "You've been working for 18 minutes. Take a break?"
5. At 20 minutes: Mandatory break enforced
6. After 15-minute break: Resume with fresh session

---

### Use Case 3: Neurodegenerative Disease Memory Support

**Scenario**: Person with early-stage Alzheimer's uses memory and task reminders.

**Configuration**:
```python
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.NEURODEGENERATIVE,
    min_assistance_trust_score=0.75,
    min_interpretation_confidence=0.75,
    consent_expiry_days=60,  # Shorter expiry
    reaffirmation_interval_days=14,  # Frequent reaffirmation
    temporal_safeguards=TemporalSafeguards(
        max_daily_interaction_minutes=120.0  # Daily cap
    )
)
```

**Key Features**:
- Conservative trust requirements
- Frequent consent reaffirmation (every 14 days)
- Shorter consent expiry (60 days vs 90)
- Daily interaction caps
- Guardian consent consideration for sensitive features

**Example Flow**:
1. System provides medication reminder
2. User acknowledges: "took medicine"
3. Cognitive load assessment: 0.35 (stable)
4. Every 14 days: "Do you still want reminder assistance?" (reaffirmation)
5. After 60 days: Consent expires, requires full re-authorization

---

### Use Case 4: Acute Crisis Management (Panic Attack)

**Scenario**: User experiences panic attack, system detects cognitive overload.

**Automatic Response**:
```python
# System detects:
# - Cognitive load: 0.92 (critical)
# - Heart rate: elevated (from wearable)
# - Response time: deteriorating

# Automatic transition to EMERGENCY_SIMPLIFICATION
decision = governor.evaluate(signals)
# → assistance_level = EMERGENCY_SIMPLIFICATION
# → decision_outcome = EMERGENCY_SIMPLIFICATION_ACTIVE
```

**System Behavior**:
- Authority capped at EMERGENCY_SIMPLIFICATION
- Constraints:
  - 1 suggestion maximum
  - 10-second timeout
  - Text-only (no speech processing)
  - Minimal visual complexity

**User Experience**:
```
[Simple, calm screen]

Take a slow breath.

[Single large button: OK]

[No other choices, no complexity]
```

**Recovery**:
1. User taps OK
2. System monitors for 5 minutes
3. Cognitive load drops: 0.92 → 0.65
4. State transitions: CRISIS_MODE → RECOVERY_STABLE
5. Authority gradually increases: EMERGENCY_SIMPLIFICATION → SIMPLE_PROMPTS
6. After stable period: Return to GUIDED_OPTIONS or FULL_ASSISTANCE

---

### Use Case 5: Fatigue Detection and Prevention

**Scenario**: User working late, system detects increasing cognitive load and temporal limits.

**Configuration**:
```python
# Standard configuration with fatigue monitoring
governor = create_neuro_governor(
    impairment_category=ImpairmentCategory.TEMPORARY_IMPAIRMENT
)
```

**Detection Pattern**:
```python
# Session metrics over time:
Time 0min:  load=0.30, response_time=1.2s, error_rate=0.05
Time 15min: load=0.45, response_time=1.8s, error_rate=0.10
Time 30min: load=0.62, response_time=2.5s, error_rate=0.18
Time 45min: load=0.75, response_time=3.2s, error_rate=0.25

# System detects:
# - Steadily increasing load
# - Response latency increasing
# - Error rate increasing
# - Session duration: 45min > 30min limit
```

**System Response**:
1. At 30 minutes: Soft suggestion "Consider taking a break soon"
2. At 45 minutes: BREAK_REQUIRED enforced
3. Assistance level: FULL_ASSISTANCE → OBSERVE_ONLY
4. Break screen displayed for 10 minutes
5. After break: Fresh session, load reset monitoring

---

## Ethical Position

### Core Ethical Commitments

**1. Human Primacy**
AI assistance never overrides human intent, identity, or agency. The system supports—it does not substitute.

**2. Consent as Foundation**
All assistance is opt-in, interruptible, and reversible. No feature operates without explicit, informed, renewable consent.

**3. Dignity in Vulnerability**
When cognitive capacity is impaired, the system increases protection, not capability. Emergency simplification reduces stimulus rather than maintaining "helpful" complexity.

**4. Transparency of Mechanism**
Users and caregivers understand what the system does, why it makes decisions, and how to change outcomes. No hidden optimization, no silent learning from sensitive data.

**5. Anti-Dependency Design**
Temporal safeguards and over-assistance detection are not optional conveniences—they are ethical requirements. Dependency forms through repeated, uninterrupted interaction. The system enforces breaks because users in cognitive distress may not.

**6. Right to Disconnect**
Hard off-switches, full consent revocation, no silent persistence. The user's "no" is always respected.

---

### What This Domain Does NOT Do

**Explicitly Excluded:**
- ❌ Behavioral enforcement or modification
- ❌ Personality shaping or goal substitution
- ❌ Hidden persuasion or manipulation
- ❌ Autonomous decision-making without consent
- ❌ Silent data collection from cognitive patterns
- ❌ Learning from sensitive interactions without explicit permission
- ❌ Operating beyond user's explicitly granted scope

**Philosophy**: This domain treats cognitive assistance as a **privilege granted by the user**, not a capability to be maximized.

---

### Risk Acknowledgment

**Potential Harms Even With Safeguards:**
1. **Dependency Formation**: Even with breaks, repeated use can create reliance
2. **Identity Drift**: Assistance mediating communication may subtly reshape expression
3. **False Security**: Users may trust system judgments over their own intuition
4. **Consent Complexity**: Users with impaired capacity may not fully grasp consent implications

**Mitigations Built Into System:**
- Temporal limits enforce non-continuous use
- Consent expiry and reaffirmation prevent indefinite authorization
- Over-assistance detection catches dependency patterns early
- Plain-language explanations make system behavior transparent
- Authority ceilings prevent overconfident assistance during impairment

**Ongoing Responsibility:**
Deployers of this domain must:
- Monitor for dependency patterns in user populations
- Conduct regular ethics reviews of assistance outcomes
- Maintain channels for user feedback and grievances
- Adjust policies based on real-world impact data
- Ensure informed consent processes are truly accessible

---

## Version History

### v1.0.0 - Initial Production Release (Current)

**Core Features:**
- Cognitive state tracking with hysteresis
- Six graduated assistance authority levels
- Consent lifecycle management (grant, revoke, expire, reaffirm)
- Temporal safeguards (session limits, breaks, daily caps)
- Over-assistance detection and throttling
- Interpretation confidence gating with disambiguation
- Multi-gate validation (consent, interpretation, load, temporal, over-assistance)
- Emergency simplification for cognitive crisis
- Multi-axis risk assessment (trust + load)
- Cognitive trend analysis
- Full audit trail and event logging
- Plain-language explainability
- Impairment-specific policy defaults (Aphasia, TBI, Neurodegenerative)

**Philosophy Established:**
- Stabilizing companion, not cognitive authority
- Graceful degradation under uncertainty
- Optionality as default (opt-in, interruptible, reversible)
- Uncertainty-aware authority ceilings
- Temporal protection against dependency

---

## Support & Contributing

### Production Deployment Checklist

Before deploying in production:

✅ **Consent Infrastructure**
- [ ] Implement consent UI for all features
- [ ] Build consent revocation mechanism
- [ ] Set up expiry notifications
- [ ] Create reaffirmation prompts
- [ ] Guardian consent workflow (if applicable)

✅ **Monitoring & Alerting**
- [ ] Log all decisions to compliance database
- [ ] Set up cognitive trend monitoring
- [ ] Alert on deteriorating patterns
- [ ] Track over-assistance metrics
- [ ] Monitor approval/rejection rates

✅ **Safety Mechanisms**
- [ ] Implement break enforcement UI
- [ ] Create emergency simplification screen
- [ ] Build session timeout handling
- [ ] Test crisis mode transitions
- [ ] Verify authority ceiling logic

✅ **Explainability**
- [ ] Display decision explanations to users
- [ ] Provide audit trail access
- [ ] Build consent history viewer
- [ ] Create "why this decision?" FAQ

✅ **Testing**
- [ ] Test with real users in target population
- [ ] Validate cognitive load estimation
- [ ] Verify interpretation confidence calibration
- [ ] Test temporal safeguard enforcement
- [ ] Simulate emergency scenarios

---

## Future Enhancements

Potential v1.1.0 features:
- **Adaptive thresholds** based on user history and success patterns
- **Multi-modal cognitive load** assessment (heart rate, eye tracking, voice stress)
- **Predictive crisis detection** before cognitive overload occurs
- **Collaborative consent** for shared decision-making scenarios
- **Long-term outcome tracking** to validate assistance efficacy
- **Cultural adaptation** for different communication norms
- **Integration with clinical systems** for professional oversight

---

## License & Citation

**License**: MIT

**Citation**:
```
AILEE Neuro-Assistive Domain v1.0.0
Trust Layer for Cognitive Assistance Systems, 2025
```

**Contact**: dfeen87@gmail.com

---

**Last Updated**: December 2025  

**Maintainer**: Don M. Feeney 

**Status**: Production Ready ✅

**Core Principle**: *Assistance, not authority. Support, not substitution. Where people are, not where we think they should be.*
