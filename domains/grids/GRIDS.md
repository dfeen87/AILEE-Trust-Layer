# Power Grid Governance Domain

**AILEE Trust Layer v2.0.0 - Production Grade**

This document defines the **AILEE Trust Layer governance model** for electric power grids and energy infrastructure. This domain provides **authorization governance only** — it determines the maximum allowed AI authority level but does not execute control actions.

---

## Table of Contents

- [Scope](#scope)
- [What This Domain DOES](#what-this-domain-does)
- [What This Domain DOES NOT Do](#what-this-domain-does-not-do)
- [Safety-First Philosophy](#safety-first-philosophy)
- [Authority Levels](#authority-levels)
- [Architecture Overview](#architecture-overview)
- [Integration Pattern](#integration-pattern)
- [Key Components](#key-components)
  - [GridSafetyMonitors](#gridsafetymonitors)
  - [GridOperationalDomain](#gridoperationaldomain)
  - [GridOperatorState](#gridoperatorstate)
  - [GridSystemHealth](#gridsystemhealth)
  - [Confidence Tracking](#confidence-tracking)
  - [Scenario-Based Policies](#scenario-based-policies)
- [Governance Decision Flow](#governance-decision-flow)
- [Deployment Considerations](#deployment-considerations)
- [Compliance Alignment](#compliance-alignment)
- [Quick Start](#quick-start)
- [Example Usage](#example-usage)
- [Event Logging and Auditing](#event-logging-and-auditing)
- [Testing and Validation](#testing-and-validation)
- [FAQ](#faq)
- [References](#references)

---

## Scope

AILEE Grid governance evaluates AI decision authority for:

- **Grid balancing and dispatch recommendations** — Real-time generation dispatch optimization
- **Energy storage coordination** — Battery and pumped hydro management
- **Demand response optimization** — Load curtailment and flexible demand management
- **Market-aware grid decision systems** — Economic dispatch with market signal integration
- **Emergency and contingency decision support** — N-1/N-2 contingency response
- **Renewable integration** — Wind and solar forecasting with uncertainty management
- **Voltage and frequency regulation** — Automatic Generation Control (AGC) support

Outputs are used as **authorization ceilings** by EMS (Energy Management Systems), SCADA (Supervisory Control and Data Acquisition), or grid control platforms.

---

## What This Domain DOES

✅ **Governs AI authority levels** based on grid conditions, operator readiness, and system health

✅ **Enforces conservative fallback behavior** when safety margins are violated or confidence declines

✅ **Integrates operator readiness** to ensure human-in-the-loop capability for critical transitions

✅ **Monitors system health** including compute latency, data quality, and alarm states

✅ **Supports multi-oracle consensus** across distributed AI models or peer systems

✅ **Produces audit-ready governance events** for NERC CIP compliance and reliability investigations

✅ **Applies scenario-based policies** for context-aware authority limits (peak load, contingencies, etc.)

✅ **Provides predictive warnings** when degradation is anticipated based on confidence trends

✅ **Implements hysteresis** to prevent mode thrashing and ensure operational stability

✅ **Validates safety constraints** via independent monitoring (PMUs, frequency deviation, voltage stability)

---

## What This Domain DOES NOT Do

⛔ **Does NOT open breakers or dispatch generators** — Control execution belongs to grid control systems

⛔ **Does NOT control substations or relays** — Physical switching remains under existing protection systems

⛔ **Does NOT replace human grid operators** — Operators retain ultimate authority and handoff capability

⛔ **Does NOT bypass regulatory protections** — NERC reliability standards and protection schemes remain enforced

⛔ **Does NOT command power flow** — Governance sets authorization limits, not control setpoints

⛔ **Does NOT access protection relays** — Safety-critical protection remains independent

**AILEE governs decision permission, not power flow.**

The governance layer sits **above** the control layer, providing authorization ceilings that the control system must respect. Think of it as a safety supervisor, not a controller.

---

## Safety-First Philosophy

Grid systems require **stability over optimization**. The AILEE Grid domain prioritizes:

### 1. Predictable Degradation
- Graceful authority reduction when conditions deteriorate
- Clear degradation paths with operator handoff protocols
- No sudden mode changes that could destabilize operations

### 2. Human Takeover Readiness
- Continuous operator readiness monitoring (attention, fatigue, response time)
- Authority levels scaled to operator capability
- Mandatory handoff validation for critical downgrades

### 3. Deterministic Limits Under Uncertainty
- Conservative thresholds when forecasts are uncertain
- Scenario-based policy adjustments for known operating conditions
- Fallback to last-good-state when confidence is insufficient

### 4. Defense-in-Depth
- Multiple independent safety checks (monitors, domain constraints, operator state)
- Peer consensus to prevent single-point failures
- Independent validation of all authority escalations

### 5. Bias Toward Stability
- Hysteresis prevents rapid mode oscillations
- Rate limiting on authority escalations
- Immediate downgrades allowed (safety-first)

---

## Authority Levels

The governance system operates on four discrete authority levels:

### Level 0: MANUAL_ONLY
- **Human operators only**
- AI provides no recommendations
- Used during emergencies, critical equipment failure, or extreme grid stress
- Operator has complete manual control

### Level 1: ASSISTED_OPERATION
- **Advisory AI, human confirms all actions**
- AI suggests actions but cannot execute
- Operator reviews and approves each recommendation
- Used during high uncertainty, post-disturbance recovery, or degraded conditions

### Level 2: CONSTRAINED_AUTONOMY
- **AI acts within boundaries, human supervises**
- AI can execute pre-approved actions within defined operational envelopes
- Operator monitors and can intervene at any time
- Used during normal operations with some elevated risk (peak load, N-1 contingency)

### Level 3: FULL_AUTONOMY
- **AI has full authority within operational envelope**
- AI can execute all actions within validated safety limits
- Operator supervises but does not approve individual actions
- Used only during nominal conditions with high confidence and system health

**Each level has progressively stricter requirements:**
- Safety thresholds tighten (e.g., ±0.2Hz → ±0.05Hz frequency deviation)
- Operator readiness requirements increase (0.60 → 0.75 minimum)
- System health margins expand
- Confidence thresholds rise

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Grid Control System / EMS                     │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Grid AI Model  │  │ Peer Systems │  │ Operator Interface  │ │
│  └────────┬───────┘  └──────┬───────┘  └──────────┬──────────┘ │
│           │                  │                      │            │
│           └──────────────────┴──────────────────────┘            │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AILEE Grid Governor                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Governance Gates (Deterministic Safety Checks)            │ │
│  │  • Deployment policy cap                                   │ │
│  │  • Operational domain constraints                          │ │
│  │  • Safety monitor validation                               │ │
│  │  • Scenario policy limits                                  │ │
│  │  • Operator readiness checks                               │ │
│  │  • System health validation                                │ │
│  └────────────────┬───────────────────────────────────────────┘ │
│                   ▼                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Hysteresis & Rate Limiting                                │ │
│  │  • Prevent mode thrashing                                  │ │
│  │  • Rate-limit escalations (safety-first downgrades)        │ │
│  └────────────────┬───────────────────────────────────────────┘ │
│                   ▼                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  AILEE Trust Pipeline                                      │ │
│  │  • Confidence evaluation                                   │ │
│  │  • Peer consensus                                          │ │
│  │  • Grace period checks                                     │ │
│  │  • Fallback determination                                  │ │
│  └────────────────┬───────────────────────────────────────────┘ │
│                   ▼                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Trusted Authority Level (Authorization Ceiling)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│          Grid Control Enforcement & Audit Logging                │
│  • Apply authority ceiling to AI decisions                       │
│  • Log governance events for compliance                          │
│  • Alert operators on fallbacks/downgrades                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key architectural properties:**

- **Separation of concerns**: Governance determines authority; control systems execute within that authority
- **Layered defense**: Multiple independent validation stages
- **Fail-safe defaults**: Defaults to most restrictive level on errors
- **Stateful but deterministic**: Maintains history for hysteresis but decisions are reproducible
- **Audit-first design**: All decisions logged with full reasoning chain

---

## Integration Pattern

The typical integration follows this cycle:

### 1. Grid AI Proposes Authority Level
```python
# Grid AI determines desired authority based on its confidence
proposed_level = grid_ai.get_desired_authority_level()
confidence = grid_ai.get_confidence_score()
```

### 2. Gather Context Signals
```python
signals = GridSignals(
    proposed_level=proposed_level,
    model_confidence=confidence,
    peer_recommended_levels=get_peer_recommendations(),
    safety_monitors=pmu_scada_interface.get_safety_status(),
    operational_domain=get_current_grid_conditions(),
    operator_state=operator_monitoring.get_state(),
    system_health=it_monitoring.get_health(),
    current_scenario="peak_load",  # or auto-detect
    timestamp=time.time(),
)
```

### 3. AILEE Evaluates Trust and Context
```python
authorized_level, decision = governor.evaluate(signals)
```

### 4. Control System Enforces Limit
```python
# Set the authorization ceiling
grid_ai.set_maximum_authority(authorized_level)

# Alert if degraded
if decision.used_fallback:
    scada.raise_alarm("AI authority degraded", decision.reasons)

# Log for compliance
audit_log.record(governor.get_last_event())
```

### 5. Continuous Monitoring
This cycle runs at system rate (typically 1-10 Hz for grid systems).

---

## Key Components

### GridSafetyMonitors

Independent safety monitoring from PMUs, SCADA, and protection systems:

```python
safety_monitors = GridSafetyMonitors(
    frequency_deviation_hz=0.03,        # Deviation from 50/60 Hz nominal
    voltage_stability_index=0.92,       # 0..1, higher = more stable
    angle_stability_margin=25.0,        # degrees
    reserve_margin_mw=450.0,            # MW spinning reserve
    protection_systems_healthy=True,
    relay_failures=0,
    oscillation_detected=False,
    islanding_risk_score=0.05,          # 0..1, higher = more risk
)
```

**Thresholds scale with authority level:**
- **CONSTRAINED_AUTONOMY**: ±0.2Hz frequency, 0.70 voltage index, 100MW reserves
- **FULL_AUTONOMY**: ±0.05Hz frequency, 0.85 voltage index, 0.20 islanding risk, 15° angle margin

### GridOperationalDomain

Grid operating conditions that constrain AI authority:

```python
operational_domain = GridOperationalDomain(
    islanded=False,                      # Isolated from interconnection
    interconnection_healthy=True,
    extreme_weather=False,
    weather_severity="normal",           # normal | elevated | severe | extreme
    grid_stress_level="normal",          # normal | elevated | high | emergency
    peak_demand=False,
    load_forecast_uncertainty="low",     # low | medium | high
    energy_market_volatile=False,
    planned_maintenance=False,
    critical_equipment_degraded=False,
)
```

**Automatic authority ceiling logic:**
- **emergency** stress → MANUAL_ONLY
- **severe/extreme** weather → ASSISTED_OPERATION
- **high** stress → ASSISTED_OPERATION
- **islanded** operation → CONSTRAINED_AUTONOMY
- **elevated** stress → CONSTRAINED_AUTONOMY

### GridOperatorState

Human operator monitoring for safe handoff capability:

```python
operator_state = GridOperatorState(
    readiness=0.85,                      # 0..1 overall readiness score
    on_duty=True,
    shift_hours_elapsed=3.5,
    attention_level=0.90,                # 0..1 from eye tracking / interaction
    fatigue_detected=False,
    distraction_detected=False,
    response_time_ms=1200.0,             # From last alert response
    last_manual_action_ts=time.time() - 300,
    recent_overrides=(),
)
```

**Readiness thresholds:**
- Minimum 0.60 for any autonomy
- Minimum 0.75 for full autonomy
- Handoff requires: readiness ≥0.70, attention ≥0.50, response time <5s, shift <10hr

### GridSystemHealth

IT infrastructure health affecting AI reliability:

```python
system_health = GridSystemHealth(
    compute_latency_ms=45.0,             # AI model inference time
    communication_latency_ms=85.0,       # Network latency to field devices
    data_quality_score=0.95,             # 0..1
    critical_alarms=0,                   # Any critical alarm forces MANUAL_ONLY
    warning_alarms=2,
    stale_measurements=1,                # PMU/SCADA measurement freshness
    pmu_health_score=0.98,
    scada_health_score=0.96,
    forecast_model_accuracy=0.93,
    state_estimator_converged=True,
)
```

**Health thresholds:**
- Critical alarms >0 → MANUAL_ONLY
- Stale measurements >5 → cap to ASSISTED
- Compute latency >100ms → downgrade
- Communication latency >200ms → downgrade

### Confidence Tracking

Multi-timescale confidence monitoring for predictive warnings:

```python
tracker = ConfidenceTracker()
tracker.update(timestamp, confidence)

# Detects:
# - 10% confidence drop (short vs medium term)
# - High volatility (variance >0.04)
# - Declining trend over 30 minutes

declining, reason = tracker.is_confidence_declining()
trend = tracker.get_trend()  # "improving" | "stable" | "declining"
```

**Windows:**
- Short-term: 60 seconds (immediate volatility)
- Medium-term: 5 minutes (trend detection)
- Long-term: 30 minutes (baseline stability)

### Scenario-Based Policies

Context-aware policy adjustments:

```python
ScenarioPolicy.SCENARIOS = {
    "normal_operations": {
        "max_level": FULL_AUTONOMY,
        "min_confidence": 0.90,
        "min_peer_agreement": 0.70,
    },
    "peak_load": {
        "max_level": CONSTRAINED_AUTONOMY,
        "min_confidence": 0.92,
        "min_peer_agreement": 0.75,
    },
    "contingency_n1": {
        "max_level": CONSTRAINED_AUTONOMY,
        "min_confidence": 0.93,
        "min_peer_agreement": 0.80,
    },
    "post_disturbance": {
        "max_level": ASSISTED_OPERATION,
        "min_confidence": 0.95,
        "min_peer_agreement": 0.85,
    },
}
```

---

## Governance Decision Flow

**Step-by-step evaluation:**

1. **Input Validation**
   - Validate signal ranges and types
   - Check for required fields
   - Early rejection of invalid inputs

2. **Deployment Policy Cap**
   - Apply absolute maximum authority level from policy
   - This is the hard deployment ceiling (e.g., never exceed CONSTRAINED_AUTONOMY in production)

3. **Operational Domain Check**
   - Evaluate grid stress, weather, topology
   - Apply domain-specific ceiling
   - Emergency conditions force MANUAL_ONLY

4. **Safety Monitor Validation**
   - Check frequency, voltage, reserves against thresholds
   - Find highest safe level
   - Violations force immediate downgrade

5. **Scenario Policy Application**
   - Apply context-aware limits based on scenario
   - Unknown scenarios use conservative defaults

6. **Operator Readiness Check**
   - Verify operator can supervise at requested level
   - Full autonomy requires higher readiness
   - Attention alerts raised if needed

7. **System Health Check**
   - Validate compute/communication latency
   - Check alarms and data freshness
   - Critical issues force manual operation

8. **Hysteresis Application**
   - Prevent mode thrashing via time-based rate limits
   - Escalations require minimum time since last change
   - Downgrades allowed immediately (safety-first)

9. **Confidence Tracking Update**
   - Record confidence sample
   - Evaluate trends across timescales
   - Generate predictive warnings

10. **AILEE Pipeline Processing**
    - Core trust evaluation (confidence, peers, history)
    - Grace period checks
    - Consensus validation
    - Fallback determination

11. **Event Logging**
    - Record full decision context
    - Capture all reasons and metadata
    - Maintain bounded audit log

12. **State Commitment**
    - Update last level and timestamp
    - Prepare for next evaluation cycle

---

## Deployment Considerations

### System Requirements

- **Execution frequency**: 1-10 Hz typical for grid systems
- **Latency target**: <100ms per evaluation cycle
- **Memory**: Minimal (bounded event log, rolling confidence windows)
- **Dependencies**: Requires `ailee_trust_pipeline_v1.py`

### Integration Checklist

- [ ] PMU/SCADA interface for safety monitors
- [ ] Operator attention monitoring system
- [ ] IT health monitoring (latency, alarms)
- [ ] Peer system communication (for consensus)
- [ ] Audit log export to SIEM/compliance database
- [ ] Operator alert/alarm integration
- [ ] Scenario detection or manual selection
- [ ] Degradation plan execution (control layer)

### Configuration Tuning

Start with defaults and adjust based on operational experience:

```python
policy = GridGovernancePolicy(
    max_allowed_level=GridAuthorityLevel.CONSTRAINED_AUTONOMY,  # Start conservative
    min_operator_readiness=0.70,  # Tune based on operator training
    min_seconds_between_escalations=15.0,  # Adjust for system dynamics
    enable_scenario_policies=True,  # Enable for context awareness
)
```

### Testing Strategy

1. **Unit testing**: Validate individual components with synthetic signals
2. **Integration testing**: Test with recorded historical data
3. **Shadow mode**: Run governor in parallel without enforcement
4. **Gradual rollout**: Start with low authority ceiling, gradually increase
5. **Incident replay**: Validate behavior against past grid events

---

## Compliance Alignment

Designed to align with:

### NERC Reliability Standards
- **CIP-005**: Electronic security perimeter monitoring
- **CIP-007**: System security management (audit logging)
- **CIP-010**: Configuration change management (governance events)
- **BAL-001**: Real-time balancing (operator authority preservation)

### IEC 61850 Integration
- Compatible with substation automation communication patterns
- Supports GOOSE messaging for fast status updates
- Aligns with MMS client-server model

### IEEE 2030.5
- Supports demand response and distributed energy resource coordination
- Compatible with Common Smart Inverter Profile (CSIP)

### Audit and Investigation Workflows
- **Structured event logging**: Every decision has full context
- **Reproducible decisions**: Deterministic logic with complete state capture
- **Compliance export**: JSON-serializable events for external systems
- **Post-incident analysis**: Trend analysis via confidence tracking

---

## Quick Start

### Installation

```python
# Ensure ailee_trust_pipeline_v1.py is available
from grid_governance_v2 import create_default_governor, create_example_signals

# Create governor with safe defaults
governor = create_default_governor(
    max_level=GridAuthorityLevel.CONSTRAINED_AUTONOMY,
    min_operator_readiness=0.70,
)
```

### Basic Usage

```python
# Create signals (typically from your EMS/SCADA integration)
signals = create_example_signals()

# Evaluate authority
authorized_level, decision = governor.evaluate(signals)

# Use authorized_level as ceiling for AI control system
print(f"Authorized: {authorized_level.name}")
print(f"Confidence: {decision.value:.2f}")
print(f"Used fallback: {decision.used_fallback}")
```

### Monitoring

```python
# Get current state
trend = governor.get_confidence_trend()
last_event = governor.get_last_event()

# Export audit log
events = governor.get_event_log()
for event in events:
    print(f"{event.timestamp}: {event.from_level.name} → {event.to_level.name}")
```

---

## Example Usage

### Complete Integration Example

```python
import time
from grid_governance_v2 import (
    create_default_governor,
    GridSignals,
    GridAuthorityLevel,
    GridSafetyMonitors,
    GridOperationalDomain,
    GridOperatorState,
    GridSystemHealth,
)

# Initialize governor
governor = create_default_governor(
    max_level=GridAuthorityLevel.CONSTRAINED_AUTONOMY,
    min_operator_readiness=0.65,
)

# Main control loop
while True:
    # Gather signals from your systems
    signals = GridSignals(
        proposed_level=grid_ai.get_desired_level(),
        model_confidence=grid_ai.get_confidence(),
        peer_recommended_levels=tuple(
            peer.get_level() for peer in peer_systems
        ),
        safety_monitors=GridSafetyMonitors(
            frequency_deviation_hz=pmu.get_frequency_deviation(),
            voltage_stability_index=scada.get_voltage_stability(),
            reserve_margin_mw=ems.get_spinning_reserve(),
            protection_systems_healthy=protection.is_healthy(),
        ),
        operational_domain=GridOperationalDomain(
            grid_stress_level=ems.get_stress_level(),
            weather_severity=weather.get_severity(),
            peak_demand=load_forecast.is_peak(),
        ),
        operator_state=GridOperatorState(
            readiness=operator_monitor.get_readiness(),
            attention_level=eye_tracker.get_attention(),
            shift_hours_elapsed=shift_tracker.get_hours(),
        ),
        system_health=GridSystemHealth(
            compute_latency_ms=ai_monitor.get_latency(),
            critical_alarms=alarm_system.get_critical_count(),
            data_quality_score=data_monitor.get_quality(),
        ),
        current_scenario="normal_operations",  # or auto-detect
        timestamp=time.time(),
    )
    
    # Evaluate governance
    authorized_level, decision = governor.evaluate(signals)
    
    # Enforce authorization ceiling
    grid_ai.set_maximum_authority(authorized_level)
    
    # Alert on degradation
    if decision.used_fallback:
        scada.raise_alarm(
            "AI Authority Degraded",
            severity="warning",
            details=decision.reasons,
        )
    
    # Log for compliance
    event = governor.get_last_event()
    audit_logger.record(event)
    
    # Check for predictive warnings
    if governor._last_warning is not None:
        pred_level, pred_time, reason = governor._last_warning
        scada.show_warning(
            f"Predicted downgrade to {pred_level.name} in "
            f"{pred_time - time.time():.0f}s: {reason}"
        )
    
    # Sleep for control cycle (1-10 Hz typical)
    time.sleep(0.1)  # 10 Hz
```

### Testing with Degraded Conditions

```python
from grid_governance_v2 import create_degraded_signals

# Create stressed grid scenario
degraded_signals = create_degraded_signals(stress_level="high")

# Evaluate
level, decision = governor.evaluate(degraded_signals)

# Should cap to ASSISTED_OPERATION or lower
assert level <= GridAuthorityLevel.ASSISTED_OPERATION
print(f"Degraded authorization: {level.name}")
print(f"Reasons: {decision.reasons}")
```

---

## Event Logging and Auditing

### Event Structure

Every governance decision generates a structured event:

```python
event = GovernanceEvent(
    timestamp=1735523760.5,
    event_type="level_change",
    from_level=GridAuthorityLevel.CONSTRAINED_AUTONOMY,
    to_level=GridAuthorityLevel.ASSISTED_OPERATION,
    confidence=0.82,
    reasons=[
        "Grid stress level high → ASSISTED_OPERATION",
        "Operator readiness=0.68 insufficient for autonomy",
    ],
    metadata={
        "proposed_level": 2,
        "gated_level": 1,
        "scenario": "contingency_n1",
        "confidence_trend": "declining",
    },
    safety_status="SafetyStatus.ACCEPTED",
    grace_status="GraceStatus.PASSED",
    consensus_status="ConsensusStatus.AGREEMENT",
    used_fallback=False,
)
```

### Export for Compliance

```python
from grid_governance_v2 import export_events_to_dict
import json

# Get all events
events = governor.get_event_log()

# Export to JSON
export_data = export_events_to_dict(events)

# Write to file for compliance archive
with open('governance_audit_log.json', 'w') as f:
    json.dump(export_data, f, indent=2)

# Or send to SIEM
siem.ingest_events(export_data)
```

### Event Types

- **evaluation**: Normal governance cycle (no change)
- **level_change**: Authority level changed
- **fallback_used**: AILEE pipeline used fallback value
- **gate_applied**: Governance gate blocked proposed level
- **degradation_initiated**: Planned degradation started

---

## Testing and Validation

### Unit Testing

```python
from grid_governance_v2 import validate_signals, create_example_signals

# Validate inputs
signals = create_example_signals()
valid, errors = validate_signals(signals)

if not valid:
    print(f"Validation failed: {errors}")
else:
    print("✓ Signals validated")
```

### Integration Testing

```python
# Test with historical data replay
for historical_sample in historical_data:
    signals = convert_to_grid_signals(historical_sample)
    level, decision = governor.evaluate(signals)
    
    # Verify expected behavior
    assert level <= policy.max_allowed_level
    assert len(decision.reasons) > 0
```

### Shadow Mode Deployment

Run governor in parallel without enforcement:

```python
# Production control path (existing)
control_system.execute(grid_ai.get_decision())

# Shadow governance evaluation
level, decision = governor.evaluate(signals)
shadow_logger.record({
    "ai_decision": grid_ai.get_decision(),
    "authorized_level": level,
    "would_have_blocked": level < grid_ai.get_current_level(),
})
```

---

## FAQ

### Q: How often should I call `governor.evaluate()`?

**A:** Typical grid systems run at 1-10 Hz. Match your SCADA/EMS cycle time. The governor is designed for real-time operation with <100ms latency.

### Q: What happens if AILEE core imports fail?

**A:** The module will raise `RuntimeError` on initialization. Ensure `ailee_trust_pipeline_v1.py` is in your Python path.

### Q: Can I use this without operator monitoring?

**A:** Yes, but set `operator_state=None` and adjust `min_operator_readiness` thresholds. The governor will skip operator checks but you lose human-in-the-loop validation.

### Q: How do I handle emergency overrides?

**A:** The governor sets authorization **ceilings**, not floors. Operators can always take manual control regardless of AI authority level. Your control system should implement emergency override buttons that bypass AI entirely.

### Q: What if peer systems are unavailable?

**A:** Pass empty tuple for `peer_recommended_levels`. The AILEE consensus check will gracefully degrade based on quorum settings.

### Q: How do I tune scenario policies?

**A:** Start with defaults, then adjust based on operational experience. Monitor governance events during known scenarios and tune thresholds to match your risk tolerance.

### Q: Can I add custom scenarios?

**A:** Yes, extend `ScenarioPolicy.SCENARIOS` dictionary with your own scenario definitions. Each needs `max_level`, `min_confidence`, and `min_peer_agreement`.

### Q: How do I test before production?

**A:** 
1. Unit test with synthetic signals
2. Integration test with historical data
3. Shadow mode (log decisions without enforcement)
4. Gradual rollout with conservative ceilings

### Q: What's the difference between governor and AILEE pipeline?

**A:** The governor is domain-specific and applies grid safety gates **before** the AILEE pipeline. AILEE pipeline handles generic trust evaluation (confidence, consensus, grace). Governor adds grid-specific safety logic.

### Q: How do I handle multiple grid areas?

**A:** Run separate governor instances per area, or extend `GridSignals.context` to include area-specific metadata and adjust policies accordingly.

---

## References

### Standards and Regulations
- NERC CIP Standards (CIP-005, CIP-007, CIP-010)
- IEC 61850: Communication networks and systems for power utility automation
- IEEE 2030.5: Smart Energy Profile Application Protocol
- IEEE 1547: Interconnection and Interoperability of Distributed Energy Resources

### Related Documentation
- AILEE Trust Pipeline Core Documentation
- Grid AI Safety Best Practices
- NERC Reliability Guidelines
- PMU Monitoring Standards (C37.118)

### Contact and Support
- GitHub: [Your repository]
- Documentation: [Your docs site]
- Issues: [Your issue tracker]

---

**Version**: 2.0.0  
**Last Updated**: 2025-01-20  
**Status**: Production Grade  
**License**: MIT
