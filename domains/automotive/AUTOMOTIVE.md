# AILEE Trust Layer — Automotive Autonomy Governance Domain

**Version 2.0.0 - Production Grade**

This document describes the **AILEE Trust Layer governance model** for automotive and vehicle-adjacent systems. This domain is focused on **decision authorization**, not vehicle control.

---

## Table of Contents

- [Scope](#scope)
- [What This Domain DOES](#what-this-domain-does)
- [What This Domain DOES NOT Do](#what-this-domain-does-not-do)
- [Core Components](#core-components)
  - [Autonomy Levels](#autonomy-levels)
  - [Safety Monitor Signals](#safety-monitor-signals)
  - [Operational Design Domain (ODD)](#operational-design-domain-odd)
  - [Driver State Monitoring](#driver-state-monitoring)
  - [System Health](#system-health)
- [Governance Architecture](#governance-architecture)
  - [Multi-Layer Decision Gates](#multi-layer-decision-gates)
  - [Hysteresis and Stability](#hysteresis-and-stability)
  - [Predictive Warnings](#predictive-warnings)
  - [Scenario-Based Policies](#scenario-based-policies)
- [Integration Pattern](#integration-pattern)
  - [Quick Start Example](#quick-start-example)
  - [Runtime Evaluation Loop](#runtime-evaluation-loop)
  - [Event Logging and Compliance](#event-logging-and-compliance)
- [Safety Features](#safety-features)
  - [Degradation Management](#degradation-management)
  - [Confidence Tracking](#confidence-tracking)
  - [Black Box Logging](#black-box-logging)
- [Design Principles](#design-principles)
- [Compliance and Certification](#compliance-and-certification)
- [API Reference](#api-reference)
- [Testing and Validation](#testing-and-validation)

---

## Scope

AILEE Automotive governance evaluates and constrains **AI decision authority** for systems such as:

- **Autonomous driving decision stacks** — L2+ to L4 systems requiring real-time authorization
- **Advanced driver-assistance systems (ADAS)** — Lane keeping, adaptive cruise control, automated parking
- **Fleet-level optimization and routing AI** — Multi-vehicle coordination and dispatch
- **Vehicle-to-infrastructure (V2I) coordination** — Smart intersection negotiation, traffic signal priority
- **Safety-critical perception fusion pipelines** — Sensor fusion governance for redundancy and fault tolerance

The output of this domain is an **authorization ceiling** that limits what an AI system is permitted to do at runtime.

---

## What This Domain DOES

✅ **Enforces deterministic approval** of AI driving decisions with full audit trails  
✅ **Applies multi-factor validation** using confidence, consensus, and stability thresholds  
✅ **Supports human-in-the-loop** with driver readiness and handoff capability monitoring  
✅ **Prevents unsafe escalation** of autonomy modes through governance gates  
✅ **Produces auditable decision records** for ISO 26262, UL 4600, and regulatory compliance  
✅ **Integrates ODD constraints** to respect operational design domain boundaries  
✅ **Monitors safety-critical signals** including collision risk, path safety, and sensor health  
✅ **Provides predictive warnings** to anticipate required degradations  
✅ **Enables graceful degradation** with context-aware fallback strategies  

---

## What This Domain DOES NOT Do

❌ Does NOT steer, brake, accelerate, or actuate vehicles  
❌ Does NOT replace vehicle control software or trajectory planning  
❌ Does NOT override OEM safety systems or redundant ASIL-D monitors  
❌ Does NOT execute trajectories, maneuvers, or motion primitives  
❌ Does NOT perform perception, sensor fusion, or object detection (uses outputs)  
❌ Does NOT handle vehicle communication protocols (CAN, Ethernet, FlexRay)  

**AILEE governs permission, not motion.**

The governor sets an **authorization ceiling** — the maximum autonomy level permitted. The vehicle's control stack remains responsible for:
- Trajectory execution
- Low-level actuator control  
- Emergency braking and collision avoidance
- Redundant safety monitoring (ASIL-D)

---

## Core Components

### Autonomy Levels

The system defines four discrete autonomy authorization levels:

| Level | Name | Description | Typical Use Cases |
|-------|------|-------------|-------------------|
| **0** | `MANUAL_ONLY` | Driver has full control, no AI assistance | System degraded, unsafe conditions, ODD exit |
| **1** | `ASSISTED_ONLY` | Basic driver assistance active | Lane keeping, adaptive cruise, parking assist |
| **2** | `CONSTRAINED_AUTONOMY` | Conditional automation with constraints | Highway pilot, geo-fenced operation, supervised autonomy |
| **3** | `FULL_AUTONOMY_ALLOWED` | Full autonomy permitted (when all safety conditions met) | Urban L4, robotaxi operation, unmanned delivery |

**Key Properties:**
- Levels are **strictly ordered** and represent increasing authority
- Governor can only **constrain** (lower) the autonomy level, never force escalation
- Transitions between levels require explicit safety validation
- Each level has associated confidence and safety thresholds

---

### Safety Monitor Signals

External safety monitors provide independent validation of driving conditions. These inputs come from **ASIL-D certified** or equivalent safety-rated systems, separate from the primary autonomy stack.

**`SafetyMonitorSignals`** includes:

```python
@dataclass(frozen=True)
class SafetyMonitorSignals:
    collision_risk_score: Optional[float]        # 0..1, higher = more risk
    path_safety_score: Optional[float]           # 0..1, higher = safer
    sensor_fusion_health: Optional[float]        # 0..1, overall sensor health
    localization_uncertainty_m: Optional[float]  # Position uncertainty (meters)
    object_detection_health: Optional[float]     # 0..1, detection reliability
    emergency_brake_available: bool              # Redundant e-brake status
    redundant_systems_online: bool               # Backup systems operational
```

**Level-Specific Thresholds:**

- **Level 1+**: Sensor fusion health ≥ 0.80
- **Level 2+**: Collision risk ≤ 0.30, Path safety ≥ 0.70, Localization uncertainty ≤ 2.0m
- **Level 3**: Collision risk ≤ 0.10, Localization uncertainty ≤ 0.5m, Redundant systems required

---

### Operational Design Domain (ODD)

The **ODD** defines where and when autonomy is permitted, based on SAE J3016 and ISO 34503 standards.

**`OperationalDesignDomain`** includes:

```python
@dataclass(frozen=True)
class OperationalDesignDomain:
    # Geographic
    geofence_authorized: bool
    hd_map_available: bool
    distance_to_boundary_m: Optional[float]
    
    # Environmental
    weather_code: str          # clear, light_rain, heavy_rain, snow, fog, ice
    visibility_m: Optional[float]
    road_type: str             # highway, urban, rural, parking, unknown
    road_surface: str          # dry, wet, snow, ice
    construction_zone: bool
    
    # Temporal
    time_of_day: str          # day, dusk, dawn, night
    traffic_density: str      # light, moderate, heavy, congested
```

**ODD Constraints:**
- Ice conditions → `MANUAL_ONLY`
- Heavy rain/snow/fog → Max `ASSISTED_ONLY`
- Low visibility (<30m) → `MANUAL_ONLY`
- Construction zones → Max `ASSISTED_ONLY`
- Outside geofence → `MANUAL_ONLY`

---

### Driver State Monitoring

Comprehensive driver monitoring beyond basic readiness, supporting Euro NCAP and NHTSA requirements.

**`DriverState`** includes:

```python
@dataclass(frozen=True)
class DriverState:
    readiness: float                    # 0..1, overall readiness score
    attention_level: Optional[float]    # 0..1, gaze/pose tracking
    distraction_detected: bool
    drowsiness_detected: bool
    hands_on_wheel: Optional[bool]
    eyes_on_road: Optional[bool]
    response_time_ms: Optional[float]   # Last measured response time
    last_manual_input_ts: Optional[float]
```

**Handoff Readiness Criteria:**
- Readiness ≥ 0.70
- Attention ≥ 0.50
- No distraction/drowsiness detected
- Response time ≤ 2000ms
- Manual input within last 5 minutes
- Hands on wheel (for immediate takeover)

---

### System Health

**`SystemHealth`** monitors computational and sensor infrastructure:

```python
@dataclass(frozen=True)
class SystemHealth:
    latency_ms: Optional[float]
    sensor_faults: int
    compute_load: Optional[float]       # 0..1
    can_bus_errors: int
    network_latency_ms: Optional[float]
    
    # Sensor-specific health scores (0..1)
    camera_health: Optional[float]
    lidar_health: Optional[float]
    radar_health: Optional[float]
    gps_health: Optional[float]
    imu_health: Optional[float]
```

**Health Gating Rules:**
- Any sensor fault → `MANUAL_ONLY`
- CAN bus errors >5 → `MANUAL_ONLY`
- Latency >150ms → Max `CONSTRAINED_AUTONOMY`
- Latency >100ms → Blocks `FULL_AUTONOMY_ALLOWED`

---

## Governance Architecture

### Multi-Layer Decision Gates

The governor applies deterministic gates in strict order:

1. **Deployment Cap** — Policy-level maximum (e.g., "never allow Level 3 in this vehicle")
2. **ODD Constraints** — Geographic, environmental, and temporal boundaries
3. **Safety Monitors** — Independent validation of collision risk, path safety
4. **Scenario Policy** — Context-aware limits (e.g., school zones max Level 1)
5. **System Health** — Latency, faults, sensor integrity
6. **Driver Readiness** — Attention, handoff capability

Each gate can only **lower** the authorization level, never raise it.

---

### Hysteresis and Stability

To prevent mode thrashing (rapid oscillation between levels):

- **Escalation**: Rate-limited (default: min 10 seconds between increases)
- **Downgrade**: Immediate (safety-first, no delay)
- **Grace Period**: Tolerates brief confidence drops without immediate downgrade

This ensures smooth operation while prioritizing safety.

---

### Predictive Warnings

The governor anticipates required downgrades using:

1. **Confidence Trend Analysis** — Multi-timescale tracking (10s, 60s, 5min windows)
2. **ODD Boundary Proximity** — Warns at 300m (urban) to 1000m (highway) before boundary
3. **Weather Deterioration** — Responds to changing conditions
4. **Driver Attention Decline** — Early alerts for distraction/drowsiness

Example warning: `"Predicted: confidence_decline, downgrade to ASSISTED in 5s"`

---

### Scenario-Based Policies

Fine-tuned governance per operational context:

| Scenario | Max Level | Min Confidence | Description |
|----------|-----------|----------------|-------------|
| `highway_cruise` | 3 (Full) | 0.90 | Steady highway, low complexity |
| `highway_merge` | 2 (Constrained) | 0.95 | Elevated risk during merge |
| `urban_intersection` | 2 (Constrained) | 0.92 | High complexity intersection |
| `school_zone` | 1 (Assisted) | 0.95 | Maximum caution required |
| `parking_lot` | 2 (Constrained) | 0.88 | Low speed maneuvering |

Scenarios are detected by the perception/planning stack and passed to the governor.

---

## Integration Pattern

### Quick Start Example

```python
from ailee_automotive_domain import (
    AutonomyGovernor,
    AutonomyGovernancePolicy,
    AutonomySignals,
    AutonomyLevel,
    DriverState,
    SystemHealth,
    OperationalDesignDomain,
    SafetyMonitorSignals,
)

# 1. Setup (once at initialization)
policy = AutonomyGovernancePolicy(
    max_allowed_level=AutonomyLevel.CONSTRAINED_AUTONOMY,
    min_driver_readiness_for_autonomy=0.65,
    max_latency_ms_for_autonomy=150.0,
)
governor = AutonomyGovernor(policy=policy)

# 2. Per-frame evaluation (typically 10-100Hz)
while vehicle_running:
    # Gather inputs from vehicle systems
    signals = AutonomySignals(
        proposed_level=autonomy_stack.get_desired_level(),
        model_confidence=autonomy_stack.get_confidence(),
        peer_recommended_levels=tuple(peer.get_level() for peer in peers),
        driver_state=DriverState(
            readiness=driver_monitor.get_readiness(),
            attention_level=driver_monitor.get_attention(),
        ),
        system_health=SystemHealth(
            latency_ms=system_monitor.get_latency(),
            sensor_faults=system_monitor.get_fault_count(),
        ),
        odd=OperationalDesignDomain(
            weather_code=environment.get_weather(),
            road_type=map_data.get_road_type(),
        ),
        safety_monitors=SafetyMonitorSignals(
            collision_risk_score=safety_stack.get_collision_risk(),
            path_safety_score=safety_stack.get_path_safety(),
        ),
    )
    
    # 3. Get authorized level
    authorized_level, decision = governor.evaluate(signals)
    
    # 4. Enforce as ceiling (NOT a command)
    autonomy_stack.set_authorization_ceiling(authorized_level)
    
    # 5. Alert on degradation
    if decision.used_fallback:
        hmi.show_warning("Autonomy degraded", decision.reasons)
    
    # 6. Log for compliance
    black_box.record_governance_event(governor.get_last_event())
```

---

### Runtime Evaluation Loop

The governor operates in a **continuous evaluation loop**:

```
┌─────────────────────────────────────────────────────────┐
│  Vehicle Systems (Perception, Planning, Control)        │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ Proposed Level + Context
                 ▼
┌─────────────────────────────────────────────────────────┐
│  AILEE Autonomy Governor                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Apply Governance Gates                        │  │
│  │    • Deployment cap                              │  │
│  │    • ODD constraints                             │  │
│  │    • Safety monitors                             │  │
│  │    • System health                               │  │
│  │    • Driver readiness                            │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2. Apply Hysteresis                              │  │
│  │    • Rate limit escalations                      │  │
│  │    • Allow immediate downgrades                  │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 3. AILEE Pipeline Validation                     │  │
│  │    • Confidence scoring                          │  │
│  │    • Peer consensus                              │  │
│  │    • Stability checks                            │  │
│  │    • Grace period evaluation                     │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 4. Predictive Analysis                           │  │
│  │    • Confidence trend monitoring                 │  │
│  │    • ODD boundary warnings                       │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ Authorized Level + Decision Metadata
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Vehicle Control Stack                                   │
│  • Enforces authorization ceiling                        │
│  • Executes degradation if needed                        │
│  • Manages handoff protocols                             │
└─────────────────────────────────────────────────────────┘
```

**Critical Integration Points:**

1. **Input Frequency**: Typically 10-100Hz depending on vehicle dynamics
2. **Authorization Ceiling**: Governor output is a **limit**, not a command
3. **Fallback Handling**: Monitor `decision.used_fallback` for degraded states
4. **Event Export**: Periodically export events for compliance logging

---

### Event Logging and Compliance

Every evaluation produces a **`GovernanceEvent`** for audit trails:

```python
@dataclass(frozen=True)
class GovernanceEvent:
    timestamp: float
    event_type: str              # "level_change" | "gate_applied" | "fallback_used"
    from_level: AutonomyLevel
    to_level: AutonomyLevel
    confidence: float
    reasons: List[str]           # Human-readable decision rationale
    safety_status: Optional[str]
    grace_status: Optional[str]
    consensus_status: Optional[str]
    used_fallback: bool
    metadata: Dict[str, Any]     # Full context snapshot
```

**Export events for post-incident analysis:**

```python
# Get events since last export
events = governor.export_events(since_ts=last_export_timestamp)

# Write to secure storage
for event in events:
    black_box.write(event)
    telemetry.send(event)
```

---

## Safety Features

### Degradation Management

Structured degradation planning with handoff validation:

```python
from ailee_automotive_domain import DegradationStrategy

strategy = DegradationStrategy(
    target_level=AutonomyLevel.ASSISTED_ONLY,
    timeout_seconds=5.0,
    handoff_required=True,
    fallback_trajectory="pull_over",  # or "minimum_risk_maneuver", "slow_in_lane"
    alert_driver=True,
    reason="ODD boundary approaching",
)

# Validate degradation is safe
if governor.initiate_degradation(strategy):
    # Governor validated: driver ready, safe to degrade
    vehicle_controller.execute_degradation(strategy)
else:
    # Blocked: driver not ready or unsafe conditions
    vehicle_controller.execute_minimum_risk_maneuver()
```

**Key Principle:** Governor validates degradation safety but does NOT execute it. Execution remains the responsibility of the vehicle control layer.

---

### Confidence Tracking

Multi-timescale confidence monitoring for early warning:

- **Short-term** (10 seconds): Detects sudden drops
- **Medium-term** (60 seconds): Identifies trends
- **Long-term** (5 minutes): Establishes baseline

```python
# Get confidence trend
trend = governor.get_confidence_trend()  # "improving" | "stable" | "declining"

# Check for active warnings
warning = governor.get_current_warning()
if warning:
    predicted_level, predicted_ts, reason = warning
    hmi.show_predictive_warning(f"Downgrade anticipated: {reason}")
```

---

### Black Box Logging

Full audit trail for compliance and post-incident analysis:

```python
# Export all events
all_events = governor.export_events()

# Export events from specific timeframe
incident_events = governor.export_events(since_ts=incident_start_time)

# Get last event immediately
last_event = governor.get_last_event()
if last_event.event_type == "level_change":
    print(f"Autonomy changed: {last_event.from_level} → {last_event.to_level}")
    print(f"Reasons: {last_event.reasons}")
```

Events include:
- Full decision rationale
- All input signals (ODD, safety monitors, driver state)
- Confidence scores and peer recommendations
- Applied gates and policies
- Metadata snapshots

---

## Design Principles

> **Autonomy must be earned continuously, not assumed.**

This domain is designed around four core principles:

1. **Safety-First Architecture**
   - Immediate downgrades allowed, escalations rate-limited
   - Multiple independent validation layers
   - Explicit fallback behavior with last-known-good states

2. **Deterministic Decisions**
   - No randomness in governance logic
   - Reproducible decisions for identical inputs
   - Clear precedence rules for conflicting signals

3. **Certification-Friendly**
   - Complete audit trails for every decision
   - ISO 26262 and UL 4600 alignment
   - Black box logging for post-incident analysis

4. **Stability and Predictability**
   - Hysteresis prevents mode thrashing
   - Predictive warnings anticipate degradation needs
   - Grace periods tolerate brief confidence drops

**Key Philosophy:** The governor's role is to **prevent harm**, not optimize performance. When in doubt, it constrains authority.

---

## Compliance and Certification

### Regulatory Alignment

This domain is designed to support:

- **ISO 26262** (Road Vehicles — Functional Safety)
  - ASIL-D compatible inputs
  - Deterministic decision logic
  - Complete traceability

- **UL 4600** (Safety for Autonomous Products)
  - Hazard analysis integration
  - Runtime monitoring requirements
  - Validation and verification support

- **SAE J3016** (Levels of Driving Automation)
  - Clear level definitions
  - Handoff protocol support
  - ODD enforcement

- **Euro NCAP / NHTSA** (Driver Monitoring)
  - Attention tracking
  - Handoff readiness validation
  - Distraction/drowsiness detection

### Audit Trail Requirements

Every decision includes:
- Timestamp with microsecond precision
- Full input snapshot (all signals, context)
- Applied governance rules and thresholds
- Decision rationale (human-readable)
- Fallback/degradation status

### Certification Testing

The domain supports:
- Unit testing with deterministic scenarios
- Integration testing with hardware-in-the-loop
- Fault injection for safety validation
- Compliance report generation

---

## API Reference

### Core Classes

**`AutonomyGovernor`** — Main governance controller
- `evaluate(signals: AutonomySignals) → (AutonomyLevel, DecisionResult)`
- `initiate_degradation(strategy: DegradationStrategy) → bool`
- `export_events(since_ts: Optional[float]) → List[GovernanceEvent]`
- `get_confidence_trend() → str`
- `get_current_warning() → Optional[Tuple[AutonomyLevel, float, str]]`

**`AutonomySignals`** — Input signals for evaluation
- Required: `proposed_level`, `model_confidence`
- Optional: `peer_recommended_levels`, `safety_monitors`, `odd`, `driver_state`, `system_health`, `current_scenario`

**`AutonomyGovernancePolicy`** — Deployment-level configuration
- Defines authorization caps, thresholds, and hysteresis parameters
- Immutable after initialization

### Configuration Factory

**`default_autonomy_config()`** — Safe defaults for AILEE pipeline
- Returns pre-tuned `AileeConfig` for automotive domain
- Optimized for stability and consensus

### Utility Functions

**`create_test_signals()`** — Factory for unit testing
- Generates valid `AutonomySignals` with sensible defaults
- Supports keyword arguments for customization

---

## Testing and Validation

### Unit Testing

```python
from ailee_automotive_domain import create_test_signals, AutonomyGovernor

# Test scenario: sensor fault should force manual
signals = create_test_signals(
    level=AutonomyLevel.CONSTRAINED_AUTONOMY,
    confidence=0.90,
    system_health=SystemHealth(sensor_faults=2),
)

governor = AutonomyGovernor()
authorized_level, decision = governor.evaluate(signals)

assert authorized_level == AutonomyLevel.MANUAL_ONLY
assert "sensor_faults" in str(decision.reasons)
```

### Integration Testing

The demo at the end of the module (`if __name__ == "__main__":`) demonstrates:
- Good conditions escalation
- Driver distraction handling
- High latency blocking
- Sensor fault forcing manual mode

Run with: `python ailee_automotive_domain.py`

### Test Coverage Requirements

For certification, ensure coverage of:
- All ODD boundary conditions
- All safety monitor thresholds
- Driver state edge cases (distraction, drowsiness, handoff failure)
- System health degradation paths
- Hysteresis timing edge cases
- Predictive warning triggers

---

## Summary

The **AILEE Automotive Governance Domain** provides production-grade authorization control for autonomous vehicle systems. It:

- ✅ Enforces multi-factor validation with deterministic logic
- ✅ Respects ODD constraints and safety monitor inputs
- ✅ Monitors driver readiness and system health
- ✅ Prevents unsafe escalation while enabling graceful degradation
- ✅ Produces complete audit trails for compliance
- ✅ Anticipates degradation needs with predictive warnings

**Remember:** AILEE governs **permission**, not motion. The governor sets an authorization ceiling; the vehicle control stack remains responsible for execution.

For questions or contributions, refer to the main AILEE Trust Layer documentation.

---

*Document Version: 2.0.0*  
*Last Updated: December 2025*  
*Compatibility: AILEE Trust Pipeline v1.x*
