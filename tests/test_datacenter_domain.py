#!/usr/bin/env python
"""
Datacenter Domain Test Suite
=============================

Tests for the AILEE Data Center domain governance interface:
1. All public types can be imported
2. DatacenterGovernor can be instantiated with factory functions
3. DatacenterSignals and validate_datacenter_signals work correctly
4. Governor.evaluate() returns a valid DatacenterDecision for each control domain
5. Trust level determination is consistent with pipeline output
6. Rate limiting works for workload migrations
7. Event and decision history recording
8. Policy enforcement (strict vs. permissive)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ailee.domains.datacenter import (
    DatacenterGovernor,
    DatacenterPolicy,
    DatacenterSignals,
    DatacenterDecision,
    DatacenterEvent,
    DatacenterTrustLevel,
    FacilityHealthStatus,
    ControlDomain,
    ControlAction,
    SensorReading,
    TelemetryProcessor,
    COOLING_CONTROL,
    POWER_CAPPING,
    WORKLOAD_PLACEMENT,
    CoolingController,
    PowerCapController,
    WorkloadPlacementGovernor,
    DataCenterMonitor,
    create_datacenter_governor,
    create_default_governor,
    create_strict_governor,
    create_permissive_governor,
    validate_datacenter_signals,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_cooling_signals(confidence: float = 0.88, n_sensors: int = 4) -> DatacenterSignals:
    readings = [
        SensorReading(22.3 + i * 0.05, time.time(), f"rack_{i:02d}_inlet")
        for i in range(n_sensors)
    ]
    return DatacenterSignals(
        control_domain=ControlDomain.COOLING,
        proposed_action=ControlAction.SETPOINT_CHANGE,
        ai_value=22.0,
        ai_confidence=confidence,
        sensor_readings=readings,
        zone_id="zone_a",
    )


def make_power_signals(confidence: float = 0.92) -> DatacenterSignals:
    return DatacenterSignals(
        control_domain=ControlDomain.POWER,
        proposed_action=ControlAction.POWER_CAP,
        ai_value=4500.0,
        ai_confidence=confidence,
        sensor_readings=[
            SensorReading(4200, time.time(), "meter_1"),
            SensorReading(4250, time.time(), "meter_2"),
            SensorReading(4180, time.time(), "meter_3"),
        ],
    )


def make_workload_signals(confidence: float = 0.85) -> DatacenterSignals:
    return DatacenterSignals(
        control_domain=ControlDomain.WORKLOAD,
        proposed_action=ControlAction.WORKLOAD_MIGRATE,
        ai_value=7.5,
        ai_confidence=confidence,
        sensor_readings=[SensorReading(7.2, time.time(), "model_1")],
        workload_id="vm-123",
        source_node="node-a",
        target_node="node-b",
    )


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

def test_imports():
    """All public types are importable."""
    assert DatacenterGovernor is not None
    assert DatacenterPolicy is not None
    assert DatacenterSignals is not None
    assert DatacenterDecision is not None
    assert DatacenterEvent is not None
    assert DatacenterTrustLevel is not None
    assert FacilityHealthStatus is not None
    assert ControlDomain is not None
    assert ControlAction is not None
    assert SensorReading is not None
    assert create_datacenter_governor is not None
    assert validate_datacenter_signals is not None
    print("✓ All imports successful")
    return True


def test_top_level_imports():
    """Key types are importable directly from the top-level ailee package."""
    import ailee
    assert hasattr(ailee, "DatacenterGovernor")
    assert hasattr(ailee, "DatacenterPolicy")
    assert hasattr(ailee, "DatacenterSignals")
    assert hasattr(ailee, "DatacenterTrustLevel")
    assert hasattr(ailee, "ControlDomain")
    assert hasattr(ailee, "ControlAction")
    assert hasattr(ailee, "create_datacenter_governor")
    assert hasattr(ailee, "validate_datacenter_signals")
    print("✓ Top-level ailee imports OK")
    return True


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

def test_enums():
    """Enum values are correct and ordered as expected."""
    assert DatacenterTrustLevel.NO_ACTION < DatacenterTrustLevel.ADVISORY
    assert DatacenterTrustLevel.ADVISORY < DatacenterTrustLevel.SUPERVISED
    assert DatacenterTrustLevel.SUPERVISED < DatacenterTrustLevel.AUTONOMOUS

    assert FacilityHealthStatus.HEALTHY == "HEALTHY"
    assert FacilityHealthStatus.CRITICAL == "CRITICAL"

    assert ControlDomain.COOLING == "COOLING"
    assert ControlDomain.POWER == "POWER"
    assert ControlDomain.WORKLOAD == "WORKLOAD"

    assert ControlAction.SETPOINT_CHANGE == "SETPOINT_CHANGE"
    assert ControlAction.POWER_CAP == "POWER_CAP"
    assert ControlAction.WORKLOAD_MIGRATE == "WORKLOAD_MIGRATE"

    print("✓ Enum values correct")
    return True


# ---------------------------------------------------------------------------
# Factory / policy tests
# ---------------------------------------------------------------------------

def test_factory_functions():
    """All factory functions create valid DatacenterGovernor instances."""
    g1 = create_datacenter_governor()
    assert isinstance(g1, DatacenterGovernor)

    g2 = create_default_governor()
    assert isinstance(g2, DatacenterGovernor)
    assert g2.policy.min_trust_for_action == DatacenterTrustLevel.SUPERVISED

    g3 = create_strict_governor()
    assert isinstance(g3, DatacenterGovernor)
    assert g3.policy.min_trust_for_action == DatacenterTrustLevel.AUTONOMOUS
    assert g3.policy.require_consensus is True
    assert g3.policy.max_setpoint_change_per_cycle == 1.0
    assert g3.policy.max_migrations_per_hour == 50

    g4 = create_permissive_governor()
    assert isinstance(g4, DatacenterGovernor)
    assert g4.policy.min_trust_for_action == DatacenterTrustLevel.ADVISORY
    assert g4.policy.require_consensus is False

    print("✓ Factory functions create correct governors")
    return True


def test_policy_overrides():
    """Factory functions respect keyword overrides."""
    g = create_strict_governor(max_migrations_per_hour=10)
    assert g.policy.max_migrations_per_hour == 10

    g2 = create_permissive_governor(enable_audit_events=True)
    assert g2.policy.enable_audit_events is True

    print("✓ Policy overrides applied correctly")
    return True


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validate_signals_valid():
    """validate_datacenter_signals returns empty list for valid signals."""
    signals = make_cooling_signals()
    issues = validate_datacenter_signals(signals)
    assert issues == [], f"Unexpected issues: {issues}"
    print("✓ Valid signals pass validation")
    return True


def test_validate_signals_bad_confidence():
    """validate_datacenter_signals catches out-of-range confidence."""
    signals = DatacenterSignals(
        control_domain=ControlDomain.COOLING,
        proposed_action=ControlAction.SETPOINT_CHANGE,
        ai_value=22.0,
        ai_confidence=1.5,  # invalid
        sensor_readings=[SensorReading(22.0, time.time(), "s1")],
    )
    issues = validate_datacenter_signals(signals)
    assert any("ai_confidence" in i for i in issues)
    print("✓ Bad confidence flagged by validation")
    return True


def test_validate_signals_no_sensors():
    """validate_datacenter_signals warns when no sensor readings are provided."""
    signals = DatacenterSignals(
        control_domain=ControlDomain.COOLING,
        proposed_action=ControlAction.SETPOINT_CHANGE,
        ai_value=22.0,
        ai_confidence=0.85,
        sensor_readings=[],
    )
    issues = validate_datacenter_signals(signals)
    assert any("sensor" in i.lower() for i in issues)
    print("✓ Missing sensors flagged by validation")
    return True


def test_validate_signals_cooling_out_of_range():
    """validate_datacenter_signals warns about extreme cooling setpoints."""
    signals = DatacenterSignals(
        control_domain=ControlDomain.COOLING,
        proposed_action=ControlAction.SETPOINT_CHANGE,
        ai_value=100.0,  # way too hot
        ai_confidence=0.85,
        sensor_readings=[SensorReading(22.0, time.time(), "s1")],
    )
    issues = validate_datacenter_signals(signals)
    assert any("out of range" in i.lower() or "setpoint" in i.lower() for i in issues)
    print("✓ Out-of-range cooling setpoint flagged")
    return True


def test_validate_workload_score_out_of_range():
    """validate_datacenter_signals warns about invalid workload placement scores."""
    signals = DatacenterSignals(
        control_domain=ControlDomain.WORKLOAD,
        proposed_action=ControlAction.WORKLOAD_MIGRATE,
        ai_value=15.0,  # should be [0-10]
        ai_confidence=0.85,
        sensor_readings=[SensorReading(7.0, time.time(), "m1")],
    )
    issues = validate_datacenter_signals(signals)
    assert any("score" in i.lower() or "placement" in i.lower() for i in issues)
    print("✓ Out-of-range workload score flagged")
    return True


# ---------------------------------------------------------------------------
# Governor.evaluate() tests
# ---------------------------------------------------------------------------

def test_evaluate_cooling():
    """Governor returns a DatacenterDecision for cooling signals."""
    governor = create_datacenter_governor()
    signals = make_cooling_signals()
    decision = governor.evaluate(signals)

    assert isinstance(decision, DatacenterDecision)
    assert decision.control_domain == ControlDomain.COOLING
    assert decision.proposed_action == ControlAction.SETPOINT_CHANGE
    assert isinstance(decision.authorized_level, DatacenterTrustLevel)
    assert isinstance(decision.actionable, bool)
    assert isinstance(decision.trusted_value, float)
    assert decision.decision_id is not None
    assert decision.timestamp > 0
    print(f"✓ Cooling evaluation: trust={decision.authorized_level.name}, actionable={decision.actionable}")
    return True


def test_evaluate_power():
    """Governor returns a DatacenterDecision for power signals."""
    governor = create_datacenter_governor()
    signals = make_power_signals()
    decision = governor.evaluate(signals)

    assert isinstance(decision, DatacenterDecision)
    assert decision.control_domain == ControlDomain.POWER
    assert decision.proposed_action == ControlAction.POWER_CAP
    assert isinstance(decision.trusted_value, float)
    print(f"✓ Power evaluation: trust={decision.authorized_level.name}, value={decision.trusted_value:.0f} kW")
    return True


def test_evaluate_workload():
    """Governor returns a DatacenterDecision for workload signals."""
    governor = create_datacenter_governor()
    signals = make_workload_signals()
    decision = governor.evaluate(signals)

    assert isinstance(decision, DatacenterDecision)
    assert decision.control_domain == ControlDomain.WORKLOAD
    print(f"✓ Workload evaluation: trust={decision.authorized_level.name}, actionable={decision.actionable}")
    return True


def test_evaluate_all_domains():
    """Governor can evaluate all ControlDomain values without error."""
    governor = create_datacenter_governor()
    for domain in ControlDomain:
        signals = DatacenterSignals(
            control_domain=domain,
            proposed_action=ControlAction.NO_ACTION,
            ai_value=5.0,
            ai_confidence=0.80,
            sensor_readings=[SensorReading(5.0, time.time(), "s1")],
        )
        decision = governor.evaluate(signals)
        assert isinstance(decision, DatacenterDecision)
    print("✓ All ControlDomain values evaluated without error")
    return True


def test_decision_fields():
    """DatacenterDecision has all required fields set correctly."""
    governor = create_datacenter_governor()
    signals = make_cooling_signals()
    decision = governor.evaluate(signals)

    assert hasattr(decision, "authorized_level")
    assert hasattr(decision, "actionable")
    assert hasattr(decision, "trusted_value")
    assert hasattr(decision, "control_domain")
    assert hasattr(decision, "proposed_action")
    assert hasattr(decision, "health_status")
    assert hasattr(decision, "safety_flags")
    assert hasattr(decision, "used_fallback")
    assert hasattr(decision, "timestamp")
    assert hasattr(decision, "decision_id")
    assert hasattr(decision, "reasons")
    assert hasattr(decision, "metadata")
    assert isinstance(decision.health_status, FacilityHealthStatus)
    print("✓ DatacenterDecision has all required fields")
    return True


# ---------------------------------------------------------------------------
# Rate limiting test
# ---------------------------------------------------------------------------

def test_workload_rate_limit():
    """Governor enforces migration rate limit per policy."""
    # Use max_migrations_per_hour=0 so rate limiting triggers immediately
    policy = DatacenterPolicy(max_migrations_per_hour=0)
    governor = DatacenterGovernor(policy=policy)

    decisions = []
    for i in range(3):
        signals = DatacenterSignals(
            control_domain=ControlDomain.WORKLOAD,
            proposed_action=ControlAction.WORKLOAD_MIGRATE,
            ai_value=8.0,
            ai_confidence=0.95,
            sensor_readings=[SensorReading(8.0, time.time(), "m1")],
            workload_id=f"vm-{i}",
        )
        decisions.append(governor.evaluate(signals))

    # With max=0, all workload decisions should be rate-limited immediately
    rate_limited = [d for d in decisions if "Rate limit" in (d.fallback_reason or "")]
    assert len(rate_limited) == 3, (
        f"Expected 3 rate-limited decisions, got {len(rate_limited)}"
    )
    # Rate-limited decisions must not be actionable
    assert all(not d.actionable for d in rate_limited)
    # Rate-limited decisions have NO_ACTION trust level
    assert all(d.authorized_level == DatacenterTrustLevel.NO_ACTION for d in rate_limited)
    print(f"✓ Rate limiting works: {len(rate_limited)} decisions rate-limited out of 3")
    return True


# ---------------------------------------------------------------------------
# Events and history tests
# ---------------------------------------------------------------------------

def test_events_recorded():
    """Governor records events for each evaluate() call when policy enables it."""
    governor = create_datacenter_governor()  # default: enable_audit_events=True
    initial_count = len(governor.get_events())

    governor.evaluate(make_cooling_signals())
    governor.evaluate(make_power_signals())

    events = governor.get_events()
    assert len(events) == initial_count + 2
    for event in events:
        assert isinstance(event, DatacenterEvent)
        assert event.timestamp > 0
        assert isinstance(event.control_domain, ControlDomain)
    print(f"✓ {len(events)} events recorded correctly")
    return True


def test_decision_history():
    """Governor records decision history when policy enables it."""
    governor = create_datacenter_governor()
    governor.evaluate(make_cooling_signals())
    governor.evaluate(make_power_signals())

    history = governor.get_decision_history()
    assert len(history) == 2
    assert all(isinstance(d, DatacenterDecision) for d in history)
    print("✓ Decision history recorded correctly")
    return True


def test_no_events_when_disabled():
    """Governor does not record events when policy disables it."""
    governor = create_permissive_governor()  # enable_audit_events=False
    governor.evaluate(make_cooling_signals())
    governor.evaluate(make_power_signals())

    assert len(governor.get_events()) == 0
    assert len(governor.get_decision_history()) == 0
    print("✓ Events/history correctly suppressed when disabled")
    return True


# ---------------------------------------------------------------------------
# Health and metrics tests
# ---------------------------------------------------------------------------

def test_get_health():
    """Governor.get_health() returns a FacilityHealthStatus."""
    governor = create_datacenter_governor()
    health = governor.get_health()
    assert isinstance(health, FacilityHealthStatus)
    print(f"✓ get_health() returned: {health.value}")
    return True


def test_get_subsystem_health():
    """Governor.get_subsystem_health() returns a dict keyed by domain name."""
    governor = create_datacenter_governor()
    subsystem = governor.get_subsystem_health()
    assert isinstance(subsystem, dict)
    for domain in ControlDomain:
        assert domain.value in subsystem
        assert isinstance(subsystem[domain.value], FacilityHealthStatus)
    print("✓ get_subsystem_health() returned correct structure")
    return True


def test_get_metrics():
    """Governor.get_metrics() returns expected keys."""
    governor = create_datacenter_governor()
    governor.evaluate(make_cooling_signals())
    metrics = governor.get_metrics()

    assert "fallback_rate" in metrics
    assert "avg_confidence" in metrics
    assert "total_decisions" in metrics
    assert "migrations_this_hour" in metrics
    assert "overall_health" in metrics
    assert metrics["total_decisions"] == 1
    print(f"✓ get_metrics() returned: {metrics}")
    return True


# ---------------------------------------------------------------------------
# Lower-level API backward-compatibility tests
# ---------------------------------------------------------------------------

def test_cooling_controller_still_works():
    """CoolingController (lower-level API) still functions correctly."""
    controller = CoolingController()
    sensors = [
        SensorReading(22.5, time.time(), f"rack_{i:02d}") for i in range(4)
    ]
    result = controller.propose_setpoint(
        ai_setpoint=22.0,
        ai_confidence=0.85,
        sensor_readings=sensors,
        zone_id="zone_a",
    )
    assert result is not None
    assert hasattr(result, "safety_status")
    print("✓ CoolingController backward-compatible")
    return True


def test_datacenter_monitor_still_works():
    """DataCenterMonitor still provides unified metrics."""
    monitor = DataCenterMonitor()
    metrics = monitor.get_unified_metrics()
    assert "cooling" in metrics
    assert "power" in metrics
    assert "workload" in metrics
    health = monitor.check_health()
    assert "cooling" in health
    print("✓ DataCenterMonitor backward-compatible")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_imports,
        test_top_level_imports,
        test_enums,
        test_factory_functions,
        test_policy_overrides,
        test_validate_signals_valid,
        test_validate_signals_bad_confidence,
        test_validate_signals_no_sensors,
        test_validate_signals_cooling_out_of_range,
        test_validate_workload_score_out_of_range,
        test_evaluate_cooling,
        test_evaluate_power,
        test_evaluate_workload,
        test_evaluate_all_domains,
        test_decision_fields,
        test_workload_rate_limit,
        test_events_recorded,
        test_decision_history,
        test_no_events_when_disabled,
        test_get_health,
        test_get_subsystem_health,
        test_get_metrics,
        test_cooling_controller_still_works,
        test_datacenter_monitor_still_works,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"✗ {test.__name__}: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
