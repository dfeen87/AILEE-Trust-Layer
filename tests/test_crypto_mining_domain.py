#!/usr/bin/env python
"""
Crypto Mining Domain Test Suite
=================================

Tests for the AILEE Crypto Mining domain governance interface:
1. All public types can be imported
2. MiningGovernor can be instantiated with factory functions
3. MiningSignals and validate_mining_signals work correctly
4. Governor.evaluate() returns a valid MiningDecision for each mining domain
5. Trust level determination is consistent with pipeline output
6. Rate limiting works for pool switches
7. Thermal override fires when observed temperature breaches threshold
8. Event and decision history recording
9. Policy enforcement (strict vs. permissive)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ailee.domains.crypto_mining import (
    MiningGovernor,
    MiningPolicy,
    MiningSignals,
    MiningDecision,
    MiningEvent,
    CryptoMiningTrustLevel,
    MiningOperationStatus,
    MiningDomain,
    MiningAction,
    HardwareReading,
    HASH_RATE_OPTIMIZATION,
    THERMAL_PROTECTION,
    POOL_SWITCHING,
    POWER_MANAGEMENT,
    create_mining_governor,
    create_default_governor,
    create_strict_governor,
    create_permissive_governor,
    validate_mining_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hash_rate_signals(confidence: float = 0.88, n_rigs: int = 3) -> MiningSignals:
    readings = [
        HardwareReading(93.0 + i * 0.5, time.time(), f"rig_{i:02d}")
        for i in range(n_rigs)
    ]
    return MiningSignals(
        mining_domain=MiningDomain.HASH_RATE,
        proposed_action=MiningAction.TUNE_HASH_RATE,
        ai_value=95.0,
        ai_confidence=confidence,
        hardware_readings=readings,
        rig_id="rig_00",
    )


def make_thermal_signals(confidence: float = 0.92, temp: float = 72.0) -> MiningSignals:
    return MiningSignals(
        mining_domain=MiningDomain.THERMAL,
        proposed_action=MiningAction.THERMAL_THROTTLE,
        ai_value=temp,
        ai_confidence=confidence,
        hardware_readings=[
            HardwareReading(temp - 1.0, time.time(), "gpu_00"),
            HardwareReading(temp,       time.time(), "gpu_01"),
            HardwareReading(temp + 0.5, time.time(), "gpu_02"),
            HardwareReading(temp - 0.5, time.time(), "gpu_03"),
        ],
        rig_id="rig_00",
    )


def make_pool_signals(confidence: float = 0.85) -> MiningSignals:
    return MiningSignals(
        mining_domain=MiningDomain.POOL_MANAGEMENT,
        proposed_action=MiningAction.SWITCH_POOL,
        ai_value=0.82,
        ai_confidence=confidence,
        hardware_readings=[
            HardwareReading(0.80, time.time(), "latency_probe_1"),
            HardwareReading(0.83, time.time(), "latency_probe_2"),
            HardwareReading(0.81, time.time(), "latency_probe_3"),
        ],
        pool_url="stratum+tcp://pool.example.com:3333",
        current_pool_url="stratum+tcp://old.pool.example.com:3333",
    )


def make_power_signals(confidence: float = 0.90) -> MiningSignals:
    return MiningSignals(
        mining_domain=MiningDomain.POWER,
        proposed_action=MiningAction.ADJUST_POWER_LIMIT,
        ai_value=280.0,
        ai_confidence=confidence,
        hardware_readings=[
            HardwareReading(275.0, time.time(), "psu_0"),
            HardwareReading(278.0, time.time(), "psu_1"),
            HardwareReading(282.0, time.time(), "psu_2"),
        ],
        rig_id="rig_00",
    )


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

def test_imports():
    """All public types are importable."""
    assert MiningGovernor is not None
    assert MiningPolicy is not None
    assert MiningSignals is not None
    assert MiningDecision is not None
    assert MiningEvent is not None
    assert CryptoMiningTrustLevel is not None
    assert MiningOperationStatus is not None
    assert MiningDomain is not None
    assert MiningAction is not None
    assert HardwareReading is not None
    assert create_mining_governor is not None
    assert validate_mining_signals is not None
    print("✓ All imports successful")
    return True


def test_top_level_imports():
    """Key types are importable directly from the top-level ailee package."""
    import ailee
    assert hasattr(ailee, "MiningGovernor")
    assert hasattr(ailee, "MiningPolicy")
    assert hasattr(ailee, "MiningSignals")
    assert hasattr(ailee, "CryptoMiningTrustLevel")
    assert hasattr(ailee, "MiningDomain")
    assert hasattr(ailee, "MiningAction")
    assert hasattr(ailee, "create_mining_governor")
    assert hasattr(ailee, "validate_mining_signals")
    print("✓ Top-level ailee imports OK")
    return True


def test_domain_registry():
    """crypto_mining appears in get_available_domains()."""
    import ailee
    domains = ailee.get_available_domains()
    assert "crypto_mining" in domains
    assert domains["crypto_mining"] is True
    print("✓ crypto_mining registered in get_available_domains()")
    return True


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

def test_enums():
    """Enum values are correct and ordered as expected."""
    assert CryptoMiningTrustLevel.NO_ACTION < CryptoMiningTrustLevel.ADVISORY
    assert CryptoMiningTrustLevel.ADVISORY < CryptoMiningTrustLevel.SUPERVISED
    assert CryptoMiningTrustLevel.SUPERVISED < CryptoMiningTrustLevel.AUTONOMOUS

    assert MiningOperationStatus.HEALTHY == "HEALTHY"
    assert MiningOperationStatus.CRITICAL == "CRITICAL"

    assert MiningDomain.HASH_RATE == "HASH_RATE"
    assert MiningDomain.THERMAL == "THERMAL"
    assert MiningDomain.POWER == "POWER"
    assert MiningDomain.POOL_MANAGEMENT == "POOL_MANAGEMENT"
    assert MiningDomain.HARDWARE == "HARDWARE"

    assert MiningAction.TUNE_HASH_RATE == "TUNE_HASH_RATE"
    assert MiningAction.ADJUST_POWER_LIMIT == "ADJUST_POWER_LIMIT"
    assert MiningAction.SWITCH_POOL == "SWITCH_POOL"
    assert MiningAction.THERMAL_THROTTLE == "THERMAL_THROTTLE"
    assert MiningAction.NO_ACTION == "NO_ACTION"

    print("✓ Enum values correct")
    return True


# ---------------------------------------------------------------------------
# Factory / policy tests
# ---------------------------------------------------------------------------

def test_factory_functions():
    """All factory functions create valid MiningGovernor instances."""
    g1 = create_mining_governor()
    assert isinstance(g1, MiningGovernor)

    g2 = create_default_governor()
    assert isinstance(g2, MiningGovernor)
    assert g2.policy.min_trust_for_action == CryptoMiningTrustLevel.SUPERVISED

    g3 = create_strict_governor()
    assert isinstance(g3, MiningGovernor)
    assert g3.policy.min_trust_for_action == CryptoMiningTrustLevel.AUTONOMOUS
    assert g3.policy.require_consensus is True
    assert g3.policy.max_hash_rate_change_pct == 5.0
    assert g3.policy.max_pool_switches_per_hour == 2
    assert g3.policy.thermal_throttle_temp_c == 80.0

    g4 = create_permissive_governor()
    assert isinstance(g4, MiningGovernor)
    assert g4.policy.min_trust_for_action == CryptoMiningTrustLevel.ADVISORY
    assert g4.policy.require_consensus is False

    print("✓ Factory functions create correct governors")
    return True


def test_policy_overrides():
    """Factory functions respect keyword overrides."""
    g = create_strict_governor(max_pool_switches_per_hour=10)
    assert g.policy.max_pool_switches_per_hour == 10

    g2 = create_permissive_governor(enable_audit_events=True)
    assert g2.policy.enable_audit_events is True

    print("✓ Policy overrides applied correctly")
    return True


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validate_signals_valid():
    """validate_mining_signals returns empty list for valid signals."""
    signals = make_hash_rate_signals()
    issues = validate_mining_signals(signals)
    assert issues == [], f"Unexpected issues: {issues}"
    print("✓ Valid signals pass validation")
    return True


def test_validate_signals_bad_confidence():
    """validate_mining_signals catches out-of-range confidence."""
    signals = MiningSignals(
        mining_domain=MiningDomain.HASH_RATE,
        proposed_action=MiningAction.TUNE_HASH_RATE,
        ai_value=95.0,
        ai_confidence=1.5,  # invalid
        hardware_readings=[HardwareReading(93.0, time.time(), "rig_01")],
    )
    issues = validate_mining_signals(signals)
    assert any("ai_confidence" in i for i in issues)
    print("✓ Bad confidence flagged by validation")
    return True


def test_validate_signals_no_hardware():
    """validate_mining_signals warns when no hardware readings are provided."""
    signals = MiningSignals(
        mining_domain=MiningDomain.HASH_RATE,
        proposed_action=MiningAction.TUNE_HASH_RATE,
        ai_value=95.0,
        ai_confidence=0.85,
        hardware_readings=[],
    )
    issues = validate_mining_signals(signals)
    assert any("hardware" in i.lower() for i in issues)
    print("✓ Missing hardware readings flagged by validation")
    return True


def test_validate_signals_thermal_out_of_range():
    """validate_mining_signals warns about extreme thermal values."""
    signals = MiningSignals(
        mining_domain=MiningDomain.THERMAL,
        proposed_action=MiningAction.THERMAL_THROTTLE,
        ai_value=200.0,  # way too hot
        ai_confidence=0.85,
        hardware_readings=[HardwareReading(75.0, time.time(), "gpu_00")],
    )
    issues = validate_mining_signals(signals)
    assert any("thermal" in i.lower() or "out of range" in i.lower() for i in issues)
    print("✓ Out-of-range thermal value flagged")
    return True


def test_validate_signals_pool_score_out_of_range():
    """validate_mining_signals warns about invalid pool scores."""
    signals = MiningSignals(
        mining_domain=MiningDomain.POOL_MANAGEMENT,
        proposed_action=MiningAction.SWITCH_POOL,
        ai_value=2.5,  # should be [0, 1]
        ai_confidence=0.85,
        hardware_readings=[HardwareReading(0.80, time.time(), "probe_1")],
    )
    issues = validate_mining_signals(signals)
    assert any("pool" in i.lower() or "score" in i.lower() for i in issues)
    print("✓ Out-of-range pool score flagged")
    return True


def test_validate_signals_negative_hash_rate():
    """validate_mining_signals warns about negative hash rate."""
    signals = MiningSignals(
        mining_domain=MiningDomain.HASH_RATE,
        proposed_action=MiningAction.TUNE_HASH_RATE,
        ai_value=-10.0,
        ai_confidence=0.85,
        hardware_readings=[HardwareReading(90.0, time.time(), "rig_01")],
    )
    issues = validate_mining_signals(signals)
    assert any("hash rate" in i.lower() or "non-negative" in i.lower() for i in issues)
    print("✓ Negative hash rate flagged")
    return True


# ---------------------------------------------------------------------------
# Governor.evaluate() tests
# ---------------------------------------------------------------------------

def test_evaluate_hash_rate():
    """Governor returns a MiningDecision for hash rate signals."""
    governor = create_mining_governor()
    signals = make_hash_rate_signals()
    decision = governor.evaluate(signals)

    assert isinstance(decision, MiningDecision)
    assert decision.mining_domain == MiningDomain.HASH_RATE
    assert decision.proposed_action == MiningAction.TUNE_HASH_RATE
    assert isinstance(decision.authorized_level, CryptoMiningTrustLevel)
    assert isinstance(decision.actionable, bool)
    assert isinstance(decision.trusted_value, float)
    assert decision.decision_id is not None
    assert decision.timestamp > 0
    print(f"✓ Hash rate evaluation: trust={decision.authorized_level.name}, actionable={decision.actionable}")
    return True


def test_evaluate_thermal():
    """Governor returns a MiningDecision for thermal signals (below throttle threshold)."""
    governor = create_mining_governor()
    signals = make_thermal_signals(temp=72.0)
    decision = governor.evaluate(signals)

    assert isinstance(decision, MiningDecision)
    assert decision.mining_domain == MiningDomain.THERMAL
    assert isinstance(decision.trusted_value, float)
    print(f"✓ Thermal evaluation: trust={decision.authorized_level.name}, actionable={decision.actionable}")
    return True


def test_evaluate_pool():
    """Governor returns a MiningDecision for pool management signals."""
    governor = create_mining_governor()
    signals = make_pool_signals()
    decision = governor.evaluate(signals)

    assert isinstance(decision, MiningDecision)
    assert decision.mining_domain == MiningDomain.POOL_MANAGEMENT
    print(f"✓ Pool evaluation: trust={decision.authorized_level.name}, actionable={decision.actionable}")
    return True


def test_evaluate_power():
    """Governor returns a MiningDecision for power management signals."""
    governor = create_mining_governor()
    signals = make_power_signals()
    decision = governor.evaluate(signals)

    assert isinstance(decision, MiningDecision)
    assert decision.mining_domain == MiningDomain.POWER
    assert decision.proposed_action == MiningAction.ADJUST_POWER_LIMIT
    assert isinstance(decision.trusted_value, float)
    print(f"✓ Power evaluation: trust={decision.authorized_level.name}, value={decision.trusted_value:.1f} W")
    return True


def test_evaluate_all_domains():
    """Governor can evaluate all MiningDomain values without error."""
    governor = create_mining_governor()
    for domain in MiningDomain:
        signals = MiningSignals(
            mining_domain=domain,
            proposed_action=MiningAction.NO_ACTION,
            ai_value=0.5,
            ai_confidence=0.80,
            hardware_readings=[HardwareReading(0.5, time.time(), "sensor_1")],
        )
        decision = governor.evaluate(signals)
        assert isinstance(decision, MiningDecision)
    print("✓ All MiningDomain values evaluated without error")
    return True


def test_decision_fields():
    """MiningDecision has all required fields set correctly."""
    governor = create_mining_governor()
    signals = make_hash_rate_signals()
    decision = governor.evaluate(signals)

    assert hasattr(decision, "authorized_level")
    assert hasattr(decision, "actionable")
    assert hasattr(decision, "trusted_value")
    assert hasattr(decision, "mining_domain")
    assert hasattr(decision, "proposed_action")
    assert hasattr(decision, "operation_status")
    assert hasattr(decision, "safety_flags")
    assert hasattr(decision, "used_fallback")
    assert hasattr(decision, "timestamp")
    assert hasattr(decision, "decision_id")
    assert hasattr(decision, "reasons")
    assert hasattr(decision, "metadata")
    assert isinstance(decision.operation_status, MiningOperationStatus)
    print("✓ MiningDecision has all required fields")
    return True


# ---------------------------------------------------------------------------
# Thermal override test
# ---------------------------------------------------------------------------

def test_thermal_override_fires():
    """Governor blocks action and emits thermal_override event when temp >= threshold."""
    policy = MiningPolicy(thermal_throttle_temp_c=80.0, enable_audit_events=True)
    governor = MiningGovernor(policy=policy)

    # Readings above throttle threshold
    signals = MiningSignals(
        mining_domain=MiningDomain.THERMAL,
        proposed_action=MiningAction.THERMAL_THROTTLE,
        ai_value=85.0,
        ai_confidence=0.90,
        hardware_readings=[
            HardwareReading(83.0, time.time(), "gpu_00"),
            HardwareReading(85.0, time.time(), "gpu_01"),
        ],
        rig_id="rig_00",
    )
    decision = governor.evaluate(signals)

    assert not decision.actionable
    assert decision.authorized_level == CryptoMiningTrustLevel.NO_ACTION
    assert decision.operation_status == MiningOperationStatus.CRITICAL
    assert decision.fallback_reason is not None
    assert "thermal override" in decision.fallback_reason.lower()

    events = governor.get_events()
    assert any(e.event_type == "thermal_override" for e in events)
    print("✓ Thermal override fires correctly")
    return True


# ---------------------------------------------------------------------------
# Rate limiting test
# ---------------------------------------------------------------------------

def test_pool_switch_rate_limit():
    """Governor enforces pool switch rate limit per policy."""
    policy = MiningPolicy(max_pool_switches_per_hour=0, enable_audit_events=True)
    governor = MiningGovernor(policy=policy)

    decisions = []
    for i in range(3):
        signals = make_pool_signals()
        decisions.append(governor.evaluate(signals))

    rate_limited = [d for d in decisions if "Rate limit" in (d.fallback_reason or "")]
    assert len(rate_limited) == 3, (
        f"Expected 3 rate-limited decisions, got {len(rate_limited)}"
    )
    assert all(not d.actionable for d in rate_limited)
    assert all(d.authorized_level == CryptoMiningTrustLevel.NO_ACTION for d in rate_limited)
    print(f"✓ Pool switch rate limiting works: {len(rate_limited)} decisions blocked out of 3")
    return True


# ---------------------------------------------------------------------------
# Events and history tests
# ---------------------------------------------------------------------------

def test_events_recorded():
    """Governor records events for each evaluate() call when policy enables it."""
    governor = create_mining_governor()
    initial_count = len(governor.get_events())

    governor.evaluate(make_hash_rate_signals())
    governor.evaluate(make_power_signals())

    events = governor.get_events()
    assert len(events) == initial_count + 2
    for event in events:
        assert isinstance(event, MiningEvent)
        assert event.timestamp > 0
        assert isinstance(event.mining_domain, MiningDomain)
    print(f"✓ {len(events)} events recorded correctly")
    return True


def test_decision_history():
    """Governor records decision history when policy enables it."""
    governor = create_mining_governor()
    governor.evaluate(make_hash_rate_signals())
    governor.evaluate(make_power_signals())

    history = governor.get_decision_history()
    assert len(history) == 2
    assert all(isinstance(d, MiningDecision) for d in history)
    print("✓ Decision history recorded correctly")
    return True


def test_no_events_when_disabled():
    """Governor does not record events when policy disables it."""
    governor = create_permissive_governor()  # enable_audit_events=False
    governor.evaluate(make_hash_rate_signals())
    governor.evaluate(make_power_signals())

    assert len(governor.get_events()) == 0
    assert len(governor.get_decision_history()) == 0
    print("✓ Events/history correctly suppressed when disabled")
    return True


# ---------------------------------------------------------------------------
# Health and metrics tests
# ---------------------------------------------------------------------------

def test_get_health():
    """Governor.get_health() returns a MiningOperationStatus."""
    governor = create_mining_governor()
    health = governor.get_health()
    assert isinstance(health, MiningOperationStatus)
    print(f"✓ get_health() returned: {health.value}")
    return True


def test_get_subsystem_health():
    """Governor.get_subsystem_health() returns a dict keyed by domain name."""
    governor = create_mining_governor()
    subsystem = governor.get_subsystem_health()
    assert isinstance(subsystem, dict)
    for domain in MiningDomain:
        assert domain.value in subsystem
        assert isinstance(subsystem[domain.value], MiningOperationStatus)
    print("✓ get_subsystem_health() returned correct structure")
    return True


def test_get_metrics():
    """Governor.get_metrics() returns expected keys."""
    governor = create_mining_governor()
    governor.evaluate(make_hash_rate_signals())
    metrics = governor.get_metrics()

    assert "fallback_rate" in metrics
    assert "avg_confidence" in metrics
    assert "total_decisions" in metrics
    assert "pool_switches_this_hour" in metrics
    assert "overall_health" in metrics
    assert metrics["total_decisions"] == 1
    print(f"✓ get_metrics() returned: {metrics}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_imports,
        test_top_level_imports,
        test_domain_registry,
        test_enums,
        test_factory_functions,
        test_policy_overrides,
        test_validate_signals_valid,
        test_validate_signals_bad_confidence,
        test_validate_signals_no_hardware,
        test_validate_signals_thermal_out_of_range,
        test_validate_signals_pool_score_out_of_range,
        test_validate_signals_negative_hash_rate,
        test_evaluate_hash_rate,
        test_evaluate_thermal,
        test_evaluate_pool,
        test_evaluate_power,
        test_evaluate_all_domains,
        test_decision_fields,
        test_thermal_override_fires,
        test_pool_switch_rate_limit,
        test_events_recorded,
        test_decision_history,
        test_no_events_when_disabled,
        test_get_health,
        test_get_subsystem_health,
        test_get_metrics,
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
