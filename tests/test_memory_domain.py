# Licensed under the PolyForm Noncommercial License 1.0.0
import pytest
import time

from ailee.domains.memory import (
    MemoryGovernor,
    MemoryPolicy,
    MemorySignals,
    MemoryDecision,
    MemoryEvent,
    MemoryTrustLevel,
    MemoryHealthStatus,
    MemoryDomain,
    MemoryAction,
    MemoryReading,
    RAM_ALLOCATION,
    HEAP_MONITORING,
    SWAP_MANAGEMENT,
    PROCESS_MEMORY,
    create_memory_governor,
    create_default_governor,
    create_strict_governor,
    create_permissive_governor,
    validate_memory_signals,
)
import ailee

def test_imports():
    assert MemoryGovernor is not None
    assert MemoryPolicy is not None

def test_top_level_imports():
    from ailee import MemoryGovernor as MG
    assert MG is not None
    assert ailee.get_available_domains().get("memory") is True

def test_enums():
    assert MemoryTrustLevel.NO_ACTION < MemoryTrustLevel.AUTONOMOUS
    assert MemoryHealthStatus.HEALTHY == "HEALTHY"
    assert MemoryDomain.RAM_ALLOCATION == "RAM_ALLOCATION"
    assert MemoryAction.ENFORCE_ALLOCATION_LIMIT == "ENFORCE_ALLOCATION_LIMIT"

def test_factory_functions():
    gov1 = create_memory_governor()
    gov2 = create_default_governor()
    gov3 = create_strict_governor()
    gov4 = create_permissive_governor()
    assert isinstance(gov1, MemoryGovernor)
    assert isinstance(gov2, MemoryGovernor)
    assert isinstance(gov3, MemoryGovernor)
    assert isinstance(gov4, MemoryGovernor)
    assert gov3.policy.min_trust_for_action == MemoryTrustLevel.AUTONOMOUS
    assert gov4.policy.min_trust_for_action == MemoryTrustLevel.ADVISORY

def test_policy_overrides():
    gov = create_memory_governor(max_oom_kills_per_hour=99)
    assert gov.policy.max_oom_kills_per_hour == 99

def test_validate_signals_valid():
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=0.9,
        memory_readings=[MemoryReading(0.5, time.time(), "s1")]
    )
    assert validate_memory_signals(signals) == []

def test_validate_signals_bad_confidence():
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=1.5,
        memory_readings=[MemoryReading(0.5, time.time(), "s1")]
    )
    issues = validate_memory_signals(signals)
    assert any("ai_confidence" in i for i in issues)

def test_validate_signals_no_readings():
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=0.9,
        memory_readings=[]
    )
    issues = validate_memory_signals(signals)
    assert any("reading" in i.lower() or "sensor" in i.lower() for i in issues)

def test_validate_signals_out_of_range():
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=1.5,
        ai_confidence=0.9,
        memory_readings=[MemoryReading(0.5, time.time(), "s1")]
    )
    issues = validate_memory_signals(signals)
    assert any("range" in i.lower() or "0.0 and 1.0" in i for i in issues)

def test_evaluate_ram_allocation():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    dec = gov.evaluate(signals)
    assert dec.memory_domain == MemoryDomain.RAM_ALLOCATION
    assert dec.decision_id is not None
    assert isinstance(dec, MemoryDecision)

def test_evaluate_heap():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.HEAP,
        proposed_action=MemoryAction.TRIGGER_GC,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    dec = gov.evaluate(signals)
    assert dec.memory_domain == MemoryDomain.HEAP

def test_evaluate_swap():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.SWAP,
        proposed_action=MemoryAction.ENABLE_SWAP,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    dec = gov.evaluate(signals)
    assert dec.memory_domain == MemoryDomain.SWAP

def test_evaluate_process():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.PROCESS,
        proposed_action=MemoryAction.THROTTLE_PROCESS,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    dec = gov.evaluate(signals)
    assert dec.memory_domain == MemoryDomain.PROCESS

def test_evaluate_all_domains():
    gov = create_memory_governor()
    for domain in MemoryDomain:
        signals = MemorySignals(
            memory_domain=domain,
            proposed_action=MemoryAction.NO_ACTION,
            ai_value=0.5,
            ai_confidence=0.95,
            memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
        )
        dec = gov.evaluate(signals)
        assert dec.memory_domain == domain

def test_oom_override():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.EVICT_CACHE,
        ai_value=0.98,
        ai_confidence=0.99,
        memory_readings=[MemoryReading(0.99, time.time(), "s1")]
    )
    dec = gov.evaluate(signals)
    assert dec.health_status == MemoryHealthStatus.CRITICAL
    assert not dec.actionable
    events = gov.get_events()
    assert events[-1].event_type == "oom_override"

def test_oom_kill_rate_limit():
    gov = create_memory_governor(max_oom_kills_per_hour=0)
    signals = MemorySignals(
        memory_domain=MemoryDomain.PROCESS,
        proposed_action=MemoryAction.KILL_PROCESS,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1")]
    )
    dec = gov.evaluate(signals)
    assert not dec.actionable
    assert dec.health_status == MemoryHealthStatus.WARNING
    assert "rate_limit" in dec.safety_flags

def test_decision_history():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    gov.evaluate(signals)
    gov.evaluate(signals)
    assert len(gov.get_decision_history()) == 2

def test_events_recorded():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    gov.evaluate(signals)
    gov.evaluate(signals)
    assert len(gov.get_events()) >= 2

def test_no_events_when_disabled():
    gov = create_permissive_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    gov.evaluate(signals)
    assert len(gov.get_events()) == 0

def test_get_health():
    gov = create_memory_governor()
    assert isinstance(gov.get_health(), MemoryHealthStatus)

def test_get_subsystem_health():
    gov = create_memory_governor()
    sub_health = gov.get_subsystem_health()
    for d in MemoryDomain:
        assert d.value in sub_health

def test_get_metrics():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    gov.evaluate(signals)
    metrics = gov.get_metrics()
    assert "fallback_rate" in metrics
    assert "avg_confidence" in metrics
    assert "total_decisions" in metrics
    assert "oom_kills_this_hour" in metrics
    assert "overall_health" in metrics

def test_decision_fields():
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.5,
        ai_confidence=0.95,
        memory_readings=[MemoryReading(0.5, time.time(), "s1"), MemoryReading(0.51, time.time(), "s2"), MemoryReading(0.49, time.time(), "s3")]
    )
    dec = gov.evaluate(signals)
    assert hasattr(dec, "authorized_level")
    assert hasattr(dec, "actionable")
    assert hasattr(dec, "trusted_value")
    assert hasattr(dec, "memory_domain")
    assert hasattr(dec, "proposed_action")
    assert hasattr(dec, "health_status")
    assert hasattr(dec, "safety_flags")
    assert hasattr(dec, "used_fallback")
    assert hasattr(dec, "fallback_reason")
    assert hasattr(dec, "timestamp")
    assert hasattr(dec, "decision_id")
    assert hasattr(dec, "reasons")
    assert hasattr(dec, "metadata")

def test_config_presets():
    for conf in [RAM_ALLOCATION, HEAP_MONITORING, SWAP_MANAGEMENT, PROCESS_MEMORY]:
        assert conf.hard_max <= 1.0
