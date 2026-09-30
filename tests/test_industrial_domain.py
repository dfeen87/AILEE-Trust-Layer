from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ailee.domains.industrial import (
    EventLedger, IndustrialEvent, MaterialObservation, ProcessInterval, ProcessState,
    TelemetryObservation, TelemetryValidity, ThroughputGovernor, ThroughputReason,
)


NOW = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
START = NOW - timedelta(minutes=1)


def state_observation(**changes):
    values = dict(
        observation_id="state-1", machine_id="machine-a", signal_id="process_state",
        timestamp=NOW - timedelta(seconds=10), value="RUNNING", unit="state",
        source="controller-readonly", received_at=NOW - timedelta(seconds=9),
    )
    values.update(changes)
    return TelemetryObservation(**values)


def material(identity, timestamp, quantity):
    return MaterialObservation(identity, "machine-a", timestamp, quantity, "kg", "scale-readonly", timestamp)


def interval(**changes):
    values = dict(
        interval_id="interval-1", machine_id="machine-a", start=START, end=NOW,
        start_state=ProcessState.RUNNING, end_state=ProcessState.RUNNING,
        source="historian-readonly", state_observations=(state_observation(),),
        material_start=material("material-1", START, 100.0),
        material_end=material("material-2", NOW, 110.0),
    )
    values.update(changes)
    return ProcessInterval(**values)


def test_running_interval_produces_quantity_over_productive_time():
    result = ThroughputGovernor().evaluate(interval(), NOW)
    assert result.available
    assert result.reason is ThroughputReason.CALCULATED
    assert result.telemetry_validity is TelemetryValidity.VALID
    assert result.elapsed_seconds == result.productive_seconds == 60
    assert result.material_delta == 10
    assert result.quantity_per_hour == 600


@pytest.mark.parametrize(
    ("state", "reason", "validity"),
    [
        (ProcessState.FAULT, ThroughputReason.FAULT_EXCLUDED, TelemetryValidity.ACTIVE_FAULT),
        (ProcessState.UNPLANNED_STOP, ThroughputReason.NON_PRODUCTIVE_STATE, TelemetryValidity.VALID),
        (ProcessState.PLANNED_HOLD, ThroughputReason.NON_PRODUCTIVE_STATE, TelemetryValidity.VALID),
    ],
)
def test_nonproductive_intervals_are_excluded(state, reason, validity):
    observation = state_observation(value=state.value)
    result = ThroughputGovernor().evaluate(
        interval(start_state=state, end_state=state, state_observations=(observation,)), NOW
    )
    assert not result.available
    assert result.reason is reason
    assert result.telemetry_validity is validity
    assert result.productive_seconds == 0
    assert result.material_delta is None


def test_missing_and_stale_telemetry_make_analytics_unavailable():
    missing = ThroughputGovernor().evaluate(interval(state_observations=()), NOW)
    stale_obs = state_observation(timestamp=NOW - timedelta(minutes=10), received_at=NOW - timedelta(minutes=9))
    stale = ThroughputGovernor().evaluate(interval(state_observations=(stale_obs,)), NOW)
    assert missing.reason is ThroughputReason.MISSING_TELEMETRY
    assert stale.reason is ThroughputReason.STALE_TELEMETRY
    assert not missing.available and not stale.available


def test_invalid_timestamp_order_is_rejected():
    result = ThroughputGovernor().evaluate(interval(start=NOW, end=START), NOW)
    assert result.reason is ThroughputReason.INVALID_TIMESTAMP
    assert result.productive_seconds == 0


def test_missing_material_does_not_fabricate_throughput():
    result = ThroughputGovernor().evaluate(interval(material_end=None), NOW)
    assert result.reason is ThroughputReason.MATERIAL_EVIDENCE_MISSING
    assert result.material_delta is None and result.quantity_per_hour is None


def test_contradictory_state_and_invalid_quantity_are_explicit():
    contradictory = ThroughputGovernor().evaluate(interval(end_state=ProcessState.FAULT), NOW)
    decreasing = ThroughputGovernor().evaluate(
        interval(material_end=material("material-2", NOW, 99.0)), NOW
    )
    assert contradictory.reason is ThroughputReason.CONTRADICTORY_STATE
    assert decreasing.reason is ThroughputReason.INVALID_MATERIAL_VALUE


def test_event_chronology_is_deterministic_and_not_root_cause_inference():
    ledger = EventLedger()
    later = IndustrialEvent("event-b", "machine-a", NOW, "ALARM", "controller", severity="high")
    earlier_b = replace(later, event_id="event-c", timestamp=START)
    earlier_a = replace(later, event_id="event-a", timestamp=START)
    for event in (later, earlier_b, earlier_a):
        ledger.append(event)
    events = ledger.chronological()
    assert [event.event_id for event in events] == ["event-a", "event-c", "event-b"]
    assert all("root" not in event.metadata for event in events)


def test_identical_evidence_produces_identical_result():
    governor = ThroughputGovernor()
    assert governor.evaluate(interval(), NOW) == governor.evaluate(interval(), NOW)
