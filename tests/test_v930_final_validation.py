"""Synthetic release-gate simulations for the two v9.3 trust dimensions."""

from dataclasses import replace
from datetime import timedelta
import math

import pytest

from ailee.domains.dual_domain import GovernedAnalyticsService
from ailee.domains.industrial import (
    EventLedger,
    IndustrialEvent,
    ProcessState,
    TelemetryValidity,
    ThroughputGovernor,
    ThroughputReason,
)
from ailee.domains.licensing import (
    CredentialStatus,
    LicenseGovernor,
    LicenseReason,
)
from tests.test_dual_domain import fixtures
from tests.test_industrial_domain import (
    NOW,
    START,
    interval,
    material,
    state_observation,
)


@pytest.mark.parametrize(
    ("change_target", "changes", "reason"),
    [
        (
            "request",
            {"capability": "unknown_capability"},
            LicenseReason.ENTITLEMENT_MISSING,
        ),
        ("request", {"schema_version": "99"}, LicenseReason.UNSUPPORTED_SCHEMA),
        ("evidence", {"schema_version": "99"}, LicenseReason.UNSUPPORTED_SCHEMA),
        ("evidence", {"asset_id": "machine-b"}, LicenseReason.INVALID_CREDENTIAL),
        ("evidence", {"issuer": "issuer-other"}, LicenseReason.INVALID_CREDENTIAL),
    ],
)
def test_licensing_schema_capability_and_credential_tampering_fail_closed(
    change_target, changes, reason
):
    contract, request, _ = fixtures()
    if change_target == "request":
        request = replace(request, **changes)
    else:
        request = replace(request, evidence=replace(request.evidence, **changes))
    decision = LicenseGovernor().authorize(contract, request)
    assert not decision.authorized
    assert decision.reason is reason
    assert decision.audit.reason_codes == (reason.value,)


class RejectingIntegrityVerifier:
    """Synthetic upstream verifier response for altered signed content."""

    def verify(self, evidence):
        return CredentialStatus.INVALID


def test_upstream_integrity_rejection_denies_tampered_entitlement_payload():
    contract, request, _ = fixtures()
    decision = LicenseGovernor(RejectingIntegrityVerifier()).authorize(
        contract, request
    )
    assert decision.reason is LicenseReason.INVALID_CREDENTIAL
    assert not decision.authorized


@pytest.mark.parametrize(
    ("changes", "reason", "validity"),
    [
        (
            {"schema_version": "99"},
            ThroughputReason.UNSUPPORTED_SCHEMA,
            TelemetryValidity.UNSUPPORTED_SCHEMA,
        ),
        (
            {"start": NOW, "end": NOW},
            ThroughputReason.INVALID_TIMESTAMP,
            TelemetryValidity.INVALID_TIMESTAMP,
        ),
        (
            {"start": NOW + timedelta(seconds=1), "end": NOW + timedelta(seconds=2)},
            ThroughputReason.INVALID_TIMESTAMP,
            TelemetryValidity.INVALID_TIMESTAMP,
        ),
    ],
)
def test_interval_schema_zero_time_and_future_time_are_explicit(
    changes, reason, validity
):
    result = ThroughputGovernor().evaluate(interval(**changes), NOW)
    assert not result.available
    assert result.reason is reason
    assert result.telemetry_validity is validity
    assert result.productive_seconds == 0


def test_future_observation_wrong_asset_and_unit_mismatch_are_rejected():
    future = state_observation(
        timestamp=NOW + timedelta(seconds=1), received_at=NOW + timedelta(seconds=1)
    )
    wrong_asset = state_observation(machine_id="machine-b")
    unit_end = replace(interval().material_end, unit="lb")
    governor = ThroughputGovernor()
    assert (
        governor.evaluate(interval(state_observations=(future,)), NOW).reason
        is ThroughputReason.INVALID_TIMESTAMP
    )
    assert (
        governor.evaluate(interval(state_observations=(wrong_asset,)), NOW).reason
        is ThroughputReason.CONTRADICTORY_STATE
    )
    assert (
        governor.evaluate(interval(material_end=unit_end), NOW).reason
        is ThroughputReason.UNIT_MISMATCH
    )


@pytest.mark.parametrize("quantity", [math.nan, math.inf, -math.inf, 99.0])
def test_invalid_material_values_never_reach_trusted_analytics(quantity):
    result = ThroughputGovernor().evaluate(
        interval(material_end=material("material-2", NOW, quantity)), NOW
    )
    assert not result.available
    assert result.reason is ThroughputReason.INVALID_MATERIAL_VALUE
    assert result.quantity_per_hour is None


def test_overlapping_intervals_are_independent_not_implicitly_aggregated():
    first = interval(interval_id="interval-1")
    second = interval(
        interval_id="interval-2",
        start=START + timedelta(seconds=30),
        material_start=material("material-3", START + timedelta(seconds=30), 105.0),
        material_end=material("material-4", NOW, 110.0),
    )
    governor = ThroughputGovernor()
    first_result = governor.evaluate(first, NOW)
    second_result = governor.evaluate(second, NOW)
    assert first_result.productive_seconds == 60
    assert second_result.productive_seconds == 30
    # There is deliberately no collection aggregator that could silently sum overlaps.
    assert not hasattr(governor, "aggregate")


def test_event_occurrence_ties_reordered_arrival_and_late_arrival_are_deterministic():
    ledger = EventLedger(evaluated_at=NOW)
    tied_b = IndustrialEvent(
        "event-b", "machine-a", START, "ALARM", "plc", received_at=START
    )
    tied_a_late = replace(tied_b, event_id="event-a", received_at=NOW)
    later = replace(
        tied_b,
        event_id="event-c",
        timestamp=START + timedelta(seconds=1),
        received_at=NOW,
    )
    for event in (later, tied_b, tied_a_late):
        ledger.append(event)
    assert [event.event_id for event in ledger.chronological()] == [
        "event-a",
        "event-b",
        "event-c",
    ]


@pytest.mark.parametrize(
    ("license_mode", "telemetry_mode", "auth_reason", "throughput_reason", "permitted"),
    [
        ("valid", "valid", LicenseReason.AUTHORIZED, ThroughputReason.CALCULATED, True),
        (
            "no_entitlement",
            "valid",
            LicenseReason.ENTITLEMENT_MISSING,
            ThroughputReason.CALCULATED,
            False,
        ),
        (
            "valid",
            "missing",
            LicenseReason.AUTHORIZED,
            ThroughputReason.MISSING_TELEMETRY,
            False,
        ),
        (
            "invalid",
            "valid",
            LicenseReason.INVALID_CREDENTIAL,
            ThroughputReason.CALCULATED,
            False,
        ),
        (
            "unavailable",
            "valid",
            LicenseReason.VERIFICATION_UNAVAILABLE,
            ThroughputReason.CALCULATED,
            False,
        ),
        (
            "no_entitlement",
            "fault",
            LicenseReason.ENTITLEMENT_MISSING,
            ThroughputReason.FAULT_EXCLUDED,
            False,
        ),
    ],
)
def test_cross_domain_matrix(
    license_mode, telemetry_mode, auth_reason, throughput_reason, permitted
):
    contract, request, process_interval = fixtures()
    service = GovernedAnalyticsService()
    if license_mode == "no_entitlement":
        contract = replace(contract, entitlements=("base_operation",))
    elif license_mode == "invalid":
        request = replace(
            request, evidence=replace(request.evidence, status=CredentialStatus.INVALID)
        )
    elif license_mode == "unavailable":

        class BrokenVerifier:
            def verify(self, evidence):
                raise RuntimeError("synthetic verifier outage")

        service = GovernedAnalyticsService(
            license_governor=LicenseGovernor(BrokenVerifier())
        )
    if telemetry_mode == "missing":
        process_interval = replace(process_interval, state_observations=())
    elif telemetry_mode == "fault":
        process_interval = replace(
            process_interval,
            start_state=ProcessState.FAULT,
            end_state=ProcessState.FAULT,
            state_observations=(
                replace(process_interval.state_observations[0], value="FAULT"),
            ),
        )
    before = (contract, request, process_interval)
    decision = service.evaluate(contract, request, process_interval)
    assert decision.authorization.reason is auth_reason
    assert decision.throughput.reason is throughput_reason
    assert decision.analytics_permitted is permitted
    assert not decision.machine_control_issued
    assert before == (contract, request, process_interval)


def test_representative_composed_result_is_repeatable():
    service = GovernedAnalyticsService()
    inputs = fixtures()
    results = [service.evaluate(*inputs) for _ in range(10)]
    assert all(result == results[0] for result in results)
    assert len({result.authorization.decision_id for result in results}) == 1
    assert len({result.throughput.decision_id for result in results}) == 1
