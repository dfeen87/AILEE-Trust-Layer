from dataclasses import replace
from datetime import datetime, timedelta, timezone

from ailee.domains.dual_domain import GovernedAnalyticsService
from ailee.domains.industrial import (
    MaterialObservation, ProcessInterval, ProcessState, TelemetryObservation,
    ThroughputReason,
)
from ailee.domains.licensing import (
    AuthorizationRequest, CredentialStatus, IntegrityEvidence, LicenseContract,
    LicenseReason,
)


NOW = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
START = NOW - timedelta(minutes=1)


def fixtures():
    contract = LicenseContract(
        "license-1", "customer-1", ("machine-a",), "issuer-1",
        NOW - timedelta(days=1), NOW + timedelta(days=1), ("advanced_analytics",),
    )
    evidence = IntegrityEvidence(
        "evidence-1", "signed", CredentialStatus.VALID, "issuer-1",
        "license-1", "machine-a", NOW, customer_id="customer-1",
    )
    request = AuthorizationRequest(
        "request-1", "customer-1", "machine-a", "advanced_analytics", NOW, evidence,
    )
    telemetry = TelemetryObservation(
        "state-1", "machine-a", "state", NOW - timedelta(seconds=5),
        "RUNNING", "state", "readonly-adapter", NOW - timedelta(seconds=4),
    )
    interval = ProcessInterval(
        "interval-1", "machine-a", START, NOW, ProcessState.RUNNING,
        ProcessState.RUNNING, "historian", (telemetry,),
        MaterialObservation("m1", "machine-a", START, 0, "kg", "scale", START),
        MaterialObservation("m2", "machine-a", NOW, 2, "kg", "scale", NOW),
    )
    return contract, request, interval


def test_valid_telemetry_without_entitlement_denies_analytics_only():
    contract, request, interval = fixtures()
    decision = GovernedAnalyticsService().evaluate(
        replace(contract, entitlements=("base_operation",)), request, interval
    )
    assert decision.authorization.reason is LicenseReason.ENTITLEMENT_MISSING
    assert decision.throughput.available
    assert not decision.analytics_permitted
    assert decision.observed_process_state == "RUNNING"
    assert not decision.machine_control_issued


def test_valid_entitlement_with_bad_telemetry_withholds_trusted_analytics():
    contract, request, interval = fixtures()
    decision = GovernedAnalyticsService().evaluate(
        contract, request, replace(interval, state_observations=())
    )
    assert decision.authorization.authorized
    assert decision.throughput.reason is ThroughputReason.MISSING_TELEMETRY
    assert not decision.analytics_permitted


def test_valid_entitlement_and_evidence_permit_calculable_analytics():
    decision = GovernedAnalyticsService().evaluate(*fixtures())
    assert decision.authorization.authorized
    assert decision.throughput.available
    assert decision.analytics_permitted
    assert decision.throughput.quantity_per_hour == 120


def test_licensing_failure_never_modifies_process_or_safety_state():
    contract, request, interval = fixtures()
    fault_interval = replace(
        interval, start_state=ProcessState.FAULT, end_state=ProcessState.FAULT
    )
    decision = GovernedAnalyticsService().evaluate(
        contract, replace(request, evidence=None), fault_interval
    )
    assert decision.authorization.reason is LicenseReason.INSUFFICIENT_EVIDENCE
    assert decision.observed_process_state == "FAULT"
    assert decision.throughput.reason is ThroughputReason.FAULT_EXCLUDED
    assert not decision.machine_control_issued
