from dataclasses import replace
from datetime import timedelta
import math

import pytest

from ailee.domains.industrial import (
    EventLedger, IndustrialEvent, ProcessState, ProcessTransitionPolicy,
    TelemetryValidator, TelemetryValidity, ThroughputGovernor, ThroughputReason,
)
from ailee.domains.licensing import LicenseGovernor, LicenseReason
from tests.test_dual_domain import fixtures
from tests.test_industrial_domain import NOW, START, interval, material, state_observation


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("asset_id", " machine-a"),
    ("capability", "advanced_analytics/*"), ("customer_id", "CUSTOMER-1"),
])
def test_malformed_or_noncanonical_request_identity_fails_closed(field, value):
    contract, request, _ = fixtures()
    decision = LicenseGovernor().authorize(contract, replace(request, **{field: value}))
    expected = LicenseReason.INSUFFICIENT_EVIDENCE if not value else LicenseReason.MALFORMED_IDENTIFIER
    if field == "customer_id" and value == "CUSTOMER-1":
        expected = LicenseReason.MALFORMED_IDENTIFIER
    assert decision.reason is expected
    assert not decision.authorized


def test_duplicate_contract_bindings_and_unsupported_schema_fail_closed():
    contract, request, _ = fixtures()
    duplicate = replace(contract, entitlements=("advanced_analytics", "advanced_analytics"))
    assert LicenseGovernor().authorize(duplicate, request).reason is LicenseReason.INVALID_CONTRACT
    assert LicenseGovernor().authorize(replace(contract, schema_version="99"), request).reason is LicenseReason.UNSUPPORTED_SCHEMA


def test_credential_customer_substitution_future_verification_and_boundaries():
    contract, request, _ = fixtures()
    substituted = replace(request.evidence, customer_id="customer-2")
    future = replace(request.evidence, verified_at=request.evaluated_at + timedelta(seconds=1))
    governor = LicenseGovernor()
    assert governor.authorize(contract, replace(request, evidence=substituted)).reason is LicenseReason.INVALID_CREDENTIAL
    assert governor.authorize(contract, replace(request, evidence=future)).reason is LicenseReason.INVALID_CREDENTIAL
    assert governor.authorize(replace(contract, valid_from=request.evaluated_at), request).authorized
    assert governor.authorize(replace(contract, valid_until=request.evaluated_at), request).reason is LicenseReason.EXPIRED


class BrokenVerifier:
    def verify(self, evidence):
        raise RuntimeError("key service unavailable")


class IndeterminateVerifier:
    def verify(self, evidence):
        return None


@pytest.mark.parametrize("verifier", [BrokenVerifier(), IndeterminateVerifier()])
def test_verifier_failure_or_indeterminate_result_fails_closed(verifier):
    contract, request, _ = fixtures()
    assert LicenseGovernor(verifier).authorize(contract, request).reason is LicenseReason.VERIFICATION_UNAVAILABLE


class OneShotReplayGuard:
    def __init__(self):
        self.seen = set()

    def accept(self, request_id, evaluated_at):
        if request_id in self.seen:
            return False
        self.seen.add(request_id)
        return True


def test_optional_replay_boundary_rejects_duplicate_request():
    contract, request, _ = fixtures()
    governor = LicenseGovernor(replay_guard=OneShotReplayGuard())
    assert governor.authorize(contract, request).authorized
    assert governor.authorize(contract, request).reason is LicenseReason.REPLAY_DETECTED


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_telemetry_never_validates(value):
    observation = state_observation(value=value)
    assert TelemetryValidator().validate(observation, NOW) is TelemetryValidity.INVALID_VALUE


def test_duplicates_reordering_and_contradictory_state_are_rejected():
    first = state_observation(observation_id="state-1", timestamp=START + timedelta(seconds=20), received_at=START + timedelta(seconds=21))
    second = state_observation(observation_id="state-2", timestamp=START + timedelta(seconds=10), received_at=START + timedelta(seconds=11))
    assert ThroughputGovernor().evaluate(interval(state_observations=(first, first)), NOW).reason is ThroughputReason.DUPLICATE_EVIDENCE
    reordered = ThroughputGovernor().evaluate(interval(state_observations=(first, second)), NOW)
    assert reordered.telemetry_validity is TelemetryValidity.OUT_OF_ORDER
    contradictory = ThroughputGovernor().evaluate(interval(state_observations=(state_observation(value="FAULT"),)), NOW)
    assert contradictory.reason is ThroughputReason.CONTRADICTORY_STATE


@pytest.mark.parametrize("quantity", [math.nan, math.inf, -math.inf])
def test_nonfinite_material_never_produces_throughput(quantity):
    result = ThroughputGovernor().evaluate(interval(material_end=material("material-2", NOW, quantity)), NOW)
    assert result.reason is ThroughputReason.INVALID_MATERIAL_VALUE
    assert result.quantity_per_hour is None


def test_material_source_run_and_time_provenance_must_agree():
    base = interval()
    wrong_source = replace(base.material_end, source="other-scale")
    mixed_run = replace(base.material_start, run_id="run-1"), replace(base.material_end, run_id="run-2")
    future = replace(base.material_end, received_at=NOW + timedelta(seconds=1))
    assert ThroughputGovernor().evaluate(replace(base, material_end=wrong_source), NOW).reason is ThroughputReason.CONTRADICTORY_STATE
    assert ThroughputGovernor().evaluate(replace(base, material_start=mixed_run[0], material_end=mixed_run[1]), NOW).reason is ThroughputReason.CONTRADICTORY_STATE
    assert ThroughputGovernor().evaluate(replace(base, material_end=future), NOW).reason is ThroughputReason.INVALID_TIMESTAMP


def test_transition_policy_requires_explicit_recovery_evidence():
    policy = ProcessTransitionPolicy(allowed={(ProcessState.FAULT, ProcessState.RUNNING)})
    assert not policy.permits(ProcessState.FAULT, ProcessState.RUNNING)
    assert policy.permits(ProcessState.FAULT, ProcessState.RUNNING, "recovery-1")
    assert not policy.permits(ProcessState.COMPLETE, ProcessState.RUNNING, "recovery-1")


def test_event_ledger_rejects_future_duplicate_conflicting_and_unknown_events():
    ledger = EventLedger(evaluated_at=NOW, allowed_categories={"process"})
    valid = IndustrialEvent("event-1", "machine-a", START, "ALARM", "plc", category="process", received_at=START)
    ledger.append(valid)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.append(valid)
    with pytest.raises(ValueError, match="chronology"):
        ledger.append(replace(valid, event_id="event-2", timestamp=NOW + timedelta(seconds=1), received_at=NOW + timedelta(seconds=1)))
    with pytest.raises(ValueError, match="unknown"):
        EventLedger(allowed_categories={"known"}).append(replace(valid, event_id="event-3", category="unknown"))
    with pytest.raises(ValueError, match="conflicting"):
        ledger.append(replace(valid, event_id="event-4", metadata={"status": "different"}))


def test_license_verifier_outage_does_not_change_valid_process_evidence():
    from ailee.domains.dual_domain import GovernedAnalyticsService
    contract, request, process_interval = fixtures()
    decision = GovernedAnalyticsService(license_governor=LicenseGovernor(BrokenVerifier())).evaluate(
        contract, request, process_interval
    )
    assert decision.authorization.reason is LicenseReason.VERIFICATION_UNAVAILABLE
    assert decision.throughput.available
    assert decision.throughput.telemetry_validity is TelemetryValidity.VALID
    assert not decision.analytics_permitted and not decision.machine_control_issued


def test_invalid_license_does_not_rewrite_valid_telemetry():
    from ailee.domains.dual_domain import GovernedAnalyticsService
    contract, request, process_interval = fixtures()
    decision = GovernedAnalyticsService().evaluate(contract, replace(request, evidence=None), process_interval)
    assert not decision.authorization.authorized
    assert decision.throughput.available
    assert decision.throughput.telemetry_validity is TelemetryValidity.VALID


def test_industrial_analytics_exception_cannot_mutate_license_or_machine_inputs():
    from ailee.domains.dual_domain import GovernedAnalyticsService

    class BrokenAnalytics:
        def evaluate(self, process_interval, evaluated_at):
            raise RuntimeError("analytics offline")

    contract, request, process_interval = fixtures()
    original = (contract, request, process_interval)
    with pytest.raises(RuntimeError, match="analytics offline"):
        GovernedAnalyticsService(throughput_governor=BrokenAnalytics()).evaluate(*original)
    assert original == (contract, request, process_interval)


def test_partial_and_corrupted_typed_evidence_returns_reason_not_exception():
    contract, request, _ = fixtures()
    assert LicenseGovernor().authorize(replace(contract, asset_ids=None), request).reason is LicenseReason.INSUFFICIENT_EVIDENCE
    assert LicenseGovernor().authorize(contract, replace(request, evidence="corrupt")).reason is LicenseReason.VERIFICATION_UNAVAILABLE
    malformed = ThroughputGovernor().evaluate(replace(interval(), state_observations=(None,)), NOW)
    assert malformed.reason is ThroughputReason.MALFORMED_TELEMETRY
