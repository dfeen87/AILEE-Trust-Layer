"""Common-contract tests; test adapters do not represent native enforcement."""

from datetime import datetime, timezone

import pytest

from ailee.local_computing import (
    Capability,
    CapabilityRequest,
    CapabilitySupport,
    DeterministicPolicyEngine,
    EnforcementResult,
    EnforcementStatus,
    ExecutionContext,
    LocalComputingError,
    LocalComputingTrust,
    PlatformCapability,
    Policy,
    PolicyDecision,
    Principal,
    ResourceTarget,
    TrustState,
)

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


class ContractTestAdapter:
    """Exercises the common protocol only; it is not an OS implementation."""

    def __init__(self, support=CapabilitySupport.SUPPORTED):
        self.support = support
        self.calls = 0

    def capability(self, requested):
        return PlatformCapability(requested, self.support)

    def enforce(self, request, constraints):
        self.calls += 1
        return EnforcementResult(
            EnforcementStatus.ENFORCED,
            attempted=True,
            enforced=True,
            completed=False,
            detail="common contract test only",
        )


def policy():
    return Policy(
        "test-policy/v1",
        frozenset({"trusted-tool", "limited-tool"}),
        {"trusted-tool": frozenset({Capability.FILESYSTEM_READ})},
        {"limited-tool": frozenset({Capability.FILESYSTEM_READ})},
        {Capability.FILESYSTEM_READ: ("read-only",)},
    )


def request(principal="trusted-tool", trust=TrustState.TRUSTED):
    return CapabilityRequest(
        "request-1",
        Principal(principal),
        Capability.FILESYSTEM_READ,
        ResourceTarget("user-file", "opaque-target-1"),
        ExecutionContext("test", "user", "correlation-1"),
        trust,
    )


def service(adapter=None, audit_sink=None):
    return LocalComputingTrust(
        DeterministicPolicyEngine(policy()), adapter, audit_sink, clock=lambda: NOW
    )


def test_trusted_request_is_allowed_but_not_reported_completed():
    adapter = ContractTestAdapter()
    result = service(adapter).govern(request())
    assert result.policy.decision is PolicyDecision.ALLOW
    assert result.enforcement.status is EnforcementStatus.ENFORCED
    assert result.enforcement.completed is False
    assert result.audited is True


def test_denied_and_unknown_principals_are_never_attempted():
    adapter = ContractTestAdapter()
    denied = service(adapter).govern(request(trust=TrustState.UNTRUSTED))
    unknown = service(adapter).govern(request(principal="unknown"))
    assert denied.policy.decision is PolicyDecision.DENY
    assert unknown.error is LocalComputingError.UNKNOWN_PRINCIPAL
    assert adapter.calls == 0


def test_restricted_request_and_degradation_reduce_authority():
    adapter = ContractTestAdapter()
    limited = service(adapter).govern(request(principal="limited-tool"))
    degraded = service(adapter).govern(request(trust=TrustState.DEGRADED))
    assert limited.policy.decision is PolicyDecision.RESTRICT
    assert limited.policy.constraints == ("read-only",)
    assert degraded.policy.decision is PolicyDecision.RESTRICT
    assert TrustState.TRUSTED.degrade() is TrustState.DEGRADED
    assert TrustState.DEGRADED.degrade() is TrustState.UNTRUSTED
    assert TrustState.UNTRUSTED.degrade() is TrustState.UNTRUSTED


@pytest.mark.parametrize(
    ("bad_request", "expected"),
    [
        (
            request().__class__(
                "",
                request().principal,
                request().capability,
                request().target,
                request().context,
                request().trust_state,
            ),
            LocalComputingError.MALFORMED_REQUEST,
        ),
        (
            request().__class__(
                "request-1",
                request().principal,
                request().capability,
                ResourceTarget("", ""),
                request().context,
                request().trust_state,
            ),
            LocalComputingError.INVALID_TARGET,
        ),
        (request(trust=TrustState.INVALID), LocalComputingError.INVALID_TRUST_STATE),
    ],
)
def test_malformed_and_invalid_requests_fail_closed(bad_request, expected):
    result = service(ContractTestAdapter()).govern(bad_request)
    assert result.policy.decision is PolicyDecision.DENY
    assert result.error is expected
    assert result.enforcement.attempted is False


def test_unknown_capability_and_invalid_enum_values_are_rejected():
    mismatched = PlatformCapability(Capability.FILE_DELETE, CapabilitySupport.SUPPORTED)
    outcome = DeterministicPolicyEngine(policy()).evaluate(request(), mismatched)
    assert outcome.error is LocalComputingError.UNSUPPORTED_CAPABILITY
    assert outcome.decision is PolicyDecision.DENY
    with pytest.raises(ValueError):
        Capability("UNKNOWN")
    with pytest.raises(ValueError):
        TrustState(99)


def test_policy_is_deterministic_and_platform_limitations_remain_separate():
    engine = DeterministicPolicyEngine(policy())
    platform = PlatformCapability(
        Capability.FILESYSTEM_READ,
        CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
        ("host permission remains authoritative",),
    )
    assert engine.evaluate(request(), platform) == engine.evaluate(request(), platform)
    result = service(ContractTestAdapter(CapabilitySupport.OBSERVABLE_ONLY)).govern(
        request()
    )
    assert result.policy.decision is PolicyDecision.RESTRICT
    assert result.platform_capability.support is CapabilitySupport.OBSERVABLE_ONLY
    assert result.enforcement.status is EnforcementStatus.NOT_ATTEMPTED
    assert result.enforcement.enforced is False


def test_policy_engine_snapshots_mutable_configuration():
    allowed = {"trusted-tool": {Capability.FILESYSTEM_READ}}
    configured = Policy("snapshot/v1", frozenset({"trusted-tool"}), allowed, {}, {})
    engine = DeterministicPolicyEngine(configured)
    first = engine.evaluate(
        request(),
        PlatformCapability(Capability.FILESYSTEM_READ, CapabilitySupport.SUPPORTED),
    )
    allowed["trusted-tool"].clear()
    second = engine.evaluate(
        request(),
        PlatformCapability(Capability.FILESYSTEM_READ, CapabilitySupport.SUPPORTED),
    )
    assert first == second
    assert second.decision is PolicyDecision.ALLOW


def test_audit_is_structured_deterministic_and_excludes_target_identifier():
    events = []
    result = service(ContractTestAdapter(), events.append).govern(request())
    assert events == [result.audit]
    assert result.audit.event_id == "audit:request-1"
    assert result.audit.timestamp == "2026-10-01T00:00:00+00:00"
    assert result.audit.target_classification == "user-file"
    assert not hasattr(result.audit, "target_identifier")


def test_audit_failure_is_explicit_and_does_not_change_policy_decision():
    def broken_sink(event):
        raise OSError("unavailable")

    result = service(ContractTestAdapter(), broken_sink).govern(request())
    assert result.policy.decision is PolicyDecision.ALLOW
    assert result.audited is False
    assert result.error is LocalComputingError.AUDIT_FAILURE


def test_standalone_default_is_honestly_unavailable():
    result = service().govern(request())
    assert result.platform_capability.support is CapabilitySupport.UNAVAILABLE
    assert result.policy.decision is PolicyDecision.DENY
    assert result.error is LocalComputingError.PLATFORM_UNAVAILABLE
    assert result.enforcement.status is EnforcementStatus.NOT_ATTEMPTED


def test_platform_failures_are_explicit_and_never_allow():
    class BrokenCapabilityAdapter(ContractTestAdapter):
        def capability(self, requested):
            raise OSError("integration failed")

    class BrokenEnforcementAdapter(ContractTestAdapter):
        def enforce(self, request, constraints):
            raise OSError("enforcement failed")

    integration = service(BrokenCapabilityAdapter()).govern(request())
    enforcement = service(BrokenEnforcementAdapter()).govern(request())
    assert integration.policy.decision is PolicyDecision.DENY
    assert integration.error is LocalComputingError.PLATFORM_INTEGRATION_FAILURE
    assert enforcement.enforcement.status is EnforcementStatus.FAILED
    assert enforcement.error is LocalComputingError.ENFORCEMENT_FAILURE


def test_policy_configuration_validation_rejects_unknown_principal_rules():
    malformed = Policy("bad", frozenset({"known"}), {"unknown": frozenset()}, {}, {})
    with pytest.raises(ValueError, match="unknown principals"):
        DeterministicPolicyEngine(malformed)
