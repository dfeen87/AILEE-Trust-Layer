"""Adversarial regressions for the Local Computing trust boundary."""

from concurrent.futures import ThreadPoolExecutor

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


class HostileAdapter:
    def __init__(self, result=None):
        self.result = result or EnforcementResult(
            EnforcementStatus.COMPLETED, True, True, True
        )
        self.calls = 0
        self.observed = []

    def capability(self, requested):
        return PlatformCapability(requested, CapabilitySupport.SUPPORTED)

    def enforce(self, request, constraints):
        self.calls += 1
        self.observed.append(request)
        return self.result


def request(*, request_id="unique", target=None):
    return CapabilityRequest(
        request_id,
        Principal("known"),
        Capability.SUBPROCESS_CREATE,
        target or ResourceTarget("tool", "/bin/true"),
        ExecutionContext("linux"),
        TrustState.TRUSTED,
    )


def service(adapter):
    policy = Policy(
        "adversarial",
        frozenset({"known"}),
        {"known": frozenset({Capability.SUBPROCESS_CREATE})},
        {},
        {},
    )
    return LocalComputingTrust(DeterministicPolicyEngine(policy), adapter)


@pytest.mark.parametrize(
    "result",
    [
        EnforcementResult(
            EnforcementStatus.COMPLETED,
            True,
            True,
            True,
            LocalComputingError.ENFORCEMENT_FAILURE,
        ),
        EnforcementResult(
            EnforcementStatus.FAILED,
            True,
            True,
            False,
            LocalComputingError.ENFORCEMENT_FAILURE,
        ),
        EnforcementResult(
            EnforcementStatus.ENFORCED,
            True,
            True,
            False,
            LocalComputingError.ENFORCEMENT_FAILURE,
        ),
        EnforcementResult(EnforcementStatus.COMPLETED, 1, True, True),
        EnforcementResult(
            EnforcementStatus.COMPLETED, True, True, True, detail="x" * 4097
        ),
    ],
)
def test_contradictory_or_malformed_adapter_evidence_fails_closed(result):
    governed = service(HostileAdapter(result)).govern(request())
    assert governed.enforcement.status is EnforcementStatus.FAILED
    assert governed.error is LocalComputingError.UNEXPECTED_RESULT


def test_capability_mismatch_fails_before_enforcement():
    adapter = HostileAdapter()
    adapter.capability = lambda requested: PlatformCapability(
        Capability.FILE_DELETE, CapabilitySupport.SUPPORTED
    )
    governed = service(adapter).govern(request())
    assert governed.policy.decision is PolicyDecision.DENY
    assert governed.error is LocalComputingError.PLATFORM_INTEGRATION_FAILURE
    assert adapter.calls == 0


def test_request_is_snapshotted_before_adapter_use():
    adapter = HostileAdapter()
    original = request(target=ResourceTarget("tool", "/bin/true", ("safe",)))
    governed = service(adapter).govern(original)
    object.__setattr__(original.target, "arguments", ("mutated",))
    assert governed.enforcement.completed
    assert adapter.observed[0] is not original
    assert adapter.observed[0].target.arguments == ("safe",)


def test_concurrent_duplicate_request_id_executes_once():
    adapter = HostileAdapter()
    trust = service(adapter)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: trust.govern(request()), range(16)))
    assert adapter.calls == 1
    assert sum(item.enforcement.completed for item in results) == 1
    assert (
        sum(item.error is LocalComputingError.REPLAYED_REQUEST for item in results)
        == 15
    )


@pytest.mark.parametrize(
    "target",
    [
        ResourceTarget("tool", "/bin/true", ("bad\0arg",)),
        ResourceTarget(
            "tool", "/bin/true", attributes=(("timeout", "1"), ("timeout", "2"))
        ),
        ResourceTarget("tool", "/bin/true", attributes=(("environment", "SECRET=x"),)),
        ResourceTarget("tool", "/bin/true", attributes=(("timeout", "nan"),)),
    ],
)
def test_hostile_arguments_and_attributes_never_reach_adapter(target):
    adapter = HostileAdapter()
    governed = service(adapter).govern(request(target=target))
    assert governed.policy.decision is PolicyDecision.DENY
    assert governed.error is LocalComputingError.INVALID_TARGET
    assert adapter.calls == 0


def test_audit_construction_failure_is_explicit_and_does_not_change_completion():
    trust = service(HostileAdapter())
    trust._clock = lambda: (_ for _ in ()).throw(RuntimeError("clock unavailable"))
    governed = trust.govern(request())
    assert governed.enforcement.completed
    assert governed.audit is None and not governed.audited
    assert governed.error is LocalComputingError.AUDIT_FAILURE


def test_policy_rejects_conflicting_and_malformed_rules():
    with pytest.raises(ValueError):
        DeterministicPolicyEngine(
            Policy(
                "conflict",
                frozenset({"known"}),
                {"known": frozenset({Capability.SUBPROCESS_CREATE})},
                {"known": frozenset({Capability.SUBPROCESS_CREATE})},
                {},
            )
        )
    with pytest.raises(ValueError):
        DeterministicPolicyEngine(
            Policy(
                "bad",
                frozenset({"known"}),
                {"known": frozenset({"not-an-enum"})},
                {},
                {},
            )
        )


def test_nonpositive_pid_cannot_gain_posix_process_group_semantics():
    policy = Policy(
        "process",
        frozenset({"known"}),
        {"known": frozenset({Capability.PROCESS_CONTROL})},
        {},
        {},
    )
    adapter = HostileAdapter()
    trust = LocalComputingTrust(DeterministicPolicyEngine(policy), adapter)
    hostile = CapabilityRequest(
        "negative-pid",
        Principal("known"),
        Capability.PROCESS_CONTROL,
        ResourceTarget("process", "-1", attributes=(("signal", "15"),)),
        ExecutionContext("linux"),
        TrustState.TRUSTED,
    )
    governed = trust.govern(hostile)
    assert governed.error is LocalComputingError.INVALID_TARGET
    assert adapter.calls == 0
