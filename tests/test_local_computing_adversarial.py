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
    PolicyOutcome,
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


def test_request_id_retention_is_bounded_without_evicting_replay_evidence():
    adapter = HostileAdapter()
    policy = Policy(
        "bounded-replay",
        frozenset({"known"}),
        {"known": frozenset({Capability.SUBPROCESS_CREATE})},
        {},
        {},
    )
    trust = LocalComputingTrust(
        DeterministicPolicyEngine(policy), adapter, max_request_ids=1
    )

    first = trust.govern(request(request_id="first"))
    exhausted = trust.govern(request(request_id="second"))
    replay = trust.govern(request(request_id="first"))

    assert first.enforcement.completed
    assert exhausted.error is LocalComputingError.REQUEST_ID_CAPACITY_EXCEEDED
    assert exhausted.enforcement.attempted is False
    assert replay.error is LocalComputingError.REPLAYED_REQUEST
    assert adapter.calls == 1


@pytest.mark.parametrize("max_request_ids", [0, -1, True, 1.5])
def test_request_id_capacity_requires_a_positive_exact_integer(max_request_ids):
    with pytest.raises(ValueError):
        LocalComputingTrust(
            DeterministicPolicyEngine(
                Policy("capacity", frozenset({"known"}), {}, {}, {})
            ),
            HostileAdapter(),
            max_request_ids=max_request_ids,
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


def test_security_metadata_rejects_controls_but_posix_argv_is_not_over_rejected():
    adapter = HostileAdapter()
    controlled_id = service(adapter).govern(request(request_id="line\nbreak"))
    assert controlled_id.error is LocalComputingError.MALFORMED_REQUEST
    assert adapter.calls == 0

    legitimate_argument = service(adapter).govern(
        request(target=ResourceTarget("tool", "/bin/true", ("line\nbreak",)))
    )
    assert legitimate_argument.enforcement.completed
    assert adapter.observed[-1].target.arguments == ("line\nbreak",)


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


def test_policy_failure_and_malformed_policy_evidence_fail_closed_and_stay_reserved():
    adapter = HostileAdapter()
    trust = service(adapter)
    trust._policy_engine.evaluate = lambda request, platform: (_ for _ in ()).throw(
        RuntimeError("policy unavailable")
    )

    failed = trust.govern(request(request_id="policy-failure"))
    replay = trust.govern(request(request_id="policy-failure"))

    assert failed.policy.decision is PolicyDecision.DENY
    assert failed.error is LocalComputingError.POLICY_EVALUATION_FAILURE
    assert failed.enforcement.attempted is False
    assert replay.error is LocalComputingError.REPLAYED_REQUEST
    assert adapter.calls == 0


def test_policy_constraints_are_canonicalized_for_equivalent_configuration():
    configured = Policy(
        "canonical",
        frozenset({"known"}),
        {},
        {"known": frozenset({Capability.SUBPROCESS_CREATE})},
        {Capability.SUBPROCESS_CREATE: ("z-limit", "a-limit", "z-limit")},
    )
    outcome = DeterministicPolicyEngine(configured).evaluate(
        request(),
        PlatformCapability(Capability.SUBPROCESS_CREATE, CapabilitySupport.SUPPORTED),
    )
    assert outcome.constraints == ("a-limit", "z-limit")


def restricted_policy(constraints):
    return Policy(
        "constraint-contract",
        frozenset({"known"}),
        {},
        {"known": frozenset({Capability.SUBPROCESS_CREATE})},
        {Capability.SUBPROCESS_CREATE: constraints},
    )


def test_policy_accepts_exactly_256_unique_constraints_and_matching_outcome():
    constraints = tuple(f"constraint-{index:03d}" for index in range(256))
    engine = DeterministicPolicyEngine(restricted_policy(constraints))
    trust = LocalComputingTrust(engine, HostileAdapter())

    governed = trust.govern(request(request_id="valid-constraint-boundary"))

    assert governed.policy.decision is PolicyDecision.RESTRICT
    assert governed.policy.constraints == constraints
    assert governed.error is None
    assert governed.enforcement.completed


def test_policy_rejects_257_unique_constraints_during_construction():
    constraints = tuple(f"constraint-{index:03d}" for index in range(257))
    with pytest.raises(ValueError, match="restrictions are malformed"):
        DeterministicPolicyEngine(restricted_policy(constraints))


def test_policy_constraint_length_boundary_is_validated_during_construction():
    DeterministicPolicyEngine(restricted_policy(("x" * 1024,)))
    with pytest.raises(ValueError, match="restrictions are malformed"):
        DeterministicPolicyEngine(restricted_policy(("x" * 1025,)))


@pytest.mark.parametrize("control", ["line\nbreak", "delete\x7fcharacter"])
def test_policy_rejects_constraint_controls_during_construction(control):
    with pytest.raises(ValueError, match="restrictions are malformed"):
        DeterministicPolicyEngine(restricted_policy((control,)))


def test_policy_counts_canonical_unique_constraints_and_sorts_them():
    raw = tuple(reversed(tuple(f"constraint-{index:03d}" for index in range(200))))
    engine = DeterministicPolicyEngine(restricted_policy(raw + raw[:100]))
    outcome = engine.evaluate(
        request(),
        PlatformCapability(Capability.SUBPROCESS_CREATE, CapabilitySupport.SUPPORTED),
    )
    assert outcome.constraints == tuple(sorted(set(raw)))
    assert len(outcome.constraints) == 200


@pytest.mark.parametrize(
    "constraints",
    [("x" * 1025,), ("control\ncharacter",), tuple(str(i) for i in range(257))],
)
def test_nonconforming_policy_outcome_still_fails_closed(constraints):
    adapter = HostileAdapter()
    trust = service(adapter)
    trust._policy_engine.evaluate = lambda request, platform: PolicyOutcome(
        PolicyDecision.ALLOW, trust._policy_engine.policy_id, constraints
    )

    governed = trust.govern(request(request_id=f"hostile-{len(constraints)}"))

    assert governed.policy.decision is PolicyDecision.DENY
    assert governed.error is LocalComputingError.POLICY_EVALUATION_FAILURE
    assert not governed.enforcement.attempted
    assert adapter.calls == 0


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
