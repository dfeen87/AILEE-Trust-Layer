"""Deterministic, fail-closed Local Computing policy foundation."""

from dataclasses import dataclass
from typing import FrozenSet, Mapping, Tuple

from .errors import LocalComputingError
from .models import (
    Capability,
    CapabilityRequest,
    CapabilitySupport,
    PlatformCapability,
    PolicyDecision,
    PolicyOutcome,
    TrustState,
)


@dataclass(frozen=True)
class Policy:
    policy_id: str
    known_principals: FrozenSet[str]
    allowed: Mapping[str, FrozenSet[Capability]]
    restricted: Mapping[str, FrozenSet[Capability]]
    restrictions: Mapping[Capability, Tuple[str, ...]]

    def validate(self) -> None:
        if not self.policy_id.strip() or not self.known_principals:
            raise ValueError("policy_id and known_principals are required")
        if any(not principal.strip() for principal in self.known_principals):
            raise ValueError("principal identifiers cannot be empty")
        unknown = (set(self.allowed) | set(self.restricted)) - set(
            self.known_principals
        )
        if unknown:
            raise ValueError("policy rules reference unknown principals")

    @classmethod
    def deny_by_default(cls, known_principals: FrozenSet[str]) -> "Policy":
        return cls("local-computing/default-deny/v1", known_principals, {}, {}, {})


class DeterministicPolicyEngine:
    def __init__(self, policy: Policy):
        policy.validate()
        # Snapshot caller-owned mappings so later mutation cannot change decisions.
        self._policy = Policy(
            policy.policy_id,
            frozenset(policy.known_principals),
            {key: frozenset(value) for key, value in policy.allowed.items()},
            {key: frozenset(value) for key, value in policy.restricted.items()},
            {key: tuple(value) for key, value in policy.restrictions.items()},
        )

    def evaluate(
        self, request: CapabilityRequest, platform: PlatformCapability
    ) -> PolicyOutcome:
        error = request.validate()
        if error:
            return PolicyOutcome(
                PolicyDecision.DENY, self._policy.policy_id, error=error
            )
        principal_id = request.principal.principal_id
        if principal_id not in self._policy.known_principals:
            return PolicyOutcome(
                PolicyDecision.DENY,
                self._policy.policy_id,
                error=LocalComputingError.UNKNOWN_PRINCIPAL,
            )
        if platform.capability is not request.capability:
            return PolicyOutcome(
                PolicyDecision.DENY,
                self._policy.policy_id,
                error=LocalComputingError.UNSUPPORTED_CAPABILITY,
            )
        if request.trust_state <= TrustState.UNTRUSTED:
            return PolicyOutcome(PolicyDecision.DENY, self._policy.policy_id)
        allowed = request.capability in self._policy.allowed.get(
            principal_id, frozenset()
        )
        restricted = request.capability in self._policy.restricted.get(
            principal_id, frozenset()
        )
        if not allowed and not restricted:
            return PolicyOutcome(PolicyDecision.DENY, self._policy.policy_id)
        if platform.support is CapabilitySupport.UNAVAILABLE:
            return PolicyOutcome(
                PolicyDecision.DENY,
                self._policy.policy_id,
                error=LocalComputingError.PLATFORM_UNAVAILABLE,
            )
        constraints = self._policy.restrictions.get(request.capability, ())
        if (
            restricted
            or request.trust_state is TrustState.DEGRADED
            or platform.support
            in {
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                CapabilitySupport.OBSERVABLE_ONLY,
            }
        ):
            return PolicyOutcome(
                PolicyDecision.RESTRICT, self._policy.policy_id, constraints
            )
        return PolicyOutcome(PolicyDecision.ALLOW, self._policy.policy_id)
