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

MAX_POLICY_CONSTRAINTS = 256
MAX_POLICY_CONSTRAINT_LENGTH = 1024


def valid_policy_constraints(values: object, *, canonical_count: bool = False) -> bool:
    """Validate the constraint contract shared by configuration and outcomes.

    Policy configuration is counted after deterministic deduplication, while an
    emitted outcome is already expected to be within the bounded tuple size.
    """
    if type(values) is not tuple:
        return False
    if any(
        type(value) is not str
        or not value.strip()
        or len(value) > MAX_POLICY_CONSTRAINT_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        for value in values
    ):
        return False
    count = len(set(values)) if canonical_count else len(values)
    return count <= MAX_POLICY_CONSTRAINTS


@dataclass(frozen=True)
class Policy:
    policy_id: str
    known_principals: FrozenSet[str]
    allowed: Mapping[str, FrozenSet[Capability]]
    restricted: Mapping[str, FrozenSet[Capability]]
    restrictions: Mapping[Capability, Tuple[str, ...]]

    def validate(self) -> None:
        if (
            type(self.policy_id) is not str
            or not self.policy_id.strip()
            or len(self.policy_id) > 256
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.policy_id
            )
            or not self.known_principals
        ):
            raise ValueError("policy_id and known_principals are required")
        if any(
            type(principal) is not str
            or not principal.strip()
            or len(principal) > 256
            or any(
                ord(character) < 32 or ord(character) == 127 for character in principal
            )
            for principal in self.known_principals
        ):
            raise ValueError("principal identifiers cannot be empty")
        unknown = (set(self.allowed) | set(self.restricted)) - set(
            self.known_principals
        )
        if unknown:
            raise ValueError("policy rules reference unknown principals")
        for rules in (self.allowed, self.restricted):
            if any(
                type(principal) is not str
                or any(
                    type(capability) is not Capability for capability in capabilities
                )
                for principal, capabilities in rules.items()
            ):
                raise ValueError("policy rules contain malformed capabilities")
        conflicts = {
            principal
            for principal in self.known_principals
            if set(self.allowed.get(principal, ()))
            & set(self.restricted.get(principal, ()))
        }
        if conflicts:
            raise ValueError("a capability cannot be both allowed and restricted")
        if any(
            type(capability) is not Capability
            or not valid_policy_constraints(values, canonical_count=True)
            for capability, values in self.restrictions.items()
        ):
            raise ValueError("policy restrictions are malformed")

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
            {
                key: tuple(sorted(set(value)))
                for key, value in policy.restrictions.items()
            },
        )

    @property
    def policy_id(self) -> str:
        return self._policy.policy_id

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
