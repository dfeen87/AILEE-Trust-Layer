"""Public orchestration boundary for Local Computing integrations."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Protocol

from .errors import LocalComputingError
from .models import (
    AuditEvent,
    Capability,
    CapabilityRequest,
    CapabilitySupport,
    EnforcementResult,
    EnforcementStatus,
    GovernanceResult,
    PlatformCapability,
    PolicyDecision,
    PolicyOutcome,
)
from .policy import DeterministicPolicyEngine


class PlatformAdapter(Protocol):
    """Prompt 1.2 integration seam; implementations must report real capability."""

    def capability(self, requested: Capability) -> PlatformCapability: ...

    def enforce(
        self, request: CapabilityRequest, constraints: tuple[str, ...]
    ) -> EnforcementResult: ...


class UnavailablePlatformAdapter:
    """Honest default: no native platform enforcement is implemented yet."""

    def capability(self, requested: Capability) -> PlatformCapability:
        return PlatformCapability(requested, CapabilitySupport.UNAVAILABLE)

    def enforce(
        self, request: CapabilityRequest, constraints: tuple[str, ...]
    ) -> EnforcementResult:
        return EnforcementResult.not_attempted("native platform adapter is unavailable")


def native_platform_adapter() -> PlatformAdapter:
    """Select only the adapter for the running OS; unknown OSes fail closed."""
    import sys

    if sys.platform.startswith("linux"):
        from ..linux import LinuxPlatformAdapter

        return LinuxPlatformAdapter()
    if sys.platform == "win32":
        from ..windows import WindowsPlatformAdapter

        return WindowsPlatformAdapter()
    if sys.platform == "darwin":
        from ..macos import MacOSPlatformAdapter

        return MacOSPlatformAdapter()
    return UnavailablePlatformAdapter()


class LocalComputingTrust:
    """Standalone public API governing requests submitted through this boundary."""

    def __init__(
        self,
        policy_engine: DeterministicPolicyEngine,
        platform: PlatformAdapter | None = None,
        audit_sink: Callable[[AuditEvent], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self._policy_engine = policy_engine
        self._platform = platform or UnavailablePlatformAdapter()
        self._audit_sink = audit_sink
        self._clock = clock

    def govern(self, request: CapabilityRequest) -> GovernanceResult:
        try:
            capability = self._platform.capability(request.capability)
            if (
                not isinstance(capability, PlatformCapability)
                or not isinstance(capability.capability, Capability)
                or not isinstance(capability.support, CapabilitySupport)
            ):
                raise TypeError("invalid platform capability result")
        except Exception:
            capability = PlatformCapability(
                request.capability, CapabilitySupport.UNAVAILABLE
            )
            outcome = PolicyOutcome(
                PolicyDecision.DENY,
                "local-computing/platform-integration-failure",
                error=LocalComputingError.PLATFORM_INTEGRATION_FAILURE,
            )
        else:
            outcome = self._policy_engine.evaluate(request, capability)
        if outcome.decision is PolicyDecision.DENY:
            enforcement = EnforcementResult.not_attempted("policy denied the request")
        elif capability.support is CapabilitySupport.OBSERVABLE_ONLY:
            enforcement = EnforcementResult.not_attempted(
                "platform capability is observable only"
            )
        else:
            try:
                enforcement = self._platform.enforce(request, outcome.constraints)
            except Exception:
                enforcement = EnforcementResult(
                    status=EnforcementStatus.FAILED,
                    attempted=True,
                    error=LocalComputingError.ENFORCEMENT_FAILURE,
                    detail="platform adapter raised during enforcement",
                )
            else:
                enforcement = self._validated_enforcement(enforcement)
        audit = AuditEvent.create(
            request,
            outcome,
            capability,
            enforcement,
            timestamp=self._clock() if self._clock else None,
        )
        audited = True
        error = enforcement.error or outcome.error
        if self._audit_sink:
            try:
                self._audit_sink(audit)
            except Exception:
                audited = False
                error = LocalComputingError.AUDIT_FAILURE
        return GovernanceResult(
            request.request_id,
            outcome,
            capability,
            enforcement,
            audit,
            audited,
            error,
        )

    @staticmethod
    def _validated_enforcement(result: object) -> EnforcementResult:
        """Fail closed when an adapter returns a contradictory result.

        Platform adapters are an integration boundary, so their return values
        cannot be trusted merely because policy evaluation authorized an
        attempt.  In particular, completion is evidence only when all three
        execution flags and the status agree and no error is present.
        """
        if not isinstance(result, EnforcementResult):
            return EnforcementResult(
                EnforcementStatus.FAILED,
                attempted=True,
                error=LocalComputingError.UNEXPECTED_RESULT,
                detail="platform adapter returned an invalid result type",
            )

        valid = {
            EnforcementStatus.NOT_ATTEMPTED: (
                not result.attempted and not result.enforced and not result.completed
            ),
            EnforcementStatus.ENFORCED: (
                result.attempted and result.enforced and not result.completed
            ),
            EnforcementStatus.COMPLETED: (
                result.attempted
                and result.enforced
                and result.completed
                and result.error is None
            ),
            EnforcementStatus.PARTIAL: (
                result.attempted
                and not result.completed
                and result.error is LocalComputingError.PARTIAL_EXECUTION
            ),
            EnforcementStatus.FAILED: (
                not result.completed and result.error is not None
            ),
        }
        if valid.get(result.status, False):
            return result
        return EnforcementResult(
            EnforcementStatus.FAILED,
            attempted=bool(result.attempted),
            error=LocalComputingError.UNEXPECTED_RESULT,
            detail="platform adapter returned contradictory enforcement evidence",
        )
