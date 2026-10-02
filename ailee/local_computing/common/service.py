"""Public orchestration boundary for Local Computing integrations."""

from __future__ import annotations

from datetime import datetime
import threading
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
    Principal,
    ResourceTarget,
    ExecutionContext,
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

    try:
        if sys.platform.startswith("linux"):
            from ..linux import LinuxPlatformAdapter

            return LinuxPlatformAdapter()
        if sys.platform == "win32":
            from ..windows import WindowsPlatformAdapter

            return WindowsPlatformAdapter()
        if sys.platform == "darwin":
            from ..macos import MacOSPlatformAdapter

            return MacOSPlatformAdapter()
    except Exception:
        # Import and construction failures are capability failures, never a
        # reason to select a foreign adapter or let an action proceed.
        return UnavailablePlatformAdapter()
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
        self._request_ids: set[str] = set()
        self._request_lock = threading.Lock()

    def govern(self, request: CapabilityRequest) -> GovernanceResult:
        if type(request) is not CapabilityRequest:
            raise TypeError("request must be an exact CapabilityRequest")
        validation_error = request.validate()
        if validation_error is not None:
            capability = PlatformCapability(
                (
                    request.capability
                    if type(request.capability) is Capability
                    else Capability.FILESYSTEM_READ
                ),
                CapabilitySupport.UNAVAILABLE,
                ("request validation failed before platform discovery",),
            )
            outcome = PolicyOutcome(
                PolicyDecision.DENY,
                "local-computing/request-validation-failure",
                error=validation_error,
            )
            return self._finish(
                request,
                outcome,
                capability,
                EnforcementResult.not_attempted("request validation failed"),
            )

        # Frozen dataclasses can still be maliciously changed with low-level
        # object APIs. Reconstruct the complete request so policy, adapter and
        # audit all observe one value even if the caller retains aliases.
        request = CapabilityRequest(
            request_id=request.request_id,
            principal=Principal(request.principal.principal_id, request.principal.kind),
            capability=request.capability,
            target=ResourceTarget(
                request.target.classification,
                request.target.identifier,
                tuple(request.target.arguments),
                tuple((key, value) for key, value in request.target.attributes),
            ),
            context=ExecutionContext(
                request.context.platform,
                request.context.privilege_scope,
                request.context.correlation_id,
            ),
            trust_state=request.trust_state,
        )

        # Reserve before policy evaluation so concurrent duplicate IDs cannot
        # both authorize and execute. IDs are deliberately single-use.
        with self._request_lock:
            replayed = request.request_id in self._request_ids
            self._request_ids.add(request.request_id)
        if replayed:
            capability = PlatformCapability(
                request.capability,
                CapabilitySupport.UNAVAILABLE,
                ("request identifier has already been consumed",),
            )
            outcome = PolicyOutcome(
                PolicyDecision.DENY,
                "local-computing/replay-denial",
                error=LocalComputingError.REPLAYED_REQUEST,
            )
            return self._finish(
                request,
                outcome,
                capability,
                EnforcementResult.not_attempted("duplicate request identifier"),
            )

        try:
            capability = self._platform.capability(request.capability)
            if (
                not isinstance(capability, PlatformCapability)
                or type(capability) is not PlatformCapability
                or type(capability.capability) is not Capability
                or type(capability.support) is not CapabilitySupport
                or capability.capability is not request.capability
                or type(capability.limitations) is not tuple
                or any(
                    type(item) is not str or len(item) > 1024
                    for item in capability.limitations
                )
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
        return self._finish(request, outcome, capability, enforcement)

    def _finish(self, request, outcome, capability, enforcement) -> GovernanceResult:
        try:
            audit = AuditEvent.create(
                request,
                outcome,
                capability,
                enforcement,
                timestamp=self._clock() if self._clock else None,
            )
        except Exception:
            return GovernanceResult(
                request.request_id if type(request.request_id) is str else "",
                outcome,
                capability,
                enforcement,
                None,
                False,
                LocalComputingError.AUDIT_FAILURE,
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
        if type(result) is not EnforcementResult:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                attempted=True,
                error=LocalComputingError.UNEXPECTED_RESULT,
                detail="platform adapter returned an invalid result type",
            )

        fields_are_valid = (
            type(result.status) is EnforcementStatus
            and type(result.attempted) is bool
            and type(result.enforced) is bool
            and type(result.completed) is bool
            and (result.error is None or type(result.error) is LocalComputingError)
            and type(result.detail) is str
            and len(result.detail) <= 4096
        )
        valid = fields_are_valid and {
            EnforcementStatus.NOT_ATTEMPTED: (
                not result.attempted and not result.enforced and not result.completed
            ),
            EnforcementStatus.ENFORCED: (
                result.attempted
                and result.enforced
                and not result.completed
                and result.error is None
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
                not result.enforced
                and not result.completed
                and result.error is not None
            ),
        }.get(result.status, False)
        if valid:
            return result
        return EnforcementResult(
            EnforcementStatus.FAILED,
            attempted=bool(result.attempted),
            error=LocalComputingError.UNEXPECTED_RESULT,
            detail="platform adapter returned contradictory enforcement evidence",
        )
