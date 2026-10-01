"""Platform-neutral contracts for governing consequential local actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Mapping, Optional, Tuple

from .errors import LocalComputingError


class Capability(str, Enum):
    FILESYSTEM_READ = "FILESYSTEM_READ"
    FILE_MODIFY = "FILE_MODIFY"
    FILE_DELETE = "FILE_DELETE"
    DIRECTORY_TRAVERSE = "DIRECTORY_TRAVERSE"
    TOOL_EXECUTE = "TOOL_EXECUTE"
    COMMAND_EXECUTE = "COMMAND_EXECUTE"
    SUBPROCESS_CREATE = "SUBPROCESS_CREATE"
    PROCESS_CONTROL = "PROCESS_CONTROL"
    PROTECTED_RESOURCE_ACCESS = "PROTECTED_RESOURCE_ACCESS"
    PRIVILEGE_SENSITIVE_OPERATION = "PRIVILEGE_SENSITIVE_OPERATION"
    NETWORK_CONNECT = "NETWORK_CONNECT"
    NETWORK_DESTINATION_ACCESS = "NETWORK_DESTINATION_ACCESS"
    RESOURCE_ALLOCATE = "RESOURCE_ALLOCATE"
    HARDWARE_REQUEST = "HARDWARE_REQUEST"
    CREDENTIAL_SENSITIVE_OPERATION = "CREDENTIAL_SENSITIVE_OPERATION"


class TrustState(IntEnum):
    """Ordered authority: lower values can never confer greater authority."""

    INVALID = 0
    UNTRUSTED = 1
    DEGRADED = 2
    TRUSTED = 3

    def degrade(self) -> "TrustState":
        return TrustState(max(TrustState.UNTRUSTED, int(self) - 1))


class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    RESTRICT = "RESTRICT"


class CapabilitySupport(str, Enum):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_LIMITATIONS = "SUPPORTED_WITH_LIMITATIONS"
    OBSERVABLE_ONLY = "OBSERVABLE_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


class EnforcementStatus(str, Enum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    ENFORCED = "ENFORCED"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Principal:
    principal_id: str
    kind: str = "external"


@dataclass(frozen=True)
class ResourceTarget:
    """Classified target only; audit records do not retain sensitive payloads."""

    classification: str
    identifier: str


@dataclass(frozen=True)
class ExecutionContext:
    platform: str
    privilege_scope: str = "user"
    correlation_id: Optional[str] = None


@dataclass(frozen=True)
class CapabilityRequest:
    request_id: str
    principal: Principal
    capability: Capability
    target: ResourceTarget
    context: ExecutionContext
    trust_state: TrustState

    def validate(self) -> Optional[LocalComputingError]:
        if not self.request_id.strip() or not self.context.platform.strip():
            return LocalComputingError.MALFORMED_REQUEST
        if not self.principal.principal_id.strip():
            return LocalComputingError.UNKNOWN_PRINCIPAL
        if self.trust_state is TrustState.INVALID:
            return LocalComputingError.INVALID_TRUST_STATE
        if not self.target.classification.strip() or not self.target.identifier.strip():
            return LocalComputingError.INVALID_TARGET
        return None


@dataclass(frozen=True)
class PlatformCapability:
    capability: Capability
    support: CapabilitySupport
    limitations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyOutcome:
    decision: PolicyDecision
    policy_id: str
    constraints: Tuple[str, ...] = ()
    error: Optional[LocalComputingError] = None


@dataclass(frozen=True)
class EnforcementResult:
    status: EnforcementStatus
    attempted: bool = False
    enforced: bool = False
    completed: bool = False
    error: Optional[LocalComputingError] = None
    detail: str = ""

    @classmethod
    def not_attempted(cls, detail: str) -> "EnforcementResult":
        return cls(EnforcementStatus.NOT_ATTEMPTED, detail=detail)


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    request_id: str
    timestamp: str
    platform: str
    principal_id: str
    capability: str
    target_classification: str
    trust_state: str
    policy_id: str
    policy_decision: str
    platform_capability: str
    enforcement_status: str
    correlation_id: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def create(
        cls,
        request: CapabilityRequest,
        outcome: PolicyOutcome,
        platform: PlatformCapability,
        enforcement: EnforcementResult,
        *,
        timestamp: Optional[datetime] = None,
    ) -> "AuditEvent":
        instant = timestamp or datetime.now(timezone.utc)
        error = enforcement.error or outcome.error
        return cls(
            event_id=f"audit:{request.request_id}",
            request_id=request.request_id,
            timestamp=instant.astimezone(timezone.utc).isoformat(),
            platform=request.context.platform,
            principal_id=request.principal.principal_id,
            capability=request.capability.value,
            target_classification=request.target.classification,
            trust_state=request.trust_state.name,
            policy_id=outcome.policy_id,
            policy_decision=outcome.decision.value,
            platform_capability=platform.support.value,
            enforcement_status=enforcement.status.value,
            correlation_id=request.context.correlation_id,
            error=error.value if error else None,
        )


@dataclass(frozen=True)
class GovernanceResult:
    request_id: str
    policy: PolicyOutcome
    platform_capability: PlatformCapability
    enforcement: EnforcementResult
    audit: Optional[AuditEvent]
    audited: bool
    error: Optional[LocalComputingError] = None
