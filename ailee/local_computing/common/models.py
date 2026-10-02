"""Platform-neutral contracts for governing consequential local actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Optional, Tuple

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
    arguments: Tuple[str, ...] = ()
    attributes: Tuple[Tuple[str, str], ...] = ()

    def attribute(self, name: str) -> Optional[str]:
        """Return a request-local native option without copying it to audit."""
        return next((value for key, value in self.attributes if key == name), None)


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
        if (
            type(self.principal) is not Principal
            or type(self.target) is not ResourceTarget
            or type(self.context) is not ExecutionContext
            or type(self.capability) is not Capability
            or type(self.trust_state) is not TrustState
        ):
            return LocalComputingError.MALFORMED_REQUEST
        text_fields = (
            self.request_id,
            self.principal.principal_id,
            self.principal.kind,
            self.context.platform,
            self.context.privilege_scope,
            self.target.classification,
            self.target.identifier,
        )
        if any(type(value) is not str for value in text_fields):
            return LocalComputingError.MALFORMED_REQUEST
        if (
            self.context.correlation_id is not None
            and type(self.context.correlation_id) is not str
        ):
            return LocalComputingError.MALFORMED_REQUEST
        if not self.request_id.strip() or not self.context.platform.strip():
            return LocalComputingError.MALFORMED_REQUEST
        if not self.principal.principal_id.strip():
            return LocalComputingError.UNKNOWN_PRINCIPAL
        if self.trust_state is TrustState.INVALID:
            return LocalComputingError.INVALID_TRUST_STATE
        if not self.target.classification.strip() or not self.target.identifier.strip():
            return LocalComputingError.INVALID_TARGET
        if _has_forbidden_control(self.target.identifier):
            return LocalComputingError.INVALID_TARGET
        if (
            type(self.target.arguments) is not tuple
            or type(self.target.attributes) is not tuple
        ):
            return LocalComputingError.INVALID_TARGET
        if len(self.request_id) > 256 or len(self.target.identifier) > 32768:
            return LocalComputingError.MALFORMED_REQUEST
        if (
            any(
                type(argument) is not str
                or len(argument) > 32768
                or _has_forbidden_control(argument)
                for argument in self.target.arguments
            )
            or len(self.target.arguments) > 256
        ):
            return LocalComputingError.INVALID_TARGET
        if any(
            type(item) is not tuple
            or len(item) != 2
            or any(type(value) is not str for value in item)
            for item in self.target.attributes
        ):
            return LocalComputingError.INVALID_TARGET
        keys = tuple(key for key, _ in self.target.attributes)
        if len(keys) != len(set(keys)) or any(
            not key.strip() or _has_forbidden_control(key) for key in keys
        ):
            return LocalComputingError.INVALID_TARGET
        allowed_attributes = {
            Capability.FILE_MODIFY: {"content"},
            Capability.TOOL_EXECUTE: {"timeout"},
            Capability.COMMAND_EXECUTE: {"timeout"},
            Capability.SUBPROCESS_CREATE: {"timeout"},
            Capability.PROCESS_CONTROL: {"signal", "exit_code"},
            Capability.NETWORK_CONNECT: {"port", "timeout"},
            Capability.NETWORK_DESTINATION_ACCESS: {"port", "timeout"},
        }.get(self.capability, set())
        if set(keys) - allowed_attributes:
            return LocalComputingError.INVALID_TARGET
        for key, value in self.target.attributes:
            if key != "content" and (
                len(value) > 1024 or _has_forbidden_control(value)
            ):
                return LocalComputingError.INVALID_TARGET
        if (
            self.target.attribute("content") is not None
            and len(self.target.attribute("content") or "") > 1_048_576
        ):
            return LocalComputingError.INVALID_TARGET
        if self.capability in {
            Capability.NETWORK_CONNECT,
            Capability.NETWORK_DESTINATION_ACCESS,
        }:
            try:
                port = int(self.target.attribute("port") or "")
            except ValueError:
                return LocalComputingError.INVALID_TARGET
            if not 1 <= port <= 65535 or any(
                ch.isspace() for ch in self.target.identifier
            ):
                return LocalComputingError.INVALID_TARGET
        timeout = self.target.attribute("timeout")
        if timeout is not None:
            try:
                numeric_timeout = float(timeout)
            except ValueError:
                return LocalComputingError.INVALID_TARGET
            if not 0 < numeric_timeout <= 300 or numeric_timeout != numeric_timeout:
                return LocalComputingError.INVALID_TARGET
        if self.capability is Capability.PROCESS_CONTROL:
            try:
                pid = int(self.target.identifier)
            except ValueError:
                return LocalComputingError.INVALID_TARGET
            if pid <= 0:
                # POSIX gives zero and negative PIDs process-group semantics.
                return LocalComputingError.INVALID_TARGET
            for name in ("signal", "exit_code"):
                value = self.target.attribute(name)
                if value is not None:
                    try:
                        numeric = int(value)
                    except ValueError:
                        return LocalComputingError.INVALID_TARGET
                    if numeric < 0 or numeric > 0xFFFFFFFF:
                        return LocalComputingError.INVALID_TARGET
        return None


def _has_forbidden_control(value: str) -> bool:
    """Reject NUL/C0 controls at native string boundaries."""
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


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
    platform_limitations: Tuple[str, ...] = ()
    enforcement_detail: str = ""
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
            platform_limitations=platform.limitations,
            enforcement_detail=enforcement.detail,
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
