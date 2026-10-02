"""macOS/Darwin user-space Local Computing adapter (no kernel extension)."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
from typing import Tuple

from ..common.errors import LocalComputingError
from ..common.models import (
    Capability,
    CapabilityRequest,
    CapabilitySupport,
    EnforcementResult,
    EnforcementStatus,
    PlatformCapability,
)

_FS = {
    Capability.FILESYSTEM_READ,
    Capability.FILE_MODIFY,
    Capability.FILE_DELETE,
    Capability.DIRECTORY_TRAVERSE,
    Capability.PROTECTED_RESOURCE_ACCESS,
}
_EXEC = {
    Capability.TOOL_EXECUTE,
    Capability.COMMAND_EXECUTE,
    Capability.SUBPROCESS_CREATE,
}
_NET = {Capability.NETWORK_CONNECT, Capability.NETWORK_DESTINATION_ACCESS}


class MacOSPlatformAdapter:
    """Darwin integration operating with the hosting process's entitlements."""

    def __init__(self) -> None:
        self._available = sys.platform == "darwin"

    def identity(self) -> dict[str, object]:
        if not self._available:
            return {"platform": "macos", "available": False}
        return {
            "platform": "macos",
            "available": True,
            "pid": os.getpid(),
            "uid": os.getuid(),
            "euid": os.geteuid(),
            "gid": os.getgid(),
            "groups": tuple(os.getgroups()),
            "privileged": os.geteuid() == 0,
            "darwin": platform.release(),
        }

    def capability(self, requested: Capability) -> PlatformCapability:
        if not self._available:
            return PlatformCapability(
                requested, CapabilitySupport.UNAVAILABLE, ("requires a macOS runtime",)
            )
        if requested in _FS:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "POSIX modes, ACLs, sandbox entitlements, TCC and SIP remain authoritative",
                    "no mediation outside adapter",
                    "symlinks are not a sandbox",
                ),
            )
        if requested in _EXEC:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "posix_spawn/fork implementation selected by Python",
                    "direct argv without shell",
                    "code signing and sandbox rules remain authoritative",
                    "executable identity and descendants are not transitively bound",
                ),
            )
        if requested is Capability.PROCESS_CONTROL:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "signal delivery only",
                    "Darwin permission and protected-process checks remain authoritative",
                    "PID identity is not pinned against reuse",
                ),
            )
        if requested in _NET:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "TCP connect only",
                    "sandbox and Network Extension policy remain authoritative",
                    "resolved address is not pinned during policy evaluation",
                    "no Network Extension or traffic interception",
                ),
            )
        if requested in {
            Capability.PRIVILEGE_SENSITIVE_OPERATION,
            Capability.RESOURCE_ALLOCATE,
            Capability.HARDWARE_REQUEST,
        }:
            return PlatformCapability(
                requested,
                CapabilitySupport.OBSERVABLE_ONLY,
                (
                    "host context can be observed; entitlements or privileges cannot be granted",
                ),
            )
        return PlatformCapability(
            requested,
            CapabilitySupport.UNAVAILABLE,
            ("no supported user-space enforcement implemented",),
        )

    def enforce(
        self, request: CapabilityRequest, constraints: Tuple[str, ...]
    ) -> EnforcementResult:
        if request.context.platform.lower() not in {"macos", "darwin"}:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                error=LocalComputingError.PLATFORM_INTEGRATION_FAILURE,
                detail="request platform does not match macOS adapter",
            )
        support = self.capability(request.capability)
        if support.support in {
            CapabilitySupport.UNAVAILABLE,
            CapabilitySupport.OBSERVABLE_ONLY,
        }:
            return EnforcementResult(
                EnforcementStatus.NOT_ATTEMPTED,
                error=LocalComputingError.PLATFORM_UNAVAILABLE,
                detail="macOS capability cannot be enforced by this adapter",
            )
        if "read-only" in constraints and request.capability not in {
            Capability.FILESYSTEM_READ,
            Capability.DIRECTORY_TRAVERSE,
        }:
            return EnforcementResult.not_attempted(
                "request restriction forbids mutation"
            )
        try:
            detail = self._perform(request)
            return EnforcementResult(
                EnforcementStatus.COMPLETED, True, True, True, detail=detail
            )
        except (ValueError, TypeError):
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=LocalComputingError.INVALID_TARGET,
                detail="invalid macOS resource parameters",
            )
        except PermissionError:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=LocalComputingError.ENFORCEMENT_FAILURE,
                detail="macOS denied permission (mode, ACL, sandbox, TCC, or SIP)",
            )
        except FileNotFoundError:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=LocalComputingError.INVALID_TARGET,
                detail="macOS resource does not exist",
            )
        except subprocess.TimeoutExpired:
            return EnforcementResult(
                EnforcementStatus.PARTIAL,
                True,
                error=LocalComputingError.PARTIAL_EXECUTION,
                detail="macOS child timed out after creation",
            )
        except OSError as exc:
            kind = (
                LocalComputingError.INVALID_TARGET
                if exc.errno in {errno.EINVAL, errno.ENOENT}
                else LocalComputingError.ENFORCEMENT_FAILURE
            )
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=kind,
                detail=f"macOS operation failed with errno {exc.errno}",
            )

    def _perform(self, request: CapabilityRequest) -> str:
        cap, target, path = (
            request.capability,
            request.target,
            Path(request.target.identifier),
        )
        if cap in {Capability.FILESYSTEM_READ, Capability.PROTECTED_RESOURCE_ACCESS}:
            path.read_bytes()
            return "macOS file read completed under host entitlements"
        if cap is Capability.FILE_MODIFY:
            content = target.attribute("content")
            if content is None:
                raise ValueError("content required")
            path.write_text(content, encoding="utf-8")
            return "macOS file write completed"
        if cap is Capability.FILE_DELETE:
            path.unlink()
            return "macOS file deletion completed"
        if cap is Capability.DIRECTORY_TRAVERSE:
            tuple(item.name for item in path.iterdir())
            return "macOS directory traversal completed"
        if cap in _EXEC:
            result = subprocess.run(
                (target.identifier, *target.arguments),
                shell=False,
                check=False,
                timeout=float(target.attribute("timeout") or "30"),
                capture_output=True,
            )
            if result.returncode:
                raise OSError(errno.ECHILD, f"child exit {result.returncode}")
            return f"macOS child completed with exit code {result.returncode}"
        if cap is Capability.PROCESS_CONTROL:
            os.kill(
                int(target.identifier),
                int(target.attribute("signal") or signal.SIGTERM),
            )
            return "macOS signal delivered"
        if cap in _NET:
            with socket.create_connection(
                (target.identifier, int(target.attribute("port") or "")),
                timeout=float(target.attribute("timeout") or "5"),
            ):
                pass
            return "macOS TCP connection established and closed"
        raise OSError(errno.ENOTSUP, "unsupported macOS operation")


__all__ = ["MacOSPlatformAdapter"]
