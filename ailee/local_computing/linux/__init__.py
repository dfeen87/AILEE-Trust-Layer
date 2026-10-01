"""Linux user-space Local Computing adapter.

This adapter governs only actions routed through it.  It uses documented POSIX/
Linux process, filesystem, signal, and socket facilities; it is not a sandbox.
"""

from __future__ import annotations

import errno
import os
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


class LinuxPlatformAdapter:
    """Explicit Linux integration using current-process authority."""

    def __init__(self) -> None:
        self._available = sys.platform.startswith("linux")

    def identity(self) -> dict[str, object]:
        if not self._available:
            return {"platform": "linux", "available": False}
        groups = tuple(os.getgroups()) if hasattr(os, "getgroups") else ()
        return {
            "platform": "linux",
            "available": True,
            "pid": os.getpid(),
            "uid": os.getuid(),
            "euid": os.geteuid(),
            "gid": os.getgid(),
            "egid": os.getegid(),
            "groups": groups,
            "privileged": os.geteuid() == 0,
            "kernel": platform.release(),
        }

    def capability(self, requested: Capability) -> PlatformCapability:
        if not self._available:
            return PlatformCapability(
                requested, CapabilitySupport.UNAVAILABLE, ("requires a Linux runtime",)
            )
        if requested in _FS:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "uses caller credentials",
                    "no mediation outside adapter",
                    "path checks are not a sandbox",
                ),
            )
        if requested in _EXEC:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "direct argv execution without a shell",
                    "child inherits caller OS security context",
                ),
            )
        if requested in _NET:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "TCP connect only",
                    "DNS and routing remain OS-authoritative",
                    "no traffic interception",
                ),
            )
        if requested is Capability.PROCESS_CONTROL:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "signal delivery only",
                    "kernel permission checks remain authoritative",
                ),
            )
        if requested in {Capability.RESOURCE_ALLOCATE, Capability.HARDWARE_REQUEST}:
            return PlatformCapability(
                requested,
                CapabilitySupport.OBSERVABLE_ONLY,
                (
                    "current-process resource context can be observed; allocation is not mediated",
                ),
            )
        return PlatformCapability(
            requested,
            CapabilitySupport.UNAVAILABLE,
            ("no safe user-space mechanism implemented",),
        )

    def enforce(
        self, request: CapabilityRequest, constraints: Tuple[str, ...]
    ) -> EnforcementResult:
        if request.context.platform.lower() not in {"linux", "linux2"}:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                error=LocalComputingError.PLATFORM_INTEGRATION_FAILURE,
                detail="request platform does not match Linux adapter",
            )
        support = self.capability(request.capability)
        if support.support in {
            CapabilitySupport.UNAVAILABLE,
            CapabilitySupport.OBSERVABLE_ONLY,
        }:
            return EnforcementResult(
                EnforcementStatus.NOT_ATTEMPTED,
                error=LocalComputingError.PLATFORM_UNAVAILABLE,
                detail="Linux capability cannot be enforced by this adapter",
            )
        if "read-only" in constraints and request.capability not in {
            Capability.FILESYSTEM_READ,
            Capability.DIRECTORY_TRAVERSE,
        }:
            return EnforcementResult(
                EnforcementStatus.NOT_ATTEMPTED,
                detail="request restriction forbids mutation",
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
                detail="invalid Linux resource parameters",
            )
        except PermissionError:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=LocalComputingError.ENFORCEMENT_FAILURE,
                detail="Linux denied permission",
            )
        except FileNotFoundError:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=LocalComputingError.INVALID_TARGET,
                detail="Linux resource does not exist",
            )
        except subprocess.TimeoutExpired:
            return EnforcementResult(
                EnforcementStatus.PARTIAL,
                True,
                error=LocalComputingError.PARTIAL_EXECUTION,
                detail="child timed out after creation",
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
                detail=f"Linux operation failed with errno {exc.errno}",
            )

    def _perform(self, request: CapabilityRequest) -> str:
        cap, target = request.capability, request.target
        if cap is Capability.FILESYSTEM_READ:
            with open(target.identifier, "rb") as stream:
                stream.read()
            return "Linux file read completed"
        if cap is Capability.FILE_MODIFY:
            data = target.attribute("content")
            if data is None:
                raise ValueError("content required")
            with open(target.identifier, "w", encoding="utf-8") as stream:
                stream.write(data)
            return "Linux file write completed"
        if cap is Capability.FILE_DELETE:
            os.remove(target.identifier)
            return "Linux file deletion completed"
        if cap is Capability.DIRECTORY_TRAVERSE:
            with os.scandir(target.identifier) as entries:
                tuple(entry.name for entry in entries)
            return "Linux directory traversal completed"
        if cap is Capability.PROTECTED_RESOURCE_ACCESS:
            with open(target.identifier, "rb") as stream:
                stream.read(1)
            return "Linux protected-resource read completed under caller credentials"
        if cap in _EXEC:
            timeout = float(target.attribute("timeout") or "30")
            completed = subprocess.run(
                (target.identifier, *target.arguments),
                shell=False,
                check=False,
                timeout=timeout,
                capture_output=True,
            )
            if completed.returncode != 0:
                raise OSError(errno.ECHILD, f"child exit {completed.returncode}")
            return f"Linux child completed with exit code {completed.returncode}"
        if cap is Capability.PROCESS_CONTROL:
            pid = int(target.identifier)
            signum = int(target.attribute("signal") or signal.SIGTERM)
            os.kill(pid, signum)
            return "Linux signal delivered"
        if cap in _NET:
            port = int(target.attribute("port") or "")
            timeout = float(target.attribute("timeout") or "5")
            with socket.create_connection((target.identifier, port), timeout=timeout):
                pass
            return "Linux TCP connection established and closed"
        raise OSError(errno.ENOTSUP, "unsupported Linux operation")


__all__ = ["LinuxPlatformAdapter"]
