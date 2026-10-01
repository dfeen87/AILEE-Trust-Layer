"""Windows-native user-space Local Computing adapter (no driver or POSIX emulation)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
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


class WindowsPlatformAdapter:
    """Use Win32 handles/tokens plus standard Windows filesystem and Winsock APIs."""

    def __init__(self) -> None:
        self._available = sys.platform == "win32"

    def identity(self) -> dict[str, object]:
        if not self._available:
            return {"platform": "windows", "available": False}
        token = wintypes.HANDLE()
        TOKEN_QUERY = 0x0008
        opened = bool(
            ctypes.windll.advapi32.OpenProcessToken(
                ctypes.windll.kernel32.GetCurrentProcess(),
                TOKEN_QUERY,
                ctypes.byref(token),
            )
        )
        if opened:
            ctypes.windll.kernel32.CloseHandle(token)
        return {
            "platform": "windows",
            "available": True,
            "pid": os.getpid(),
            "token_queryable": opened,
            "username": os.environ.get("USERNAME", ""),
        }

    def capability(self, requested: Capability) -> PlatformCapability:
        if not self._available:
            return PlatformCapability(
                requested,
                CapabilitySupport.UNAVAILABLE,
                ("requires a Windows runtime",),
            )
        if requested in _FS:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "Windows ACL and sharing checks remain authoritative",
                    "reparse points are not sandboxed",
                    "no mediation outside adapter",
                ),
            )
        if requested in _EXEC:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "CreateProcess semantics via direct argument vector",
                    "child inherits the current security token",
                    "not a Job Object sandbox",
                ),
            )
        if requested is Capability.PROCESS_CONTROL:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "TerminateProcess only",
                    "target ACL and protected-process rules remain authoritative",
                ),
            )
        if requested in _NET:
            return PlatformCapability(
                requested,
                CapabilitySupport.SUPPORTED_WITH_LIMITATIONS,
                (
                    "TCP connect through Winsock only",
                    "Windows Firewall remains authoritative",
                    "no traffic interception",
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
                    "security/resource context is observable but this adapter does not grant or confine it",
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
        if request.context.platform.lower() not in {"windows", "win32"}:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                error=LocalComputingError.PLATFORM_INTEGRATION_FAILURE,
                detail="request platform does not match Windows adapter",
            )
        support = self.capability(request.capability)
        if support.support in {
            CapabilitySupport.UNAVAILABLE,
            CapabilitySupport.OBSERVABLE_ONLY,
        }:
            return EnforcementResult(
                EnforcementStatus.NOT_ATTEMPTED,
                error=LocalComputingError.PLATFORM_UNAVAILABLE,
                detail="Windows capability cannot be enforced by this adapter",
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
                detail="invalid Windows resource parameters",
            )
        except PermissionError:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=LocalComputingError.ENFORCEMENT_FAILURE,
                detail="Windows access check denied the operation",
            )
        except FileNotFoundError:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=LocalComputingError.INVALID_TARGET,
                detail="Windows resource does not exist",
            )
        except subprocess.TimeoutExpired:
            return EnforcementResult(
                EnforcementStatus.PARTIAL,
                True,
                error=LocalComputingError.PARTIAL_EXECUTION,
                detail="Windows child timed out after creation",
            )
        except OSError as exc:
            return EnforcementResult(
                EnforcementStatus.FAILED,
                True,
                error=LocalComputingError.ENFORCEMENT_FAILURE,
                detail=f"Windows operation failed with code {getattr(exc, 'winerror', None) or exc.errno}",
            )

    def _perform(self, request: CapabilityRequest) -> str:
        cap, target, path = (
            request.capability,
            request.target,
            Path(request.target.identifier),
        )
        if cap in {Capability.FILESYSTEM_READ, Capability.PROTECTED_RESOURCE_ACCESS}:
            path.read_bytes()
            return "Windows file read completed under current token"
        if cap is Capability.FILE_MODIFY:
            content = target.attribute("content")
            if content is None:
                raise ValueError("content required")
            path.write_text(content, encoding="utf-8")
            return "Windows file write completed"
        if cap is Capability.FILE_DELETE:
            path.unlink()
            return "Windows file deletion completed"
        if cap is Capability.DIRECTORY_TRAVERSE:
            tuple(item.name for item in path.iterdir())
            return "Windows directory traversal completed"
        if cap in _EXEC:
            timeout = float(target.attribute("timeout") or "30")
            result = subprocess.run(
                (target.identifier, *target.arguments),
                shell=False,
                check=False,
                timeout=timeout,
                capture_output=True,
                creationflags=0x08000000,
            )
            if result.returncode:
                raise OSError(f"child exit {result.returncode}")
            return f"Windows child completed with exit code {result.returncode}"
        if cap is Capability.PROCESS_CONTROL:
            PROCESS_TERMINATE = 0x0001
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_TERMINATE, False, int(target.identifier)
            )
            if not handle:
                raise ctypes.WinError()
            try:
                if not ctypes.windll.kernel32.TerminateProcess(
                    handle, int(target.attribute("exit_code") or "1")
                ):
                    raise ctypes.WinError()
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
            return "Windows process termination requested"
        if cap in _NET:
            with socket.create_connection(
                (target.identifier, int(target.attribute("port") or "")),
                timeout=float(target.attribute("timeout") or "5"),
            ):
                pass
            return "Windows TCP connection established and closed"
        raise OSError("unsupported Windows operation")


__all__ = ["WindowsPlatformAdapter"]
