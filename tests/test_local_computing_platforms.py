"""Platform tests label native Linux execution separately from import simulations."""

import socket
import sys
import threading

import pytest

from ailee.local_computing import (
    Capability,
    CapabilityRequest,
    CapabilitySupport,
    DeterministicPolicyEngine,
    EnforcementStatus,
    ExecutionContext,
    LocalComputingError,
    LocalComputingTrust,
    Policy,
    PolicyDecision,
    Principal,
    ResourceTarget,
    TrustState,
    native_platform_adapter,
)
from ailee.local_computing.linux import LinuxPlatformAdapter
from ailee.local_computing.macos import MacOSPlatformAdapter
from ailee.local_computing.windows import WindowsPlatformAdapter


def req(
    capability,
    identifier,
    *,
    attributes=(),
    arguments=(),
    platform="linux",
    trust=TrustState.TRUSTED
):
    return CapabilityRequest(
        "native-1",
        Principal("agent"),
        capability,
        ResourceTarget(
            "test-resource", str(identifier), tuple(arguments), tuple(attributes)
        ),
        ExecutionContext(platform, "user", "platform-test"),
        trust,
    )


def service(capability, adapter, *, restricted=False, events=None):
    rules = {"agent": frozenset({capability})}
    policy = Policy(
        "native-test/v1",
        frozenset({"agent"}),
        {} if restricted else rules,
        rules if restricted else {},
        {capability: ("read-only",)} if restricted else {},
    )
    return LocalComputingTrust(
        DeterministicPolicyEngine(policy),
        adapter,
        events.append if events is not None else None,
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux integration test"
)
def test_linux_filesystem_allow_deny_restrict_failure_and_audit(tmp_path):
    adapter = LinuxPlatformAdapter()
    path = tmp_path / "governed.txt"
    path.write_text("old", encoding="utf-8")
    events = []
    read = service(Capability.FILESYSTEM_READ, adapter, events=events).govern(
        req(Capability.FILESYSTEM_READ, path)
    )
    assert (
        read.policy.decision is PolicyDecision.RESTRICT
    )  # native limitation is explicit
    assert read.enforcement.status is EnforcementStatus.COMPLETED
    assert read.enforcement.completed and events[0].platform_limitations
    assert "file read completed" in events[0].enforcement_detail

    denied = service(Capability.FILE_MODIFY, adapter).govern(
        req(
            Capability.FILE_MODIFY,
            path,
            attributes=(("content", "new"),),
            trust=TrustState.UNTRUSTED,
        )
    )
    assert denied.enforcement.status is EnforcementStatus.NOT_ATTEMPTED
    assert path.read_text(encoding="utf-8") == "old"

    restricted = service(Capability.FILE_MODIFY, adapter, restricted=True).govern(
        req(Capability.FILE_MODIFY, path, attributes=(("content", "new"),))
    )
    assert restricted.enforcement.status is EnforcementStatus.NOT_ATTEMPTED
    assert path.read_text(encoding="utf-8") == "old"

    written = service(Capability.FILE_MODIFY, adapter).govern(
        req(Capability.FILE_MODIFY, path, attributes=(("content", "new"),))
    )
    assert written.enforcement.completed and path.read_text(encoding="utf-8") == "new"
    traversed = service(Capability.DIRECTORY_TRAVERSE, adapter).govern(
        req(Capability.DIRECTORY_TRAVERSE, tmp_path)
    )
    assert traversed.enforcement.completed
    deleted = service(Capability.FILE_DELETE, adapter).govern(
        req(Capability.FILE_DELETE, path)
    )
    assert deleted.enforcement.completed and not path.exists()

    missing = service(Capability.FILESYSTEM_READ, adapter).govern(
        req(Capability.FILESYSTEM_READ, tmp_path / "missing")
    )
    assert missing.error is LocalComputingError.INVALID_TARGET


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux integration test"
)
def test_linux_process_network_and_identity_are_native(tmp_path):
    adapter = LinuxPlatformAdapter()
    process = service(Capability.SUBPROCESS_CREATE, adapter).govern(
        req(
            Capability.SUBPROCESS_CREATE,
            sys.executable,
            arguments=("-c", "raise SystemExit(0)"),
        )
    )
    assert process.enforcement.status is EnforcementStatus.COMPLETED
    assert adapter.identity()["uid"] >= 0

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    accepted = []
    thread = threading.Thread(
        target=lambda: accepted.append(listener.accept()[0]), daemon=True
    )
    thread.start()
    network = service(Capability.NETWORK_CONNECT, adapter).govern(
        req(Capability.NETWORK_CONNECT, "127.0.0.1", attributes=(("port", str(port)),))
    )
    thread.join(timeout=2)
    for connection in accepted:
        connection.close()
    listener.close()
    assert network.enforcement.status is EnforcementStatus.COMPLETED


def test_platform_dispatch_matches_runtime():
    selected = native_platform_adapter()
    if sys.platform.startswith("linux"):
        assert isinstance(selected, LinuxPlatformAdapter)
    elif sys.platform == "win32":
        assert isinstance(selected, WindowsPlatformAdapter)
    elif sys.platform == "darwin":
        assert isinstance(selected, MacOSPlatformAdapter)


@pytest.mark.skipif(
    sys.platform not in {"win32", "darwin"},
    reason="Windows/macOS native integration test",
)
def test_windows_and_macos_native_file_process_network_and_identity(tmp_path):
    """Exercise real native calls on their host; never simulate foreign calls."""
    adapter = native_platform_adapter()
    platform_name = "windows" if sys.platform == "win32" else "macos"
    path = tmp_path / "native-governed.txt"
    path.write_text("local", encoding="utf-8")

    read = service(Capability.FILESYSTEM_READ, adapter).govern(
        req(Capability.FILESYSTEM_READ, path, platform=platform_name)
    )
    process = service(Capability.SUBPROCESS_CREATE, adapter).govern(
        req(
            Capability.SUBPROCESS_CREATE,
            sys.executable,
            arguments=("-c", "raise SystemExit(0)"),
            platform=platform_name,
        )
    )

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    accepted = []
    thread = threading.Thread(
        target=lambda: accepted.append(listener.accept()[0]), daemon=True
    )
    thread.start()
    network = service(Capability.NETWORK_CONNECT, adapter).govern(
        req(
            Capability.NETWORK_CONNECT,
            "127.0.0.1",
            attributes=(("port", str(listener.getsockname()[1])),),
            platform=platform_name,
        )
    )
    thread.join(timeout=2)
    for connection in accepted:
        connection.close()
    listener.close()

    assert read.enforcement.status is EnforcementStatus.COMPLETED
    assert process.enforcement.status is EnforcementStatus.COMPLETED
    assert network.enforcement.status is EnforcementStatus.COMPLETED
    assert adapter.identity()["available"] is True


def test_foreign_platforms_import_and_report_unavailable_on_current_host():
    adapters = [
        (WindowsPlatformAdapter(), "windows"),
        (MacOSPlatformAdapter(), "macos"),
    ]
    if sys.platform == "win32":
        adapters.pop(0)
    if sys.platform == "darwin":
        adapters.pop(1)
    for adapter, name in adapters:
        capability = adapter.capability(Capability.FILESYSTEM_READ)
        assert capability.support is CapabilitySupport.UNAVAILABLE
        result = adapter.enforce(
            req(Capability.FILESYSTEM_READ, "missing", platform=name), ()
        )
        assert result.status is EnforcementStatus.NOT_ATTEMPTED
        assert result.error is LocalComputingError.PLATFORM_UNAVAILABLE


def test_each_platform_has_distinct_capability_truth_table():
    # Logic test only: force discovery mode; no foreign native call is executed.
    linux, windows, macos = (
        LinuxPlatformAdapter(),
        WindowsPlatformAdapter(),
        MacOSPlatformAdapter(),
    )
    linux._available = windows._available = macos._available = True
    assert (
        linux.capability(Capability.PRIVILEGE_SENSITIVE_OPERATION).support
        is CapabilitySupport.UNAVAILABLE
    )
    assert (
        windows.capability(Capability.PRIVILEGE_SENSITIVE_OPERATION).support
        is CapabilitySupport.OBSERVABLE_ONLY
    )
    assert (
        macos.capability(Capability.PRIVILEGE_SENSITIVE_OPERATION).support
        is CapabilitySupport.OBSERVABLE_ONLY
    )
    assert "TCC" in macos.capability(Capability.FILESYSTEM_READ).limitations[0]
    assert "ACL" in windows.capability(Capability.FILESYSTEM_READ).limitations[0]


def test_context_mismatch_and_observable_capability_fail_closed():
    adapter = LinuxPlatformAdapter()
    mismatch = adapter.enforce(
        req(Capability.FILESYSTEM_READ, "/dev/null", platform="windows"), ()
    )
    assert mismatch.error is LocalComputingError.PLATFORM_INTEGRATION_FAILURE
    observable = adapter.enforce(req(Capability.RESOURCE_ALLOCATE, "memory"), ())
    assert observable.status is EnforcementStatus.NOT_ATTEMPTED
