# AILEE v9.4 Local Computing

## Authority boundary

AILEE governs agency, not computation. Local Computing evaluates consequential
actions that an external principal, agentic system, tool, or automation submits
through `ailee.local_computing`. The host OS remains authoritative. These
adapters neither intercept unrelated processes nor constitute a sandbox,
endpoint firewall, antivirus, kernel module, driver, or kernel extension.

The lifecycle is **request → trust/policy decision → native adapter → result →
audit**. Policy authorization, advertised platform capability, and successful
OS execution are deliberately separate facts. Native adapter selection is by
`native_platform_adapter()` and unknown platforms retain the fail-closed
unavailable adapter.

## Request data and evidence

`ResourceTarget.identifier` is the native path, executable, PID, or network
host. `arguments` is an immutable direct process argument vector. `attributes`
is an immutable set of native options (for example `content`, `port`, `timeout`,
`signal`, or Windows `exit_code`). Integrators must treat all three as sensitive.
They are used for the request but omitted from `AuditEvent`.

Audit evidence records the classified target, policy state, capability support,
platform limitations, enforcement state, sanitized result detail, correlation
ID, and typed error. It does not record file content, argv, paths, hosts, ports,
credentials, stdout, or stderr.

## Linux

`LinuxPlatformAdapter` discovers PID, real/effective UID and GID, supplementary
groups, root status, and kernel release. It implements file read/write/delete,
directory traversal, read access to a protected resource under caller
credentials, direct argv subprocess execution, signal-based process control,
and outbound TCP connection attempts using supported Python/POSIX facilities.

Those operations are `SUPPORTED_WITH_LIMITATIONS`: Linux permissions remain
authoritative; paths and symlinks are not containment; children inherit the
host process security context; signals are permission checked by the kernel;
and DNS/routing remain external. Resource and hardware requests are
`OBSERVABLE_ONLY`. Privilege elevation and credential-sensitive operations are
`UNAVAILABLE`. No namespaces, seccomp policy, cgroups, LSM rules, capabilities,
or kernel hooks are installed.

## Windows

`WindowsPlatformAdapter` is import-safe on other systems and activates only on
Win32. It queries whether the current process token can be opened, uses native
Windows path semantics and ACL checks through file APIs, direct argument-vector
process creation, `OpenProcess`/`TerminateProcess` with closed handles for
process control, and Winsock-backed TCP connections.

These paths are `SUPPORTED_WITH_LIMITATIONS`: the current token, ACLs, sharing
modes, reparse points, protected-process rules, Windows Firewall, and OS policy
remain authoritative. The adapter does not create a restricted token, Job
Object sandbox, AppContainer, WFP filter, or driver. Privilege/resource/hardware
context is `OBSERVABLE_ONLY`; credential-sensitive operations are `UNAVAILABLE`.

## macOS

`MacOSPlatformAdapter` discovers Darwin PID, UID/GID, supplementary groups,
root status, and release. It implements file operations under host permissions,
direct argv child creation, signal delivery, and outbound TCP connections.

These paths are `SUPPORTED_WITH_LIMITATIONS`: POSIX modes, ACLs, sandbox
entitlements, code signing, Transparency Consent and Control (TCC), System
Integrity Protection (SIP), and protected-process checks remain authoritative.
The adapter cannot bypass or grant those controls. It installs no Endpoint
Security client, Network Extension, kernel extension, or deprecated kernel
hook. Privilege/resource/hardware context is `OBSERVABLE_ONLY`;
credential-sensitive operations are `UNAVAILABLE`.

## Failure semantics

Malformed parameters, missing resources, permission failures, nonzero child
exit, timeout after child creation, platform mismatch, unavailable capability,
and unexpected OS or adapter results are explicit results. A policy decision
may remain `ALLOW` or `RESTRICT` while enforcement separately fails, but these
conditions never become a successful enforcement or completion claim. A
timeout is `PARTIAL`; observation-only and unavailable mechanisms are not
attempted. Policy denial always precedes OS execution. The `read-only`
constraint blocks mutating adapter actions. Contradictory adapter status,
attempt, enforcement, completion, and error fields are normalized to a failed
`UNEXPECTED_RESULT` rather than trusted.

## Local networking only

Networking is a local outbound TCP action: policy evaluates a host/port request,
the current host's adapter asks its OS to connect, records success/failure, and
closes the socket. This is not host-to-host AILEE communication. There is no
mesh, federation, consensus, cluster orchestration, remote trust exchange, or
cloud control plane.

## Verification scope

The test suite runs real temporary-file, child-process, loopback TCP, identity,
and audit integration paths on the hosting platform. CI has a dedicated
Ubuntu, Windows, and macOS matrix for those focused tests. All platform modules
are also parsed/imported on the current host, and foreign discovery logic is
simulated only for truth-table tests. A local run is evidence solely for its
host OS; native CI evidence must be read from the corresponding matrix job.
