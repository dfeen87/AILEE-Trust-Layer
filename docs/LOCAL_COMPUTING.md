# AILEE v9.4 Local Computing

> Final operator documentation is organized in
> [`docs/local_computing/`](local_computing/README.md), including platform
> manuals and the v9.4.0 acceptance evidence matrix. This overview remains the
> compact implementation reference.

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

Requests are validated and snapshotted before policy evaluation. Request IDs
are single-use within a `LocalComputingTrust` instance and are atomically
reserved before evaluation, so replay and concurrent duplicate submission fail
closed. Retention lasts for the instance lifetime (not across process restart)
and is bounded by `max_request_ids` (100,000 by default). When that bound is
reached, new IDs fail closed rather than evicting replay evidence; callers must
rotate the service instance only at an intentional replay-boundary reset.
Invalid requests are rejected before reservation because they cannot authorize
or execute an action. Duplicate/unknown attributes, NUL in native path or argv
values, control characters in audit identifiers, oversized request data,
invalid ports, and invalid timeouts are rejected before adapter discovery.
Legitimate Unicode and non-NUL characters in native arguments remain valid.
Policy construction rejects malformed capabilities and conflicting
allow/restrict rules, and canonicalizes duplicate/reordered constraints.

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

Adapter capability evidence must identify the requested capability and contain
well-formed support and limitation fields. Enforcement evidence must be an
exact contract object with typed booleans/enums, bounded detail, and a
non-contradictory status/error combination. Audit construction or sink failure
is reported as `AUDIT_FAILURE`; it never changes native failure into success or
rewrites a completed native side effect as unexecuted. The returned enforcement
evidence remains available when audit creation fails, while `audit` is absent
and `audited` is false; sink failure retains the constructed event but marks it
unpersisted. Rollback is not claimed. Policy exceptions or malformed policy
outcomes fail closed and consume the already-reserved request ID.

## Local networking only

Networking is a local outbound TCP action: policy evaluates a host/port request,
the current host's adapter asks its OS to connect, records success/failure, and
closes the socket. This is not host-to-host AILEE communication. There is no
mesh, federation, consensus, cluster orchestration, remote trust exchange, or
cloud control plane.

## Remaining user-space race boundaries

Pathnames may be replaced (including by symlink, mount, junction, or reparse
point changes) between policy evaluation and native use. Hostname resolution
and routing may change before a connection. POSIX process control identifies a
target by PID, which can be reused; Windows obtains a process handle only at
enforcement time. Executables can be replaced before process creation, and
child behavior is not transitively governed. The adapters report these as
limitations and do not claim atomic identity binding or sandbox containment.

## Verification scope

The test suite runs real temporary-file, child-process, loopback TCP, identity,
and audit integration paths on the hosting platform. CI has a dedicated
Ubuntu, Windows, and macOS matrix for those focused tests. All platform modules
are also parsed/imported on the current host, and foreign discovery logic is
simulated only for truth-table tests. A local run is evidence solely for its
host OS; native CI evidence must be read from the corresponding matrix job.
