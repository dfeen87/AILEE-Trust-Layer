# Linux operating manual (v9.4.0)

## Architecture and authority

`LinuxPlatformAdapter` activates only when `sys.platform` starts with `linux`.
It operates in user space with the hosting process's real/effective UID/GID and
supplementary groups. `identity()` reports that context, PID, root status, and
kernel release; it does not authenticate a requesting principal or change
credentials. Kernel permission checks remain authoritative.

| Capability group | Classification | Mechanism |
|---|---|---|
| file read/write/delete, directory traversal, protected-resource read | `SUPPORTED_WITH_LIMITATIONS` | Python file APIs, `os.remove`, `os.scandir` |
| tool/command/subprocess creation | `SUPPORTED_WITH_LIMITATIONS` | `subprocess.run`, direct argv, `shell=False` |
| process control | `SUPPORTED_WITH_LIMITATIONS` | `os.kill` signal delivery |
| network connect/destination access | `SUPPORTED_WITH_LIMITATIONS` | outbound `socket.create_connection` TCP |
| resource allocation, hardware request | `OBSERVABLE_ONLY` | context may be observed; action is not mediated |
| privilege-sensitive and credential-sensitive operations | `UNAVAILABLE` | no safe mechanism implemented |

There is currently no capability classified simply `SUPPORTED`: every native
action retains material user-space limitations.

## Operation and failure behavior

Policy and trust evaluation occurs before an adapter call. A `read-only`
restriction prevents mutation. File modification requires the `content`
attribute. Process execution passes the executable and immutable arguments
directly and captures output without auditing it; nonzero exit is failure and a
timeout is `PARTIAL` because creation may already have occurred. Process control
requires a positive PID and optionally a numeric signal. Networking requires a
valid port and optionally a timeout.

Missing/invalid resources, Linux permission denial, OS errors, platform
mismatch, or unavailable capability produce explicit non-completion. Audit
semantics are those in the common manual. Run under a dedicated least-privilege
account: root authority expands what the kernel permits but does not improve
AILEE's mediation guarantees.

## Deployment and evidence

Install the Python package and pass `native_platform_adapter()` (or an explicit
`LinuxPlatformAdapter`) to `LocalComputingTrust`. The v9.4.0 local acceptance
run on Linux exercised temporary-file read/write/delete/traversal, direct child
execution, loopback TCP, identity, policy denial/restriction, failure, replay,
malformed evidence, and audit paths. The workflow also *configures* an Ubuntu
native matrix job. No authenticated GitHub run results were available during
this acceptance pass, so CI verification is not claimed.

## Limits, not guarantees

The adapter is not a sandbox and does not install namespaces, seccomp, cgroups,
LSM rules, Linux capabilities, or kernel hooks. It cannot mediate calls that do
not pass through it. Path checking and native opening are not atomic: symlink,
mount, rename, and replacement races remain (user-space TOCTOU). Executable
identity and descendants are not pinned. PID identity is not pinned against
reuse. DNS resolution and routing can change after policy evaluation, and
traffic is not intercepted.

