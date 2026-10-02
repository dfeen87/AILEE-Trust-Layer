# macOS operating manual (v9.4.0)

## Darwin-native boundary

`MacOSPlatformAdapter` activates only on Darwin. It reports PID, UID/effective
UID, GID, supplementary groups, root status, and Darwin release. Operations use
the host process's permissions, sandbox profile, and entitlements; AILEE neither
grants nor bypasses them.

| Capability group | Classification on macOS | Mechanism |
|---|---|---|
| file read/write/delete, directory traversal, protected-resource read | `SUPPORTED_WITH_LIMITATIONS` | Darwin/Python file APIs under modes, ACLs, TCC, SIP, sandbox |
| tool/command/subprocess creation | `SUPPORTED_WITH_LIMITATIONS` | direct argv; Python-selected `posix_spawn`/fork implementation |
| process control | `SUPPORTED_WITH_LIMITATIONS` | `os.kill` signal delivery |
| network connect/destination access | `SUPPORTED_WITH_LIMITATIONS` | outbound TCP connect |
| privilege-sensitive, resource allocation, hardware request | `OBSERVABLE_ONLY` | host context only; no grant/confinement |
| credential-sensitive operation | `UNAVAILABLE` | no enforcement implementation |

No capability is unconditionally `SUPPORTED`.

## Operation, privilege, and failure

Policy/trust gates precede native calls; `read-only` blocks mutation. Direct
process argv avoids shell interpolation. Nonzero child exit, timeout, missing
resource, invalid input, permission denial, platform mismatch, socket/OS error,
and unavailable support remain explicit and distinct from authorization.
Darwin permissions, code signing, application sandbox, Transparency Consent and
Control (TCC), System Integrity Protection (SIP), and protected-process rules
remain authoritative. Root does not bypass every macOS protection and does not
turn this adapter into containment; deploy with the minimum required identity,
entitlements, and consent.

## Evidence and limitations

The module was imported and its distinct capability truth table simulated on
Linux; foreign execution reported `UNAVAILABLE`. It was **not natively run or
compiled on macOS in the local acceptance environment**. A `macos-latest`
native test job is configured, but no authenticated run result was accessible;
macOS CI success is therefore `NOT VERIFIED`.

The adapter installs no Endpoint Security client, Network Extension, kernel
extension, or deprecated kernel hook. It cannot mediate unrelated operations.
Symlink/mount/replacement TOCTOU remains; executable and descendant identity is
not transitively bound; PID reuse remains possible; DNS/routing is not pinned;
and traffic is not intercepted.

