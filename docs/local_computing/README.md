# Local Computing operator manual (v9.4.0)

**AILEE governs agency, not computation.** Local Computing is a foundational,
domain-independent Python boundary for actions deliberately submitted by an
application, agentic AI, tool, or automation. It does not govern ordinary host
activity and does not install an AILEE agent, sandbox, kernel component, or
network control plane.

```text
User / Applications
        ↓
Agentic AI / Tools / Automation
        ↓
AILEE Local Computing Trust Governance
        ↓
Supported OS Interfaces
        ↓
OS / Kernel
        ↓
Hardware
```

Each installation governs only its local host. There is no cross-machine AILEE
federation. Other domains may compose with this package only through its public
interfaces.

## Public boundary and lifecycle

Import contracts from `ailee.local_computing`. An integrator creates a
`Policy`, wraps it in `DeterministicPolicyEngine`, and submits an exact
`CapabilityRequest` to `LocalComputingTrust.govern()`. Use
`native_platform_adapter()` explicitly when native action is intended; the
service default is honestly unavailable.

The recorded lifecycle is:

```text
input → principal → capability → classified resource → trust state → policy
      → platform capability → policy decision → enforcement attempt
      → enforcement result / operation status → audit result
```

These are separate claims:

* **Policy decision** is `ALLOW`, `DENY`, or `RESTRICT`.
* **Platform capability** is `SUPPORTED`, `SUPPORTED_WITH_LIMITATIONS`,
  `OBSERVABLE_ONLY`, or `UNAVAILABLE`.
* **Enforcement result** records whether an attempt occurred and whether the
  adapter reports enforcement.
* **Operation completion** is a separate `completed` fact. An allowed request
  can fail, and enforcement need not mean completion.
* **Audit result** says whether an event was constructed and, when a sink is
  configured, accepted by that sink. Audit failure does not undo a side effect.

Malformed requests, unknown principals, invalid trust, unconfigured or
unsupported capabilities, unavailable adapters, malformed policy evidence, and
contradictory enforcement evidence fail closed. `TRUSTED` permits configured
authority; `DEGRADED` can only restrict it; `UNTRUSTED` denies; `INVALID` is a
validation error. Policy is default-deny.

## Requests, replay, and canonical behavior

`ResourceTarget.identifier` is the native path, executable, PID, or host;
`arguments` is an immutable direct argv; and `attributes` contains the small
capability-specific option set. Validation rejects duplicate/unknown options,
NULs in native strings, invalid ports/timeouts/PIDs, control characters in
evidence identifiers, and bounded-size violations. No shell interpolation is
introduced.

The service validates and snapshots a request before policy or adapter use.
Request IDs are atomically reserved before policy evaluation, are single-use
for the service lifetime, and cannot be mutated into a second operation.
Concurrent duplicates execute at most once. The bounded in-memory registry
holds 100,000 IDs by default and fails closed with an explicit capacity error;
it never evicts an ID to permit replay and is not durable across restart.
Restrictions are deduplicated and sorted when policy is constructed, giving
equivalent policy configuration a canonical constraint tuple. A policy may
configure at most 256 unique constraints per capability. Each constraint must
be a non-empty string of at most 1,024 characters without ASCII control
characters or DEL; invalid configuration is rejected when the policy engine is
constructed.

## Audit and sensitive data

An `AuditEvent` contains request ID, UTC time, platform, principal, capability,
resource **classification**, trust state, policy identity/decision, platform
support and limitations, enforcement status/detail, correlation ID, and typed
error. It intentionally excludes the target identifier, file content, argv,
host/port, credentials, stdout, and stderr. A failed sink returns
`AUDIT_FAILURE`, `audited=False`; it neither upgrades nor erases enforcement.

## Example

This Linux example uses a Linux path; choose an operator-approved native path
and matching `ExecutionContext.platform` on Windows or macOS.

```python
from ailee.local_computing import *

policy = Policy("example/v1", frozenset({"tool"}),
                {"tool": frozenset({Capability.FILESYSTEM_READ})}, {}, {})
trust = LocalComputingTrust(DeterministicPolicyEngine(policy),
                            native_platform_adapter())
result = trust.govern(CapabilityRequest(
    "request-1", Principal("tool"), Capability.FILESYSTEM_READ,
    ResourceTarget("operator-approved-input", "/tmp/input.txt"),
    ExecutionContext("linux"), TrustState.TRUSTED))
# Inspect policy, platform_capability, enforcement, audited, and error separately.
```

## Scope and common limitations

Networking means one outbound TCP connect from the local host and immediate
close. DNS and routing remain OS-controlled; policy does not bind a resolved
address. Paths, symlinks/reparse points, mounts, and executables can change
between decision and use. PIDs can be reused. Children and actions outside the
adapter are not transitively governed. Native permissions and privileges are
always authoritative.

AILEE does **not** intercept system calls, elevate privilege, provide mandatory
access control, guarantee rollback, mediate all computation, inspect traffic,
pin DNS, provide distributed trust, or claim that an allowed operation
completed. See [Linux](LINUX.md), [Windows](WINDOWS.md), [macOS](MACOS.md), and
the [final evidence report](EVIDENCE.md).
