# AILEE v9.4 Local Computing Foundation

## Authority boundary

AILEE governs agency, not computation. Local Computing evaluates consequential
actions that an external principal, agentic system, tool, or automation submits
through the public `ailee.local_computing` boundary. The host operating system
provides computation and remains authoritative. AILEE does not claim to observe
or intercept arbitrary third-party processes.

The common lifecycle is **request → observe → normalize → contextualize → trust
evaluation → policy decision → platform enforcement → result → audit**. Prompt
1.1 implements the typed common contract and deterministic policy portion of
that lifecycle. It does not install kernel modules, drivers, kernel extensions,
patches, undocumented hooks, or replacement OS behavior.

## Package layout

- `ailee/local_computing/common/` contains platform-neutral requests, policy,
  capability reporting, enforcement results, errors, audits, and orchestration.
- `ailee/local_computing/linux/`, `windows/`, and `macos/` are explicit native
  integration locations. They contain no native enforcement in Prompt 1.1.
- `ailee.local_computing.LocalComputingTrust` is the public integration boundary;
  domains must not depend on common implementation internals.

## Safety semantics

Policy decisions (`ALLOW`, `DENY`, `RESTRICT`), reported platform capability,
and enforcement results are separate values. The default platform adapter
reports `UNAVAILABLE`; therefore a valid policy cannot be mistaken for native
enforcement. Unknown principals, invalid trust, malformed targets, unavailable
capabilities, and missing rules fail closed.

Trust states are ordered and deterministic. Degradation can only move from
`TRUSTED` to `DEGRADED` to `UNTRUSTED`; it cannot increase authority. Platform
support is reported as `SUPPORTED`, `SUPPORTED_WITH_LIMITATIONS`,
`OBSERVABLE_ONLY`, or `UNAVAILABLE`.

Audit events retain classified evidence, identifiers needed for correlation,
and explicit policy/capability/enforcement states. They intentionally omit the
target identifier and request payload so credentials, secrets, file contents,
and authentication tokens are not copied into common audit evidence.

## Deferred work

Linux, Windows, and macOS adapters are intentionally deferred to Prompt 1.2.
The test adapter verifies only the public common contract and is not evidence of
native OS interception or enforcement.
