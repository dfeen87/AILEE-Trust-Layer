# AILEE Trust Layer v9.3 Domain Governance

## 1. Purpose, scope, and claim level

This document describes the **implemented** Python v9.3 licensing and industrial-process domains. The repository is the authority. These modules govern evidence and analytical availability; they are not a certification, regulatory approval, formal verification result, penetration-test report, production validation, or machine-safety qualification.

Status terms used here:

- **IMPLEMENTED** — behavior present in code and exercised by repository tests.
- **CONFIGURABLE** — behavior supplied through a constructor policy or protocol.
- **EXTERNAL RESPONSIBILITY** — behavior deliberately delegated to an integrator or upstream system.
- **FUTURE / OUT OF SCOPE** — behavior not exposed by the current implementation.

Primary implementation paths are `ailee/domains/licensing/licensing.py`, `ailee/domains/industrial/industrial.py`, and `ailee/domains/dual_domain.py`. Public exports are in each domain's `__init__.py` and selected governors are exposed at package level.

## 2. Shared trust philosophy and v9.3 invariants

The implementation treats trust as separate, explainable evidence dimensions rather than a probability or aggregate score:

1. Caller-supplied timestamps and immutable inputs make decisions repeatable for identical inputs, subject to explicitly injected stateful dependencies such as a replay guard.
2. Authorization and throughput decisions carry enum reasons, boolean availability/authorization, deterministic decision identifiers, and audit evidence.
3. Missing, malformed, stale, contradictory, unsupported, duplicate, or invalid evidence does not silently become valid evidence.
4. Adapters only supply observations. Governors revalidate core constraints.
5. Audit records retain relevant identifiers, sources, timestamps, validity, and reason codes; they do not retain credential secrets.
6. Licensing authorization and industrial telemetry validity remain separate objects.
7. License denial cannot mutate telemetry or machine state. Industrial evaluation cannot create entitlement.
8. The industrial interfaces are supervisory/read-only. There is no method to command a machine, write a PLC value, bypass an interlock, suppress native safety behavior, or replace a native safety controller.
9. Trusted throughput is available only when both process-state and material evidence pass validation.
10. The composed service reports both dimensions and uses a conjunction only for `analytics_permitted`; it does not collapse them into a trust score.

“Deterministic” here means stable output for the same value inputs and the same dependency state. It does not mean that calls through a stateful replay guard return the same result after the guard records a request. It also does not mean that IDs or clocks are generated internally: callers supply request IDs, evidence IDs, interval IDs, and evaluation time.

## 3. Licensing Trust Domain

### 3.1 Architecture and evidence model

The implemented path is:

```text
LicenseContract + AuthorizationRequest
                         |
                         v
          injected CredentialVerifier
                         |
                         v
                 LicenseGovernor
                         |
                         v
      AuthorizationDecision + LicenseAuditEvidence
```

`LicenseContract` binds a license ID, customer, issuer, non-empty asset tuple, half-open validity window, entitlement tuple, and schema version. `AuthorizationRequest` supplies request, customer, asset, capability, evaluation time, schema version, and optional `IntegrityEvidence`. `IntegrityEvidence` records the upstream credential status plus license/customer/asset/issuer bindings, evidence identity, credential type, verification time, and schema.

All domain records are frozen dataclasses. This prevents ordinary field mutation after construction, but nested `Mapping` values are not deep-frozen by Python.

### 3.2 Credential and integrity boundary

**IMPLEMENTED:** `EvidenceStatusVerifier` refuses absent evidence, non-enum statuses, and any status other than the upstream explicit status. The governor fails closed if the verifier throws, returns no status, or returns an unsupported value.

**CONFIGURABLE:** `CredentialVerifier` is a protocol. A deployment can inject an established signature/credential verifier.

**EXTERNAL RESPONSIBILITY:** Cryptographic signature validation, key custody, revocation, entitlement-payload canonicalization, and issuer trust are not implemented by this module. The default verifier consumes an upstream status; it does not prove a signature. Consequently, tamper detection over the entitlement payload exists only when the injected verifier covers that content and returns `INVALID`.

### 3.3 Customer, asset, entitlement, and validity semantics

Authorization uses a fixed fail-closed validation pipeline:

1. Require typed contract/request objects and supported schema `1.0`.
2. Require non-empty identities, canonical identifiers, aware datetimes, non-empty assets and entitlements, no duplicate bindings, and `valid_from < valid_until`.
3. Call the credential verifier and require `CredentialStatus.VALID`.
4. Require typed, supported integrity evidence with canonical identifiers and an aware `verified_at`.
5. Require request customer to equal contract customer.
6. Require request asset membership in the contract assets.
7. Require evidence to bind the same license, asset, customer, and issuer; future verification time is invalid.
8. Apply the half-open validity window: `valid_from <= evaluated_at < valid_until`.
9. Require exact capability membership in `entitlements`.
10. If configured, require the replay guard to accept `(request_id, evaluated_at)`.

Identifiers are not case-normalized or whitespace-trimmed to grant access. Unknown well-formed capabilities therefore receive `ENTITLEMENT_MISSING`; malformed capability strings receive `MALFORMED_IDENTIFIER`.

### 3.4 Decision and failure semantics

`AuthorizationDecision` contains `authorized`, `LicenseReason`, a SHA-256 decision ID, an audit record, and a derived `AUTHORIZED`/`DENIED` result. Reasons include insufficient evidence, invalid credential, customer/asset mismatch, invalid window, missing entitlement, malformed identity, invalid contract, verifier unavailable, unsupported schema, and replay detection.

Malformed untyped top-level inputs now yield an auditable `INSUFFICIENT_EVIDENCE` denial rather than failing while the audit record is assembled. The audit's `checks_performed` tuple names the configured validation pipeline; it is not a per-branch execution trace and later checks may not have run after an earlier failure.

The default governor is stateless and provides no replay protection. Replay rejection is **CONFIGURABLE** through the `replay_guard` boundary. Durable, distributed replay storage is an **EXTERNAL RESPONSIBILITY**.

## 4. Industrial Process Trust Domain

### 4.1 Architecture and read-only boundary

The implemented evidence path is:

```text
Machine/PLC -> external read-only acquisition -> TelemetryObservation
                                                   + MaterialObservation
                                                   + ProcessInterval
                                                           |
                                      TelemetryValidator + ThroughputGovernor
                                                           |
                                  ThroughputResult + IndustrialAuditEvidence
```

`ReadOnlyTelemetryAdapter` exposes only `observations(machine_id)`. Neither it nor any industrial governor exposes writes or control commands. The text “read-only” describes the software interface in this repository; physical/network enforcement and adapter implementation review are **EXTERNAL RESPONSIBILITIES**.

### 4.2 Telemetry validation

`TelemetryValidator` requires schema `1.0`, canonical identifiers, aware occurrence/receipt/evaluation timestamps, `occurrence <= receipt <= evaluation`, and age within a positive configurable maximum (five minutes by default). Missing values and non-finite numeric values fail. Optional bounds must be finite and coherent. Allowed signals, sources, and units are configurable allowlists.

A `ProcessInterval` must have a canonical identity/source, a positive aware time range ending no later than evaluation, a non-empty tuple of uniquely identified typed state observations, correct machine binding, in-range timestamps, and nondecreasing observation order. At least one signal named `process_state` must support the interval's start state; unrelated valid telemetry cannot substitute for state evidence. All `process_state` values must agree with the interval state.

The current interval model accepts only a constant state (`start_state == end_state`) for throughput. A changing-state interval is contradictory and must be split by the caller. `FAULT` is excluded with `ACTIVE_FAULT`; every non-`RUNNING` constant state is non-productive.

### 4.3 Transition and recovery policy

`ProcessTransitionPolicy` is a separate **CONFIGURABLE** transition check. Same-state transitions pass. Listed ordinary transitions pass when allowed. `FAULT`, `UNPLANNED_STOP`, or `COMPLETE` to `RUNNING` requires both explicit policy permission and a canonical recovery evidence ID.

The throughput governor does not invoke this policy or infer recovery from the next `RUNNING` interval. Orchestration that constructs intervals must apply transition policy. This is an **EXTERNAL RESPONSIBILITY** and an important limitation: a standalone constant `RUNNING` interval does not prove the preceding recovery transition.

### 4.4 Productive time and throughput

Wall-clock elapsed time is the positive interval duration. Validated productive time equals elapsed time only for a fully valid constant `RUNNING` interval whose required material evidence also validates; otherwise productive time is zero. Thus elapsed and productive time are distinct result fields.

Material start/end observations must be typed, supported, canonically identified, uniquely identified, machine-bound, aware, chronologically coherent, received by evaluation, inside the interval, and from the same source/run. Units must match exactly. Quantities must be finite and nondecreasing. A decrease/counter reset is `INVALID_MATERIAL_VALUE`; this implementation does not infer rollover or reset. Throughput is:

```text
(end quantity - start quantity) / (productive seconds / 3600)
```

A zero/negative duration is rejected before division. Missing material makes throughput unavailable. There is no unit-conversion service, and a matching string is treated only as a shared unit label—not dimensional verification.

The governor evaluates one interval at a time. It has no collection aggregator, so it does not itself sum overlapping intervals. Detection/deduplication of overlap across separately evaluated intervals is **EXTERNAL RESPONSIBILITY**; callers must not naively sum overlapping `productive_seconds`.

### 4.5 Events and alarm chronology

`EventLedger` is in-memory and append-only through its public API. It validates schema, identifiers, aware timestamps, occurrence no later than receipt, optional receipt no later than a configured evaluation time, optional category allowlists, unique event IDs, and conflicting same-machine/time/type/source evidence. `chronological()` sorts by occurrence timestamp and then event ID, producing deterministic ties independent of arrival order. `received_at` remains available to distinguish late arrival.

Chronological first appearance is not root-cause inference. Persistence, distributed ordering, clock synchronization, causal analysis, and ledger durability are **EXTERNAL RESPONSIBILITIES**.

### 4.6 Industrial result and audit model

`ThroughputResult` retains availability, a `ThroughputReason`, `TelemetryValidity`, wall-clock and productive seconds, optional material delta/unit/rate, a deterministic SHA-256 decision ID, and `IndustrialAuditEvidence`. Audit evidence includes interval/machine identity, observation IDs, deduplicated sorted sources, caller-supplied evaluation time, validity, and reason.

No audit sink abstraction exists. Sink durability, availability, access control, and retention are **EXTERNAL RESPONSIBILITIES**.

## 5. Cross-domain separation

`GovernedAnalyticsService` evaluates licensing and throughput independently using the same caller-supplied evaluation time. `GovernedAnalyticsDecision` exposes both complete decisions, the observed process state, and `machine_control_issued=False`. `analytics_permitted` is true only when authorization succeeds **and** throughput is available.

This conjunction is a capability gate, not an aggregate trust score:

| Licensing | Industrial evidence | Protected analytics | Machine/process effect |
|---|---|---|---|
| authorized | valid | permitted | none |
| denied | valid | denied; throughput result remains valid | none |
| authorized | invalid/insufficient | withheld; authorization remains valid | none |
| denied/unavailable | invalid/faulted | denied; both reasons retained separately | none |

If an injected analytics governor raises, the exception currently propagates; frozen input records remain unchanged. The composition layer does not convert arbitrary analytics exceptions into a result. Exception mapping and service availability are **EXTERNAL RESPONSIBILITIES**.

## 6. Determinism and serialization

Decision hashes use sorted-key compact JSON over selected stable identities, evaluation time, evidence IDs, and reason. Audit source ordering is sorted. Event order is `(timestamp, event_id)`. Tests repeat representative licensing, throughput, and composed decisions.

Intentional/state-dependent boundaries:

- replay-guard state can turn the second identical request into `REPLAY_DETECTED`;
- caller-supplied IDs and clocks may differ between otherwise similar real-world calls;
- event arrival order is preserved only in `received_at`, while chronological projection uses occurrence time;
- floating-point rate calculation uses Python binary floating point with no rounding policy;
- no domain-specific JSON serializer is provided. Dataclass/enum/datetime encoding policy is an **EXTERNAL RESPONSIBILITY**.

## 7. Security and trust boundaries

- Credential authenticity, key management, revocation, trusted time, and protected transport are external.
- Telemetry source authentication, sensor calibration, PLC/historian integrity, time synchronization, and physical safety are external.
- Allowlist correctness and transition-policy invocation are deployment responsibilities.
- Audit mappings may contain integrator-provided metadata/details. The governors do not log them, but callers must avoid placing secrets in those fields.
- The code performs governance over supplied evidence; it does not attest the hardware that produced it.

## 8. Explicit non-goals and known limitations

**FUTURE / OUT OF SCOPE:** machine control, PLC writes, interlock control, native safety replacement, automatic remediation, signature algorithms, license issuance, network services, persistent ledgers, distributed replay protection, interval construction, overlap aggregation, unit conversion, counter-reset inference, causal/root-cause analysis, audit-sink delivery, production deployment validation, certification, and regulatory approval.

Known implementation limits include: default credential status is trusted from upstream rather than cryptographically proven; state-transition policy is not automatically coupled to throughput; freshness applies to state observations but material observations have chronology rather than a separate max-age threshold; exact material sampling at interval endpoints is not required; overlap handling is per-caller; mappings are not deep immutable; and industrial exceptions from injected components are not normalized by the composition service.
