# AILEE v9.3 Simulation and Validation Evidence

## 1. Methodology and environment

Validation was performed against the repository working tree on 2026-09-30 using only synthetic identifiers, quantities, events, and timestamps. The fixed Python fixtures use `2026-01-15T12:00:00+00:00`; no employer, vendor, recipe, private PLC-tag, or confidential machine data was used.

Environment actually observed: Python 3.14.4, pytest 9.0.3, Rust/Cargo 1.95.0, Node 24.15.0, npm 11.4.2, CMake 3.28.3, and GNU C++ 13.3.0. Scenario evidence is in `tests/test_licensing_domain.py`, `tests/test_industrial_domain.py`, `tests/test_dual_domain.py`, `tests/test_dual_domain_adversarial.py`, and `tests/test_v930_final_validation.py`.

Status vocabulary: **PASS** means the stated command/scenario executed and assertions passed; **FAIL** means it executed unsuccessfully; **NOT RUN** means it was not executed; **ENVIRONMENTALLY BLOCKED** means an environment dependency prevented execution; **OUT OF SCOPE** means the repository exposes no such mechanism.

## 2. Licensing simulations

| Scenario | Expected result | Actual observed result | Status |
|---|---|---|---|
| Valid customer/asset/credential/window/entitlement | authorize | `AUTHORIZED`; repeat decision equality and stable ID | PASS |
| Missing entitlement / unknown capability | explicit denial, no expansion | `ENTITLEMENT_MISSING` | PASS |
| Expired (`now == valid_until`) | deny | `EXPIRED` | PASS |
| Not yet valid | deny | `NOT_YET_VALID` | PASS |
| Activation boundary (`now == valid_from`) | authorize | authorized | PASS |
| Customer mismatch | deny | `CUSTOMER_MISMATCH` | PASS |
| Asset mismatch | deny | `ASSET_MISMATCH` | PASS |
| Missing/unverified credential | fail closed | `INSUFFICIENT_EVIDENCE` | PASS |
| Invalid credential | fail closed | `INVALID_CREDENTIAL` | PASS |
| Altered credential bindings | reject | license/customer/asset/issuer substitutions yield `INVALID_CREDENTIAL` | PASS |
| Tampered signed payload reported by upstream verifier | reject | injected verifier returns invalid; `INVALID_CREDENTIAL` | PASS |
| Malformed identifiers or untyped payload | explicit denial | `MALFORMED_IDENTIFIER` or auditable `INSUFFICIENT_EVIDENCE`; no authorization exception | PASS |
| Contradictory/future verification evidence | no escalation | `INVALID_CREDENTIAL` | PASS |
| Duplicate contract bindings | invalid | `INVALID_CONTRACT` | PASS |
| Contract/request/evidence schema `99` | unsupported | `UNSUPPORTED_SCHEMA` | PASS |
| Verifier raises/returns indeterminate | fail closed | `VERIFICATION_UNAVAILABLE` | PASS |
| Duplicate request with one-shot guard | reject replay | first authorized; second `REPLAY_DETECTED` | PASS |
| Replay without configured guard | document boundary | default governor is stateless; durable replay prevention is external | OUT OF SCOPE |

The tamper scenario proves propagation of an injected verifier's integrity result, not cryptographic verification by AILEE. The default `EvidenceStatusVerifier` does not implement cryptography.

## 3. Industrial-process simulations

| Scenario | Expected result | Actual observed result | Status |
|---|---|---|---|
| Normal 60-second RUNNING, 100→110 kg | validated productive time/rate | 60 productive seconds, delta 10, 600 kg/hour, `CALCULATED` | PASS |
| Planned hold | excluded | zero productive time, `NON_PRODUCTIVE_STATE` | PASS |
| Unplanned stop | excluded | zero productive time, `NON_PRODUCTIVE_STATE` | PASS |
| Fault during interval | excluded | zero productive time, `FAULT_EXCLUDED`/`ACTIVE_FAULT` | PASS |
| Fault recovery policy | require explicit evidence | allowed recovery needs allowed transition plus recovery ID | PASS |
| Standalone RUNNING after fault | do not infer recovery | governor has no history; transition orchestration remains external | OUT OF SCOPE |
| Missing state telemetry | unavailable | `MISSING_TELEMETRY` | PASS |
| Unrelated telemetry substituted for state | unavailable | `MISSING_TELEMETRY` after Implementation 3 correction | PASS |
| Stale telemetry | unavailable | `STALE_TELEMETRY` | PASS |
| Reordered observations | deterministic reject | `OUT_OF_ORDER` validity | PASS |
| Duplicate observation/material identity | no double count | `DUPLICATE_EVIDENCE` | PASS |
| Future-dated telemetry/receipt | reject | `INVALID_TIMESTAMP` | PASS |
| End before/equal start; interval in future | reject | `INVALID_TIMESTAMP`, zero productive time | PASS |
| NaN, +∞, -∞ telemetry/material | reject | `INVALID_VALUE` or `INVALID_MATERIAL_VALUE`; no rate | PASS |
| Incompatible unit strings | unavailable | `UNIT_MISMATCH` | PASS |
| Missing material evidence | unavailable | `MATERIAL_EVIDENCE_MISSING` | PASS |
| Zero/negative productive duration | no division | `INVALID_TIMESTAMP`; no rate | PASS |
| Counter decrease/reset | no inference | `INVALID_MATERIAL_VALUE` | PASS |
| Wrong-machine state/material | reject | `CONTRADICTORY_STATE` | PASS |
| Overlapping intervals | no implicit aggregation | each interval evaluated independently; no aggregate API exists | PASS (boundary demonstrated) |
| Normal, tied, reordered, and late events | deterministic chronology | sorted by occurrence then event ID; late `received_at` retained | PASS |
| Duplicate/conflicting/future/malformed event | reject | `ValueError` at explicit ledger ingestion boundary | PASS |
| Root cause from first appearance | do not claim | ledger records chronology only | PASS |
| Telemetry adapter disappears mid-run | no runtime adapter orchestration exists | adapter protocol only; empty evidence path tested as missing | OUT OF SCOPE |

The overlap test does **not** prove safe behavior in a caller-created aggregate. It proves that this implementation has no implicit sum. Cross-interval overlap detection is external.

## 4. Cross-domain trust matrix

The parameterized matrix in `tests/test_v930_final_validation.py` and focused tests in `tests/test_dual_domain.py`/`tests/test_dual_domain_adversarial.py` were executed.

| Case | Expected result | Actual observed result | Status |
|---|---|---|---|
| 1. Valid license + entitlement + valid telemetry | permit protected trusted analytics | both decisions valid; `analytics_permitted=True` | PASS |
| 2. Valid license, entitlement absent, telemetry valid | deny capability; retain telemetry/process | `ENTITLEMENT_MISSING`, throughput remains calculated, state unchanged | PASS |
| 3. Valid entitlement, telemetry missing | authorization may pass; analytics unavailable | authorized plus `MISSING_TELEMETRY`; not permitted | PASS |
| 4. Invalid license, telemetry valid | commercial denial only | `INVALID_CREDENTIAL`; throughput remains calculated | PASS |
| 5. Verifier unavailable, telemetry valid | fail closed commercially | `VERIFICATION_UNAVAILABLE`; throughput remains calculated | PASS |
| 6. Analytics component raises | cannot mutate inputs/licensing/machine | exception propagates; frozen input tuple remains equal | PASS |
| 7. Entitlement absent, process fault | retain separate reasons | `ENTITLEMENT_MISSING` and `FAULT_EXCLUDED`; no control issued | PASS |

Every composed result sets `machine_control_issued=False`. More importantly, the exposed interfaces contain no machine-control operation.

## 5. Failure injection

| Injected failure | Actual behavior | Status |
|---|---|---|
| Credential verifier throws | `VERIFICATION_UNAVAILABLE`, denied | PASS |
| Verifier returns `None` | `VERIFICATION_UNAVAILABLE`, denied | PASS |
| Missing/corrupt licensing objects | auditable `INSUFFICIENT_EVIDENCE`, denied | PASS |
| Unsupported licensing/telemetry schema | explicit unsupported result | PASS |
| State source stale/disappears (empty tuple) | no throughput | PASS |
| Malformed state member | `MALFORMED_TELEMETRY` | PASS |
| Impossible/future interval | `INVALID_TIMESTAMP` | PASS |
| Invalid material/source/run binding | explicit invalid/contradictory result | PASS |
| Industrial analytics raises | exception propagates; inputs and license evidence unchanged | PASS |
| Audit sink unavailable | no audit sink abstraction exists | OUT OF SCOPE |

No executed failure increased authorization or created throughput. The analytics exception is explainable only as an exception, not a normalized governed result; this is a documented limitation.

## 6. Determinism and repeatability

Representative licensing, throughput, event ordering, and composed scenarios were repeated. A final composed result was evaluated ten times: object equality, authorization decision ID, and throughput decision ID were stable. Sorted source sets and `(timestamp, event_id)` chronology avoid collection/arrival nondeterminism.

The replay test intentionally differs on the second call because the injected guard is stateful. IDs/clocks are caller supplied. Python binary-float arithmetic has no explicit cross-runtime rounding/canonical serialization guarantee. No domain JSON serializer exists, so domain JSON serialization was **NOT RUN**; stable in-memory dataclass equality and decision hashes were tested instead.

## 7. Defects found and corrections made

1. **Malformed top-level license inputs could fail while building audit output.** `LicenseGovernor.authorize` now uses safe field extraction and returns an auditable denial; regression cases cover `None` and corrupt strings.
2. **Unrelated valid telemetry could support a claimed RUNNING interval.** Throughput validation now requires at least one `process_state` observation and verifies its value; tests and the composed fixture were corrected.
3. **New-domain typing findings.** Licensing audit time is optional for malformed top-level requests, safe decision-ID extraction was narrowed, numeric comparisons are explicitly narrowed, and the event ledger collection is typed.

Affected scenario suites were rerun after each correction; the focused v9.3 run finished with 74 passing tests.

## 8. Complete validation results

| Command | Result | Classification |
|---|---|---|
| `python -m pytest tests/test_licensing_domain.py tests/test_industrial_domain.py tests/test_dual_domain.py tests/test_dual_domain_adversarial.py tests/test_v930_final_validation.py -q` | 74 passed | PASS |
| `python -m pytest -q` | 264 passed, 142 pre-existing warnings | PASS with warnings |
| `cargo test --all-targets` | 32 passed (25 unit + 7 integration), 0 failed | PASS |
| `cargo fmt --all -- --check` | clean | PASS |
| `cargo clippy --all-targets --all-features -- -D warnings` | pre-existing `new_without_default` in `src/video_temporal_provenance.rs` | FAIL (outside v9.3 domain scope) |
| `python -m compileall -q ailee tests` | success | PASS |
| `python -m black --check ailee tests` | 114 pre-existing files would be reformatted | FAIL (repository-wide baseline; not rewritten) |
| `python -m mypy ailee` | 389 errors in 42 files, including broad pre-existing domains | FAIL (repository-wide baseline) |
| `python -m mypy --follow-imports=skip ailee/domains/licensing/licensing.py ailee/domains/industrial/industrial.py ailee/domains/dual_domain.py` | no issues in the three v9.3 implementation files | PASS |
| `npm test --prefix packages/ailee-ts` | 178 passed | PASS |
| `npm run typecheck --prefix packages/ailee-ts` | success | PASS |
| `npm run build --prefix packages/ailee-ts` | CJS, ESM, declarations built | PASS |
| `cmake -S . -B build && cmake --build build` | configured and built | PASS |
| `ctest --test-dir build --output-on-failure` | 1 passed | PASS |

Pytest warnings include test functions returning values and deprecated naive `utcnow()` usage outside the v9.3 dual domains. npm printed an environment-configuration deprecation warning for `http-proxy`; commands still passed. None of these outcomes is represented as certification.

## 9. Traceability and release gate

Major behavior maps to the three implementation modules and the five test modules named in §1. Documentation was checked against those sources after the final runs.

- [x] Valid licensing requires contract structure, credential status/binding, time, customer, asset, and entitlement—not presence alone.
- [x] Customer/asset/entitlement boundaries are explicit.
- [x] Missing or invalid licensing evidence fails closed.
- [x] Licensing failure cannot alter machine/process state: **NO alteration is implemented.**
- [x] Industrial governance is read-only: **YES at the repository interface boundary.**
- [x] Analytics cannot override PLC/interlock/native safety behavior: **NO control API exists.**
- [x] Invalid telemetry cannot produce trusted throughput.
- [x] Wall-clock elapsed and validated productive time are distinct.
- [x] Entitlement and telemetry validity are independent.
- [x] Decisions expose deterministic machine-readable statuses/reasons.
- [x] Cross-domain simulations were executed.
- [x] Failure paths were executed where architecture supports them.
- [x] Documentation claims were checked against code/tests.
- [x] Limitations and external responsibilities are explicit.

## 10. Reproduction

From the repository root, run the commands in §8. No network service, hardware, real credential, PLC, or proprietary data is required for the Python v9.3 simulations. Build products (`build/`, `target/`, and `packages/ailee-ts/dist/`) are generated artifacts and are not evidence beyond the recorded command outcomes.
