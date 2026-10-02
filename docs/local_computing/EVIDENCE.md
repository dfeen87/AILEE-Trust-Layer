# AILEE v9.4.0 Local Computing final release evidence

**Acceptance environment:** Linux, Python 3.14.4, 2026-10-02. This report
records this checkout and this run only. It does not convert configured CI into
executed CI evidence.

## 1. Architecture and material documentation

Local Computing is the standalone, domain-independent governance boundary
between agentic callers and supported user-space OS interfaces. The OS/kernel
remains below and authoritative; there are no required native agents, kernel
components, remote federation, or cross-host trust. The final manuals are
`README.md`, `LINUX.md`, `WINDOWS.md`, and `MACOS.md` in this directory. This
acceptance pass also updated the repository architecture and release notes. No
platform-adapter defect was discovered. Prompt 3C corrected a common policy
configuration validation mismatch before final acceptance.

## 2. Trust, policy, replay, enforcement, and audit

An exact, validated `CapabilityRequest` binds the request ID, principal,
capability, classified resource target, execution context, and one of
`TRUSTED`, `DEGRADED`, `UNTRUSTED`, or `INVALID`. Deterministic policy is
default-deny: configured trusted authority may allow; degraded authority and
limited/observable support restrict; untrusted, invalid, unknown, malformed,
unsupported, and unavailable states deny or do not execute.

Requests are snapshotted, then valid IDs are atomically and permanently
reserved within that service instance before evaluation. Replay, mutation, and
concurrent duplicate tests confirm at-most-once adapter entry. Capacity
exhaustion fails closed rather than evicting IDs. Constraint tuples are
canonicalized. Policy construction applies the same non-empty, control-free,
1,024-character constraint contract as service-side outcome validation and
limits the canonical unique set to 256 entries. Retention is bounded and
in-memory, not restart-durable.

Platform support, policy decision, enforcement attempt, enforcement truth,
operation completion, and audit persistence remain distinct fields.
Contradictory or malformed adapter/policy evidence becomes explicit failure.
Audit construction/sink failure is explicit and does not rewrite a completed
side effect. Audit excludes native targets and payloads.

## 3. Deterministic scenario reconciliation

The established pytest harness was used rather than creating a second
simulation system. Each case exercises the conceptual chain from input through
audit. Labels below mean **NATIVE EXECUTION**, **SIMULATION/MOCKED LOGIC**, or
**NOT EXECUTED** in this acceptance environment.

| Scenario set | Evidence mode | Result |
|---|---|---|
| trusted configured request; explicit/default deny; restriction; degraded trust; unknown principal | SIMULATION/MOCKED LOGIC | passed |
| malformed/invalid request and target; unsupported/unavailable capability; unknown-platform fallback | SIMULATION/MOCKED LOGIC | passed fail-closed assertions |
| replayed ID; mutated request attempt; concurrent duplicate; bounded replay capacity | SIMULATION/MOCKED LOGIC | passed; adapter entered at most once |
| malformed/contradictory adapter evidence; capability mismatch; policy exception/evidence; audit failure | SIMULATION/MOCKED LOGIC | passed explicit-failure assertions |
| hostile argv/attributes, NULs, controls, invalid port/timeout, nonpositive PID | SIMULATION/MOCKED LOGIC | passed pre-adapter rejection assertions |
| Linux allowed read/write/delete/traverse; denied/restricted write; missing file; audit | NATIVE EXECUTION | passed with temporary resources |
| Linux direct argv child and identity | NATIVE EXECUTION | passed |
| Linux outbound loopback TCP | NATIVE EXECUTION | passed |
| Linux protected path/race, permission failure, process signal/PID lifecycle, DNS change | NOT EXECUTED | limitations tested/documented conceptually; no destructive or race claim |
| Windows/macOS imports, unavailable foreign adapter, distinct capability tables | SIMULATION/MOCKED LOGIC | passed; no foreign native call |
| Windows/macOS file, child, identity, TCP, process control | NOT EXECUTED | host OS unavailable |

Filesystem policy denial is verified before adapter entry. Directory traversal
is a native listing, not path containment. Protected-resource classification
does not grant access. Process execution uses direct argv without shell
interpolation. Process-control permission/nonexistent-target outcomes are
implemented, but were not induced locally. Network tests used loopback only;
no external destination was contacted.

## 4. Platform capability/evidence matrix

Categories are deliberately not collapsed. `COMPILED` for these Python modules
means byte-compilation/import validation, not a platform-native binary build.

| Platform / major paths | Implemented | Compiled/imported here | Unit tested | Simulated here | Native-runtime tested here | CI verified |
|---|---:|---:|---:|---:|---:|---:|
| Linux filesystem | yes | yes | yes | yes | yes | **NOT VERIFIED** |
| Linux process/subprocess | yes | yes | yes | yes | child creation: yes; process control: no | **NOT VERIFIED** |
| Linux outbound TCP | yes | yes | yes | yes | loopback: yes | **NOT VERIFIED** |
| Windows filesystem/process/TCP | yes | yes | common and truth-table logic | yes | **NOT EXECUTED** | **NOT VERIFIED** |
| macOS filesystem/process/TCP | yes | yes | common and truth-table logic | yes | **NOT EXECUTED** | **NOT VERIFIED** |

On every native OS, filesystem, subprocess, process control, and outbound TCP
are `SUPPORTED_WITH_LIMITATIONS`; they are never presented as unconditional
support. Linux resource/hardware and Windows/macOS privilege/resource/hardware
are `OBSERVABLE_ONLY`. Remaining unimplemented sensitive capabilities are
`UNAVAILABLE` as detailed in each manual.

## 5. Validation totals and builds

* Full Python suite: **329 passed, 1 skipped, 142 warnings**. The skip was the
  Windows/macOS-native test on Linux. Warnings are pre-existing pytest warnings
  for test functions returning values rather than asserting.
* Focused Local Computing suites: **59 passed, 1 skipped**.
* Python byte compilation and API-visible `9.4.0` check: passed.
* Rust formatting/check/test: passed; **32 tests passed** (25 unit + 7
  integration), with the example target containing zero tests.
* TypeScript typecheck/build/test: passed; **178 tests passed in 8 files**.
* CMake configure/build and CTest: passed; **1/1 test passed**.

Metadata inspection confirmed `9.4.0` in Python setup/API, Cargo, TypeScript
package/lock/export, CMake project, README, changelog, and the CI wheel smoke
expectation. Historical version records were left intact.

## 6. CI evidence

`.github/workflows/ci.yml` configures one workflow triggered by pushes and pull
requests: Python quality; Python tests on 3.9/3.10/3.11; native Local Computing
on Ubuntu/Windows/macOS; Python packaging; v9.3 governance validation; Rust;
TypeScript; and C++. This expands to 12 configured job executions per event.

GitHub CLI was installed but unauthenticated, and this checkout exposed no
usable run result. Consequently, **zero GitHub jobs are claimed executed by
this report**; Linux, Windows, and macOS CI results, failures, and skips are all
**NOT VERIFIED / not accessible**, not “green.” Local commands above are not
GitHub CI evidence.

## 7. Bugs, downgrades, limitations, and readiness

Prompt 3C reproduced a configuration-boundary defect: `Policy.validate()`
accepted restrictions that a matching `PolicyOutcome` validator later
rejected. The shared constraint contract now rejects invalid configuration
during `DeterministicPolicyEngine` construction, counts canonical unique
entries, and retains service validation as defense-in-depth. Boundary,
canonicalization, matching-outcome, and hostile-outcome regressions passed.
Capability truth was already bounded by `SUPPORTED_WITH_LIMITATIONS`,
`OBSERVABLE_ONLY`, or `UNAVAILABLE`; no further platform downgrade was
required.

Remaining boundaries include user-space pathname/executable TOCTOU, symlink or
reparse-point changes, DNS/policy non-binding and routing changes, PID reuse,
child/descendant behavior outside the boundary, native permissions and
privilege requirements, non-durable replay state, audit without rollback, and
no mediation of actions not submitted through AILEE. Windows and macOS lack
native local and accessible CI evidence. Repository-wide pre-existing pytest
warning debt remains; the CI file also documents repository-wide Black/mypy
debt outside its targeted type baseline. Packaging success may remain dependent
on platform/toolchain availability even though local language builds passed.

**Evidence-bounded readiness:** the common model and Linux native temporary
file, direct-child, identity, and loopback paths passed in this Linux
environment. Windows and macOS implementations are import/unit-simulation
verified here but require successful native runs before native-runtime or CI
verification can be claimed.
