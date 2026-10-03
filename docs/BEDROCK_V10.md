# AILEE v10.0.0 BEDROCK engineering report

## Release rationale and preserved architecture

Version 10 is a strict-SemVer major release because it strengthens malformed
input behavior and the Rust lineage contract. It does not redesign AILEE. The
existing model/system-output → safety → GRACE → consensus → fallback → audited
output pipeline remains intact, as do the independent Python domain governors,
Local Computing policy/enforcement boundary, Rust generative trust core,
TypeScript runtime, and C++ video temporal provenance engine.

The architecture was preserved where sound. The guarantees underneath it were
strengthened.

## Bottom-up invariant map

| Behavior | Required invariant | Previous enforcement/test | v10 enforcement |
|---|---|---|---|
| Python configuration | Thresholds/ratios are finite and bounded; windows and quorum are positive; bounds are ordered | Partial validation; several invalid domains were assumed | Construction rejects non-finite, out-of-range, contradictory, and invalid-cardinality configuration; parameterized regression tests cover boundaries |
| Python `process` | Untrusted numeric evidence is finite before it can influence trust or state | Types implied validity; NaN/infinity could enter audit data and rejected fallback history | Inputs are snapshotted and validated before any state access or mutation; malformed calls raise deterministically and preserve history, last-good value, and last result |
| Rust trust construction | Every trust dimension and aggregate is finite in `[0, 1]` | Finite values were clamped; `NaN` survived `f64::clamp` | Non-finite dimensions become zero, making malformed evidence fail closed |
| Rust thresholds | Invalid thresholds cannot authorize a result | Builders clamped finite values but retained NaN; direct threshold checks accepted values outside the documented domain | Non-finite builder thresholds become the most restrictive threshold; result checks require finite `[0, 1]` thresholds and scores |
| Rust lineage | Equivalent inputs produce the same hash and all represented output evidence is integrity-covered | Outputs were sorted, but request `HashMap` serialization was not canonical; latency/token/metadata changes were omitted | Request/output maps are sorted, complete output evidence is hashed, and public lineage summaries use deterministic order |
| C++ tests in CI | Registered assertions execute in Release builds | CTest ran the binary, but `NDEBUG` could compile out its `assert` checks | The test target explicitly undefines `NDEBUG` while the production library remains a Release build |
| Release metadata | One authoritative active version exists across runtimes | Values were manually aligned | A Python release-gate test checks Python, Rust, TypeScript, CMake, citation, README, and CI metadata |

## Defects confirmed and regression evidence

The audit confirmed three material weaknesses: incomplete Python numeric-domain
validation; Rust NaN propagation and incomplete/non-canonical lineage hashing;
and C++ Release tests whose assertions could be disabled. Regression tests were
first added for the Python failures and observed failing before correction.
Rust tests now cover non-finite dimensions and thresholds, insertion-order
independence, deterministic summary ordering, and metadata-sensitive hashes.

No partial mutation is performed by a malformed Python call. Ordinary trust
rejection still commits the selected fallback value to the trusted-stream
history, preserving the established fallback architecture; malformed input is
distinguished because it is not valid evidence from which to make a decision.

## Trust, integration, and compatibility contracts

Unknown or non-finite numeric evidence never becomes permission. Python rejects
it before scoring; Rust converts malformed trust dimensions to zero and rejects
invalid threshold checks. Canonical Rust lineage now connects request context,
complete model-output evidence, the selected output, and deterministic public
summaries under one contract rather than recomputing only a subset.

Public function signatures were preserved. Valid v9 configurations and runtime
inputs retain their behavior. Compatibility changes are deliberate:

1. malformed/non-finite Python configuration and runtime values now raise;
2. contradictory bounds and zero/negative windows or quorum are rejected;
3. Rust non-finite scores no longer survive as NaN;
4. existing Rust lineage hashes are not byte-compatible because v10 hashes a
   canonical and more complete evidence record.

## CI and release enforcement

CI continues to run the complete Python suite on supported Python versions,
native Local Computing tests on Linux/Windows/macOS, wheel installation, focused
governance tests, Rust formatting/check/build/tests, TypeScript typecheck/build/
tests, and the C++ Release build/CTest suite. Rust Clippy warnings now fail CI,
version consistency is a test, and C++ assertions are retained specifically for
its test executable.

## Risks investigated but not changed

- Fallback values are intentionally committed as the governed output stream;
  changing that would redesign historical stability semantics.
- Rust degraded consensus intentionally returns the best available output while
  marking consensus unachieved. Callers must inspect consensus metadata; v10
  does not silently reinterpret this documented result type as authorization.
- Local Computing replay state remains bounded, process-local, and non-durable,
  as documented in its v9.4 manuals. Durable distributed replay prevention is
  an external deployment responsibility.
- File-ledger durability and multi-process locking are not claimed. The
  in-process lock preserves the existing single-process boundary; production
  multi-process storage requires a transactional backend.
- Optional FEEN hardware and external model/provider integrations were not
  available for physical or live-service validation in this repository pass.

## Remaining external and real-world validation

Repository tests establish software behavior only. They are not simulation,
hardware, clinical, automotive, industrial, financial, security certification,
or production approval. Deployments must separately validate calibrated domain
bounds, model/provider behavior, host permissions, filesystem/process/network
races, persistent audit storage, hardware adapters, operational monitoring,
hazard analysis, and applicable regulatory or certification requirements.
