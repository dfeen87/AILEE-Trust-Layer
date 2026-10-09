# AILEE v10.0.1 post-BEDROCK architectural review

## Baseline and scope

The starting checkout was clean on branch `work` at
`bc3f274062c237814d9cc920b3031696dde917d0`, the merged BEDROCK PR #93.
The review branch is `codex/post-bedrock-hardening`. Architecture, versioning,
BEDROCK implementation and regressions, changelog, and CI were inspected.
No additional AGENTS.md instructions were found in applicable workspace paths.

BEDROCK already validates Python configuration and numeric ingress before
state mutation, bounds Rust score construction and invalid threshold builders,
canonicalizes represented Rust lineage evidence, retains C++ Release assertions,
and gates aligned active release metadata. Consensus `SKIPPED` routing and
governed fallback-history updates are intentional. Local Computing remains
user-space governance within host permissions, with process-local replay and
separate policy, capability, enforcement, completion, and audit evidence.

Three findings were selected in one hardening iteration. Direct authorization
failures and incorrect quorum evidence ranked above conditional extreme-number
and specialized integration risks. Secondary reviews followed the corrected
boundaries; the three-finding limit was retained.

## Confirmed findings and corrections

### 1. High: malformed governance numeric evidence could authorize

Ownership: `ailee/domains/governance/governance.py`,
`GovernanceGovernor.evaluate`, `_validate_temporal`, and numeric configuration.

An otherwise valid strict signal with certified authority, recognized mandate,
consent and jurisdiction, `valid_from=1`, `valid_until=2`, and `timestamp=10000`
is expired. Substituting NaN for the timestamp or expiry produced `FULL_TRUST`,
`actionable=True`, and `temporal_status=VALID`, and committed that decision.
Zero expiry bypassed the truthiness check. Negative/NaN delegation depth could
likewise bypass the positive-depth rejection gate despite failed delegation
evidence. NaN clock grace defeated expiry comparisons through configuration.

The violated invariant is that malformed or contradictory evidence cannot earn
authorization or enter a successful decision history. The public boundary now
rejects non-finite temporal evidence, booleans, contradictory bounds, and invalid
delegation cardinality before decision/history mutation. Relevant numeric policy
settings are validated at construction and before evaluation. Zero time bounds
and issuance times are evaluated; default-expiry overflow fails before commit.
Exact finite numeric values are preserved to avoid rounding authorization
boundaries or changing valid audit identifiers. Ordinary clock grace,
registered delegation, and missing-required-bounds denial remain intact.

The initial focused numeric regressions produced 52 failures and 2 passes
before correction. Additional precision/compatibility checks verify the final
implementation rather than treating float conversion as harmless validation.

### 2. High: unknown scope did not stop strict authorization

Ownership: `GovernanceGovernor._evaluate_signal` and `_validate_scope` in the
same file. A valid strict signal with `jurisdiction=None`, an empty jurisdiction,
or an unrecognized jurisdiction returned `SCOPE_UNKNOWN` but continued to
`FULL_TRUST`, `actionable=True`, and successful history evidence. The gate only
denied `OUT_OF_SCOPE`.

Scope enforcement now denies both `OUT_OF_SCOPE` and `SCOPE_UNKNOWN`. The
existing decision type records `NO_TRUST` and `actionable=False`, and every
history view agrees with that denial. Explicitly disabled scope enforcement and
optional missing jurisdiction preserve their existing behavior. This restores
the configured requirement; it does not add jurisdiction discovery or external
credential verification. Three scope regressions failed before correction.

### 3. High evidence integrity: Rust ignored its configured model quorum

Ownership: `src/consensus.rs`, `ConsensusEngine::reach_consensus`.
`with_min_models(3)` with one threshold-eligible output reported
`consensus_achieved=true` under all four strategies. `min_models` was assigned
but never consulted. Callers relying on that evidence could mistake inadequate
participation for achieved consensus.

The engine now requires the configured count of threshold-eligible outputs
before entering an achieved-consensus strategy. Under-quorum calls retain the
existing degraded selection and score, report consensus unachieved, and give
available/required counts without falsely claiming the score was below its
threshold. Default/zero minimum normalization, empty/low evidence, and
successful quorum behavior are preserved. The baseline regression failed;
five final tests cover all strategies, exact/excess quorum, missing/low/NaN
scores, output/score mismatches, and default behavior.

## Progressive review and final reassessment

Numeric authorization failures led to review of policy mutation, default expiry,
delegation, exact comparisons, and all three governance history views. Scope
failures led to checks of optional and disabled enforcement and short-circuit
status representation. The quorum correction led to eligibility filtering and
the distinction between a high degraded score, achieved consensus, and actual
authorization. No failed selected operation commits a partial decision.

The Python core was traced through safety, GRACE, consensus, fallback, commit,
and audit serialization. Related arithmetic/evidence paths were inspected in
TypeScript, Rust lineage, C++ provenance, Local Computing and the math engine.
Runtime APIs were compared only where they share a contract. Independent diff
review checked precision, valid-input compatibility, metadata, and regression
test strength. No consensus `SKIPPED`, fallback-history, lineage encoding,
platform capability, or mathematical-permission contract was redefined.

## Validation

All commands ran in the repository root unless otherwise stated, using the
configured Python 3.11 and Rust toolchains. The final complete Python suite
passed 492 tests with one Windows/macOS native-platform skip and 68 existing
warnings. Focused governance validation passed 91 tests. Compileall, pip check,
the established three-file mypy gate, and the isolated distribution build passed.

- Python: `python -m pytest tests -q`; targeted
  `python -m pytest tests/test_governance_integrity.py tests/test_governance_domain.py -q`;
  `python -m compileall -q ailee tests`; `python -m pip check`;
  `python -m mypy --follow-imports=skip ailee/domains/licensing/licensing.py ailee/domains/industrial/industrial.py ailee/domains/dual_domain.py`.
- Packaging: `python -m build --outdir /workspace/work/python-dist`, followed
  by wheel installation and version/behavior checks in a clean virtual environment
  outside the repository — passed installed imports, version, exact temporal
  comparisons, strict scope denial, malformed-evidence rejection and history
  preservation checks.
- Rust: `cargo fmt --all -- --check`;
  `cargo clippy --all-targets --all-features --locked -- -D warnings`;
  `cargo check --all-targets --locked`; `cargo build --all-targets --locked`;
  `cargo test --all-targets --locked` — 40 tests passed.
- TypeScript, in `packages/ailee-ts`: `npm run typecheck`; `npm run build`;
  `npm test` — 178 tests passed in eight files.
- C++: `cmake -S . -B /workspace/work/cpp-final-release -DCMAKE_BUILD_TYPE=Release`;
  `cmake --build /workspace/work/cpp-final-release --config Release --parallel 2`;
  `ctest --test-dir /workspace/work/cpp-final-release --build-config Release --output-on-failure`
  — one registered test passed; assertions remain enabled for its Release target.

The syscall sandbox stalled an existing in-process FastAPI test and denied
native loopback fixtures. The FastAPI test passed outside that sandbox; full
Python/native Linux validation uses that mode. A sandboxed isolated packaging
attempt could not connect to the configured proxy; the isolated build passed
with the proxy available. These environment failures were not relaxed tests.
Windows/macOS native evidence, other Python versions, live providers and
physical hardware were not established locally. The existing CI matrix and
release gates remain intact; configured CI is not a claim of completed CI.

## Release classification

`10.0.1` is PATCH: required scope evidence, finite numeric authorization evidence,
and a configured consensus quorum were already intended contracts. The fixes
restore those checks, preserve valid exact arithmetic and public signatures,
and add no unrelated capabilities or dependencies. Active Python, Rust,
TypeScript, CMake, citation, governance metadata, documentation and CI versions
are aligned. Historical BEDROCK/changelog evidence and lineage hashes are intact.

## Verified unresolved risks and intentional limits

The three-finding cap leaves concrete follow-up work; this is not a complete
safety certification:

- Python fallback clamps can override hard bounds when their ranges conflict:
  hard range `[0,1]` plus fallback minimum `2` returns and commits `2`. An even
  median of two finite `1e308` history values can return and commit infinity.
- TypeScript finite-input arithmetic can return an accepted infinite weighted
  result or accepted NaN trust scores through peer/history mean overflow;
  fallback midpoint arithmetic can also overflow. Its exported `TPEFFIBridge`
  retains caller-owned frame evidence, allowing mutation after ingestion to
  produce accepted NaN metrics. Focused failing tests/reproductions were retained
  outside the proposed diff.
- Rust scoring mutates history while iterating unordered maps, so equivalent
  fresh batches can select different outputs. Public/deserialized aggregate
  scores above one bypass constructor bounds and can satisfy consensus/result
  threshold checks. Non-finite confidence values encode as JSON null, so NaN
  and infinity can collide in the lineage hash. These were reproduced rather
  than inferred from passing tests.
- Local Computing skips a callable audit sink whose boolean value is false and
  reports successful audit without delivery; a callable empty-list collector
  demonstrated this with a deterministic adapter double. No real operation was
  used to reproduce it.
- Existing governance hierarchy/delegation relationships are simplified;
  submitted mandate and consent strings are not cryptographically or externally
  verified credentials. Domain signals, software provenance and mathematical
  evidence do not independently establish permission or physical authenticity.
- Host permissions, user-space resource races, non-durable process-local replay,
  external audit durability, provider behavior, physical hardware and production
  safety require separate deployment validation. Shared mutable core pipeline
  instances also require consumer coordination; this patch adds no concurrency
  or distributed-state guarantee.

Only the three selected weaknesses were corrected. Remaining evidence should
guide a separately scoped review rather than expanding this patch speculatively.

## Changed files

| Purpose | Files |
|---|---|
| Implementation | `ailee/domains/governance/governance.py`, `src/consensus.rs` |
| Adversarial regressions | `tests/test_governance_integrity.py`, `tests/rust_consensus_quorum.rs` |
| Documentation | `CHANGELOG.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/VERSIONING.md`, `docs/POST_BEDROCK_V10_0_1.md` |
| Active release metadata and its checks | `.github/workflows/ci.yml`, `CITATION.cff`, `CMakeLists.txt`, `Cargo.toml`, `Cargo.lock`, `setup.py`, `ailee/__init__.py`, `ailee/governance_v1/approval/approval.py`, `ailee/governance_v1/compartments/registry.py`, `ailee/governance_v1/ledger/ledger.py`, `ailee/governance_v1/schemas.py`, `packages/ailee-ts/package.json`, `packages/ailee-ts/package-lock.json`, `packages/ailee-ts/src/index.ts`, `packages/ailee-ts/tests/governance.test.ts`, `tests/test_governance_v1.py`, `tests/test_release_metadata.py` |
