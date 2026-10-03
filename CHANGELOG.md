# Changelog

### **v10.0.0 — BEDROCK Engineering Baseline**

**Type:** Major (strict-SemVer behavioral contract hardening)

- Preserved the layered trust pipeline, multi-runtime structure, Local
  Computing boundary, domain governors, and public method signatures.
- Made the Python pipeline's numeric contract explicit: configuration rejects
  non-finite, contradictory, out-of-domain, and invalid cardinality values.
- Made malformed runtime numbers fail before state mutation, preventing NaN or
  infinity from entering confidence calculations, fallback history, or audit
  metadata.
- Made Rust trust scores and confidence/threshold builders fail closed for
  non-finite inputs, and made result threshold checks reject invalid domains.
- Canonicalized Rust lineage across map insertion orders, sorted exposed
  summaries, and expanded the verification hash to cover output execution and
  model metadata.
- Added regression coverage for numeric boundaries, mutation atomicity,
  non-finite Rust evidence, canonical lineage, and metadata tampering.
- Ensured C++ assertions remain enabled in the Release test target, enforced a
  warning-free Rust Clippy baseline, and added a cross-runtime
  version-consistency release gate.
- Updated active Python, Rust, TypeScript, CMake, citation, documentation, and
  CI metadata to `10.0.0`; historical release evidence remains unchanged.

Compatibility note: method signatures and sound inputs remain compatible, but
previously accepted malformed configuration/runtime numbers are now rejected.
Rust lineage hashes intentionally change because they now use a canonical,
complete evidence representation. These behavioral changes warrant the major
increment under the repository's strict Semantic Versioning policy.

### **v9.4.0 — Local Computing Foundation**

#### BONUS — Executable Δv Mathematical Engine

- Added the propulsion-derived `ailee-delta-v/v1` reference implementation,
  immutable typed inputs/results, and deterministic composite trapezoidal integration.
- Added explicit validation and numerical-overflow failures without heavy numerical dependencies.
- Added compact workflow evidence integration that does not control trust or Local Computing authorization.
- Added analytic, regression, integration, mutation-safety, and workflow-independence tests plus an illustrative propulsion example.

- Renamed the public TypeScript Brooks Domain API and package tree to the vendor-neutral Air Flow Domain (`AirFlowDomain`, `AirFlowHardwareAdapter`, `AirFlowSafetyPolicy`, `DEFAULT_AIR_FLOW_POLICY`, and `AIR_FLOW_PRESETS`), without retaining legacy aliases or changing safety semantics.
- Converted the model-specific manifest identity into an explicitly repository-defined Reference Mass Flow Controller profile; its deterministic wire layout is preserved and is not presented as a universal MFC standard.
- Fixed the shared policy-constraint validator to reject whitespace-only restrictions during policy-engine construction and retained the same validation contract for fail-closed service-side outcomes.
- Established Local Computing as a foundational, domain-independent package.
- Added deterministic common trust, policy, capability, enforcement, error, and audit contracts.
- Added explicit Linux, Windows, and macOS integration locations, followed by the v9.4 user-space native adapter implementations.
- Aligned active Python, Rust, TypeScript, CMake, package, and CI release metadata to 9.4.0.
- Completed user-space native adapters for governed filesystem, subprocess, process-control, and outbound TCP actions, with OS-specific capability limits.
- Added fail-closed validation for contradictory platform results and native Linux, Windows, and macOS CI coverage.
- Bounded in-memory request-ID replay retention without eviction, fail-closed policy-result validation, canonicalized restrictions, and tightened audit-identifier validation.
- Completed final acceptance documentation with separate Linux, Windows, and macOS operating manuals, deterministic scenario reconciliation, and an evidence matrix that does not treat simulation or configured CI as native verification.
- Final local verification exercised Linux filesystem, direct-child, identity, loopback TCP, replay, adversarial, audit, and failure paths; Windows/macOS native and GitHub CI status remain explicitly unverified where run evidence was unavailable.
- Documented user-space path/executable races, DNS binding and routing limits, PID lifecycle, host permission requirements, non-durable replay state, and observable-only/unavailable capabilities as release limitations.
- Corrected policy construction to reject outcome-incompatible constraints at configuration time while retaining fail-closed service validation; the 256-entry limit applies to canonical unique constraints and each constraint is bounded to 1,024 control-free characters.
- Integrated Local Computing into the root README immediately after the domain use cases and hardened cross-links, platform-evidence language, replay semantics, and user-space limitations across the v9.4 manuals.

### **v9.1.1 — Version Alignment**

**Type:** Patch (Bug fixes & maintenance)

#### Consistency
- Updated the public Python, Rust, and TypeScript package versions to 9.1.1
- Aligned governance metadata, documentation, dashboard labels, manifests, lockfiles, and tests with the 9.1.1 release

---

### **v9.1.0 — Version Alignment**

**Type:** Minor (Additive, backward-compatible)

#### Consistency
- Updated the public Python, Rust, and TypeScript package versions to 9.1.0
- Aligned governance metadata, documentation, dashboard labels, manifests, lockfiles, and tests with the 9.1.0 release

---

### **v4.7.0 — Memory Domain Hardening & Benchmark Coverage**

**Type:** Minor (Additive, backward-compatible)

#### Memory Domain
- Hardened `MemoryGovernor.evaluate()` with optional pre-flight signal validation
- Added `MemoryPolicy.__post_init__` validation for critical policy fields
- Replaced string-literal `SafetyStatus` comparisons with proper enum references
- Fixed `fallback_reason` field to always carry an explicit `str` value (`.value`)
- Updated `_determine_trust_level` return type to use `Tuple` from `typing` for Python 3.8+ compatibility
- Added `from __future__ import annotations` for clean forward-reference handling
- Created `ailee/domains/memory/BENCHMARK.md` with simulated performance and control quality metrics

#### Consistency
- Updated version strings to 4.7.0 across all Python packages, Rust crate, markdown docs, and deployment configs

---

### **v4.3.0 — Topology Domain: Full Uplift**

The Topology domain has been elevated to full production standard, completing the interior work that 4.2.0 began. The domain now runs all governance decisions through the AILEE Trust Pipeline across five real control domains — node connectivity, trust relationships, deployment graph, structural integrity, and route reliability — each with its own tuned configuration, rate limiting, and domain-aware validation. Accompanying this release are production documentation and validated benchmarks, bringing the Topology domain to full parity with the Datacenters reference implementation in implementation quality, documentation depth, and benchmark coverage.

---

### v4.2.0 — Security Hardening & Robustness (March 2026)

**Type:** Minor (Additive, backward-compatible)

#### Security
- Hardened CORS configuration with environment-based origin control
- Added query length limits to prevent resource exhaustion
- Restricted HTTP methods to GET-only for API endpoints
- Pinned dependency versions to prevent supply-chain drift
- Added input sanitization for search and model generation queries
- Clamped confidence extraction to valid [0, 100] range

#### Robustness
- Fixed frozen dataclass mutation pattern in FEEN backend fallback
- Added `__post_init__` validation to `AileeConfig` for all critical fields
- Added fallback_mode enum validation
- Added confidence weight sum validation
- Added request timeout (60s) to frontend fetch calls
- Added localStorage session eviction (max 50 sessions)
- Replaced `unwrap()` with `unwrap_or_default()` in Rust lineage timestamp
- Added input clamping to Rust `TrustScore::new()`
- Replaced O(n) `Vec::remove(0)` with `VecDeque::pop_front()` in Rust scorer

#### Architecture
- Created `ARCHITECTURE.md` documenting repository layout
- Made `AileeBackend` protocol `runtime_checkable`
- Exported `AileeBackend` and `BackendCapabilities` from backends package

#### Testing
- Added `tests/test_pipeline_smoke.py` with 5 smoke tests
- CI now validates core pipeline behavior on every push

#### Consistency
- Unified version strings across Python, Rust, and deployment configs
- Updated deprecated FastAPI patterns (`regex` → `pattern`, event handlers → lifespan)

---

### v2.2.0 Fixes & Stability

- **Auditory domain:** Repaired structural duplication and method scoping issues that could
  lead to incorrect governance behavior under edge conditions.
- Hardened safety gating, uncertainty aggregation, and precautionary penalty handling.
- Finalized production-grade auditory governance logic with consistent event logging
  and decision explainability.
- **Auditory BENCHMARKS.md**

---

### v1.9.0 — Clarified trust-boundary documentation and fixed a minor metadata typo. No behavioral changes.

---

### v1.8.0 Validation & Assurance Roadmap

The **Governance** and **Cross-Ecosystem** domains intentionally do not include
traditional performance benchmarks.

These domains are **normative and safety-critical**, and are evaluated by
**deterministic invariants, guarantees, and restraint** rather than throughput,
latency, or optimization metrics.

- The **Governance** domain is validated through authority enforcement,
  jurisdictional scope containment, temporal correctness, delegation safety,
  and deterministic decision outcomes.
- The **Cross-Ecosystem** domain is validated through semantic fidelity,
  consent preservation, capability alignment, and safe continuity across
  incompatible platforms.

Formal documentation files (e.g., `ASSURANCE.md`, `INVARIANTS.md`) for both
domains are **actively being developed** and will be introduced once real-world
usage patterns and adversarial scenarios meaningfully inform their structure.

This approach is intentional and preserves architectural correctness while
avoiding premature or misleading evaluation artifacts.

---

### AILEE Trust Layer — v1.4.0

**Release Type:** Minor (Domain Expansion & Packaging)  
**Status:** Production / Stable

### Added
- **IMAGING domain** governance layer  
  - `domains/imaging/imaging.py` — production-grade imaging trust and QA governance  
  - `domains/imaging/__init__.py` — domain exports  
  - `domains/imaging/IMAGING.md` — imaging domain conceptual framework
- **Python packaging support**
  - `setup.py` — minimal, production-safe package configuration for installation and distribution

### Updated
- `README.md` — added IMAGING domain overview
- Root `__init__.py` — exposed IMAGING domain (non-invasive, optional)

### Notes
- No changes to the AILEE core trust pipeline
- No breaking API changes
- Existing deployments remain fully compatible

---

### Documentation Note (v1.3.0)

Documentation clarification:
Automotive and Power Grid governance domains were conceptually part of the AILEE architecture in v1.3.0 but were not yet accompanied by standalone domain documentation at release time.

Additionally, the Data Center documentation file was relocated to a domain-consistent path (domains/datacenter/) to align with the evolving domain structure.

These were documentation-only oversights. No behavioral, API, or governance logic changes were introduced.
