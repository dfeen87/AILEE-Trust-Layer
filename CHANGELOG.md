### v4.1.0 — Security Hardening & Robustness (March 2026)

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
