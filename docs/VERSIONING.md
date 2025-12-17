# Versioning Policy

AILEE Trust Layer follows a **strict, trust-preserving versioning policy** designed to ensure stability, predictability, and long-term usability in production systems.

This project treats version numbers as **contracts**, not marketing signals.

---

## Version Format

Versions follow semantic versioning:

```
MAJOR.MINOR.PATCH
```

Example: `1.1.0`

---

## Major Versions (X.0.0)

A major version change indicates **breaking changes**.

This includes:
- Changes to core trust semantics
- Removal or renaming of public APIs
- Alteration of safety, grace, consensus, or fallback behavior
- Changes that invalidate prior audit assumptions
- Modifications to the AILEE equation or decision pipeline

**Breaking changes will only occur in a new major version.**

**Migration guides will be provided for all major version upgrades.**

---

## Minor Versions (1.X.0)

Minor versions introduce **additive, backward-compatible improvements**.

This may include:
- New optional modules (monitoring, serialization, replay)
- New configuration presets for additional domains
- Additional peer adapters
- Enhanced observability features
- New utility functions and convenience helpers
- Expanded documentation and examples

Minor releases will **never alter existing behavior**.

**All v1.0.0 code runs identically on v1.1.0+**

---

## Patch Versions (1.0.X)

Patch versions are reserved for:
- Bug fixes that restore intended behavior
- Documentation corrections and clarifications
- Performance optimizations with identical outputs
- Internal refactoring with no behavioral change
- Security patches

Patch releases will **not change outputs for identical inputs**.

---

## Stability Guarantees (v1.x)

Within the v1 series, AILEE Trust Layer guarantees:

✅ **Stable public APIs** — No function signature changes  
✅ **Stable trust semantics** — Identical inputs = identical trust decisions  
✅ **Deterministic behavior** — No randomness introduced  
✅ **No silent changes** — All changes documented in changelog  
✅ **No forced migrations** — Upgrades are opt-in and safe  

Users can upgrade within v1.x with confidence.

---

## Experimental Features

Any experimental or non-stable features will be:
- Clearly labeled with `EXPERIMENTAL` in docstrings
- Opt-in only (never imported by default)
- Isolated from the core trust pipeline
- Documented with stability warnings

Experimental features will **never** affect default behavior or core pipeline.

---

## Deprecation Policy

When features must be deprecated:

1. **Announce** in release notes with migration path
2. **Deprecate** in MINOR version with runtime warnings
3. **Remove** only in next MAJOR version (minimum 6 months later)

**No features will be removed without clear advance notice.**

---

## Trust Commitment

AILEE Trust Layer prioritizes **predictability over velocity**.

- Changes are made deliberately
- Stability is preserved intentionally
- Trust is built by honoring contracts over time
- Production systems deserve stable foundations

---

## Version History & Changelog

### v1.1.0 (January 17, 2025) — Production Arsenal Release

**Type:** Minor (Additive, Backward Compatible)

**New Features:**
- ✅ 17 domain-optimized configuration presets
- ✅ 6 advanced peer adapters for multi-model systems
- ✅ Enterprise monitoring with AlertingMonitor and PrometheusExporter
- ✅ Comprehensive serialization (audit logs, CSV, compact formats)
- ✅ Deterministic replay and regression testing utilities
- ✅ Enhanced package initialization with `create_pipeline()` convenience function

**Improvements:**
- Enhanced TrustMonitor with 10+ new metrics
- Added MetricSnapshot for point-in-time captures
- Improved documentation across all modules
- Added performance benchmarks

**Breaking Changes:** None

**Migration Required:** No

**Backward Compatibility:** 100% — All v1.0.0 code runs unchanged

---

### v1.0.0 (Initial Release)

**Type:** Major (Initial Stable Release)

**Core Features:**
- ✅ Multi-layered trust pipeline (Safety, Grace, Consensus, Fallback)
- ✅ Deterministic decision-making with full auditability
- ✅ GRACE Layer for borderline mediation
- ✅ Peer-based consensus without centralization
- ✅ Stability-preserving fallback mechanisms
- ✅ Complete audit schema and traceability
- ✅ 3 initial configuration presets (LLM, Sensor, Financial)
- ✅ Basic peer adapters (Static, Rolling)
- ✅ Core monitoring (TrustMonitor)
- ✅ Basic serialization (dict, JSON)

**Philosophy Established:**
- Trust is structure, not probability
- Decisions are earned, not assumed
- Explainability is mandatory
- Safety over convenience

---

## Upcoming Releases

### v1.2.0 (Planned)

**Tentative Features:**
- Streaming support for real-time pipelines
- Async adapters for high-throughput systems
- Time-series specific Grace policies
- Extended consensus protocols (Byzantine fault tolerance)
- Additional domain presets based on community feedback

**Status:** Under consideration  
**Timeline:** TBD based on production feedback

**The core architecture will remain stable.**

---

### v2.0.0 (Future)

A v2.0.0 release would only be considered if:
- Fundamental trust semantics require revision
- Core pipeline architecture needs restructuring
- Industry standards demand incompatible changes

**Current Status:** Not planned

**Commitment:** v1.x will be maintained for minimum 2 years from v2.0.0 release.

---

## Release Cycle

AILEE Trust Layer does not follow a fixed release schedule.

**Releases occur when:**
1. Sufficient value has been added (minor)
2. Critical bugs are fixed (patch)
3. Breaking changes are necessary (major, with extensive notice)

**Quality over cadence.**

---

## Security Updates

Security patches will be:
- Released immediately as PATCH versions
- Backported to all supported MAJOR versions
- Announced via GitHub Security Advisories
- Documented with CVE numbers when applicable

**Current security support:**
- v1.x: Full support
- v0.x: No longer supported (if any existed)

---

## Version Support Policy

| Version | Status | Support End Date |
|---------|--------|------------------|
| v1.1.x | **Current** | Until v2.0.0 + 2 years |
| v1.0.x | Supported | Until v1.2.0 + 6 months |
| v0.x | Not applicable | N/A |

---

## Community Contributions

Community contributions follow the same versioning rules:
- New features → Minor release
- Bug fixes → Patch release
- Breaking changes → Require maintainer approval and major version planning

All contributions must maintain backward compatibility within v1.x.

---

## Version Verification

Check your installed version:

```python
import ailee
print(ailee.__version__)  # "1.1.0"
print(ailee.get_info())   # Full package info
```

Or via command line:

```bash
pip show ailee-trust-layer
```

---

## Summary

- **v1.x is stable** and safe for production
- **Breaking changes require v2.0.0** (not planned)
- **Additive improvements** remain backward-compatible
- **Documentation and clarity** are first-class concerns
- **Security patches** are prioritized and immediate

**Versioning exists to protect users — not to signal activity.**

---

## Questions?

For version-specific questions or concerns:
- Open a [GitHub Discussion](https://github.com/dfeen87/ailee-trust-layer/discussions)
- Check [Release Notes](https://github.com/dfeen87/ailee-trust-layer/releases)
- Review [Migration Guides](docs/MIGRATION.md) (if applicable)

---

**AILEE Trust Layer**  
*Trust is a contract, not a promise.*

Last Updated: December 17, 2025
