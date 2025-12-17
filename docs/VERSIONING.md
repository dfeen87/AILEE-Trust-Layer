# Versioning Policy

AILEE Trust Layer follows a **strict, trust-preserving versioning policy** designed
to ensure stability, predictability, and long-term usability in production systems.

This project treats version numbers as **contracts**, not marketing signals.

---

## Version Format

Versions follow semantic versioning:

MAJOR.MINOR.PATCH

Example:


---

## Major Versions (X.0.0)

A major version change indicates **breaking changes**.

This includes:
- Changes to core trust semantics
- Removal or renaming of public APIs
- Alteration of safety, grace, consensus, or fallback behavior
- Changes that invalidate prior audit assumptions

**Breaking changes will only occur in a new major version.**

---

## Minor Versions (1.X.0)

Minor versions introduce **additive, backward-compatible improvements**.

This may include:
- New optional modules
- New configuration presets
- Additional documentation
- New adapters or examples

Minor releases will **never alter existing behavior**.

---

## Patch Versions (1.0.X)

Patch versions are reserved for:
- Bug fixes
- Documentation corrections
- Clarifications or comments
- Internal refactoring with no behavioral change

Patch releases will **not change outputs for identical inputs**.

---

## Stability Guarantees (v1.x)

Within the v1 series, AILEE Trust Layer guarantees:

- Stable public APIs
- Stable trust semantics
- Deterministic behavior
- No silent changes
- No forced migrations

Users can upgrade within v1.x with confidence.

---

## Experimental Features

Any experimental or non-stable features will be:
- Clearly labeled
- Opt-in only
- Isolated from the core trust pipeline

Experimental features will **never** affect default behavior.

---

## Trust Commitment

AILEE Trust Layer prioritizes **predictability over velocity**.

Changes are made deliberately.
Stability is preserved intentionally.

Trust is built by honoring contracts over time.

---

## Summary

- v1.x is stable and safe for production
- Breaking changes require v2.0.0
- Additive improvements remain backward-compatible
- Documentation and clarity are first-class concerns

Versioning exists to protect users — not to signal activity.
