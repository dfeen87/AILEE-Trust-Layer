# OpenNAM research phase record (append only)

## Phase 1 — contracts and scope

- Base: `4ccf4b8e2e50ed3c055afebeedc669ffbf6c89cf` on `main`, version 10.2.0.
- Deliverables: opt-in immutable synthetic contracts, closed bounded dict and
  JSON ingress, separate reviewed catalog and policy, canonical metadata SHA-256,
  contract tests and explicit non-scientific scope boundary.
- No operational governor, scientific predictions, prototype reconstruction,
  biological measurements, real datasets, CI/dependency/release changes or
  existing-domain edits in this phase.
- Local verification, Python 3.11.14: `python -m pytest -q tests/test_opennam_contracts.py`
  — 37 passed. Contract conformance only; not scientific validation.

## Phase 2 — deterministic evidence metadata governor

- Implemented the opt-in governor with independent immutable copied snapshots,
  nested exact-type revalidation, explicit clock, exact-scope catalog coverage,
  metadata digest binding, QC/reporting/applicability, chronology, conflicting
  or inconclusive categorical tokens, duplicate artifacts and distinct-group
  checks. No raw biological data, model, quantitative inference or integration.
- Stable reason codes, canonical audit fingerprints of the policy/catalog and
  only fully bound synthetic request, and a minimized detached audit projection.
- Local verification, Python 3.11.14: `python -m pytest -q
  tests/test_opennam_contracts.py tests/test_opennam_governor.py` — 74 passed.
  This demonstrates engineered behavior on original fixtures, not scientific
  validity, actual source independence, toxicological interpretation or safety.
