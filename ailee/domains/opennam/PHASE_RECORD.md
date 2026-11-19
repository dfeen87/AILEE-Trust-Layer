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
