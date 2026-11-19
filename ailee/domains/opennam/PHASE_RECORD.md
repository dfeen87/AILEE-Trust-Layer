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

## Phase 3 — authored evaluation and adversarial conformance

- Wrote and separately froze 40 original synthetic scenarios and manually
  assigned outcome/reason expectations before measuring the corpus; retained
  the complete canonical observed per-case audit and comparator results.
- `cases.json` SHA-256 `244b3102f50e8714d6f07640c68d8a98d1875e8a600dbf245891910a3201f0d2`;
  `expected.json` SHA-256 `8579732148981d64405f9ec6f147133b86ee3f177ba82fd296b86d9885691330`;
  `observed.json` SHA-256 `a07b17b90153e36432f21a44e122cd7225679e68c904a9910fbe97f17b80f2e9`.
- Exact outcome/reason matches: **40/40**; 6 expected ready and 34 expected
  not ready, zero false ready and zero false not-ready in this authored suite.
  Accept-all false ready 34/34; abstain-all false not-ready 6/6; metadata-presence
  false ready 32/34. These are intentionally weak engineering baselines.
- Local verification, Python 3.11.14: 115 cumulative OpenNAM tests passed.
  No real scientific datasets, ground truth, blind or held-out evaluation,
  or validated performance claim. See corpus provenance and benchmarks.
