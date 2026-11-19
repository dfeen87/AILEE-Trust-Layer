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

## Phase 4 — deterministic reproduction and final boundary audit

- Added checkout-only, stdlib reproduction runner and a frozen 17-file
  source, testing, methodology, corpus, labeling and observed-report manifest.
  Manifest SHA-256 `b87a85b86e3556c0117cf242179fb7e21daa5df013879f00764c81c462ce69cc`.
  The runner compares exact artifact bytes before reconstruction, reproduces
  the complete case-level output, requires all authored exact labels, returns
  nonzero on mismatch, and emits no raw rejected metadata. The frozen report
  digest remains `a07b17b90153e36432f21a44e122cd7225679e68c904a9910fbe97f17b80f2e9`.
- Python 3.11.14, 3.12.12, 3.13.5 and 3.14.6 checkout runner from outside the
  checkout, each with `PYTHONHASHSEED=0,7,31` (12 executions): exact same
  report digest. This proves deterministic replay only in these tested
  environments, not scientific reproducibility or all-platform compatibility.
- Local Python 3.11.14 OpenNAM tests: **125 passed**; independent Python
  3.12.12 OpenNAM tests: **125 passed**. Full Python 3.11.14 project suite with
  native socket test permissions: **1140 passed, 3 existing skips, 68 existing
  warnings**. Python 3.9 grammar parsing of nine new Python files and project
  `compileall` for `ailee`, `tests`, `scripts` passed. Actual Python 3.9/3.10
  runtime is covered by the existing hosted CI matrix, not executed locally.
- Offline local wheel build for unchanged package version `10.2.0`, clean
  environment install with `--no-index --no-deps`, isolated installed-package
  opt-in import and ready/omission checks passed. It was not published; the
  reproduction fixture corpus remains checkout-only.
- Independently compared all 346 tracked baseline blobs against the working
  tree; all unchanged. The separately organized user checkout stayed clean
  on its original branch. Existing release metadata, root package exports,
  CI/dependency files, Health Safety domain and other tracked baseline files
  remain untouched.
- Independent engineering/scientific-boundary review covered exact scope and
  roster, digest binding, age/chronology, mutable-input handling, source-group
  counting, audit scope correlation, frozen case and label fingerprints,
  byte replay and scientific framing. The Phase 2 audit-binding correction
  distinguishes equivalent-ready scopes; no blocking findings remained.
  No independent scientific or regulatory validation was performed. Manifest
  scope is explicitly limited to selected files under a trusted checkout;
  it is not a signed dependency lock or attestation, parent-package modules
  reside outside it, and optional additional malformed-manifest checks are
  future defensive improvements. No claim extends beyond synthetic metadata
  conformance on the tested finite cases.
- Prior hosted Phase 3 push/PR checks succeeded (runs 44606076685 and
  44606092068); final Phase 4 hosted checks must run against the final commit.
  No release, merge, deployment, live integration, external action, scientific
  prediction or animal replacement has been authorized by this work.

## Post-Phase 4, follow-up 1 — bounded synthetic adapter foundation

- Commit `404315b5bc8e4e70399b3596c1393f1cd7322d89` adds only the optional
  `adapters` module, its dedicated tests and experiment-scope documentation.
  Two explicit revision-1 parser formats (`JSON_ENVELOPE` and `CSV_ROW`) admit
  already-categorized fictional metadata within independently selected exact
  method/endpoint/protocol/unit/material/context profiles and catalog reference.
  No executable adapter identifiers, dynamic modules, normalization of raw
  scientific values, unit conversion, scope fallback or trust annotation
  creation. New formats require explicit parser/mapping additions and review.
- One bounded nonempty envelope and one opaque artifact byte string per packet;
  1–32 packets and an aggregate 1 MiB cap checked before decoding. Actual
  artifact-byte SHA-256 must match the declared checksum; duplicate evidence
  or artifact digests reject the entire batch. This detects inconsistent bytes,
  not semantic correspondence, completeness, origin, scientific validity or
  re-encoded duplication. Distinct bytes cannot establish independent studies.
- Canonical untrusted `EvidenceRequest` plus ordered descriptive receipts bind
  parser/profile versions, exact scope/catalog descriptor hash, source envelope
  bytes, opaque artifact bytes and record metadata SHA-256. Receipts are not
  authorization, approved-adapter attestation or an authenticity credential.
  The unchanged governor consumes only the request with its independent catalog,
  policy and clock; unknown controls and incomplete rosters remain abstentions.
- Local Python 3.11.14: **70 adapter tests passed**, including strict format and
  context closure, actual byte mismatch, duplicate/group handling, bounds,
  typed snapshot mutation, forged receipt limitations and independent-governor
  readiness/abstention under two distinct fictional profiles. Original Phase 4
  reproduction passed unchanged. Hosted push/PR checks succeeded (runs
  44608947177 and 44608955162). Neither transport is an actual assay adapter.

## Post-Phase 4, follow-up 2 — reproducibility hardening and final audit

- Archived original 17-file manifest verbatim as `reproducibility-v1.json`
  (SHA-256 `b87a85b86e3556c0117cf242179fb7e21daa5df013879f00764c81c462ce69cc`).
  Current revision 2 checks 21 selected files including the archive, adapters,
  their documentation/tests and prior selected sources. The revision 2 manifest
  SHA-256 is `618117ecd32da90307e838c2814390ff1b27fd53fbf8dc426385a637d3869aae`.
  Requires bounded strict UTF-8 manifest parsing, duplicate-key rejection,
  exact revision/phase/path/digest schema, ≤1 MiB streaming hash limit per
  selected file, and pins original cases/labels/report fingerprints to the
  historical v1 archive. A regenerated v2 cannot rebaseline those assets.
- No change to the existing core contracts, governor, original corpus harness,
  40 authored cases, fixed expectations or original per-case observations.
  Behavioral replay remains the original 40-case evaluation; adapter behavior
  is covered by the dedicated adapter suite. Runner default output SHA-256
  remains `a07b17b90153e36432f21a44e122cd7225679e68c904a9910fbe97f17b80f2e9`.
  Twelve v2 replay executions outside the checkout on Python 3.11.14, 3.12.12,
  3.13.5 and 3.14.6, each hash seeds 0, 7 and 31, produced that exact digest.
- Local Python 3.11.14: **228 cumulative OpenNAM tests passed**; independent
  Python 3.12.12: **228 passed**. Full Python 3.11.14 project suite with native
  socket test permissions: **1243 passed, 3 existing skips, 68 existing
  warnings**. Python 3.9 grammar parsing of 11 new Python files, `compileall`,
  and `git diff --check` passed; no claim of local 3.9 runtime validation.
- Offline wheel build at unchanged `10.2.0`, fresh environment installation
  with `--no-index --no-deps`, isolated installed-package adapter/core import,
  independently catalog-governed readiness, omission abstention, byte-tamper
  rejection, receipt-as-request invalidation and `pip check` passed. No package
  publication. Independent reviewers reproduced 228 tests and 21-file replay,
  confirmed the preserved fingerprints and found no remaining blockers within
  the explicitly synthetic engineering boundary.
- All 346 baseline tracked blobs exactly unchanged; original core contracts,
  governor, harness and three corpus files match the original Phase 4 commit.
  Existing organized checkout remained clean on its original branch. The
  package version, root exports, CI/dependencies, Health Safety and other
  domains remain unchanged. These follow-ups support further separately
  qualified adapter opportunities, not real method implementations, scientific
  comparability, biophysical inference, validated performance, certification,
  approved release or workflow integration. Final hosted checks must run on the
  follow-up head; no merge, version bump or release is included here.
