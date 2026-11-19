# AHS evaluation-corpus provenance

The Phase 4 benchmark uses the exact Phase 3 freeze
[`92379d20205c72a5dd6cc5638d3650e3db036d48`](https://github.com/dfeen87/AILEE-Trust-Layer/commit/92379d20205c72a5dd6cc5638d3650e3db036d48).
All case artifacts, scope identifiers, claims, source-group assignments,
consent/urgency facts and privacy canaries are intentionally synthetic.
No externally sourced health records, real patient information or actual
clinical advice was introduced.

## Partition register

| Partition | Origin/version | Cases | Label/review status |
| --- | --- | ---: | --- |
| Phase 1/2 development | `ahs-phase1-synthetic-v1`, schema 1.0; repository-authored contract examples under Don Michael Feeney Jr.'s project direction, frozen in Phase 1 | 49 | Separate provisional engineering labels written for the contract; operationalized without relabeling in Phase 2; no independent adjudication |
| Visible reservation (development file) | Four preexisting scenario ideas in original corpus | 4 | `AWAITING_INDEPENDENT_ADJUDICATION`; no case outcome labels, not executed or scored |
| Phase 3 defensive validation | `ahs-phase3-defensive-v1`, schema 1.0; original tool-assisted engineering fixtures authored under the same repository owner's Phase 3 assignment | 62 | Separate expected decision/support/artifact/auxiliary labels fixed and hashed before measurement against the Phase 1 contract; no independent adjudication |
| Genuinely held-out independent partition | Not available | 0 | Not created, not labeled, not evaluated |

The Phase 1/2 set and Phase 3 set were both visible to the implementation
effort. The authored expectations measure engineering conformance only;
they are **not** blinded reference standards, clinically validated
annotations, independent medical evidence, or estimates of independent
effectiveness. There is no separate independent reviewer count, agreement
study or adjudication procedure completed for either scored partition.

### Original development corpus

- Origin: Phase 1 contract-first specification in commit
  `43e86da5c7a5413064178a9cafca40205281b63f`. The original scenario and
  expectation bytes were retained in Phase 2, commit
  `66ab477857509b3a3e271e529d14cd88ac58f504`, and Phase 3.
- Versioned trusted synthetic policy `policy-ahs-phase1`/1, catalog
  `catalog-ahs-synthetic`/1, review ID `review-ahs-phase1-spec`, stage
  `PROVISIONAL_ENGINEERING`. Fixed explicit fixture time
  `2026-01-15T12:00:00Z` or controlled absent/invalid clock.
- Labels specify structural checks, candidate outcome, controlled primary
  reason, fixed public support and exact artifact-permission reference. The
  historical `PENDING_PHASE_2` metadata remains as frozen authorship context,
  not a statement that the actual Phase 2 tests were unexecuted later.
- Limits: small curated examples written around documented invariants,
  uneven class composition, known to the implementer, no independent
  source verification or medical adjudication.

### Phase 3 defensive validation corpus

- Origin: new fixed conformance examples, ten original inert artifact
  profiles, nine specified validation categories, separate request and
  trusted-context profile selections. Neither the original corpus nor the
  governor was modified.
- Policy `policy-ahs-phase3`/1, catalog `catalog-ahs-phase3`/1, review ID
  `review-ahs-phase3-spec`, stage `PROVISIONAL_ENGINEERING`. Fixed UTC
  `2026-06-02T12:00:00Z` or explicit unavailable trusted context.
- Label procedure: write dedicated reference outcome/reason/support and
  exact permitted references, plus applicable Boolean integrity, privacy,
  controlled decoder, prior eligibility and replay labels before the first
  new case execution. Preserve the separately measured observation file
  and the exact freeze. No label revision after measurement.
- Limits: same known contract and project-directed authoring, finite
  developer-facing controls and invalid-shape examples, honest offline
  harness; not a blind holdout, not an attack simulation or clinical study.

## Exact fixture checksums

The following files are located at `tests/fixtures/health_safety/`.
Checksums are SHA-256 of exact bytes, including the final newline.

| Artifact | SHA-256 |
| --- | --- |
| [scenarios.json](../../../tests/fixtures/health_safety/scenarios.json) | `ced7860b3e25f6f158092575029bb07fd88e26043337042ab1061162aca8e8ee` |
| [expected.json](../../../tests/fixtures/health_safety/expected.json) | `b824a06427781d0deab5779c466999c17f5d95cf65e751d4977297b85757733e` |
| [phase3_scenarios.json](../../../tests/fixtures/health_safety/phase3_scenarios.json) | `3f4bfc1c1950a411e34cf495f0b6e6d4169eaa042e526c5604f0e35ed9ed8a1d` |
| [phase3_expected.json](../../../tests/fixtures/health_safety/phase3_expected.json) | `a5368d7ecf8563a3868549e3d8851686d137c0faaca2458b38049275adc30f0d` |
| [phase3_observed.json](../../../tests/fixtures/health_safety/phase3_observed.json) — observations, not labels | `b75202fc5ec804e32f0040cdca68970a38a7dd044fce69c7f65b8d08615e50b1` |

The four named reserved cases retain exactly their original status in
`expected.json`. They are visible, unlabelled and excluded from Phase 4
metrics; reserving them is not independent holdout construction.

## Licensing and redistribution status

The repository's [MIT license](../../../LICENSE) applies to its licensed
software and associated documentation. Both corpus metadata records
explicitly retain **PENDING_REVIEW** for corpus licensing/redistribution.
No independent corpus-license or contributor-rights review has occurred;
do not infer separate corpus-release clearance from the software license.
The Phase 4 work reuses the original synthetic fixtures inside the
existing draft research repository only; no separate dataset publication
or external patient-data authorization is claimed.

An independent review should establish provenance of label and source-group
assignments, reviewer criteria, disagreement/adjudication rules and
redistribution terms, then define a genuinely separated synthetic holdout
before any broader claims. Clinical validation would require a distinct,
appropriately governed study; this experiment is not such a study.
