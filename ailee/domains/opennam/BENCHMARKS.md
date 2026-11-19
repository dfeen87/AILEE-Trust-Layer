# OpenNAM Phase 3 — synthetic engineering conformance

This is a finite deterministic conformance exercise, not a biological or
scientific benchmark. See [corpus provenance](CORPUS_PROVENANCE.md) for origin,
freeze chronology and SHA-256 fingerprints.

## Method

The test harness reconstructs each of 40 synthetic fixture cases with an
independently selected catalog and policy, an explicit clock, and a separate
untrusted request or malformed input. It evaluates the real governor, compares
outcome and controlled reason set against the fixed provisional labels, and
records the actual minimized decision audit and comparator readiness for every
case. Six fixtures are expected engineering-ready, 34 not ready. `ABSTAIN` and
`INVALID` both mean not ready for the binary comparator; only the governor's
exact outcome and reason tuple are assessed in the three-outcome comparison.

Comparators deliberately omit gates:

- Accept all: every fixture ready, regardless of content.
- Abstain all: no fixture ready, regardless of content.
- Metadata presence: top-level required keys present and at least two rows
  with required keys present, permissive JSON decoding, no value or semantic
  validation, authority, exact-scope coverage, time, catalog checks, QC,
  source-group deduplication or token consistency.

All comparators and the gate are compared with the same labels. They are not
scientific baseline models, endorsed decision approaches, or external systems.

## Measured result (Python 3.11.14)

| Engine | Expected ready, observed ready | Expected not ready, observed not ready | False ready | False not ready |
| --- | ---: | ---: | ---: | ---: |
| OpenNAM governor | 6 | 34 | 0 | 0 |
| Accept all | 6 | 0 | 34 | 0 |
| Abstain all | 0 | 34 | 0 | 6 |
| Metadata presence | 6 | 2 | 32 | 0 |

The governor matched **40/40** fixed exact outcome-and-reason labels on the
authored suite. Additional metamorphic checks permute rows and serialize
dicts as JSON, compare repeated evaluations, and enforce byte-for-byte
canonical report reconstruction. The case-level audit and comparator results
are in `tests/fixtures/opennam/observed.json`. There is no tuning, sampling,
uncertainty estimate, confidence score, threshold calibration, scientific
validation, held-out test, statistical generalization, or evidence regarding
actual study quality or a real material. These results do not establish
superiority over any scientific assessment, efficacy, safety or suitability
for regulatory decision making.
