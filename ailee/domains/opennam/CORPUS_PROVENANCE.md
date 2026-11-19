# OpenNAM synthetic fixture provenance

## Origin and boundary

This phase uses **40 original fictional engineering cases** authored for this
research scaffold. All material, method, endpoint, protocol, unit, evidence,
source-group and context IDs are invented; all categorical observations and
review flags are invented. Repeated hexadecimal digits represent **declared**
artifact digests for imaginary metadata references. No artifact bytes exist or
are verified. There are no real chemical identifiers, organisms, experimental
measurements, protocols, study subjects, providers, downloaded datasets,
private data, retrospective annotations or prototype-derived cases. Nothing
is sampled from a scientific population. The fixtures are test assets in the
same repository under its MIT license.

The original OpenNAM v0.1.0 prototype could not be inspected, so this corpus
does not reconstruct, rerun or benchmark it. No adaptation from external data.

## Freeze and chronology

The fixture definitions and provisional engineering outcome/reason labels were
written separately from the observation-report generator, after Phase 2 and
before running the Phase 3 corpus. They are designed to exercise the specified
gate and were **not blind, held-out, third-party or scientifically adjudicated**.
The implementer sent their premeasurement digests for independent comparison
within the project team. Local file creation timestamps (not trusted external
timestamps): corpus and expectations `2026-11-19T15:07:40-05:00`;
frozen observations `2026-11-19T15:08:49-05:00`. Parent phase commit:
`5bcc0d4b6088ce118b68ba0487b6af78ab59439a`.

| File (`tests/fixtures/opennam/`) | SHA-256 of exact bytes |
| --- | --- |
| `cases.json` | `244b3102f50e8714d6f07640c68d8a98d1875e8a600dbf245891910a3201f0d2` |
| `expected.json` | `8579732148981d64405f9ec6f147133b86ee3f177ba82fd296b86d9885691330` |
| `observed.json` | `a07b17b90153e36432f21a44e122cd7225679e68c904a9910fbe97f17b80f2e9` |

The trusted fixture manifest separately specifies review-catalog membership
and quality/group labels; the evaluator harness assembles the catalog first.
The untrusted request selects from the fictional record definitions separately
and may be incomplete, conflicting, altered or malformed. Request-selection
or mutation never updates the independent catalog. This is synthetic harness
separation, **not** proof of real independent review, source independence,
provenance or scientific applicability. Labels in `expected.json` are fixed
engineering expectations, not toxicological ground truth.

## Scientific context, not dependencies or implemented methods

Metadata reporting, reliability, relevance, applicability, applicability domain,
reproducibility, and practical qualification are distinct problems. Useful
authoritative context for understanding that distinction:

- OECD, [Defined Approaches on Skin Sensitisation (TG 497)](https://doi.org/10.1787/b92879a4-en),
  2026 edition. This domain does not implement any of its test methods,
  interpretation procedures or decision rules.
- OECD, [Guidance Document for Describing Non-Guideline In Vitro Test Methods](https://doi.org/10.1787/9789264274730-en).
  Completeness of reporting helps reconstruction but does not establish validity.
- OECD, [Guidance Document on Good In Vitro Method Practices (GIVIMP)](https://doi.org/10.1787/9789264304796-en).
  This work does not implement laboratory operations or assess GIVIMP compliance.

No scientific evaluation data is taken from these references. No regulatory,
scientific-validation, hazard classification, animal-replacement, performance,
or safety claim follows from this prototype's metadata gates.
