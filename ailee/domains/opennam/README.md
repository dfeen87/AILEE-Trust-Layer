# OpenNAM research metadata domain (experimental)

This optional offline domain explores **engineering metadata completeness for
expert review** of original, fictitious non-animal-study records. It neither
interprets a biological assay nor predicts a material's hazard, potency,
toxicity, safety, therapeutic properties or regulatory acceptance. It does not
implement an OECD test guideline, defined approach, biological classifier,
animal-replacement protocol, model, or validated scientific workflow. No real
materials, organisms, measurements or confidential study artifacts are in scope.

## Phase 1 — contracts

Baseline `4ccf4b8e2e50ed3c055afebeedc669ffbf6c89cf` (`main`, version 10.2.0)
includes the existing opt-in Health Safety experiment. Like that domain, this
package uses isolated frozen records, strict categorical metadata, explicit
clock input and separate trusted configuration, with no global registration,
runtime adapter, root exports, dependencies, persistence, network, ledger,
learning or external actions. The referenced OpenNAM v0.1.0 prototype was not
available for inspection; this is a new independent research scaffold, not
verified prototype code, a migration or a reproduced result.

- `Scope` describes exact fictitious material, endpoint, method, protocol
  version, unit and context IDs. Exact string equality does not establish
  actual scientific comparability or independence.
- `EvidenceRecord` carries synthetic ID, scope, UTC observation time,
  categorical observation, and declared artifact SHA-256. Its metadata digest
  binds all fields using sorted compact ASCII JSON and SHA-256. It does not
  read, verify or attest an external artifact.
- `EvidenceRequest` is entirely untrusted and supports strict dict or bounded
  bytes JSON ingestion. Only these fields are admitted, with schema `1.0`,
  explicit catalog reference and at most 32 distinct evidence IDs.
- `EvidenceCatalog` is independently assembled by a trusted offline harness.
  Up to 256 unique entries bind evidence ID, exact scope and metadata digest to synthetic
  review time, declared source-group identity and separate reviewed reporting,
  positive and negative control, and applicability flags. A typed catalog is
  not authentication; it must never be assembled from the request itself.
- `ReviewPolicy` pins that catalog version, exact allowed scopes, a group
  counting requirement (2–8), and whole-day maximum age (1–3650). These are
  engineering fixture thresholds, not scientific decision thresholds.

IDs use lowercase ASCII `syn-` names, up to 63 characters total; digests are
64 lowercase hex characters. UTC times must be exact whole seconds (JSON
`YYYY-MM-DDTHH:MM:SSZ`). No coercions, extra keys, duplicate IDs or catalog
keys, arbitrary prose, quantitative values or raw data. JSON is UTF-8 bytes
only, ≤32 KiB, rejects duplicate object keys at every level, floats, NaN,
infinities and out-of-range integers. Canonical records, scopes and catalog
entries are ordered, and nested data is frozen. These syntactic constraints
do not guarantee data is fictitious; the harness must enforce the scope.

Phase 2 will add a deterministic governor with outcomes `INVALID`, `ABSTAIN`,
and `REVIEW_READY`. `REVIEW_READY` will mean only that the supplied synthetic
metadata met the configured engineering gate; never safety, regulatory
approval, recommended action or scientific validity. Both positive and
negative observations can qualify. Uncertainty, missing reviews, stale records,
scope mismatch, contradictory comparable records or insufficient declared
source groups must not qualify. All records in the requested exact scope in
the independently selected catalog must be accounted for: omitting a reviewed
record must not turn a conflicting result into a ready one. Coverage is bounded
to that supplied catalog, not proof of completeness of literature or evidence.
Phase 1 defines contracts only, not a gate.
