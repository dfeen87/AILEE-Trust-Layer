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

## Phase 2 — deterministic metadata governor

`EvidenceGovernor(policy=..., catalog=...).evaluate(request,
evaluation_time=...)` accepts a typed request, strict dict or bounded JSON
bytes. No default policy, catalog, evaluation time or permissive fallback.
The harness must separately establish trusted snapshots; the constructor
validates and copies nested records, and evaluation revalidates exact typed
structures including nested fields. There is no isolation from arbitrary
code execution inside the same Python process.

The gate has fixed precedence: configuration and time, request structure,
catalog/scope compatibility, nonempty full exact-scope roster coverage,
metadata binding, timestamp and reviewed quality prerequisites, then
categorical agreement, unique artifact digests and minimum distinct declared
source groups. The timestamps must satisfy `observed <= reviewed <= evaluation`
and age `<= max_age_days` (inclusive). Bad/unknown reporting, either control,
or applicability abstains. An abstract inconclusive result or a mixture of
positive and negative abstract tokens in the exact scope also abstains. This
is not an assay interpreter or a hazard classification. Mixed scope is never
compared as contradiction; even exact scope does not establish actual
comparability. One artifact copied under multiple IDs disqualifies the set;
multiple records from the same declared source group count only once. Group
identity is a reviewed catalog annotation, not proof of independence.

`INVALID` indicates malformed request/configuration or unavailable clock;
`ABSTAIN` indicates a structurally valid request blocked by evidence criteria;
`REVIEW_READY` with `METADATA_COMPLETE` indicates only that these configured
synthetic metadata checks passed. The output has stable controlled reasons,
explicit UTC time and canonical policy/catalog SHA-256 fingerprints. Once the
entire exact roster is bound, it also contains the normalized synthetic request
digest; malformed, unbound and partial requests never receive that digest.
Digests identify chosen snapshot content (including catalog annotations) and
bound metadata, not authentication; the harness must control provenance and
versioning. Digests are not encryption or a substitute for privacy controls.
The minimized `decision.audit` projection includes those fields only, no IDs,
raw request content, observation tokens or individual artifact digests. No audit persistence is
implemented. Exact-scope coverage is limited to the supplied catalog; a scope
with more than 32 entries cannot pass this bounded gate. The engine does not
discover studies, review raw artifacts, infer uncertainty, assess actual method
reliability or authorize any action.

## Phase 3 — original synthetic engineering cases

See [benchmarks](BENCHMARKS.md) for the finite conformance comparison and
[corpus provenance](CORPUS_PROVENANCE.md) for the frozen original fixture
identities and limits. The gate matched 40/40 manually assigned engineering
outcome/reason labels, not scientific classifications. No real scientific data,
animal replacement claim, performance validation or prototype reproduction.
