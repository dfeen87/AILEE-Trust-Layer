# AILEE Health Safety (AHS) — experimental domain

**Phase 1 of 4: contracts and synthetic specifications only.**

**Design down. Prove up. Govern the answer. Preserve the help.**

This opt-in, offline Python package defines the data and authority boundaries
for a future categorical candidate-release governor. It has **no operational
governor, release adapter, clinical reasoning, or support-template renderer**.
No Phase 1 function authorizes an answer. In particular, constructing an
ALLOW-shaped dataclass for a test does not grant authority.

Only original synthetic fixtures are in scope. No real patient data, medical
records, patient identifiers, diagnosis, prescribing, treatment selection,
clinical triage, medical action, health service, external tool execution,
network access, telemetry, or efficacy claims are supported.

## 1. Verified architectural baseline

- Repository: `dfeen87/AILEE-Trust-Layer`.
- Baseline `main`: [`a001af813ce9416989540b6af05bfb0eb9f7bad2`](https://github.com/dfeen87/AILEE-Trust-Layer/commit/a001af813ce9416989540b6af05bfb0eb9f7bad2).
- [PR #98](https://github.com/dfeen87/AILEE-Trust-Layer/pull/98) is merged.
  The published [v10.1.1 — TRUST CONSISTENCY release](https://github.com/dfeen87/AILEE-Trust-Layer/releases/tag/v10.1.1)
  is neither a draft nor a prerelease; its tag points to the exact baseline above.
  [Hosted CI on that exact SHA](https://github.com/dfeen87/AILEE-Trust-Layer/actions/runs/44439076143)
  completed successfully (12/12 jobs, including the Python 3.9/3.10/3.11 matrix).
- Work proceeds on `feat/health-safety-domain`, in one shared draft PR across
  all four phases. The existing release metadata is unchanged.

The [Phase 0 audit disposition](../../../docs/PHASE_0_5_AUDIT_DISPOSITION.md)
is the architectural starting point. The audited domains retain their public
behavior. Review of the verified baseline informs these choices:

| Existing domain / audit boundary | AHS design decision |
| --- | --- |
| [Industrial](../industrial/industrial.py): typed records, explicit evaluation time, validity reasons, read-only boundary | Use explicit typed outcomes and times; no telemetry, process state, ledger, or nested mutable metadata. |
| [Licensing](../licensing/licensing.py): deterministic ordered checks; an upstream status is not authentication | Separate declared references from independently established authority; no caller-supplied credential or clock as proof. |
| [Cybersecurity](../cybersecurity/cybersecurity.py): structural provenance checks, controlled denial with unavailable time | Adopt typed failure and optional audit timestamps; never treat the default verifier, token, issuer label, signature format, or evidence presence as Health authority. |
| [Neuro-assistive](../neuro_assistive/neuro_assistive.py): consent precedence and alignment of decisions with effective constraints | Require independently scoped consent for candidate release. Do not import its state tracker, numeric pipeline, historical fallback, or event history. Generic public support is a separate nonclinical channel. |
| Deferred `/trust`, Governance v1, FEEN, ledger, and training-dispatch findings | Preserve isolation. No use of numeric confidence, generic approval, shared query state, provider diagnostics, persisted ledger, training callbacks, or fallback values as candidate authorization. |

The package follows the existing domain-specific `__init__.py` and frozen
dataclass conventions. `setup.py` discovers packages via `find_packages`;
there is no top-level registration, wrapper, new mandatory dependency, or
change to any existing domain.

## 2. Frozen API and trust boundaries

The following public API is specified for **Phase 2, not implemented here**:

~~~python
# Planned interface only; this class does not exist in Phase 1.
class HealthSafetyGovernor:
    def __init__(self, *, policy: Optional[HealthPolicy],
                 catalog: Optional[HealthCatalog]) -> None: ...

    def evaluate(self, request: Union[HealthRequest, dict], *,
                 context: HealthContext) -> HealthDecision: ...
~~~

The constructor will receive explicit, immutable harness snapshots and never
choose permissive defaults. A trusted assembly failure can be represented by
`None` for policy or catalog so evaluation still returns a controlled denial
and the independently determined support disposition. The evaluator will
accept an exact typed `HealthRequest` or an already decoded built-in `dict`
using the closed schema; other values are malformed. It must verify the
typed object's contract rather than treating construction as authority.

| Contract | Fields and origin |
| --- | --- |
| `HealthPolicy` | **Trusted configuration**: `PolicyRef(policy_id, version)`, pinned `CatalogRef(catalog_id, version)`, explicitly allowed safe purposes, integer minimum independent sources per claim. No setting enables prohibited purposes or reduces the minimum below two. An empty allowed-purpose tuple denies all candidates. |
| `HealthRequest` | **Entirely untrusted**: `schema_version`, declared `FixtureScope(fixture_id, turn_id)`, catalog reference, `ArtifactRef(artifact_id, sha256)`, declared purpose, claim IDs, `(claim_id, evidence_id)` references, closed synthetic input markers. No free text, raw payloads, source weights, confidence scores, consent, authority, clock, urgency, policy, or catalog contents. `from_dict` checks structure only. |
| `HealthContext` | **Independent harness only**: `scope`, `evaluation_time`, `consent`, separately established `urgency`, and `ContextState` (`ASSEMBLED` or `INVALID`). Neither request/model text nor matching request-provided IDs establishes these facts. `None` explicitly represents unavailable time, consent, or urgency; an unavailable scope requires `INVALID`. `INVALID` denies candidate release even if separately trusted support facts survive. |
| `SyntheticConsent` | **Independent harness only**: exact scope, catalog, artifact ID + digest and purpose; explicit `GRANTED`, `DENIED`, or `REVOKED` state as of evaluation; validity window. This is synthetic permission to consider that artifact only, not consent of a real patient or authorization for a medical action. |
| `UrgencyFact` | **Independent harness only**: explicit scope and `UrgencyLevel` (`ORDINARY`, `QUALIFIED_REVIEW`, `IMMEDIATE_PUBLIC_REFERRAL`). No inference from symptoms, health prose, keyword detection, numeric trust, or candidate disposition. |
| `HealthCatalog` | **Independently selected synthetic manifest**: schema version, catalog identity/version, review ID and explicit review stage, frozen artifact annotations and evidence rows. See section 3. |
| `HealthDecision` | **Future trusted evaluator output only**: outcome, exactly one controlled primary reason in `reason_codes`, optional single exact permitted `ArtifactRef`, separate support disposition and fixed public template ID, policy/catalog references, trusted scope and evaluation time. Unknown trusted metadata is `None` only on denial. |
| `HealthAuditRecord` | Frozen projection via `decision.audit`: controlled outcome, reason, support/template ID, policy/catalog references, trusted scope and evaluation time. No request data, candidate/evidence references, consent details, or rejected-content digest. No ledger or persistence. |

All contracts are frozen dataclasses of immutable standard-library types
(tuples, strings, integers, explicit enums, UTC datetimes, and other frozen
records). Type checks and identifier syntax are **not authentication**.
The trusted harness owns the actual bootstrap and assembly boundary. It must
not deserialize request-supplied data into `HealthContext`, `SyntheticConsent`,
`UrgencyFact`, `HealthPolicy`, or `HealthCatalog`, or accept a request-supplied
`HealthDecision` as an authorization token. This is an offline experiment
inside an honest harness, not isolation from arbitrary Python code running in
the same process.

### Schema and time rules

- Request and catalog schema version: exactly `"1.0"`. A different schema
  string gives `UNSUPPORTED_SCHEMA`; other malformed values give
  `MALFORMED_SCHEMA`. `HealthRequest.from_dict` accepts exact built-in dicts
  with precisely the documented fields, exact nested keys and bounded lists.
  Extra keys (including injected policy/consent/urgency or an arbitrary text
  field) are rejected. It does not parse JSON or infer intent. A future
  ingress must bound encoded JSON to 16 KiB and reject duplicate keys
  **before** constructing a dict; this transport adapter is not Phase 1.
- IDs are at most 64 lowercase ASCII alphanumeric characters with internal
  `.`, `_`, `-`, ending alphanumeric and with the appropriate nonempty
  namespace: `policy-`, `catalog-`, `artifact-`, `fixture-`, `turn-`,
  `claim-`, `evidence-`, `source-`, `review-`. Versions are exact integers in
  `[1, 2147483647]`; bool is not an integer version. Digests are exactly 64
  lowercase hex characters; no case folding or coercion.
- Max 16 distinct claims, 64 evidence references (duplicates retained), and
  eight distinct input markers from the closed `SyntheticInput` enum. Empty
  claim/evidence lists are structurally valid but cannot authorize an artifact
  requiring them. Policy's minimum independent groups is 2–8 per claim.
  Catalog maxima: 128 artifacts, 512 evidence rows, 64 evidence rows per
  artifact. The catalog must have at least one artifact.
- Claims, input markers, allowed purposes, evidence-reference multisets and
  manifest rows are canonically sorted by stable identifiers or enum values.
  Duplicate claims, markers, purposes, artifact IDs and catalog evidence IDs
  are invalid; repeated request evidence refs remain a sorted multiset.
  No unordered iteration, process-random hash, clock or ambient environment
  may affect a future decision.
- Typed constructors require exact built-in `datetime` with `timezone.utc`,
  whole seconds and `fold=0`. No default timezone, wall clock, arbitrary
  timezone provider, normalization, fractional timestamps, or implicit
  conversion. Fixture strings use `YYYY-MM-DDTHH:MM:SSZ`. Validity windows
  require `valid_from < valid_until`; checks use
  `valid_from <= evaluation_time < valid_until`. Invalid/unavailable time
  is a denial, never a default to "now"; the audit time is absent (`None`).
- Unknown enum values, wrong types, mutable lists within typed records,
  malformed identifiers, noncanonical timestamps, conflicting catalog links
  and out-of-bounds counts are structural errors. A valid shape with a revoked,
  expired, or mismatched consent fact remains valid *structure* to be denied
  by the future evaluator. Structure does not grant authority.

When the harness encounters an invalid clock it must retain an independently
established scope/urgency and supply `evaluation_time=None`; this preserves
the referral pathway. For malformed consent, urgency or another context
component, it must discard the malformed values and construct an `INVALID`
snapshot, retaining only independently verified facts. Do not downgrade
malformed urgency to `None` in an otherwise assembled context in a way that
could authorize a candidate. An unknown/unreliable scope is `None` and
`INVALID`; no urgency can be considered bound in that case. These are explicit
failure representations, not bypasses. If the entire context is an arbitrary
untrusted object, it supplies no reliable fact at all.

## 3. Reviewed synthetic catalog

`HealthCatalog` is a versioned, immutable manifest provided by the harness.
The Phase 1 catalog has `review_stage=PROVISIONAL_ENGINEERING` and a named
review revision. The reserved `INDEPENDENT_SYNTHETIC_ADJUDICATION` enum is for
later genuinely independent synthetic label review, not clinical validation;
the label is not self-authenticating. The fixture annotations are authored
engineering expectations, not assertions of clinical truth.

- `ReviewedArtifact`: exact `ArtifactRef`, `FixtureScope`, `HealthPurpose`,
  `ArtifactAssessment`, full claim set and closed input-marker set. The only
  candidate purposes potentially eligible are `GENERAL_EDUCATION` and
  `UNCERTAINTY_AND_REFERRAL`, and the assessment must explicitly be `ELIGIBLE`.
  Diagnosis, treatment selection, prescribing, triage, external action, and
  privacy/authority overreach can never be eligible. Other assessment values
  represent unresolved review, unsupported certainty, or explicit
  prohibitions. A malformed manifest labeling a prohibited purpose/marker
  or a claimless candidate as eligible is invalid.
- `ReviewedEvidence`: unique ID, exact artifact reference (including digest),
  scope, purpose, claim ID, independently assigned `source_group_id`,
  `EvidenceRelation` (`SUPPORTS`, `CONTRADICTS`, `UNRESOLVED`) and UTC validity
  window. It must refer to a known artifact and one of its claims with all
  bindings equal. Each row inherits the exact catalog ID/version; a request
  cannot add, relabel, or reassign a row or source group.
- The exact reviewed artifact bytes are kept separately by the trusted
  synthetic fixture harness (`artifact_payloads` in the fixture file).
  `sha256` covers unchanged UTF-8 bytes, not a paraphrase or normalized
  string. Catalog identity/version, review revision, and immutable payload
  store must be pinned together by the harness. Changing a payload,
  annotation, group assignment, or catalog entry requires a new reviewed
  catalog revision/version. A digest verifies identity of bytes only, not
  medical correctness, authenticity of a catalog author, or clinical safety.

The complete matching catalog claim/evidence set is authoritative. Merely
presenting two evidence IDs or two differently worded citations does not
prove independent support. All correlated entries remain in the same
source group. Future evaluation checks the *whole* matching manifest set,
not only favorable references presented by a request, and counts distinct
supporting `source_group_id` values **per claim**. A repeated reference, two
IDs from the same group, an unknown reference, or a different turn cannot
manufacture support. A known contradiction cannot be hidden by omitting its
reference or waiting for its validity window to expire; only a new reviewed
catalog revision can change that annotation.

## 4. Candidate-release precedence (Phase 2 specification)

| Candidate outcome | Authorization |
| --- | --- |
| `ALLOW` | Authorizes release of exactly one synthetic, explicitly eligible, catalog-reviewed artifact ID **and digest**, only after affirmative checks of every claim, complete evidence, scope, consent and policy and the exact-release boundary recheck. No medical or external action authority. |
| `REVIEW` | Withhold the candidate and its identifier from the release field. Structurally valid but unresolved, incomplete, contradictory, stale or unsupported evidence/review. Review itself does not later authorize release. |
| `DENY` | Withhold the candidate; return a controlled denial reason. Malformed schema, prohibited medical/action request, privacy or authority violation, binding/consent breach, invalid context/configuration, or evaluation failure. |

**DENY > REVIEW > ALLOW.** No score, fallback, majority, urgency, model
statement, or evidence count overrides a deny gate. Specified v1 returns
exactly one primary code (a single-element `reason_codes` tuple). The
following ordered checks determine it; the first applicable gate wins.
All applicable deny gates must be cleared before considering any review
gate. Only validated data may be inspected at each gate.

1. Independently establish the support disposition (section 5); do not use it
   for candidate authorization. Validate candidate type/schema; return
   `MALFORMED_SCHEMA` or `UNSUPPORTED_SCHEMA` on decoder/contract rejection.
2. Reject non-harness/malformed/`INVALID` context or missing reliable scope
   (`INVALID_CONTEXT`), then invalid/unavailable time (`INVALID_TIMESTAMP`).
3. Validate trusted policy (`INVALID_POLICY`) and trusted catalog
   (`INVALID_CATALOG`), including exact pinned catalog reference. A malformed
   catalog entry or duplicate/dangling ID invalidates the entire snapshot;
   do not fall back to a previous catalog. An unavailable assembly uses
   explicit `None` and the corresponding controlled denial reason.
4. Safely look up the declared artifact ID in the trusted catalog. Deny the
   following in order: privacy marker or manifest privacy prohibition
   (`PRIVACY_VIOLATION`); claimed authority marker or manifest authority
   prohibition (`AUTHORITY_VIOLATION`); any prohibited purpose, prohibited
   action marker or manifest medical/action prohibition
   (`PROHIBITED_MEDICAL_ACTION`). These checks only use closed enum markers
   and trusted annotations, not an interpretation of medical text. Omitting a
   marker cannot suppress a manifest prohibition.
5. All bindings must match (`BINDING_MISMATCH` for known disagreement):
   request scope equals context scope; a supplied urgency fact is scoped to
   that same context; declared catalog equals the policy-pinned catalog; a
   known artifact ID matches its exact digest, scope, purpose and marker set;
   no extra/foreign claim IDs; each evidence reference names a declared claim
   and any known row has the exact claim, artifact, purpose, scope and catalog.
   A known row from another artifact or turn is a binding violation, even if
   the artifact bytes or claim description are identical. Unknown IDs are
   unresolved, not affirmative support; a known mismatch takes precedence.
6. The remaining safe purpose must be in the explicit policy
   (`POLICY_PURPOSE_NOT_ALLOWED`).
7. Consent is a separate deny gate, in order: missing (`CONSENT_REQUIRED`);
   any scope/catalog/artifact/purpose mismatch (`CONSENT_MISMATCH`);
   revoked (`CONSENT_REVOKED`); denied (`CONSENT_DENIED`); not yet valid
   (`CONSENT_NOT_YET_VALID`); expired (`CONSENT_EXPIRED`). Only a current
   independently granted fact for the exact candidate meets the gate.
   Consent never overrides a prohibition or substitutes for evidence.
8. If no deny applies, withhold with the first applicable `REVIEW` reason
   in this order:
   - `CANDIDATE_NOT_REVIEWED`: candidate absent from the manifest or its
     assessment is unresolved.
   - `UNSUPPORTED_CERTAINTY`: the trusted assessment identifies unresolved
     certainty even if supporting annotations exist.
   - `CONTRADICTORY_EVIDENCE`: any matching manifest contradiction, even if
     not supplied by the request.
   - `MISSING_CLAIMS`: a proper subset of the reviewed claim set.
   - `MISSING_EVIDENCE`: any required matching catalog row omitted, or a
     required claim has no evidence row.
   - `UNRESOLVED_EVIDENCE`: unknown evidence reference or any matching row
     with an unresolved relation.
   - `STALE_EVIDENCE`: any matching row outside its half-open validity window.
   - `INSUFFICIENT_INDEPENDENT_SUPPORT`: at least one claim has fewer
     distinct current supporting source groups than the policy minimum.
9. Only an explicitly `ELIGIBLE` artifact with every preceding check passed
   may return `ALLOW`, reason `SYNTHETIC_ARTIFACT_ELIGIBLE` and the exact
   **catalog-owned** `ArtifactRef`. `REVIEW` and `DENY` always carry
   `permitted_artifact=None`.

Unexpected evaluator/assembly exceptions must fail closed as
`EVALUATOR_FAILURE` (or the controlled structural/configuration reason when
known), never return a partial ALLOW or echo exception text. A later review
requires a new reviewed snapshot and fresh evaluation, not an implicit
approval, override button, or previously returned decision.

### Exact release binding

Only a future trusted in-process consumer of an evaluator-produced decision
may consider release. It must verify the decision's origin; bind it to the
same policy, catalog version, fixture/turn, and permitted artifact; retrieve
that one immutable reviewed payload by ID; and verify the exact SHA-256 before
release. A mismatch, absent artifact, time/turn change or assembly failure
withholds it. Never release the original model text, append rejected material,
rewrite the reviewed artifact, release multiple artifacts, or treat a
hand-constructed `HealthDecision` as proof. These consumer checks are
specified here but not implemented or exercised in Phase 1. No ALLOW can
authorize diagnosis, treatment, prescribing, clinical triage, external tool
execution, or any clinical action.

## 5. Independent public support pathway

The support disposition is **not** a modification of the candidate outcome.
It is determined only by a separately established, reliable harness urgency
fact matching the independently known fixture and turn.

| Independently established urgency | Support disposition | Fixed public template ID |
| --- | --- | --- |
| Ordinary, absent, or unreliable/incorrectly scoped | `ORDINARY_HELP` | `ahs.public.ordinary-help.v1` |
| Qualified review, correctly scoped | `QUALIFIED_REVIEW` | `ahs.public.qualified-review.v1` |
| Immediate public referral, correctly scoped | `IMMEDIATE_PUBLIC_REFERRAL` | `ahs.public.immediate-referral.v1` |

No inference from the request, rejected content, urgency-shaped untrusted
markers, confidence or scores is permitted. Use a prevalidated support
snapshot before candidate checks so failures in request decoding, clock,
consent, policy/catalog assembly or evaluator execution cannot suppress a
reliable public referral. An `INVALID` context may carry an independently
validated scope and urgency; that support survives, while the candidate is
denied. If the scope or urgency is unreliable/unavailable, use the fixed
ordinary-help template, not invented urgency. No consent is required to make
these fixed, nonpersonal public template IDs available; no data is disclosed.

Example specification (not execution):

| Inert synthetic case | Candidate | Support |
| --- | --- | --- |
| Prohibited treatment-selection placeholder; no urgency | `DENY` | `ORDINARY_HELP` |
| Same prohibited placeholder; independent trusted immediate urgency | `DENY` | `IMMEDIATE_PUBLIC_REFERRAL` |
| Malformed candidate; no reliable urgency | `DENY` | `ORDINARY_HELP` |
| Malformed candidate; independent trusted immediate urgency | `DENY` | `IMMEDIATE_PUBLIC_REFERRAL` |

Phase 1 defines IDs and checks the legal disposition/template pairs only.
No public template body or clinical guidance is emitted; later rendering may
use only fixed, reviewed public help/referral wording, with no personalization,
private content, automated contact, location inference, symptom triage,
external API, or tool action. A `QUALIFIED_REVIEW` support ID does not mean
the candidate is approved by a qualified person.

## 6. Privacy, failure, and audit boundary

`ContractError` exposes only a controlled `HealthReason`, never arbitrary
values, unknown enum spelling, malformed keys or raw exception messages.
`HealthRequest` and potentially untrusted reference reprs omit their values.
Do not log request dictionaries, rejected bytes, their digests, raw IDs,
stack traces with captured input, or health prose. Constructor and decoding
errors must be contained at the future boundary, not propagated to the
consumer as a free-form message. No rejected content is returned in a decision.

An audit is a minimal immutable in-memory projection:
`outcome`, `reason_codes`, `support`, `support_template_id`, `policy`,
`catalog`, `scope`, `evaluation_time`. The versions, scope, and time are
copied only from independently verified harness metadata, or are `None` when
not established. Even syntactically valid caller-supplied IDs may contain
private data; they are not suitable audit metadata by virtue of grammar.
Only registered public synthetic IDs from trusted snapshots may appear.
Do not include request IDs, evidence IDs, proposed/rejected artifact IDs or
digests, people, locations, consent details, textual descriptions, or
exception content. The decision itself contains a reviewed artifact reference
only for ALLOW. There is no audit logging, shared event history, network,
training, telemetry, callback, or external persistence.

## 7. Nonnegotiable invariants and test obligations

The test module implements only executable **structural** checks in Phase 1.
The behavioral specification tests below are individually collected as skipped
`test_phase2_governor_behavior_pending[case-id]` and
`test_phase2_operational_invariant_pending[invariant]`. Each must be made
executable against the real public evaluator in Phase 2; the skip markers
and placeholder bodies must be removed. Passing shape checks of authored
expectations is not evidence that any behavior is implemented.

| # | Invariant | Phase 1 proof / Phase 2 executable obligation |
| --- | --- | --- |
| 1 | Evidence presence never proves medical correctness. | Manifest/digest shape tests; `evidence-presence-not-medical-correctness`: supporting annotations cannot clear unsupported certainty or a prohibited medical purpose (`dev-unsupported-certainty`, `dev-diagnosis-prohibited`). |
| 2 | Every claim, evidence, artifact, purpose and catalog version match. | Catalog internal-link/corruption tests; `complete-claim-evidence-artifact-purpose-catalog-binding`: reject foreign claim, purpose, version, digest and evidence substitutions. |
| 3 | Repeated/correlated evidence cannot manufacture independence. | Frozen group metadata and canonical reference multiset; `deduplicate-underlying-source-groups`: compare `dev-education-eligible`, `dev-correlated-evidence`, `dev-repeated-evidence`, and duplicates with already sufficient support. |
| 4 | Unsupported certainty and contradiction cannot reach ALLOW. | Explicit assessment/relation enums; `unsupported-certainty-and-contradictions-withheld`: check known contradiction even if the untrusted ref is omitted. |
| 5 | Consent/authority never originate from model text. | Closed request decoding, noncoercing context, restrictive policy tests; `consent-and-authority-only-from-trusted-harness`: injected fields, authority marker, expired/revoked/denied/missing/mismatched consent, and a valid safe purpose excluded by a narrower or empty trusted policy. |
| 6 | Malformed inputs and evaluator failures never release. | Bounded schema, time, unknown enum, catalog corruption and result-shape tests; `malformed-input-and-exceptions-fail-closed`: test malformed request, context and catalog, unavailable time or policy, injected evaluator exception. |
| 7 | Denial never suppresses independent public referral. | Representable DENY + immediate support and fixed-template pair; `denial-preserves-independent-public-referral`: treatment, malformed request/clock/catalog, invalid context and evaluator-failure pairs. |
| 8 | Release binds to exactly one reviewed artifact. | SHA-256 fixture checks, exact permitted ref or `None` on withholding; `release-only-exact-reviewed-artifact-from-trusted-decision`: reject changed bytes, unrelated IDs and untrusted decision objects at a synthetic release boundary. |
| 9 | Untrusted context cannot cross fixture, turn or consent boundaries. | Typed scope and manifest links; `fixture-turn-and-consent-isolation`: same bytes across turns, cross-turn evidence, consent mismatch, unbound urgency/unknown scope. |
| 10 | Audit and error outputs never reveal private/rejected content. | Controlled errors, no repr echo, audit allowlist and synthetic privacy canaries; `audit-and-error-canaries-never-disclosed`: test evaluator error, malformed fields and rejected artifact through the actual output/audit path. |
| 11 | Equal canonical inputs + explicit policy/context produce identical results. | Canonical ordering, deep immutability and defensive-copy tests; `equal-canonical-inputs-identical-decisions`: repeat and reorder evaluations, compare outcomes, reasons, artifact and audit across independent instances. |
| 12 | Health evaluation cannot mutate existing AILEE domain state. | Incremental import-isolation check and stdlib-only dependency scan; `existing-domain-state-unchanged`: snapshot existing domain state before/after evaluation, including failure paths; no dispatch or side effects. |

## 8. Synthetic fixtures and future adjudication

[`scenarios.json`](../../../tests/fixtures/health_safety/scenarios.json)
defines corpus metadata, policy, versioned catalog, exact original synthetic
artifact payloads, separate complete request and trusted-context templates,
and case partitions. Each case has an ID/description, independent
`request_template` and `context_template` keys, explicit shallow top-level
`request_overrides` and `context_overrides` (copy before replacement), and a
fixture-only `harness_fault`. No template or trusted fact is computed from
an untrusted request. Overrides deliberately permit malformed raw fixture
values so the schema can be tested. `DUPLICATE_CATALOG_ARTIFACT` exercises
catalog construction now; `EVALUATOR_EXCEPTION` is a future fault marker
and is **not executed** in Phase 1. No fault or override metadata is an
authorization field.

[`expected.json`](../../../tests/fixtures/health_safety/expected.json)
contains separately specified, **provisional engineering** labels for 49
development cases: schema checks, future candidate/support outcome, primary
reason, fixed template ID and exact permitted ref (or null). These include
4 provisional ALLOW, 12 REVIEW and 33 DENY labels. The schema test validates
their structure and reference consistency, not their clinical validity or
future evaluator correctness. Four additional visible cases are reserved,
with only `AWAITING_INDEPENDENT_ADJUDICATION` status and no outcome labels.
This visible seed reservation is not a blind or statistically independent
holdout; do not train/tune on it or report it as such.

The following are explicit future release gates:

- Independently specify the label adjudication protocol, reviewer criteria,
  disagreement procedure, and a genuinely held-out synthetic corpus before
  using results as research evidence. No independent clinical or other label
  review has occurred. Independent synthetic adjudication does not establish
  clinical efficacy, clinical readiness or regulatory compliance.
- Confirm the corpus licensing/redistribution terms and reviewer-created
  contributions before any corpus release. All current examples are original
  synthetic engineering fixtures; no external health data or real cases were
  imported. Corpus licensing review remains pending.
- Implement and verify the pure governor and trusted failure boundary in
  Phase 2, adversarial testing/independent adjudication in later phases,
  then a reproducible benchmark and measured research report. No performance
  or medical-security effectiveness claims until supported by actual evidence.
  Template text review, provenance of catalog labels/source groups, and the
  trusted release consumer are also outstanding. No integration or clinical
  deployment is authorized.

## 9. Verification and stop boundary

From a checkout of the exact baseline (and the same environment for the
feature branch), the reproducible existing Python test command is:

~~~sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m pytest -q --tb=short -p no:cacheprovider tests
~~~

The verified baseline's 327 tracked blobs matched the GitHub commit.
With Python 3.13 the existing suite produced 765 passes, 3 skips and one
known environment-specific failure in
`test_windows_and_macos_native_file_process_network_and_identity`
(localhost `socket.bind`, `PermissionError`). This is not a Health-domain
failure or justification to change existing code; record the same test in
both runs, and compare with that single environment-restricted test
deselected if necessary.

~~~sh
# Phase 1 structural tests; -rs lists intentionally pending Phase 2 tests.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m pytest -q -rs -p no:cacheprovider tests/test_health_safety_domain.py
~~~

Phase 1 local verification (same isolated source and environment):

| Check | Observed result |
| --- | --- |
| Health structural suite, Python 3.11 | 101 passed, 61 skipped |
| Health structural suite, Python 3.13 | 101 passed, 61 skipped |
| Existing + Health suite, Python 3.13 | 866 passed, 64 skipped, same one localhost socket permission failure |
| Same suite with only the restricted native test deselected | 866 passed, 64 skipped, 1 deselected |
| Pending operational cases | 49 scenario verdicts + 12 invariant tests (61 explicit skips, not passing tests) |
| Python 3.9 grammar and compile | All three added Python files passed `ast.parse(feature_version=(3, 9))` and compilation |
| Package discovery and wheel smoke | Health subpackage discovered; offline `--no-deps --no-build-isolation` wheel built, installed to an isolated directory, and imported; version remains 10.1.1 |
| Incremental Health import | No new sibling domain modules, output, logging changes, threads, environment/path changes, file writes or network events detected by the import test |

The full Python 3.9 runtime was not available locally. The grammar check is
not proof of a complete Python 3.9 runtime test on this branch; the hosted
Python 3.9/3.10/3.11 matrix remains a separate verification gate. Hosted
results for the Phase 1 commit belong in the draft PR's check record.
No release tag is created for AHS.

**STOP after Phase 1**. No operational evaluator, automatic authorization,
public template rendering, adversarial execution, benchmark, clinical
assessment, PR merge, release tag or existing-domain modification belongs
in this phase.
