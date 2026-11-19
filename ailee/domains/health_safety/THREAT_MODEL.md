# AILEE Health Safety — experimental threat model

**Scope:** the pure, offline synthetic `HealthSafetyGovernor` and its
documented Boolean artifact-integrity helper frozen at Phase 3
`92379d20205c72a5dd6cc5638d3650e3db036d48`. This is a defensive
contract model for an honest Python fixture harness. It is not a claim of
production trust, medical safety, clinical triage, or attack resistance.

## Assets and assumptions

| Asset | Intended property |
| --- | --- |
| Candidate disposition and controlled primary reason | Deterministic categorical DENY/REVIEW/ALLOW; withhold unless all requirements pass |
| Synthetic immutable artifact reference | Only the exact reviewed ID and SHA-256 can appear as permitted on ALLOW |
| Independently established support disposition | Preserve correctly scoped reliable support even when a candidate is withheld |
| Trusted configuration/context | Explicit pinned policy/catalog, scope, whole UTC second, synthetic consent and urgency; no candidate field grants trust |
| Audit projection | Minimal immutable metadata; no private/rejected content, evidence references or rejected digest |
| Existing AILEE domains | Opt-in Health import and evaluation do not modify sibling state or existing behavior |
| Frozen research evidence | Versioned scenarios, separate provisional labels, actual observed projections, checksums and reproducible commands |

The harness independently selects and validates the policy, catalog,
context, consent and urgency snapshots. The catalog reviewer/source
group labels and immutable synthetic payload store are assumed correct for
the experiment, not authenticated or independently verified by the
governor. A context's structural validity is not proof of its origin.
The evaluator's constructors are ordinary Python records, not capabilities
or transferable authorization tokens. The trust boundary is the honest
harness, not a hostile Python interpreter with arbitrary code execution.

## Boundary and data ownership

| Interface | Origin and restrictions |
| --- | --- |
| Request passed to `evaluate(request, *, context)` | Untrusted decoded built-in dict or exact typed `HealthRequest`; bounded fields, closed enums, declared bindings only |
| Policy and catalog at construction | Trusted immutable snapshots; pinned version, explicit allowed purposes and source-group rule |
| Context at evaluation | Separate harness assembly; reliable scope and urgency can survive controlled assembly failures; no ambient time |
| Decision and audit | Evaluator-produced controlled records; withholding always has no permitted artifact; support is separate |
| `verify_synthetic_artifact` | Fresh eligibility evaluation plus exact ID and digest of bounded inert bytes; Boolean only; not a release/delivery interface |

The untrusted input surface comprises only the closed request schema and
typed contract. The experiment does not accept raw text as a medical
request, perform symptom assessment, decode raw network payloads, or
operate a service. The benchmark uses only the two fixed local synthetic
corpora and their authored conditions; it is not a red-team framework.

## Defensive threat categories and coverage

The contract's ordering is deny before review before allow. Malformed or
unsupported input and invalid trusted configurations fail closed; scoped
consent, purpose, manifest and evidence bindings must agree. Unresolved
review, certainty, contradiction, completeness, validity and independence
are separately evaluated before any eligible synthetic output. An urgency
fact affects only the fixed public-support disposition.

| Invariant (README §7) | Concern | Control and tested coverage |
| --- | --- | --- |
| 1 | Apparent evidence mistaken for correctness | Explicit eligible catalog assessment; original evidence-presence invariant, Phase 3 certainty and benign controls |
| 2 | Inconsistent manifest bindings | Exact catalog, artifact, purpose, scope, claim and evidence matches; original binding invariant, Phase 3 evidence/consent cases |
| 3 | Count-based overstatement of independent evidence | Count distinct source groups per claim; original deduplication invariant, repeated/correlated and complete Phase 3 controls |
| 4 | Unresolved certainty or contradictory/incomplete evidence treated as eligible | Complete trusted manifest and ordered review gates; original contradiction invariant and Phase 3 certainty/evidence checks |
| 5 | Missing, invalid or mis-scoped authorization | Independently selected consent, fixed windows, restrictive policy and closed authority markers; original consent/authority invariant and Phase 3 consent/schema controls |
| 6 | Malformed input or evaluator/assembly fault | Strict typed/schema checks, controlled fail-closed reasons; original malformed/failure invariant and Phase 3 schema and controlled-fault cases |
| 7 | Candidate denial inadvertently suppresses public help | Independent scoped urgency snapshot before candidate evaluation; original referral invariant and Phase 3 public-support/fault cases |
| 8 | Content different from the one reviewed | Exact permitted reference and fresh, bounded Boolean digest verification; original integrity invariant and ten Phase 3 Boolean checks |
| 9 | Fixture, turn, catalog or consent cross-contamination | Exact independent scope/binding checks; original isolation invariant and Phase 3 context/consent checks |
| 10 | Private-looking rejected data exposed | Minimal decision/audit shape, controlled exception views, quiet-capture and inert privacy canaries; original privacy invariant and four new Phase 3 privacy cases |
| 11 | Nondeterminism | Canonical immutable records and explicit UTC; original replay invariant, Phase 3 order/fresh-instance and hash-seed tests, Phase 4 reproducibility |
| 12 | Existing domains affected | Separate package, no state/history/network/rendering, existing import-isolation and representative state snapshots; full regression and packaging CI |

The Phase 3 test record had **62/62** reference matches and no confirmed
in-scope governor defect requiring correction. The narrowed
decision-type check uses a constructed withheld result as an ordinary
wrong request type and checks the public method signature. It does not
establish authentication or release-consumer enforcement. The finite
coverage above is not proof against all input combinations, unknown faults,
host compromise or operational misuse. No exploit-development or
external-service testing is part of this experiment.

## Open boundaries and residual risks

- **Ingress and authentication:** no authenticated remote request
  transport, independent identity/authorization provider, duplicate-key and
  encoded-size ingress adapter, process sandbox, or isolation from arbitrary
  in-process Python code. Constructors, fixture labels and SHA-256 alone
  do not authenticate provenance.
- **Evidence and consent provenance:** no independent source verification
  or source-group adjudication, independent review of synthetic labels,
  actual patient permission system, external freshness protocol or immutable
  production catalog/payload storage. The caller/harness is responsible
  for the truth and freshness of those facts. A consumed or stale
  decision is not an authorization credential.
- **Artifact consumer:** no production release consumer, renderer,
  delivery path, origin-verification or current-turn release integration.
  The Boolean helper's exact byte contract is not an end-to-end release
  guarantee.
- **Privacy:** no persistence, monitoring service, real private records,
  telemetry, clinical note handling or transport in the experiment. The
  canary tests inspect specific decision/audit/exception views, captured
  Python output and propagated logs, not all conceivable sinks, processes,
  snapshots or downstream integrations.
- **Public support:** only fixed template IDs are tested. No template
  wording has been independently reviewed; no medical reasoning, clinical
  triage/referral classification from symptoms, automatic emergency calls,
  positioning, health service invocation or personalization exists.
- **External validity:** bounded known provisional engineering fixtures;
  no blind held-out corpus, no independent synthetic adjudication and no
  real-patient or clinical evidence. Case agreement does not establish
  clinical effectiveness, clinical safety, regulatory approval or
  production readiness.

Independent engineering review, corpus licensing/redistribution review,
independent synthetic adjudication and genuinely held-out case design
remain important before any broader research claim. Do not integrate
or deploy this experiment for medical decision-making.
