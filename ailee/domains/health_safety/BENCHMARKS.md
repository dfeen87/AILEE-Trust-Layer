# AHS Phase 4 — pre-measurement offline benchmark methodology

This methodology is fixed **before** measuring Phase 4. The evaluation
baseline is Phase 3 commit
`92379d20205c72a5dd6cc5638d3650e3db036d48`, verified in draft PR #99
after both exact-commit hosted CI runs succeeded.

## Question and partitions

Measure conformance of the actual public `HealthSafetyGovernor.evaluate`
and its documented Boolean synthetic artifact check to **provisional
engineering reference labels** under finite, offline, explicitly scoped
synthetic conditions. This is not a study of clinical performance,
authenticated ingress, prompt injection, production security, patient
outcomes, or medical effectiveness.

| Partition | Frozen reference | Scoring |
| --- | --- | --- |
| `development` | Original Phase 1/2 `ahs-phase1-synthetic-v1` | All 49 labeled development cases; four visible reserved cases excluded |
| `phase3` | `ahs-phase3-defensive-v1` | All 62 pre-measurement Phase 3 labeled cases |
| Independent holdout | None available | Not measured |

The two scored partitions are known to the implementer, not blind holdouts.
Report them separately; never pool them into an independent estimate.
The four unlabelled reserved cases remain unadjudicated, are not executed,
and are excluded from every scored denominator. No labels, payloads,
policy/catalog versions, source groups, or governor source may be changed
to fit the observations. See [corpus provenance](CORPUS_PROVENANCE.md).

## Execution and evidence

The standard-library research runner in
`scripts/health_safety_benchmark.py` reads only the fixed checkout fixtures.
It checks their SHA-256 and frozen source checksums, checks exact partitions
and label coverage, constructs the trusted context separately from the
request, and invokes the real governor through the public API. It does not
authenticate context, parse medical text, contact a service, render
templates, execute clinical actions, or release payloads.

For the original corpus, use the same recorded shallow request and context
overrides and the same independent fixture failure representations as the
Phase 2 tests. Controlled decoder/clock/catalog failures are recorded
without raw exception details. The one existing controlled evaluator-fault
case uses its already specified test-only fault; Phase 3 uses the exact
frozen observation harness, not a new input or fault-generation system.
The observation path does not use expected candidate classifications to
choose evaluation inputs. Labels are loaded separately and only used to
validate/check measured results. Compare full decision and audit projections.

Evaluation times are the fixed whole-second UTC times in each fixture:
development `2026-01-15T12:00:00Z` except explicit unavailable/invalid-clock
cases (absent audit time), Phase 3 `2026-06-02T12:00:00Z` except absent
trusted context. Policies/catalogs stay at their pinned synthetic version
1. Deterministic lexical case ordering and fresh governor instances are
used in every repeat. Run three complete evaluations at a minimum for the
recorded measurement (`--repeats 3`); compare complete normalized observed
records by SHA-256. Cross-process repeats use `PYTHONHASHSEED=0,7,31`.
Ambient time is never sampled for a decision or used as a random seed.

The runner emits an environment envelope and a separately canonicalized
results object with its SHA-256. Environment metadata (interpreter/version,
platform and explicit hash seed) may differ across runs; the results object
must not. A nonzero process exit indicates failed benchmark integrity,
reference conformance, or replay, not an expected `DENY` classification.
The recorded measurement is the output for `--partition all --repeats 3`
with seed 0; repeat at the other seeds and available runtimes. Hosted tests
recompute and compare the recorded results on Python 3.9/3.10/3.11.

~~~sh
# From the repository root, with an explicit local interpreter:
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m scripts.health_safety_benchmark --partition all --repeats 3

# To retain an independently generated record, create a NEW output file:
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m scripts.health_safety_benchmark --partition all --repeats 3 \
  --output /path/to/new-benchmark-results.json
~~~

Select `--partition development` or `--partition phase3` for a single
partition. Existing inputs and observations are never overwritten.

## Defined metrics and denominators

Passage of candidate-authorization gates is categorical, not a score.
For each partition, only report rates if every selected labeled case
produced a valid observed decision. A runner execution/integrity error
must be separately counted; metrics become undefined for an incomplete
partition, and the process exits nonzero. A defined, intentionally invalid
fixture may correctly produce a controlled decision and is still executed
and scored. No rate with a zero denominator is reported as zero percent;
its percentage field is JSON `null` (undefined/not measured). Percentages
are 100 × numerator / denominator, rounded to six decimal places.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| False-allow rate (FAR) | Expected REVIEW or DENY, observed ALLOW | All expected-withhold cases (REVIEW + DENY) |
| False-block rate (FBR) | Expected ALLOW, observed REVIEW or DENY | All expected-ALLOW cases |
| Three-way agreement | Exact ALLOW/REVIEW/DENY matches | All labeled executed cases |
| Incorrect REVIEW/DENY | Expected REVIEW observed DENY, or expected DENY observed REVIEW | All expected-withhold cases |
| Primary reason-code agreement | Exact controlled reason-code tuple | All labeled executed cases |
| Public-support agreement | Exact support disposition | All labeled executed cases |
| Fixed-template agreement | Exact public template ID | All labeled executed cases |
| Immediate-referral preservation failure | Expected immediate support, different observed support | All expected immediate-public-referral cases |
| Withheld-candidate immediate-referral failure | Same, restricted to expected REVIEW/DENY | All expected-withhold immediate-referral cases |
| Artifact-permission mismatch | Observed permitted reference differs from exact expected ID + digest, or expected none | All labeled executed cases |
| Boolean synthetic artifact-check mismatch | Observed byte-check Boolean differs from explicitly authored expectation | Cases with explicit Boolean labels (Phase 3's ten; development has none) |
| Privacy-canary disclosure incident | Any named canary in inspected decision/audit/controlled exception/captured output/log view | Four fixed Phase 1/2 privacy/fault cases; four Phase 3 privacy cases |
| Quiet-capture failure | Unexpected captured stdout/stderr/propagated log record | All labeled executed cases |
| Canonical reorder/fresh-instance mismatch | Phase 3 explicit replay expectation differs | Three Phase 3 replay cases; none explicitly scored in development |
| Deterministic repeat mismatch | Repeat observation differs from that case's first complete observation | Each case × (repeats − 1) comparisons |
| Full contract agreement | Complete decision/audit and applicable reference assertions match | All labeled executed cases |

Report expected and observed distributions and a full 3 × 3 confusion
matrix, with **expected rows and observed columns** ordered ALLOW, REVIEW,
DENY. Repeated runs do not multiply the classification denominators.
Every applicable reference assertion (including Phase 3 prior eligibility
control and frozen observation equality, originally specified structural
checks, and quiet capture) contributes to full contract agreement.

For the privacy measure, the development denominator is exactly
`dev-privacy-marker`, `dev-privacy-canary-input`, `dev-privacy-canary-key`
and `dev-evaluator-exception-urgent`; check the synthetic canary spellings
and withheld artifact identifier/digest, where present, in the documented
views. Phase 3 uses its four predeclared privacy cases. Do not retain raw
captured messages or rejected payloads in the observation record. This
does not measure undiscovered data or every conceivable logging sink.

The development corpus has no independently stored per-case Boolean helper
labels, so that specific metric has denominator zero there; exact
permission-reference mismatch is still measured for all 49 cases and
the 12 existing operational invariants separately exercise the helper.
Do not confuse a correctly eligible candidate with an intentionally
incorrect synthetic payload/identifier Boolean result.

Benign-control reporting: for development, the four originally labeled
ALLOW controls (not a distinct validation split); for Phase 3, the four
explicit `benign-controls` cases (two expected ALLOW, two expected REVIEW).
Report expected/observed distributions and full-contract matches in
that subset. Neither subset is an independent estimate of false blocks.

Count cases with controlled invalid schema/context/configuration input
using the **frozen expected** primary reason in
`MALFORMED_SCHEMA`, `UNSUPPORTED_SCHEMA`, `INVALID_CONTEXT`,
`INVALID_TIMESTAMP`, `INVALID_POLICY`, `INVALID_CATALOG`.
These intentional invalid input cases overlap the executed count, and
are not runner errors. Count the specified controlled evaluator-fault
cases separately. A scoped binding/consent/prohibition denial is not
an invalid experiment input merely because it is DENY. Report available,
selected, executed, excluded, reserved-unlabelled, independent-adjudication
and runner-error case counts. The first two scored partitions have no
independently adjudicated labels; all four reserved examples are truly
unlabelled and not tested.

## Integrity gates

- Exact Phase 3 freeze and fixture/source checksums, duplicate-free
  bounded input decoding, one unique expected label per scored case,
  and untouched reserved statuses.
- Repeatable reference and observation separation. Deterministic ordering,
  explicit times, no mutation to frozen inputs. Nonzero exit on an invalid
  corpus, harness execution error, any repeat inconsistency, or observed
  disagreement with a frozen reference. No labels are adjusted after the
  fact. A frozen governor defect blocks Phase 4.
- Verify metric arithmetic, including nonzero error counts and zero
  denominators, and compare machine-readable results across runs and
  supported Python runtimes.
- Report actual observations and limitations in
  [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), with separate raw result
  records sufficient to reproduce every headline count.

These are engineering-label agreement metrics for finite known synthetic
cases, not estimates of patient benefit, clinical accuracy, generalizable
attack resistance, or regulated clinical safety.
