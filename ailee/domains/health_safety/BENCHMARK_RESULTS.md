# AHS Phase 4 — observed offline benchmark results

**Baseline:** [frozen Phase 3 commit
`92379d20205c72a5dd6cc5638d3650e3db036d48`](https://github.com/dfeen87/AILEE-Trust-Layer/commit/92379d20205c72a5dd6cc5638d3650e3db036d48).
The [pre-measurement methodology](BENCHMARKS.md) was written and SHA-256
hashed before the first benchmark evaluation; no input, expected label, or
frozen governor was changed. See [corpus provenance](CORPUS_PROVENANCE.md)
and the [experimental threat model](THREAT_MODEL.md).

## Execution record

The canonical recorded execution ran in the verified research checkout on
CPython **3.13.13**, Darwin-arm64, with explicit hash seed 0, no bytecode
writes, no network, and the exact command:

~~~sh
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /Users/donfeeney/miniforge3/bin/python3.13 -m scripts.health_safety_benchmark \
  --partition all --repeats 3 \
  --output tests/fixtures/health_safety/phase4_benchmark_results.json
~~~

The runner returned process status **0**. The environment envelope,
complete decision/audit observations, all repeat digests, partition
counts, confusion matrices, metrics and frozen file hashes are in
[phase4_benchmark_results.json](../../../tests/fixtures/health_safety/phase4_benchmark_results.json).
The raw archived file includes runtime-specific metadata. Its canonical
`results` object (sorted-key compact ASCII JSON, no trailing newline) has a
separate SHA-256 that is independent of the environment envelope.

| Evidence | SHA-256 |
| --- | --- |
| Pre-measurement `BENCHMARKS.md` | `7b3521185e672ba020ee3f5357eb1db576cfa6093ed4a6c384813d981ffc6cf1` |
| Canonical results object | `c16e602280e64faa30601956d547b4a8e60dfb485082404a6a70cf301c9a1405` |
| Raw result file | `802fee38669c18237d3b702a721812add1129fac02d76423b058e4b6bec7794d` |
| [Local reproducibility record](../../../tests/fixtures/health_safety/phase4_reproducibility.json) | `a91e55c2769376f760997ac3c5b1dde533de977fa46723f675ffbcfa5036ec82` |

All five original/Phase 3 scenario, label and observation files have the
exact checksums in [CORPUS_PROVENANCE.md](CORPUS_PROVENANCE.md); the runner
verifies those and five frozen source/test checksums before and after
evaluation. Policy/catalog settings are unchanged:

| Partition | Policy | Catalog | UTC evaluation |
| --- | --- | --- | --- |
| Development | `policy-ahs-phase1` v1 | `catalog-ahs-synthetic` v1 | 2026-01-15 12:00:00Z; 2 explicit absent-time outcomes |
| Phase 3 | `policy-ahs-phase3` v1 | `catalog-ahs-phase3` v1 | 2026-06-02 12:00:00Z; 1 absent-context/time outcome |

## Case disposition and coverage

| Count | Development | Phase 3 |
| --- | ---: | ---: |
| Available | 53 | 62 |
| Selected provisional labeled cases | 49 | 62 |
| Executed with valid observed decision | 49 | 62 |
| Intentional invalid-input cases (overlap executed) | 13 | 13 |
| Controlled evaluator-fault cases (overlap executed) | 1 | 1 |
| Excluded visible reserved/unlabelled | 4 | 0 |
| Runner execution errors | 0 | 0 |
| Independently adjudicated labels | 0 | 0 |

All 111 distinct selected cases executed; each was evaluated three times
(333 primary case observations, plus the existing applicable integrity,
prior-eligibility and canonical replay checks). Repeats do not inflate
the scored classification denominators. The four visible reserved cases
were not executed or scored and remain unadjudicated. **No genuinely
held-out independently adjudicated corpus was available**.

## Three-way classification

The confusion matrices use expected rows and observed columns. Counts and
distributions are taken from the actual machine-readable record, not
assumed from the test tally.

**Frozen development corpus (49):**

| Expected \ Observed | ALLOW | REVIEW | DENY | Total |
| --- | ---: | ---: | ---: | ---: |
| ALLOW | 4 | 0 | 0 | 4 |
| REVIEW | 0 | 12 | 0 | 12 |
| DENY | 0 | 0 | 33 | 33 |
| **Total observed** | **4** | **12** | **33** | **49** |

**Frozen Phase 3 defensive validation corpus (62):**

| Expected \ Observed | ALLOW | REVIEW | DENY | Total |
| --- | ---: | ---: | ---: | ---: |
| ALLOW | 18 | 0 | 0 | 18 |
| REVIEW | 0 | 13 | 0 | 13 |
| DENY | 0 | 0 | 31 | 31 |
| **Total observed** | **18** | **13** | **31** | **62** |

## Metrics (numerator / denominator)

| Metric | Development | Phase 3 |
| --- | --- | --- |
| False allow (FAR) | **0 / 45 = 0%** | **0 / 44 = 0%** |
| False block (FBR) | **0 / 4 = 0%** | **0 / 18 = 0%** |
| Three-way agreement | 49 / 49 = 100% | 62 / 62 = 100% |
| Incorrect REVIEW/DENY | 0 / 45 = 0% | 0 / 44 = 0% |
| Controlled reason-code agreement | 49 / 49 = 100% | 62 / 62 = 100% |
| Public-support disposition agreement | 49 / 49 = 100% | 62 / 62 = 100% |
| Fixed support-template ID agreement | 49 / 49 = 100% | 62 / 62 = 100% |
| Immediate-referral preservation failures | 0 / 6 = 0% | 0 / 5 = 0% |
| Withheld-candidate immediate-referral failures | 0 / 6 = 0% | 0 / 4 = 0% |
| Exact artifact-permission mismatches | 0 / 49 = 0% | 0 / 62 = 0% |
| Boolean synthetic artifact-check mismatches | 0 / 0 — **undefined** | 0 / 10 = 0% |
| Privacy-canary incidents | 0 / 4 = 0% | 0 / 4 = 0% |
| Quiet-capture failures | 0 / 49 = 0% | 0 / 62 = 0% |
| Explicit canonical reorder/replay mismatches | 0 / 0 — **undefined** | 0 / 3 = 0% |
| Complete repeat mismatches | 0 / 98 = 0% | 0 / 124 = 0% |
| Full contract agreement | 49 / 49 = 100% | 62 / 62 = 100% |

The development Boolean/reorder metrics have no per-case reference labels,
so their percentage fields are null, not 0%. The separately executed 12
operational invariants cover the original Boolean helper and replay
obligations. All Phase 3 byte-check results matched (one true, nine false);
no content was rendered or delivered. The observed supports were:

| Support | Development | Phase 3 |
| --- | ---: | ---: |
| Ordinary help | 42 | 54 |
| Qualified review | 1 | 3 |
| Immediate public referral | 6 | 5 |

The four development originally expected-ALLOW controls were all allowed,
with complete matches (4/4). The four distinct Phase 3 benign controls
matched exactly (4/4): two eligible educational/uncertainty fixtures
ALLOW, two unresolved-review/evidence cases REVIEW, none DENY. This
supports finite legitimate-output preservation, not a general estimate
of false blocking.

## Reproducibility

Six independent local process invocations ran the same all-partition
three-repeat selection. The canonical results objects were compared
for exact equality, not just equal headline metrics. All six returned
status 0 and the same canonical SHA-256 above.

| Interpreter / platform | Hash seeds | Process outcomes |
| --- | --- | --- |
| CPython 3.13.13 / Darwin-arm64 (`/Users/donfeeney/miniforge3/bin/python3.13`) | 0, 7, 31 | 3/3 status 0; exact match |
| CPython 3.11.14 / Darwin-arm64 (`/opt/homebrew/bin/python3.11`) | 0, 7, 31 | 3/3 status 0; exact match |

The five additional runs used the identical command with the listed
interpreter/seed and a different new output path. Python 3.9 and 3.10
were not available as local runtimes. Hosted supported-runtime tests
explicitly recompute the entire recorded canonical results object and
compare fresh-process results at all three seeds. Exact Phase 4 hosted
results are recorded in the draft PR only after the Phase 4 commit runs
CI; no unobserved hosted result is claimed here.

To reproduce without overwriting archived evidence:

~~~sh
for seed in 0 7 31; do
  PYTHONHASHSEED="$seed" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
    python -m scripts.health_safety_benchmark --partition all --repeats 3
done
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m pytest -q -p no:cacheprovider tests/test_health_safety_benchmark.py
~~~

The documented methodology, input checksums, full raw observations, repeat
digests and separate references allow all reported numbers to be recomputed.

## Interpretation and residual limitations

These observations establish conformance to the frozen synthetic engineering
expectations under the tested conditions: categorical decisions, controlled
reason codes, independently established support, exact artifact-reference
permission and applicable integrity/privacy/replay controls. No observed
in-scope frozen-governor discrepancy required a correction.

**Engineering-label agreement is not clinical accuracy or patient
outcomes.** The authors/implementer knew both scored partitions; neither
was independently adjudicated or blinded. A zero observed false-allow or
false-block count in 45/4 or 44/18 expected cases respectively cannot
establish a population error rate of zero. The corpus has limited coverage,
is not a probability sample, and no independent holdout estimate is
available. No latency, throughput, clinical safety, therapeutic validity,
regulatory approval, production readiness, universal adversarial
resistance, or efficacy claim is made. There is no authenticated real-world
ingress, independent source verification, medical reasoning, clinical
triage, actual release consumer, or template delivery. Corpus licensing
and independent synthetic adjudication remain pending.

Retain the draft and seek independent engineering review of the contract,
code, labels, harness ownership, provenance and reproducibility before any
further decision.
