# OpenNAM Phase 4 — reproduce the frozen engineering observations

From a trusted repository checkout, with a supported Python interpreter:

```sh
python scripts/opennam_reproduce.py --summary
python scripts/opennam_reproduce.py
python -m pytest -q tests/test_opennam_contracts.py tests/test_opennam_governor.py tests/test_opennam_evaluation.py tests/test_opennam_reproducibility.py
```

The runner is checkout-only (the fixture corpus and test harness are not
installed package assets). It needs no third-party dependencies, network,
data downloads, credentials, fresh wall-clock evaluation, or external
services. For a different working directory, use the absolute path to the
checkout's `scripts/opennam_reproduce.py`; the same checkout's source and
fixtures are selected regardless of current directory.

The runner first verifies the exact bytes of 17 selected files against
`tests/fixtures/opennam/reproducibility.json`, including domain source,
fixture inputs/labels/observations, harness, tests, method/scope documents
and the runner. The manifest and cumulative phase log are intentionally not
self-hashed. It then reconstructs the real evaluation with frozen fixture
clocks, compares the complete emitted canonical JSON bytes against
`tests/fixtures/opennam/observed.json`, requires exact labels, and prints only
that report (or the stable summary). It does not persist/rewrite observations.
A missing/mismatched artifact or result exits nonzero without printing a
report. Output includes neither runtime timestamps nor platform-specific
paths; it is intended for hash seed/version comparison. Exact result digest:

```text
a07b17b90153e36432f21a44e122cd7225679e68c904a9910fbe97f17b80f2e9
```

The test suite exercises different Python hash seeds (`0`, `7`, `31`),
working-directory independence, exact byte reproduction, and integrity
failure in isolated copies altered at source, method, label, case, report
and harness paths. This is deterministic conformance to 40 original
authored provisional engineering cases, not scientific reproducibility of
a non-animal study, OECD guideline or unavailable OpenNAM v0.1.0 prototype.

The JSON manifest is an integrity check within a known checkout, not a signed
manifest, dependency lock, security guarantee or independent attestation.
Refer to the reviewed Git history to establish exact source provenance. A
Phase 4 result does not certify operational, scientific or legal readiness.
See [the threat model](THREAT_MODEL.md), [corpus provenance](CORPUS_PROVENANCE.md)
and [benchmarks](BENCHMARKS.md) for boundaries and the complete method.
