# OpenNAM Phase 4 — reproduce the frozen engineering observations

From a trusted repository checkout, with a supported Python interpreter:

```sh
python scripts/opennam_reproduce.py --summary
python scripts/opennam_reproduce.py
python -m pytest -q tests/test_opennam_contracts.py tests/test_opennam_governor.py tests/test_opennam_evaluation.py tests/test_opennam_adapters.py tests/test_opennam_reproducibility.py
```

The runner is checkout-only (the fixture corpus and test harness are not
installed package assets). It needs no third-party dependencies, network,
data downloads, credentials, fresh wall-clock evaluation, or external
services. For a different working directory, use the absolute path to the
checkout's `scripts/opennam_reproduce.py`; the same checkout's source and
fixtures are selected regardless of current directory.

The original Phase 4 runner verified 17 selected files; its exact original
manifest is preserved verbatim as `tests/fixtures/opennam/reproducibility-v1.json`
(SHA-256 `b87a85b86e3556c0117cf242179fb7e21daa5df013879f00764c81c462ce69cc`)
and refers to the original Phase 4 commit, not the current files. The current
**revision 2** runner verifies the exact bytes of 21 selected files against
`tests/fixtures/opennam/reproducibility.json`, including the historical manifest
with its pinned fingerprint, adapter source/docs/tests, prior domain source,
fixture inputs/labels/observations, harness, tests, method/scope documents
and runner. The active manifest and cumulative phase log are not self-hashed.
The active manifest cannot rebaseline the original inputs, labels or observed
report: their three fingerprints must agree with the hash-pinned v1 archive.
It requires a closed, bounded (≤64 KiB) UTF-8 manifest, rejects duplicate keys
at every level or invalid revision/phase/path/digest fields, and streams known
selected files in bounded chunks (≤1 MiB each). It then reconstructs only the
original 40-case governor evaluation with unchanged frozen fixture clocks,
compares the complete emitted canonical JSON bytes against
`tests/fixtures/opennam/observed.json`, requires exact labels, and prints only
that report (or the stable summary). It does not persist/rewrite observations.
A missing/mismatched artifact or result exits nonzero without printing a
report. The default report remains byte-identical to the original run; the
`--summary` schema is revision 2 and includes the original Phase 4 commit and
historical manifest fingerprint, not a new corpus. Output includes neither runtime timestamps nor platform-specific
paths; it is intended for hash seed/version comparison. Exact result digest:

```text
a07b17b90153e36432f21a44e122cd7225679e68c904a9910fbe97f17b80f2e9
```

The test suite exercises different Python hash seeds (`0`, `7`, `31`),
working-directory independence, exact byte reproduction, and integrity
failure in isolated copies altered at source, method, label, case, report,
archive, adapter and harness paths, and malformed, oversize or missing manifests.
Adapter behavior is exercised separately by its dedicated pytest suite, not
by this original-corpus runner. This is deterministic conformance to 40 original
authored provisional engineering cases, not scientific reproducibility of
a non-animal study, OECD guideline or unavailable OpenNAM v0.1.0 prototype.

The JSON manifest is an integrity check within a known checkout, not a signed
manifest, dependency lock, security guarantee or independent attestation.
Refer to the reviewed Git history to establish exact source provenance. A
Phase 4 result does not certify operational, scientific or legal readiness.
See [the threat model](THREAT_MODEL.md), [corpus provenance](CORPUS_PROVENANCE.md)
and [benchmarks](BENCHMARKS.md) for boundaries and the complete method.
