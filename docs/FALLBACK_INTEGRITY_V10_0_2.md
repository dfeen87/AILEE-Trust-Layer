# AILEE v10.0.2 fallback integrity hardening

## Baseline and scope

This patch starts from merged `main` at
`bf9bc8cbd21630b8a9f897db546831ef0fb0cf3d` (PR #94, v10.0.1).
The isolated branch is `codex/v10.0.2-fallback-integrity`. Its scope is the
Python core fallback boundary and the active release metadata. Historical
BEDROCK and v10.0.1 reports remain unchanged.

The invariant is that every completed fallback returns and commits a finite
value inside the configured hard safety envelope. An operation that cannot
establish valid output or numeric audit evidence raises before changing history,
last-known-good value, or the last result.

## Confirmed root causes

All five fallback call sites applied hard bounds before fallback clamps.
`hard_min=0`, `hard_max=1`, `fallback_clamp_min=2` and rejected input `9`
therefore returned and committed `2`. Configuration validated each interval
separately but did not reject disjoint intervals.

For finite history `[1e308, 1e308]`, the even median's sum overflowed and a
safety rejection returned and committed infinity with no hard bounds. The
negative counterpart committed negative infinity. `statistics.fmean` raised
`OverflowError` for those same histories even though their means are finite.
With empty history and hard bounds `[1e308, 1.5e308]`, the midpoint overflowed;
the subsequent hard clamp concealed that overflow by selecting `1.5e308`
instead of the finite midpoint `1.25e308`.

## Correction and compatibility

Fallback mode and bound validation now reject disjoint intervals and recheck
mutable configuration before a decision. Wider, overlapping fallback intervals,
one-sided bounds, touching intervals, and all three existing fallback modes
remain supported. A disjoint policy is rejected explicitly rather than silently
reinterpreted.

Fallback calculation and clamping use one boundary: validate the selected
numeric evidence, calculate the candidate, apply fallback clamps, and apply
hard bounds last. Even medians and empty-history midpoints retain the original
arithmetic when finite and use a half-sum when the intermediate sum overflows.
Means retain `fmean` for ordinary data and retry the standard library's exact
accumulation in `statistics.mean` on overflow. No dependency or arbitrary numeric
clipping is introduced.

Final result construction and commit validate the output before mutation.
Pipeline-generated numeric audit fields are checked for finiteness; caller
context remains arbitrary application metadata. Last-known-good remains the
selected policy when available; otherwise `last_good` still uses median history.
Ordinary governed fallback results still enter history and never replace the
last-known-good accepted value.

This is PATCH under the strict versioning policy: hard constraints and finite
trusted evidence were already required. The fixes restore those invariants for
unsafe configurations and overflow cases. Ordinary arithmetic, public
signatures, safety/GRACE/consensus routing, intentional `SKIPPED` acceptance,
and valid fallback policies retain their behavior. No governing layer is added
or removed.

## Verification

The initial 77 new cases on the uncorrected pipeline produced **48 failures and
29 passes**. After correction, eight further immediate-boundary cases cover
tightened runtime hard bounds, integer extreme bounds, and atomic variance
overflow in an earlier layer. The final focused run passed **108 tests**: all
**85 new regressions** and the **23 existing pipeline tests**.

The final full Python suite passed **670 tests**, with **one platform skip** and
**68 existing warnings**, on Python **3.11.16**. No assertions were relaxed.

| Command/check | Actual outcome |
|---|---|
| `python -m pytest tests/test_fallback_integrity.py tests/test_pipeline_smoke.py -q` | 108 passed |
| `python -m pytest tests -q` | 670 passed, 1 skipped, 68 warnings |
| `python -m compileall -q ailee tests` | Passed |
| `python -m pip check` | Passed |
| Existing three-file mypy gate, with `--follow-imports=skip` | No issues in licensing, industrial, and dual-domain files |
| Release metadata and governance-v1 tests | 18 passed |
| `python -m build` and fresh installed-wheel smoke outside the repository | sdist/wheel built; version 10.0.2, eight finite fallback scenarios, inconsistent-config rejection and atomic runtime rejection passed |
| `vitest run tests/governance.test.ts --maxWorkers=1 --minWorkers=1` in `packages/ailee-ts` | 2 passed |
| CMake Release configuration | Passed; project version 10.0.2 |
| Cargo manifest/lock metadata | Python release gate passed; Cargo tooling unavailable locally |
| `git diff --check` | Passed |

Python commands used the existing virtual environment with `PYTHONPATH` pointing
to the isolated worktree. The full suite required network-enabled sandbox
permissions for the existing in-process FastAPI test and native loopback
fixtures; a metadata/governance run under the default sandbox stalled and was
interrupted before rerunning successfully. No test was weakened. Package builds
emitted the existing setuptools license-classifier deprecation warning.

The final diff was reviewed independently for clamp authority, median/mean
compatibility, atomic failure, audit validation, and preservation of historical
release evidence. Full cross-runtime execution was left to the established CI
matrix because other runtime algorithms did not change.

## Remaining risks and limits

Extreme arithmetic in confidence, GRACE, or consensus may still raise before
fallback selection; such failures preserve state. This patch does not broaden
those layers' numerical algorithms. Caller mutation of public history or
configuration does not acquire a concurrency or recovery guarantee.

Other Python versions, Windows/macOS native behavior, and the complete Rust,
TypeScript, and C++ runtime matrix remain CI responsibilities. The independent
issues recorded in the historical v10.0.1 report remain outside this patch.

## Changed-file summary

- Runtime correction: `ailee/ailee_trust_pipeline_v1.py`.
- Focused regressions: `tests/test_fallback_integrity.py`.
- Release evidence: this report, changelog, architecture and versioning notes,
  and the README current-release description.
- Active version alignment: Python/package/ALCOA metadata, Cargo manifests,
  TypeScript package/lock/export/test metadata, CMake, citation, CI wheel smoke,
  and Python version-consistency assertions.
