#!/usr/bin/env python3
"""Offline checkout-only reproduction of frozen OpenNAM engineering observations."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "opennam" / "reproducibility.json"
PHASE3_COMMIT = "fe8418dbb5dce85867a8a536255685bc3ff2914a"
FROZEN_PATHS = (
    "ailee/domains/opennam/BENCHMARKS.md",
    "ailee/domains/opennam/CORPUS_PROVENANCE.md",
    "ailee/domains/opennam/README.md",
    "ailee/domains/opennam/REPRODUCIBILITY.md",
    "ailee/domains/opennam/THREAT_MODEL.md",
    "ailee/domains/opennam/__init__.py",
    "ailee/domains/opennam/contracts.py",
    "ailee/domains/opennam/governor.py",
    "scripts/opennam_reproduce.py",
    "tests/fixtures/opennam/cases.json",
    "tests/fixtures/opennam/expected.json",
    "tests/fixtures/opennam/observed.json",
    "tests/opennam_evaluation.py",
    "tests/test_opennam_contracts.py",
    "tests/test_opennam_evaluation.py",
    "tests/test_opennam_governor.py",
    "tests/test_opennam_reproducibility.py",
)


def verify_artifacts():
    """Check known source/fixture paths against the checked-in manifest."""
    try:
        manifest = json.loads(MANIFEST.read_bytes())
        if type(manifest) is not dict or set(manifest) != {"format", "phase3_commit", "sha256"}:
            raise ValueError("invalid reproducibility manifest")
        if manifest["format"] != "opennam-reproduction-v1" or manifest["phase3_commit"] != PHASE3_COMMIT:
            raise ValueError("unexpected reproducibility manifest version or phase")
        hashes = manifest["sha256"]
        if type(hashes) is not dict or set(hashes) != set(FROZEN_PATHS):
            raise ValueError("unexpected reproducibility manifest path set")
        for name in FROZEN_PATHS:
            expected = hashes[name]
            if (type(expected) is not str or len(expected) != 64 or
                    any(c not in "0123456789abcdef" for c in expected)):
                raise ValueError(f"invalid recorded fingerprint: {name}")
            if sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"frozen artifact mismatch: {name}")
        return manifest
    except (OSError, UnicodeError, TypeError, KeyError, RecursionError) as error:
        raise ValueError("missing or malformed reproducibility artifacts") from error


def reproduce():
    manifest = verify_artifacts()
    # Explicitly select the checkout under inspection, independent of cwd.
    # No installed baseline package is silently substituted for local source.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    from tests.opennam_evaluation import canonical_report

    actual = canonical_report().encode("ascii")
    observed = (ROOT / "tests" / "fixtures" / "opennam" / "observed.json").read_bytes()
    if actual != observed:
        raise ValueError("reconstructed observations differ from frozen report")
    parsed = json.loads(actual)
    if parsed["exact_governor_outcome_and_reason_matches"] != parsed["case_count"]:
        raise ValueError("provisional engineering labels do not match")
    summary = {
        "format": "opennam-engineering-reproduction-summary-v1",
        "phase3_commit": manifest["phase3_commit"],
        "observation_sha256": sha256(actual).hexdigest(),
        "verified_artifacts": len(FROZEN_PATHS),
        "case_count": parsed["case_count"],
        "exact_governor_outcome_and_reason_matches": parsed["exact_governor_outcome_and_reason_matches"],
        "readiness_comparison": parsed["readiness_comparison"],
    }
    return actual, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="emit only the stable comparison summary")
    args = parser.parse_args()
    try:
        actual, summary = reproduce()
    except (ValueError, OSError, ImportError):
        # Fixed diagnostic avoids emitting file content or malformed metadata.
        print("OpenNAM reproduction failed; check frozen checkout artifacts and observations.", file=sys.stderr)
        return 1
    if args.summary:
        actual = (json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")
    sys.stdout.buffer.write(actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
