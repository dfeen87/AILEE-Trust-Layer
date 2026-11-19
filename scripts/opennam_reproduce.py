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
PHASE4_COMMIT = "f6090ea94d093e08fcfc461501c957280e5cb0a3"
PRIOR_MANIFEST_PATH = "tests/fixtures/opennam/reproducibility-v1.json"
PRIOR_MANIFEST_SHA256 = "b87a85b86e3556c0117cf242179fb7e21daa5df013879f00764c81c462ce69cc"
ORIGINAL_CORPUS_PATHS = (
    "tests/fixtures/opennam/cases.json",
    "tests/fixtures/opennam/expected.json",
    "tests/fixtures/opennam/observed.json",
)
MAX_MANIFEST_BYTES = 65536
MAX_SELECTED_FILE_BYTES = 1048576
FROZEN_PATHS = (
    "ailee/domains/opennam/ADAPTERS.md",
    "ailee/domains/opennam/BENCHMARKS.md",
    "ailee/domains/opennam/CORPUS_PROVENANCE.md",
    "ailee/domains/opennam/README.md",
    "ailee/domains/opennam/REPRODUCIBILITY.md",
    "ailee/domains/opennam/THREAT_MODEL.md",
    "ailee/domains/opennam/__init__.py",
    "ailee/domains/opennam/adapters.py",
    "ailee/domains/opennam/contracts.py",
    "ailee/domains/opennam/governor.py",
    "scripts/opennam_reproduce.py",
    "tests/fixtures/opennam/cases.json",
    "tests/fixtures/opennam/expected.json",
    "tests/fixtures/opennam/observed.json",
    PRIOR_MANIFEST_PATH,
    "tests/opennam_evaluation.py",
    "tests/test_opennam_adapters.py",
    "tests/test_opennam_contracts.py",
    "tests/test_opennam_evaluation.py",
    "tests/test_opennam_governor.py",
    "tests/test_opennam_reproducibility.py",
)


def _read_bounded(path, limit):
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("oversized reproduction artifact")
    return data


def _hash_file(path):
    # Bounded streaming prevents a replaced selected file from causing an
    # unbounded allocation before it fails its recorded hash.
    digest = sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                return digest.hexdigest()
            total += len(chunk)
            if total > MAX_SELECTED_FILE_BYTES:
                raise ValueError("oversized reproduction artifact")
            digest.update(chunk)


def _parse_manifest(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate reproducibility manifest key")
            result[key] = value
        return result

    def reject_number(raw):
        raise ValueError("numeric reproducibility manifest value")

    # Exact UTF-8, no JSON autodetection of UTF-16/32, no arbitrary numbers.
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_int=reject_number, parse_float=reject_number,
                      parse_constant=reject_number)


def verify_artifacts():
    """Check known source/fixture paths against the checked-in manifest."""
    try:
        manifest = _parse_manifest(_read_bounded(MANIFEST, MAX_MANIFEST_BYTES))
        if type(manifest) is not dict or set(manifest) != {
            "format", "phase3_commit", "phase4_commit", "prior_manifest_sha256", "sha256"
        }:
            raise ValueError("invalid reproducibility manifest")
        if (manifest["format"] != "opennam-reproduction-v2" or manifest["phase3_commit"] != PHASE3_COMMIT
                or manifest["phase4_commit"] != PHASE4_COMMIT or
                manifest["prior_manifest_sha256"] != PRIOR_MANIFEST_SHA256):
            raise ValueError("unexpected reproducibility manifest version or phase")
        hashes = manifest["sha256"]
        if type(hashes) is not dict or set(hashes) != set(FROZEN_PATHS):
            raise ValueError("unexpected reproducibility manifest path set")
        if hashes[PRIOR_MANIFEST_PATH] != PRIOR_MANIFEST_SHA256:
            raise ValueError("historical reproducibility manifest fingerprint mismatch")
        historical = _read_bounded(ROOT / PRIOR_MANIFEST_PATH, MAX_MANIFEST_BYTES)
        if sha256(historical).hexdigest() != PRIOR_MANIFEST_SHA256:
            raise ValueError("historical reproducibility manifest fingerprint mismatch")
        original = _parse_manifest(historical)
        if any(hashes[name] != original["sha256"][name] for name in ORIGINAL_CORPUS_PATHS):
            raise ValueError("original corpus fingerprint mismatch")
        for name in FROZEN_PATHS:
            expected = hashes[name]
            if (type(expected) is not str or len(expected) != 64 or
                    any(c not in "0123456789abcdef" for c in expected)):
                raise ValueError(f"invalid recorded fingerprint: {name}")
            if _hash_file(ROOT / name) != expected:
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
    observed = _read_bounded(ROOT / "tests" / "fixtures" / "opennam" / "observed.json", MAX_SELECTED_FILE_BYTES)
    if actual != observed:
        raise ValueError("reconstructed observations differ from frozen report")
    parsed = json.loads(actual)
    if parsed["exact_governor_outcome_and_reason_matches"] != parsed["case_count"]:
        raise ValueError("provisional engineering labels do not match")
    summary = {
        "format": "opennam-engineering-reproduction-summary-v2",
        "phase3_commit": manifest["phase3_commit"],
        "phase4_commit": manifest["phase4_commit"],
        "prior_manifest_sha256": manifest["prior_manifest_sha256"],
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
