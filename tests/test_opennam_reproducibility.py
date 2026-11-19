"""Checkout runner, hash-seed/cwd invariance and frozen artifact rejection."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "opennam_reproduce.py"
MANIFEST = Path("tests/fixtures/opennam/reproducibility.json")
OBSERVED = ROOT / "tests" / "fixtures" / "opennam" / "observed.json"


def run(runner, cwd, *args, seed="0"):
    return subprocess.run([sys.executable, str(runner), *args], cwd=cwd,
                          env={**os.environ, "PYTHONHASHSEED": seed},
                          capture_output=True, check=False, timeout=30)


@pytest.mark.parametrize("seed", ["0", "7", "31"])
def test_exact_checkout_runner_from_arbitrary_directory(tmp_path, seed):
    actual = run(RUNNER, tmp_path, seed=seed)
    assert actual.returncode == 0, actual.stderr
    assert actual.stderr == b""
    assert actual.stdout == OBSERVED.read_bytes()


def test_summary_and_manifest_scope(tmp_path):
    result = run(RUNNER, tmp_path, "--summary")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["case_count"] == summary["exact_governor_outcome_and_reason_matches"] == 40
    manifest = json.loads((ROOT / MANIFEST).read_bytes())
    assert summary["verified_artifacts"] == len(manifest["sha256"])
    assert summary["observation_sha256"] == manifest["sha256"][str(OBSERVED.relative_to(ROOT).as_posix())]


@pytest.mark.parametrize("target", [
    "ailee/domains/opennam/governor.py",
    "tests/fixtures/opennam/cases.json",
    "tests/fixtures/opennam/expected.json",
    "tests/fixtures/opennam/observed.json",
    "tests/opennam_evaluation.py",
    "ailee/domains/opennam/BENCHMARKS.md",
])
def test_frozen_source_data_and_method_tampering_rejected_before_replay(tmp_path, target):
    manifest = json.loads((ROOT / MANIFEST).read_bytes())
    # This is a tiny inert copy, not the checked-out repository. No mutate/restore
    # cycle on tracked files and no import if frozen data differs.
    for relative in [*manifest["sha256"], MANIFEST.as_posix()]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    with (tmp_path / target).open("ab") as file:
        file.write(b"\nsynthetic integrity test\n")
    result = run(tmp_path / "scripts" / RUNNER.name, tmp_path)
    assert result.returncode == 1
    assert result.stdout == b""
    assert b"reproduction failed" in result.stderr
