"""Checkout runner, hash-seed/cwd invariance and frozen artifact rejection."""

import json
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import opennam_reproduce as verifier

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "opennam_reproduce.py"
MANIFEST = Path("tests/fixtures/opennam/reproducibility.json")
OBSERVED = ROOT / "tests" / "fixtures" / "opennam" / "observed.json"
FROZEN_ARCHIVE = "tests/fixtures/opennam/reproducibility-v1.json"


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
    assert summary["format"] == "opennam-engineering-reproduction-summary-v2"
    assert summary["phase4_commit"] == "f6090ea94d093e08fcfc461501c957280e5cb0a3"
    assert summary["prior_manifest_sha256"] == "b87a85b86e3556c0117cf242179fb7e21daa5df013879f00764c81c462ce69cc"
    manifest = json.loads((ROOT / MANIFEST).read_bytes())
    assert summary["verified_artifacts"] == len(manifest["sha256"])
    assert summary["observation_sha256"] == manifest["sha256"][str(OBSERVED.relative_to(ROOT).as_posix())]
    assert summary["verified_artifacts"] == 21
    assert manifest["sha256"][FROZEN_ARCHIVE] == summary["prior_manifest_sha256"]


def copy_selected(tmp_path):
    manifest = json.loads((ROOT / MANIFEST).read_bytes())
    for relative in [*manifest["sha256"], MANIFEST.as_posix()]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return manifest


@pytest.mark.parametrize("target", [
    "ailee/domains/opennam/governor.py",
    "tests/fixtures/opennam/cases.json",
    "tests/fixtures/opennam/expected.json",
    "tests/fixtures/opennam/observed.json",
    "tests/opennam_evaluation.py",
    "ailee/domains/opennam/BENCHMARKS.md",
    "ailee/domains/opennam/adapters.py",
    "ailee/domains/opennam/ADAPTERS.md",
    "tests/test_opennam_adapters.py",
    FROZEN_ARCHIVE,
])
def test_frozen_source_data_and_method_tampering_rejected_before_replay(tmp_path, target):
    # This is a tiny inert copy, not the checked-out repository. No mutate/restore
    # cycle on tracked files and no import if frozen data differs.
    copy_selected(tmp_path)
    with (tmp_path / target).open("ab") as file:
        file.write(b"\nsynthetic integrity test\n")
    result = run(tmp_path / "scripts" / RUNNER.name, tmp_path)
    assert result.returncode == 1
    assert result.stdout == b""
    assert b"reproduction failed" in result.stderr


def configured_copy(tmp_path, monkeypatch):
    manifest = copy_selected(tmp_path)
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "MANIFEST", tmp_path / MANIFEST)
    # Verify the intact copy first so a later failure cannot merely be caused
    # by missing selected files or the absent full runtime in this small copy.
    assert verifier.verify_artifacts() == manifest
    return manifest


@pytest.mark.parametrize("mutation,diagnostic", [
    (lambda b: b'{"format":"opennam-reproduction-v2",' + b[1:], "duplicate"),
    (lambda b: b.replace(b'"sha256": {', b'"sha256": {"ailee/domains/opennam/ADAPTERS.md":"duplicate",', 1), "duplicate"),
    (lambda b: b.decode("utf-8").encode("utf-16"), "malformed"),
    (lambda b: b"\xff", "malformed"),
    (lambda b: b" " * (verifier.MAX_MANIFEST_BYTES + 1), "oversized"),
    (lambda b: b[:-4], "Expecting"),
    (lambda b: b'{"x":NaN}', "numeric"),
    (lambda b: b'{"x":1e300}', "numeric"),
    (lambda b: b'{"x":' + b'9' * 1000 + b'}', "numeric"),
])
def test_manifest_bytes_fail_closed_before_replay(tmp_path, monkeypatch, mutation, diagnostic):
    configured_copy(tmp_path, monkeypatch)
    path = tmp_path / MANIFEST
    path.write_bytes(mutation(path.read_bytes()))
    with pytest.raises(ValueError, match=diagnostic):
        verifier.verify_artifacts()


@pytest.mark.parametrize("mutation,diagnostic", [
    (lambda m: {**m, "extra": "value"}, "invalid reproducibility manifest"),
    (lambda m: {**m, "format": "opennam-reproduction-v1"}, "version or phase"),
    (lambda m: {**m, "phase3_commit": "f" * 40}, "version or phase"),
    (lambda m: {**m, "phase4_commit": "f" * 40}, "version or phase"),
    (lambda m: {**m, "prior_manifest_sha256": "f" * 64}, "version or phase"),
    (lambda m: {**m, "sha256": []}, "path set"),
    (lambda m: {**m, "sha256": {**m["sha256"], "../outside": "f" * 64}}, "path set"),
    (lambda m: {**m, "sha256": {k: v for k, v in m["sha256"].items() if k != FROZEN_ARCHIVE}}, "path set"),
    (lambda m: {**m, "sha256": {**m["sha256"], FROZEN_ARCHIVE: "f" * 64}}, "historical"),
    (lambda m: {**m, "sha256": {**m["sha256"], "ailee/domains/opennam/adapters.py": True}}, "fingerprint"),
    (lambda m: {**m, "sha256": {**m["sha256"], "ailee/domains/opennam/adapters.py": "F" * 64}}, "fingerprint"),
    (lambda m: {**m, "sha256": {**m["sha256"], "ailee/domains/opennam/adapters.py": "f" * 63}}, "fingerprint"),
    (lambda m: {**m, "sha256": {**m["sha256"], "ailee/domains/opennam/adapters.py": "f" * 64}}, "frozen artifact mismatch"),
])
def test_manifest_shape_paths_revision_and_hashes(tmp_path, monkeypatch, mutation, diagnostic):
    manifest = configured_copy(tmp_path, monkeypatch)
    (tmp_path / MANIFEST).write_text(json.dumps(mutation(manifest)), encoding="utf-8")
    with pytest.raises(ValueError, match=diagnostic):
        verifier.verify_artifacts()


@pytest.mark.parametrize("target", [MANIFEST.as_posix(), "ailee/domains/opennam/adapters.py"])
def test_manifest_and_selected_file_missing(tmp_path, monkeypatch, target):
    configured_copy(tmp_path, monkeypatch)
    (tmp_path / target).unlink()
    with pytest.raises(ValueError, match="missing or malformed"):
        verifier.verify_artifacts()


def test_selected_file_hashing_is_size_bounded(tmp_path, monkeypatch):
    configured_copy(tmp_path, monkeypatch)
    target = tmp_path / "ailee/domains/opennam/adapters.py"
    target.write_bytes(b"x" * (verifier.MAX_SELECTED_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="oversized"):
        verifier.verify_artifacts()


@pytest.mark.parametrize("target", verifier.ORIGINAL_CORPUS_PATHS)
def test_regenerated_manifest_cannot_rebaseline_original_corpus(tmp_path, monkeypatch, target):
    manifest = configured_copy(tmp_path, monkeypatch)
    changed = (tmp_path / target).read_bytes() + b"\n"
    (tmp_path / target).write_bytes(changed)
    manifest["sha256"][target] = sha256(changed).hexdigest()
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="original corpus fingerprint"):
        verifier.verify_artifacts()


def test_controlled_cli_failure_for_bad_manifest(tmp_path):
    copy_selected(tmp_path)
    (tmp_path / MANIFEST).write_bytes(b'{"secret rejected content":NaN}')
    result = run(tmp_path / "scripts" / RUNNER.name, tmp_path)
    assert result.returncode == 1 and result.stdout == b""
    assert result.stderr == b"OpenNAM reproduction failed; check frozen checkout artifacts and observations.\n"
