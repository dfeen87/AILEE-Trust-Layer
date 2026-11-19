"""Phase 3 defensive contract conformance and independent observations."""

import ast
import hashlib
import inspect
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from ailee.domains import health_safety as hs
from tests import health_safety_phase3 as phase3


# Recorded before any Phase 3 observation. The original development corpus
# and separately authored Phase 3 scenarios/labels are not adjusted to fit
# the evaluator. Hosted tests also verify these exact bytes.
FROZEN_INPUT_SHA256 = {
    "scenarios.json": "ced7860b3e25f6f158092575029bb07fd88e26043337042ab1061162aca8e8ee",
    "expected.json": "b824a06427781d0deab5779c466999c17f5d95cf65e751d4977297b85757733e",
    "phase3_scenarios.json": "3f4bfc1c1950a411e34cf495f0b6e6d4169eaa042e526c5604f0e35ed9ed8a1d",
    "phase3_expected.json": "a5368d7ecf8563a3868549e3d8851686d137c0faaca2458b38049275adc30f0d",
}


def test_phase3_frozen_corpus_and_labels():
    phase3.validate_corpus()
    for filename, checksum in FROZEN_INPUT_SHA256.items():
        assert hashlib.sha256((phase3.FIXTURES / filename).read_bytes()).hexdigest() == checksum
    assert len(phase3.CASES) == 62
    development = phase3.read_fixture(phase3.FIXTURES / "scenarios.json")
    previous_ids = {
        row["id"] for key in ("development", "reserved_for_adjudication")
        for row in development[key]
    }
    assert len(previous_ids) == 53
    assert previous_ids.isdisjoint({case["id"] for case in phase3.CASES})
    assert phase3.SCENARIOS["corpus"]["independent_adjudication"] == "NOT_PERFORMED"


@pytest.mark.parametrize("case", phase3.CASES, ids=lambda case: case["id"])
def test_phase3_defensive_case(case):
    observed = phase3.observe_case(case)
    # On a failure expose only controlled field names, not fixture data or
    # exception text. The separate observations record contains actual results.
    assert phase3.mismatches(case, observed) == ()


def test_phase3_typed_contract_and_integrity_surface():
    case = next(row for row in phase3.CASES if row["id"] == "p3-benign-education")
    typed = phase3.build_request(case)
    assert type(typed) is hs.HealthRequest
    with pytest.raises(hs.ContractError) as error:
        replace(typed, purpose="GENERAL_EDUCATION")
    assert error.value.reason is hs.HealthReason.MALFORMED_SCHEMA
    assert str(error.value) == "MALFORMED_SCHEMA"
    assert "GENERAL_EDUCATION" not in repr(error.value)
    signature = inspect.signature(hs.HealthSafetyGovernor.verify_synthetic_artifact)
    assert tuple(signature.parameters) == ("self", "request", "context", "artifact_id", "payload")
    for name in ("context", "artifact_id", "payload"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert not hasattr(phase3.build_governor(), "release")


def test_phase3_observation_record_reproduced():
    archived = phase3.read_fixture(phase3.OBSERVED_FILE)
    assert archived["case_count"] == len(phase3.CASES)
    assert hashlib.sha256(phase3.observation_json(phase3.observe_all()).encode("utf-8")).hexdigest() == (
        hashlib.sha256(phase3.observation_json(archived).encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize("seed", phase3.HASH_SEEDS)
def test_phase3_full_replay_across_python_hash_seeds(seed):
    # Reproduce *complete* decision/audit projections in fresh Python processes.
    # All Phase 3 cases, including independently verified review and denial
    # cases, are included; the three replay cases additionally reverse order.
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run(
        [sys.executable, "-m", "tests.health_safety_phase3"],
        cwd=phase3.ROOT, env=env, capture_output=True, timeout=30, check=False,
    )
    assert run.returncode == 0
    assert run.stderr == b""
    assert hashlib.sha256(run.stdout).hexdigest() == hashlib.sha256(phase3.OBSERVED_FILE.read_bytes()).hexdigest()


def test_phase3_python39_syntax():
    for filename in ("health_safety_phase3.py", "test_health_safety_phase3.py"):
        source = (phase3.ROOT / "tests" / filename).read_text(encoding="utf-8")
        ast.parse(source, filename=filename, feature_version=(3, 9))
        compile(source, filename, "exec")
