"""Phase 4 research harness integrity, metric arithmetic and reproducibility."""

import ast
import copy
import json
import os
import subprocess
import sys

import pytest

from scripts import health_safety_benchmark as benchmark
from tests import health_safety_phase3 as phase3


RESULTS = phase3.FIXTURES / "phase4_benchmark_results.json"
REPRODUCIBILITY = phase3.FIXTURES / "phase4_reproducibility.json"


def _archive():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _arithmetic_pair(case_id, expected, actual, *, immediate=False, preserved=True,
                     integrity=None, integrity_actual=None, privacy=None, privacy_actual=None):
    # Pure metric arithmetic examples, never submitted to an evaluator or used
    # as decisions/credentials. These labels test the denominators and errors.
    ref_decision = {
        "outcome": expected, "reason_codes": ["REFERENCE"],
        "support": benchmark.IMMEDIATE if immediate else "ORDINARY_HELP",
        "support_template_id": "REFERENCE_TEMPLATE", "permitted_artifact": None,
    }
    observed_decision = dict(
        ref_decision, outcome=actual,
        support="ORDINARY_HELP" if immediate and not preserved else ref_decision["support"],
    )
    reference = {"fields": {
        "decision": ref_decision, "integrity_check": integrity, "privacy_clear": privacy,
        "quiet": True, "replay_equal": None,
    }, "benign_control": expected == "ALLOW"}
    row = {
        "id": case_id, "status": "EXECUTED",
        "observed": {"decision": observed_decision, "integrity_check": integrity_actual,
                     "privacy_clear": privacy_actual, "quiet": True},
        "contract_match": expected == actual and (not immediate or preserved)
        and integrity == integrity_actual and privacy == privacy_actual,
        "replay_mismatches": 0, "replay_comparisons": 2,
    }
    return row, reference


def test_benchmark_metric_arithmetic():
    examples = (
        _arithmetic_pair("a", "ALLOW", "ALLOW"),
        _arithmetic_pair("b", "ALLOW", "REVIEW", integrity=True, integrity_actual=False),
        _arithmetic_pair("c", "REVIEW", "ALLOW", immediate=True, preserved=False),
        _arithmetic_pair("d", "REVIEW", "DENY"),
        _arithmetic_pair("e", "DENY", "REVIEW", immediate=True, privacy=True, privacy_actual=False),
        _arithmetic_pair("f", "DENY", "DENY"),
    )
    rows = [row for row, _ in examples]
    refs = {row["id"]: reference for row, reference in examples}
    metrics = benchmark.compute_metrics(rows, refs)
    assert metrics["expected_distribution"] == {"ALLOW": 2, "REVIEW": 2, "DENY": 2}
    assert metrics["observed_distribution"] == {"ALLOW": 2, "REVIEW": 2, "DENY": 2}
    assert metrics["confusion_matrix_expected_rows"] == {
        "ALLOW": {"ALLOW": 1, "REVIEW": 1, "DENY": 0},
        "REVIEW": {"ALLOW": 1, "REVIEW": 0, "DENY": 1},
        "DENY": {"ALLOW": 0, "REVIEW": 1, "DENY": 1},
    }
    assert metrics["false_allow_rate"] == benchmark.rate(1, 4)
    assert metrics["false_block_rate"] == benchmark.rate(1, 2)
    assert metrics["three_way_agreement"] == benchmark.rate(2, 6)
    assert metrics["incorrect_review_deny"] == benchmark.rate(2, 4)
    assert metrics["immediate_referral_failures"] == benchmark.rate(1, 2)
    assert metrics["withheld_immediate_referral_failures"] == benchmark.rate(1, 2)
    assert metrics["artifact_integrity_mismatches"] == benchmark.rate(1, 1)
    assert metrics["privacy_canary_incidents"] == benchmark.rate(1, 1)
    assert metrics["deterministic_repeat_mismatches"] == benchmark.rate(0, 12)
    assert metrics["false_allow_rate"]["percent"] == 25.0
    assert metrics["false_block_rate"]["percent"] == 50.0


def test_benchmark_zero_denominator_is_undefined():
    assert benchmark.rate(0, 0) == {"numerator": 0, "denominator": 0, "percent": None}
    assert benchmark.rate(1, 3)["percent"] == 33.333333
    row, reference = _arithmetic_pair("only-allow", "ALLOW", "ALLOW")
    metrics = benchmark.compute_metrics([row], {row["id"]: reference})
    for name in ("false_allow_rate", "immediate_referral_failures", "artifact_integrity_mismatches",
                 "privacy_canary_incidents", "canonical_replay_mismatches"):
        assert metrics[name]["percent"] is None and metrics[name]["denominator"] == 0
    row, reference = _arithmetic_pair("only-deny", "DENY", "DENY")
    metrics = benchmark.compute_metrics([row], {row["id"]: reference})
    assert metrics["false_block_rate"]["percent"] is None
    with pytest.raises(benchmark.BenchmarkError, match="INVALID_METRIC_ARITHMETIC"):
        benchmark.rate(1, 0)


def test_benchmark_rejects_incomplete_metric_inputs():
    row, reference = _arithmetic_pair("case", "DENY", "DENY")
    with pytest.raises(benchmark.BenchmarkError, match="METRIC_CASE_COVERAGE"):
        benchmark.compute_metrics([], {"case": reference})
    with pytest.raises(benchmark.BenchmarkError, match="INCOMPLETE_MEASUREMENT"):
        benchmark.compute_metrics([dict(row, status="EXECUTION_ERROR")], {"case": reference})


def _development_case(case_id):
    scenarios, labels, cases, _ = benchmark.load_development()
    case = next(item for item in cases if item["id"] == case_id)
    ref = benchmark.reference_development(scenarios, labels, case)
    return scenarios, case, ref


@pytest.mark.parametrize("case_id,invalid", (
    ("dev-consent-missing", False),
    ("dev-malformed-candidate", True),
))
def test_benchmark_expected_denial_is_a_valid_observation(case_id, invalid):
    scenarios, case, reference = _development_case(case_id)
    row = benchmark.execute_case(case, reference, lambda item: benchmark.observe_development(scenarios, item), 3)
    assert row["status"] == "EXECUTED"
    assert row["observed"]["decision"]["outcome"] == "DENY"
    assert row["contract_match"] and row["replay_mismatches"] == 0
    assert reference["expected_invalid_input"] is invalid


def test_benchmark_execution_error_is_not_a_withholding_verdict():
    _, case, reference = _development_case("dev-consent-missing")
    def fail(_):
        raise RuntimeError("SYNTHETIC_RUNNER_FAILURE_CANARY")
    result = benchmark.summarize_partition(
        [case], {case["id"]: reference}, fail, 2, [], {},
    )
    assert result["counts"]["runner_errors"] == 1
    assert result["counts"]["executed"] == 0
    assert result["metrics"] is None
    assert result["observations"] == [{
        "id": case["id"], "status": "EXECUTION_ERROR", "error": "CASE_EVALUATION_FAILED",
    }]
    assert benchmark.exit_status({"partitions": {"development": result}}) == 2


@pytest.mark.parametrize("mode,expected_status", (("mismatch", 1), ("nonrepeatable", 3)))
def test_benchmark_result_integrity_failure_status(mode, expected_status):
    # Alter only test-harness observation records, not the frozen governor or
    # fixture inputs. An unexpected observation must not be a successful run.
    scenarios, case, reference = _development_case("dev-consent-missing")
    valid = benchmark.observe_development(scenarios, case)
    observations = [copy.deepcopy(valid) for _ in range(2)]
    observations[1]["quiet"] = False
    if mode == "mismatch":
        observations[0]["quiet"] = False
    pending = iter(observations)
    partition = benchmark.summarize_partition(
        [case], {case["id"]: reference}, lambda _: next(pending), 2, [], {},
    )
    assert partition["counts"]["runner_errors"] == 0
    assert benchmark.exit_status({"partitions": {"development": partition}}) == expected_status


def test_benchmark_frozen_input_integrity_and_reserved_status():
    benchmark.verify_frozen_inputs()
    scenarios, labels, cases, reserved = benchmark.load_development()
    assert len(cases) == 49 and len(reserved) == 4
    assert set(reserved).isdisjoint(labels["development"])
    assert all(labels["reserved_for_adjudication"][key] == "AWAITING_INDEPENDENT_ADJUDICATION" for key in reserved)
    assert scenarios["corpus"]["corpus_licensing"] == "PENDING_REVIEW"
    assert len(phase3.CASES) == 62
    assert phase3.SCENARIOS["corpus"]["corpus_licensing"] == "PENDING_REVIEW"
    with pytest.raises(benchmark.BenchmarkError, match="UNKNOWN_PARTITION"):
        benchmark.evaluate("other", 3)
    with pytest.raises(benchmark.BenchmarkError, match="REPEAT_COUNT_BOUND"):
        benchmark.evaluate("all", 1)


@pytest.mark.parametrize("selection,selected,excluded", (("development", 49, 4), ("phase3", 62, 0)))
def test_benchmark_explicit_partition_selection(selection, selected, excluded):
    result = benchmark.evaluate(selection, 2)
    assert set(result["partitions"]) == {selection}
    partition = result["partitions"][selection]
    assert partition["counts"]["selected_labeled"] == selected
    assert partition["counts"]["excluded"] == excluded
    assert partition["counts"]["executed"] == selected
    assert partition["counts"]["runner_errors"] == 0
    assert benchmark.exit_status(result) == 0


def test_benchmark_record_matches_recomputed_results():
    # This test is also executed in all hosted supported-runtime CI jobs.
    archived = _archive()
    assert archived["exit_status"] == 0
    assert archived["results"]["phase3_baseline_sha"] == benchmark.PHASE3_SHA
    assert archived["results_sha256"] == benchmark.digest(archived["results"])
    fresh = benchmark.evaluate("all", 3)
    assert benchmark.exit_status(fresh) == 0
    assert benchmark.digest(fresh) == archived["results_sha256"]
    assert fresh == archived["results"]


@pytest.mark.parametrize("seed", phase3.HASH_SEEDS)
def test_benchmark_fresh_process_seed_reproducibility(seed):
    run = subprocess.run(
        [sys.executable, "-m", "scripts.health_safety_benchmark", "--partition", "all", "--repeats", "3"],
        cwd=phase3.ROOT, env=dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, timeout=45, check=False,
    )
    assert run.returncode == 0 and run.stderr == b""
    observed = json.loads(run.stdout)
    assert observed["environment"]["python_hash_seed"] == seed
    assert observed["exit_status"] == 0
    assert observed["results_sha256"] == benchmark.digest(observed["results"]) == _archive()["results_sha256"]


def test_benchmark_output_never_overwrites_existing_evidence(tmp_path, capsys):
    existing = tmp_path / "existing.json"
    existing.write_text("preserved", encoding="utf-8")
    result = benchmark.main(["--partition", "development", "--repeats", "2", "--output", str(existing)])
    assert result == 2
    assert existing.read_text(encoding="utf-8") == "preserved"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "BENCHMARK_INTEGRITY_OR_EXECUTION_ERROR\n"


def test_benchmark_reproducibility_record_and_docs():
    record = json.loads(REPRODUCIBILITY.read_text(encoding="utf-8"))
    expected_digest = _archive()["results_sha256"]
    assert record["results_sha256"] == expected_digest
    assert len(record["runs"]) == 6
    assert {item["python_hash_seed"] for item in record["runs"]} == set(phase3.HASH_SEEDS)
    assert len({item["python_version"] for item in record["runs"]}) == 2
    for item in record["runs"]:
        assert item["exit_status"] == 0 and item["results_sha256"] == expected_digest
    model_docs = phase3.ROOT / "ailee" / "domains" / "health_safety"
    provenance = (model_docs / "CORPUS_PROVENANCE.md").read_text(encoding="utf-8")
    for path, checksum in benchmark.FROZEN_SHA256.items():
        if path.startswith("tests/fixtures/health_safety/"):
            assert checksum in provenance
    assert expected_digest in (model_docs / "BENCHMARK_RESULTS.md").read_text(encoding="utf-8")


def test_benchmark_python39_syntax():
    files = (phase3.ROOT / "scripts" / "health_safety_benchmark.py",
             phase3.ROOT / "tests" / "test_health_safety_benchmark.py")
    for path in files:
        code = path.read_text(encoding="utf-8")
        ast.parse(code, filename=str(path), feature_version=(3, 9))
        compile(code, str(path), "exec")
