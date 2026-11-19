"""Offline, fixed-corpus Phase 4 Health Safety conformance measurements.

Run from the exact research checkout, not as a production integration:
python -m scripts.health_safety_benchmark --partition all --repeats 3
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from ailee.domains import health_safety as hs
from tests import health_safety_phase3 as phase3

ROOT = phase3.ROOT
FIXTURES = phase3.FIXTURES
PHASE3_SHA = "92379d20205c72a5dd6cc5638d3650e3db036d48"
FORMAT = "ahs-phase4-offline-benchmark-v1"
METHODOLOGY = "ailee/domains/health_safety/BENCHMARKS.md"
METHODOLOGY_SHA256 = "7b3521185e672ba020ee3f5357eb1db576cfa6093ed4a6c384813d981ffc6cf1"
FROZEN_SHA256 = {
    "ailee/domains/health_safety/health_safety.py": "7801f9eba68b64d461ace349c3c2b84cb78650598c96f20ac85ca3861241d687",
    "ailee/domains/health_safety/__init__.py": "f347c4817b87533e2d41425eded0b483e866623484bf6f99ebf63f4b2cc95222",
    "tests/test_health_safety_domain.py": "9b1ae126a0644fd7677c64d2a8fc5eb993eb48790cc86d2c3318ea83516cb9d1",
    "tests/health_safety_phase3.py": "724ff2b3409c35e4077fd0784cb29af812e583b7095d6c95d3b2a4383a207caf",
    "tests/test_health_safety_phase3.py": "33673a6b0ebb57bd54ae6fa005801c5614979573180834d468c8d9cbbe1054ae",
    "tests/fixtures/health_safety/scenarios.json": "ced7860b3e25f6f158092575029bb07fd88e26043337042ab1061162aca8e8ee",
    "tests/fixtures/health_safety/expected.json": "b824a06427781d0deab5779c466999c17f5d95cf65e751d4977297b85757733e",
    "tests/fixtures/health_safety/phase3_scenarios.json": "3f4bfc1c1950a411e34cf495f0b6e6d4169eaa042e526c5604f0e35ed9ed8a1d",
    "tests/fixtures/health_safety/phase3_expected.json": "a5368d7ecf8563a3868549e3d8851686d137c0faaca2458b38049275adc30f0d",
    "tests/fixtures/health_safety/phase3_observed.json": "b75202fc5ec804e32f0040cdca68970a38a7dd044fce69c7f65b8d08615e50b1",
}
OUTCOMES = ("ALLOW", "REVIEW", "DENY")
IMMEDIATE = "IMMEDIATE_PUBLIC_REFERRAL"
INVALID_REASONS = frozenset((
    "MALFORMED_SCHEMA", "UNSUPPORTED_SCHEMA", "INVALID_CONTEXT",
    "INVALID_TIMESTAMP", "INVALID_POLICY", "INVALID_CATALOG",
))
DEVELOPMENT_CANARY = "SYNTHETIC_PRIVATE_CANARY_NOT_REAL_6042"
DEVELOPMENT_PRIVACY_IDS = frozenset((
    "dev-privacy-marker", "dev-privacy-canary-input",
    "dev-privacy-canary-key", "dev-evaluator-exception-urgent",
))
DECISION_FIELDS = frozenset((
    "outcome", "reason_codes", "permitted_artifact", "support", "support_template_id",
    "scope", "policy", "catalog", "evaluation_time",
))


class BenchmarkError(ValueError):
    """Controlled benchmark failure, distinct from a candidate DENY."""


def require(condition, code):
    if not condition:
        raise BenchmarkError(code)


def digest(value):
    """Digest a complete deterministic JSON value (not runtime metadata)."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_frozen_inputs():
    for path, expected in dict(FROZEN_SHA256, **{METHODOLOGY: METHODOLOGY_SHA256}).items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                "FROZEN_INPUT_CHECKSUM_MISMATCH")


def load_development():
    scenarios = phase3.read_fixture(FIXTURES / "scenarios.json")
    labels = phase3.read_fixture(FIXTURES / "expected.json")
    require(scenarios["schema_version"] == labels["schema_version"] == "1.0", "DEVELOPMENT_SCHEMA")
    require(scenarios["corpus"]["id"] == labels["corpus_id"], "DEVELOPMENT_ID")
    require(scenarios["corpus"]["synthetic_only"] is True, "DEVELOPMENT_NOT_SYNTHETIC")
    cases = tuple(sorted(scenarios["development"], key=lambda item: item["id"]))
    ids = {case["id"] for case in cases}
    reserved = tuple(sorted(case["id"] for case in scenarios["reserved_for_adjudication"]))
    require(len(cases) == len(ids) == 49 and ids == set(labels["development"]), "DEVELOPMENT_LABELS")
    require(len(reserved) == len(set(reserved)) == 4 and ids.isdisjoint(reserved), "RESERVED_PARTITION")
    require(labels["reserved_for_adjudication"] ==
            {case_id: "AWAITING_INDEPENDENT_ADJUDICATION" for case_id in reserved}, "RESERVED_STATUS")
    return scenarios, labels, cases, reserved


def _time(value):
    try:
        return phase3.fixture_time(value)
    except phase3.FixtureError:
        raise hs.ContractError(hs.HealthReason.INVALID_TIMESTAMP) from None


def _policy(scenarios):
    raw = scenarios["policy"]
    return hs.HealthPolicy(
        hs.PolicyRef(**raw["ref"]), hs.CatalogRef(**raw["catalog"]),
        tuple(hs.HealthPurpose(item) for item in raw["allowed_purposes"]),
        raw["min_independent_sources_per_claim"],
    )


def _catalog(scenarios, fault):
    raw = scenarios["catalog"]
    artifacts = tuple(hs.ReviewedArtifact(
        hs.ArtifactRef(**row["artifact"]), hs.FixtureScope(**row["scope"]),
        hs.HealthPurpose(row["purpose"]), hs.ArtifactAssessment(row["assessment"]),
        tuple(row["claim_ids"]), tuple(hs.SyntheticInput(item) for item in row["synthetic_inputs"]),
    ) for row in raw["artifacts"])
    evidence = tuple(hs.ReviewedEvidence(
        evidence_id=row["evidence_id"], artifact=hs.ArtifactRef(**row["artifact"]),
        scope=hs.FixtureScope(**row["scope"]), purpose=hs.HealthPurpose(row["purpose"]),
        claim_id=row["claim_id"], source_group_id=row["source_group_id"],
        relation=hs.EvidenceRelation(row["relation"]), valid_from=_time(row["valid_from"]),
        valid_until=_time(row["valid_until"]),
    ) for row in raw["evidence"])
    if fault == "DUPLICATE_CATALOG_ARTIFACT":
        artifacts += (artifacts[0],)
    return hs.HealthCatalog(
        raw["schema_version"], hs.CatalogRef(**raw["ref"]), raw["review_id"],
        hs.CatalogReviewStage(raw["review_stage"]), artifacts, evidence,
    )


def _context(raw):
    # Same independent fixture assembly contract as the frozen Phase 2 tests.
    row = raw["consent"]
    consent = None if row is None else hs.SyntheticConsent(
        hs.FixtureScope(**row["scope"]), hs.CatalogRef(**row["catalog"]),
        hs.ArtifactRef(**row["artifact"]), hs.HealthPurpose(row["purpose"]),
        hs.ConsentState(row["state"]), _time(row["valid_from"]), _time(row["valid_until"]),
    )
    urgency = raw["urgency"]
    return hs.HealthContext(
        None if raw["scope"] is None else hs.FixtureScope(**raw["scope"]),
        None if raw["evaluation_time"] is None else _time(raw["evaluation_time"]),
        consent,
        None if urgency is None else hs.UrgencyFact(hs.FixtureScope(**urgency["scope"]),
                                                   hs.UrgencyLevel(urgency["level"])),
        hs.ContextState(raw["state"]),
    )


def _development_request(scenarios, case):
    raw = copy.deepcopy(scenarios["request_templates"][case["request_template"]])
    raw.update(copy.deepcopy(case["request_overrides"]))
    return raw


def _development_context(scenarios, case):
    raw = copy.deepcopy(scenarios["context_templates"][case["context_template"]])
    raw.update(copy.deepcopy(case["context_overrides"]))
    return raw


def observe_development(scenarios, case):
    """Evaluate a frozen development case without consulting its labels."""
    request = _development_request(scenarios, case)
    raw_context = _development_context(scenarios, case)
    fault = case["harness_fault"]
    require(fault in ("NONE", "DUPLICATE_CATALOG_ARTIFACT", "EVALUATOR_EXCEPTION"),
            "UNREGISTERED_DEVELOPMENT_FAULT")
    schema = {"request": "VALID", "context": "VALID", "catalog": "VALID"}
    exception_views = []
    with phase3._capture() as (out, err, log):
        try:
            hs.HealthRequest.from_dict(request)
        except hs.ContractError as exc:
            schema["request"] = exc.reason.value
            exception_views.extend((str(exc), repr(exc), repr(exc.args)))
        try:
            context = _context(raw_context)
        except hs.ContractError as exc:
            # A known fixture clock failure is represented by absent time,
            # keeping independently established urgency/scope, exactly as Phase 2.
            require(exc.reason is hs.HealthReason.INVALID_TIMESTAMP, "UNEXPECTED_CONTEXT_ASSEMBLY_ERROR")
            schema["context"] = exc.reason.value
            exception_views.extend((str(exc), repr(exc), repr(exc.args)))
            context = _context(dict(raw_context, evaluation_time=None))
        try:
            catalog = _catalog(scenarios, fault)
        except hs.ContractError as exc:
            require(exc.reason is hs.HealthReason.INVALID_CATALOG, "UNEXPECTED_CATALOG_ASSEMBLY_ERROR")
            schema["catalog"] = exc.reason.value
            exception_views.extend((str(exc), repr(exc), repr(exc.args)))
            catalog = None
        governor = hs.HealthSafetyGovernor(policy=_policy(scenarios), catalog=catalog)
        injected = (
            patch.object(hs.HealthSafetyGovernor, "_candidate_reason",
                         side_effect=RuntimeError(DEVELOPMENT_CANARY))
            if fault == "EVALUATOR_EXCEPTION" else nullcontext()
        )
        with injected:
            decision = governor.evaluate(request, context=context)
        require(type(decision) is hs.HealthDecision, "NON_DECISION_RESULT")
        decision_data = phase3.projection(decision)
        audit_data = phase3.projection(decision.audit)
        views = (repr(decision), repr(decision.audit),
                 json.dumps(decision_data, sort_keys=True), json.dumps(audit_data, sort_keys=True))
    captured = (out.getvalue(), err.getvalue(), log.getvalue())
    privacy_clear = None
    if case["id"] in DEVELOPMENT_PRIVACY_IDS:
        canaries = [DEVELOPMENT_CANARY, "synthetic-private-canary-not-real-6042"]
        # Inspect withheld identifier/digest as synthetic canaries, never as
        # authority. No rejected content or captured diagnostic is persisted.
        rejected = request.get("artifact")
        if type(rejected) is dict:
            canaries.extend(value for key in ("artifact_id", "sha256")
                            for value in (rejected.get(key),) if type(value) is str and value)
        privacy_clear = not any(token in view for token in canaries
                                for view in (*views, *exception_views, *captured))
    return {
        "id": case["id"], "decision": decision_data, "audit": audit_data,
        "schema_checks": schema, "privacy_clear": privacy_clear,
        "integrity_check": None, "replay_equal": None,
        "quiet": all(not text for text in captured),
    }


def _reference_fields(decision, **auxiliary):
    return dict(
        decision=decision,
        audit={key: value for key, value in decision.items() if key != "permitted_artifact"},
        quiet=True, **auxiliary,
    )


def reference_development(scenarios, labels, case):
    label = labels["development"][case["id"]]
    raw_context = _development_context(scenarios, case)
    schema = label["schema_checks"]
    decision = {
        "outcome": label["candidate"], "reason_codes": list(label["reason_codes"]),
        "support": label["support"], "support_template_id": label["support_template_id"],
        "permitted_artifact": label["permitted_artifact"], "policy": scenarios["policy"]["ref"],
        "catalog": scenarios["catalog"]["ref"] if schema["catalog"] == "VALID" else None,
        "scope": raw_context["scope"],
        "evaluation_time": raw_context["evaluation_time"] if schema["context"] == "VALID" else None,
    }
    return {
        "fields": _reference_fields(
            decision, schema_checks=schema, integrity_check=None, replay_equal=None,
            privacy_clear=True if case["id"] in DEVELOPMENT_PRIVACY_IDS else None,
        ),
        "expected_invalid_input": decision["reason_codes"][0] in INVALID_REASONS,
        "controlled_fault": case["harness_fault"] == "EVALUATOR_EXCEPTION",
        "benign_control": decision["outcome"] == "ALLOW",
    }


def reference_phase3(case, archive):
    case_id = case["id"]
    decision = phase3.expected_decision(case)
    return {
        "fields": _reference_fields(
            decision, integrity_check=phase3.EXPECTED["integrity_checks"].get(case_id),
            privacy_clear=phase3.EXPECTED["privacy_checks"].get(case_id),
            replay_equal=phase3.EXPECTED["replay_checks"].get(case_id),
            controlled_decoder_reason=phase3.EXPECTED["controlled_decoder_reasons"].get(case_id),
            prior_candidate=phase3.EXPECTED["prior_candidates"].get(case_id),
        ),
        "frozen_observation": archive[case_id],
        "expected_invalid_input": decision["reason_codes"][0] in INVALID_REASONS,
        "controlled_fault": case.get("fault") == "controlled-evaluator-error",
        "benign_control": case["category"] == "benign-controls",
    }


def _valid_observation(case_id, observed):
    require(type(observed) is dict and observed.get("id") == case_id, "OBSERVATION_ID")
    decision = observed.get("decision")
    audit = observed.get("audit")
    require(type(decision) is dict and set(decision) == DECISION_FIELDS, "OBSERVATION_DECISION_SHAPE")
    require(decision["outcome"] in OUTCOMES, "OBSERVATION_OUTCOME")
    require(type(audit) is dict and set(audit) == DECISION_FIELDS - {"permitted_artifact"},
            "OBSERVATION_AUDIT_SHAPE")


def execute_case(case, reference, observer, repeats):
    """Run one *fixed labeled* case; intentional DENY is a valid observation."""
    case_id = case["id"]
    try:
        observed = observer(case)
        _valid_observation(case_id, observed)
        first = digest(observed)
        hashes = [first]
        for _ in range(repeats - 1):
            repeated = observer(case)
            _valid_observation(case_id, repeated)
            hashes.append(digest(repeated))
        contract_match = all(key in observed and value == observed[key]
                             for key, value in reference["fields"].items())
        if "frozen_observation" in reference:
            contract_match = contract_match and observed == reference["frozen_observation"]
        return {
            "id": case_id, "status": "EXECUTED", "observed": observed,
            "contract_match": contract_match, "repeat_sha256": hashes,
            "replay_comparisons": repeats - 1,
            "replay_mismatches": sum(value != first for value in hashes[1:]),
        }
    except Exception:
        # Execution errors are controlled harness failures, never synthetic
        # authorization labels. Never persist the input or exception text.
        return {"id": case_id, "status": "EXECUTION_ERROR", "error": "CASE_EVALUATION_FAILED"}


def rate(numerator, denominator):
    require(type(numerator) is int and type(denominator) is int and 0 <= numerator <= denominator,
            "INVALID_METRIC_ARITHMETIC")
    return {
        "numerator": numerator, "denominator": denominator,
        "percent": None if denominator == 0 else round(100 * numerator / denominator, 6),
    }


def _distribution(values):
    return {outcome: sum(item == outcome for item in values) for outcome in OUTCOMES}


def compute_metrics(rows, references):
    """Calculate rates on complete, valid labeled observations only."""
    require(len(rows) == len(references) and {row["id"] for row in rows} == set(references),
            "METRIC_CASE_COVERAGE")
    require(all(row["status"] == "EXECUTED" for row in rows), "INCOMPLETE_MEASUREMENT")
    pairs = [(references[row["id"]], row["observed"], row) for row in rows]
    expected = [ref["fields"]["decision"]["outcome"] for ref, _, _ in pairs]
    observed = [actual["decision"]["outcome"] for _, actual, _ in pairs]
    total = len(rows)
    require(all(outcome in OUTCOMES for outcome in expected + observed), "METRIC_OUTCOME")
    expected_dist = _distribution(expected)
    observed_dist = _distribution(observed)
    matrix = {
        left: {right: sum(a == left and b == right for a, b in zip(expected, observed))
               for right in OUTCOMES}
        for left in OUTCOMES
    }
    withhold = expected_dist["REVIEW"] + expected_dist["DENY"]

    def mismatch_count(field):
        return sum(ref["fields"]["decision"][field] != actual["decision"][field]
                   for ref, actual, _ in pairs)

    def auxiliary_mismatches(field):
        eligible = [(ref, actual) for ref, actual, _ in pairs if ref["fields"].get(field) is not None]
        return rate(sum(ref["fields"][field] != actual.get(field) for ref, actual in eligible), len(eligible))

    def referral_failures(*, withheld_only=False):
        eligible = [
            (ref, actual) for ref, actual, _ in pairs
            if ref["fields"]["decision"]["support"] == IMMEDIATE
            and (not withheld_only or ref["fields"]["decision"]["outcome"] != "ALLOW")
        ]
        return rate(sum(actual["decision"]["support"] != IMMEDIATE for _, actual in eligible), len(eligible))

    benign = [(ref, actual, row) for ref, actual, row in pairs if ref["benign_control"]]
    return {
        "expected_distribution": expected_dist,
        "observed_distribution": observed_dist,
        "confusion_matrix_expected_rows": matrix,
        "false_allow_rate": rate(matrix["REVIEW"]["ALLOW"] + matrix["DENY"]["ALLOW"], withhold),
        "false_block_rate": rate(matrix["ALLOW"]["REVIEW"] + matrix["ALLOW"]["DENY"], expected_dist["ALLOW"]),
        "three_way_agreement": rate(sum(matrix[item][item] for item in OUTCOMES), total),
        "incorrect_review_deny": rate(matrix["REVIEW"]["DENY"] + matrix["DENY"]["REVIEW"], withhold),
        "reason_code_agreement": rate(total - mismatch_count("reason_codes"), total),
        "public_support_agreement": rate(total - mismatch_count("support"), total),
        "fixed_template_agreement": rate(total - mismatch_count("support_template_id"), total),
        "immediate_referral_failures": referral_failures(),
        "withheld_immediate_referral_failures": referral_failures(withheld_only=True),
        "artifact_permission_mismatches": rate(mismatch_count("permitted_artifact"), total),
        "artifact_integrity_mismatches": auxiliary_mismatches("integrity_check"),
        "privacy_canary_incidents": auxiliary_mismatches("privacy_clear"),
        "quiet_capture_failures": auxiliary_mismatches("quiet"),
        "canonical_replay_mismatches": auxiliary_mismatches("replay_equal"),
        "deterministic_repeat_mismatches": rate(sum(row["replay_mismatches"] for row in rows),
                                                 sum(row["replay_comparisons"] for row in rows)),
        "full_contract_agreement": rate(sum(row["contract_match"] for row in rows), total),
        "benign_controls": {
            "count": len(benign),
            "expected_distribution": _distribution([ref["fields"]["decision"]["outcome"] for ref, _, _ in benign]),
            "observed_distribution": _distribution([actual["decision"]["outcome"] for _, actual, _ in benign]),
            "full_contract_agreement": rate(sum(row["contract_match"] for _, _, row in benign), len(benign)),
        },
    }


def summarize_partition(cases, references, observer, repeats, excluded, profile):
    rows = [execute_case(case, references[case["id"]], observer, repeats) for case in cases]
    successful = sum(row["status"] == "EXECUTED" for row in rows)
    errors = len(rows) - successful
    counts = {
        "available": len(cases) + len(excluded), "selected_labeled": len(cases),
        "executed": successful, "excluded": len(excluded), "reserved_unlabelled": len(excluded),
        "independently_adjudicated": 0, "provisional_labeled": len(cases),
        "expected_invalid_input": sum(ref["expected_invalid_input"] for ref in references.values()),
        "controlled_fault": sum(ref["controlled_fault"] for ref in references.values()),
        "runner_errors": errors,
    }
    metrics = None if errors else compute_metrics(rows, references)
    return {
        "profile": profile, "counts": counts, "excluded": list(excluded),
        "metrics": metrics, "observations": rows,
    }


def evaluate(selection, repeats):
    require(selection in ("development", "phase3", "all"), "UNKNOWN_PARTITION")
    require(type(repeats) is int and 2 <= repeats <= 5, "REPEAT_COUNT_BOUND")
    verify_frozen_inputs()
    scenarios, labels, development, reserved = load_development()
    phase3.validate_corpus()
    archive = phase3.read_fixture(phase3.OBSERVED_FILE)
    archived_by_id = {row["id"]: row for row in archive["observations"]}
    require(len(archived_by_id) == archive["case_count"] == len(phase3.CASES)
            and set(archived_by_id) == {case["id"] for case in phase3.CASES}, "FROZEN_OBSERVATION_COVERAGE")
    partitions = {}
    if selection in ("development", "all"):
        references = {case["id"]: reference_development(scenarios, labels, case) for case in development}
        partitions["development"] = summarize_partition(
            development, references, lambda case: observe_development(scenarios, case), repeats,
            [{"id": case_id, "status": "AWAITING_INDEPENDENT_ADJUDICATION"} for case_id in reserved],
            {"corpus_id": scenarios["corpus"]["id"], "policy": scenarios["policy"]["ref"],
             "catalog": scenarios["catalog"]["ref"], "review_id": scenarios["catalog"]["review_id"],
             "review_stage": scenarios["catalog"]["review_stage"],
             "evaluation_time": "2026-01-15T12:00:00Z", "unavailable_time_cases": 2},
        )
    if selection in ("phase3", "all"):
        references = {case["id"]: reference_phase3(case, archived_by_id) for case in phase3.CASES}
        partitions["phase3"] = summarize_partition(
            phase3.CASES, references, phase3.observe_case, repeats, [],
            {"corpus_id": phase3.EXPECTED["corpus_id"], "policy": {
                "policy_id": phase3.SCENARIOS["policy"]["policy_id"],
                "version": phase3.SCENARIOS["policy"]["version"]}, "catalog": {
                "catalog_id": phase3.SCENARIOS["catalog"]["catalog_id"],
                "version": phase3.SCENARIOS["catalog"]["version"]},
             "review_id": phase3.SCENARIOS["catalog"]["review_id"],
             "review_stage": "PROVISIONAL_ENGINEERING",
             "evaluation_time": phase3.SCENARIOS["evaluation_time"], "unavailable_time_cases": 1},
        )
    verify_frozen_inputs()
    return {
        "format": FORMAT, "phase3_baseline_sha": PHASE3_SHA,
        "methodology_sha256": METHODOLOGY_SHA256, "frozen_files_sha256": dict(FROZEN_SHA256),
        "selection": selection, "repeats": repeats,
        "independent_holdout": "NOT_AVAILABLE_NOT_MEASURED",
        "partitions": partitions,
    }


def exit_status(results):
    if any(part["counts"]["runner_errors"] for part in results["partitions"].values()):
        return 2
    if any(part["metrics"]["deterministic_repeat_mismatches"]["numerator"]
           for part in results["partitions"].values()):
        return 3
    if any(part["metrics"]["full_contract_agreement"]["numerator"] != part["counts"]["selected_labeled"]
           for part in results["partitions"].values()):
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", required=True, choices=("development", "phase3", "all"))
    parser.add_argument("--repeats", type=int, default=3, choices=range(2, 6))
    parser.add_argument("--output", type=Path, help="Create a new result file; never overwrite.")
    args = parser.parse_args(argv)
    try:
        results = evaluate(args.partition, args.repeats)
        status = exit_status(results)
        envelope = {
            "environment": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.system() + "-" + platform.machine(),
                "python_hash_seed": os.environ.get("PYTHONHASHSEED", "UNSET"),
            },
            "exit_status": status, "results_sha256": digest(results), "results": results,
        }
        output = json.dumps(envelope, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
        if args.output is None:
            sys.stdout.write(output)
        else:
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(output)
        return status
    except Exception:
        print("BENCHMARK_INTEGRITY_OR_EXECUTION_ERROR", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
