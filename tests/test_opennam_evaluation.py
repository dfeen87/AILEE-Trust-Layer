"""Fixed authored engineering labels and deterministic observations."""

from hashlib import sha256
import json

import pytest

from tests.opennam_evaluation import (
    FIXTURES, FROZEN_CASES_SHA256, FROZEN_EXPECTED_SHA256, build_case,
    canonical_report, read_frozen, report,
)

FROZEN_OBSERVED_SHA256 = "a07b17b90153e36432f21a44e122cd7225679e68c904a9910fbe97f17b80f2e9"
CORPUS = read_frozen("cases.json", FROZEN_CASES_SHA256)
EXPECTED = read_frozen("expected.json", FROZEN_EXPECTED_SHA256)["labels"]


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda c: c["id"])
def test_fixed_provisional_outcome_and_reason(case):
    governor, request, clock = build_case(CORPUS, case)
    actual = governor.evaluate(request, evaluation_time=clock)
    assert [actual.outcome.value, [reason.value for reason in actual.reason_codes]] == EXPECTED[case["id"]]
    assert governor.evaluate(request, evaluation_time=clock) == actual
    if type(request) is dict:
        # Reordering supplied rows cannot change admission or audit.
        reordered = {**request, "records": request["records"][::-1]}
        assert governor.evaluate(reordered, evaluation_time=clock) == actual
        assert governor.evaluate(json.dumps(reordered, sort_keys=True).encode(), evaluation_time=clock) == actual
    audit = actual.audit
    assert not any("syn-" in str(value) for value in audit.values())


def test_frozen_complete_observations_and_comparisons():
    observed = (FIXTURES / "observed.json").read_bytes()
    assert sha256(observed).hexdigest() == FROZEN_OBSERVED_SHA256
    assert canonical_report().encode("ascii") == observed
    actual = report()
    assert actual["case_count"] == 40
    assert actual["exact_governor_outcome_and_reason_matches"] == 40
    assert actual["readiness_comparison"] == {
        "governor": {"true_ready": 6, "true_not_ready": 34, "false_ready": 0, "false_not_ready": 0},
        "accept_all": {"true_ready": 6, "true_not_ready": 0, "false_ready": 34, "false_not_ready": 0},
        "abstain_all": {"true_ready": 0, "true_not_ready": 34, "false_ready": 0, "false_not_ready": 6},
        "metadata_presence": {"true_ready": 6, "true_not_ready": 2, "false_ready": 32, "false_not_ready": 0},
    }
