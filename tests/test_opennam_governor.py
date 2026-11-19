"""Conformance of an offline metadata gate on independently configured fixtures."""

from dataclasses import replace
from datetime import timedelta, timezone
import json

import pytest

from ailee.domains.opennam import (
    CatalogEntry, ContractError, EvidenceCatalog, EvidenceGovernor, EvidenceRequest,
    Observation, Outcome, Reason, ReviewStatus,
)
from tests.test_opennam_contracts import REF, SCOPE, TIME, policy, record

NOW = TIME + timedelta(days=5)


def entry(r, group):
    return CatalogEntry(r.evidence_id, r.scope, r.metadata_sha256, group,
                        r.observed_at, *([ReviewStatus.PASS] * 4))


def fixture(records=None, entries=None, configured_policy=None):
    records = (record(), record(2)) if records is None else tuple(records)
    entries = tuple(entry(r, f"syn-group-{i}") for i, r in enumerate(records)) if entries is None else tuple(entries)
    configured_policy = policy() if configured_policy is None else configured_policy
    return (EvidenceGovernor(policy=configured_policy, catalog=EvidenceCatalog(REF, entries)),
            EvidenceRequest("1.0", REF, SCOPE, records))


@pytest.mark.parametrize("observation", [Observation.POSITIVE, Observation.NEGATIVE])
def test_ready_for_abstract_categorical_records(observation):
    records = tuple(replace(record(n), observation=observation) for n in (1, 2))
    governor, request = fixture(records)
    result = governor.evaluate(request, evaluation_time=NOW)
    assert result.outcome is Outcome.REVIEW_READY
    assert result.reason_codes == (Reason.METADATA_COMPLETE,)
    assert governor.evaluate(request.to_dict(), evaluation_time=NOW) == result
    assert governor.evaluate(json.dumps(request.to_dict()).encode(), evaluation_time=NOW) == result


@pytest.mark.parametrize("config", [None, {}, object()])
def test_invalid_config(config):
    _, request = fixture()
    decision = EvidenceGovernor(policy=config, catalog=config).evaluate(request, evaluation_time=NOW)
    assert decision.reason_codes == (Reason.INVALID_CONFIGURATION,)
    assert decision.outcome is Outcome.INVALID
    assert decision.audit["policy_sha256"] is None


@pytest.mark.parametrize("clock", [None, "2025-01-06T00:00:00Z", NOW.replace(tzinfo=None),
                                     NOW.replace(fold=1), NOW.replace(microsecond=1), NOW.replace(tzinfo=timezone(timedelta(hours=1)))])
def test_clock_fail_closed(clock):
    governor, request = fixture()
    decision = governor.evaluate(request, evaluation_time=clock)
    assert (decision.outcome, decision.reason_codes, decision.evaluation_time) == (Outcome.INVALID, (Reason.INVALID_TIME,), None)


@pytest.mark.parametrize("bad", [None, {}, object(), b'{"x":NaN}', b'{"x":1,"x":2}', {"secret": "raw content"}])
def test_malformed_request(bad):
    governor, _ = fixture()
    decision = governor.evaluate(bad, evaluation_time=NOW)
    assert decision.reason_codes == (Reason.MALFORMED_REQUEST,)
    assert decision.outcome is Outcome.INVALID
    assert "raw content" not in repr(decision)
    assert decision.bound_request_sha256 is None


def test_catalog_ref_and_scope_disagreements():
    governor, request = fixture()
    wrong_ref = replace(REF, version=2)
    assert governor.evaluate(replace(request, catalog=wrong_ref), evaluation_time=NOW).reason_codes == (Reason.CATALOG_MISMATCH,)
    other_scope = replace(SCOPE, protocol_version="syn-v2")
    assert governor.evaluate(replace(request, scope=other_scope), evaluation_time=NOW).reason_codes == (Reason.SCOPE_NOT_ALLOWED,)
    mixed = replace(request, records=(record(), replace(record(2), scope=other_scope)))
    assert governor.evaluate(mixed, evaluation_time=NOW).reason_codes == (Reason.SCOPE_MISMATCH,)
    mismatch = EvidenceGovernor(policy=policy(), catalog=EvidenceCatalog(wrong_ref, ()))
    assert mismatch.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.INVALID_CONFIGURATION,)


def test_exact_catalog_scope_coverage_and_cherry_picking():
    other_scope = replace(SCOPE, context_id="syn-context-other")
    records = (record(), record(2), replace(record(3), observation=Observation.POSITIVE))
    entries = tuple(entry(r, f"syn-group-{i}") for i, r in enumerate(records))
    another_study = replace(record(4), scope=other_scope, observation=Observation.POSITIVE)
    governor, request = fixture(records, entries + (entry(another_study, "syn-group-4"),))
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.CONFLICT,)
    cherry_picked = replace(request, records=request.records[:2])
    assert governor.evaluate(cherry_picked, evaluation_time=NOW).reason_codes == (Reason.INCOMPLETE_CATALOG_SCOPE,)
    assert governor.evaluate(replace(request, records=()), evaluation_time=NOW).reason_codes == (Reason.MISSING_EVIDENCE,)
    extra = replace(request, records=request.records + (record(5),))
    assert governor.evaluate(extra, evaluation_time=NOW).reason_codes == (Reason.UNREVIEWED_EVIDENCE,)
    scope_governor, valid = fixture(records[:2], entries[:2] + (entry(another_study, "syn-group-4"),))
    assert scope_governor.evaluate(valid, evaluation_time=NOW).outcome is Outcome.REVIEW_READY


@pytest.mark.parametrize("patch", [
    {"artifact_sha256": "f" * 64}, {"observation": Observation.POSITIVE},
    {"observed_at": TIME + timedelta(seconds=1)},
])
def test_digest_binding(patch):
    governor, request = fixture()
    altered = replace(request, records=(replace(record(), **patch), record(2)))
    result = governor.evaluate(altered, evaluation_time=NOW)
    assert result.reason_codes == (Reason.BINDING_MISMATCH,)
    assert result.bound_request_sha256 is None


@pytest.mark.parametrize("field,reason", [
    ("reporting", Reason.REPORTING_NOT_PASSED), ("positive_control", Reason.CONTROLS_NOT_PASSED),
    ("negative_control", Reason.CONTROLS_NOT_PASSED), ("applicability", Reason.APPLICABILITY_NOT_PASSED),
])
@pytest.mark.parametrize("status", [ReviewStatus.FAIL, ReviewStatus.UNKNOWN])
def test_catalog_flags(field, reason, status):
    governor, request = fixture(entries=(replace(entry(record(), "syn-group-a"), **{field: status}),
                                         entry(record(2), "syn-group-b")))
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (reason,)


def test_contradiction_inconclusive_and_multiple_blockers():
    governor, request = fixture((replace(record(), observation=Observation.POSITIVE), record(2), replace(record(3), observation=Observation.INCONCLUSIVE)))
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.CONFLICT, Reason.INCONCLUSIVE)
    different = replace(record(), observation=Observation.POSITIVE)
    governor, request = fixture((different, record(2)), (replace(entry(different, "syn-a"), positive_control=ReviewStatus.FAIL, reporting=ReviewStatus.UNKNOWN), entry(record(2), "syn-b")))
    # No contradiction reported from observations with failed prerequisite metadata.
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.CONTROLS_NOT_PASSED, Reason.REPORTING_NOT_PASSED)


def test_source_group_count_and_duplicate_artifact():
    governor, request = fixture(entries=(entry(record(), "syn-a"), entry(record(2), "syn-a")))
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.INSUFFICIENT_GROUPS,)
    governor, request = fixture((record(), replace(record(2), artifact_sha256=record().artifact_sha256)))
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.DUPLICATE_ARTIFACT,)
    governor, request = fixture((record(), record(2), record(3)), (entry(record(), "syn-a"), entry(record(2), "syn-a"), entry(record(3), "syn-b")))
    assert governor.evaluate(request, evaluation_time=NOW).outcome is Outcome.REVIEW_READY


def test_explicit_temporal_boundaries_and_review_chronology():
    governor, request = fixture()
    for clock in (TIME, TIME + timedelta(days=30)):
        assert governor.evaluate(request, evaluation_time=clock).outcome is Outcome.REVIEW_READY
    for clock in (TIME - timedelta(seconds=1), TIME + timedelta(days=30, seconds=1)):
        assert governor.evaluate(request, evaluation_time=clock).reason_codes == (Reason.STALE_OR_FUTURE,)
    for reviewed_at in (TIME - timedelta(seconds=1), NOW + timedelta(seconds=1)):
        governor, request = fixture(entries=(replace(entry(record(), "syn-a"), reviewed_at=reviewed_at), entry(record(2), "syn-b")))
        assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.STALE_OR_FUTURE,)


def test_typed_input_revalidated_deeply():
    governor, request = fixture()
    object.__setattr__(request.records[0], "observation", "negative")
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.MALFORMED_REQUEST,)
    governor, request = fixture()
    object.__setattr__(request.records[0], "observed_at", TIME.replace(microsecond=1))
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.MALFORMED_REQUEST,)
    governor, request = fixture()
    object.__setattr__(request.records[0], "scope", object.__new__(type(SCOPE)))
    assert governor.evaluate(request, evaluation_time=NOW).reason_codes == (Reason.MALFORMED_REQUEST,)
    broken = object.__new__(EvidenceRequest)
    assert governor.evaluate(broken, evaluation_time=NOW).reason_codes == (Reason.MALFORMED_REQUEST,)


def test_trusted_snapshot_is_copied_and_malformed_snapshot_denied():
    configuration = policy()
    first, second = entry(record(), "syn-a"), entry(record(2), "syn-b")
    catalog = EvidenceCatalog(REF, (first, second))
    governor = EvidenceGovernor(policy=configuration, catalog=catalog)
    _, request = fixture()
    baseline = governor.evaluate(request, evaluation_time=NOW)
    object.__setattr__(first, "negative_control", ReviewStatus.UNKNOWN)
    object.__setattr__(configuration, "max_age_days", True)
    assert governor.evaluate(request, evaluation_time=NOW) == baseline
    assert EvidenceGovernor(policy=configuration, catalog=catalog).evaluate(request, evaluation_time=NOW).reason_codes == (Reason.INVALID_CONFIGURATION,)
    object.__setattr__(first, "negative_control", "pass")
    assert EvidenceGovernor(policy=policy(), catalog=catalog).evaluate(request, evaluation_time=NOW).reason_codes == (Reason.INVALID_CONFIGURATION,)


def test_audit_canonical_fingerprints_and_minimizes_request_metadata():
    governor, request = fixture()
    before = governor.evaluate(request, evaluation_time=NOW)
    reverse, reordered = fixture(request.records[::-1], (entry(record(2), "syn-group-1"), entry(record(), "syn-group-0")))
    assert reverse.evaluate(reordered, evaluation_time=NOW) == before
    altered_catalog, _ = fixture(entries=(entry(record(), "syn-different"), entry(record(2), "syn-group-1")))
    assert altered_catalog.evaluate(request, evaluation_time=NOW).catalog_sha256 != before.catalog_sha256
    altered_policy, _ = fixture(configured_policy=replace(policy(), version=2))
    assert altered_policy.evaluate(request, evaluation_time=NOW).policy_sha256 != before.policy_sha256
    assert set(before.audit) == {"outcome", "reason_codes", "evaluation_time", "policy_sha256", "catalog_sha256", "bound_request_sha256"}
    assert "syn-" not in repr(before.audit) and record().artifact_sha256 not in repr(before.audit)
    detached = before.audit
    detached["reason_codes"].append("tampered")
    assert before.audit["reason_codes"] == ["metadata_complete"]


def test_bound_request_digest_distinguishes_canonical_scopes():
    other = replace(SCOPE, context_id="syn-other-context")
    records = (record(), record(2), replace(record(3), scope=other), replace(record(4), scope=other))
    catalog = EvidenceCatalog(REF, tuple(entry(r, f"syn-group-{i}") for i, r in enumerate(records)))
    governor = EvidenceGovernor(policy=replace(policy(), allowed_scopes=(SCOPE, other)), catalog=catalog)
    first = governor.evaluate(EvidenceRequest("1.0", REF, SCOPE, records[:2]), evaluation_time=NOW)
    second = governor.evaluate(EvidenceRequest("1.0", REF, other, records[2:]), evaluation_time=NOW)
    assert first.outcome is second.outcome is Outcome.REVIEW_READY
    assert first.bound_request_sha256 != second.bound_request_sha256
    assert first.catalog_sha256 == second.catalog_sha256
    assert first.policy_sha256 == second.policy_sha256
