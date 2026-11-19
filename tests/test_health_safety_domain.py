"""Phase 1 structural tests and explicitly pending Phase 2 specifications."""

import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone, tzinfo
from enum import Enum
from pathlib import Path

import pytest

from ailee.domains import health_safety as hs
from ailee.domains.health_safety import health_safety as contracts


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "health_safety"
CANARY = "SYNTHETIC_PRIVATE_CANARY_NOT_REAL_6042"


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_FIXTURE_KEY")
        result[key] = value
    return result


def _read_fixture(filename):
    return json.loads(
        (FIXTURES / filename).read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_keys,
    )


SCENARIOS = _read_fixture("scenarios.json")
EXPECTED = _read_fixture("expected.json")
DEVELOPMENT = SCENARIOS["development"]
RESERVED = SCENARIOS["reserved_for_adjudication"]


def _time(value):
    # Fixture-only parser, not a production trust ingress or medical parser.
    if type(value) is not str or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value
    ):
        raise hs.ContractError(hs.HealthReason.INVALID_TIMESTAMP)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise hs.ContractError(hs.HealthReason.INVALID_TIMESTAMP) from None


def _scope(raw):
    return hs.FixtureScope(**raw)


def _artifact(raw):
    return hs.ArtifactRef(**raw)


def _catalog_ref(raw):
    return hs.CatalogRef(**raw)


def _policy():
    row = SCENARIOS["policy"]
    return hs.HealthPolicy(
        ref=hs.PolicyRef(**row["ref"]),
        catalog=_catalog_ref(row["catalog"]),
        allowed_purposes=tuple(hs.HealthPurpose(p) for p in row["allowed_purposes"]),
        min_independent_sources_per_claim=row["min_independent_sources_per_claim"],
    )


def _catalog(fault="NONE"):
    raw = SCENARIOS["catalog"]
    artifacts = tuple(
        hs.ReviewedArtifact(
            artifact=_artifact(row["artifact"]),
            scope=_scope(row["scope"]),
            purpose=hs.HealthPurpose(row["purpose"]),
            assessment=hs.ArtifactAssessment(row["assessment"]),
            claim_ids=tuple(row["claim_ids"]),
            synthetic_inputs=tuple(hs.SyntheticInput(v) for v in row["synthetic_inputs"]),
        )
        for row in raw["artifacts"]
    )
    evidence = tuple(
        hs.ReviewedEvidence(
            evidence_id=row["evidence_id"],
            artifact=_artifact(row["artifact"]),
            scope=_scope(row["scope"]),
            purpose=hs.HealthPurpose(row["purpose"]),
            claim_id=row["claim_id"],
            source_group_id=row["source_group_id"],
            relation=hs.EvidenceRelation(row["relation"]),
            valid_from=_time(row["valid_from"]),
            valid_until=_time(row["valid_until"]),
        )
        for row in raw["evidence"]
    )
    if fault == "DUPLICATE_CATALOG_ARTIFACT":
        artifacts += (artifacts[0],)
    return hs.HealthCatalog(
        schema_version=raw["schema_version"],
        ref=_catalog_ref(raw["ref"]),
        review_id=raw["review_id"],
        review_stage=hs.CatalogReviewStage(raw["review_stage"]),
        artifacts=artifacts,
        evidence=evidence,
    )


def _context(raw):
    # The trusted fixture path is deliberately separate: this function never
    # takes a HealthRequest, and cannot derive authority from its fields.
    row = raw["consent"]
    consent = None if row is None else hs.SyntheticConsent(
        scope=_scope(row["scope"]),
        catalog=_catalog_ref(row["catalog"]),
        artifact=_artifact(row["artifact"]),
        purpose=hs.HealthPurpose(row["purpose"]),
        state=hs.ConsentState(row["state"]),
        valid_from=_time(row["valid_from"]),
        valid_until=_time(row["valid_until"]),
    )
    urgency = raw["urgency"]
    return hs.HealthContext(
        scope=None if raw["scope"] is None else _scope(raw["scope"]),
        evaluation_time=None if raw["evaluation_time"] is None else _time(raw["evaluation_time"]),
        consent=consent,
        urgency=None if urgency is None else hs.UrgencyFact(
            scope=_scope(urgency["scope"]),
            level=hs.UrgencyLevel(urgency["level"]),
        ),
        state=hs.ContextState(raw["state"]),
    )


def _inputs(case):
    # Shallow, explicit top-level replacements after copying each template
    # independently. An override can intentionally make the request malformed.
    request = copy.deepcopy(SCENARIOS["request_templates"][case["request_template"]])
    request.update(copy.deepcopy(case["request_overrides"]))
    context = copy.deepcopy(SCENARIOS["context_templates"][case["context_template"]])
    context.update(copy.deepcopy(case["context_overrides"]))
    return request, context


def _education():
    return (
        hs.HealthRequest.from_dict(copy.deepcopy(SCENARIOS["request_templates"]["education"])),
        _context(copy.deepcopy(SCENARIOS["context_templates"]["education"])),
    )


def _check_schema(build, expected):
    if expected == "VALID":
        return build()
    with pytest.raises(hs.ContractError) as caught:
        build()
    assert caught.value.reason is hs.HealthReason(expected)
    assert caught.value.args == (expected,)
    assert CANARY not in repr(caught.value)
    return None


def _label_record(label, *, scope, evaluation_time, catalog):
    """Validate an authored expectation's result shape, never evaluate it."""
    proof = label["permitted_artifact"]
    return hs.HealthDecision(
        outcome=hs.CandidateDisposition(label["candidate"]),
        reason_codes=tuple(hs.HealthReason(code) for code in label["reason_codes"]),
        permitted_artifact=None if proof is None else _artifact(proof),
        support=hs.SupportDisposition(label["support"]),
        support_template_id=hs.SupportTemplateId(label["support_template_id"]),
        policy=_policy().ref,
        catalog=None if catalog is None else catalog.ref,
        scope=scope,
        evaluation_time=evaluation_time,
    )


def test_fixture_partitions_and_provisional_label_status():
    assert SCENARIOS["schema_version"] == EXPECTED["schema_version"] == hs.SCHEMA_VERSION
    corpus = SCENARIOS["corpus"]
    assert corpus["id"] == EXPECTED["corpus_id"]
    assert corpus["synthetic_only"] is True
    assert corpus["label_status"] == "PROVISIONAL_ENGINEERING"
    assert corpus["independent_adjudication"] == "PENDING"
    assert corpus["corpus_licensing"] == "PENDING_REVIEW"
    assert corpus["reserved_partition"] == "VISIBLE_UNADJUDICATED_NOT_A_BLIND_HOLDOUT"
    assert EXPECTED["label_status"] == "PROVISIONAL_ENGINEERING_NOT_CLINICALLY_OR_INDEPENDENTLY_REVIEWED"
    assert EXPECTED["governor_behavior"] == "PENDING_PHASE_2"
    dev_ids = [case["id"] for case in DEVELOPMENT]
    reserved_ids = [case["id"] for case in RESERVED]
    assert len(set(dev_ids + reserved_ids)) == len(dev_ids) + len(reserved_ids)
    assert set(dev_ids) == set(EXPECTED["development"])
    assert set(reserved_ids) == set(EXPECTED["reserved_for_adjudication"])
    assert set(EXPECTED["reserved_for_adjudication"].values()) == {
        "AWAITING_INDEPENDENT_ADJUDICATION"
    }
    assert dev_ids and reserved_ids
    for case in DEVELOPMENT + RESERVED:
        assert set(case) == {
            "id", "description", "request_template", "request_overrides",
            "context_template", "context_overrides", "harness_fault",
        }
        assert case["description"]
        assert case["harness_fault"] in {
            "NONE", "DUPLICATE_CATALOG_ARTIFACT", "EVALUATOR_EXCEPTION",
        }
        assert type(case["request_overrides"]) is dict
        assert type(case["context_overrides"]) is dict
        _inputs(case)  # Only materialize fixture syntax; no reserved verdict.


def test_catalog_payload_digests_and_independent_fixture_templates():
    catalog, policy = _catalog(), _policy()
    assert catalog.ref == policy.catalog
    assert catalog.review_stage is hs.CatalogReviewStage.PROVISIONAL_ENGINEERING
    payloads = SCENARIOS["artifact_payloads"]
    manifests = {row.artifact.artifact_id: row for row in catalog.artifacts}
    assert set(payloads) == set(manifests)
    for artifact_id, payload in payloads.items():
        assert type(payload) is str and 0 < len(payload.encode("utf-8")) <= 4096
        assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == manifests[artifact_id].artifact.sha256

    assert set(SCENARIOS["request_templates"]) == set(SCENARIOS["context_templates"])
    for key, raw in SCENARIOS["request_templates"].items():
        request = hs.HealthRequest.from_dict(raw)
        context = _context(SCENARIOS["context_templates"][key])
        manifest = manifests[request.artifact.artifact_id]
        assert (request.artifact, request.scope, request.purpose) == (
            manifest.artifact, manifest.scope, manifest.purpose
        )
        assert request.claim_ids == manifest.claim_ids
        assert request.synthetic_inputs == manifest.synthetic_inputs
        assert request.catalog == catalog.ref
        assert context.scope == request.scope
        assert context.consent is not None
        assert (context.consent.scope, context.consent.catalog, context.consent.artifact, context.consent.purpose) == (
            request.scope, request.catalog, request.artifact, request.purpose
        )
        assert context.urgency is None or context.urgency.scope == context.scope


@pytest.mark.parametrize("case", DEVELOPMENT, ids=lambda case: case["id"])
def test_development_scenario_schema_and_expected_record_shape(case):
    """Executable schema checks; the separately specified verdict is NOT executed."""
    raw_request, raw_context = _inputs(case)
    label = EXPECTED["development"][case["id"]]
    schema = label["schema_checks"]
    request = _check_schema(lambda: hs.HealthRequest.from_dict(raw_request), schema["request"])
    context = _check_schema(lambda: _context(raw_context), schema["context"])
    catalog = _check_schema(lambda: _catalog(case["harness_fault"]), schema["catalog"])
    record = _label_record(
        label,
        scope=None if raw_context["scope"] is None else _scope(raw_context["scope"]),
        evaluation_time=None if context is None else context.evaluation_time,
        catalog=catalog,
    )
    assert record.audit.reason_codes == record.reason_codes
    assert record.audit.support is record.support
    if record.outcome is hs.CandidateDisposition.ALLOW:
        # Check only the consistency of the written label and exact fixture ID.
        assert request is not None and catalog is not None and context is not None
        assert record.permitted_artifact == request.artifact
        assert request.artifact in tuple(row.artifact for row in catalog.artifacts)
    else:
        assert record.permitted_artifact is None


def test_canonicalization_defensive_copy_and_frozen_contracts():
    request, context = _education()
    policy, catalog = _policy(), _catalog()
    raw = copy.deepcopy(SCENARIOS["request_templates"]["education"])
    raw["claim_ids"].reverse()
    raw["evidence_refs"].reverse()
    parsed = hs.HealthRequest.from_dict(raw)
    assert parsed == request and hash(parsed) == hash(request)
    raw["artifact"]["sha256"] = "f" * 64
    raw["claim_ids"].clear()
    assert parsed == request
    assert replace(policy, allowed_purposes=tuple(reversed(policy.allowed_purposes))) == policy
    assert replace(catalog, artifacts=tuple(reversed(catalog.artifacts)),
                   evidence=tuple(reversed(catalog.evidence))) == catalog

    record = _label_record(
        EXPECTED["development"]["dev-education-eligible"],
        scope=context.scope, evaluation_time=context.evaluation_time, catalog=catalog,
    )
    # Deeply immutable ordinary Python values, not just a frozen outer shell.
    def assert_frozen(value):
        if is_dataclass(value):
            assert type(value).__dataclass_params__.frozen
            with pytest.raises(FrozenInstanceError):
                setattr(value, fields(value)[0].name, None)
            for item in fields(value):
                assert_frozen(getattr(value, item.name))
        elif type(value) is tuple:
            for item in value:
                assert_frozen(item)
        else:
            assert value is None or isinstance(value, (str, int, datetime, Enum))

    for value in (request, context, policy, catalog, record, record.audit):
        assert_frozen(value)


@pytest.mark.parametrize("changes", [
    {"claim_ids": ["claim-duplicate", "claim-duplicate"]},
    {"claim_ids": ["claim-" + "a" * 60]},
    {"claim_ids": ["foreign-id"]},
    {"claim_ids": ["claim-" + str(i) for i in range(contracts.MAX_CLAIMS + 1)]},
    {"evidence_refs": [{"claim_id": "claim-a", "evidence_id": "evidence-a"}] * (contracts.MAX_EVIDENCE_REFS + 1)},
    {"evidence_refs": [{"claim_id": "claim-a", "evidence_id": "evidence-a", "authority": True}]},
    {"synthetic_inputs": ["GENERAL_QUESTION"] * (contracts.MAX_INPUT_MARKERS + 1)},
    {"synthetic_inputs": ["GENERAL_QUESTION", "GENERAL_QUESTION"]},
    {"synthetic_inputs": ("GENERAL_QUESTION",)},
    {"catalog": {"catalog_id": "catalog-ahs-synthetic", "version": True}},
    {"artifact": {"artifact_id": "artifact-a", "sha256": "F" * 64}},
    {"scope": {"fixture_id": "fixture- a", "turn_id": "turn-1"}},
    {"evaluation_time": "2026-01-15T12:00:00Z"},
    {"urgency": "IMMEDIATE_PUBLIC_REFERRAL"},
    {"policy": {"min_independent_sources_per_claim": 0}},
])
def test_request_closed_schema_and_bounds(changes):
    raw = copy.deepcopy(SCENARIOS["request_templates"]["education"])
    raw.update(changes)
    with pytest.raises(hs.ContractError, match="^MALFORMED_SCHEMA$"):
        hs.HealthRequest.from_dict(raw)


def test_malformed_typed_fields_do_not_coerce_strings_or_lists():
    request, context = _education()
    for changes in (
        {"purpose": "GENERAL_EDUCATION"},
        {"claim_ids": list(request.claim_ids)},
        {"synthetic_inputs": ("GENERAL_QUESTION",)},
        {"evidence_refs": list(request.evidence_refs)},
    ):
        with pytest.raises(hs.ContractError):
            replace(request, **changes)
    with pytest.raises(hs.ContractError):
        replace(context.consent, state="GRANTED")
    with pytest.raises(hs.ContractError):
        replace(context.urgency, level="IMMEDIATE_PUBLIC_REFERRAL")
    with pytest.raises(hs.ContractError, match="^INVALID_CONTEXT$"):
        replace(context, consent={"state": "GRANTED"})
    with pytest.raises(hs.ContractError, match="^INVALID_CONTEXT$"):
        replace(context, state="ASSEMBLED")
    with pytest.raises(hs.ContractError, match="^INVALID_CONTEXT$"):
        replace(context, scope=None)
    failed = replace(context, consent=None, state=hs.ContextState.INVALID)
    assert failed.urgency == context.urgency and failed.consent is None


@pytest.mark.parametrize("changes", [
    {"min_independent_sources_per_claim": 1},
    {"min_independent_sources_per_claim": contracts.MAX_INDEPENDENT_SOURCES + 1},
    {"min_independent_sources_per_claim": True},
    {"allowed_purposes": (hs.HealthPurpose.DIAGNOSIS,)},
    {"allowed_purposes": (hs.HealthPurpose.TREATMENT_SELECTION,)},
    {"allowed_purposes": (hs.HealthPurpose.PRESCRIBING,)},
    {"allowed_purposes": (hs.HealthPurpose.CLINICAL_TRIAGE,)},
    {"allowed_purposes": (hs.HealthPurpose.EXTERNAL_ACTION,)},
    {"allowed_purposes": ("GENERAL_EDUCATION",)},
])
def test_policy_cannot_lower_nonnegotiable_limits(changes):
    with pytest.raises(hs.ContractError, match="^INVALID_POLICY$"):
        replace(_policy(), **changes)


class _UnexpectedTimezone(tzinfo):
    def utcoffset(self, dt):
        raise AssertionError("Contract validation must not call a timezone provider")


@pytest.mark.parametrize("value", [
    "not-a-timestamp",
    True,
    datetime(2026, 1, 15, 12),
    datetime(2026, 1, 15, 12, tzinfo=timezone(timedelta(hours=1))),
    datetime(2026, 1, 15, 12, microsecond=1, tzinfo=timezone.utc),
    datetime(2026, 1, 15, 12, tzinfo=timezone.utc, fold=1),
    datetime(2026, 1, 15, 12, tzinfo=_UnexpectedTimezone()),
], ids=["string", "bool", "naive", "offset", "fraction", "fold", "provider"])
def test_timestamp_validation_is_explicit_utc_and_safe(value):
    _, context = _education()
    with pytest.raises(hs.ContractError, match="^INVALID_TIMESTAMP$"):
        replace(context, evaluation_time=value)
    assert replace(context, evaluation_time=None).evaluation_time is None


def test_validity_window_shape_and_unknown_catalog_enums():
    _, context = _education()
    evidence = _catalog().evidence[0]
    for item in (context.consent, evidence):
        with pytest.raises(hs.ContractError, match="^INVALID_TIMESTAMP$"):
            replace(item, valid_until=item.valid_from)
        with pytest.raises(hs.ContractError, match="^INVALID_TIMESTAMP$"):
            replace(item, valid_until=item.valid_from - timedelta(seconds=1))
    with pytest.raises(hs.ContractError, match="^INVALID_CATALOG$"):
        replace(evidence, relation="SUPPORTS")
    with pytest.raises(hs.ContractError, match="^INVALID_CATALOG$"):
        replace(_catalog(), review_stage="PROVISIONAL_ENGINEERING")


@pytest.mark.parametrize("fault", [
    "duplicate-artifact", "duplicate-evidence", "dangling-claim", "wrong-digest",
    "wrong-scope", "wrong-purpose", "prohibited-eligible", "private-eligible",
    "authority-eligible", "claimless-eligible", "unbounded-evidence",
])
def test_catalog_structural_corruption_rejected(fault):
    catalog = _catalog()
    entry = catalog.artifacts[0]
    evidence = catalog.evidence[0]
    with pytest.raises(hs.ContractError, match="^INVALID_CATALOG$"):
        if fault == "duplicate-artifact":
            replace(catalog, artifacts=catalog.artifacts + (entry,))
        elif fault == "duplicate-evidence":
            replace(catalog, evidence=catalog.evidence + (evidence,))
        elif fault == "dangling-claim":
            replace(catalog, evidence=(replace(evidence, claim_id="claim-foreign"),))
        elif fault == "wrong-digest":
            changed = replace(evidence.artifact, sha256="f" * 64)
            replace(catalog, evidence=(replace(evidence, artifact=changed),))
        elif fault == "wrong-scope":
            replace(catalog, evidence=(replace(evidence, scope=hs.FixtureScope("fixture-foreign", "turn-2")),))
        elif fault == "wrong-purpose":
            replace(catalog, evidence=(replace(evidence, purpose=hs.HealthPurpose.DIAGNOSIS),))
        elif fault == "prohibited-eligible":
            replace(entry, purpose=hs.HealthPurpose.DIAGNOSIS, assessment=hs.ArtifactAssessment.ELIGIBLE,
                    synthetic_inputs=(hs.SyntheticInput.GENERAL_QUESTION,))
        elif fault == "private-eligible":
            replace(entry, purpose=hs.HealthPurpose.GENERAL_EDUCATION, assessment=hs.ArtifactAssessment.ELIGIBLE,
                    synthetic_inputs=(hs.SyntheticInput.PRIVATE_DATA_CANARY,))
        elif fault == "authority-eligible":
            replace(entry, purpose=hs.HealthPurpose.GENERAL_EDUCATION, assessment=hs.ArtifactAssessment.ELIGIBLE,
                    synthetic_inputs=(hs.SyntheticInput.ASSERTED_AUTHORITY,))
        elif fault == "claimless-eligible":
            replace(entry, purpose=hs.HealthPurpose.GENERAL_EDUCATION, assessment=hs.ArtifactAssessment.ELIGIBLE,
                    synthetic_inputs=(hs.SyntheticInput.GENERAL_QUESTION,), claim_ids=())
        elif fault == "unbounded-evidence":
            replace(catalog, evidence=tuple(
                replace(evidence, evidence_id=f"evidence-overflow-{i}")
                for i in range(contracts.MAX_EVIDENCE_REFS + 1)
            ))


def test_decision_shape_preserves_separate_support_and_exact_reference():
    request, context = _education()
    catalog = _catalog()
    allow = _label_record(
        EXPECTED["development"]["dev-education-eligible"],
        scope=context.scope, evaluation_time=context.evaluation_time, catalog=catalog,
    )
    review = replace(
        allow, outcome=hs.CandidateDisposition.REVIEW,
        reason_codes=(hs.HealthReason.MISSING_EVIDENCE,), permitted_artifact=None,
    )
    urgent_denial = replace(
        review, outcome=hs.CandidateDisposition.DENY,
        reason_codes=(hs.HealthReason.PROHIBITED_MEDICAL_ACTION,),
        support=hs.SupportDisposition.IMMEDIATE_PUBLIC_REFERRAL,
        support_template_id=hs.SupportTemplateId.IMMEDIATE_PUBLIC_REFERRAL_V1,
        evaluation_time=None,
    )
    assert allow.permitted_artifact == request.artifact
    assert review.permitted_artifact is urgent_denial.permitted_artifact is None
    assert urgent_denial.audit.support is hs.SupportDisposition.IMMEDIATE_PUBLIC_REFERRAL
    assert urgent_denial.audit.evaluation_time is None

    for original, changes in (
        (allow, {"permitted_artifact": None}),
        (allow, {"reason_codes": (hs.HealthReason.MISSING_EVIDENCE,)}),
        (allow, {"policy": None}),
        (allow, {"evaluation_time": None}),
        (review, {"permitted_artifact": allow.permitted_artifact}),
        (review, {"reason_codes": (hs.HealthReason.SYNTHETIC_ARTIFACT_ELIGIBLE,)}),
        (urgent_denial, {"permitted_artifact": allow.permitted_artifact}),
        (urgent_denial, {"reason_codes": ()}),
        (urgent_denial, {"reason_codes": (hs.HealthReason.EVALUATOR_FAILURE, hs.HealthReason.MALFORMED_SCHEMA)}),
        (urgent_denial, {"scope": None}),
        (urgent_denial, {"support_template_id": hs.SupportTemplateId.ORDINARY_HELP_V1}),
        (urgent_denial, {"outcome": "DENY"}),
    ):
        with pytest.raises(hs.ContractError):
            replace(original, **changes)


def test_privacy_canaries_are_not_echoed_by_schema_or_audit_shape(capsys, caplog):
    request, context = _education()
    raw = copy.deepcopy(SCENARIOS["request_templates"]["education"])
    for changes in (
        {"purpose": CANARY},
        {"synthetic_inputs": [CANARY]},
        {CANARY: CANARY},
        {"artifact": {"artifact_id": CANARY, "sha256": CANARY}},
    ):
        malformed = dict(raw, **changes)
        with pytest.raises(hs.ContractError) as caught:
            hs.HealthRequest.from_dict(malformed)
        error = caught.value
        assert error.args == ("MALFORMED_SCHEMA",)
        assert CANARY not in str(error) and CANARY not in repr(error)
        assert error.__context__ is None and error.__cause__ is None

    # Even a lexical identifier is not evidence of trust or suitable for logs.
    canary_id = "artifact-synthetic-private-canary-not-real-6042"
    valid_shape = hs.HealthRequest.from_dict(
        dict(raw, artifact={"artifact_id": canary_id, "sha256": "0" * 64})
    )
    assert canary_id not in repr(valid_shape) and canary_id not in repr(valid_shape.artifact)
    assert repr(request) == repr(valid_shape)
    denial = hs.HealthDecision(
        outcome=hs.CandidateDisposition.DENY,
        reason_codes=(hs.HealthReason.MALFORMED_SCHEMA,),
        permitted_artifact=None,
        support=hs.SupportDisposition.ORDINARY_HELP,
        support_template_id=hs.SupportTemplateId.ORDINARY_HELP_V1,
        policy=_policy().ref, catalog=_catalog().ref, scope=context.scope,
        evaluation_time=None,
    )
    assert set(asdict(denial.audit)) == {
        "outcome", "reason_codes", "support", "support_template_id",
        "policy", "catalog", "scope", "evaluation_time",
    }
    encoded = json.dumps(asdict(denial.audit), default=str)
    assert CANARY not in encoded and canary_id not in encoded
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_malformed_values_do_not_call_custom_representations():
    class Unexpected:
        def __repr__(self):
            raise AssertionError("Must not represent malformed objects")

        def __eq__(self, other):
            raise AssertionError("Must not compare malformed objects")

    raw = copy.deepcopy(SCENARIOS["request_templates"]["education"])
    for malformed in (Unexpected(), dict(raw, purpose=Unexpected())):
        with pytest.raises(hs.ContractError, match="^MALFORMED_SCHEMA$"):
            hs.HealthRequest.from_dict(malformed)


def test_phase1_public_surface_and_python39_syntax():
    assert hs.SCHEMA_VERSION == "1.0"
    assert not hasattr(hs, "HealthSafetyGovernor")
    assert not hasattr(contracts, "HealthSafetyGovernor")
    assert not hasattr(hs, "evaluate")
    for name in hs.__all__:
        assert hasattr(hs, name)
    for path in (Path(contracts.__file__), Path(hs.__file__), Path(__file__)):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
    syntax = ast.parse(Path(contracts.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module)
    assert imported <= {"__future__", "re", "dataclasses", "datetime", "enum", "typing"}


def test_health_import_has_no_incremental_side_effects():
    # Preload parent packages and the standard library. Only measure the new
    # opt-in domain; older ailee package initialization is unchanged.
    script = """
import __future__, contextlib, dataclasses, datetime, enum, io, logging
import os, re, sys, threading, typing
import ailee
import ailee.domains
assert "ailee.domains.health_safety" not in sys.modules
before = set(sys.modules)
env = dict(os.environ)
path = tuple(sys.path)
threads = tuple(threading.enumerate())
handlers = tuple(logging.getLogger().handlers)
exports = tuple(ailee.__all__)
events = []
def observe(event, args):
    if event == "open" and isinstance(args[2], int) and args[2] & (
        os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    ):
        events.append("write")
    elif event.startswith("socket.") or event in {
        "subprocess.Popen", "os.putenv", "os.unsetenv", "os.chdir",
        "os.mkdir", "os.remove", "os.rename", "os.rmdir"
    }:
        events.append(event)
sys.addaudithook(observe)
out, err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
    import ailee.domains.health_safety as hs
assert not out.getvalue() and not err.getvalue()
assert set(sys.modules) - before <= {
    "ailee.domains.health_safety", "ailee.domains.health_safety.health_safety"
}
assert not events
assert dict(os.environ) == env and tuple(sys.path) == path
assert tuple(threading.enumerate()) == threads
assert tuple(logging.getLogger().handlers) == handlers
assert tuple(ailee.__all__) == exports
assert not hasattr(ailee, "HealthSafetyGovernor")
assert not hasattr(hs, "HealthSafetyGovernor")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


# These tests are intentionally SKIPPED, not passing and not expected failures.
# Phase 2 must replace the placeholder bodies, remove the skip markers, and
# assert the authored expectations against the actual public evaluate() path.
@pytest.mark.skip(reason="Pending Phase 2: no HealthSafetyGovernor exists in Phase 1")
@pytest.mark.parametrize("case", DEVELOPMENT, ids=lambda case: case["id"])
def test_phase2_governor_behavior_pending(case):
    pytest.fail("Phase 2 must evaluate this case against expected.json")


PENDING_INVARIANTS = (
    "evidence-presence-not-medical-correctness",
    "complete-claim-evidence-artifact-purpose-catalog-binding",
    "deduplicate-underlying-source-groups",
    "unsupported-certainty-and-contradictions-withheld",
    "consent-and-authority-only-from-trusted-harness",
    "malformed-input-and-exceptions-fail-closed",
    "denial-preserves-independent-public-referral",
    "release-only-exact-reviewed-artifact-from-trusted-decision",
    "fixture-turn-and-consent-isolation",
    "audit-and-error-canaries-never-disclosed",
    "equal-canonical-inputs-identical-decisions",
    "existing-domain-state-unchanged",
)


@pytest.mark.skip(reason="Pending Phase 2: executable operational invariant")
@pytest.mark.parametrize("invariant", PENDING_INVARIANTS)
def test_phase2_operational_invariant_pending(invariant):
    pytest.fail("Phase 2 must exercise this invariant through the actual governor")
