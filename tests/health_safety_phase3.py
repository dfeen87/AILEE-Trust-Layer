"""Fixed offline Phase 3 conformance fixtures and observation projection.

This is a research test harness, not an ingress, authority provider, content
consumer, or configurable security-testing tool. Only the versioned synthetic
fixtures beside the tests are evaluated. Trusted context is assembled from its
own fixture profile, never from the request presented to the governor.

Run `python -m tests.health_safety_phase3` from the repository root to reproduce
the machine-readable observations; `--output` creates a new evidence file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import logging
import re
import sys
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ailee.domains import health_safety as hs

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "health_safety"
SCENARIO_FILE = FIXTURES / "phase3_scenarios.json"
EXPECTED_FILE = FIXTURES / "phase3_expected.json"
OBSERVED_FILE = FIXTURES / "phase3_observed.json"
MAX_FIXTURE_BYTES = 256 * 1024
MAX_CASES = 128
CANARY = "SYNTHETIC_CANARY_P3_7319"
CANARIES = (CANARY, "artifact-p3-synthetic-canary-7319", "0" * 64)
HASH_SEEDS = ("0", "7", "31")
CATEGORY_COUNTS = {
    "unsupported-certainty": 2,
    "evidence-consistency": 11,
    "consent": 12,
    "authority-schema": 8,
    "public-support": 8,
    "privacy-failures": 4,
    "artifact-integrity": 10,
    "deterministic-replay": 3,
    "benign-controls": 4,
}


class FixtureError(ValueError):
    """Controlled failure of the fixed synthetic research harness."""


def _require(condition, code):
    if not condition:
        raise FixtureError(code)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "DUPLICATE_FIXTURE_KEY")
        result[key] = value
    return result


def read_fixture(path):
    """Read a bounded versioned local JSON fixture without duplicate keys."""
    with path.open("rb") as handle:
        raw = handle.read(MAX_FIXTURE_BYTES + 1)
    _require(len(raw) <= MAX_FIXTURE_BYTES, "FIXTURE_TOO_LARGE")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError):
        raise FixtureError("INVALID_FIXTURE_JSON") from None


SCENARIOS = read_fixture(SCENARIO_FILE)
EXPECTED = read_fixture(EXPECTED_FILE)
CASES = tuple(sorted(SCENARIOS["cases"], key=lambda item: item["id"]))


def fixture_time(value):
    _require(
        type(value) is str
        and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value)
        is not None,
        "INVALID_FIXTURE_TIME",
    )
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise FixtureError("INVALID_FIXTURE_TIME") from None


NOW = fixture_time(SCENARIOS["evaluation_time"])
START = fixture_time(SCENARIOS["valid_from"])
END = fixture_time(SCENARIOS["valid_until"])


def _catalog_ref():
    config = SCENARIOS["catalog"]
    return hs.CatalogRef(config["catalog_id"], config["version"])


def _policy_ref():
    config = SCENARIOS["policy"]
    return hs.PolicyRef(config["policy_id"], config["version"])


def _payload(profile):
    payload = SCENARIOS["artifacts"][profile]["payload"].encode("utf-8")
    _require(0 < len(payload) <= 4096, "INVALID_FIXTURE_PAYLOAD_LENGTH")
    return payload


def _artifact(profile):
    return hs.ArtifactRef("artifact-p3-" + profile, hashlib.sha256(_payload(profile)).hexdigest())


def _scope(profile):
    return hs.FixtureScope("fixture-p3-" + profile, "turn-3")


def _purpose(profile):
    return hs.HealthPurpose(SCENARIOS["artifacts"][profile]["purpose"])


def _policy_and_catalog(*, reordered=False):
    artifacts = []
    evidence = []
    for profile, spec in SCENARIOS["artifacts"].items():
        artifacts.append(hs.ReviewedArtifact(
            _artifact(profile), _scope(profile), _purpose(profile),
            hs.ArtifactAssessment(spec["assessment"]), tuple(spec["claims"]),
            (hs.SyntheticInput(spec["marker"]),),
        ))
        for row in spec["evidence"]:
            _require(len(row) in (4, 5), "INVALID_FIXTURE_EVIDENCE_ROW")
            evidence.append(hs.ReviewedEvidence(
                evidence_id=row[0], claim_id=row[1], source_group_id=row[2],
                relation=hs.EvidenceRelation(row[3]), artifact=_artifact(profile),
                scope=_scope(profile), purpose=_purpose(profile), valid_from=START,
                valid_until=fixture_time(row[4]) if len(row) == 5 else END,
            ))
    purposes = (hs.HealthPurpose.GENERAL_EDUCATION, hs.HealthPurpose.UNCERTAINTY_AND_REFERRAL)
    policy = hs.HealthPolicy(
        _policy_ref(), _catalog_ref(), purposes[::-1] if reordered else purposes,
        SCENARIOS["policy"]["min_independent_sources_per_claim"],
    )
    catalog = hs.HealthCatalog(
        SCENARIOS["schema_version"], _catalog_ref(), SCENARIOS["catalog"]["review_id"],
        hs.CatalogReviewStage.PROVISIONAL_ENGINEERING,
        tuple(reversed(artifacts) if reordered else artifacts),
        tuple(reversed(evidence) if reordered else evidence),
    )
    return policy, catalog


def _reorder(value):
    # Reverse insertion/list ordering only; no semantic mutation or randomness.
    if type(value) is dict:
        return {key: _reorder(value[key]) for key in reversed(value)}
    if type(value) is list:
        return [_reorder(item) for item in reversed(value)]
    return value


def build_request(case, *, reordered=False):
    """The untrusted half of a fixed, independently specified fixture."""
    profile = case["request"]
    spec = SCENARIOS["artifacts"][profile]
    raw = {
        "schema_version": SCENARIOS["schema_version"],
        "scope": asdict(_scope(profile)),
        "catalog": asdict(_catalog_ref()),
        "artifact": asdict(_artifact(profile)),
        "purpose": spec["purpose"],
        "claim_ids": list(spec["claims"]),
        "evidence_refs": [
            {"claim_id": row[1], "evidence_id": row[0]} for row in spec["evidence"]
        ],
        "synthetic_inputs": [spec["marker"]],
    }
    raw.update(copy.deepcopy(case.get("request_overrides", {})))
    if reordered:
        raw = _reorder(raw)
    form = case.get("request_form", "wire")
    if form == "typed":
        return hs.HealthRequest.from_dict(raw)
    if form == "null":
        return None
    _require(form == "wire", "INVALID_FIXTURE_REQUEST_FORM")
    return raw


def build_context(case):
    """The trusted half; no request object or request declaration is used."""
    profile = case["context"]
    if profile is None:
        return None
    scope = _scope(profile)
    consent = hs.SyntheticConsent(
        scope, _catalog_ref(), _artifact(profile), _purpose(profile),
        hs.ConsentState.GRANTED, START, END,
    )
    mode = case.get("consent", "granted")
    if mode == "missing":
        consent = None
    elif mode == "revoked":
        consent = replace(consent, state=hs.ConsentState.REVOKED)
    elif mode == "denied":
        consent = replace(consent, state=hs.ConsentState.DENIED)
    elif mode == "expired":
        consent = replace(consent, valid_until=NOW)
    elif mode == "future":
        consent = replace(consent, valid_from=NOW + timedelta(seconds=1))
    elif mode == "starts-now":
        consent = replace(consent, valid_from=NOW)
    elif mode == "other-scope":
        consent = replace(consent, scope=hs.FixtureScope(scope.fixture_id, "turn-other"))
    elif mode == "other-artifact":
        consent = replace(consent, artifact=_artifact("edu2" if profile == "edu" else "edu"))
    elif mode == "other-catalog":
        consent = replace(consent, catalog=replace(_catalog_ref(), version=2))
    elif mode == "other-purpose":
        consent = replace(consent, purpose=hs.HealthPurpose.UNCERTAINTY_AND_REFERRAL)
    else:
        _require(mode == "granted", "INVALID_FIXTURE_CONSENT_MODE")
    level = case.get("urgency", "ORDINARY")
    urgency = None if level == "NONE" else hs.UrgencyFact(
        _scope(case.get("urgency_scope", profile)), hs.UrgencyLevel(level),
    )
    return hs.HealthContext(
        scope, NOW, consent, urgency, hs.ContextState(case.get("context_state", "ASSEMBLED")),
    )


def build_governor(*, reordered=False):
    policy, catalog = _policy_and_catalog(reordered=reordered)
    return hs.HealthSafetyGovernor(policy=policy, catalog=catalog)


def _encode(value):
    if type(value) is datetime and value.tzinfo is timezone.utc:
        return value.isoformat(timespec="seconds").replace("+00:00", "Z")
    raise FixtureError("UNEXPECTED_OBSERVATION_TYPE")


def projection(record):
    """Normalize only evaluator-produced records, not raw request fields."""
    return json.loads(json.dumps(asdict(record), default=_encode, sort_keys=True))


@contextmanager
def _capture():
    # Local synchronous tests only: collect stdout/stderr and propagated Python
    # log records, and restore the root logger configuration even on failure.
    out, err, log = io.StringIO(), io.StringIO(), io.StringIO()
    root = logging.getLogger()
    old_level = root.level
    handler = logging.StreamHandler(log)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            yield out, err, log
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()


def _integrity_check(case, governor, request, context, decision):
    mode = case.get("integrity")
    if mode is None:
        return None
    profile = case["request"]
    artifact_id = _artifact(profile).artifact_id
    payload = _payload(profile)
    if mode == "other-id":
        artifact_id = _artifact("edu" if profile != "edu" else "edu2").artifact_id
    elif mode == "invalid-id":
        artifact_id = ""
    elif mode == "changed-bytes":
        payload += b"."
    elif mode == "nonbytes":
        payload = payload.decode("utf-8")
    elif mode == "empty-payload":
        payload = b""
    elif mode == "withheld-decision-type":
        # Ordinary wrong-type contract check. Use a newly constructed copy of a
        # WITHHELD result, not a fabricated affirmative authorization object.
        withheld = governor.evaluate(request, context=build_context(dict(case, consent="missing")))
        _require(withheld.outcome is hs.CandidateDisposition.DENY, "EXPECTED_WITHHELD_CONTROL")
        request = replace(withheld)
    else:
        _require(mode in ("exact", "fresh-consent"), "INVALID_FIXTURE_INTEGRITY_MODE")
    checked = governor.verify_synthetic_artifact(
        request, context=context, artifact_id=artifact_id, payload=payload,
    )
    _require(type(checked) is bool, "NON_BOOLEAN_INTEGRITY_RESULT")
    return checked


def observe_case(case):
    """Measure an authored case once, without consulting its expected label."""
    decoder_reason = None
    decoder_views = []
    prior_candidate = None
    replay_equal = None
    with _capture() as (out, err, log):
        governor = build_governor()
        request = build_request(case)
        context = build_context(case)
        if case.get("integrity") == "fresh-consent":
            prior = governor.evaluate(request, context=build_context(dict(case, consent="granted")))
            prior_candidate = prior.outcome.value
        if case.get("check_decoder"):
            try:
                hs.HealthRequest.from_dict(request)
            except hs.ContractError as exc:
                decoder_reason = exc.reason.value
                decoder_views = [str(exc), repr(exc), repr(exc.args)]

        fault = case.get("fault")
        _require(fault in (None, "controlled-evaluator-error"), "INVALID_FIXTURE_FAULT")
        injected = (
            patch.object(hs.HealthSafetyGovernor, "_candidate_reason", side_effect=RuntimeError(CANARY))
            if fault == "controlled-evaluator-error" else nullcontext()
        )
        with injected:
            decision = governor.evaluate(request, context=context)
        integrity = _integrity_check(case, governor, request, context, decision)
        if case.get("replay"):
            replay_equal = True
            for reordered in (False, True, True):
                repeated = build_governor(reordered=reordered).evaluate(
                    build_request(case, reordered=reordered), context=build_context(case),
                )
                replay_equal = replay_equal and repeated == decision and repeated.audit == decision.audit

        decision_data = projection(decision)
        audit_data = projection(decision.audit)
        views = [repr(decision), repr(decision.audit),
                 json.dumps(decision_data, sort_keys=True), json.dumps(audit_data, sort_keys=True)]
    captured = (out.getvalue(), err.getvalue(), log.getvalue())
    privacy_clear = None
    if case["category"] == "privacy-failures":
        privacy_clear = not any(
            canary in view for canary in CANARIES for view in (*views, *decoder_views, *captured)
        )
    return {
        "id": case["id"], "category": case["category"],
        "decision": decision_data, "audit": audit_data,
        "integrity_check": integrity, "privacy_clear": privacy_clear,
        "controlled_decoder_reason": decoder_reason, "prior_candidate": prior_candidate,
        "quiet": all(not value for value in captured), "replay_equal": replay_equal,
    }


def expected_decision(case):
    """Construct the reference projection from labels and trusted fixture data."""
    outcome, reason, support, permitted = EXPECTED["decisions"][case["id"]]
    context = build_context(case)
    return {
        "outcome": outcome, "reason_codes": [reason], "support": support,
        "support_template_id": EXPECTED["support_templates"][support],
        "permitted_artifact": None if permitted is None else EXPECTED["permitted_artifacts"][permitted],
        "policy": asdict(_policy_ref()), "catalog": asdict(_catalog_ref()),
        "scope": None if context is None else asdict(context.scope),
        "evaluation_time": None if context is None else SCENARIOS["evaluation_time"],
    }


def mismatches(case, observed):
    """Identify mismatched fields without printing fixture inputs or raw errors."""
    reference = expected_decision(case)
    audit = {key: value for key, value in reference.items() if key != "permitted_artifact"}
    checks = {
        "id": case["id"], "category": case["category"],
        "decision": reference, "audit": audit,
        "integrity_check": EXPECTED["integrity_checks"].get(case["id"]),
        "privacy_clear": EXPECTED["privacy_checks"].get(case["id"]),
        "controlled_decoder_reason": EXPECTED["controlled_decoder_reasons"].get(case["id"]),
        "prior_candidate": EXPECTED["prior_candidates"].get(case["id"]),
        "replay_equal": EXPECTED["replay_checks"].get(case["id"]), "quiet": True,
    }
    return tuple(key for key, value in checks.items() if observed.get(key) != value)


def validate_corpus():
    _require(SCENARIOS["schema_version"] == EXPECTED["schema_version"] == "1.0", "CORPUS_VERSION")
    _require(SCENARIOS["corpus"]["synthetic_only"] is True, "NON_SYNTHETIC_FIXTURE")
    _require(SCENARIOS["corpus"]["id"] == EXPECTED["corpus_id"], "CORPUS_ID_MISMATCH")
    _require(0 < len(CASES) <= MAX_CASES, "CASE_COUNT_BOUND")
    ids = {case["id"] for case in CASES}
    _require(len(ids) == len(CASES) and ids == set(EXPECTED["decisions"]), "CASE_LABEL_COVERAGE")
    counts = {}
    for case in CASES:
        category = case["category"]
        counts[category] = counts.get(category, 0) + 1
        _require(case["request"] in SCENARIOS["artifacts"], "UNKNOWN_REQUEST_PROFILE")
        _require(case["context"] is None or case["context"] in SCENARIOS["artifacts"], "UNKNOWN_CONTEXT_PROFILE")
        _require(bool(case["invariants"]) and all(type(i) is int and 1 <= i <= 12 for i in case["invariants"]),
                 "INVALID_INVARIANT_REFERENCE")
        outcome, reason, support, permitted = EXPECTED["decisions"][case["id"]]
        hs.CandidateDisposition(outcome)
        hs.HealthReason(reason)
        hs.SupportDisposition(support)
        _require((outcome == "ALLOW") == (permitted is not None), "INVALID_EXPECTED_PERMISSION")
        if permitted is not None:
            _require(permitted in EXPECTED["permitted_artifacts"], "UNKNOWN_REFERENCE_ARTIFACT")
    _require(counts == CATEGORY_COUNTS, "CATEGORY_COUNTS_MISMATCH")
    for profile, reference in EXPECTED["permitted_artifacts"].items():
        _require(asdict(_artifact(profile)) == reference, "EXPECTED_ARTIFACT_DIGEST_MISMATCH")
    for field, selector in (
        ("integrity_checks", lambda case: case.get("integrity") is not None),
        ("privacy_checks", lambda case: case["category"] == "privacy-failures"),
        ("controlled_decoder_reasons", lambda case: case.get("check_decoder")),
        ("prior_candidates", lambda case: case.get("integrity") == "fresh-consent"),
        ("replay_checks", lambda case: case.get("replay")),
    ):
        _require(set(EXPECTED[field]) == {case["id"] for case in CASES if selector(case)},
                 "AUXILIARY_LABEL_COVERAGE")
    _policy_and_catalog()


def observe_all():
    validate_corpus()
    return {
        "corpus_id": EXPECTED["corpus_id"], "case_count": len(CASES),
        "observations": [observe_case(case) for case in CASES],
    }


def observation_json(record):
    return json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new observation file; never overwrite.")
    args = parser.parse_args(argv)
    try:
        result = observe_all()
        failures = [
            (case["id"], mismatches(case, record))
            for case, record in zip(CASES, result["observations"])
            if mismatches(case, record)
        ]
        serialized = observation_json(result)
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
        if failures:
            for case_id, fields in failures:
                print(case_id + ": " + ", ".join(fields), file=sys.stderr)
            return 1
        return 0
    except Exception:
        # Raw input, log content and exception representations are never emitted
        # by this command, including failures in the experiment harness itself.
        print("PHASE3_OBSERVATION_ERROR", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
