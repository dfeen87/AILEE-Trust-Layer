"""Reconstruct original fictional study metadata and score engineering expectations.

The fixture manifest is trusted test-author input, not runtime request data.
No artifact payload, assay data, scientific inference or third-party data is used.
"""

from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from ailee.domains.opennam import (
    CatalogEntry, CatalogRef, EvidenceCatalog, EvidenceGovernor, EvidenceRecord,
    Observation, ReviewPolicy, ReviewStatus, Scope,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "opennam"
FROZEN_CASES_SHA256 = "244b3102f50e8714d6f07640c68d8a98d1875e8a600dbf245891910a3201f0d2"
FROZEN_EXPECTED_SHA256 = "8579732148981d64405f9ec6f147133b86ee3f177ba82fd296b86d9885691330"
SCOPE = Scope("syn-material", "syn-endpoint", "syn-method", "syn-v1", "syn-unit", "syn-context")
REF = CatalogRef("syn-research-catalog", 1)
OBSERVED_AT = "2025-01-01T00:00:00Z"


def read_frozen(name, fingerprint):
    raw = (FIXTURES / name).read_bytes()
    if sha256(raw).hexdigest() != fingerprint:
        raise ValueError(f"frozen manifest hash mismatch: {name}")
    return json.loads(raw)


def _time(value):
    return None if value is None else datetime.fromisoformat(value[:-1] + "+00:00")


def _fixture_record(record_id, specification):
    scope = replace(SCOPE, context_id=specification.get("scope_context", SCOPE.context_id))
    return EvidenceRecord(record_id, scope, _time(specification.get("observed_at", OBSERVED_AT)),
                          Observation(specification.get("observation", "negative")),
                          specification["artifact"] * 64)


def build_case(corpus, case):
    """Separate trusted catalog selection from untrusted request reconstruction."""
    definitions = corpus["records"]
    entries = []
    for record_id in case["catalog"]:
        # Origin is the fixed, independently selected fixture catalog roster.
        # Do not construct authority from request rows, including mutations below.
        spec = definitions[record_id]
        record = _fixture_record(record_id, spec)
        group = case.get("catalog_group_override", {}).get(record_id, spec["group"])
        entries.append(CatalogEntry(
            record_id, record.scope, record.metadata_sha256, group,
            _time(spec.get("reviewed_at", spec.get("observed_at", OBSERVED_AT))),
            ReviewStatus(spec.get("reporting", "pass")),
            ReviewStatus(spec.get("positive_control", "pass")),
            ReviewStatus(spec.get("negative_control", "pass")),
            ReviewStatus(spec.get("applicability", "pass")),
        ))
    catalog = None if case.get("invalid_config") else EvidenceCatalog(REF, tuple(entries))
    policy = ReviewPolicy("syn-research-policy", 1, REF, (SCOPE,), case.get("min_groups", 2), 30)
    governor = EvidenceGovernor(policy=policy, catalog=catalog)

    # Separately generate request rows from the original fictional data, not from
    # catalog objects. Request mutations do not update the independent manifest.
    request = {"schema_version": "1.0", "catalog": REF.to_dict(), "scope": SCOPE.to_dict(),
               "records": [_fixture_record(r, definitions[r]).to_dict() for r in case["request"]]}
    mutation = case.get("request_mutation")
    if mutation == "artifact":
        request["records"][0]["artifact_sha256"] = "f" * 64
    elif mutation == "observation":
        request["records"][0]["observation"] = "positive"
    elif mutation == "catalog_version":
        request["catalog"]["version"] = 2
    elif mutation == "scope":
        request["scope"]["context_id"] = "syn-not-allowed"
    elif mutation == "schema":
        request["schema_version"] = "2.0"
    elif mutation == "extra_review":
        request["records"][0]["negative_control"] = "pass"
    elif mutation == "bool_version":
        request["catalog"]["version"] = True
    elif mutation == "duplicate_key_json":
        request = b'{"schema_version":"1.0",' + json.dumps(request).encode()[1:]
    elif mutation == "nan_json":
        request["schema_version"] = float("nan")
        request = json.dumps(request).encode()
    elif mutation == "valid_json":
        request = json.dumps(request).encode()
    elif mutation is not None:
        raise ValueError("unknown synthetic mutation")
    return governor, request, _time(case.get("evaluation_time", corpus["evaluation_time"]))


def metadata_presence(request):
    """Naive comparator: required field presence and at least two rows only.

    Does not check types, schema/version values, semantic validity, chronology,
    exact scope, reviewed authority/QC, source dependence or contradictions.
    Python's permissive JSON decoder is intentional for this weak baseline.
    """
    if type(request) is bytes:
        try:
            request = json.loads(request)
        except (ValueError, UnicodeError, RecursionError):
            return False
    if type(request) is not dict or not {"schema_version", "catalog", "scope", "records"}.issubset(request):
        return False
    record_fields = {"evidence_id", "scope", "observed_at", "observation", "artifact_sha256"}
    rows = request["records"]
    return type(rows) is list and len(rows) >= 2 and all(type(r) is dict and record_fields.issubset(r) for r in rows)


def _counts(expected, observed):
    return {"true_ready": sum(a and b for a, b in zip(expected, observed)),
            "true_not_ready": sum(not a and not b for a, b in zip(expected, observed)),
            "false_ready": sum(not a and b for a, b in zip(expected, observed)),
            "false_not_ready": sum(a and not b for a, b in zip(expected, observed))}


def report():
    corpus = read_frozen("cases.json", FROZEN_CASES_SHA256)
    expected = read_frozen("expected.json", FROZEN_EXPECTED_SHA256)["labels"]
    ids = [case["id"] for case in corpus["cases"]]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("invalid fixture / label identity set")
    results = []
    for case in corpus["cases"]:
        governor, request, clock = build_case(corpus, case)
        decision = governor.evaluate(request, evaluation_time=clock)
        results.append({"id": case["id"], "expected": expected[case["id"]],
                        "observed": decision.audit,
                        "comparison_ready": {"accept_all": True, "abstain_all": False,
                                             "metadata_presence": metadata_presence(request)}})
    expected_ready = [r["expected"][0] == "review_ready" for r in results]
    models = {"governor": _counts(expected_ready, [r["observed"]["outcome"] == "review_ready" for r in results])}
    for name in ("accept_all", "abstain_all", "metadata_presence"):
        models[name] = _counts(expected_ready, [r["comparison_ready"][name] for r in results])
    exact_matches = sum(r["expected"] == [r["observed"]["outcome"], r["observed"]["reason_codes"]] for r in results)
    return {"format": "opennam-engineering-observations-v1", "case_count": len(results),
            "cases_sha256": FROZEN_CASES_SHA256, "expected_sha256": FROZEN_EXPECTED_SHA256,
            "exact_governor_outcome_and_reason_matches": exact_matches,
            "readiness_comparison": models, "results": results}


def canonical_report():
    return json.dumps(report(), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
