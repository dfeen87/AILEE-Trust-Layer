"""Synthetic contract conformance, not assay validation."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from ailee.domains.opennam import (
    CatalogEntry, CatalogRef, ContractError, EvidenceCatalog, EvidenceRecord,
    EvidenceRequest, Observation, ReviewPolicy, ReviewStatus, Scope,
)

TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)
SCOPE = Scope("syn-material", "syn-endpoint", "syn-method", "syn-v1", "syn-unit", "syn-context")
REF = CatalogRef("syn-catalog", 1)


def record(n=1):
    return EvidenceRecord(f"syn-e{n}", SCOPE, TIME, Observation.NEGATIVE, f"{n:064x}")


def request():
    return EvidenceRequest("1.0", REF, SCOPE, (record(2), record()))


def reviewed():
    return CatalogEntry("syn-e1", SCOPE, record().metadata_sha256, "syn-group-a", TIME,
                        *([ReviewStatus.PASS] * 4))


def policy():
    return ReviewPolicy("syn-policy", 1, REF, (SCOPE,), 2, 30)


def test_immutable_canonical_metadata():
    value = request()
    assert [r.evidence_id for r in value.records] == ["syn-e1", "syn-e2"]
    assert EvidenceRequest.from_dict(value.to_dict()) == value
    assert EvidenceRequest.from_json(json.dumps(value.to_dict()).encode()) == value
    assert EvidenceRequest.from_dict({**value.to_dict(), "records": value.to_dict()["records"][::-1]}) == value
    assert len(value.records[0].metadata_sha256) == 64
    assert record().metadata_sha256 != replace(record(), observation=Observation.POSITIVE).metadata_sha256
    assert record().metadata_sha256 != replace(record(), artifact_sha256="a" * 64).metadata_sha256
    with pytest.raises(FrozenInstanceError):
        value.records[0].evidence_id = "syn-changed"
    detached = value.to_dict()
    detached["records"][0]["scope"]["material_id"] = "syn-modified"
    assert value.records[0].scope == SCOPE


@pytest.mark.parametrize("patch", [
    {"schema_version": "2.0"}, {"schema_version": 1}, {"policy": {}},
    {"catalog": {"catalog_id": "syn-catalog", "version": True}},
    {"catalog": {"catalog_id": "syn-catalog", "version": 10**100}},
    {"catalog": {"catalog_id": "syn-catalog", "version": float("nan")}},
    {"scope": {**SCOPE.to_dict(), "extra": 1}}, {"records": [record().to_dict()] * 33},
    {"records": [record().to_dict()] * 2}, {"records": tuple()},
])
def test_closed_dict_ingress(patch):
    with pytest.raises(ContractError):
        EvidenceRequest.from_dict({**request().to_dict(), **patch})


@pytest.mark.parametrize("patch", [
    {"evidence_id": "sensitive content"}, {"evidence_id": "syn-α"},
    {"evidence_id": "syn-a" * 30}, {"artifact_sha256": "A" * 64},
    {"observation": "approved"}, {"observation": True}, {"observed_at": "2025-02-30T00:00:00Z"},
    {"observed_at": "2025-01-01T00:00:00+00:00"}, {"observed_at": "2025-01-01T00:00:00.0Z"},
    {"source_group": "syn-untrusted"}, {"positive_control": "pass"}, {"values": [1.0]},
])
def test_closed_evidence(patch):
    with pytest.raises(ContractError) as exc:
        EvidenceRequest.from_dict({**request().to_dict(), "records": [{**record().to_dict(), **patch}]})
    assert "sensitive content" not in str(exc.value)


@pytest.mark.parametrize("encoded", [
    b'{"schema_version":"1.0","schema_version":"1.0"}',
    b'{"catalog":{"version":1,"version":2}}', b'{"x":NaN}', b'{"x":Infinity}',
    b'{"x":1.0}', b'{"x":1e300}', b'{"x":' + b'9' * 1000 + b'}',
    b'[' * 2000 + b']' * 2000, b' ' * 32769, b'\xff', b'{} trailing', "{}", b'',
])
def test_strict_json(encoded):
    with pytest.raises(ContractError):
        EvidenceRequest.from_json(encoded)


def test_typed_records_and_configuration_reject_invalid_values():
    for bad in (TIME.replace(tzinfo=None), TIME.replace(microsecond=1),
                TIME.replace(tzinfo=timezone(timedelta(hours=1))), TIME.replace(fold=1)):
        with pytest.raises(ContractError):
            replace(record(), observed_at=bad)
    with pytest.raises(ContractError):
        replace(request(), records=[record()])
    with pytest.raises(ContractError):
        replace(reviewed(), reporting="pass")
    with pytest.raises(ContractError):
        EvidenceCatalog(REF, (reviewed(), reviewed()))
    for bad in (True, 1, 9, 2.0):
        with pytest.raises(ContractError):
            replace(policy(), min_source_groups=bad)
    with pytest.raises(ContractError):
        replace(policy(), max_age_days=10**100)
    with pytest.raises(ContractError):
        replace(policy(), allowed_scopes=(SCOPE, SCOPE))
