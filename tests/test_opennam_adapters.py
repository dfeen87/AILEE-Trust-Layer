"""Offline fictional metadata adapters, not scientific method validation."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from hashlib import sha256
import json

import pytest

from ailee.domains.opennam import (
    CatalogEntry, CatalogRef, ContractError, EvidenceCatalog, EvidenceGovernor,
    EvidenceRecord, Observation, Outcome, Reason, ReviewPolicy, ReviewStatus, Scope,
)
from ailee.domains.opennam.adapters import (
    ADAPTER_VERSION, CSV_HEADER, MAX_ARTIFACT_BYTES, MAX_BATCH_BYTES, MAX_ENVELOPE_BYTES,
    AdaptationError, AdaptationResult, AdapterFormat, AdapterProfile, AdapterReason,
    EvidenceAdapter, SourcePacket,
)

TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)
TIME_STRING = "2025-01-01T00:00:00Z"
ARTIFACT_A = b"original fictional report A; opaque bytes"
ARTIFACT_B = b"original fictional report B; opaque bytes\x00\xff"
SCOPE_A = Scope("syn-material", "syn-endpoint-a", "syn-method-a", "syn-protocol-a1", "syn-unit-a", "syn-context-a")
SCOPE_B = Scope("syn-material", "syn-endpoint-b", "syn-method-b", "syn-protocol-b2", "syn-unit-b", "syn-context-b")
CATALOG = CatalogRef("syn-catalog", 1)


def profile_a():
    return AdapterProfile(AdapterFormat.JSON_ENVELOPE, ADAPTER_VERSION, "syn-profile-a", 1, SCOPE_A, CATALOG)


def profile_b():
    return AdapterProfile(AdapterFormat.CSV_ROW, ADAPTER_VERSION, "syn-profile-b", 2, SCOPE_B, CATALOG)


def value(profile, evidence_id="syn-row-a", artifact=ARTIFACT_A, token=None):
    default_token = "negative" if profile.adapter_format is AdapterFormat.JSON_ENVELOPE else "ABSTRACT_NEG"
    return {**profile.to_dict(), "record": {
        "record_id": evidence_id, "observed_at": TIME_STRING,
        "result_token": default_token if token is None else token,
        "artifact_sha256": sha256(artifact).hexdigest(),
    }}


def serialize(profile, payload):
    if profile.adapter_format is AdapterFormat.JSON_ENVELOPE:
        return json.dumps(payload, sort_keys=True).encode("utf-8")
    fields = {**payload, **payload["scope"], **payload["record"],
              "catalog_id": payload["catalog"]["catalog_id"], "catalog_version": payload["catalog"]["version"]}
    return (CSV_HEADER + "\n" + ",".join(str(fields[k]) for k in CSV_HEADER.split(",")) + "\n").encode("ascii")


def packet(profile, evidence_id="syn-row-a", artifact=ARTIFACT_A, token=None):
    return SourcePacket(serialize(profile, value(profile, evidence_id, artifact, token)), artifact)


def reject(expected, function, *args, **kwargs):
    with pytest.raises(AdaptationError) as exc:
        function(*args, **kwargs)
    assert exc.value.reason is expected
    assert str(exc.value) == "synthetic adapter rejected input: " + expected.value


def expected_record(scope, evidence_id, artifact, observation=Observation.NEGATIVE):
    return EvidenceRecord(evidence_id, scope, TIME, observation, sha256(artifact).hexdigest())


def independent_governor(profile, *, missing_controls=False, same_group=False, omit=False):
    # Known authored fixture facts independently constructed before any adapter
    # results; never lift reporting, QC, group or authority from an adapter.
    records = (expected_record(profile.scope, "syn-row-a", ARTIFACT_A),
               expected_record(profile.scope, "syn-row-b", ARTIFACT_B))
    entries = tuple(CatalogEntry(r.evidence_id, r.scope, r.metadata_sha256,
                                "syn-group-a" if same_group else f"syn-group-{i}", TIME,
                                ReviewStatus.PASS, ReviewStatus.PASS,
                                ReviewStatus.UNKNOWN if missing_controls else ReviewStatus.PASS,
                                ReviewStatus.PASS) for i, r in enumerate(records))
    return EvidenceGovernor(policy=ReviewPolicy("syn-policy", 1, profile.catalog, (profile.scope,), 2, 30),
                            catalog=EvidenceCatalog(profile.catalog, entries[:1] if omit else entries))


@pytest.mark.parametrize("make_profile", [profile_a, profile_b])
def test_two_distinct_fictional_profiles_keep_catalog_authority(make_profile):
    profile = make_profile()
    adapter = EvidenceAdapter(profile=profile)
    governor = independent_governor(profile)
    first = packet(profile)
    second = packet(profile, "syn-row-b", ARTIFACT_B)
    actual = adapter.adapt((second, first))
    assert actual.request.records == (expected_record(profile.scope, "syn-row-a", ARTIFACT_A),
                                      expected_record(profile.scope, "syn-row-b", ARTIFACT_B))
    assert actual.request.scope == profile.scope and actual.request.catalog == profile.catalog
    assert actual == adapter.adapt((first, second))
    assert actual == adapter.adapt((second, first))
    assert governor.evaluate(actual.request, evaluation_time=TIME).outcome is Outcome.REVIEW_READY
    assert governor.evaluate(adapter.adapt((first,)).request, evaluation_time=TIME).reason_codes == (Reason.INCOMPLETE_CATALOG_SCOPE,)
    assert independent_governor(profile, missing_controls=True).evaluate(actual.request, evaluation_time=TIME).reason_codes == (Reason.CONTROLS_NOT_PASSED,)
    assert independent_governor(profile, same_group=True).evaluate(actual.request, evaluation_time=TIME).reason_codes == (Reason.INSUFFICIENT_GROUPS,)
    assert independent_governor(profile, omit=True).evaluate(actual.request, evaluation_time=TIME).reason_codes == (Reason.UNREVIEWED_EVIDENCE,)
    assert EvidenceGovernor(policy=None, catalog=None).evaluate(actual.request, evaluation_time=TIME).reason_codes == (Reason.INVALID_CONFIGURATION,)
    with pytest.raises(FrozenInstanceError):
        actual.receipts[0].profile_id = "syn-changed"
    for source, record, receipt in zip((first, second), actual.request.records, actual.receipts):
        assert receipt.evidence_id == record.evidence_id
        assert receipt.adapter_format is profile.adapter_format and receipt.adapter_version == 1
        assert receipt.profile_id == profile.profile_id and receipt.profile_version == profile.profile_version
        assert receipt.profile_sha256 == profile.profile_sha256
        assert receipt.envelope_sha256 == sha256(source.envelope).hexdigest()
        assert receipt.artifact_sha256 == sha256(source.artifact).hexdigest()
        assert receipt.metadata_sha256 == record.metadata_sha256
        assert len(receipt.receipt_sha256) == 64
        assert "original fictional report" not in repr(receipt)


@pytest.mark.parametrize("json_token,csv_token,observation", [
    ("positive", "ABSTRACT_POS", Observation.POSITIVE),
    ("negative", "ABSTRACT_NEG", Observation.NEGATIVE),
    ("inconclusive", "ABSTRACT_UNRESOLVED", Observation.INCONCLUSIVE),
])
def test_equivalent_categorical_representation_and_distinct_receipts(json_token, csv_token, observation):
    first = profile_a()
    second = replace(first, adapter_format=AdapterFormat.CSV_ROW, profile_id="syn-profile-csv-same-scope")
    a = EvidenceAdapter(profile=first).adapt((packet(first, token=json_token),))
    b = EvidenceAdapter(profile=second).adapt((packet(second, token=csv_token),))
    assert a.request == b.request
    assert a.request.records[0].observation is observation
    assert a.receipts[0].metadata_sha256 == b.receipts[0].metadata_sha256
    assert a.receipts[0].artifact_sha256 == b.receipts[0].artifact_sha256
    assert a.receipts[0].envelope_sha256 != b.receipts[0].envelope_sha256
    assert a.receipts[0].profile_sha256 != b.receipts[0].profile_sha256
    assert a.receipts[0].receipt_sha256 != b.receipts[0].receipt_sha256


def test_canonical_output_but_byte_sensitive_receipt():
    profile = profile_a()
    adapter = EvidenceAdapter(profile=profile)
    payload = value(profile)
    raw = json.dumps(payload, indent=2).encode("utf-8")
    canonical = adapter.adapt((packet(profile),))
    alternate = adapter.adapt((SourcePacket(raw, ARTIFACT_A),))
    assert alternate.request == canonical.request
    assert alternate.receipts[0].envelope_sha256 != canonical.receipts[0].envelope_sha256
    assert alternate.receipts[0].metadata_sha256 == canonical.receipts[0].metadata_sha256
    changed = replace(profile, profile_version=2)
    updated = EvidenceAdapter(profile=changed).adapt((packet(changed),))
    assert updated.request == canonical.request
    assert updated.receipts[0].profile_sha256 != canonical.receipts[0].profile_sha256
    assert updated.receipts[0].receipt_sha256 != canonical.receipts[0].receipt_sha256


@pytest.mark.parametrize("field,new_value", [
    ("adapter_id", "syn-importlib.import_module"), ("adapter_version", 2),
    ("profile_id", "syn-os.system"), ("profile_version", 2),
    ("catalog", {"catalog_id": "syn-other", "version": 1}),
])
def test_raw_profile_and_catalog_binding(field, new_value):
    profile = profile_a()
    payload = {**value(profile), field: new_value}
    reject(AdapterReason.PROFILE_MISMATCH, EvidenceAdapter(profile=profile).adapt,
           (SourcePacket(serialize(profile, payload), ARTIFACT_A),))


@pytest.mark.parametrize("field", ["material_id", "endpoint_id", "method_id", "protocol_version", "unit_id", "context_id"])
@pytest.mark.parametrize("make_profile", [profile_a, profile_b])
def test_exact_scope_rejects_cross_method_unit_and_other_drift(field, make_profile):
    profile = make_profile()
    payload = value(profile)
    payload["scope"][field] = "syn-other"
    reject(AdapterReason.PROFILE_MISMATCH, EvidenceAdapter(profile=profile).adapt,
           (SourcePacket(serialize(profile, payload), ARTIFACT_A),))


@pytest.mark.parametrize("patch", [
    {"adapter_version": True}, {"adapter_version": 1.0}, {"adapter_version": float("nan")},
    {"profile_version": 10**100}, {"profile_version": "1"}, {"catalog": {"catalog_id": "syn-catalog", "version": False}},
    {"schema_version": "1.0"}, {"policy": {}}, {"catalog_entries": []},
    {"scope": {**SCOPE_A.to_dict(), "extra": "syn-info"}},
])
def test_strict_json_versions_and_closed_envelope(patch):
    profile = profile_a()
    payload = {**value(profile), **patch}
    reject(AdapterReason.MALFORMED_ENVELOPE, EvidenceAdapter(profile=profile).adapt,
           (SourcePacket(serialize(profile, payload), ARTIFACT_A),))


@pytest.mark.parametrize("patch", [
    {"result_token": "N/A"}, {"result_token": "unknown"}, {"result_token": "NEGATIVE"},
    {"result_token": 0}, {"result_token": True}, {"result_token": "0.1"},
    {"negative_control": "pass"}, {"positive_control": "N/A"}, {"reporting": "pass"},
    {"applicability": "pass"}, {"source_group": "syn-pretend-independent"}, {"measurement": [1, 2]},
    {"observed_at": "2025-01-01T00:00:00+00:00"}, {"observed_at": "2025-02-30T00:00:00Z"},
    {"record_id": "secret rejected content"}, {"artifact_sha256": "A" * 64},
])
def test_uninterpreted_or_authority_injection_and_bad_metadata_rejected(patch):
    profile = profile_a()
    payload = value(profile)
    payload["record"].update(patch)
    source = SourcePacket(serialize(profile, payload), ARTIFACT_A)
    assert "secret rejected content" not in repr(source) and "fictional report" not in repr(source)
    reject(AdapterReason.MALFORMED_ENVELOPE, EvidenceAdapter(profile=profile).adapt, (source,))


def test_json_duplicate_keys_utf8_truncation_and_deep_nesting():
    profile = profile_a()
    source = packet(profile).envelope
    valid = json.loads(source)
    duplicate_nested = source.replace(b'"record": {', b'"record": {"record_id": "syn-other",', 1)
    duplicate_top = b'{"adapter_id":"syn-json-envelope",' + source[1:]
    for raw in (duplicate_nested, duplicate_top, source[:-1], source + b' trailing', b'\xff',
                b'[' * 2000 + b']' * 2000, b'{"record": Infinity}'):
        reject(AdapterReason.MALFORMED_ENVELOPE, EvidenceAdapter(profile=profile).adapt, (SourcePacket(raw, ARTIFACT_A),))
    assert valid["record"]["record_id"] == "syn-row-a"


@pytest.mark.parametrize("mutation", [
    lambda b: b[:-1], lambda b: b.replace(b"\n", b"\r\n"), lambda b: b + b"extra\n",
    lambda b: b.replace(b"adapter_version", b"adapter_id", 1),
    lambda b: b.replace(b"adapter_id", b"unknown", 1), lambda b: b[:-1] + b",extra\n",
    lambda b: b.replace(b"ABSTRACT_NEG", b"N/A"), lambda b: b.replace(b"ABSTRACT_NEG", b"0.1"),
    lambda b: b.replace(b"ABSTRACT_NEG", b'"ABSTRACT_NEG"'), lambda b: b.replace(b"ABSTRACT_NEG", b"\xff"),
    lambda b: b.replace(b"syn-csv-row,1,", b"syn-csv-row,01,"),
    lambda b: b.replace(b"syn-csv-row,1,", b"syn-csv-row,True,"),
    lambda b: b.replace(b"syn-csv-row,1,", b"syn-csv-row,99999999999,"),
    lambda b: b.replace(b"syn-csv-row,1,", b"syn-csv-row,NaN,"),
])
def test_csv_exact_dialect_and_malformed_rows(mutation):
    profile = profile_b()
    raw = mutation(packet(profile).envelope)
    reject(AdapterReason.MALFORMED_ENVELOPE, EvidenceAdapter(profile=profile).adapt, (SourcePacket(raw, ARTIFACT_A),))


def test_csv_profile_binding_and_wrong_format_cannot_relabel():
    profile = profile_b()
    raw = packet(profile).envelope.replace(b"syn-csv-row,1,", b"syn-csv-row,2,")
    reject(AdapterReason.PROFILE_MISMATCH, EvidenceAdapter(profile=profile).adapt, (SourcePacket(raw, ARTIFACT_A),))
    reject(AdapterReason.MALFORMED_ENVELOPE, EvidenceAdapter(profile=profile).adapt, (packet(profile_a()),))


@pytest.mark.parametrize("make_profile", [profile_a, profile_b])
def test_actual_artifact_bytes_mismatch_and_duplicate_safety(make_profile):
    profile = make_profile()
    adapter = EvidenceAdapter(profile=profile)
    first = packet(profile)
    reject(AdapterReason.ARTIFACT_MISMATCH, adapter.adapt, (SourcePacket(first.envelope, ARTIFACT_A + b"!"),))
    reject(AdapterReason.DUPLICATE_EVIDENCE, adapter.adapt, (first, packet(profile, artifact=ARTIFACT_B)))
    reject(AdapterReason.DUPLICATE_ARTIFACT, adapter.adapt, (first, packet(profile, "syn-row-b", ARTIFACT_A)))
    # Distinct encoding/bytes can still represent one study; catalog groups,
    # not receipt or byte uniqueness, enforce the declared group requirement.
    distinct = adapter.adapt((first, packet(profile, "syn-row-b", ARTIFACT_B)))
    assert independent_governor(profile, same_group=True).evaluate(distinct.request, evaluation_time=TIME).reason_codes == (Reason.INSUFFICIENT_GROUPS,)


def test_packet_and_aggregate_bounds_precede_decode():
    reject(AdapterReason.INVALID_PACKET, SourcePacket, "payload", ARTIFACT_A)
    reject(AdapterReason.INVALID_PACKET, SourcePacket, b"payload", bytearray(ARTIFACT_A))
    reject(AdapterReason.BOUNDS_EXCEEDED, SourcePacket, b"x" * (MAX_ENVELOPE_BYTES + 1), ARTIFACT_A)
    reject(AdapterReason.BOUNDS_EXCEEDED, SourcePacket, b"x", b"a" * (MAX_ARTIFACT_BYTES + 1))
    reject(AdapterReason.BOUNDS_EXCEEDED, SourcePacket, b"x", b"")
    adapter = EvidenceAdapter(profile=profile_a())
    reject(AdapterReason.INVALID_PACKET, adapter.adapt, [packet(profile_a())])
    reject(AdapterReason.BOUNDS_EXCEEDED, adapter.adapt, ())
    reject(AdapterReason.BOUNDS_EXCEEDED, adapter.adapt, (packet(profile_a()),) * 33)
    reject(AdapterReason.INVALID_PACKET, adapter.adapt, (object(),))
    # Entire bounded set is counted before malformed envelopes are parsed.
    large = SourcePacket(b"invalid envelope", b"x" * MAX_ARTIFACT_BYTES)
    count = MAX_BATCH_BYTES // MAX_ARTIFACT_BYTES + 1
    reject(AdapterReason.BOUNDS_EXCEEDED, adapter.adapt, (large,) * count)


def test_profiles_and_packets_are_deep_revalidated_and_detached():
    profile = profile_a()
    adapter = EvidenceAdapter(profile=profile)
    source = packet(profile)
    expected = adapter.adapt((source,))
    object.__setattr__(profile.scope, "unit_id", "syn-tampered")
    object.__setattr__(profile.catalog, "version", True)
    assert adapter.adapt((source,)) == expected
    reject(AdapterReason.INVALID_PROFILE, EvidenceAdapter, profile=profile)
    reject(AdapterReason.INVALID_PROFILE, EvidenceAdapter, profile=object.__new__(AdapterProfile))
    reject(AdapterReason.INVALID_PROFILE, EvidenceAdapter, profile=object())
    reject(AdapterReason.INVALID_PROFILE, replace, profile_a(), adapter_format="syn-os.system")
    reject(AdapterReason.INVALID_PROFILE, replace, profile_a(), adapter_version=True)
    reject(AdapterReason.INVALID_PROFILE, replace, profile_a(), adapter_version=2)
    object.__setattr__(source, "artifact", bytearray(ARTIFACT_A))
    reject(AdapterReason.INVALID_PACKET, adapter.adapt, (source,))
    reject(AdapterReason.INVALID_PACKET, adapter.adapt, (object.__new__(SourcePacket),))
    reject(AdapterReason.INVALID_PROFILE, object.__new__(EvidenceAdapter).adapt, (packet(profile_a()),))


def test_receipt_linkage_does_not_authenticate_or_authorize():
    profile = profile_a()
    adapter = EvidenceAdapter(profile=profile)
    result = adapter.adapt((packet(profile),))
    receipt = result.receipts[0]
    with pytest.raises(ContractError):
        AdaptationResult(result.request, (replace(receipt, metadata_sha256="a" * 64),))
    with pytest.raises(ContractError):
        AdaptationResult(result.request, (replace(receipt, profile_version=2),))
    # A structurally consistent receipt can be manufactured without original
    # envelope bytes. This is intentional: the receipt is NOT a trust credential.
    forged = AdaptationResult(result.request, (replace(receipt, envelope_sha256="f" * 64),))
    assert forged.receipts[0].receipt_sha256 != receipt.receipt_sha256
    governor = EvidenceGovernor(policy=None, catalog=None)
    assert governor.evaluate(forged.request, evaluation_time=TIME).outcome is Outcome.INVALID
    assert governor.evaluate(forged, evaluation_time=TIME).outcome is Outcome.INVALID
