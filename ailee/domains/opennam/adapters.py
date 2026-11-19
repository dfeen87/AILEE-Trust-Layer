# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Closed offline adapters for fictional, already-categorical metadata only.

The adapter does not produce trusted catalog entries or scientific conclusions.
Each input packet is one metadata envelope plus exactly one opaque artifact.
"""

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Dict, NoReturn, Tuple

from .contracts import (
    CatalogRef, ContractError, EvidenceRecord, EvidenceRequest, MAX_RECORDS,
    Observation, SCHEMA_VERSION, Scope, _digest, _id, _keys, _parse_time,
    _require, _version,
)

ADAPTER_VERSION = 1
MAX_ENVELOPE_BYTES = 8192
MAX_ARTIFACT_BYTES = 65536
MAX_BATCH_BYTES = 1048576


class AdapterFormat(str, Enum):
    """Closed parser allowlist. No import, plugin or executable identifier."""

    JSON_ENVELOPE = "syn-json-envelope"
    CSV_ROW = "syn-csv-row"


class AdapterReason(str, Enum):
    INVALID_PROFILE = "invalid_profile"
    INVALID_PACKET = "invalid_packet"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    MALFORMED_ENVELOPE = "malformed_envelope"
    PROFILE_MISMATCH = "profile_mismatch"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    DUPLICATE_EVIDENCE = "duplicate_evidence"
    DUPLICATE_ARTIFACT = "duplicate_artifact"


class AdaptationError(ValueError):
    """Fixed failure category without an input-derived diagnostic."""

    def __init__(self, reason: AdapterReason) -> None:
        self.reason = reason
        super().__init__("synthetic adapter rejected input: " + reason.value)


def _fail(reason: AdapterReason) -> NoReturn:
    raise AdaptationError(reason)


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _scope(value: Any) -> Scope:
    _require(type(value) is Scope)
    return Scope(value.material_id, value.endpoint_id, value.method_id,
                 value.protocol_version, value.unit_id, value.context_id)


def _reference(value: Any) -> CatalogRef:
    _require(type(value) is CatalogRef)
    return CatalogRef(value.catalog_id, value.version)


@dataclass(frozen=True)
class AdapterProfile:
    """Host-selected parser revision and exact fictional scope/catalog binding."""

    adapter_format: AdapterFormat
    adapter_version: int
    profile_id: str
    profile_version: int
    scope: Scope
    catalog: CatalogRef

    def __post_init__(self) -> None:
        try:
            _require(type(self.adapter_format) is AdapterFormat and self.adapter_format in
                     (AdapterFormat.JSON_ENVELOPE, AdapterFormat.CSV_ROW))
            _require(type(self.adapter_version) is int and self.adapter_version == ADAPTER_VERSION)
            _id(self.profile_id)
            _version(self.profile_version)
            object.__setattr__(self, "scope", _scope(self.scope))
            object.__setattr__(self, "catalog", _reference(self.catalog))
        except (ContractError, AttributeError, TypeError):
            raise AdaptationError(AdapterReason.INVALID_PROFILE) from None

    def to_dict(self) -> Dict[str, Any]:
        return {"adapter_id": self.adapter_format.value,
                "adapter_version": self.adapter_version,
                "profile_id": self.profile_id, "profile_version": self.profile_version,
                "scope": self.scope.to_dict(), "catalog": self.catalog.to_dict()}

    @property
    def profile_sha256(self) -> str:
        return _hash(self.to_dict())


def _profile(value: Any) -> AdapterProfile:
    if type(value) is not AdapterProfile:
        _fail(AdapterReason.INVALID_PROFILE)
    try:
        return AdapterProfile(value.adapter_format, value.adapter_version, value.profile_id,
                              value.profile_version, value.scope, value.catalog)
    except (AttributeError, TypeError):
        raise AdaptationError(AdapterReason.INVALID_PROFILE) from None


@dataclass(frozen=True, repr=False)
class SourcePacket:
    """One exact byte envelope and one nonempty opaque artifact, not a path."""

    envelope: bytes
    artifact: bytes

    def __post_init__(self) -> None:
        if type(self.envelope) is not bytes or type(self.artifact) is not bytes:
            _fail(AdapterReason.INVALID_PACKET)
        if not (0 < len(self.envelope) <= MAX_ENVELOPE_BYTES and
                0 < len(self.artifact) <= MAX_ARTIFACT_BYTES):
            _fail(AdapterReason.BOUNDS_EXCEEDED)

    def __repr__(self) -> str:
        return "SourcePacket(<omitted>)"


def _packet(value: Any) -> SourcePacket:
    if type(value) is not SourcePacket:
        _fail(AdapterReason.INVALID_PACKET)
    try:
        return SourcePacket(value.envelope, value.artifact)
    except (AttributeError, TypeError):
        raise AdaptationError(AdapterReason.INVALID_PACKET) from None


_PROFILE_KEYS = ("adapter_id", "adapter_version", "profile_id", "profile_version", "scope", "catalog")
_RECORD_KEYS = ("record_id", "observed_at", "result_token", "artifact_sha256")
_SCOPE_KEYS = ("material_id", "endpoint_id", "method_id", "protocol_version", "unit_id", "context_id")
_CSV_HEADER = ("adapter_id", "adapter_version", "profile_id", "profile_version",
               "catalog_id", "catalog_version", *_SCOPE_KEYS, *_RECORD_KEYS)
CSV_HEADER = ",".join(_CSV_HEADER)


def _json(envelope: bytes) -> Dict[str, Any]:
    def pairs(items: Any) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for key, item in items:
            _require(key not in values)
            values[key] = item
        return values

    def integer(raw: str) -> int:
        _require(len(raw) <= 10)
        value = int(raw)
        _version(value)
        return value

    def invalid_number(raw: str) -> Any:
        raise ContractError("invalid adapter number")

    value = json.loads(envelope.decode("utf-8"), object_pairs_hook=pairs,
                       parse_int=integer, parse_float=invalid_number,
                       parse_constant=invalid_number)
    _keys(value, (*_PROFILE_KEYS, "record"))
    _keys(value["record"], _RECORD_KEYS)
    return value


def _decimal(value: str) -> int:
    _require(1 <= len(value) <= 10 and value.isascii() and value.isdecimal()
             and not value.startswith("0"))
    integer = int(value)
    _version(integer)
    return integer


def _csv(envelope: bytes) -> Dict[str, Any]:
    # Deliberately constrained row format, not a general-purpose CSV importer.
    # No quoting, line folding, BOM, CRLF, extra columns or multirow batches.
    value = envelope.decode("ascii")
    _require(value.endswith("\n") and "\r" not in value and '"' not in value)
    lines = value.split("\n")
    _require(len(lines) == 3 and lines[0] == CSV_HEADER and lines[2] == "")
    cells = lines[1].split(",")
    _require(len(cells) == len(_CSV_HEADER) and all(cells))
    row = dict(zip(_CSV_HEADER, cells))
    return {"adapter_id": row["adapter_id"], "adapter_version": _decimal(row["adapter_version"]),
            "profile_id": row["profile_id"], "profile_version": _decimal(row["profile_version"]),
            "catalog": {"catalog_id": row["catalog_id"], "version": _decimal(row["catalog_version"])},
            "scope": {k: row[k] for k in _SCOPE_KEYS}, "record": {k: row[k] for k in _RECORD_KEYS}}


def _record(profile: AdapterProfile, packet: SourcePacket) -> EvidenceRecord:
    try:
        # Explicit finite dispatch; packet data can never select executable code.
        if profile.adapter_format is AdapterFormat.JSON_ENVELOPE:
            value = _json(packet.envelope)
        elif profile.adapter_format is AdapterFormat.CSV_ROW:
            value = _csv(packet.envelope)
        else:
            _fail(AdapterReason.INVALID_PROFILE)
        _require(type(value["adapter_id"]) is str)
        _version(value["adapter_version"])
        _id(value["profile_id"])
        _version(value["profile_version"])
        source_scope = Scope.from_dict(value["scope"])
        source_catalog = CatalogRef.from_dict(value["catalog"])
        if (value["adapter_id"] != profile.adapter_format.value or
                value["adapter_version"] != profile.adapter_version or
                value["profile_id"] != profile.profile_id or
                value["profile_version"] != profile.profile_version or
                source_scope != profile.scope or source_catalog != profile.catalog):
            _fail(AdapterReason.PROFILE_MISMATCH)
        row = value["record"]
        _id(row["record_id"])
        _digest(row["artifact_sha256"])
        _require(type(row["result_token"]) is str)
        if profile.adapter_format is AdapterFormat.JSON_ENVELOPE:
            observation = Observation(row["result_token"])
        elif profile.adapter_format is AdapterFormat.CSV_ROW:
            # Already interpreted, abstract tokens. No numeric or scientific mapping.
            observation = {"ABSTRACT_POS": Observation.POSITIVE, "ABSTRACT_NEG": Observation.NEGATIVE,
                           "ABSTRACT_UNRESOLVED": Observation.INCONCLUSIVE}[row["result_token"]]
        else:
            _fail(AdapterReason.INVALID_PROFILE)
        time = _parse_time(row["observed_at"])
    except AdaptationError:
        raise
    except (ContractError, ValueError, UnicodeError, RecursionError, KeyError, TypeError):
        raise AdaptationError(AdapterReason.MALFORMED_ENVELOPE) from None
    actual = sha256(packet.artifact).hexdigest()
    if actual != row["artifact_sha256"]:
        _fail(AdapterReason.ARTIFACT_MISMATCH)
    return EvidenceRecord(row["record_id"], source_scope, time, observation, actual)


@dataclass(frozen=True)
class AdaptationReceipt:
    """Descriptive checksum linkage only; a receipt is not authentication or trust."""

    evidence_id: str
    adapter_format: AdapterFormat
    adapter_version: int
    profile_id: str
    profile_version: int
    profile_sha256: str
    envelope_sha256: str
    artifact_sha256: str
    metadata_sha256: str

    def __post_init__(self) -> None:
        _id(self.evidence_id)
        _require(type(self.adapter_format) is AdapterFormat)
        _require(type(self.adapter_version) is int and self.adapter_version == ADAPTER_VERSION)
        _id(self.profile_id)
        _version(self.profile_version)
        for digest in (self.profile_sha256, self.envelope_sha256, self.artifact_sha256, self.metadata_sha256):
            _digest(digest)

    def to_dict(self) -> Dict[str, Any]:
        return {"evidence_id": self.evidence_id, "adapter_id": self.adapter_format.value,
                "adapter_version": self.adapter_version, "profile_id": self.profile_id,
                "profile_version": self.profile_version, "profile_sha256": self.profile_sha256,
                "envelope_sha256": self.envelope_sha256, "artifact_sha256": self.artifact_sha256,
                "metadata_sha256": self.metadata_sha256}

    @property
    def receipt_sha256(self) -> str:
        return _hash(self.to_dict())


@dataclass(frozen=True)
class AdaptationResult:
    """Untrusted governor request and companion receipts; never catalog or policy."""

    request: EvidenceRequest
    receipts: Tuple[AdaptationReceipt, ...]

    def __post_init__(self) -> None:
        try:
            self._validate()
        except (AttributeError, TypeError):
            raise ContractError("invalid synthetic adapter result") from None

    def _validate(self) -> None:
        # Structural linkage among returned fields. This cannot authenticate a
        # receipt or prove possession of the original bytes; the governor does
        # not consume receipts as trusted input.
        value = self.request
        _require(type(value) is EvidenceRequest and type(value.records) is tuple
                 and 0 < len(value.records) <= MAX_RECORDS)
        records = []
        for record in value.records:
            _require(type(record) is EvidenceRecord)
            records.append(EvidenceRecord(record.evidence_id, _scope(record.scope),
                           record.observed_at, record.observation, record.artifact_sha256))
        request = EvidenceRequest(value.schema_version, _reference(value.catalog), _scope(value.scope), tuple(records))
        _require(all(r.scope == request.scope for r in request.records))
        _require(len({r.artifact_sha256 for r in request.records}) == len(request.records))
        _require(type(self.receipts) is tuple and len(self.receipts) == len(request.records))
        receipts = []
        for receipt in self.receipts:
            _require(type(receipt) is AdaptationReceipt)
            receipts.append(AdaptationReceipt(
                receipt.evidence_id, receipt.adapter_format, receipt.adapter_version,
                receipt.profile_id, receipt.profile_version, receipt.profile_sha256,
                receipt.envelope_sha256, receipt.artifact_sha256, receipt.metadata_sha256))
        ordered = tuple(sorted(receipts, key=lambda r: r.evidence_id))
        first = ordered[0]
        profile = AdapterProfile(first.adapter_format, first.adapter_version, first.profile_id,
                                 first.profile_version, request.scope, request.catalog)
        for record, receipt in zip(request.records, ordered):
            _require(receipt.evidence_id == record.evidence_id
                     and receipt.artifact_sha256 == record.artifact_sha256
                     and receipt.metadata_sha256 == record.metadata_sha256
                     and receipt.adapter_format is profile.adapter_format
                     and receipt.adapter_version == profile.adapter_version
                     and receipt.profile_id == profile.profile_id
                     and receipt.profile_version == profile.profile_version
                     and receipt.profile_sha256 == profile.profile_sha256)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "receipts", ordered)


@dataclass(frozen=True, init=False, repr=False)
class EvidenceAdapter:
    """One explicit immutable profile per batch; all-or-nothing adaptation."""

    _selected: AdapterProfile

    def __init__(self, *, profile: Any) -> None:
        object.__setattr__(self, "_selected", _profile(profile))

    def adapt(self, packets: Any) -> AdaptationResult:
        try:
            profile = _profile(self._selected)
        except AttributeError:
            raise AdaptationError(AdapterReason.INVALID_PROFILE) from None
        if type(packets) is not tuple:
            _fail(AdapterReason.INVALID_PACKET)
        if not (0 < len(packets) <= MAX_RECORDS):
            _fail(AdapterReason.BOUNDS_EXCEEDED)
        detached = tuple(_packet(p) for p in packets)
        if sum(len(p.envelope) + len(p.artifact) for p in detached) > MAX_BATCH_BYTES:
            _fail(AdapterReason.BOUNDS_EXCEEDED)

        records = []
        receipts = []
        for packet in detached:
            record = _record(profile, packet)
            records.append(record)
            receipts.append(AdaptationReceipt(
                record.evidence_id, profile.adapter_format, profile.adapter_version,
                profile.profile_id, profile.profile_version, profile.profile_sha256,
                sha256(packet.envelope).hexdigest(), record.artifact_sha256, record.metadata_sha256,
            ))
        if len({r.evidence_id for r in records}) != len(records):
            _fail(AdapterReason.DUPLICATE_EVIDENCE)
        if len({r.artifact_sha256 for r in records}) != len(records):
            _fail(AdapterReason.DUPLICATE_ARTIFACT)
        return AdaptationResult(EvidenceRequest(SCHEMA_VERSION, _reference(profile.catalog), _scope(profile.scope),
                                               tuple(records)),
                                tuple(sorted(receipts, key=lambda r: r.evidence_id)))


__all__ = [
    "ADAPTER_VERSION", "CSV_HEADER", "MAX_ENVELOPE_BYTES", "MAX_ARTIFACT_BYTES", "MAX_BATCH_BYTES",
    "AdapterFormat", "AdapterReason", "AdaptationError", "AdapterProfile", "SourcePacket",
    "AdaptationReceipt", "AdaptationResult", "EvidenceAdapter",
]
