# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Closed, immutable synthetic study-metadata contracts; no scientific inference."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Any, Dict, Tuple

SCHEMA_VERSION = "1.0"
MAX_RECORDS = 32
MAX_CATALOG_ENTRIES = 256
MAX_JSON_BYTES = 32_768
_ID = re.compile(r"syn-[a-z0-9](?:[a-z0-9._-]{0,57}[a-z0-9])?\Z")
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


class ContractError(ValueError):
    """Controlled structural error; intentionally does not echo input values."""


def _require(condition: bool) -> None:
    if not condition:
        raise ContractError("invalid synthetic evidence metadata")


def _id(value: Any) -> None:
    _require(type(value) is str and _ID.fullmatch(value) is not None)


def _digest(value: Any) -> None:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None)


def _version(value: Any) -> None:
    _require(type(value) is int and 1 <= value <= 2**31 - 1)


def validate_time(value: Any) -> None:
    _require(type(value) is datetime and value.tzinfo is timezone.utc
             and value.microsecond == 0 and value.fold == 0)


def _parse_time(value: Any) -> datetime:
    _require(type(value) is str and _TIME.fullmatch(value) is not None)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ContractError("invalid synthetic evidence timestamp") from None


def _format_time(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _keys(value: Any, names: Tuple[str, ...]) -> None:
    _require(type(value) is dict and len(value) == len(names)
             and all(type(k) is str for k in value) and set(value) == set(names))


def _enum(cls: Any, value: Any) -> Any:
    _require(type(value) is str)
    try:
        return cls(value)
    except ValueError:
        raise ContractError("invalid categorical metadata") from None


class Observation(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"


class ReviewStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Outcome(str, Enum):
    INVALID = "invalid"
    ABSTAIN = "abstain"
    REVIEW_READY = "review_ready"


class Reason(str, Enum):
    MALFORMED_REQUEST = "malformed_request"
    INVALID_CONFIGURATION = "invalid_configuration"
    INVALID_TIME = "invalid_time"
    CATALOG_MISMATCH = "catalog_mismatch"
    SCOPE_NOT_ALLOWED = "scope_not_allowed"
    SCOPE_MISMATCH = "scope_mismatch"
    MISSING_EVIDENCE = "missing_evidence"
    UNREVIEWED_EVIDENCE = "unreviewed_evidence"
    INCOMPLETE_CATALOG_SCOPE = "incomplete_catalog_scope"
    BINDING_MISMATCH = "binding_mismatch"
    DUPLICATE_ARTIFACT = "duplicate_artifact"
    STALE_OR_FUTURE = "stale_or_future"
    REPORTING_NOT_PASSED = "reporting_not_passed"
    CONTROLS_NOT_PASSED = "controls_not_passed"
    APPLICABILITY_NOT_PASSED = "applicability_not_passed"
    INCONCLUSIVE = "inconclusive"
    CONFLICT = "conflict"
    INSUFFICIENT_GROUPS = "insufficient_groups"
    METADATA_COMPLETE = "metadata_complete"


@dataclass(frozen=True, order=True)
class Scope:
    """A declared exact-match scope, not proof of scientific comparability."""

    material_id: str
    endpoint_id: str
    method_id: str
    protocol_version: str
    unit_id: str
    context_id: str

    def __post_init__(self) -> None:
        for value in (self.material_id, self.endpoint_id, self.method_id,
                      self.protocol_version, self.unit_id, self.context_id):
            _id(value)

    @classmethod
    def from_dict(cls, value: Any) -> "Scope":
        _keys(value, ("material_id", "endpoint_id", "method_id", "protocol_version",
                      "unit_id", "context_id"))
        return cls(**value)

    def to_dict(self) -> Dict[str, str]:
        return {"material_id": self.material_id, "endpoint_id": self.endpoint_id,
                "method_id": self.method_id, "protocol_version": self.protocol_version,
                "unit_id": self.unit_id, "context_id": self.context_id}


@dataclass(frozen=True)
class CatalogRef:
    catalog_id: str
    version: int

    def __post_init__(self) -> None:
        _id(self.catalog_id)
        _version(self.version)

    @classmethod
    def from_dict(cls, value: Any) -> "CatalogRef":
        _keys(value, ("catalog_id", "version"))
        return cls(**value)

    def to_dict(self) -> Dict[str, Any]:
        return {"catalog_id": self.catalog_id, "version": self.version}


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    scope: Scope
    observed_at: datetime
    observation: Observation
    artifact_sha256: str

    def __post_init__(self) -> None:
        _id(self.evidence_id)
        _require(type(self.scope) is Scope and type(self.observation) is Observation)
        validate_time(self.observed_at)
        _digest(self.artifact_sha256)

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceRecord":
        _keys(value, ("evidence_id", "scope", "observed_at", "observation", "artifact_sha256"))
        return cls(value["evidence_id"], Scope.from_dict(value["scope"]),
                   _parse_time(value["observed_at"]), _enum(Observation, value["observation"]),
                   value["artifact_sha256"])

    def to_dict(self) -> Dict[str, Any]:
        return {"evidence_id": self.evidence_id, "scope": self.scope.to_dict(),
                "observed_at": _format_time(self.observed_at), "observation": self.observation.value,
                "artifact_sha256": self.artifact_sha256}

    @property
    def metadata_sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvidenceRequest:
    schema_version: str
    catalog: CatalogRef
    scope: Scope
    records: Tuple[EvidenceRecord, ...]

    def __post_init__(self) -> None:
        _require(type(self.schema_version) is str and self.schema_version == SCHEMA_VERSION)
        _require(type(self.catalog) is CatalogRef and type(self.scope) is Scope)
        _require(type(self.records) is tuple and len(self.records) <= MAX_RECORDS
                 and all(type(r) is EvidenceRecord for r in self.records))
        _require(len({r.evidence_id for r in self.records}) == len(self.records))
        object.__setattr__(self, "records", tuple(sorted(self.records, key=lambda r: r.evidence_id)))

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceRequest":
        _keys(value, ("schema_version", "catalog", "scope", "records"))
        records = value["records"]
        _require(type(records) is list and len(records) <= MAX_RECORDS)
        return cls(value["schema_version"], CatalogRef.from_dict(value["catalog"]),
                   Scope.from_dict(value["scope"]), tuple(EvidenceRecord.from_dict(r) for r in records))

    @classmethod
    def from_json(cls, value: Any) -> "EvidenceRequest":
        """Bounded UTF-8 bytes; duplicate object keys and noninteger numbers fail."""
        _require(type(value) is bytes and 0 < len(value) <= MAX_JSON_BYTES)

        def pairs(items: Any) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for key, item in items:
                _require(key not in result)
                result[key] = item
            return result

        def bounded_int(raw: str) -> int:
            _require(len(raw) <= 10)
            value = int(raw)
            _version(value)
            return value

        def invalid_number(raw: str) -> Any:
            raise ContractError("invalid JSON number")

        try:
            decoded = json.loads(value.decode("utf-8"), object_pairs_hook=pairs,
                                 parse_int=bounded_int, parse_float=invalid_number,
                                 parse_constant=invalid_number)
        except (UnicodeError, ValueError, RecursionError):
            raise ContractError("invalid synthetic evidence JSON") from None
        return cls.from_dict(decoded)

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "catalog": self.catalog.to_dict(),
                "scope": self.scope.to_dict(), "records": [r.to_dict() for r in self.records]}


@dataclass(frozen=True)
class CatalogEntry:
    """Independent reviewed metadata facts, not assertions in the request."""

    evidence_id: str
    scope: Scope
    metadata_sha256: str
    source_group: str
    reviewed_at: datetime
    reporting: ReviewStatus
    positive_control: ReviewStatus
    negative_control: ReviewStatus
    applicability: ReviewStatus

    def __post_init__(self) -> None:
        _id(self.evidence_id)
        _require(type(self.scope) is Scope)
        _digest(self.metadata_sha256)
        _id(self.source_group)
        validate_time(self.reviewed_at)
        _require(all(type(item) is ReviewStatus for item in
                     (self.reporting, self.positive_control, self.negative_control, self.applicability)))


@dataclass(frozen=True)
class EvidenceCatalog:
    reference: CatalogRef
    entries: Tuple[CatalogEntry, ...]

    def __post_init__(self) -> None:
        _require(type(self.reference) is CatalogRef and type(self.entries) is tuple
                 and len(self.entries) <= MAX_CATALOG_ENTRIES
                 and all(type(e) is CatalogEntry for e in self.entries))
        _require(len({e.evidence_id for e in self.entries}) == len(self.entries))
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda e: e.evidence_id)))


@dataclass(frozen=True)
class ReviewPolicy:
    """Trusted engineering gate configuration. No defaults or release authority."""

    policy_id: str
    version: int
    catalog: CatalogRef
    allowed_scopes: Tuple[Scope, ...]
    min_source_groups: int
    max_age_days: int

    def __post_init__(self) -> None:
        _id(self.policy_id)
        _version(self.version)
        _require(type(self.catalog) is CatalogRef and type(self.allowed_scopes) is tuple
                 and len(self.allowed_scopes) <= 32
                 and all(type(s) is Scope for s in self.allowed_scopes))
        _require(len(set(self.allowed_scopes)) == len(self.allowed_scopes))
        _require(type(self.min_source_groups) is int and 2 <= self.min_source_groups <= 8)
        _require(type(self.max_age_days) is int and 1 <= self.max_age_days <= 3650)
        object.__setattr__(self, "allowed_scopes", tuple(sorted(self.allowed_scopes)))
