# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Offline synthetic Health Safety contracts and deterministic Phase 2 governor.

Constructing any of these values establishes structure only, not trust, clinical
correctness, consent, or release authority. See README.md for the scoped API
and the independent harness boundary.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple, Type, TypeVar, Union, cast

SCHEMA_VERSION = "1.0"
MAX_IDENTIFIER_LENGTH = 64
MAX_CLAIMS = 16
MAX_EVIDENCE_REFS = 64
MAX_INPUT_MARKERS = 8
MAX_CATALOG_ARTIFACTS = 128
MAX_CATALOG_EVIDENCE = 512
MAX_INDEPENDENT_SOURCES = 8
MAX_SYNTHETIC_ARTIFACT_BYTES = 4096

_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CandidateDisposition(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    DENY = "DENY"


class SupportDisposition(str, Enum):
    ORDINARY_HELP = "ORDINARY_HELP"
    QUALIFIED_REVIEW = "QUALIFIED_REVIEW"
    IMMEDIATE_PUBLIC_REFERRAL = "IMMEDIATE_PUBLIC_REFERRAL"


class SupportTemplateId(str, Enum):
    ORDINARY_HELP_V1 = "ahs.public.ordinary-help.v1"
    QUALIFIED_REVIEW_V1 = "ahs.public.qualified-review.v1"
    IMMEDIATE_PUBLIC_REFERRAL_V1 = "ahs.public.immediate-referral.v1"


class HealthPurpose(str, Enum):
    GENERAL_EDUCATION = "GENERAL_EDUCATION"
    UNCERTAINTY_AND_REFERRAL = "UNCERTAINTY_AND_REFERRAL"
    DIAGNOSIS = "DIAGNOSIS"
    TREATMENT_SELECTION = "TREATMENT_SELECTION"
    PRESCRIBING = "PRESCRIBING"
    CLINICAL_TRIAGE = "CLINICAL_TRIAGE"
    EXTERNAL_ACTION = "EXTERNAL_ACTION"


class SyntheticInput(str, Enum):
    GENERAL_QUESTION = "GENERAL_QUESTION"
    UNCERTAINTY = "UNCERTAINTY"
    PROHIBITED_ACTION_REQUEST = "PROHIBITED_ACTION_REQUEST"
    ASSERTED_AUTHORITY = "ASSERTED_AUTHORITY"
    PRIVATE_DATA_CANARY = "PRIVATE_DATA_CANARY"


class ConsentState(str, Enum):
    GRANTED = "GRANTED"
    DENIED = "DENIED"
    REVOKED = "REVOKED"


class ContextState(str, Enum):
    ASSEMBLED = "ASSEMBLED"
    INVALID = "INVALID"


class UrgencyLevel(str, Enum):
    ORDINARY = "ORDINARY"
    QUALIFIED_REVIEW = "QUALIFIED_REVIEW"
    IMMEDIATE_PUBLIC_REFERRAL = "IMMEDIATE_PUBLIC_REFERRAL"


class CatalogReviewStage(str, Enum):
    PROVISIONAL_ENGINEERING = "PROVISIONAL_ENGINEERING"
    INDEPENDENT_SYNTHETIC_ADJUDICATION = "INDEPENDENT_SYNTHETIC_ADJUDICATION"


class ArtifactAssessment(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    UNRESOLVED = "UNRESOLVED"
    UNSUPPORTED_CERTAINTY = "UNSUPPORTED_CERTAINTY"
    PROHIBITED_MEDICAL_ACTION = "PROHIBITED_MEDICAL_ACTION"
    PROHIBITED_PRIVACY = "PROHIBITED_PRIVACY"
    PROHIBITED_AUTHORITY = "PROHIBITED_AUTHORITY"


class EvidenceRelation(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    UNRESOLVED = "UNRESOLVED"


class HealthReason(str, Enum):
    SYNTHETIC_ARTIFACT_ELIGIBLE = "SYNTHETIC_ARTIFACT_ELIGIBLE"

    CANDIDATE_NOT_REVIEWED = "CANDIDATE_NOT_REVIEWED"
    UNSUPPORTED_CERTAINTY = "UNSUPPORTED_CERTAINTY"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    MISSING_CLAIMS = "MISSING_CLAIMS"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    UNRESOLVED_EVIDENCE = "UNRESOLVED_EVIDENCE"
    STALE_EVIDENCE = "STALE_EVIDENCE"
    INSUFFICIENT_INDEPENDENT_SUPPORT = "INSUFFICIENT_INDEPENDENT_SUPPORT"

    MALFORMED_SCHEMA = "MALFORMED_SCHEMA"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    INVALID_CONTEXT = "INVALID_CONTEXT"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    INVALID_POLICY = "INVALID_POLICY"
    INVALID_CATALOG = "INVALID_CATALOG"
    PRIVACY_VIOLATION = "PRIVACY_VIOLATION"
    AUTHORITY_VIOLATION = "AUTHORITY_VIOLATION"
    PROHIBITED_MEDICAL_ACTION = "PROHIBITED_MEDICAL_ACTION"
    POLICY_PURPOSE_NOT_ALLOWED = "POLICY_PURPOSE_NOT_ALLOWED"
    BINDING_MISMATCH = "BINDING_MISMATCH"
    CONSENT_REQUIRED = "CONSENT_REQUIRED"
    CONSENT_MISMATCH = "CONSENT_MISMATCH"
    CONSENT_REVOKED = "CONSENT_REVOKED"
    CONSENT_DENIED = "CONSENT_DENIED"
    CONSENT_NOT_YET_VALID = "CONSENT_NOT_YET_VALID"
    CONSENT_EXPIRED = "CONSENT_EXPIRED"
    EVALUATOR_FAILURE = "EVALUATOR_FAILURE"


_ELIGIBLE_PURPOSES = frozenset(
    (HealthPurpose.GENERAL_EDUCATION, HealthPurpose.UNCERTAINTY_AND_REFERRAL)
)
_PROHIBITED_INPUTS = frozenset(
    (
        SyntheticInput.PROHIBITED_ACTION_REQUEST,
        SyntheticInput.ASSERTED_AUTHORITY,
        SyntheticInput.PRIVATE_DATA_CANARY,
    )
)
_REVIEW_REASONS = frozenset(
    (
        HealthReason.CANDIDATE_NOT_REVIEWED,
        HealthReason.UNSUPPORTED_CERTAINTY,
        HealthReason.CONTRADICTORY_EVIDENCE,
        HealthReason.MISSING_CLAIMS,
        HealthReason.MISSING_EVIDENCE,
        HealthReason.UNRESOLVED_EVIDENCE,
        HealthReason.STALE_EVIDENCE,
        HealthReason.INSUFFICIENT_INDEPENDENT_SUPPORT,
    )
)
_DENY_REASONS = frozenset(
    (
        HealthReason.MALFORMED_SCHEMA,
        HealthReason.UNSUPPORTED_SCHEMA,
        HealthReason.INVALID_CONTEXT,
        HealthReason.INVALID_TIMESTAMP,
        HealthReason.INVALID_POLICY,
        HealthReason.INVALID_CATALOG,
        HealthReason.PRIVACY_VIOLATION,
        HealthReason.AUTHORITY_VIOLATION,
        HealthReason.PROHIBITED_MEDICAL_ACTION,
        HealthReason.POLICY_PURPOSE_NOT_ALLOWED,
        HealthReason.BINDING_MISMATCH,
        HealthReason.CONSENT_REQUIRED,
        HealthReason.CONSENT_MISMATCH,
        HealthReason.CONSENT_REVOKED,
        HealthReason.CONSENT_DENIED,
        HealthReason.CONSENT_NOT_YET_VALID,
        HealthReason.CONSENT_EXPIRED,
        HealthReason.EVALUATOR_FAILURE,
    )
)
_SUPPORT_PAIRS = frozenset(
    (
        (SupportDisposition.ORDINARY_HELP, SupportTemplateId.ORDINARY_HELP_V1),
        (
            SupportDisposition.QUALIFIED_REVIEW,
            SupportTemplateId.QUALIFIED_REVIEW_V1,
        ),
        (
            SupportDisposition.IMMEDIATE_PUBLIC_REFERRAL,
            SupportTemplateId.IMMEDIATE_PUBLIC_REFERRAL_V1,
        ),
    )
)


class ContractError(ValueError):
    """Controlled structural error; its message never includes input values."""

    def __init__(self, reason: HealthReason):
        if type(reason) is not HealthReason or reason not in _DENY_REASONS:
            reason = HealthReason.MALFORMED_SCHEMA
        self.reason = reason
        super().__init__(reason.value)


def _require(condition: bool, reason: HealthReason = HealthReason.MALFORMED_SCHEMA) -> None:
    if not condition:
        raise ContractError(reason)


def _identifier(value: str, prefix: str) -> None:
    _require(
        type(value) is str
        and len(prefix) < len(value) <= MAX_IDENTIFIER_LENGTH
        and value.startswith(prefix)
        and _IDENTIFIER.fullmatch(value) is not None
    )


def _version(value: int) -> None:
    _require(type(value) is int and 1 <= value <= 2147483647)


def _schema(version: str) -> None:
    _require(type(version) is str)
    _require(version == SCHEMA_VERSION, HealthReason.UNSUPPORTED_SCHEMA)


def _utc(value: datetime) -> None:
    # Use exact built-in UTC datetimes. Do not call arbitrary tzinfo providers,
    # assume a timezone, consult a clock, or normalize ambiguous timestamps.
    _require(
        type(value) is datetime
        and value.tzinfo is timezone.utc
        and value.microsecond == 0
        and value.fold == 0,
        HealthReason.INVALID_TIMESTAMP,
    )


def _window(start: datetime, end: datetime) -> None:
    _utc(start)
    _utc(end)
    _require(start < end, HealthReason.INVALID_TIMESTAMP)


def _tuple_of(value: tuple, item_type: type, limit: int) -> None:
    _require(
        type(value) is tuple
        and len(value) <= limit
        and all(type(item) is item_type for item in value)
    )


def _claim_ids(values: Tuple[str, ...]) -> Tuple[str, ...]:
    _tuple_of(values, str, MAX_CLAIMS)
    for value in values:
        _identifier(value, "claim-")
    _require(len(set(values)) == len(values))
    return tuple(sorted(values))


_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum_tuple(
    values: Tuple[_EnumT, ...], enum_type: Type[_EnumT], limit: int
) -> Tuple[_EnumT, ...]:
    _tuple_of(values, enum_type, limit)
    _require(len(set(values)) == len(values))
    return tuple(sorted(values, key=lambda value: value.value))


def _wire_dict(value: object, *keys: str) -> dict:
    _require(
        type(value) is dict
        and len(value) == len(keys)
        and all(type(key) is str for key in value)
        and value.keys() == set(keys)
    )
    return cast(dict, value)


def _wire_list(value: object, limit: int) -> list:
    _require(type(value) is list and len(value) <= limit)
    return cast(list, value)


def _wire_enum(enum_type: Type[_EnumT], value: object) -> _EnumT:
    _require(type(value) is str and len(value) <= MAX_IDENTIFIER_LENGTH)
    # Avoid creating Enum's ordinary ValueError, which includes the raw value.
    for member in enum_type:
        if member.value == value:
            return member
    raise ContractError(HealthReason.MALFORMED_SCHEMA)


@dataclass(frozen=True)
class PolicyRef:
    policy_id: str = field(repr=False)
    version: int

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "policy-")
        _version(self.version)


@dataclass(frozen=True)
class CatalogRef:
    catalog_id: str = field(repr=False)
    version: int

    def __post_init__(self) -> None:
        _identifier(self.catalog_id, "catalog-")
        _version(self.version)


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reviewed artifact ID and SHA-256 of its exact UTF-8 bytes."""

    artifact_id: str = field(repr=False)
    sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, "artifact-")
        _require(
            type(self.sha256) is str
            and len(self.sha256) == 64
            and _SHA256.fullmatch(self.sha256) is not None
        )


@dataclass(frozen=True)
class FixtureScope:
    fixture_id: str = field(repr=False)
    turn_id: str = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.fixture_id, "fixture-")
        _identifier(self.turn_id, "turn-")


@dataclass(frozen=True)
class EvidenceRef:
    claim_id: str = field(repr=False)
    evidence_id: str = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.claim_id, "claim-")
        _identifier(self.evidence_id, "evidence-")


@dataclass(frozen=True)
class HealthPolicy:
    """Trusted, versioned synthetic policy; cannot opt into medical actions."""

    ref: PolicyRef
    catalog: CatalogRef
    allowed_purposes: Tuple[HealthPurpose, ...]
    min_independent_sources_per_claim: int

    def __post_init__(self) -> None:
        _require(
            type(self.ref) is PolicyRef and type(self.catalog) is CatalogRef,
            HealthReason.INVALID_POLICY,
        )
        try:
            purposes = _enum_tuple(self.allowed_purposes, HealthPurpose, 2)
        except ContractError:
            raise ContractError(HealthReason.INVALID_POLICY) from None
        _require(
            set(purposes).issubset(_ELIGIBLE_PURPOSES)
            and type(self.min_independent_sources_per_claim) is int
            and 2 <= self.min_independent_sources_per_claim <= MAX_INDEPENDENT_SOURCES,
            HealthReason.INVALID_POLICY,
        )
        object.__setattr__(self, "allowed_purposes", purposes)


@dataclass(frozen=True)
class HealthRequest:
    """All fields are untrusted declarations. No prose or authority fields."""

    schema_version: str = field(repr=False)
    scope: FixtureScope = field(repr=False)
    catalog: CatalogRef = field(repr=False)
    artifact: ArtifactRef = field(repr=False)
    purpose: HealthPurpose = field(repr=False)
    claim_ids: Tuple[str, ...] = field(repr=False)
    evidence_refs: Tuple[EvidenceRef, ...] = field(repr=False)
    synthetic_inputs: Tuple[SyntheticInput, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _require(
            type(self.scope) is FixtureScope
            and type(self.catalog) is CatalogRef
            and type(self.artifact) is ArtifactRef
            and type(self.purpose) is HealthPurpose
        )
        object.__setattr__(self, "claim_ids", _claim_ids(self.claim_ids))
        _tuple_of(self.evidence_refs, EvidenceRef, MAX_EVIDENCE_REFS)
        # Preserve duplicate evidence references for future deduplication checks.
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(
                sorted(self.evidence_refs, key=lambda ref: (ref.claim_id, ref.evidence_id))
            ),
        )
        object.__setattr__(
            self,
            "synthetic_inputs",
            _enum_tuple(self.synthetic_inputs, SyntheticInput, MAX_INPUT_MARKERS),
        )

    @classmethod
    def from_dict(cls, value: object) -> HealthRequest:
        """Decode only the closed, already-parsed synthetic request schema.

        This is structural validation, not trust verification. No context or
        catalog contents can be supplied here. A future ingress must bound raw
        bytes and reject duplicate JSON keys before decoding a dict.
        """
        data = _wire_dict(
            value,
            "schema_version",
            "scope",
            "catalog",
            "artifact",
            "purpose",
            "claim_ids",
            "evidence_refs",
            "synthetic_inputs",
        )
        _schema(data["schema_version"])
        return cls(
            schema_version=data["schema_version"],
            scope=FixtureScope(**_wire_dict(data["scope"], "fixture_id", "turn_id")),
            catalog=CatalogRef(**_wire_dict(data["catalog"], "catalog_id", "version")),
            artifact=ArtifactRef(**_wire_dict(data["artifact"], "artifact_id", "sha256")),
            purpose=_wire_enum(HealthPurpose, data["purpose"]),
            claim_ids=tuple(_wire_list(data["claim_ids"], MAX_CLAIMS)),
            evidence_refs=tuple(
                EvidenceRef(**_wire_dict(ref, "claim_id", "evidence_id"))
                for ref in _wire_list(data["evidence_refs"], MAX_EVIDENCE_REFS)
            ),
            synthetic_inputs=tuple(
                _wire_enum(SyntheticInput, marker)
                for marker in _wire_list(data["synthetic_inputs"], MAX_INPUT_MARKERS)
            ),
        )


@dataclass(frozen=True)
class SyntheticConsent:
    """Harness-issued synthetic fact; never constructed from request claims."""

    scope: FixtureScope
    catalog: CatalogRef
    artifact: ArtifactRef
    purpose: HealthPurpose
    state: ConsentState
    valid_from: datetime
    valid_until: datetime

    def __post_init__(self) -> None:
        _require(
            type(self.scope) is FixtureScope
            and type(self.catalog) is CatalogRef
            and type(self.artifact) is ArtifactRef
            and type(self.purpose) is HealthPurpose
            and type(self.state) is ConsentState
        )
        _window(self.valid_from, self.valid_until)


@dataclass(frozen=True)
class UrgencyFact:
    """Separately established synthetic urgency, scoped to exactly one turn."""

    scope: FixtureScope
    level: UrgencyLevel

    def __post_init__(self) -> None:
        _require(type(self.scope) is FixtureScope and type(self.level) is UrgencyLevel)


@dataclass(frozen=True)
class HealthContext:
    """Trusted harness snapshot, including safe representations of failure.

    INVALID preserves independently checked support facts when another context
    component cannot be assembled. It will never permit candidate release.
    """

    scope: Optional[FixtureScope]
    evaluation_time: Optional[datetime]
    consent: Optional[SyntheticConsent]
    urgency: Optional[UrgencyFact]
    state: ContextState

    def __post_init__(self) -> None:
        _require(
            type(self.state) is ContextState
            and (
                type(self.scope) is FixtureScope
                or (self.scope is None and self.state is ContextState.INVALID)
            )
            and (self.consent is None or type(self.consent) is SyntheticConsent)
            and (self.urgency is None or type(self.urgency) is UrgencyFact),
            HealthReason.INVALID_CONTEXT,
        )
        if self.evaluation_time is not None:
            _utc(self.evaluation_time)


@dataclass(frozen=True)
class ReviewedArtifact:
    """Catalog annotation of an exact synthetic artifact, not clinical review."""

    artifact: ArtifactRef
    scope: FixtureScope
    purpose: HealthPurpose
    assessment: ArtifactAssessment
    claim_ids: Tuple[str, ...] = field(repr=False)
    synthetic_inputs: Tuple[SyntheticInput, ...]

    def __post_init__(self) -> None:
        _require(
            type(self.artifact) is ArtifactRef
            and type(self.scope) is FixtureScope
            and type(self.purpose) is HealthPurpose
            and type(self.assessment) is ArtifactAssessment,
            HealthReason.INVALID_CATALOG,
        )
        claims = _claim_ids(self.claim_ids)
        markers = _enum_tuple(self.synthetic_inputs, SyntheticInput, MAX_INPUT_MARKERS)
        if self.assessment is ArtifactAssessment.ELIGIBLE:
            _require(
                self.purpose in _ELIGIBLE_PURPOSES
                and bool(claims)
                and not _PROHIBITED_INPUTS.intersection(markers),
                HealthReason.INVALID_CATALOG,
            )
        object.__setattr__(self, "claim_ids", claims)
        object.__setattr__(self, "synthetic_inputs", markers)


@dataclass(frozen=True)
class ReviewedEvidence:
    """An expressly synthetic annotation bound to an artifact, claim and turn."""

    evidence_id: str = field(repr=False)
    artifact: ArtifactRef
    scope: FixtureScope
    purpose: HealthPurpose
    claim_id: str = field(repr=False)
    source_group_id: str = field(repr=False)
    relation: EvidenceRelation
    valid_from: datetime
    valid_until: datetime

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence-")
        _identifier(self.claim_id, "claim-")
        _identifier(self.source_group_id, "source-")
        _require(
            type(self.artifact) is ArtifactRef
            and type(self.scope) is FixtureScope
            and type(self.purpose) is HealthPurpose
            and type(self.relation) is EvidenceRelation,
            HealthReason.INVALID_CATALOG,
        )
        _window(self.valid_from, self.valid_until)


@dataclass(frozen=True)
class HealthCatalog:
    """Trusted versioned manifest; all rows inherit its exact CatalogRef."""

    schema_version: str
    ref: CatalogRef
    review_id: str = field(repr=False)
    review_stage: CatalogReviewStage
    artifacts: Tuple[ReviewedArtifact, ...]
    evidence: Tuple[ReviewedEvidence, ...]

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _identifier(self.review_id, "review-")
        _require(
            type(self.ref) is CatalogRef
            and type(self.review_stage) is CatalogReviewStage,
            HealthReason.INVALID_CATALOG,
        )
        try:
            _tuple_of(self.artifacts, ReviewedArtifact, MAX_CATALOG_ARTIFACTS)
            _tuple_of(self.evidence, ReviewedEvidence, MAX_CATALOG_EVIDENCE)
        except ContractError:
            raise ContractError(HealthReason.INVALID_CATALOG) from None
        artifacts = {entry.artifact.artifact_id: entry for entry in self.artifacts}
        _require(
            bool(artifacts)
            and len(artifacts) == len(self.artifacts)
            and len({entry.evidence_id for entry in self.evidence}) == len(self.evidence),
            HealthReason.INVALID_CATALOG,
        )
        counts = {}
        for entry in self.evidence:
            artifact = artifacts.get(entry.artifact.artifact_id)
            _require(
                artifact is not None
                and entry.artifact == artifact.artifact
                and entry.scope == artifact.scope
                and entry.purpose == artifact.purpose
                and entry.claim_id in artifact.claim_ids,
                HealthReason.INVALID_CATALOG,
            )
            artifact_id = entry.artifact.artifact_id
            counts[artifact_id] = counts.get(artifact_id, 0) + 1
            _require(
                counts[artifact_id] <= MAX_EVIDENCE_REFS, HealthReason.INVALID_CATALOG
            )
        object.__setattr__(
            self,
            "artifacts",
            tuple(sorted(self.artifacts, key=lambda row: row.artifact.artifact_id)),
        )
        object.__setattr__(
            self, "evidence", tuple(sorted(self.evidence, key=lambda row: row.evidence_id))
        )


def _result_metadata(
    outcome: CandidateDisposition,
    reason_codes: Tuple[HealthReason, ...],
    support: SupportDisposition,
    support_template_id: SupportTemplateId,
    policy: Optional[PolicyRef],
    catalog: Optional[CatalogRef],
    scope: Optional[FixtureScope],
    evaluation_time: Optional[datetime],
) -> None:
    _require(
        type(outcome) is CandidateDisposition
        and type(support) is SupportDisposition
        and type(support_template_id) is SupportTemplateId
        and (support, support_template_id) in _SUPPORT_PAIRS
        and (policy is None or type(policy) is PolicyRef)
        and (catalog is None or type(catalog) is CatalogRef)
        and (scope is None or type(scope) is FixtureScope)
    )
    _tuple_of(reason_codes, HealthReason, 1)
    _require(len(reason_codes) == 1)
    reason = reason_codes[0]
    _require(
        (
            outcome is CandidateDisposition.ALLOW
            and reason is HealthReason.SYNTHETIC_ARTIFACT_ELIGIBLE
        )
        or (outcome is CandidateDisposition.REVIEW and reason in _REVIEW_REASONS)
        or (outcome is CandidateDisposition.DENY and reason in _DENY_REASONS)
    )
    if evaluation_time is not None:
        _utc(evaluation_time)
    if support is not SupportDisposition.ORDINARY_HELP:
        _require(scope is not None)
    if outcome is not CandidateDisposition.DENY:
        _require(
            policy is not None
            and catalog is not None
            and scope is not None
            and evaluation_time is not None
        )


@dataclass(frozen=True)
class HealthAuditRecord:
    """Controlled, in-memory metadata; no request, evidence or artifact text."""

    outcome: CandidateDisposition
    reason_codes: Tuple[HealthReason, ...]
    support: SupportDisposition
    support_template_id: SupportTemplateId
    policy: Optional[PolicyRef]
    catalog: Optional[CatalogRef]
    scope: Optional[FixtureScope]
    evaluation_time: Optional[datetime]

    def __post_init__(self) -> None:
        _result_metadata(
            self.outcome,
            self.reason_codes,
            self.support,
            self.support_template_id,
            self.policy,
            self.catalog,
            self.scope,
            self.evaluation_time,
        )


@dataclass(frozen=True)
class HealthDecision:
    """Result shape only; even an ALLOW instance is not an authorization token."""

    outcome: CandidateDisposition
    reason_codes: Tuple[HealthReason, ...]
    permitted_artifact: Optional[ArtifactRef]
    support: SupportDisposition
    support_template_id: SupportTemplateId
    policy: Optional[PolicyRef]
    catalog: Optional[CatalogRef]
    scope: Optional[FixtureScope]
    evaluation_time: Optional[datetime]

    def __post_init__(self) -> None:
        _result_metadata(
            self.outcome,
            self.reason_codes,
            self.support,
            self.support_template_id,
            self.policy,
            self.catalog,
            self.scope,
            self.evaluation_time,
        )
        _require(
            (
                self.outcome is CandidateDisposition.ALLOW
                and type(self.permitted_artifact) is ArtifactRef
            )
            or (
                self.outcome is not CandidateDisposition.ALLOW
                and self.permitted_artifact is None
            )
        )

    @property
    def audit(self) -> HealthAuditRecord:
        # A projection of controlled result metadata, not a log or history.
        return HealthAuditRecord(
            self.outcome,
            self.reason_codes,
            self.support,
            self.support_template_id,
            self.policy,
            self.catalog,
            self.scope,
            self.evaluation_time,
        )


_RecordT = TypeVar("_RecordT")


def _leaf_snapshot(value: object, record_type: Type[_RecordT]) -> _RecordT:
    """Revalidate and copy an exact, leaf-level contract without coercion."""
    _require(type(value) is record_type)
    try:
        return cast(_RecordT, replace(value))
    except AttributeError:
        raise ContractError(HealthReason.MALFORMED_SCHEMA) from None


def _request_snapshot(value: object) -> HealthRequest:
    if type(value) is dict:
        return HealthRequest.from_dict(value)
    _require(type(value) is HealthRequest)
    # A typed declaration remains untrusted. Validate nested records before
    # the constructor sorts or compares any of their fields.
    _schema(value.schema_version)
    _tuple_of(value.evidence_refs, EvidenceRef, MAX_EVIDENCE_REFS)
    return replace(
        value,
        scope=_leaf_snapshot(value.scope, FixtureScope),
        catalog=_leaf_snapshot(value.catalog, CatalogRef),
        artifact=_leaf_snapshot(value.artifact, ArtifactRef),
        evidence_refs=tuple(_leaf_snapshot(ref, EvidenceRef) for ref in value.evidence_refs),
    )


def _policy_snapshot(value: object) -> HealthPolicy:
    _require(type(value) is HealthPolicy, HealthReason.INVALID_POLICY)
    return replace(
        value,
        ref=_leaf_snapshot(value.ref, PolicyRef),
        catalog=_leaf_snapshot(value.catalog, CatalogRef),
    )


def _catalog_snapshot(value: object) -> HealthCatalog:
    _require(type(value) is HealthCatalog, HealthReason.INVALID_CATALOG)
    _tuple_of(value.artifacts, ReviewedArtifact, MAX_CATALOG_ARTIFACTS)
    _tuple_of(value.evidence, ReviewedEvidence, MAX_CATALOG_EVIDENCE)
    # Validate every nested record before checking the aggregate manifest.
    # The governor has its own immutable catalog, independent of caller data.
    return replace(
        value,
        ref=_leaf_snapshot(value.ref, CatalogRef),
        artifacts=tuple(
            replace(row, artifact=_leaf_snapshot(row.artifact, ArtifactRef),
                    scope=_leaf_snapshot(row.scope, FixtureScope))
            for row in value.artifacts
        ),
        evidence=tuple(
            replace(row, artifact=_leaf_snapshot(row.artifact, ArtifactRef),
                    scope=_leaf_snapshot(row.scope, FixtureScope))
            for row in value.evidence
        ),
    )


def _consent_snapshot(value: object) -> SyntheticConsent:
    _require(type(value) is SyntheticConsent, HealthReason.INVALID_CONTEXT)
    return replace(
        value,
        scope=_leaf_snapshot(value.scope, FixtureScope),
        catalog=_leaf_snapshot(value.catalog, CatalogRef),
        artifact=_leaf_snapshot(value.artifact, ArtifactRef),
    )


@dataclass(frozen=True, init=False)
class HealthSafetyGovernor:
    """Pure categorical evaluator of reviewed, offline synthetic artifacts.

    Policy, catalog and context must be independently assembled by the trusted
    harness. Their types do not authenticate their provenance. No result is a
    clinical judgment, a transferable credential, or a release operation.
    """

    _policy: Optional[HealthPolicy] = field(repr=False)
    _catalog: Optional[HealthCatalog] = field(repr=False)
    _policy_error: Optional[HealthReason] = field(repr=False)
    _catalog_error: Optional[HealthReason] = field(repr=False)

    def __init__(self, *, policy: Optional[HealthPolicy], catalog: Optional[HealthCatalog]) -> None:
        checked_policy = None
        checked_catalog = None
        policy_error = None
        catalog_error = None
        try:
            checked_policy = _policy_snapshot(policy)
        except (ContractError, AttributeError):
            policy_error = HealthReason.INVALID_POLICY
        except Exception:
            policy_error = HealthReason.EVALUATOR_FAILURE
        try:
            checked_catalog = _catalog_snapshot(catalog)
            if checked_policy is not None and checked_catalog.ref != checked_policy.catalog:
                checked_catalog = None
                catalog_error = HealthReason.INVALID_CATALOG
        except (ContractError, AttributeError):
            catalog_error = HealthReason.INVALID_CATALOG
        except Exception:
            catalog_error = HealthReason.EVALUATOR_FAILURE
        object.__setattr__(self, "_policy", checked_policy)
        object.__setattr__(self, "_catalog", checked_catalog)
        object.__setattr__(self, "_policy_error", policy_error)
        object.__setattr__(self, "_catalog_error", catalog_error)

    def evaluate(self, request: Union[HealthRequest, dict], *, context: HealthContext) -> HealthDecision:
        # Independently validated audit/support metadata is never derived from
        # request fields. Capture it before inspecting the candidate, so a
        # candidate-side exception cannot discard an established referral.
        scope = None
        now = None
        urgency = None
        consent = None
        state = None
        support = SupportDisposition.ORDINARY_HELP
        template = SupportTemplateId.ORDINARY_HELP_V1
        policy_ref = None if self._policy is None else self._policy.ref
        catalog_ref = None if self._catalog is None else self._catalog.ref

        def result(reason: HealthReason, artifact: Optional[ArtifactRef] = None) -> HealthDecision:
            if reason is HealthReason.SYNTHETIC_ARTIFACT_ELIGIBLE:
                outcome = CandidateDisposition.ALLOW
            elif reason in _REVIEW_REASONS:
                outcome = CandidateDisposition.REVIEW
            else:
                outcome = CandidateDisposition.DENY
            return HealthDecision(
                outcome=outcome,
                reason_codes=(reason,),
                permitted_artifact=artifact if outcome is CandidateDisposition.ALLOW else None,
                support=support,
                support_template_id=template,
                policy=policy_ref,
                catalog=catalog_ref,
                scope=scope,
                evaluation_time=now,
            )

        try:
            invalid_context = type(context) is not HealthContext
            if not invalid_context:
                # Scope and urgency are validated separately from consent,
                # time, context state and candidate disposition.
                try:
                    if context.scope is not None:
                        scope = _leaf_snapshot(context.scope, FixtureScope)
                except (ContractError, AttributeError):
                    invalid_context = True
                try:
                    if context.urgency is not None:
                        _require(type(context.urgency) is UrgencyFact, HealthReason.INVALID_CONTEXT)
                        urgency = replace(
                            context.urgency, scope=_leaf_snapshot(context.urgency.scope, FixtureScope)
                        )
                except (ContractError, AttributeError):
                    invalid_context = True
                if scope is not None and urgency is not None and urgency.scope == scope:
                    if urgency.level is UrgencyLevel.IMMEDIATE_PUBLIC_REFERRAL:
                        support = SupportDisposition.IMMEDIATE_PUBLIC_REFERRAL
                        template = SupportTemplateId.IMMEDIATE_PUBLIC_REFERRAL_V1
                    elif urgency.level is UrgencyLevel.QUALIFIED_REVIEW:
                        support = SupportDisposition.QUALIFIED_REVIEW
                        template = SupportTemplateId.QUALIFIED_REVIEW_V1

                # Unavailable/invalid evaluation time is explicitly absent in
                # the audit; never substitute a clock or request timestamp.
                try:
                    if context.evaluation_time is not None:
                        _utc(context.evaluation_time)
                        now = context.evaluation_time
                except (ContractError, AttributeError):
                    pass
                try:
                    state = context.state
                    _require(type(state) is ContextState, HealthReason.INVALID_CONTEXT)
                    if context.consent is not None:
                        consent = _consent_snapshot(context.consent)
                except (ContractError, AttributeError):
                    invalid_context = True

            # Structural request rejection precedes context/configuration
            # rejection, even though support was captured independently first.
            try:
                candidate = _request_snapshot(request)
            except ContractError as error:
                return result(error.reason)
            except AttributeError:
                return result(HealthReason.MALFORMED_SCHEMA)
            if invalid_context or state is not ContextState.ASSEMBLED or scope is None:
                return result(HealthReason.INVALID_CONTEXT)
            if now is None:
                return result(HealthReason.INVALID_TIMESTAMP)
            if self._policy_error is not None or self._policy is None:
                return result(self._policy_error or HealthReason.INVALID_POLICY)
            if self._catalog_error is not None or self._catalog is None:
                return result(self._catalog_error or HealthReason.INVALID_CATALOG)

            checked_context = HealthContext(scope, now, consent, urgency, state)
            reason, artifact = self._candidate_reason(
                candidate, checked_context, self._policy, self._catalog
            )
            return result(reason, artifact)
        except Exception:
            # No exception text, raw request values or partially authorized
            # artifacts escape. The independently captured support survives.
            return result(HealthReason.EVALUATOR_FAILURE)

    @staticmethod
    def _candidate_reason(
        request: HealthRequest, context: HealthContext, policy: HealthPolicy,
        catalog: HealthCatalog,
    ) -> Tuple[HealthReason, Optional[ArtifactRef]]:
        """Ordered gates from the frozen Phase 1 specification."""
        reason = HealthReason
        manifest = next((row for row in catalog.artifacts
                         if row.artifact.artifact_id == request.artifact.artifact_id), None)
        assessment = None if manifest is None else manifest.assessment
        markers = set(request.synthetic_inputs)
        if manifest is not None:
            markers.update(manifest.synthetic_inputs)

        if (SyntheticInput.PRIVATE_DATA_CANARY in markers
                or assessment is ArtifactAssessment.PROHIBITED_PRIVACY):
            return reason.PRIVACY_VIOLATION, None
        if (SyntheticInput.ASSERTED_AUTHORITY in markers
                or assessment is ArtifactAssessment.PROHIBITED_AUTHORITY):
            return reason.AUTHORITY_VIOLATION, None
        if (request.purpose not in _ELIGIBLE_PURPOSES
                or (manifest is not None and manifest.purpose not in _ELIGIBLE_PURPOSES)
                or SyntheticInput.PROHIBITED_ACTION_REQUEST in markers
                or assessment is ArtifactAssessment.PROHIBITED_MEDICAL_ACTION):
            return reason.PROHIBITED_MEDICAL_ACTION, None

        if (request.scope != context.scope or request.catalog != policy.catalog
                or request.catalog != catalog.ref
                or (context.urgency is not None and context.urgency.scope != context.scope)):
            return reason.BINDING_MISMATCH, None
        declared_claims = set(request.claim_ids)
        if manifest is not None and (
            request.artifact != manifest.artifact or request.scope != manifest.scope
            or request.purpose is not manifest.purpose
            or request.synthetic_inputs != manifest.synthetic_inputs
            or not declared_claims.issubset(manifest.claim_ids)
        ):
            return reason.BINDING_MISMATCH, None

        evidence_by_id = {row.evidence_id: row for row in catalog.evidence}
        unknown_evidence = False
        for ref in request.evidence_refs:
            if ref.claim_id not in declared_claims:
                return reason.BINDING_MISMATCH, None
            row = evidence_by_id.get(ref.evidence_id)
            if row is None:
                unknown_evidence = True
            elif (row.claim_id != ref.claim_id or row.artifact != request.artifact
                    or row.scope != request.scope or row.purpose is not request.purpose):
                return reason.BINDING_MISMATCH, None

        if request.purpose not in policy.allowed_purposes:
            return reason.POLICY_PURPOSE_NOT_ALLOWED, None
        consent = context.consent
        now = context.evaluation_time
        if consent is None:
            return reason.CONSENT_REQUIRED, None
        if (consent.scope != request.scope or consent.catalog != request.catalog
                or consent.artifact != request.artifact or consent.purpose is not request.purpose):
            return reason.CONSENT_MISMATCH, None
        if consent.state is ConsentState.REVOKED:
            return reason.CONSENT_REVOKED, None
        if consent.state is ConsentState.DENIED:
            return reason.CONSENT_DENIED, None
        if consent.state is not ConsentState.GRANTED or now is None:
            return reason.INVALID_CONTEXT, None
        if now < consent.valid_from:
            return reason.CONSENT_NOT_YET_VALID, None
        if now >= consent.valid_until:
            return reason.CONSENT_EXPIRED, None

        # No review gate is reached until every denial gate has cleared.
        if manifest is None or assessment is ArtifactAssessment.UNRESOLVED:
            return reason.CANDIDATE_NOT_REVIEWED, None
        if assessment is ArtifactAssessment.UNSUPPORTED_CERTAINTY:
            return reason.UNSUPPORTED_CERTAINTY, None

        # Inspect the complete trusted matching manifest, not only the rows
        # the untrusted request chose to cite. Contradictions survive omission
        # and expiry; evidence IDs and group IDs have no authority by presence.
        rows = tuple(row for row in catalog.evidence if row.artifact == manifest.artifact)
        if any(row.relation is EvidenceRelation.CONTRADICTS for row in rows):
            return reason.CONTRADICTORY_EVIDENCE, None
        if declared_claims != set(manifest.claim_ids):
            return reason.MISSING_CLAIMS, None
        referenced = {ref.evidence_id for ref in request.evidence_refs}
        if (not {row.evidence_id for row in rows}.issubset(referenced)
                or not declared_claims.issubset({row.claim_id for row in rows})):
            return reason.MISSING_EVIDENCE, None
        if unknown_evidence or any(row.relation is EvidenceRelation.UNRESOLVED for row in rows):
            return reason.UNRESOLVED_EVIDENCE, None
        if any(not row.valid_from <= now < row.valid_until for row in rows):
            return reason.STALE_EVIDENCE, None
        for claim in manifest.claim_ids:
            groups = {row.source_group_id for row in rows
                      if row.claim_id == claim and row.relation is EvidenceRelation.SUPPORTS}
            if len(groups) < policy.min_independent_sources_per_claim:
                return reason.INSUFFICIENT_INDEPENDENT_SUPPORT, None

        if assessment is not ArtifactAssessment.ELIGIBLE:
            return reason.CANDIDATE_NOT_REVIEWED, None
        return reason.SYNTHETIC_ARTIFACT_ELIGIBLE, manifest.artifact

    def verify_synthetic_artifact(
        self, request: Union[HealthRequest, dict], *, context: HealthContext,
        artifact_id: str, payload: bytes,
    ) -> bool:
        """Check exact synthetic fixture bytes after a fresh evaluation only.

        This check returns no bytes or release handle and accepts no decision
        credential. A trusted harness still owns the immutable payload store,
        provenance, current-turn assembly and any future release consumer.
        """
        try:
            _identifier(artifact_id, "artifact-")
            if type(payload) is not bytes or not 0 < len(payload) <= MAX_SYNTHETIC_ARTIFACT_BYTES:
                return False
            decision = self.evaluate(request, context=context)
            artifact = decision.permitted_artifact
            return (
                decision.outcome is CandidateDisposition.ALLOW
                and artifact is not None
                and self._catalog is not None
                and any(artifact is row.artifact for row in self._catalog.artifacts)
                and artifact.artifact_id == artifact_id
                and hashlib.sha256(payload).hexdigest() == artifact.sha256
            )
        except Exception:
            return False
