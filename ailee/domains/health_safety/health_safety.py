# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Phase 1: offline synthetic Health Safety contracts, not an evaluator.

Constructing any of these values establishes structure only, not trust, clinical
correctness, consent, or release authority. See README.md for the future API
and the independent harness boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple, Type, TypeVar, cast

SCHEMA_VERSION = "1.0"
MAX_IDENTIFIER_LENGTH = 64
MAX_CLAIMS = 16
MAX_EVIDENCE_REFS = 64
MAX_INPUT_MARKERS = 8
MAX_CATALOG_ARTIFACTS = 128
MAX_CATALOG_EVIDENCE = 512
MAX_INDEPENDENT_SOURCES = 8

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
