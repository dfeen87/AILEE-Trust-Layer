# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Deterministic synthetic-metadata readiness gate; not a scientific decision model."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
from typing import Any, Dict, Optional, Tuple

from .contracts import (
    CatalogEntry, CatalogRef, ContractError, EvidenceCatalog, EvidenceRecord,
    EvidenceRequest, Observation, Outcome, Reason, ReviewPolicy, ReviewStatus,
    Scope, MAX_CATALOG_ENTRIES, MAX_RECORDS, _digest, _format_time, _require,
    validate_time,
)


def _scope(value: Any) -> Scope:
    _require(type(value) is Scope)
    return Scope(value.material_id, value.endpoint_id, value.method_id,
                 value.protocol_version, value.unit_id, value.context_id)


def _reference(value: Any) -> CatalogRef:
    _require(type(value) is CatalogRef)
    return CatalogRef(value.catalog_id, value.version)


def _record(value: Any) -> EvidenceRecord:
    _require(type(value) is EvidenceRecord)
    return EvidenceRecord(value.evidence_id, _scope(value.scope), value.observed_at,
                          value.observation, value.artifact_sha256)


def _request(value: Any) -> EvidenceRequest:
    if type(value) is dict:
        return EvidenceRequest.from_dict(value)
    if type(value) is bytes:
        return EvidenceRequest.from_json(value)
    _require(type(value) is EvidenceRequest)
    _require(type(value.records) is tuple and len(value.records) <= MAX_RECORDS)
    return EvidenceRequest(value.schema_version, _reference(value.catalog), _scope(value.scope),
                           tuple(_record(r) for r in value.records))


def _entry(value: Any) -> CatalogEntry:
    _require(type(value) is CatalogEntry)
    return CatalogEntry(value.evidence_id, _scope(value.scope), value.metadata_sha256,
                        value.source_group, value.reviewed_at, value.reporting,
                        value.positive_control, value.negative_control, value.applicability)


def _catalog(value: Any) -> EvidenceCatalog:
    _require(type(value) is EvidenceCatalog and type(value.entries) is tuple
             and len(value.entries) <= MAX_CATALOG_ENTRIES)
    return EvidenceCatalog(_reference(value.reference), tuple(_entry(e) for e in value.entries))


def _policy(value: Any) -> ReviewPolicy:
    _require(type(value) is ReviewPolicy and type(value.allowed_scopes) is tuple
             and len(value.allowed_scopes) <= 32)
    return ReviewPolicy(value.policy_id, value.version, _reference(value.catalog),
                        tuple(_scope(s) for s in value.allowed_scopes), value.min_source_groups,
                        value.max_age_days)


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _policy_hash(policy: ReviewPolicy) -> str:
    return _hash({"policy_id": policy.policy_id, "version": policy.version,
                  "catalog": policy.catalog.to_dict(),
                  "allowed_scopes": [s.to_dict() for s in policy.allowed_scopes],
                  "min_source_groups": policy.min_source_groups, "max_age_days": policy.max_age_days})


def _catalog_hash(catalog: EvidenceCatalog) -> str:
    return _hash({"reference": catalog.reference.to_dict(), "entries": [
        {"evidence_id": e.evidence_id, "scope": e.scope.to_dict(),
         "metadata_sha256": e.metadata_sha256, "source_group": e.source_group,
         "reviewed_at": _format_time(e.reviewed_at), "reporting": e.reporting.value,
         "positive_control": e.positive_control.value, "negative_control": e.negative_control.value,
         "applicability": e.applicability.value} for e in catalog.entries]})


@dataclass(frozen=True)
class Decision:
    """Engineering outcome; never scientific validity or authorization."""

    outcome: Outcome
    reason_codes: Tuple[Reason, ...]
    evaluation_time: Optional[datetime]
    policy_sha256: Optional[str]
    catalog_sha256: Optional[str]
    bound_request_sha256: Optional[str]

    def __post_init__(self) -> None:
        _require(type(self.outcome) is Outcome and type(self.reason_codes) is tuple
                 and bool(self.reason_codes) and all(type(r) is Reason for r in self.reason_codes))
        _require(tuple(sorted(set(self.reason_codes), key=lambda r: r.value)) == self.reason_codes)
        if self.evaluation_time is not None:
            validate_time(self.evaluation_time)
        for digest in (self.policy_sha256, self.catalog_sha256, self.bound_request_sha256):
            if digest is not None:
                _digest(digest)
        _require((self.outcome is Outcome.REVIEW_READY) == (self.reason_codes == (Reason.METADATA_COMPLETE,)))
        invalid_reasons = {Reason.INVALID_CONFIGURATION, Reason.INVALID_TIME, Reason.MALFORMED_REQUEST}
        if self.outcome is Outcome.INVALID:
            _require(len(self.reason_codes) == 1 and self.reason_codes[0] in invalid_reasons
                     and self.bound_request_sha256 is None)
        else:
            _require(not invalid_reasons.intersection(self.reason_codes))
            if self.outcome is Outcome.ABSTAIN:
                _require(Reason.METADATA_COMPLETE not in self.reason_codes)
        if self.outcome is Outcome.REVIEW_READY:
            _require(self.evaluation_time is not None and self.policy_sha256 is not None
                     and self.catalog_sha256 is not None and self.bound_request_sha256 is not None)

    @property
    def audit(self) -> Dict[str, Any]:
        """Detached minimized projection; no request fields, evidence or artifact hashes."""
        return {"outcome": self.outcome.value, "reason_codes": [r.value for r in self.reason_codes],
                "evaluation_time": None if self.evaluation_time is None else _format_time(self.evaluation_time),
                "policy_sha256": self.policy_sha256, "catalog_sha256": self.catalog_sha256,
                "bound_request_sha256": self.bound_request_sha256}


@dataclass(frozen=True, init=False)
class EvidenceGovernor:
    """An offline gate configured only with independently assembled trusted snapshots.

    Snapshot construction/type checks do not authenticate the snapshot's origin.
    Arbitrary code execution in this same Python process is not a security boundary.
    """

    _policy: Optional[ReviewPolicy]
    _catalog: Optional[EvidenceCatalog]

    def __init__(self, *, policy: Any, catalog: Any) -> None:
        validated_policy: Optional[ReviewPolicy]
        validated_catalog: Optional[EvidenceCatalog]
        try:
            validated_policy = _policy(policy)
            validated_catalog = _catalog(catalog)
            _require(validated_policy.catalog == validated_catalog.reference)
        except (ContractError, AttributeError, TypeError):
            validated_policy = None
            validated_catalog = None
        object.__setattr__(self, "_policy", validated_policy)
        object.__setattr__(self, "_catalog", validated_catalog)

    def evaluate(self, request: Any, *, evaluation_time: Any) -> Decision:
        """Evaluate explicit UTC time and typed request, exact dict, or bounded JSON bytes."""
        try:
            validate_time(evaluation_time)
            clock = evaluation_time
        except ContractError:
            clock = None
        try:
            policy = _policy(self._policy)
            catalog = _catalog(self._catalog)
            _require(policy.catalog == catalog.reference)
            policy_digest, catalog_digest = _policy_hash(policy), _catalog_hash(catalog)
        except (ContractError, AttributeError, TypeError):
            return Decision(Outcome.INVALID, (Reason.INVALID_CONFIGURATION,), clock, None, None, None)

        bound_request_digest: Optional[str] = None

        def decision(outcome: Outcome, *reasons: Reason) -> Decision:
            return Decision(outcome, tuple(sorted(set(reasons), key=lambda r: r.value)),
                            clock, policy_digest, catalog_digest, bound_request_digest)

        if clock is None:
            return decision(Outcome.INVALID, Reason.INVALID_TIME)
        try:
            supplied = _request(request)
        except (ContractError, AttributeError, TypeError):
            return decision(Outcome.INVALID, Reason.MALFORMED_REQUEST)

        if supplied.catalog != catalog.reference:
            return decision(Outcome.ABSTAIN, Reason.CATALOG_MISMATCH)
        if supplied.scope not in policy.allowed_scopes:
            return decision(Outcome.ABSTAIN, Reason.SCOPE_NOT_ALLOWED)
        if any(r.scope != supplied.scope for r in supplied.records):
            return decision(Outcome.ABSTAIN, Reason.SCOPE_MISMATCH)
        if not supplied.records:
            return decision(Outcome.ABSTAIN, Reason.MISSING_EVIDENCE)

        roster = {e.evidence_id: e for e in catalog.entries if e.scope == supplied.scope}
        ids = {r.evidence_id for r in supplied.records}
        if not ids.issubset(roster):
            return decision(Outcome.ABSTAIN, Reason.UNREVIEWED_EVIDENCE)
        if ids != set(roster):
            return decision(Outcome.ABSTAIN, Reason.INCOMPLETE_CATALOG_SCOPE)
        if any(r.metadata_sha256 != roster[r.evidence_id].metadata_sha256 for r in supplied.records):
            return decision(Outcome.ABSTAIN, Reason.BINDING_MISMATCH)
        # Only fingerprint catalog-bound metadata; never hash unbound or malformed requests.
        bound_request_digest = _hash(supplied.to_dict())

        # Only compare/categorize observations after exact scope, roster and digest checks.
        blockers = set()
        for record in supplied.records:
            entry = roster[record.evidence_id]
            if not (record.observed_at <= entry.reviewed_at <= clock) or (
                    clock - record.observed_at > timedelta(days=policy.max_age_days)):
                blockers.add(Reason.STALE_OR_FUTURE)
            if entry.reporting is not ReviewStatus.PASS:
                blockers.add(Reason.REPORTING_NOT_PASSED)
            if (entry.positive_control is not ReviewStatus.PASS or
                    entry.negative_control is not ReviewStatus.PASS):
                blockers.add(Reason.CONTROLS_NOT_PASSED)
            if entry.applicability is not ReviewStatus.PASS:
                blockers.add(Reason.APPLICABILITY_NOT_PASSED)
        if blockers:
            return decision(Outcome.ABSTAIN, *blockers)

        observations = {r.observation for r in supplied.records}
        if Observation.INCONCLUSIVE in observations:
            blockers.add(Reason.INCONCLUSIVE)
        if Observation.POSITIVE in observations and Observation.NEGATIVE in observations:
            blockers.add(Reason.CONFLICT)
        if len({r.artifact_sha256 for r in supplied.records}) != len(supplied.records):
            blockers.add(Reason.DUPLICATE_ARTIFACT)
        if len({roster[r.evidence_id].source_group for r in supplied.records}) < policy.min_source_groups:
            blockers.add(Reason.INSUFFICIENT_GROUPS)
        if blockers:
            return decision(Outcome.ABSTAIN, *blockers)
        return decision(Outcome.REVIEW_READY, Reason.METADATA_COMPLETE)
