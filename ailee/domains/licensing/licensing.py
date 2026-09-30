# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Deterministic runtime licensing and entitlement governance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Optional, Protocol, Tuple


SCHEMA_VERSION = "1.0"
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")


class CredentialStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    MISSING = "MISSING"
    UNVERIFIED = "UNVERIFIED"


class LicenseReason(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INVALID_CREDENTIAL = "INVALID_CREDENTIAL"
    CUSTOMER_MISMATCH = "CUSTOMER_MISMATCH"
    ASSET_MISMATCH = "ASSET_MISMATCH"
    NOT_YET_VALID = "NOT_YET_VALID"
    EXPIRED = "EXPIRED"
    ENTITLEMENT_MISSING = "ENTITLEMENT_MISSING"
    MALFORMED_IDENTIFIER = "MALFORMED_IDENTIFIER"
    INVALID_CONTRACT = "INVALID_CONTRACT"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    REPLAY_DETECTED = "REPLAY_DETECTED"


@dataclass(frozen=True)
class IntegrityEvidence:
    evidence_id: str
    credential_type: str
    status: CredentialStatus
    issuer: str
    license_id: str
    asset_id: str
    verified_at: Optional[datetime]
    details: Mapping[str, str] = field(default_factory=dict)
    customer_id: str = ""
    schema_version: str = SCHEMA_VERSION


class CredentialVerifier(Protocol):
    """Boundary for an established signature/credential verifier.

    Implementations return an explicit status; this foundation intentionally does
    not provide home-grown cryptography.
    """

    def verify(self, evidence: Optional[IntegrityEvidence]) -> CredentialStatus: ...


class EvidenceStatusVerifier:
    """Accept an upstream verifier's explicit result, never mere file presence."""

    def verify(self, evidence: Optional[IntegrityEvidence]) -> CredentialStatus:
        if evidence is None:
            return CredentialStatus.MISSING
        if not isinstance(evidence.status, CredentialStatus):
            return CredentialStatus.INVALID
        return evidence.status


@dataclass(frozen=True)
class LicenseContract:
    license_id: str
    customer_id: str
    asset_ids: Tuple[str, ...]
    issuer: str
    valid_from: datetime
    valid_until: datetime
    entitlements: Tuple[str, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class AuthorizationRequest:
    request_id: str
    customer_id: str
    asset_id: str
    capability: str
    evaluated_at: datetime
    evidence: Optional[IntegrityEvidence]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class LicenseAuditEvidence:
    request_id: str
    license_id: str
    customer_id: str
    asset_id: str
    capability: str
    issuer: str
    evaluated_at: datetime
    credential_evidence_id: Optional[str]
    checks_performed: Tuple[str, ...]
    reason_codes: Tuple[str, ...]
    domain: str = "LICENSING"
    schema_version: str = SCHEMA_VERSION
    time_source: str = "CALLER_SUPPLIED_UNTRUSTED_CLOCK"


@dataclass(frozen=True)
class AuthorizationDecision:
    authorized: bool
    reason: LicenseReason
    decision_id: str
    audit: LicenseAuditEvidence

    @property
    def result(self) -> str:
        return "AUTHORIZED" if self.authorized else "DENIED"


def _utc(value: datetime) -> Optional[datetime]:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return value.astimezone(timezone.utc)


def _valid_identifier(value: object) -> bool:
    """Identifiers are already canonical; normalization never grants access."""
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _decision_id(contract: LicenseContract, request: AuthorizationRequest, reason: LicenseReason) -> str:
    payload = {
        "asset_id": request.asset_id,
        "capability": request.capability,
        "customer_id": request.customer_id,
        "evaluated_at": request.evaluated_at.isoformat() if isinstance(request.evaluated_at, datetime) else repr(request.evaluated_at),
        "evidence_id": getattr(request.evidence, "evidence_id", None),
        "license_id": contract.license_id,
        "reason": reason.value,
        "request_id": request.request_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class LicenseGovernor:
    """Evaluate a license contract in a fixed, documented fail-closed order."""

    CHECKS = (
        "required_fields", "credential", "customer_binding", "asset_binding",
        "credential_binding", "validity_window", "entitlement",
    )

    def __init__(self, verifier: Optional[CredentialVerifier] = None, replay_guard=None):
        self._verifier = verifier or EvidenceStatusVerifier()
        self._replay_guard = replay_guard

    def authorize(self, contract: LicenseContract, request: AuthorizationRequest) -> AuthorizationDecision:
        try:
            reason = self._reason(contract, request)
        except (AttributeError, TypeError, ValueError):
            reason = LicenseReason.INSUFFICIENT_EVIDENCE
        audit = LicenseAuditEvidence(
            request_id=request.request_id,
            license_id=contract.license_id,
            customer_id=request.customer_id,
            asset_id=request.asset_id,
            capability=request.capability,
            issuer=contract.issuer,
            evaluated_at=request.evaluated_at,
            credential_evidence_id=getattr(request.evidence, "evidence_id", None),
            checks_performed=self.CHECKS,
            reason_codes=(reason.value,),
        )
        return AuthorizationDecision(
            authorized=reason is LicenseReason.AUTHORIZED,
            reason=reason,
            decision_id=_decision_id(contract, request, reason),
            audit=audit,
        )

    def _reason(self, contract: LicenseContract, request: AuthorizationRequest) -> LicenseReason:
        if not isinstance(contract, LicenseContract) or not isinstance(request, AuthorizationRequest):
            return LicenseReason.INSUFFICIENT_EVIDENCE
        if contract.schema_version != SCHEMA_VERSION or request.schema_version != SCHEMA_VERSION:
            return LicenseReason.UNSUPPORTED_SCHEMA
        required = (
            contract.license_id, contract.customer_id, contract.issuer,
            request.request_id, request.customer_id, request.asset_id, request.capability,
        )
        start, end, now = _utc(contract.valid_from), _utc(contract.valid_until), _utc(request.evaluated_at)
        if not all(isinstance(value, str) and value.strip() for value in required):
            return LicenseReason.INSUFFICIENT_EVIDENCE
        identities = required + tuple(contract.asset_ids) + tuple(contract.entitlements)
        if not all(_valid_identifier(value) for value in identities):
            return LicenseReason.MALFORMED_IDENTIFIER
        if len(set(contract.asset_ids)) != len(contract.asset_ids) or len(set(contract.entitlements)) != len(contract.entitlements):
            return LicenseReason.INVALID_CONTRACT
        if not contract.asset_ids or not contract.entitlements or start is None or end is None or now is None:
            return LicenseReason.INSUFFICIENT_EVIDENCE
        if start >= end:
            return LicenseReason.INVALID_CONTRACT
        try:
            credential_status = self._verifier.verify(request.evidence)
        except Exception:
            return LicenseReason.VERIFICATION_UNAVAILABLE
        if credential_status is None or not isinstance(credential_status, CredentialStatus):
            return LicenseReason.VERIFICATION_UNAVAILABLE
        if credential_status in (CredentialStatus.MISSING, CredentialStatus.UNVERIFIED):
            return LicenseReason.INSUFFICIENT_EVIDENCE
        if credential_status is not CredentialStatus.VALID:
            return LicenseReason.INVALID_CREDENTIAL
        evidence = request.evidence
        if not isinstance(evidence, IntegrityEvidence):
            return LicenseReason.INSUFFICIENT_EVIDENCE
        if evidence.schema_version != SCHEMA_VERSION:
            return LicenseReason.UNSUPPORTED_SCHEMA
        verified_at = _utc(evidence.verified_at) if evidence.verified_at is not None else None
        if not all(_valid_identifier(value) for value in (
            evidence.evidence_id, evidence.credential_type, evidence.issuer,
            evidence.license_id, evidence.asset_id, evidence.customer_id,
        )) or verified_at is None:
            return LicenseReason.INSUFFICIENT_EVIDENCE
        if request.customer_id != contract.customer_id:
            return LicenseReason.CUSTOMER_MISMATCH
        if request.asset_id not in contract.asset_ids:
            return LicenseReason.ASSET_MISMATCH
        if (evidence.license_id != contract.license_id or evidence.asset_id != request.asset_id
                or evidence.customer_id != request.customer_id or evidence.issuer != contract.issuer
                or verified_at > now):
            return LicenseReason.INVALID_CREDENTIAL
        if now < start:
            return LicenseReason.NOT_YET_VALID
        if now >= end:
            return LicenseReason.EXPIRED
        if request.capability not in contract.entitlements:
            return LicenseReason.ENTITLEMENT_MISSING
        if self._replay_guard is not None:
            try:
                accepted = self._replay_guard.accept(request.request_id, now)
            except Exception:
                return LicenseReason.VERIFICATION_UNAVAILABLE
            if accepted is not True:
                return LicenseReason.REPLAY_DETECTED
        return LicenseReason.AUTHORIZED
