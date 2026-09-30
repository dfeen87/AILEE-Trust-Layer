# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Deterministic runtime licensing and entitlement governance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Optional, Protocol, Tuple


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


@dataclass(frozen=True)
class AuthorizationRequest:
    request_id: str
    customer_id: str
    asset_id: str
    capability: str
    evaluated_at: datetime
    evidence: Optional[IntegrityEvidence]


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


def _decision_id(contract: LicenseContract, request: AuthorizationRequest, reason: LicenseReason) -> str:
    payload = {
        "asset_id": request.asset_id,
        "capability": request.capability,
        "customer_id": request.customer_id,
        "evaluated_at": request.evaluated_at.isoformat() if isinstance(request.evaluated_at, datetime) else repr(request.evaluated_at),
        "evidence_id": request.evidence.evidence_id if request.evidence else None,
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

    def __init__(self, verifier: Optional[CredentialVerifier] = None):
        self._verifier = verifier or EvidenceStatusVerifier()

    def authorize(self, contract: LicenseContract, request: AuthorizationRequest) -> AuthorizationDecision:
        reason = self._reason(contract, request)
        audit = LicenseAuditEvidence(
            request_id=request.request_id,
            license_id=contract.license_id,
            customer_id=request.customer_id,
            asset_id=request.asset_id,
            capability=request.capability,
            issuer=contract.issuer,
            evaluated_at=request.evaluated_at,
            credential_evidence_id=request.evidence.evidence_id if request.evidence else None,
            checks_performed=self.CHECKS,
        )
        return AuthorizationDecision(
            authorized=reason is LicenseReason.AUTHORIZED,
            reason=reason,
            decision_id=_decision_id(contract, request, reason),
            audit=audit,
        )

    def _reason(self, contract: LicenseContract, request: AuthorizationRequest) -> LicenseReason:
        required = (
            contract.license_id, contract.customer_id, contract.issuer,
            request.request_id, request.customer_id, request.asset_id, request.capability,
        )
        start, end, now = _utc(contract.valid_from), _utc(contract.valid_until), _utc(request.evaluated_at)
        if not all(isinstance(value, str) and value.strip() for value in required):
            return LicenseReason.INSUFFICIENT_EVIDENCE
        if not contract.asset_ids or not contract.entitlements or start is None or end is None or now is None or start >= end:
            return LicenseReason.INSUFFICIENT_EVIDENCE
        credential_status = self._verifier.verify(request.evidence)
        if credential_status in (CredentialStatus.MISSING, CredentialStatus.UNVERIFIED):
            return LicenseReason.INSUFFICIENT_EVIDENCE
        if credential_status is not CredentialStatus.VALID:
            return LicenseReason.INVALID_CREDENTIAL
        evidence = request.evidence
        if evidence is None or not evidence.evidence_id or not evidence.issuer or evidence.verified_at is None:
            return LicenseReason.INSUFFICIENT_EVIDENCE
        if request.customer_id != contract.customer_id:
            return LicenseReason.CUSTOMER_MISMATCH
        if request.asset_id not in contract.asset_ids:
            return LicenseReason.ASSET_MISMATCH
        if evidence.license_id != contract.license_id or evidence.asset_id != request.asset_id or evidence.issuer != contract.issuer:
            return LicenseReason.INVALID_CREDENTIAL
        if now < start:
            return LicenseReason.NOT_YET_VALID
        if now >= end:
            return LicenseReason.EXPIRED
        if request.capability not in contract.entitlements:
            return LicenseReason.ENTITLEMENT_MISSING
        return LicenseReason.AUTHORIZED
