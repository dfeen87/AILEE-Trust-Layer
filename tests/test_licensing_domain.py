from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ailee.domains.licensing import (
    AuthorizationRequest,
    CredentialStatus,
    IntegrityEvidence,
    LicenseContract,
    LicenseGovernor,
    LicenseReason,
)

NOW = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)


@pytest.fixture
def contract():
    return LicenseContract(
        "license-1",
        "customer-1",
        ("machine-a",),
        "issuer-1",
        NOW - timedelta(days=1),
        NOW + timedelta(days=1),
        ("base_operation", "advanced_analytics"),
    )


@pytest.fixture
def auth_request():
    evidence = IntegrityEvidence(
        "credential-1",
        "signed_license",
        CredentialStatus.VALID,
        "issuer-1",
        "license-1",
        "machine-a",
        NOW - timedelta(minutes=1),
        customer_id="customer-1",
    )
    return AuthorizationRequest(
        "request-1",
        "customer-1",
        "machine-a",
        "advanced_analytics",
        NOW,
        evidence,
    )


def test_valid_bound_license_authorizes_deterministically(contract, auth_request):
    governor = LicenseGovernor()
    first = governor.authorize(contract, auth_request)
    second = governor.authorize(contract, auth_request)
    assert first == second
    assert first.authorized and first.result == "AUTHORIZED"
    assert first.reason is LicenseReason.AUTHORIZED
    assert first.audit.credential_evidence_id == "credential-1"


@pytest.mark.parametrize(
    ("contract_change", "request_change", "reason"),
    [
        (
            {},
            {"capability": "predictive_maintenance"},
            LicenseReason.ENTITLEMENT_MISSING,
        ),
        ({"valid_until": NOW}, {}, LicenseReason.EXPIRED),
        ({"valid_from": NOW + timedelta(seconds=1)}, {}, LicenseReason.NOT_YET_VALID),
        ({}, {"asset_id": "machine-b"}, LicenseReason.ASSET_MISMATCH),
        ({}, {"customer_id": "customer-2"}, LicenseReason.CUSTOMER_MISMATCH),
    ],
)
def test_license_failure_reasons_are_explicit(
    contract, auth_request, contract_change, request_change, reason
):
    changed_contract = replace(contract, **contract_change)
    changed_request = replace(auth_request, **request_change)
    decision = LicenseGovernor().authorize(changed_contract, changed_request)
    assert not decision.authorized
    assert decision.result == "DENIED"
    assert decision.reason is reason


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (None, LicenseReason.INSUFFICIENT_EVIDENCE),
        (
            replace(
                IntegrityEvidence(
                    "e",
                    "type",
                    CredentialStatus.VALID,
                    "issuer-1",
                    "license-1",
                    "machine-a",
                    NOW,
                    customer_id="customer-1",
                ),
                status=CredentialStatus.INVALID,
            ),
            LicenseReason.INVALID_CREDENTIAL,
        ),
        (
            replace(
                IntegrityEvidence(
                    "e",
                    "type",
                    CredentialStatus.VALID,
                    "issuer-1",
                    "license-1",
                    "machine-a",
                    NOW,
                    customer_id="customer-1",
                ),
                status=CredentialStatus.UNVERIFIED,
            ),
            LicenseReason.INSUFFICIENT_EVIDENCE,
        ),
    ],
)
def test_invalid_or_missing_credentials_fail_closed(
    contract, auth_request, evidence, reason
):
    assert (
        LicenseGovernor()
        .authorize(contract, replace(auth_request, evidence=evidence))
        .reason
        is reason
    )


def test_missing_required_identity_is_insufficient_evidence(contract, auth_request):
    decision = LicenseGovernor().authorize(
        contract, replace(auth_request, request_id="")
    )
    assert decision.reason is LicenseReason.INSUFFICIENT_EVIDENCE


def test_credential_for_other_license_is_invalid(contract, auth_request):
    evidence = replace(auth_request.evidence, license_id="license-other")
    assert (
        LicenseGovernor()
        .authorize(contract, replace(auth_request, evidence=evidence))
        .reason
        is LicenseReason.INVALID_CREDENTIAL
    )


@pytest.mark.parametrize(
    "contract_value,request_value", [(None, None), ("corrupt", "corrupt")]
)
def test_untyped_malformed_inputs_return_auditable_denial(
    contract_value, request_value
):
    decision = LicenseGovernor().authorize(contract_value, request_value)
    assert not decision.authorized
    assert decision.reason is LicenseReason.INSUFFICIENT_EVIDENCE
    assert decision.audit.reason_codes == (LicenseReason.INSUFFICIENT_EVIDENCE.value,)
    assert decision.decision_id


@pytest.mark.parametrize(
    "target",
    (
        "request_id",
        "asset_id",
        "capability",
        "customer_id",
        "evidence_id",
        "license_id",
    ),
)
def test_non_serializable_identity_values_produce_deterministic_denials(
    contract, auth_request, target
):
    def malformed_inputs():
        malformed_contract = contract
        malformed_request = auth_request
        if target == "evidence_id":
            malformed_request = replace(
                auth_request,
                evidence=replace(auth_request.evidence, evidence_id=object()),
            )
        elif target == "license_id":
            malformed_contract = replace(contract, license_id=object())
        else:
            malformed_request = replace(auth_request, **{target: object()})
        return malformed_contract, malformed_request

    first = LicenseGovernor().authorize(*malformed_inputs())
    second = LicenseGovernor().authorize(*malformed_inputs())

    assert first.result == "DENIED"
    assert not first.authorized
    assert first.reason is LicenseReason.INSUFFICIENT_EVIDENCE
    assert first.decision_id
    assert first.decision_id == second.decision_id
