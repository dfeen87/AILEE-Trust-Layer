# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Focused unit and integration tests for the Cybersecurity Trust Domain."""

from datetime import datetime, timedelta, timezone
import pytest

from ailee.domains.cybersecurity import (
    ActionTier,
    AdvisorySignal,
    CanonicalURL,
    CybersecurityActionWrapper,
    CybersecurityAuditRecord,
    CybersecurityAuthorizationRequest,
    CybersecurityDecision,
    CybersecurityGovernor,
    CybersecurityHealthStatus,
    CybersecurityReason,
    DecisionOutcome,
    EvidenceProvenanceVerifier,
    PolicyIdentifier,
    ProvenanceStatus,
    PublisherProvenanceEvidence,
    UntrustedContentContext,
    check_markdown_link,
    create_cybersecurity_action_wrapper,
    create_cybersecurity_governor,
    create_permissive_cybersecurity_governor,
    create_strict_cybersecurity_governor,
    get_decision_history,
    get_events,
    get_health,
    get_metrics,
    get_subsystem_health,
    parse_and_canonicalize_url,
    scan_content_advisories,
)

EVAL_TIME = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def valid_provenance():
    return PublisherProvenanceEvidence(
        evidence_id="ev-tool-101",
        software_id="devtools",
        publisher_id="corp.dev",
        publisher_name="Developer Corp",
        canonical_domains=("developer.example.com", "example.com"),
        allowed_download_domains=("developer.example.com", "downloads.example.com"),
        status=ProvenanceStatus.VALID,
        valid_from=EVAL_TIME - timedelta(days=30),
        valid_until=EVAL_TIME + timedelta(days=30),
        verified_at=EVAL_TIME - timedelta(days=1),
        verifier_identity="sigstore-verifier-v1",
        signature_or_digest="sha256:abcdef0123456789",
    )


def test_url_canonicalization_normalizes_ports_paths_queries():
    url = "HTTPS://Example.COM:443/docs/../guide//install/?b=2&a=1#section"
    canonical, reason, msg = parse_and_canonicalize_url(url, allowed_schemes=("https",))

    assert reason is None
    assert canonical is not None
    assert canonical.scheme == "https"
    assert canonical.hostname == "example.com"
    assert canonical.port is None  # default 443 stripped
    assert canonical.path == "/guide/install/"
    assert canonical.query == "a=1&b=2"  # deterministic parameter sorting
    assert canonical.canonical_url == "https://example.com/guide/install/?a=1&b=2"


def test_url_rejection_for_unsupported_schemes():
    for bad_url in [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html;base64,PHNjcmlwdD4=",
        "ftp://ftp.example.com/file.dmg",
    ]:
        canonical, reason, _ = parse_and_canonicalize_url(bad_url)
        assert canonical is None
        assert reason is CybersecurityReason.UNSUPPORTED_SCHEME


def test_governor_evaluates_informational_link_cleanly():
    gov = create_cybersecurity_governor()
    req = CybersecurityAuthorizationRequest(
        request_id="req-info-1",
        action_tier=ActionTier.DISPLAY_INFORMATIONAL_LINK,
        url="https://docs.python.org/3/library/urllib.parse.html",
        evaluation_time=EVAL_TIME,
    )
    decision = gov.authorize(req)

    assert decision.authorized is True
    assert decision.outcome == DecisionOutcome.ALLOW
    assert decision.reason == CybersecurityReason.AUTHORIZED
    assert decision.policy_id == PolicyIdentifier.POLICY_ACTION_AUTHORIZATION
    assert decision.canonical_url == "https://docs.python.org/3/library/urllib.parse.html"


def test_software_recommendation_requires_valid_publisher_provenance(valid_provenance):
    gov = create_cybersecurity_governor()

    # Request without evidence must fail closed
    req_no_ev = CybersecurityAuthorizationRequest(
        request_id="req-rec-no-ev",
        action_tier=ActionTier.RECOMMEND_SOFTWARE,
        url="https://developer.example.com/app",
        software_id="devtools",
        evaluation_time=EVAL_TIME,
        evidence=None,
    )
    dec_no_ev = gov.authorize(req_no_ev)
    assert dec_no_ev.authorized is False
    assert dec_no_ev.outcome == DecisionOutcome.DENY
    assert dec_no_ev.reason == CybersecurityReason.PROVENANCE_MISSING

    # Request with valid evidence matching canonical domain succeeds
    req_valid = CybersecurityAuthorizationRequest(
        request_id="req-rec-valid",
        action_tier=ActionTier.RECOMMEND_SOFTWARE,
        url="https://developer.example.com/app",
        software_id="devtools",
        evaluation_time=EVAL_TIME,
        evidence=valid_provenance,
    )
    dec_valid = gov.authorize(req_valid)
    assert dec_valid.authorized is True
    assert dec_valid.outcome == DecisionOutcome.ALLOW
    assert dec_valid.reason == CybersecurityReason.AUTHORIZED


def test_file_download_requires_allowed_download_domain(valid_provenance):
    gov = create_cybersecurity_governor()

    # Domain in canonical list but NOT in allowed_download_domains
    req_unauth = CybersecurityAuthorizationRequest(
        request_id="req-dl-unauth",
        action_tier=ActionTier.DOWNLOAD_FILE,
        url="https://example.com/downloads/tool.dmg",
        software_id="devtools",
        evaluation_time=EVAL_TIME,
        evidence=valid_provenance,
    )
    dec_unauth = gov.authorize(req_unauth)
    assert dec_unauth.authorized is False
    assert dec_unauth.outcome == DecisionOutcome.DENY
    assert dec_unauth.reason == CybersecurityReason.UNAUTHORIZED_DOWNLOAD_DOMAIN

    # Domain in allowed_download_domains succeeds
    req_auth = CybersecurityAuthorizationRequest(
        request_id="req-dl-auth",
        action_tier=ActionTier.DOWNLOAD_FILE,
        url="https://downloads.example.com/packages/tool.dmg",
        software_id="devtools",
        evaluation_time=EVAL_TIME,
        evidence=valid_provenance,
    )
    dec_auth = gov.authorize(req_auth)
    assert dec_auth.authorized is True
    assert dec_auth.outcome == DecisionOutcome.ALLOW
    assert dec_auth.reason == CybersecurityReason.AUTHORIZED


def test_install_software_requires_explicit_caller_mandate(valid_provenance):
    gov = create_cybersecurity_governor()

    # Missing mandate fails closed
    req_no_mandate = CybersecurityAuthorizationRequest(
        request_id="req-inst-no-man",
        action_tier=ActionTier.INSTALL_SOFTWARE,
        url="https://downloads.example.com/packages/tool.pkg",
        software_id="devtools",
        evaluation_time=EVAL_TIME,
        evidence=valid_provenance,
        caller_mandate=None,
    )
    dec_no_mandate = gov.authorize(req_no_mandate)
    assert dec_no_mandate.authorized is False
    assert dec_no_mandate.outcome == DecisionOutcome.DENY
    assert dec_no_mandate.reason == CybersecurityReason.INSUFFICIENT_ACTION_AUTHORIZATION

    # With explicit caller mandate
    req_mandate = CybersecurityAuthorizationRequest(
        request_id="req-inst-man",
        action_tier=ActionTier.INSTALL_SOFTWARE,
        url="https://downloads.example.com/packages/tool.pkg",
        software_id="devtools",
        evaluation_time=EVAL_TIME,
        evidence=valid_provenance,
        caller_mandate="mandate:user-explicit-approval-token-42",
    )
    dec_mandate = gov.authorize(req_mandate)
    assert dec_mandate.authorized is True
    assert dec_mandate.outcome == DecisionOutcome.ALLOW
    assert dec_mandate.reason == CybersecurityReason.AUTHORIZED


def test_action_wrapper_blocks_denied_calls(valid_provenance):
    called = []

    def mock_installer_tool(target_url: str):
        called.append(target_url)
        return "SUCCESSFUL_INSTALL"

    wrapper = create_cybersecurity_action_wrapper(mock_installer_tool)

    # Denied request (no mandate)
    req_denied = CybersecurityAuthorizationRequest(
        request_id="req-wrap-1",
        action_tier=ActionTier.INSTALL_SOFTWARE,
        url="https://downloads.example.com/packages/tool.pkg",
        software_id="devtools",
        evaluation_time=EVAL_TIME,
        evidence=valid_provenance,
        caller_mandate=None,
    )
    result = wrapper.execute(req_denied, "https://downloads.example.com/packages/tool.pkg")

    assert result["blocked"] is True
    assert result["outcome"] == "DENY"
    assert result["result"] is None
    assert len(called) == 0  # Underlying tool was never invoked!

    # Allowed request (with mandate)
    req_allowed = CybersecurityAuthorizationRequest(
        request_id="req-wrap-2",
        action_tier=ActionTier.INSTALL_SOFTWARE,
        url="https://downloads.example.com/packages/tool.pkg",
        software_id="devtools",
        evaluation_time=EVAL_TIME,
        evidence=valid_provenance,
        caller_mandate="mandate:install-token-123",
    )
    result_ok = wrapper.execute(req_allowed, "https://downloads.example.com/packages/tool.pkg")

    assert result_ok["blocked"] is False
    assert result_ok["outcome"] == "ALLOW"
    assert result_ok["result"] == "SUCCESSFUL_INSTALL"
    assert len(called) == 1


def test_governor_health_and_metrics_telemetry():
    gov = create_strict_cybersecurity_governor()
    assert get_health(gov) == CybersecurityHealthStatus.OPTIMAL
    sub_health = get_subsystem_health(gov)
    assert isinstance(sub_health, dict)
    assert "url_integrity" in sub_health

    # Send a request and check metrics
    req = CybersecurityAuthorizationRequest(
        request_id="req-tel-1",
        action_tier=ActionTier.DISPLAY_INFORMATIONAL_LINK,
        url="https://example.com/info",
        evaluation_time=EVAL_TIME,
    )
    gov.authorize(req)

    metrics = get_metrics(gov)
    assert metrics["total_decisions"] == 1
    assert metrics["allowed_decisions"] == 1

    events = get_events(gov)
    assert len(events) == 1
    assert events[0].request_id == "req-tel-1"

    history = get_decision_history(gov)
    assert len(history) == 1
    assert history[0].authorized is True
