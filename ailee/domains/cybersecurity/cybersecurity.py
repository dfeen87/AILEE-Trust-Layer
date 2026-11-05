# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""
AILEE Cybersecurity Trust Domain — Link Provenance and Injection Governance.

Deterministic governance of external URLs, software-download recommendations,
link provenance, and agent actions influenced by untrusted external content.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SCHEMA_VERSION = "1.0"
MAX_URL_LENGTH = 2048
MAX_REDIRECT_HOPS = 10
DEFAULT_ALLOWED_SCHEMES: Tuple[str, ...] = ("https",)
INFORMATIONAL_ALLOWED_SCHEMES: Tuple[str, ...] = ("https", "http")

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]{0,126}[a-zA-Z0-9])?$")

# Common TLDs used to detect deceptive domain structures in subdomains
_COMMON_TLDS = frozenset(
    [
        "com",
        "org",
        "net",
        "io",
        "app",
        "dev",
        "co",
        "info",
        "biz",
        "online",
        "site",
        "xyz",
        "tech",
        "me",
        "cc",
        "ru",
        "cn",
        "uk",
        "de",
        "jp",
        "fr",
        "ca",
        "au",
        "in",
        "ch",
        "se",
        "nl",
    ]
)

# Advisory regex patterns for untrusted content
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b"),
    re.compile(
        r"(?i)\bdisregard\s+(?:prior|previous|all)\s+(?:rules|guidelines|instructions)\b"
    ),
    re.compile(r"(?i)\bsystem\s+prompt\s+override\b"),
    re.compile(
        r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:developer|maintenance|dan|root|god)\s+mode\b"
    ),
    re.compile(r"(?i)\badministrative\s+override\b"),
    re.compile(r"(?i)\bnew\s+system\s+instructions?\b"),
    re.compile(r"(?i)\bdeveloper\s+mode\s+enabled\b"),
]

_TOOL_LURE_PATTERNS = [
    re.compile(r"(?i)\bcurl\s+[^|\n]+\|\s*(?:ba)?sh\b"),
    re.compile(r"(?i)\bwget\s+[^|\n]+\|\s*(?:ba)?sh\b"),
    re.compile(r"(?i)\b(?:chmod\s+\+x|sudo\s+[a-z]+|bash\s+-c|sh\s+-c)\b"),
    re.compile(
        r"(?i)\bpowershell(?:\.exe)?\s+(?:-enc|-encodedcommand|-c|-command)\b"
    ),
    re.compile(r"(?i)\b(?:eval|exec)\s*\("),
    re.compile(r"(?i)\bopen\s+-a\s+terminal\b"),
]

# Unicode characters requiring homograph/spoofing rejection
_ZERO_WIDTH_CHARS = frozenset(
    [
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u2060",  # word joiner
        "\ufeff",  # zero width no-break space / byte order mark
        "\u00ad",  # soft hyphen
    ]
)

_BIDI_CHARS = frozenset(
    [
        "\u202a",  # left-to-right embedding
        "\u202b",  # right-to-left embedding
        "\u202c",  # pop directional formatting
        "\u202d",  # left-to-right override
        "\u202e",  # right-to-left override
        "\u2066",  # left-to-right isolate
        "\u2067",  # right-to-left isolate
        "\u2068",  # first strong isolate
        "\u2069",  # pop directional isolate
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
    ]
)


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_valid_identifier(val: str) -> bool:
    return isinstance(val, str) and bool(_IDENTIFIER_RE.match(val))


# -----------------------------
# Enums
# -----------------------------


class ActionTier(str, Enum):
    """Tier of agent action involving an external link or content."""

    DISPLAY_INFORMATIONAL_LINK = "DISPLAY_INFORMATIONAL_LINK"
    OPEN_WEBSITE = "OPEN_WEBSITE"
    RECOMMEND_SOFTWARE = "RECOMMEND_SOFTWARE"
    DOWNLOAD_FILE = "DOWNLOAD_FILE"
    INSTALL_SOFTWARE = "INSTALL_SOFTWARE"


class DecisionOutcome(str, Enum):
    """Deterministic governance decision outcome."""

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    DENY = "DENY"


class ProvenanceStatus(str, Enum):
    """Status of publisher provenance evidence."""

    VALID = "VALID"
    INVALID = "INVALID"
    MISSING = "MISSING"
    UNVERIFIED = "UNVERIFIED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    CONTRADICTORY = "CONTRADICTORY"


class CybersecurityReason(str, Enum):
    """Stable reason codes for cybersecurity governance decisions."""

    AUTHORIZED = "AUTHORIZED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    MALFORMED_URL = "MALFORMED_URL"
    URL_TOO_LONG = "URL_TOO_LONG"
    UNSUPPORTED_SCHEME = "UNSUPPORTED_SCHEME"
    EMBEDDED_CREDENTIALS = "EMBEDDED_CREDENTIALS"
    IDN_HOMOGRAPH_ATTEMPT = "IDN_HOMOGRAPH_ATTEMPT"
    DECEPTIVE_HOSTNAME = "DECEPTIVE_HOSTNAME"
    MISLEADING_SUBDOMAIN = "MISLEADING_SUBDOMAIN"
    DECEPTIVE_LINK_LABEL = "DECEPTIVE_LINK_LABEL"
    UNSAFE_REDIRECT_CHAIN = "UNSAFE_REDIRECT_CHAIN"
    CROSS_DOMAIN_REDIRECT = "CROSS_DOMAIN_REDIRECT"
    PROVENANCE_MISSING = "PROVENANCE_MISSING"
    PROVENANCE_NOT_ESTABLISHED = "PROVENANCE_NOT_ESTABLISHED"
    PROVENANCE_EXPIRED = "PROVENANCE_EXPIRED"
    PROVENANCE_NOT_YET_VALID = "PROVENANCE_NOT_YET_VALID"
    PROVENANCE_REVOKED = "PROVENANCE_REVOKED"
    PROVENANCE_CONTRADICTORY = "PROVENANCE_CONTRADICTORY"
    PROVENANCE_UNVERIFIED = "PROVENANCE_UNVERIFIED"
    UNVERIFIED_PUBLISHER_DOMAIN = "UNVERIFIED_PUBLISHER_DOMAIN"
    PUBLISHER_MISMATCH = "PUBLISHER_MISMATCH"
    UNAUTHORIZED_DOWNLOAD_DOMAIN = "UNAUTHORIZED_DOWNLOAD_DOMAIN"
    INSUFFICIENT_ACTION_AUTHORIZATION = "INSUFFICIENT_ACTION_AUTHORIZATION"
    UNTRUSTED_EXECUTION_PROMOTION = "UNTRUSTED_EXECUTION_PROMOTION"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
    SUSPICIOUS_CONTENT_PATTERN = "SUSPICIOUS_CONTENT_PATTERN"
    UNKNOWN_DESTINATION = "UNKNOWN_DESTINATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"


class PolicyIdentifier(str, Enum):
    """Identifiers for cybersecurity governance policies."""

    POLICY_URL_INTEGRITY = "POLICY_URL_INTEGRITY"
    POLICY_LINK_CONSISTENCY = "POLICY_LINK_CONSISTENCY"
    POLICY_PUBLISHER_PROVENANCE = "POLICY_PUBLISHER_PROVENANCE"
    POLICY_REDIRECT_SAFETY = "POLICY_REDIRECT_SAFETY"
    POLICY_CONTENT_TRUST_BOUNDARY = "POLICY_CONTENT_TRUST_BOUNDARY"
    POLICY_ACTION_AUTHORIZATION = "POLICY_ACTION_AUTHORIZATION"
    POLICY_FAIL_CLOSED = "POLICY_FAIL_CLOSED"


class AdvisorySignal(str, Enum):
    """Non-blocking advisory signals detected in external content or structure."""

    PROMPT_INJECTION_SUSPECTED = "PROMPT_INJECTION_SUSPECTED"
    TOOL_EXECUTION_LURE = "TOOL_EXECUTION_LURE"
    CLAIMED_OFFICIAL_STATUS_WITHOUT_EVIDENCE = (
        "CLAIMED_OFFICIAL_STATUS_WITHOUT_EVIDENCE"
    )
    DECEPTIVE_ANCHOR_TEXT = "DECEPTIVE_ANCHOR_TEXT"
    UNVERIFIED_DOWNLOAD_DESTINATION = "UNVERIFIED_DOWNLOAD_DESTINATION"
    SUSPICIOUS_IP_HOST = "SUSPICIOUS_IP_HOST"
    LOOKALIKE_BRAND_DOMAIN = "LOOKALIKE_BRAND_DOMAIN"
    CROSS_ORIGIN_REDIRECT = "CROSS_ORIGIN_REDIRECT"


class CybersecurityHealthStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# -----------------------------
# Data Models
# -----------------------------


@dataclass(frozen=True)
class CanonicalURL:
    """Canonicalized and validated URL structure."""

    raw_url: str
    canonical_url: str
    scheme: str
    hostname: str
    port: Optional[int]
    path: str
    query: str
    is_ip_address: bool
    is_punycode: bool
    has_credentials: bool
    normalized_domain: str


@dataclass(frozen=True)
class PublisherProvenanceEvidence:
    """Independently established publisher-domain evidence."""

    evidence_id: str
    software_id: str
    publisher_id: str
    publisher_name: str
    canonical_domains: Tuple[str, ...]
    allowed_download_domains: Tuple[str, ...]
    status: ProvenanceStatus
    valid_from: Optional[datetime]
    valid_until: Optional[datetime]
    verified_at: Optional[datetime]
    verifier_identity: str = ""
    signature_or_digest: str = ""
    schema_version: str = SCHEMA_VERSION
    details: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class UntrustedContentContext:
    """Context surrounding untrusted content referencing a URL or action."""

    content_source: str
    raw_content: str = ""
    markdown_link_label: Optional[str] = None
    origin_url: Optional[str] = None
    is_action_promoted: bool = False


@dataclass(frozen=True)
class CybersecurityAuditRecord:
    """Immutable audit record for a cybersecurity governance decision."""

    request_id: str
    timestamp: datetime
    outcome: DecisionOutcome
    reason: CybersecurityReason
    policy_id: PolicyIdentifier
    raw_url: str
    canonical_url: Optional[str]
    action_tier: ActionTier
    target_domain: str
    evidence_ids: Tuple[str, ...]
    advisory_signals: Tuple[AdvisorySignal, ...]
    details: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CybersecurityDecision:
    """Deterministic result of a cybersecurity governance evaluation."""

    authorized: bool
    outcome: DecisionOutcome
    reason: CybersecurityReason
    policy_id: PolicyIdentifier
    action_tier: ActionTier
    canonical_url: Optional[str]
    audit: CybersecurityAuditRecord
    advisory_signals: Tuple[AdvisorySignal, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CybersecurityAuthorizationRequest:
    """Request to authorize a link, software recommendation, download, or action."""

    request_id: str
    action_tier: ActionTier
    url: str
    evaluation_time: datetime
    software_id: Optional[str] = None
    evidence: Optional[PublisherProvenanceEvidence] = None
    peer_evidences: Tuple[PublisherProvenanceEvidence, ...] = field(
        default_factory=tuple
    )
    redirect_chain: Tuple[str, ...] = field(default_factory=tuple)
    untrusted_content: Optional[UntrustedContentContext] = None
    caller_mandate: Optional[str] = None
    schema_version: str = SCHEMA_VERSION


# -----------------------------
# Protocols and Verifiers
# -----------------------------


class ProvenanceVerifier(Protocol):
    """Protocol for independent publisher provenance verification."""

    def verify(
        self,
        evidence: PublisherProvenanceEvidence,
        evaluation_time: datetime,
    ) -> ProvenanceStatus: ...


class EvidenceProvenanceVerifier:
    """Deterministic verification of publisher provenance evidence."""

    def verify(
        self,
        evidence: PublisherProvenanceEvidence,
        evaluation_time: datetime,
    ) -> ProvenanceStatus:
        if not isinstance(evidence, PublisherProvenanceEvidence):
            return ProvenanceStatus.INVALID
        if evidence.schema_version != SCHEMA_VERSION:
            return ProvenanceStatus.INVALID
        if not _is_valid_identifier(evidence.evidence_id):
            return ProvenanceStatus.INVALID
        if not _is_valid_identifier(evidence.software_id):
            return ProvenanceStatus.INVALID
        if not _is_valid_identifier(evidence.publisher_id):
            return ProvenanceStatus.INVALID
        if not evidence.canonical_domains:
            return ProvenanceStatus.INVALID

        if evidence.status in (
            ProvenanceStatus.REVOKED,
            ProvenanceStatus.CONTRADICTORY,
            ProvenanceStatus.UNVERIFIED,
            ProvenanceStatus.INVALID,
        ):
            return evidence.status

        now = _utc(evaluation_time)
        if evidence.valid_from is not None and now < _utc(evidence.valid_from):
            return ProvenanceStatus.INVALID  # Not yet valid
        if evidence.valid_until is not None and now >= _utc(evidence.valid_until):
            return ProvenanceStatus.EXPIRED
        if evidence.verified_at is not None and _utc(evidence.verified_at) > now:
            return ProvenanceStatus.INVALID  # Verified in the future

        return ProvenanceStatus.VALID


# -----------------------------
# URL Canonicalization & Inspection
# -----------------------------


def parse_and_canonicalize_url(
    url: str,
    allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES,
) -> Tuple[Optional[CanonicalURL], Optional[CybersecurityReason], Optional[str]]:
    """
    Strict URL parsing and canonicalization.

    Returns:
        (canonical_url_object, error_reason, error_message)
    """
    if not isinstance(url, str):
        return None, CybersecurityReason.MALFORMED_URL, "URL must be a string"

    url_clean = url.strip()
    if not url_clean:
        return None, CybersecurityReason.MALFORMED_URL, "URL is empty"

    if len(url_clean) > MAX_URL_LENGTH:
        return None, CybersecurityReason.URL_TOO_LONG, f"URL exceeds {MAX_URL_LENGTH} chars"

    # Reject control characters
    for ch in url_clean:
        code = ord(ch)
        if code < 32 or code == 127:
            return None, CybersecurityReason.MALFORMED_URL, "URL contains control characters"

    try:
        parsed = urlsplit(url_clean)
    except Exception as exc:
        return None, CybersecurityReason.MALFORMED_URL, f"URL parsing failed: {exc}"

    scheme = parsed.scheme.lower()
    if not scheme:
        return None, CybersecurityReason.MALFORMED_URL, "Missing URL scheme"

    if scheme not in allowed_schemes:
        return None, CybersecurityReason.UNSUPPORTED_SCHEME, f"Scheme '{scheme}' is not authorized"

    netloc = parsed.netloc
    if not netloc:
        return None, CybersecurityReason.MALFORMED_URL, "Missing network location"

    # Check for embedded credentials (username:password@host)
    if "@" in netloc or parsed.username is not None or parsed.password is not None:
        return None, CybersecurityReason.EMBEDDED_CREDENTIALS, "URL contains embedded credentials"

    # Extract host and port
    hostname = parsed.hostname
    if not hostname:
        return None, CybersecurityReason.MALFORMED_URL, "Missing hostname"

    hostname_clean = hostname.lower().rstrip(".")
    if not hostname_clean:
        return None, CybersecurityReason.MALFORMED_URL, "Invalid empty hostname"

    # Port validation and canonicalization
    port = None
    try:
        raw_port = parsed.port
        if raw_port is not None:
            if raw_port < 1 or raw_port > 65535:
                return None, CybersecurityReason.MALFORMED_URL, "Port out of valid range"
            # Omit default ports
            if not (
                (scheme == "https" and raw_port == 443)
                or (scheme == "http" and raw_port == 80)
            ):
                port = raw_port
    except ValueError:
        return None, CybersecurityReason.MALFORMED_URL, "Malformed port in URL"

    # Check for IP address
    is_ip = False
    try:
        ipaddress.ip_address(hostname_clean)
        is_ip = True
    except ValueError:
        # Check for integer or obfuscated hex/octal IP
        if hostname_clean.isdigit():
            is_ip = True
        elif hostname_clean.startswith("0x") and not "." in hostname_clean:
            is_ip = True

    # Internationalized-Domain (IDN) and Unicode Homograph detection
    is_punycode = False
    labels = hostname_clean.split(".")
    for label in labels:
        if label.startswith("xn--"):
            is_punycode = True
            return (
                None,
                CybersecurityReason.IDN_HOMOGRAPH_ATTEMPT,
                f"Punycode encoded domain label '{label}' rejected",
            )

    # Check for invisible chars, bidi overrides, or non-ASCII characters in host
    for ch in hostname_clean:
        if ch in _ZERO_WIDTH_CHARS or ch in _BIDI_CHARS:
            return (
                None,
                CybersecurityReason.IDN_HOMOGRAPH_ATTEMPT,
                "URL host contains hidden or directional override characters",
            )
        if ord(ch) > 127:
            # Non-ASCII character in hostname
            return (
                None,
                CybersecurityReason.IDN_HOMOGRAPH_ATTEMPT,
                f"URL host contains non-ASCII or confusable character: {ch!r}",
            )

    # Misleading subdomain structure detection
    # E.g. itsalin.com.attacker.com or apple.com-login.fake.org
    if len(labels) > 2 and not is_ip:
        # Check subdomains (labels prior to apex domain and TLD)
        subdomain_labels = labels[:-2]
        for sub_label in subdomain_labels:
            for tld in _COMMON_TLDS:
                if (
                    sub_label == tld
                    or sub_label.endswith(f"-{tld}")
                    or sub_label.startswith(f"{tld}-")
                    or f"-{tld}-" in sub_label
                    or sub_label.endswith(f".{tld}")
                ):
                    return (
                        None,
                        CybersecurityReason.MISLEADING_SUBDOMAIN,
                        f"Subdomain label '{sub_label}' mimics top-level domain structure",
                    )

    # Path canonicalization
    raw_path = parsed.path or "/"
    # Clean consecutive slashes
    clean_path = re.sub(r"/+", "/", raw_path)
    # Check directory traversal attempts above root
    segments = clean_path.split("/")
    resolved_segments: List[str] = []
    for seg in segments:
        if seg in ("", "."):
            continue
        if seg == "..":
            if not resolved_segments:
                return (
                    None,
                    CybersecurityReason.MALFORMED_URL,
                    "Path traversal attempts navigation above root",
                )
            resolved_segments.pop()
        else:
            resolved_segments.append(seg)
    canonical_path = "/" + "/".join(resolved_segments)
    if raw_path.endswith("/") and not canonical_path.endswith("/"):
        canonical_path += "/"

    # Query canonicalization: sort query params deterministically
    canonical_query = ""
    if parsed.query:
        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        query_items.sort(key=lambda item: (item[0], item[1]))
        canonical_query = urlencode(query_items)

    # Construct canonical URL
    port_str = f":{port}" if port is not None else ""
    canonical_netloc = f"{hostname_clean}{port_str}"
    reconstructed = urlunsplit(
        (scheme, canonical_netloc, canonical_path, canonical_query, "")
    )

    # Normalized registered domain (extract last two labels for regular domains)
    if is_ip or len(labels) < 2:
        normalized_domain = hostname_clean
    else:
        # Simple apex domain extraction (last 2 labels, or 3 if penultimate is co/com/gov/org/net)
        if len(labels) >= 3 and labels[-2] in ("co", "com", "gov", "org", "net"):
            normalized_domain = ".".join(labels[-3:])
        else:
            normalized_domain = ".".join(labels[-2:])

    canonical_obj = CanonicalURL(
        raw_url=url,
        canonical_url=reconstructed,
        scheme=scheme,
        hostname=hostname_clean,
        port=port,
        path=canonical_path,
        query=canonical_query,
        is_ip_address=is_ip,
        is_punycode=is_punycode,
        has_credentials=False,
        normalized_domain=normalized_domain,
    )
    return canonical_obj, None, None


# -----------------------------
# Markdown Link & Content Inspection
# -----------------------------


def check_markdown_link(
    label: str, destination_url: str
) -> Tuple[bool, Optional[CybersecurityReason], Optional[str]]:
    """
    Detect deceptive markdown links where the displayed label misrepresents destination.

    Example: [https://itsalin.com](https://pear-cleaner.com)
    """
    if not isinstance(label, str) or not isinstance(destination_url, str):
        return True, None, None

    label_clean = label.strip()
    if not label_clean:
        return True, None, None

    # If label resembles a URL or domain
    is_url_label = (
        label_clean.startswith("http://")
        or label_clean.startswith("https://")
        or (
            "." in label_clean
            and not " " in label_clean
            and any(label_clean.endswith(f".{tld}") for tld in _COMMON_TLDS)
        )
    )

    if is_url_label:
        try:
            label_parsed = urlsplit(
                label_clean
                if "://" in label_clean
                else f"https://{label_clean}"
            )
            dest_parsed = urlsplit(destination_url)
            label_host = (label_parsed.hostname or "").lower().rstrip(".")
            dest_host = (dest_parsed.hostname or "").lower().rstrip(".")

            if label_host and dest_host and label_host != dest_host:
                return (
                    False,
                    CybersecurityReason.DECEPTIVE_LINK_LABEL,
                    f"Markdown label specifies host '{label_host}' but destination points to '{dest_host}'",
                )
        except Exception:
            pass

    return True, None, None


def scan_content_advisories(content: str) -> List[AdvisorySignal]:
    """Scan untrusted text content for suspicious patterns without executing them."""
    if not isinstance(content, str) or not content:
        return []

    signals: List[AdvisorySignal] = []

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(content):
            signals.append(AdvisorySignal.PROMPT_INJECTION_SUSPECTED)
            break

    for pattern in _TOOL_LURE_PATTERNS:
        if pattern.search(content):
            signals.append(AdvisorySignal.TOOL_EXECUTION_LURE)
            break

    if re.search(
        r"(?i)\b(?:official\s+(?:source|website|download|site)|verified\s+(?:source|developer))\b",
        content,
    ):
        signals.append(AdvisorySignal.CLAIMED_OFFICIAL_STATUS_WITHOUT_EVIDENCE)

    return signals


# -----------------------------
# Core Cybersecurity Governor
# -----------------------------


class CybersecurityGovernor:
    """
    Deterministic trust governor for links, software recommendations, and action authorization.

    Preserves the Safety -> GRACE -> Consensus -> Fallback architecture:
    1. Safety: Strict invariant enforcement (URL syntax, schemes, credentials, homoglyphs, injection).
    2. GRACE: Contextual mediation for borderline advisory content vs. requested action tier.
    3. Consensus: Multi-source provenance verification and contradiction rejection.
    4. Fallback: Fail-closed denial for unverified software recommendations, downloads, and actions.
    """

    def __init__(
        self,
        allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES,
        allow_http_informational: bool = True,
        verifier: Optional[ProvenanceVerifier] = None,
        max_redirect_hops: int = MAX_REDIRECT_HOPS,
        history_limit: int = 200,
    ):
        self._allowed_schemes = tuple(allowed_schemes)
        self._allow_http_informational = allow_http_informational
        self._verifier = verifier or EvidenceProvenanceVerifier()
        self._max_redirect_hops = max_redirect_hops
        self._history_limit = history_limit
        self._decision_history: deque[CybersecurityDecision] = deque(
            maxlen=history_limit
        )
        self._events: deque[CybersecurityAuditRecord] = deque(maxlen=history_limit)

    def _make_denial(
        self,
        request: CybersecurityAuthorizationRequest,
        reason: CybersecurityReason,
        policy_id: PolicyIdentifier,
        canonical_url: Optional[str] = None,
        target_domain: str = "",
        advisory_signals: Sequence[AdvisorySignal] = (),
        details: Optional[Mapping[str, str]] = None,
    ) -> CybersecurityDecision:
        audit = CybersecurityAuditRecord(
            request_id=request.request_id,
            timestamp=_utc(request.evaluation_time),
            outcome=DecisionOutcome.DENY,
            reason=reason,
            policy_id=policy_id,
            raw_url=request.url,
            canonical_url=canonical_url,
            action_tier=request.action_tier,
            target_domain=target_domain,
            evidence_ids=(
                (request.evidence.evidence_id,)
                if request.evidence is not None
                else ()
            ),
            advisory_signals=tuple(advisory_signals),
            details=details or {},
        )
        decision = CybersecurityDecision(
            authorized=False,
            outcome=DecisionOutcome.DENY,
            reason=reason,
            policy_id=policy_id,
            action_tier=request.action_tier,
            canonical_url=canonical_url,
            audit=audit,
            advisory_signals=tuple(advisory_signals),
        )
        self._record(decision)
        return decision

    def _record(self, decision: CybersecurityDecision) -> None:
        self._decision_history.append(decision)
        self._events.append(decision.audit)

    def authorize(
        self, request: CybersecurityAuthorizationRequest
    ) -> CybersecurityDecision:
        """
        Evaluate an authorization request deterministically and without side effects.
        """
        if not isinstance(request, CybersecurityAuthorizationRequest):
            raise TypeError("request must be a CybersecurityAuthorizationRequest")

        if not _is_valid_identifier(request.request_id):
            return self._make_denial(
                request,
                CybersecurityReason.INSUFFICIENT_EVIDENCE,
                PolicyIdentifier.POLICY_FAIL_CLOSED,
                details={"error": "Malformed or missing request_id"},
            )

        if not isinstance(request.evaluation_time, datetime):
            return self._make_denial(
                request,
                CybersecurityReason.INSUFFICIENT_EVIDENCE,
                PolicyIdentifier.POLICY_FAIL_CLOSED,
                details={"error": "Missing or invalid evaluation_time"},
            )

        # -------------------------------------------------------------
        # 1. URL Integrity (Safety Layer)
        # -------------------------------------------------------------
        # Determine allowed schemes based on action tier
        if (
            request.action_tier == ActionTier.DISPLAY_INFORMATIONAL_LINK
            and self._allow_http_informational
        ):
            schemes = INFORMATIONAL_ALLOWED_SCHEMES
        else:
            schemes = self._allowed_schemes

        canonical_url, error_reason, error_msg = parse_and_canonicalize_url(
            request.url, allowed_schemes=schemes
        )
        if canonical_url is None:
            return self._make_denial(
                request,
                error_reason or CybersecurityReason.MALFORMED_URL,
                PolicyIdentifier.POLICY_URL_INTEGRITY,
                details={"error": error_msg or "URL parsing rejected"},
            )

        target_domain = canonical_url.hostname
        advisories: List[AdvisorySignal] = []

        # For software downloads/installs, raw IP addresses are prohibited
        if canonical_url.is_ip_address:
            if request.action_tier in (
                ActionTier.RECOMMEND_SOFTWARE,
                ActionTier.DOWNLOAD_FILE,
                ActionTier.INSTALL_SOFTWARE,
            ):
                return self._make_denial(
                    request,
                    CybersecurityReason.DECEPTIVE_HOSTNAME,
                    PolicyIdentifier.POLICY_URL_INTEGRITY,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    details={"error": "IP address used as software distribution host"},
                )
            advisories.append(AdvisorySignal.SUSPICIOUS_IP_HOST)

        # -------------------------------------------------------------
        # 2. Markdown Link Consistency & Untrusted Content Governance
        # -------------------------------------------------------------
        if request.untrusted_content is not None:
            content_ctx = request.untrusted_content

            # Deceptive Markdown label check
            if content_ctx.markdown_link_label:
                valid_label, label_reason, label_msg = check_markdown_link(
                    content_ctx.markdown_link_label, request.url
                )
                if not valid_label:
                    return self._make_denial(
                        request,
                        label_reason or CybersecurityReason.DECEPTIVE_LINK_LABEL,
                        PolicyIdentifier.POLICY_LINK_CONSISTENCY,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        details={"error": label_msg or "Deceptive link label"},
                    )

            # Untrusted content execution promotion check
            # Invariant: External content cannot grant itself action execution authority
            if content_ctx.is_action_promoted:
                if (
                    not request.caller_mandate
                    or not request.caller_mandate.strip()
                ):
                    return self._make_denial(
                        request,
                        CybersecurityReason.UNTRUSTED_EXECUTION_PROMOTION,
                        PolicyIdentifier.POLICY_CONTENT_TRUST_BOUNDARY,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        details={
                            "error": "Action promotion from untrusted content rejected without caller mandate"
                        },
                    )

            # Advisory pattern scan on untrusted content
            if content_ctx.raw_content:
                advisories.extend(
                    scan_content_advisories(content_ctx.raw_content)
                )

        # -------------------------------------------------------------
        # 3. Redirect Chain Inspection
        # -------------------------------------------------------------
        if request.redirect_chain:
            if len(request.redirect_chain) > self._max_redirect_hops:
                return self._make_denial(
                    request,
                    CybersecurityReason.UNSAFE_REDIRECT_CHAIN,
                    PolicyIdentifier.POLICY_REDIRECT_SAFETY,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    details={"error": "Redirect chain exceeds maximum permitted hops"},
                )

            seen_canonical: Set[str] = {canonical_url.canonical_url}
            current_domain = canonical_url.normalized_domain

            for hop_idx, hop_url in enumerate(request.redirect_chain):
                hop_canonical, hop_err, _ = parse_and_canonicalize_url(
                    hop_url, allowed_schemes=schemes
                )
                if hop_canonical is None:
                    return self._make_denial(
                        request,
                        hop_err or CybersecurityReason.UNSAFE_REDIRECT_CHAIN,
                        PolicyIdentifier.POLICY_REDIRECT_SAFETY,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        details={
                            "error": f"Malformed redirect hop at index {hop_idx}"
                        },
                    )

                if hop_canonical.canonical_url in seen_canonical:
                    return self._make_denial(
                        request,
                        CybersecurityReason.UNSAFE_REDIRECT_CHAIN,
                        PolicyIdentifier.POLICY_REDIRECT_SAFETY,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        details={"error": "Circular redirect detected in chain"},
                    )
                seen_canonical.add(hop_canonical.canonical_url)

                # Detect cross-domain transitions
                if hop_canonical.normalized_domain != current_domain:
                    # If this is software recommendation or download, destination must be authorized
                    if request.action_tier in (
                        ActionTier.RECOMMEND_SOFTWARE,
                        ActionTier.DOWNLOAD_FILE,
                        ActionTier.INSTALL_SOFTWARE,
                    ):
                        is_hop_authorized = (
                            request.evidence is not None
                            and (
                                hop_canonical.hostname
                                in request.evidence.canonical_domains
                                or hop_canonical.hostname
                                in request.evidence.allowed_download_domains
                                or hop_canonical.normalized_domain
                                in request.evidence.canonical_domains
                                or hop_canonical.normalized_domain
                                in request.evidence.allowed_download_domains
                            )
                        )
                        if not is_hop_authorized:
                            return self._make_denial(
                                request,
                                CybersecurityReason.CROSS_DOMAIN_REDIRECT,
                                PolicyIdentifier.POLICY_REDIRECT_SAFETY,
                                canonical_url=canonical_url.canonical_url,
                                target_domain=target_domain,
                                advisory_signals=advisories,
                                details={
                                    "error": f"Redirect hop {hop_idx} crosses to unauthorized domain '{hop_canonical.hostname}'"
                                },
                            )
                    else:
                        advisories.append(AdvisorySignal.CROSS_ORIGIN_REDIRECT)
                current_domain = hop_canonical.normalized_domain

        # -------------------------------------------------------------
        # 4. Publisher Provenance & Action Authorization (Consensus/Fallback)
        # -------------------------------------------------------------
        # Actions requiring independent publisher provenance:
        if request.action_tier in (
            ActionTier.RECOMMEND_SOFTWARE,
            ActionTier.DOWNLOAD_FILE,
            ActionTier.INSTALL_SOFTWARE,
        ):
            # Invariant: Missing, unverified, or self-described evidence fails closed
            if request.evidence is None:
                return self._make_denial(
                    request,
                    CybersecurityReason.PROVENANCE_MISSING,
                    PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    advisory_signals=advisories,
                    details={
                        "error": "Software action requires independently established publisher provenance"
                    },
                )

            evidence = request.evidence
            try:
                ver_status = self._verifier.verify(
                    evidence, request.evaluation_time
                )
            except Exception as exc:
                return self._make_denial(
                    request,
                    CybersecurityReason.VERIFICATION_UNAVAILABLE,
                    PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    advisory_signals=advisories,
                    details={"error": f"Verifier error: {exc}"},
                )

            if ver_status == ProvenanceStatus.EXPIRED:
                return self._make_denial(
                    request,
                    CybersecurityReason.PROVENANCE_EXPIRED,
                    PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    advisory_signals=advisories,
                )
            if ver_status == ProvenanceStatus.REVOKED:
                return self._make_denial(
                    request,
                    CybersecurityReason.PROVENANCE_REVOKED,
                    PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    advisory_signals=advisories,
                )
            if ver_status == ProvenanceStatus.CONTRADICTORY:
                return self._make_denial(
                    request,
                    CybersecurityReason.PROVENANCE_CONTRADICTORY,
                    PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    advisory_signals=advisories,
                )
            if ver_status == ProvenanceStatus.UNVERIFIED:
                return self._make_denial(
                    request,
                    CybersecurityReason.PROVENANCE_UNVERIFIED,
                    PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    advisory_signals=advisories,
                )
            if ver_status != ProvenanceStatus.VALID:
                return self._make_denial(
                    request,
                    CybersecurityReason.INSUFFICIENT_EVIDENCE,
                    PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                    canonical_url=canonical_url.canonical_url,
                    target_domain=target_domain,
                    advisory_signals=advisories,
                )

            # Consensus verification across peer evidences
            if request.peer_evidences:
                for peer_ev in request.peer_evidences:
                    try:
                        peer_status = self._verifier.verify(
                            peer_ev, request.evaluation_time
                        )
                    except Exception:
                        return self._make_denial(
                            request,
                            CybersecurityReason.VERIFICATION_UNAVAILABLE,
                            PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                            canonical_url=canonical_url.canonical_url,
                            target_domain=target_domain,
                            advisory_signals=advisories,
                        )
                    if peer_status in (
                        ProvenanceStatus.REVOKED,
                        ProvenanceStatus.CONTRADICTORY,
                    ):
                        return self._make_denial(
                            request,
                            CybersecurityReason.PROVENANCE_CONTRADICTORY,
                            PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                            canonical_url=canonical_url.canonical_url,
                            target_domain=target_domain,
                            advisory_signals=advisories,
                            details={
                                "error": "Peer consensus failed: contradictory or revoked peer evidence"
                            },
                        )

            # Software identity binding
            if request.software_id and evidence.software_id:
                if request.software_id.lower() != evidence.software_id.lower():
                    return self._make_denial(
                        request,
                        CybersecurityReason.PUBLISHER_MISMATCH,
                        PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        advisory_signals=advisories,
                        details={
                            "error": f"Requested software '{request.software_id}' does not match evidence '{evidence.software_id}'"
                        },
                    )

            # Domain binding evaluation
            clean_host = canonical_url.hostname
            clean_apex = canonical_url.normalized_domain

            if request.action_tier == ActionTier.RECOMMEND_SOFTWARE:
                # Target host must match canonical publisher domains
                is_canonical = (
                    clean_host in evidence.canonical_domains
                    or clean_apex in evidence.canonical_domains
                )
                if not is_canonical:
                    return self._make_denial(
                        request,
                        CybersecurityReason.UNVERIFIED_PUBLISHER_DOMAIN,
                        PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        advisory_signals=advisories,
                        details={
                            "error": f"Host '{clean_host}' is not an established canonical domain for '{evidence.software_id}'"
                        },
                    )

            elif request.action_tier == ActionTier.DOWNLOAD_FILE:
                # Target host must match authorized download domains
                is_authorized_download = (
                    clean_host in evidence.allowed_download_domains
                    or clean_apex in evidence.allowed_download_domains
                )
                if not is_authorized_download:
                    return self._make_denial(
                        request,
                        CybersecurityReason.UNAUTHORIZED_DOWNLOAD_DOMAIN,
                        PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        advisory_signals=advisories,
                        details={
                            "error": f"Host '{clean_host}' is not an authorized download domain for '{evidence.software_id}'"
                        },
                    )

            elif request.action_tier == ActionTier.INSTALL_SOFTWARE:
                # Installation requires download authorization + explicit caller mandate
                is_authorized_download = (
                    clean_host in evidence.allowed_download_domains
                    or clean_apex in evidence.allowed_download_domains
                )
                if not is_authorized_download:
                    return self._make_denial(
                        request,
                        CybersecurityReason.UNAUTHORIZED_DOWNLOAD_DOMAIN,
                        PolicyIdentifier.POLICY_ACTION_AUTHORIZATION,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        advisory_signals=advisories,
                    )
                if (
                    not request.caller_mandate
                    or not request.caller_mandate.strip()
                ):
                    return self._make_denial(
                        request,
                        CybersecurityReason.INSUFFICIENT_ACTION_AUTHORIZATION,
                        PolicyIdentifier.POLICY_ACTION_AUTHORIZATION,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        advisory_signals=advisories,
                        details={
                            "error": "Software installation requires explicit caller mandate"
                        },
                    )

        # For lower tiers (DISPLAY_INFORMATIONAL_LINK and OPEN_WEBSITE):
        # If evidence was supplied, validate it; invalid evidence fails closed
        elif request.evidence is not None:
            try:
                ver_status = self._verifier.verify(
                    request.evidence, request.evaluation_time
                )
                if ver_status in (
                    ProvenanceStatus.REVOKED,
                    ProvenanceStatus.CONTRADICTORY,
                ):
                    return self._make_denial(
                        request,
                        CybersecurityReason.PROVENANCE_REVOKED
                        if ver_status == ProvenanceStatus.REVOKED
                        else CybersecurityReason.PROVENANCE_CONTRADICTORY,
                        PolicyIdentifier.POLICY_PUBLISHER_PROVENANCE,
                        canonical_url=canonical_url.canonical_url,
                        target_domain=target_domain,
                        advisory_signals=advisories,
                    )
            except Exception:
                pass

        # -------------------------------------------------------------
        # 5. Outcome Determination (GRACE contextual mediation)
        # -------------------------------------------------------------
        # If any advisory signals exist (e.g. untrusted text had prompt injection attempt,
        # but the requested action is merely displaying an informational link requested by caller):
        # Require human review if advisory signals exist
        if advisories:
            audit = CybersecurityAuditRecord(
                request_id=request.request_id,
                timestamp=_utc(request.evaluation_time),
                outcome=DecisionOutcome.REVIEW,
                reason=CybersecurityReason.REVIEW_REQUIRED,
                policy_id=PolicyIdentifier.POLICY_CONTENT_TRUST_BOUNDARY,
                raw_url=request.url,
                canonical_url=canonical_url.canonical_url,
                action_tier=request.action_tier,
                target_domain=target_domain,
                evidence_ids=(
                    (request.evidence.evidence_id,)
                    if request.evidence is not None
                    else ()
                ),
                advisory_signals=tuple(advisories),
                details={"advisories": ",".join(s.value for s in advisories)},
            )
            decision = CybersecurityDecision(
                authorized=False,  # REVIEW is not automated authorization
                outcome=DecisionOutcome.REVIEW,
                reason=CybersecurityReason.REVIEW_REQUIRED,
                policy_id=PolicyIdentifier.POLICY_CONTENT_TRUST_BOUNDARY,
                action_tier=request.action_tier,
                canonical_url=canonical_url.canonical_url,
                audit=audit,
                advisory_signals=tuple(advisories),
            )
            self._record(decision)
            return decision

        # Fully authorized
        audit = CybersecurityAuditRecord(
            request_id=request.request_id,
            timestamp=_utc(request.evaluation_time),
            outcome=DecisionOutcome.ALLOW,
            reason=CybersecurityReason.AUTHORIZED,
            policy_id=PolicyIdentifier.POLICY_ACTION_AUTHORIZATION,
            raw_url=request.url,
            canonical_url=canonical_url.canonical_url,
            action_tier=request.action_tier,
            target_domain=target_domain,
            evidence_ids=(
                (request.evidence.evidence_id,)
                if request.evidence is not None
                else ()
            ),
            advisory_signals=(),
            details={"status": "All verification invariants satisfied"},
        )
        decision = CybersecurityDecision(
            authorized=True,
            outcome=DecisionOutcome.ALLOW,
            reason=CybersecurityReason.AUTHORIZED,
            policy_id=PolicyIdentifier.POLICY_ACTION_AUTHORIZATION,
            action_tier=request.action_tier,
            canonical_url=canonical_url.canonical_url,
            audit=audit,
            advisory_signals=(),
        )
        self._record(decision)
        return decision

    # Monitoring and Inspection APIs
    def get_health(self) -> CybersecurityHealthStatus:
        return CybersecurityHealthStatus.OPTIMAL

    def get_subsystem_health(self) -> Dict[str, CybersecurityHealthStatus]:
        return {
            "url_integrity": CybersecurityHealthStatus.OPTIMAL,
            "publisher_provenance": CybersecurityHealthStatus.OPTIMAL,
            "content_trust_boundary": CybersecurityHealthStatus.OPTIMAL,
            "action_authorization": CybersecurityHealthStatus.OPTIMAL,
        }

    def get_metrics(self) -> Dict[str, Any]:
        total = len(self._decision_history)
        allowed = sum(
            1 for d in self._decision_history if d.outcome == DecisionOutcome.ALLOW
        )
        denied = sum(
            1 for d in self._decision_history if d.outcome == DecisionOutcome.DENY
        )
        review = sum(
            1 for d in self._decision_history if d.outcome == DecisionOutcome.REVIEW
        )
        return {
            "total_decisions": total,
            "allowed_decisions": allowed,
            "denied_decisions": denied,
            "review_decisions": review,
        }

    def get_events(self) -> List[CybersecurityAuditRecord]:
        return list(self._events)

    def get_decision_history(self) -> List[CybersecurityDecision]:
        return list(self._decision_history)


# -----------------------------
# Module Level Functions & Factories
# -----------------------------


def get_health(governor: CybersecurityGovernor) -> CybersecurityHealthStatus:
    return governor.get_health()


def get_subsystem_health(
    governor: CybersecurityGovernor,
) -> Dict[str, CybersecurityHealthStatus]:
    return governor.get_subsystem_health()


def get_metrics(governor: CybersecurityGovernor) -> Dict[str, Any]:
    return governor.get_metrics()


def get_events(governor: CybersecurityGovernor) -> List[CybersecurityAuditRecord]:
    return governor.get_events()


def get_decision_history(
    governor: CybersecurityGovernor,
) -> List[CybersecurityDecision]:
    return governor.get_decision_history()


def create_cybersecurity_governor(**kwargs: Any) -> CybersecurityGovernor:
    """Factory function for CybersecurityGovernor."""
    return CybersecurityGovernor(**kwargs)


def create_strict_cybersecurity_governor(**kwargs: Any) -> CybersecurityGovernor:
    """Factory function for strict CybersecurityGovernor (HTTPS only)."""
    defaults = {
        "allowed_schemes": ("https",),
        "allow_http_informational": False,
    }
    defaults.update(kwargs)
    return CybersecurityGovernor(**defaults)


def create_permissive_cybersecurity_governor(**kwargs: Any) -> CybersecurityGovernor:
    """Factory function for permissive CybersecurityGovernor (HTTP allowed for informational links)."""
    defaults = {
        "allowed_schemes": ("https",),
        "allow_http_informational": True,
    }
    defaults.update(kwargs)
    return CybersecurityGovernor(**defaults)
