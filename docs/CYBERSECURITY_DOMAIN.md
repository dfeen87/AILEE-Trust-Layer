# AILEE Cybersecurity Trust Domain: Link Provenance & Injection Governance (v10.1.0)

## 1. Purpose, Scope, and Claim Level

The **Cybersecurity Trust Domain** (`ailee.domains.cybersecurity`) provides deterministic, side-effect-free governance of external URLs, software-download recommendations, link provenance, and agent actions influenced by untrusted external content.

This domain was motivated by a real-world failure mode where an AI assistant recommended an unverified software distribution website (`pear-cleaner.com` rather than the developer's authentic source `itsalin.com`), leading to an unexpected DMG download and a macOS malware alert.

Status terms used here (following AILEE standard conventions):
- **IMPLEMENTED** — Behavior present in code and exercised by repository tests.
- **CONFIGURABLE** — Behavior supplied through constructor parameters or verifier protocols.
- **EXTERNAL RESPONSIBILITY** — Collection or cryptographic generation of publisher provenance evidence and caller mandates.
- **OUT OF SCOPE** — Dynamic runtime sandboxing, kernel-level process isolation, or active network packet sniffing.

**Core Invariant:** AILEE governs whether an external recommendation or agent action is authorized; it never assumes that an AI-generated link is trustworthy simply because the AI generated it.

---

## 2. Threat Model & v10.1.0 Security Hardening

### 2.1 Threat Vectors & Defense Invariants

| Threat Vector | Description | Attacker Objective | AILEE Defense / Policy |
| :--- | :--- | :--- | :--- |
| **Lookalike / Typosquatting Domains** | Domains that visually resemble legitimate software projects (e.g. `pear-cleaner.com` vs `itsalin.com`). | Distribute modified binaries, Trojans, or bundleware under the guise of an authentic project. | `POLICY_PUBLISHER_PROVENANCE`: Requires independently verified publisher provenance via `TrustedProvenanceVerifier`. Invariant: Domain resemblance, HTTPS, search ranking, and website self-description never establish authenticity. |
| **Execution Target Substitution** | Passing one authorized URL in the request while passing a different malicious URL to the underlying tool. | Confused deputy attack bypassing authorization checks at execution time. | Strict parameter binding in `CybersecurityActionWrapper`: Compares all execution arguments against the authorized target; rejects mismatches with `PARAMETER_MISMATCH`. |
| **Shadow Mode / Review Bypass** | Attempting to use `shadow_mode=True` or `require_confirmation_for_review=False` to execute unverified downloads or installs. | Execute unauthorized high-risk side effects under observation or permissive settings. | Hard enforcement invariant: `shadow_mode` and disabled review confirmation can never bypass `DENY` or `REVIEW` for high-risk actions. |
| **Forged / Self-Asserted Provenance** | Submitting evidence marked `VALID` without authentic cryptographic signature or from an unknown issuer. | Trick the governor into accepting attacker-fabricated publisher claims. | `TrustedProvenanceVerifier` enforces authentic verifier identities from recognized trust registries and cryptographic signatures/digests. |
| **Cross-Boundary Redirect Chains** | An initial request to an authorized domain that hops through intermediate untrusted hosts to a payload. | Exploit initial domain trust to deliver unverified payloads. | `POLICY_REDIRECT_SAFETY`: Independently validates every single intermediate hop and final destination against exact authorized publisher hosts. |
| **Unauthenticated Installation Requests** | Submitting arbitrary strings or tokens to authorize local binary installation. | Elevate agent privileges to execute local software installations without user authority. | `POLICY_CALLER_MANDATE`: Requires an authenticated, scoped, time-bound `ActionMandate` binding caller, tier, software ID, target URL, and signature. |
| **Deceptive Markdown Links** | Markdown link syntax where the visible anchor label is a URL differing from the href destination. | Trick users or automated reviewers into clicking an unverified destination. | `POLICY_LINK_CONSISTENCY`: Compares parsed hostnames between anchor label and href destination; fails closed with `DECEPTIVE_LINK_LABEL`. |
| **Embedded URL Credentials** | URLs containing userinfo in authority/netloc (e.g. `https://user:password@target.com/`). | Phishing, credential leakage, or abusing HTTP authentication semantics. | `POLICY_URL_INTEGRITY`: Strict parsing identifies credentials in netloc and fails closed with `EMBEDDED_CREDENTIALS`. |
| **IDN Homograph & Punycode Spoofing** | Use of Punycode (`xn--`), mixed scripts (Cyrillic/Greek homoglyphs), zero-width spaces, or bidi overrides. | Visually deceive reviewers or bypass naive string-matching filters. | `POLICY_URL_INTEGRITY`: Comprehensive rejection of Punycode, non-ASCII confusables, zero-width spaces, and bidi override formatting (`IDN_HOMOGRAPH_ATTEMPT`). |
| **Misleading Subdomain Structures** | Embedding TLDs or authentic brand names into subdomains (e.g. `itsalin.com.attacker.com`). | Create visual authority while routing traffic to an attacker server. | `POLICY_URL_INTEGRITY`: Inspects subdomain labels against TLD hierarchies and hyphenated delimiters (`MISLEADING_SUBDOMAIN`). |
| **Prompt Injection in External Content** | Retrieved webpages, search snippets, or tool outputs containing execution overrides. | Hijack agent execution flow and trigger unauthorized tools. | `POLICY_CONTENT_TRUST_BOUNDARY`: External content is strictly treated as data; advisory scanning detects injections; action parameter promotion without caller mandate fails closed (`UNTRUSTED_EXECUTION_PROMOTION`). |

---

## 3. Architecture & Data Model

```
External URL / Action Request + Untrusted Content + Provenance Evidence
                                 |
                                 v
                 [ Safety Layer: URL & Syntax Invariants ]
                 - Scheme allowlist (HTTPS enforced for software)
                 - Embedded credentials rejection
                 - Punycode & IDN homoglyph rejection
                 - Misleading subdomain rejection
                 - Deceptive markdown link check
                                 |
                                 v
                 [ Content Trust Boundary Layer ]
                 - Treat external content strictly as data
                 - Block action promotion without caller mandate
                 - Advisory pattern scanning
                                 |
                                 v
                 [ Provenance & Consensus Layer ]
                 - Validate independent PublisherProvenanceEvidence
                 - Validate half-open validity window [valid_from, valid_until)
                 - Match software_id and canonical/download domains
                 - Multi-source peer consensus (contradiction rejection)
                                 |
                                 v
                 [ Tiered Action Authorization Layer ]
                 - DISPLAY_INFORMATIONAL_LINK -> Clean URL
                 - OPEN_WEBSITE -> HTTPS + clean host
                 - RECOMMEND_SOFTWARE -> Verified canonical domain
                 - DOWNLOAD_FILE -> Verified download domain
                 - INSTALL_SOFTWARE -> Verified domain + explicit caller mandate
                                 |
                                 v
                       CybersecurityDecision
                    (ALLOW / REVIEW / DENY)
                                 +
                      CybersecurityAuditRecord
```

### 3.1 Action Tiers

The domain categorizes actions into 5 progressively governed risk tiers:

1. `DISPLAY_INFORMATIONAL_LINK` (Low Risk): Presenting an informational reference link to the user. Requires URL integrity and consistent Markdown labels.
2. `OPEN_WEBSITE` (Moderate Risk): Opening a remote webpage in a browser or web client. Requires HTTPS and safe domain structure.
3. `RECOMMEND_SOFTWARE` (Substantial Risk): Advising a user to obtain or use a specific software application. **Requires independently established publisher provenance matching the destination domain.**
4. `DOWNLOAD_FILE` (High Risk): Fetching a binary, package, installer (DMG, PKG, EXE, ZIP). Requires valid publisher provenance and destination domain matching `allowed_download_domains`.
5. `INSTALL_SOFTWARE` (Critical Risk): Executing an installer or launching downloaded software. Requires download authorization plus an explicit `caller_mandate` granting execution privileges.

### 3.2 Decision Outcomes and Stable Reason Codes

Decisions return one of three explicit outcomes:
- `ALLOW`: All verification invariants and evidence criteria are satisfied.
- `REVIEW`: Hard invariants are satisfied, but advisory content signals (e.g. prompt injection phrases in search snippets) require human review before proceeding.
- `DENY`: Security invariant violated, provenance missing or contradictory, or unauthorized domain.

Stable reason codes include:
- `AUTHORIZED`
- `REVIEW_REQUIRED`
- `MALFORMED_URL`
- `URL_TOO_LONG`
- `UNSUPPORTED_SCHEME`
- `EMBEDDED_CREDENTIALS`
- `IDN_HOMOGRAPH_ATTEMPT`
- `DECEPTIVE_HOSTNAME`
- `MISLEADING_SUBDOMAIN`
- `DECEPTIVE_LINK_LABEL`
- `UNSAFE_REDIRECT_CHAIN`
- `CROSS_DOMAIN_REDIRECT`
- `PROVENANCE_MISSING`
- `PROVENANCE_NOT_ESTABLISHED`
- `PROVENANCE_EXPIRED`
- `PROVENANCE_NOT_YET_VALID`
- `PROVENANCE_REVOKED`
- `PROVENANCE_CONTRADICTORY`
- `PROVENANCE_UNVERIFIED`
- `UNVERIFIED_PUBLISHER_DOMAIN`
- `PUBLISHER_MISMATCH`
- `UNAUTHORIZED_DOWNLOAD_DOMAIN`
- `INSUFFICIENT_ACTION_AUTHORIZATION`
- `UNTRUSTED_EXECUTION_PROMOTION`
- `PROMPT_INJECTION_DETECTED`
- `INSUFFICIENT_EVIDENCE`
- `VERIFICATION_UNAVAILABLE`

---

## 4. Public API & Usage Examples

### 4.1 Recommending Software (The Pearcleaner Scenario)

```python
from datetime import datetime, timezone
from ailee.domains.cybersecurity import (
    CybersecurityGovernor,
    CybersecurityAuthorizationRequest,
    PublisherProvenanceEvidence,
    ActionTier,
    ProvenanceStatus,
    DecisionOutcome,
)

now = datetime.now(timezone.utc)
governor = CybersecurityGovernor()

# 1. Without independent evidence, recommending 'https://pear-cleaner.com' fails closed:
request = CybersecurityAuthorizationRequest(
    request_id="req-rec-1",
    action_tier=ActionTier.RECOMMEND_SOFTWARE,
    url="https://pear-cleaner.com/",
    software_id="pearcleaner",
    evaluation_time=now,
    evidence=None,
)
decision = governor.authorize(request)
assert decision.outcome == DecisionOutcome.DENY
assert decision.reason.value == "PROVENANCE_MISSING"

# 2. When authentic publisher evidence for Pearcleaner is supplied:
evidence = PublisherProvenanceEvidence(
    evidence_id="ev-pearcleaner-2026",
    software_id="pearcleaner",
    publisher_id="alin_panaitiu",
    publisher_name="Alin Panaitiu",
    canonical_domains=("itsalin.com",),
    allowed_download_domains=("itsalin.com", "github.com"),
    status=ProvenanceStatus.VALID,
    valid_from=now,
    valid_until=None,
    verified_at=now,
    verifier_identity="official-registry",
)

# Evaluating the lookalike domain against verified evidence:
decision_lookalike = governor.authorize(
    CybersecurityAuthorizationRequest(
        request_id="req-rec-2",
        action_tier=ActionTier.RECOMMEND_SOFTWARE,
        url="https://pear-cleaner.com/",
        software_id="pearcleaner",
        evaluation_time=now,
        evidence=evidence,
    )
)
# Denied deterministically without falsely labeling the domain malicious:
assert decision_lookalike.outcome == DecisionOutcome.DENY
assert decision_lookalike.reason.value == "UNVERIFIED_PUBLISHER_DOMAIN"

# Evaluating the authentic publisher domain:
decision_official = governor.authorize(
    CybersecurityAuthorizationRequest(
        request_id="req-rec-3",
        action_tier=ActionTier.RECOMMEND_SOFTWARE,
        url="https://itsalin.com/",
        software_id="pearcleaner",
        evaluation_time=now,
        evidence=evidence,
    )
)
assert decision_official.outcome == DecisionOutcome.ALLOW
assert decision_official.authorized is True
```

### 4.2 Intercepting Agent Tools with `CybersecurityActionWrapper`

```python
from ailee.domains.cybersecurity import (
    create_cybersecurity_action_wrapper,
    CybersecurityAuthorizationRequest,
    ActionTier,
)

def download_file_tool(url: str) -> str:
    # Physical download logic
    return f"Downloaded {url}"

wrapper = create_cybersecurity_action_wrapper(download_file_tool)

# If an unverified URL is requested, wrapper halts execution before the tool runs:
result = wrapper.execute(
    request=CybersecurityAuthorizationRequest(
        request_id="req-dl-1",
        action_tier=ActionTier.DOWNLOAD_FILE,
        url="https://pear-cleaner.com/Pearcleaner.dmg",
        software_id="pearcleaner",
        evaluation_time=now,
        evidence=evidence,  # pear-cleaner.com not in allowed_download_domains
    ),
    url="https://pear-cleaner.com/Pearcleaner.dmg",
)

assert result["blocked"] is True
assert result["result"] is None  # download_file_tool was never invoked!
```

---

## 5. Offline Adversarial Verification

All tests are side-effect-free:
- **Zero network requests:** No sockets, HTTP requests, or DNS lookups are performed during evaluation or testing.
- **Pure offline test fixtures:** Lookalike domains, redirect chains, homoglyph attacks, and simulated prompt injections are executed in isolated test fixtures (`tests/test_cybersecurity_adversarial.py`).
- **Pearcleaner incident regression fixture:** Validates the exact scenario that motivated this capability, demonstrating fail-closed rejection of unverified software distribution websites.

---

## 6. Limitations

1. **Deterministic Static Governance:** The governor does not perform runtime binary analysis, dynamic malware detonating, or real-time network traffic inspection.
2. **Provenance Dependency:** Authenticity decisions depend on the validity and freshness of `PublisherProvenanceEvidence`. An independent adapter or trust registry (e.g., Sigstore, OS developer notarization) must supply this evidence.
3. **Prompt Injection as Advisory Signal:** Adversarial pattern matching on untrusted text is advisory. True security enforcement relies on the immutable parameter trust boundary (`UNTRUSTED_EXECUTION_PROMOTION`) and explicit caller mandates rather than heuristic regex sanitization.
4. **Replay & State:** The core governor is purely deterministic and stateless. Stateful request de-duplication or replay defense should be configured via an injected replay guard where necessary.
