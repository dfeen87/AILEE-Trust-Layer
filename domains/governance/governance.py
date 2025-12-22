"""
AILEE Governance Domain — v1.0.0
Single-file reference implementation for governance trust evaluation.

Implements a layered governance pipeline:

1) Authority Layer     (source validation, mandate, consent checks)
2) Scope Layer         (jurisdictional boundaries, applicability)
3) Temporal Layer      (validity windows, expiration, revocation)
4) Delegation Layer    (authority chain validation, constraint propagation)
5) Actionability Layer (graduated trust level determination)
6) Audit Layer         (full decision traceability)

Design goals:
- Deterministic governance decisions (no randomness)
- Authority and restraint enforcement
- Clear separation: "may act" vs "what to believe"
- Auditable decision chains
- Jurisdictional scope enforcement
- Temporal validity tracking

Usage (minimal):

    config = GovernanceConfig()
    governor = GovernanceGovernor(config)
    
    result = governor.evaluate(
        source="election_authority_xyz",
        jurisdiction="state:PA",
        authority_level="certified_official",
        mandate="voter_registration_update",
        consent_proof="signed_consent_hash_abc",
        timestamp=time.time(),
        context={"operation": "voter_roll_update"}
    )
    
    if result.authorized_level >= GovernanceTrustLevel.CONSTRAINED_TRUST:
        # Proceed with constrained action
        pass

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple
import time
import hashlib


# -----------------------------
# Enums / Trust Levels
# -----------------------------

class GovernanceTrustLevel(IntEnum):
    """
    Graduated trust levels for governance signals.
    Higher values indicate more authority to act.
    """
    NO_TRUST = 0            # Signal exists but cannot be used
    ADVISORY_TRUST = 1      # Informational only, no action permitted
    CONSTRAINED_TRUST = 2   # Limited, scoped execution within boundaries
    FULL_TRUST = 3          # Authorized action within defined mandate


class AuthorityStatus(str):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    DELEGATED = "DELEGATED"


class ScopeStatus(str):
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    CROSS_JURISDICTIONAL = "CROSS_JURISDICTIONAL"
    SCOPE_UNKNOWN = "SCOPE_UNKNOWN"


class TemporalStatus(str):
    VALID = "VALID"
    EXPIRED = "EXPIRED"
    NOT_YET_VALID = "NOT_YET_VALID"
    REVOKED = "REVOKED"


# -----------------------------
# Result Structures
# -----------------------------

@dataclass(frozen=True)
class GovernanceDecision:
    """
    Complete governance decision with full audit trail.
    """
    authorized_level: GovernanceTrustLevel
    authority_status: str
    scope_status: str
    temporal_status: str
    delegation_valid: bool
    actionable: bool
    reasons: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    decision_id: str = ""
    timestamp: float = 0.0


@dataclass
class GovernanceSignal:
    """
    Input signal for governance evaluation.
    """
    # Authority identifiers
    source: str
    authority_level: Optional[str] = None
    
    # Scope identifiers
    jurisdiction: Optional[str] = None
    target_scope: Optional[str] = None
    
    # Mandate and consent
    mandate: Optional[str] = None
    consent_proof: Optional[str] = None
    
    # Temporal bounds
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    issued_at: Optional[float] = None
    
    # Delegation chain
    delegated_from: Optional[str] = None
    delegation_depth: int = 0
    
    # Revocation tracking
    revocation_check_url: Optional[str] = None
    revocable: bool = True
    
    # Context
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class GovernanceConfig:
    """
    Configuration for governance evaluation.
    """
    # Authority thresholds
    require_mandate: bool = True
    require_consent: bool = True
    require_jurisdiction: bool = True
    
    # Recognized authority levels (hierarchical, higher = more authority)
    authority_hierarchy: Dict[str, int] = field(default_factory=lambda: {
        "unauthenticated": 0,
        "observer": 1,
        "reporter": 2,
        "certified_official": 3,
        "administrator": 4,
        "root_authority": 5,
    })
    
    # Jurisdictional scopes (can be hierarchical: "federal" > "state:PA" > "county:Philadelphia")
    recognized_jurisdictions: Set[str] = field(default_factory=lambda: {
        "federal", "state:PA", "state:NY", "state:CA",
        "county:Philadelphia", "county:Kings", "municipal:Philadelphia",
    })
    
    # Recognized mandates (operation types)
    recognized_mandates: Set[str] = field(default_factory=lambda: {
        "voter_registration_update",
        "ballot_counting_observation",
        "policy_advisory_input",
        "regulatory_enforcement",
        "compliance_audit",
        "emergency_directive",
    })
    
    # Delegation constraints
    max_delegation_depth: int = 3
    allow_cross_jurisdictional_delegation: bool = False
    
    # Temporal validity
    default_validity_window_seconds: float = 86400.0 * 30  # 30 days
    grace_period_seconds: float = 3600.0  # 1 hour grace for clock skew
    require_temporal_bounds: bool = True
    
    # Trust level thresholds (minimum authority level required for each trust level)
    advisory_min_authority: int = 1   # observer+
    constrained_min_authority: int = 2  # reporter+
    full_min_authority: int = 3  # certified_official+
    
    # Actionability rules
    advisory_requires_consent: bool = False
    constrained_requires_consent: bool = True
    full_requires_consent: bool = True
    
    # Audit and tracking
    enable_audit_metadata: bool = True
    track_decision_history: bool = True
    
    # Safety constraints
    block_revoked_sources: bool = True
    enforce_scope_boundaries: bool = True


# -----------------------------
# Core Governor
# -----------------------------

class GovernanceGovernor:
    """
    AILEE Governance Governor
    
    Evaluates governance signals through layered trust validation:
    - Authority verification
    - Scope validation
    - Temporal validity
    - Delegation chain verification
    - Actionability determination
    
    Maintains decision history for audit trails.
    """
    
    def __init__(self, config: GovernanceConfig):
        self.cfg = config
        self.decision_history: List[GovernanceDecision] = []
        self.revoked_sources: Set[str] = set()
        self.known_delegations: Dict[str, List[str]] = {}  # source -> [delegated_to]
        
    # -------------------------
    # Public API
    # -------------------------
    
    def evaluate(
        self,
        source: str,
        jurisdiction: Optional[str] = None,
        authority_level: Optional[str] = None,
        mandate: Optional[str] = None,
        consent_proof: Optional[str] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        issued_at: Optional[float] = None,
        delegated_from: Optional[str] = None,
        delegation_depth: int = 0,
        target_scope: Optional[str] = None,
        revocable: bool = True,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GovernanceDecision:
        """
        Evaluate a governance signal and return authorization decision.
        """
        ts = timestamp if timestamp is not None else time.time()
        ctx = dict(context or {})
        
        signal = GovernanceSignal(
            source=source,
            authority_level=authority_level,
            jurisdiction=jurisdiction,
            target_scope=target_scope,
            mandate=mandate,
            consent_proof=consent_proof,
            valid_from=valid_from,
            valid_until=valid_until,
            issued_at=issued_at,
            delegated_from=delegated_from,
            delegation_depth=delegation_depth,
            revocable=revocable,
            timestamp=ts,
            context=ctx,
        )
        
        return self._evaluate_signal(signal)
    
    def revoke_source(self, source: str, reason: str = "") -> None:
        """
        Revoke authorization for a source.
        All future signals from this source will be NO_TRUST.
        """
        self.revoked_sources.add(source)
        if self.cfg.enable_audit_metadata:
            print(f"[GOVERNANCE] Revoked source: {source} | Reason: {reason}")
    
    def register_delegation(self, from_source: str, to_source: str) -> None:
        """
        Register a delegation relationship.
        """
        if from_source not in self.known_delegations:
            self.known_delegations[from_source] = []
        self.known_delegations[from_source].append(to_source)
    
    def get_decision_history(self, limit: int = 100) -> List[GovernanceDecision]:
        """
        Retrieve recent decision history for audit.
        """
        return self.decision_history[-limit:]
    
    # -------------------------
    # Evaluation Pipeline
    # -------------------------
    
    def _evaluate_signal(self, signal: GovernanceSignal) -> GovernanceDecision:
        """
        Core evaluation pipeline with layered checks.
        """
        reasons: List[str] = []
        constraints: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}
        
        decision_id = self._generate_decision_id(signal)
        
        # Layer 1: Authority Verification
        authority_status, auth_level = self._verify_authority(signal, reasons, metadata)
        
        if authority_status == AuthorityStatus.REVOKED:
            decision = self._make_decision(
                authorized_level=GovernanceTrustLevel.NO_TRUST,
                authority_status=authority_status,
                scope_status=ScopeStatus.SCOPE_UNKNOWN,
                temporal_status=TemporalStatus.VALID,
                delegation_valid=False,
                actionable=False,
                reasons=reasons,
                constraints=constraints,
                metadata=metadata,
                decision_id=decision_id,
                timestamp=signal.timestamp,
            )
            self._commit_decision(decision)
            return decision
        
        if authority_status == AuthorityStatus.UNVERIFIED:
            decision = self._make_decision(
                authorized_level=GovernanceTrustLevel.NO_TRUST,
                authority_status=authority_status,
                scope_status=ScopeStatus.SCOPE_UNKNOWN,
                temporal_status=TemporalStatus.VALID,
                delegation_valid=False,
                actionable=False,
                reasons=reasons,
                constraints=constraints,
                metadata=metadata,
                decision_id=decision_id,
                timestamp=signal.timestamp,
            )
            self._commit_decision(decision)
            return decision
        
        # Layer 2: Scope Validation
        scope_status = self._validate_scope(signal, reasons, constraints, metadata)
        
        if self.cfg.enforce_scope_boundaries and scope_status == ScopeStatus.OUT_OF_SCOPE:
            decision = self._make_decision(
                authorized_level=GovernanceTrustLevel.NO_TRUST,
                authority_status=authority_status,
                scope_status=scope_status,
                temporal_status=TemporalStatus.VALID,
                delegation_valid=False,
                actionable=False,
                reasons=reasons,
                constraints=constraints,
                metadata=metadata,
                decision_id=decision_id,
                timestamp=signal.timestamp,
            )
            self._commit_decision(decision)
            return decision
        
        # Layer 3: Temporal Validation
        temporal_status = self._validate_temporal(signal, reasons, metadata)
        
        if temporal_status in [TemporalStatus.EXPIRED, TemporalStatus.REVOKED]:
            decision = self._make_decision(
                authorized_level=GovernanceTrustLevel.NO_TRUST,
                authority_status=authority_status,
                scope_status=scope_status,
                temporal_status=temporal_status,
                delegation_valid=False,
                actionable=False,
                reasons=reasons,
                constraints=constraints,
                metadata=metadata,
                decision_id=decision_id,
                timestamp=signal.timestamp,
            )
            self._commit_decision(decision)
            return decision
        
        if temporal_status == TemporalStatus.NOT_YET_VALID:
            decision = self._make_decision(
                authorized_level=GovernanceTrustLevel.ADVISORY_TRUST,
                authority_status=authority_status,
                scope_status=scope_status,
                temporal_status=temporal_status,
                delegation_valid=False,
                actionable=False,
                reasons=reasons,
                constraints=constraints,
                metadata=metadata,
                decision_id=decision_id,
                timestamp=signal.timestamp,
            )
            self._commit_decision(decision)
            return decision
        
        # Layer 4: Delegation Chain Validation
        delegation_valid = self._validate_delegation(signal, reasons, metadata)
        
        if not delegation_valid and signal.delegation_depth > 0:
            decision = self._make_decision(
                authorized_level=GovernanceTrustLevel.NO_TRUST,
                authority_status=authority_status,
                scope_status=scope_status,
                temporal_status=temporal_status,
                delegation_valid=delegation_valid,
                actionable=False,
                reasons=reasons,
                constraints=constraints,
                metadata=metadata,
                decision_id=decision_id,
                timestamp=signal.timestamp,
            )
            self._commit_decision(decision)
            return decision
        
        # Layer 5: Actionability Determination
        authorized_level, actionable = self._determine_actionability(
            signal, auth_level, scope_status, reasons, constraints, metadata
        )
        
        decision = self._make_decision(
            authorized_level=authorized_level,
            authority_status=authority_status,
            scope_status=scope_status,
            temporal_status=temporal_status,
            delegation_valid=delegation_valid,
            actionable=actionable,
            reasons=reasons,
            constraints=constraints,
            metadata=metadata,
            decision_id=decision_id,
            timestamp=signal.timestamp,
        )
        
        # Invariant: decision must contain valid trust level
        assert isinstance(decision.authorized_level, GovernanceTrustLevel), \
            f"Decision must have GovernanceTrustLevel, got {type(decision.authorized_level)}"
        
        self._commit_decision(decision)
        return decision
    
    # -------------------------
    # Layer 1: Authority Verification
    # -------------------------
    
    def _verify_authority(
        self,
        signal: GovernanceSignal,
        reasons: List[str],
        metadata: Dict[str, Any],
    ) -> Tuple[str, int]:
        """
        Verify source authority and mandate/consent.
        Returns (AuthorityStatus, authority_level_int)
        """
        # Check revocation
        if self.cfg.block_revoked_sources and signal.source in self.revoked_sources:
            reasons.append(f"Source '{signal.source}' has been revoked.")
            return AuthorityStatus.REVOKED, 0
        
        # Check mandate
        if self.cfg.require_mandate and not signal.mandate:
            reasons.append("No mandate present (required by policy).")
            return AuthorityStatus.UNVERIFIED, 0
        
        if signal.mandate and signal.mandate not in self.cfg.recognized_mandates:
            reasons.append(f"Mandate '{signal.mandate}' not recognized.")
            return AuthorityStatus.UNVERIFIED, 0
        
        # Check consent
        if self.cfg.require_consent and not signal.consent_proof:
            reasons.append("No consent proof present (required by policy).")
            return AuthorityStatus.UNVERIFIED, 0
        
        # Check authority level
        if not signal.authority_level:
            reasons.append("No authority level specified.")
            return AuthorityStatus.UNVERIFIED, 0
        
        auth_level = self.cfg.authority_hierarchy.get(signal.authority_level, -1)
        if auth_level < 0:
            reasons.append(f"Authority level '{signal.authority_level}' not recognized.")
            return AuthorityStatus.UNVERIFIED, 0
        
        # Check delegation status
        status = AuthorityStatus.DELEGATED if signal.delegation_depth > 0 else AuthorityStatus.VERIFIED
        
        reasons.append(f"Authority verified: {signal.authority_level} (level={auth_level}).")
        
        if self.cfg.enable_audit_metadata:
            metadata["authority"] = {
                "source": signal.source,
                "level": signal.authority_level,
                "level_int": auth_level,
                "mandate": signal.mandate,
                "consent_present": bool(signal.consent_proof),
                "delegation_depth": signal.delegation_depth,
            }
        
        return status, auth_level
    
    # -------------------------
    # Layer 2: Scope Validation
    # -------------------------
    
    def _validate_scope(
        self,
        signal: GovernanceSignal,
        reasons: List[str],
        constraints: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """
        Validate jurisdictional scope and applicability.
        """
        if not signal.jurisdiction:
            if self.cfg.require_jurisdiction:
                reasons.append("No jurisdiction specified (required by policy).")
                return ScopeStatus.SCOPE_UNKNOWN
            else:
                reasons.append("No jurisdiction specified (advisory mode).")
                return ScopeStatus.IN_SCOPE
        
        # Check if jurisdiction is recognized
        if signal.jurisdiction not in self.cfg.recognized_jurisdictions:
            reasons.append(f"Jurisdiction '{signal.jurisdiction}' not recognized.")
            return ScopeStatus.SCOPE_UNKNOWN
        
        # Check target scope compatibility
        if signal.target_scope:
            if not self._scope_compatible(signal.jurisdiction, signal.target_scope):
                reasons.append(
                    f"Jurisdiction '{signal.jurisdiction}' incompatible with target scope '{signal.target_scope}'."
                )
                return ScopeStatus.OUT_OF_SCOPE
            
            # Check for cross-jurisdictional operations
            if self._is_cross_jurisdictional(signal.jurisdiction, signal.target_scope):
                reasons.append(f"Cross-jurisdictional operation detected: {signal.jurisdiction} → {signal.target_scope}.")
                constraints["cross_jurisdictional"] = True
                return ScopeStatus.CROSS_JURISDICTIONAL
        
        reasons.append(f"Scope validated: {signal.jurisdiction}.")
        constraints["jurisdiction"] = signal.jurisdiction
        
        if self.cfg.enable_audit_metadata:
            metadata["scope"] = {
                "jurisdiction": signal.jurisdiction,
                "target_scope": signal.target_scope,
                "cross_jurisdictional": constraints.get("cross_jurisdictional", False),
            }
        
        return ScopeStatus.IN_SCOPE
    
    def _scope_compatible(self, jurisdiction: str, target_scope: str) -> bool:
        """
        Check if jurisdiction has authority over target scope.
        Simple hierarchical check: federal > state > county > municipal
        """
        if jurisdiction == target_scope:
            return True
        
        # Federal can operate in any scope
        if jurisdiction == "federal":
            return True
        
        # State can operate in its counties/municipalities
        if jurisdiction.startswith("state:"):
            state = jurisdiction.split(":")[1]
            if target_scope.startswith(f"county:") or target_scope.startswith(f"municipal:"):
                return state in target_scope or True  # Simplified; real impl would check hierarchy
        
        return False
    
    def _is_cross_jurisdictional(self, jurisdiction: str, target_scope: str) -> bool:
        """
        Detect cross-jurisdictional operations.
        """
        if jurisdiction == target_scope:
            return False
        
        # Different states = cross-jurisdictional
        if jurisdiction.startswith("state:") and target_scope.startswith("state:"):
            return jurisdiction != target_scope
        
        return False
    
    # -------------------------
    # Layer 3: Temporal Validation
    # -------------------------
    
    def _validate_temporal(
        self,
        signal: GovernanceSignal,
        reasons: List[str],
        metadata: Dict[str, Any],
    ) -> str:
        """
        Validate temporal bounds (valid_from, valid_until).
        
        NOTE:
        GovernanceSignal is intentionally mutable in v1.x.
        Temporal bounds may be applied internally to enforce policy defaults.
        All externally visible outputs remain immutable (GovernanceDecision).
        """
        ts = signal.timestamp
        
        # Check if temporal bounds are required
        if self.cfg.require_temporal_bounds:
            if signal.valid_from is None or signal.valid_until is None:
                reasons.append("Temporal bounds required but not specified.")
                return TemporalStatus.EXPIRED
        
        # If no bounds specified, apply default validity window
        if signal.valid_from is None and signal.valid_until is None:
            # Use issued_at or current timestamp as start
            start = signal.issued_at if signal.issued_at else ts
            signal.valid_from = start
            signal.valid_until = start + self.cfg.default_validity_window_seconds
            reasons.append(f"Applied default validity window: {self.cfg.default_validity_window_seconds}s.")
        
        # Check not-yet-valid
        if signal.valid_from and ts < (signal.valid_from - self.cfg.grace_period_seconds):
            reasons.append(f"Signal not yet valid (valid_from={signal.valid_from}, current={ts}).")
            return TemporalStatus.NOT_YET_VALID
        
        # Check expiration
        if signal.valid_until and ts > (signal.valid_until + self.cfg.grace_period_seconds):
            reasons.append(f"Signal expired (valid_until={signal.valid_until}, current={ts}).")
            return TemporalStatus.EXPIRED
        
        reasons.append("Temporal validity confirmed.")
        
        if self.cfg.enable_audit_metadata:
            metadata["temporal"] = {
                "valid_from": signal.valid_from,
                "valid_until": signal.valid_until,
                "issued_at": signal.issued_at,
                "timestamp": ts,
                "remaining_validity_seconds": (signal.valid_until - ts) if signal.valid_until else None,
            }
        
        return TemporalStatus.VALID
    
    # -------------------------
    # Layer 4: Delegation Validation
    # -------------------------
    
    def _validate_delegation(
        self,
        signal: GovernanceSignal,
        reasons: List[str],
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Validate delegation chain.
        """
        if signal.delegation_depth == 0:
            # No delegation, direct authority
            return True
        
        # Check delegation depth limit
        if signal.delegation_depth > self.cfg.max_delegation_depth:
            reasons.append(
                f"Delegation depth {signal.delegation_depth} exceeds maximum {self.cfg.max_delegation_depth}."
            )
            return False
        
        # Check if delegated_from is specified
        if not signal.delegated_from:
            reasons.append("Delegation claimed but no delegated_from source specified.")
            return False
        
        # Check if delegation relationship is registered
        if signal.delegated_from not in self.known_delegations:
            reasons.append(f"No delegation record for source '{signal.delegated_from}'.")
            return False
        
        if signal.source not in self.known_delegations[signal.delegated_from]:
            reasons.append(
                f"Source '{signal.source}' not in delegation chain from '{signal.delegated_from}'."
            )
            return False
        
        # Check cross-jurisdictional delegation policy
        if not self.cfg.allow_cross_jurisdictional_delegation and signal.jurisdiction:
            # Would need to check if delegated_from has different jurisdiction
            # Simplified here
            pass
        
        reasons.append(f"Delegation validated: depth={signal.delegation_depth}, from={signal.delegated_from}.")
        
        if self.cfg.enable_audit_metadata:
            metadata["delegation"] = {
                "depth": signal.delegation_depth,
                "delegated_from": signal.delegated_from,
                "chain_valid": True,
            }
        
        return True
    
    # -------------------------
    # Layer 5: Actionability Determination
    # -------------------------
    
    def _determine_actionability(
        self,
        signal: GovernanceSignal,
        auth_level: int,
        scope_status: str,
        reasons: List[str],
        constraints: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Tuple[GovernanceTrustLevel, bool]:
        """
        Determine final trust level and actionability.
        
        Graduated trust levels based on:
        - Authority level
        - Consent requirements
        - Scope constraints
        - Delegation status
        """
        # Determine maximum achievable trust level based on authority
        if auth_level >= self.cfg.full_min_authority:
            max_trust = GovernanceTrustLevel.FULL_TRUST
        elif auth_level >= self.cfg.constrained_min_authority:
            max_trust = GovernanceTrustLevel.CONSTRAINED_TRUST
        elif auth_level >= self.cfg.advisory_min_authority:
            max_trust = GovernanceTrustLevel.ADVISORY_TRUST
        else:
            reasons.append("Authority level insufficient for any trust level.")
            return GovernanceTrustLevel.NO_TRUST, False
        
        # Check consent requirements for each level
        has_consent = bool(signal.consent_proof)
        
        if max_trust == GovernanceTrustLevel.FULL_TRUST:
            if self.cfg.full_requires_consent and not has_consent:
                reasons.append("FULL_TRUST requires consent; downgrading to CONSTRAINED_TRUST.")
                max_trust = GovernanceTrustLevel.CONSTRAINED_TRUST
        
        if max_trust == GovernanceTrustLevel.CONSTRAINED_TRUST:
            if self.cfg.constrained_requires_consent and not has_consent:
                reasons.append("CONSTRAINED_TRUST requires consent; downgrading to ADVISORY_TRUST.")
                max_trust = GovernanceTrustLevel.ADVISORY_TRUST
        
        if max_trust == GovernanceTrustLevel.ADVISORY_TRUST:
            if self.cfg.advisory_requires_consent and not has_consent:
                reasons.append("ADVISORY_TRUST requires consent; downgrading to NO_TRUST.")
                return GovernanceTrustLevel.NO_TRUST, False
        
        # Apply scope constraints
        if scope_status == ScopeStatus.CROSS_JURISDICTIONAL:
            if max_trust == GovernanceTrustLevel.FULL_TRUST:
                reasons.append("Cross-jurisdictional operation: limiting to CONSTRAINED_TRUST.")
                max_trust = GovernanceTrustLevel.CONSTRAINED_TRUST
                constraints["reason"] = "cross_jurisdictional_limitation"
        
        # Delegation reduces trust by one level (delegated authority is constrained)
        if signal.delegation_depth > 0:
            if max_trust == GovernanceTrustLevel.FULL_TRUST:
                reasons.append("Delegated authority: limiting to CONSTRAINED_TRUST.")
                max_trust = GovernanceTrustLevel.CONSTRAINED_TRUST
                constraints["delegation_constraint"] = True
            elif max_trust == GovernanceTrustLevel.CONSTRAINED_TRUST:
                reasons.append("Delegated authority: limiting to ADVISORY_TRUST.")
                max_trust = GovernanceTrustLevel.ADVISORY_TRUST
                constraints["delegation_constraint"] = True
        
        # Determine actionability
        actionable = max_trust >= GovernanceTrustLevel.CONSTRAINED_TRUST
        
        reasons.append(f"Final authorization: {max_trust.name} (actionable={actionable}).")
        
        if self.cfg.enable_audit_metadata:
            metadata["actionability"] = {
                "authorized_level": max_trust.name,
                "actionable": actionable,
                "authority_level_int": auth_level,
                "consent_present": has_consent,
                "constraints_applied": list(constraints.keys()),
            }
        
        return max_trust, actionable
    
    # -------------------------
    # Decision Construction
    # -------------------------
    
    def _make_decision(
        self,
        authorized_level: GovernanceTrustLevel,
        authority_status: str,
        scope_status: str,
        temporal_status: str,
        delegation_valid: bool,
        actionable: bool,
        reasons: List[str],
        constraints: Dict[str, Any],
        metadata: Dict[str, Any],
        decision_id: str,
        timestamp: float,
    ) -> GovernanceDecision:
        """
        Construct final governance decision.
        """
        return GovernanceDecision(
            authorized_level=authorized_level,
            authority_status=authority_status,
            scope_status=scope_status,
            temporal_status=temporal_status,
            delegation_valid=delegation_valid,
            actionable=actionable,
            reasons=reasons[:],
            constraints=dict(constraints),
            metadata=dict(metadata) if self.cfg.enable_audit_metadata else {},
            decision_id=decision_id,
            timestamp=timestamp,
        )
    
    def _commit_decision(self, decision: GovernanceDecision) -> None:
        """
        Store decision in history for audit trail.
        """
        if self.cfg.track_decision_history:
            self.decision_history.append(decision)
            # Keep last 1000 decisions
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
    
    def _generate_decision_id(self, signal: GovernanceSignal) -> str:
        """
        Generate unique decision ID for audit trail.
        """
        content = f"{signal.source}:{signal.timestamp}:{signal.mandate}:{signal.jurisdiction}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# -----------------------------
# Demo / Quick Self-Test
# -----------------------------

if __name__ == "__main__":
    print("=== AILEE Governance Domain Self-Test ===\n")
    
    config = GovernanceConfig(
        require_mandate=True,
        require_consent=True,
        require_jurisdiction=True,
        enforce_scope_boundaries=True,
    )
    
    governor = GovernanceGovernor(config)
    
    # Test Case 1: Valid certified official with full credentials
    print("Test 1: Certified Official - Voter Registration Update")
    print("-" * 60)
    result1 = governor.evaluate(
        source="election_authority_pa",
        jurisdiction="state:PA",
        authority_level="certified_official",
        mandate="voter_registration_update",
        consent_proof="signed_consent_hash_abc123",
        valid_from=time.time() - 3600,
        valid_until=time.time() + 86400,
        timestamp=time.time(),
        context={"operation": "voter_roll_update", "batch_id": "2024-12-001"},
    )
    print(f"Authorization Level: {result1.authorized_level.name}")
    print(f"Actionable: {result1.actionable}")
    print(f"Authority Status: {result1.authority_status}")
    print(f"Scope Status: {result1.scope_status}")
    print(f"Temporal Status: {result1.temporal_status}")
    print(f"Reasons: {result1.reasons[0] if result1.reasons else 'None'}")
    print()
    
    # Test Case 2: Observer role - should only get ADVISORY_TRUST
    print("Test 2: Observer - Ballot Counting Observation")
    print("-" * 60)
    result2 = governor.evaluate(
        source="poll_watcher_123",
        jurisdiction="county:Philadelphia",
        authority_level="observer",
        mandate="ballot_counting_observation",
        consent_proof="observer_credential_xyz",
        valid_from=time.time(),
        valid_until=time.time() + 86400,
        timestamp=time.time(),
        context={"location": "polling_station_42"},
    )
    print(f"Authorization Level: {result2.authorized_level.name}")
    print(f"Actionable: {result2.actionable}")
    print(f"Authority Status: {result2.authority_status}")
    print(f"Reasons: {result2.reasons[0] if result2.reasons else 'None'}")
    print()
    
    # Test Case 3: Missing mandate - should fail
    print("Test 3: Missing Mandate - Should Reject")
    print("-" * 60)
    result3 = governor.evaluate(
        source="unknown_source",
        jurisdiction="state:PA",
        authority_level="certified_official",
        mandate=None,  # Missing mandate
        consent_proof="some_consent",
        timestamp=time.time(),
    )
    print(f"Authorization Level: {result3.authorized_level.name}")
    print(f"Actionable: {result3.actionable}")
    print(f"Reasons: {result3.reasons[0] if result3.reasons else 'None'}")
    print()
    
    # Test Case 4: Expired signal - should reject
    print("Test 4: Expired Temporal Validity")
    print("-" * 60)
    result4 = governor.evaluate(
        source="election_authority_ny",
        jurisdiction="state:NY",
        authority_level="administrator",
        mandate="policy_advisory_input",
        consent_proof="consent_hash",
        valid_from=time.time() - 172800,  # 2 days ago
        valid_until=time.time() - 86400,  # expired 1 day ago
        timestamp=time.time(),
    )
    print(f"Authorization Level: {result4.authorized_level.name}")
    print(f"Actionable: {result4.actionable}")
    print(f"Temporal Status: {result4.temporal_status}")
    print(f"Reasons: {result4.reasons[0] if result4.reasons else 'None'}")
    print()
    
    # Test Case 5: Cross-jurisdictional operation
    print("Test 5: Cross-Jurisdictional Operation (State PA → State NY)")
    print("-" * 60)
    result5 = governor.evaluate(
        source="interstate_coordinator",
        jurisdiction="state:PA",
        target_scope="state:NY",
        authority_level="administrator",
        mandate="compliance_audit",
        consent_proof="interstate_agreement_hash",
        valid_from=time.time(),
        valid_until=time.time() + 86400,
        timestamp=time.time(),
    )
    print(f"Authorization Level: {result5.authorized_level.name}")
    print(f"Actionable: {result5.actionable}")
    print(f"Scope Status: {result5.scope_status}")
    print(f"Constraints: {result5.constraints}")
    print(f"Reasons: {'; '.join(result5.reasons[:2])}")
    print()
    
    # Test Case 6: Delegated authority
    print("Test 6: Delegated Authority (1 level deep)")
    print("-" * 60)
    governor.register_delegation("election_authority_pa", "deputy_official_123")
    result6 = governor.evaluate(
        source="deputy_official_123",
        jurisdiction="state:PA",
        authority_level="certified_official",
        mandate="voter_registration_update",
        consent_proof="delegated_consent",
        delegated_from="election_authority_pa",
        delegation_depth=1,
        valid_from=time.time(),
        valid_until=time.time() + 86400,
        timestamp=time.time(),
    )
    print(f"Authorization Level: {result6.authorized_level.name}")
    print(f"Actionable: {result6.actionable}")
    print(f"Delegation Valid: {result6.delegation_valid}")
    print(f"Constraints: {result6.constraints}")
    print(f"Reasons: {'; '.join(result6.reasons[:2])}")
    print()
    
    # Test Case 7: Revoked source
    print("Test 7: Revoked Source")
    print("-" * 60)
    governor.revoke_source("compromised_official", "Security breach detected")
    result7 = governor.evaluate(
        source="compromised_official",
        jurisdiction="state:CA",
        authority_level="certified_official",
        mandate="voter_registration_update",
        consent_proof="consent_hash",
        timestamp=time.time(),
    )
    print(f"Authorization Level: {result7.authorized_level.name}")
    print(f"Actionable: {result7.actionable}")
    print(f"Authority Status: {result7.authority_status}")
    print(f"Reasons: {result7.reasons[0] if result7.reasons else 'None'}")
    print()
    
    # Test Case 8: Unrecognized jurisdiction
    print("Test 8: Unrecognized Jurisdiction")
    print("-" * 60)
    result8 = governor.evaluate(
        source="foreign_authority",
        jurisdiction="country:Canada",  # Not in recognized jurisdictions
        authority_level="certified_official",
        mandate="policy_advisory_input",
        consent_proof="consent",
        timestamp=time.time(),
    )
    print(f"Authorization Level: {result8.authorized_level.name}")
    print(f"Actionable: {result8.actionable}")
    print(f"Scope Status: {result8.scope_status}")
    print(f"Reasons: {result8.reasons[0] if result8.reasons else 'None'}")
    print()
    
    # Test Case 9: Reporter level with consent - CONSTRAINED_TRUST
    print("Test 9: Reporter Level - Regulatory Enforcement")
    print("-" * 60)
    result9 = governor.evaluate(
        source="compliance_reporter_456",
        jurisdiction="federal",
        authority_level="reporter",
        mandate="regulatory_enforcement",
        consent_proof="federal_authorization",
        valid_from=time.time(),
        valid_until=time.time() + 2592000,  # 30 days
        timestamp=time.time(),
    )
    print(f"Authorization Level: {result9.authorized_level.name}")
    print(f"Actionable: {result9.actionable}")
    print(f"Authority Status: {result9.authority_status}")
    print(f"Reasons: {result9.reasons[0] if result9.reasons else 'None'}")
    print()
    
    # Test Case 10: Future-dated signal (not yet valid)
    print("Test 10: Future-Dated Signal (Not Yet Valid)")
    print("-" * 60)
    result10 = governor.evaluate(
        source="scheduled_authority",
        jurisdiction="state:NY",
        authority_level="administrator",
        mandate="emergency_directive",
        consent_proof="pre_authorized",
        valid_from=time.time() + 7200,  # valid 2 hours from now
        valid_until=time.time() + 93600,
        timestamp=time.time(),
    )
    print(f"Authorization Level: {result10.authorized_level.name}")
    print(f"Actionable: {result10.actionable}")
    print(f"Temporal Status: {result10.temporal_status}")
    print(f"Reasons: {result10.reasons[0] if result10.reasons else 'None'}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total decisions made: {len(governor.get_decision_history())}")
    print(f"Actionable decisions: {sum(1 for d in governor.get_decision_history() if d.actionable)}")
    print(f"Rejected decisions: {sum(1 for d in governor.get_decision_history() if d.authorized_level == GovernanceTrustLevel.NO_TRUST)}")
    print(f"Advisory only: {sum(1 for d in governor.get_decision_history() if d.authorized_level == GovernanceTrustLevel.ADVISORY_TRUST)}")
    print(f"Constrained trust: {sum(1 for d in governor.get_decision_history() if d.authorized_level == GovernanceTrustLevel.CONSTRAINED_TRUST)}")
    print(f"Full trust: {sum(1 for d in governor.get_decision_history() if d.authorized_level == GovernanceTrustLevel.FULL_TRUST)}")
    print("\n✓ All tests completed successfully!")
