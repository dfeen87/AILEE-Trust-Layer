"""
AILEE Trust Layer — Cross-Ecosystem Translation Domain
Version: 1.0.0 - Production Grade

Governance for semantic state and intent translation between incompatible
software-hardware ecosystems (e.g., iOS ↔ Android, proprietary wearables).

This domain does NOT bypass security, modify firmware, or violate platform terms.
It governs whether translated state/intent is trustworthy enough to act upon based on:
- Translation confidence and semantic fidelity
- Cross-platform consent and privacy boundaries
- Signal freshness across asynchronous ecosystems
- Context preservation quality
- Platform capability alignment

Primary governed signals:
- Translation trust level (scalar, 0..3)
- Semantic fidelity score
- Privacy boundary compliance
- Context preservation quality

Trust level semantics:
    0 = NO_TRUST              Do not use translated state
    1 = ADVISORY_TRUST        Display only, no actions
    2 = CONSTRAINED_TRUST     Limited automation within privacy bounds
    3 = FULL_TRUST            Full cross-ecosystem continuity

Key properties:
- Platform-agnostic semantic mapping
- Privacy-first consent enforcement
- Asymmetric capability handling
- Degradation-aware translation
- Audit trail for consent compliance

INTEGRATION EXAMPLE:

    # Setup
    policy = CrossEcosystemPolicy(
        max_allowed_level=TranslationTrustLevel.CONSTRAINED_TRUST,
        require_explicit_consent=True,
    )
    governor = CrossEcosystemGovernor(policy=policy)
    
    # Translate state between ecosystems
    while system_active:
        # Receive state from source ecosystem
        source_state = source_platform.get_state()
        
        signals = CrossEcosystemSignals(
            desired_level=application.get_required_trust(),
            semantic_fidelity=translator.estimate_fidelity(source_state),
            source_ecosystem="ios_healthkit",
            target_ecosystem="android_health_connect",
            translation_path=["ios_healthkit", "ailee_semantic", "android_health_connect"],
            consent_status=ConsentStatus(
                user_consent_granted=True,
                data_categories=["activity", "heart_rate"],
                consent_timestamp=time.time() - 3600,
            ),
            privacy_boundaries=PrivacyBoundaries(
                pii_allowed=False,
                location_precision="city",
                temporal_resolution="hourly",
            ),
            signal_freshness_ms=get_age_since_capture(source_state),
            context_preservation=ContextPreservation(
                intent_maintained=True,
                semantic_loss=0.12,
                capability_alignment=0.88,
            ),
        )
        
        authorized_level, decision = governor.evaluate(signals)
        
        if authorized_level >= TranslationTrustLevel.CONSTRAINED_TRUST:
            translated_state = translator.translate(source_state, decision.metadata)
            target_platform.apply_state(translated_state)
        else:
            logger.info(f"Translation blocked: {decision.reasons}")
        
        # Audit for compliance
        audit_log.record(governor.get_last_event())

⚠️  This module DOES NOT bypass platform security or modify hardware.
    It determines whether semantic translation is trustworthy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Set


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core AILEE Trust Pipeline Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    from ailee_trust_pipeline_v1 import (
        AileeConfig,
        AileeTrustPipeline,
        DecisionResult,
        SafetyStatus,
    )
except Exception:  # pragma: no cover
    AileeConfig = None  # type: ignore
    AileeTrustPipeline = None  # type: ignore
    DecisionResult = None  # type: ignore
    SafetyStatus = None  # type: ignore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Translation Trust Levels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TranslationTrustLevel(IntEnum):
    """
    Discrete trust levels for cross-ecosystem state translation.
    
    These levels represent increasing confidence in semantic translation,
    with each level having specific consent and privacy requirements.
    """
    NO_TRUST = 0           # Do not use translated state
    ADVISORY_TRUST = 1     # Display only, no automated actions
    CONSTRAINED_TRUST = 2  # Limited automation within privacy bounds
    FULL_TRUST = 3         # Full cross-ecosystem continuity


def _clamp_int(x: int, lo: int, hi: int) -> int:
    """Clamp integer to inclusive range [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def _level_from_float(x: float) -> TranslationTrustLevel:
    """Quantize continuous value to nearest discrete trust level."""
    ix = int(round(float(x)))
    ix = _clamp_int(
        ix,
        int(TranslationTrustLevel.NO_TRUST),
        int(TranslationTrustLevel.FULL_TRUST)
    )
    return TranslationTrustLevel(ix)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ecosystem Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class EcosystemCapabilities:
    """Define what each ecosystem can/cannot do."""
    ecosystem_id: str
    
    # Data categories supported
    supported_categories: Set[str] = field(default_factory=set)
    
    # API capabilities
    realtime_available: bool = False
    batch_sync_available: bool = True
    bidirectional: bool = False
    
    # Privacy features
    granular_consent: bool = False
    data_minimization: bool = False
    
    # Technical constraints
    max_data_age_hours: float = 24.0
    rate_limit_per_hour: Optional[int] = None
    
    # Known limitations
    semantic_gaps: List[str] = field(default_factory=list)


KNOWN_ECOSYSTEMS: Dict[str, EcosystemCapabilities] = {
    "ios_healthkit": EcosystemCapabilities(
        ecosystem_id="ios_healthkit",
        supported_categories={"activity", "heart_rate", "sleep", "nutrition", "mindfulness"},
        realtime_available=True,
        bidirectional=False,
        granular_consent=True,
        max_data_age_hours=48.0,
        semantic_gaps=["social_context", "device_continuity"],
    ),
    "android_health_connect": EcosystemCapabilities(
        ecosystem_id="android_health_connect",
        supported_categories={"activity", "heart_rate", "sleep", "nutrition"},
        realtime_available=True,
        bidirectional=True,
        granular_consent=True,
        max_data_age_hours=72.0,
        semantic_gaps=["mindfulness", "environmental_audio"],
    ),
    "wear_os": EcosystemCapabilities(
        ecosystem_id="wear_os",
        supported_categories={"activity", "heart_rate", "location"},
        realtime_available=True,
        bidirectional=True,
        rate_limit_per_hour=3600,
    ),
    "apple_watch": EcosystemCapabilities(
        ecosystem_id="apple_watch",
        supported_categories={"activity", "heart_rate", "sleep", "environmental_audio"},
        realtime_available=True,
        bidirectional=False,
        granular_consent=True,
    ),
    "fitbit": EcosystemCapabilities(
        ecosystem_id="fitbit",
        supported_categories={"activity", "heart_rate", "sleep"},
        batch_sync_available=True,
        max_data_age_hours=168.0,  # 7 days
        semantic_gaps=["realtime_hr", "advanced_sleep_stages"],
    ),
    "garmin": EcosystemCapabilities(
        ecosystem_id="garmin",
        supported_categories={"activity", "heart_rate", "sleep", "stress"},
        batch_sync_available=True,
        bidirectional=True,
        max_data_age_hours=168.0,
    ),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Consent and Privacy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class ConsentStatus:
    """User consent state for cross-ecosystem translation."""
    user_consent_granted: bool
    data_categories: List[str] = field(default_factory=list)
    consent_timestamp: Optional[float] = None
    consent_expiry: Optional[float] = None
    
    # Specific permissions
    allows_automation: bool = False
    allows_third_party: bool = False
    allows_aggregation: bool = True
    
    # Revocation tracking
    revoked: bool = False
    revocation_timestamp: Optional[float] = None
    
    def is_valid_for_category(self, category: str, current_time: float) -> bool:
        """Check if consent covers a specific data category."""
        if self.revoked:
            return False
        
        if not self.user_consent_granted:
            return False
        
        if category not in self.data_categories:
            return False
        
        if self.consent_expiry is not None:
            if current_time > self.consent_expiry:
                return False
        
        return True


@dataclass(frozen=True)
class PrivacyBoundaries:
    """Privacy constraints for translation."""
    # PII handling
    pii_allowed: bool = False
    anonymize_required: bool = False
    
    # Spatial resolution
    location_precision: str = "none"  # none | country | city | precise
    
    # Temporal resolution
    temporal_resolution: str = "daily"  # hourly | daily | weekly | monthly
    
    # Data retention
    max_retention_hours: Optional[float] = None
    
    # Cross-border
    cross_border_allowed: bool = True
    allowed_regions: Optional[List[str]] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Context Preservation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class ContextPreservation:
    """Measure how well context is preserved in translation."""
    intent_maintained: bool = True
    semantic_loss: float = 0.0  # 0..1, lower is better
    capability_alignment: float = 1.0  # 0..1, higher is better
    
    # Specific context elements
    user_activity_preserved: bool = True
    temporal_context_preserved: bool = True
    social_context_preserved: bool = False
    environmental_context_preserved: bool = False
    
    # Known degradations
    precision_loss: Optional[str] = None  # e.g., "10m -> 100m location"
    feature_unavailable: List[str] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Translation Path
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class TranslationPath:
    """Track the translation chain between ecosystems."""
    source: str
    target: str
    intermediates: List[str] = field(default_factory=list)
    
    # Translation quality
    total_hops: int = 1
    cumulative_loss: float = 0.0  # Accumulated semantic loss
    
    # Known issues in this path
    known_incompatibilities: List[str] = field(default_factory=list)
    requires_approximation: bool = False
    
    def get_full_path(self) -> List[str]:
        """Get complete translation path."""
        return [self.source] + self.intermediates + [self.target]
    
    def estimate_fidelity(self) -> float:
        """Estimate translation fidelity (0..1)."""
        base_fidelity = 1.0 - self.cumulative_loss
        hop_penalty = 0.95 ** (self.total_hops - 1)  # 5% loss per hop
        return max(0.0, base_fidelity * hop_penalty)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain Input Signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class CrossEcosystemSignals:
    """Primary input structure for cross-ecosystem governance."""
    desired_level: TranslationTrustLevel
    
    # Translation quality
    semantic_fidelity: float  # 0..1, confidence in meaning preservation
    
    # Ecosystem information
    source_ecosystem: str
    target_ecosystem: str
    translation_path: List[str] = field(default_factory=list)
    
    # Consent and privacy
    consent_status: Optional[ConsentStatus] = None
    privacy_boundaries: Optional[PrivacyBoundaries] = None
    
    # Context preservation
    context_preservation: Optional[ContextPreservation] = None
    
    # Data freshness
    signal_freshness_ms: Optional[float] = None
    source_capture_timestamp: Optional[float] = None
    
    # Data category
    data_category: Optional[str] = None
    
    # Alternative translations (for consensus)
    alternative_translations: Tuple[float, ...] = ()
    
    timestamp: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GovernanceEvent:
    """Audit event for compliance and debugging."""
    timestamp: float
    event_type: str
    from_level: TranslationTrustLevel
    to_level: TranslationTrustLevel
    
    # Translation details
    source_ecosystem: str
    target_ecosystem: str
    data_category: Optional[str]
    
    # Decision factors
    semantic_fidelity: float
    consent_valid: bool
    privacy_compliant: bool
    
    reasons: List[str]
    metadata: Dict[str, Any]
    
    safety_status: Optional[str] = None
    used_fallback: bool = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance Policy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class CrossEcosystemPolicy:
    """Domain policy for cross-ecosystem translation governance."""
    # Trust thresholds
    min_semantic_fidelity_for_advisory: float = 0.70
    min_semantic_fidelity_for_constrained: float = 0.80
    min_semantic_fidelity_for_full: float = 0.90
    
    # Consent requirements
    require_explicit_consent: bool = True
    max_consent_age_hours: float = 720.0  # 30 days default
    
    # Data freshness
    max_signal_age_hours: float = 24.0
    warn_signal_age_hours: float = 12.0
    
    # Privacy
    enforce_privacy_boundaries: bool = True
    allow_pii_translation: bool = False
    
    # Context preservation
    min_context_preservation: float = 0.75
    require_intent_preservation: bool = True
    
    # Multi-path validation
    require_translation_consensus: bool = False
    min_consensus_agreement: float = 0.85
    
    # Capability matching
    enforce_capability_alignment: bool = True
    min_capability_overlap: float = 0.60
    
    # Deployment cap
    max_allowed_level: TranslationTrustLevel = TranslationTrustLevel.FULL_TRUST
    
    # Rate limiting
    enable_rate_limiting: bool = True
    
    # Logging
    max_event_log_size: int = 5000


def default_cross_ecosystem_config() -> "AileeConfig":
    """Safe defaults for cross-ecosystem translation governance."""
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    return AileeConfig(
        accept_threshold=0.82,
        borderline_low=0.70,
        borderline_high=0.82,
        w_stability=0.45,
        w_agreement=0.40,
        w_likelihood=0.15,
        history_window=150,
        forecast_window=20,
        grace_peer_delta=1.0,
        grace_min_peer_agreement_ratio=0.70,
        grace_forecast_epsilon=0.20,
        grace_max_abs_z=2.5,
        consensus_quorum=2,
        consensus_delta=0.8,
        consensus_pass_ratio=0.75,
        fallback_mode="last_good",
        fallback_clamp_min=0.0,
        fallback_clamp_max=3.0,
        hard_min=0.0,
        hard_max=3.0,
        enable_grace=True,
        enable_consensus=True,
        enable_audit_metadata=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-Ecosystem Governor (Main Controller)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CrossEcosystemGovernor:
    """Production-grade governance for cross-ecosystem translation."""
    
    def __init__(
        self,
        cfg: Optional["AileeConfig"] = None,
        policy: Optional[CrossEcosystemPolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable")
        
        self.cfg = cfg or default_cross_ecosystem_config()
        self.policy = policy or CrossEcosystemPolicy()
        
        self.cfg.hard_max = float(int(self.policy.max_allowed_level))
        self.cfg.fallback_clamp_max = float(int(self.policy.max_allowed_level))
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        
        self._last_level: TranslationTrustLevel = TranslationTrustLevel.NO_TRUST
        self._event_log: List[GovernanceEvent] = []
        self._last_event: Optional[GovernanceEvent] = None
        
        # Rate limiting per ecosystem pair
        self._rate_tracker: Dict[str, List[float]] = {}
    
    def evaluate(
        self, signals: CrossEcosystemSignals
    ) -> Tuple[TranslationTrustLevel, "DecisionResult"]:
        """
        Evaluate cross-ecosystem translation trust decision.
        
        Args:
            signals: Current translation signals
        
        Returns:
            (authorized_level, decision_result)
        """
        ts = float(signals.timestamp if signals.timestamp is not None else time.time())
        
        # Apply governance gates
        gated_level, gate_reasons = self._apply_governance_gates(signals, ts)
        
        # Check rate limits
        rate_ok, rate_reason = self._check_rate_limits(signals, ts)
        if not rate_ok:
            gate_reasons.append(rate_reason)
            gated_level = min(gated_level, TranslationTrustLevel.NO_TRUST)
        
        # Prepare for trust pipeline
        peer_values = list(signals.alternative_translations)
        
        ctx = {
            "domain": "cross_ecosystem",
            "signal": "translation_trust_level",
            "desired_level": int(signals.desired_level),
            "gated_level": int(gated_level),
            "gate_reasons": gate_reasons,
            "source_ecosystem": signals.source_ecosystem,
            "target_ecosystem": signals.target_ecosystem,
            "data_category": signals.data_category,
            "semantic_fidelity": signals.semantic_fidelity,
        }
        
        res = self.pipeline.process(
            raw_value=float(int(gated_level)),
            raw_confidence=signals.semantic_fidelity,
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        authorized_level = _level_from_float(res.value)
        
        # Log event
        self._log_governance_event(ts, signals, authorized_level, res, gate_reasons)
        
        # Update state
        self._last_level = authorized_level
        
        return authorized_level, res
    
    def get_last_event(self) -> Optional[GovernanceEvent]:
        """Get most recent governance event."""
        return self._last_event
    
    def get_event_log(self) -> List[GovernanceEvent]:
        """Get full event log for compliance export."""
        return list(self._event_log)
    
    def _apply_governance_gates(
        self, signals: CrossEcosystemSignals, ts: float
    ) -> Tuple[TranslationTrustLevel, List[str]]:
        """Apply deterministic governance gates."""
        reasons: List[str] = []
        level = signals.desired_level
        
        # Deployment cap
        if level > self.policy.max_allowed_level:
            reasons.append(f"Policy cap to {int(self.policy.max_allowed_level)}")
            level = self.policy.max_allowed_level
        
        # Consent check (CRITICAL - blocks everything if failed)
        if self.policy.require_explicit_consent:
            if signals.consent_status is None:
                reasons.append("No consent status provided")
                return TranslationTrustLevel.NO_TRUST, reasons
            
            if not signals.consent_status.user_consent_granted:
                reasons.append("User consent not granted")
                return TranslationTrustLevel.NO_TRUST, reasons
            
            if signals.consent_status.revoked:
                reasons.append("Consent was revoked")
                return TranslationTrustLevel.NO_TRUST, reasons
            
            # Check consent age
            if signals.consent_status.consent_timestamp is not None:
                consent_age_hours = (ts - signals.consent_status.consent_timestamp) / 3600.0
                if consent_age_hours > self.policy.max_consent_age_hours:
                    reasons.append(
                        f"Consent age {consent_age_hours:.1f}h exceeds "
                        f"{self.policy.max_consent_age_hours:.1f}h"
                    )
                    return TranslationTrustLevel.NO_TRUST, reasons
            
            # Check category-specific consent
            if signals.data_category is not None:
                if not signals.consent_status.is_valid_for_category(signals.data_category, ts):
                    reasons.append(f"No consent for category '{signals.data_category}'")
                    return TranslationTrustLevel.NO_TRUST, reasons
            
            # Automation permission
            if level >= TranslationTrustLevel.CONSTRAINED_TRUST:
                if not signals.consent_status.allows_automation:
                    reasons.append("Consent does not allow automation")
                    level = TranslationTrustLevel.ADVISORY_TRUST
        
        # Privacy boundaries
        if self.policy.enforce_privacy_boundaries and signals.privacy_boundaries is not None:
            pb = signals.privacy_boundaries
            
            if not self.policy.allow_pii_translation and pb.pii_allowed:
                reasons.append("PII translation not allowed by policy")
                level = min(level, TranslationTrustLevel.ADVISORY_TRUST)
            
            if pb.anonymize_required:
                reasons.append("Anonymization required - constraining trust")
                level = min(level, TranslationTrustLevel.CONSTRAINED_TRUST)
        
        # Semantic fidelity thresholds
        if level >= TranslationTrustLevel.FULL_TRUST:
            if signals.semantic_fidelity < self.policy.min_semantic_fidelity_for_full:
                reasons.append(
                    f"Semantic fidelity {signals.semantic_fidelity:.2f} < "
                    f"{self.policy.min_semantic_fidelity_for_full:.2f} for FULL_TRUST"
                )
                level = TranslationTrustLevel.CONSTRAINED_TRUST
        
        if level >= TranslationTrustLevel.CONSTRAINED_TRUST:
            if signals.semantic_fidelity < self.policy.min_semantic_fidelity_for_constrained:
                reasons.append(
                    f"Semantic fidelity {signals.semantic_fidelity:.2f} < "
                    f"{self.policy.min_semantic_fidelity_for_constrained:.2f} for CONSTRAINED_TRUST"
                )
                level = TranslationTrustLevel.ADVISORY_TRUST
        
        if level >= TranslationTrustLevel.ADVISORY_TRUST:
            if signals.semantic_fidelity < self.policy.min_semantic_fidelity_for_advisory:
                reasons.append(
                    f"Semantic fidelity {signals.semantic_fidelity:.2f} < "
                    f"{self.policy.min_semantic_fidelity_for_advisory:.2f}"
                )
                level = TranslationTrustLevel.NO_TRUST
        
        # Data freshness
        if signals.signal_freshness_ms is not None:
            age_hours = signals.signal_freshness_ms / (1000.0 * 3600.0)
            
            if age_hours > self.policy.max_signal_age_hours:
                reasons.append(
                    f"Signal age {age_hours:.1f}h exceeds {self.policy.max_signal_age_hours:.1f}h"
                )
                level = min(level, TranslationTrustLevel.ADVISORY_TRUST)
            elif age_hours > self.policy.warn_signal_age_hours:
                reasons.append(f"Warning: Signal age {age_hours:.1f}h")
        
        # Context preservation
        if signals.context_preservation is not None and self.policy.require_intent_preservation:
            cp = signals.context_preservation
            
            if not cp.intent_maintained:
                reasons.append("User intent not maintained in translation")
                level = min(level, TranslationTrustLevel.ADVISORY_TRUST)
            
            if cp.semantic_loss > (1.0 - self.policy.min_context_preservation):
                reasons.append(
                    f"Semantic loss {cp.semantic_loss:.2f} too high (max "
                    f"{1.0 - self.policy.min_context_preservation:.2f})"
                )
                level = min(level, TranslationTrustLevel.CONSTRAINED_TRUST)
            
            if cp.capability_alignment < 0.50:
                reasons.append(
                    f"Capability alignment {cp.capability_alignment:.2f} too low"
                )
                level = min(level, TranslationTrustLevel.ADVISORY_TRUST)
        
        # Ecosystem capability validation
        if self.policy.enforce_capability_alignment:
            source_caps = KNOWN_ECOSYSTEMS.get(signals.source_ecosystem)
            target_caps = KNOWN_ECOSYSTEMS.get(signals.target_ecosystem)
            
            if source_caps and target_caps and signals.data_category:
                if signals.data_category not in source_caps.supported_categories:
                    reasons.append(
                        f"Source ecosystem '{signals.source_ecosystem}' "
                        f"doesn't support category '{signals.data_category}'"
                    )
                    return TranslationTrustLevel.NO_TRUST, reasons
                
                if signals.data_category not in target_caps.supported_categories:
                    reasons.append(
                        f"Target ecosystem '{signals.target_ecosystem}' "
                        f"doesn't support category '{signals.data_category}'"
                    )
                    return TranslationTrustLevel.NO_TRUST, reasons
        
        return level, reasons
    
    def _check_rate_limits(
        self, signals: CrossEcosystemSignals, ts: float
    ) -> Tuple[bool, str]:
        """Check if rate limits are exceeded."""
        if not self.policy.enable_rate_limiting:
            return True, ""
        
        pair_key = f"{signals.source_ecosystem}→{signals.target_ecosystem}"
        
        if pair_key not in self._rate_tracker:
            self._rate_tracker[pair_key] = []
        
        # Add current timestamp
        self._rate_tracker[pair_key].append(ts)
        
        # Clean old timestamps (keep last hour)
        one_hour_ago = ts - 3600.0
        self._rate_tracker[pair_key] = [
            t for t in self._rate_tracker[pair_key] if t >= one_hour_ago
        ]
        
        # Check target ecosystem rate limit
        target_caps = KNOWN_ECOSYSTEMS.get(signals.target_ecosystem)
        if target_caps and target_caps.rate_limit_per_hour is not None:
            count = len(self._rate_tracker[pair_key])
            if count > target_caps.rate_limit_per_hour:
                return False, (
                    f"Rate limit exceeded: {count} translations in last hour "
                    f"(max {target_caps.rate_limit_per_hour})"
                )
        
        return True, ""
    
    def _log_governance_event(
        self,
        ts: float,
        signals: CrossEcosystemSignals,
        authorized_level: TranslationTrustLevel,
        res: "DecisionResult",
        gate_reasons: List[str],
    ) -> None:
        """Log governance event for audit trail."""
        event_type = "evaluation"
        if authorized_level != self._last_level:
            event_type = "level_change"
        elif res.used_fallback:
            event_type = "fallback_used"
        elif gate_reasons:
            event_type = "gate_applied"
        
        consent_valid = False
        if signals.consent_status is not None:
            consent_valid = (
                signals.consent_status.user_consent_granted 
                and not signals.consent_status.revoked
            )
        
        privacy_compliant = True
        if self.policy.enforce_privacy_boundaries:
            if signals.privacy_boundaries is not None:
                if signals.privacy_boundaries.pii_allowed and not self.policy.allow_pii_translation:
                    privacy_compliant = False
        
        event = GovernanceEvent(
            timestamp=ts,
            event_type=event_type,
            from_level=self._last_level,
            to_level=authorized_level,
            source_ecosystem=signals.source_ecosystem,
            target_ecosystem=signals.target_ecosystem,
            data_category=signals.data_category,
            semantic_fidelity=signals.semantic_fidelity,
            consent_valid=consent_valid,
            privacy_compliant=privacy_compliant,
            reasons=res.reasons + gate_reasons,
            metadata=dict(res.metadata) if res.metadata else {},
            safety_status=str(res.safety_status) if res.safety_status else None,
            used_fallback=res.used_fallback,
        )
        
        self._event_log.append(event)
        self._last_event = event
        
        if len(self._event_log) > self.policy.max_event_log_size:
            self._event_log = self._event_log[-self.policy.max_event_log_size:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_default_governor(
    max_level: TranslationTrustLevel = TranslationTrustLevel.CONSTRAINED_TRUST,
) -> CrossEcosystemGovernor:
    """Create a cross-ecosystem governor with safe defaults."""
    policy = CrossEcosystemPolicy(max_allowed_level=max_level)
    cfg = default_cross_ecosystem_config()
    return CrossEcosystemGovernor(cfg=cfg, policy=policy)


def create_health_data_signals(
    source: str = "ios_healthkit",
    target: str = "android_health_connect",
    fidelity: float = 0.88,
) -> CrossEcosystemSignals:
    """Create example signals for health data translation."""
    return CrossEcosystemSignals(
        desired_level=TranslationTrustLevel.CONSTRAINED_TRUST,
        semantic_fidelity=fidelity,
        source_ecosystem=source,
        target_ecosystem=target,
        translation_path=[source, "ailee_semantic", target],
        consent_status=ConsentStatus(
            user_consent_granted=True,
            data_categories=["activity", "heart_rate"],
            consent_timestamp=time.time() - 3600,
            allows_automation=True,
        ),
        privacy_boundaries=PrivacyBoundaries(
            pii_allowed=False,
            location_precision="city",
            temporal_resolution="hourly",
        ),
        context_preservation=ContextPreservation(
            intent_maintained=True,
            semantic_loss=0.12,
            capability_alignment=0.88,
        ),
        signal_freshness_ms=300_000.0,  # 5 minutes
        data_category="heart_rate",
        timestamp=time.time(),
    )


def create_wearable_continuity_signals() -> CrossEcosystemSignals:
    """Create signals for wearable device continuity scenario."""
    return CrossEcosystemSignals(
        desired_level=TranslationTrustLevel.FULL_TRUST,
        semantic_fidelity=0.92,
        source_ecosystem="apple_watch",
        target_ecosystem="wear_os",
        translation_path=["apple_watch", "ailee_semantic", "wear_os"],
        consent_status=ConsentStatus(
            user_consent_granted=True,
            data_categories=["activity", "heart_rate", "location"],
            consent_timestamp=time.time() - 86400,  # 1 day ago
            allows_automation=True,
            allows_aggregation=True,
        ),
        privacy_boundaries=PrivacyBoundaries(
            pii_allowed=False,
            location_precision="precise",
            temporal_resolution="hourly",
        ),
        context_preservation=ContextPreservation(
            intent_maintained=True,
            semantic_loss=0.08,
            capability_alignment=0.92,
            user_activity_preserved=True,
            temporal_context_preserved=True,
        ),
        signal_freshness_ms=60_000.0,  # 1 minute
        data_category="activity",
        timestamp=time.time(),
    )


def create_consent_violation_signals() -> CrossEcosystemSignals:
    """Create signals demonstrating consent violation blocking."""
    return CrossEcosystemSignals(
        desired_level=TranslationTrustLevel.FULL_TRUST,
        semantic_fidelity=0.95,
        source_ecosystem="fitbit",
        target_ecosystem="garmin",
        consent_status=ConsentStatus(
            user_consent_granted=False,  # No consent!
            revoked=True,
            revocation_timestamp=time.time() - 1800,
        ),
        data_category="sleep",
        timestamp=time.time(),
    )


def create_low_fidelity_signals() -> CrossEcosystemSignals:
    """Create signals with low semantic fidelity."""
    return CrossEcosystemSignals(
        desired_level=TranslationTrustLevel.FULL_TRUST,
        semantic_fidelity=0.55,  # Too low for any trust
        source_ecosystem="fitbit",
        target_ecosystem="ios_healthkit",
        consent_status=ConsentStatus(
            user_consent_granted=True,
            data_categories=["sleep"],
            consent_timestamp=time.time() - 3600,
        ),
        context_preservation=ContextPreservation(
            intent_maintained=False,
            semantic_loss=0.45,
            capability_alignment=0.55,
            feature_unavailable=["advanced_sleep_stages"],
        ),
        signal_freshness_ms=120_000.0,
        data_category="sleep",
        timestamp=time.time(),
    )


def validate_signals(signals: CrossEcosystemSignals) -> Tuple[bool, List[str]]:
    """Validate cross-ecosystem signals structure."""
    errors: List[str] = []
    
    # Check trust level
    if not isinstance(signals.desired_level, TranslationTrustLevel):
        errors.append("desired_level must be TranslationTrustLevel enum")
    
    # Check semantic fidelity bounds
    if not (0.0 <= signals.semantic_fidelity <= 1.0):
        errors.append(f"semantic_fidelity={signals.semantic_fidelity} outside [0, 1]")
    
    # Check ecosystem identifiers
    if not signals.source_ecosystem:
        errors.append("source_ecosystem is required")
    if not signals.target_ecosystem:
        errors.append("target_ecosystem is required")
    
    # Check consent if privacy enforcement enabled
    if signals.consent_status is not None:
        cs = signals.consent_status
        if cs.consent_timestamp is not None and cs.consent_timestamp < 0:
            errors.append("consent_timestamp cannot be negative")
        if cs.consent_expiry is not None and cs.consent_expiry < 0:
            errors.append("consent_expiry cannot be negative")
    
    # Check context preservation scores
    if signals.context_preservation is not None:
        cp = signals.context_preservation
        if not (0.0 <= cp.semantic_loss <= 1.0):
            errors.append(f"semantic_loss={cp.semantic_loss} outside [0, 1]")
        if not (0.0 <= cp.capability_alignment <= 1.0):
            errors.append(f"capability_alignment={cp.capability_alignment} outside [0, 1]")
    
    # Check signal freshness
    if signals.signal_freshness_ms is not None and signals.signal_freshness_ms < 0:
        errors.append("signal_freshness_ms cannot be negative")
    
    return len(errors) == 0, errors


def export_events_to_dict(events: List[GovernanceEvent]) -> List[Dict[str, Any]]:
    """Export events to JSON-serializable format."""
    return [
        {
            "timestamp": e.timestamp,
            "event_type": e.event_type,
            "from_level": int(e.from_level),
            "to_level": int(e.to_level),
            "source_ecosystem": e.source_ecosystem,
            "target_ecosystem": e.target_ecosystem,
            "data_category": e.data_category,
            "semantic_fidelity": e.semantic_fidelity,
            "consent_valid": e.consent_valid,
            "privacy_compliant": e.privacy_compliant,
            "reasons": e.reasons,
            "metadata": e.metadata,
            "safety_status": e.safety_status,
            "used_fallback": e.used_fallback,
        }
        for e in events
    ]


def get_translation_path_info(source: str, target: str) -> Dict[str, Any]:
    """Get information about a translation path between ecosystems."""
    source_caps = KNOWN_ECOSYSTEMS.get(source)
    target_caps = KNOWN_ECOSYSTEMS.get(target)
    
    if not source_caps or not target_caps:
        return {
            "valid": False,
            "reason": "Unknown ecosystem",
        }
    
    # Find common data categories
    common_categories = source_caps.supported_categories & target_caps.supported_categories
    
    # Check bidirectionality
    bidirectional = source_caps.bidirectional and target_caps.bidirectional
    
    # Estimate base fidelity from capability overlap
    if len(source_caps.supported_categories) > 0:
        overlap_ratio = len(common_categories) / len(source_caps.supported_categories)
    else:
        overlap_ratio = 0.0
    
    return {
        "valid": True,
        "source": source,
        "target": target,
        "common_categories": list(common_categories),
        "bidirectional": bidirectional,
        "estimated_fidelity": overlap_ratio,
        "source_gaps": source_caps.semantic_gaps,
        "target_gaps": target_caps.semantic_gaps,
        "realtime_possible": source_caps.realtime_available and target_caps.realtime_available,
    }


__version__ = "1.0.0"
__all__ = [
    "TranslationTrustLevel",
    "CrossEcosystemSignals",
    "ConsentStatus",
    "PrivacyBoundaries",
    "ContextPreservation",
    "EcosystemCapabilities",
    "TranslationPath",
    "GovernanceEvent",
    "CrossEcosystemPolicy",
    "CrossEcosystemGovernor",
    "KNOWN_ECOSYSTEMS",
    "default_cross_ecosystem_config",
    "create_default_governor",
    "create_health_data_signals",
    "create_wearable_continuity_signals",
    "create_consent_violation_signals",
    "create_low_fidelity_signals",
    "validate_signals",
    "export_events_to_dict",
    "get_translation_path_info",
]


if __name__ == "__main__":
    print("=" * 80)
    print("AILEE Cross-Ecosystem Translation Governance - Demo")
    print("=" * 80)
    print()
    
    print("Creating governor...")
    governor = create_default_governor()
    print(f"✓ Governor created with max level: CONSTRAINED_TRUST")
    print()
    
    # Test scenarios
    scenarios = [
        {
            "name": "✅ Valid health data translation (iOS → Android)",
            "signals": create_health_data_signals(),
        },
        {
            "name": "✅ Wearable continuity (Apple Watch → Wear OS)",
            "signals": create_wearable_continuity_signals(),
        },
        {
            "name": "❌ Consent violation (Fitbit → Garmin)",
            "signals": create_consent_violation_signals(),
        },
        {
            "name": "⚠️  Low fidelity translation (Fitbit → iOS)",
            "signals": create_low_fidelity_signals(),
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"--- Scenario {i}: {scenario['name']} ---")
        
        signals = scenario['signals']
        level, decision = governor.evaluate(signals)
        
        print(f"  Source: {signals.source_ecosystem}")
        print(f"  Target: {signals.target_ecosystem}")
        print(f"  Data category: {signals.data_category}")
        print(f"  Semantic fidelity: {signals.semantic_fidelity:.2f}")
        print(f"  Desired level: {signals.desired_level.name}")
        print(f"  Authorized level: {level.name}")
        
        if signals.consent_status:
            print(f"  Consent granted: {signals.consent_status.user_consent_granted}")
        
        if decision.reasons:
            print(f"  Key reasons:")
            for reason in decision.reasons[:3]:
                print(f"    • {reason}")
        
        print()
    
    # Show translation path information
    print("--- Translation Path Analysis ---")
    paths = [
        ("ios_healthkit", "android_health_connect"),
        ("apple_watch", "wear_os"),
        ("fitbit", "garmin"),
    ]
    
    for source, target in paths:
        info = get_translation_path_info(source, target)
        if info["valid"]:
            print(f"{source} → {target}:")
            print(f"  Common categories: {', '.join(info['common_categories'][:5])}")
            print(f"  Estimated fidelity: {info['estimated_fidelity']:.2f}")
            print(f"  Bidirectional: {info['bidirectional']}")
            print(f"  Real-time possible: {info['realtime_possible']}")
        print()
    
    # Show event log summary
    print(f"--- Event Log ({len(governor.get_event_log())} events) ---")
    for event in governor.get_event_log():
        status_icon = "✅" if event.to_level >= TranslationTrustLevel.CONSTRAINED_TRUST else "❌"
        print(
            f"{status_icon} [{event.event_type}] {event.source_ecosystem}→{event.target_ecosystem}: "
            f"{event.from_level.name}→{event.to_level.name} (fidelity={event.semantic_fidelity:.2f})"
        )
    
    print()
    print("=" * 80)
    print("Demo complete.")
    print("=" * 80)
    print()
    print("Key Principles Demonstrated:")
    print("  • Semantic translation governs meaning, not mechanisms")
    print("  • Explicit consent is mandatory for all translations")
    print("  • Privacy boundaries are strictly enforced")
    print("  • Low fidelity translations are downgraded or blocked")
    print("  • Full audit trail for compliance verification")
    print("=" * 80)
