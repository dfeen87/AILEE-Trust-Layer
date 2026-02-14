"""
AILEE Trust Layer — NEURO-ASSISTIVE Domain
Version: 1.0.0 - Production Grade

Neuro-assistive and cognitive stability governance for AI systems that assist
human cognition, communication, and perception while preserving autonomy,
consent, identity, and dignity.

Core principle: Stabilizing companion, not cognitive authority.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time
import statistics


# ---- Core imports ----
try:
    from ...ailee_trust_pipeline_v1 import (
        AileeTrustPipeline,
        AileeConfig,
        DecisionResult,
        SafetyStatus
    )
except Exception:
    AileeTrustPipeline = None
    AileeConfig = None
    DecisionResult = None
    SafetyStatus = None


# ===== SECTION 1: ENUMERATIONS =====

class CognitiveState(str, Enum):
    """Current cognitive state of user"""
    BASELINE_STABLE = "BASELINE_STABLE"
    ELEVATED_LOAD = "ELEVATED_LOAD"
    ACUTE_DISTRESS = "ACUTE_DISTRESS"
    IMPAIRED_CAPACITY = "IMPAIRED_CAPACITY"
    CRISIS_MODE = "CRISIS_MODE"
    RECOVERY_STABLE = "RECOVERY_STABLE"
    UNKNOWN = "UNKNOWN"


class AssistanceLevel(str, Enum):
    """Graduated assistance authority levels"""
    NO_ASSISTANCE = "NO_ASSISTANCE"
    PASSIVE_MONITORING = "PASSIVE_MONITORING"
    SIMPLE_PROMPTS = "SIMPLE_PROMPTS"
    GUIDED_OPTIONS = "GUIDED_OPTIONS"
    FULL_ASSISTANCE = "FULL_ASSISTANCE"
    EMERGENCY_SIMPLIFICATION = "EMERGENCY_SIMPLIFICATION"


class AssistanceOutcome(str, Enum):
    """Decision outcomes for assistance requests"""
    DENIED = "DENIED"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    PROMPT_APPROVED = "PROMPT_APPROVED"
    GUIDANCE_APPROVED = "GUIDANCE_APPROVED"
    FULL_APPROVED = "FULL_APPROVED"
    EMERGENCY_SIMPLIFICATION_ACTIVE = "EMERGENCY_SIMPLIFICATION_ACTIVE"
    CONSENT_REQUIRED = "CONSENT_REQUIRED"
    BREAK_REQUIRED = "BREAK_REQUIRED"


class ImpairmentCategory(str, Enum):
    """User impairment classifications"""
    NONE = "NONE"
    APHASIA = "APHASIA"
    TBI_STROKE = "TBI_STROKE"
    NEURODEVELOPMENTAL = "NEURODEVELOPMENTAL"
    NEURODEGENERATIVE = "NEURODEGENERATIVE"
    TEMPORARY_IMPAIRMENT = "TEMPORARY_IMPAIRMENT"
    UNKNOWN = "UNKNOWN"


class ConsentStatus(str, Enum):
    """Consent state for features"""
    GRANTED = "GRANTED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"
    REVOKED = "REVOKED"


# ===== SECTION 2: CORE DATA STRUCTURES =====

@dataclass(frozen=True)
class ConsentRecord:
    """Individual consent record for a feature"""
    feature_name: str
    status: ConsentStatus
    
    granted_at: Optional[float] = None
    expires_at: Optional[float] = None
    revoked_at: Optional[float] = None
    
    requires_periodic_reaffirmation: bool = False
    reaffirmation_interval_days: Optional[int] = None
    last_reaffirmed_at: Optional[float] = None
    
    can_be_revoked: bool = True
    revocation_acknowledged: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self, current_time: float) -> Tuple[bool, str]:
        """Check if consent is currently valid"""
        if self.status != ConsentStatus.GRANTED:
            return False, f"consent_status={self.status.value}"
        
        if self.expires_at and current_time > self.expires_at:
            return False, "consent_expired"
        
        if self.requires_periodic_reaffirmation and self.reaffirmation_interval_days:
            if self.last_reaffirmed_at:
                days_since = (current_time - self.last_reaffirmed_at) / 86400
                if days_since > self.reaffirmation_interval_days:
                    return False, "reaffirmation_required"
            else:
                return False, "reaffirmation_required"
        
        return True, "valid"


@dataclass(frozen=True)
class InterpretationResult:
    """Result of interpreting user input"""
    raw_input: str
    interpreted_intent: str
    confidence: float
    
    ambiguity_detected: bool = False
    ambiguity_flags: List[str] = field(default_factory=list)
    alternative_interpretations: List[str] = field(default_factory=list)
    
    disambiguation_offered: bool = False
    user_confirmed: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CognitiveLoadMetrics:
    """Metrics for cognitive load assessment"""
    estimated_load: float  # [0-1]
    load_confidence: float
    
    response_latency_seconds: Optional[float] = None
    error_rate: Optional[float] = None
    interaction_complexity: Optional[float] = None
    
    load_trend: Optional[str] = None  # "increasing", "stable", "decreasing"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionMetrics:
    """Current session tracking metrics"""
    session_duration_minutes: float
    time_since_last_break_minutes: float
    
    assistance_events_count: int
    user_initiated_requests: int
    system_proactive_suggestions: int
    
    user_acceptance_rate: float
    user_rejection_rate: float
    
    consecutive_sessions_today: int
    total_interaction_minutes_today: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_over_assisting(
        self,
        max_proactive_ratio: float = 0.5,
        max_rejection_rate: float = 0.60
    ) -> Tuple[bool, str]:
        """Check if system is over-assisting"""
        if self.user_initiated_requests > 0:
            proactive_ratio = self.system_proactive_suggestions / self.user_initiated_requests
            if proactive_ratio > max_proactive_ratio:
                return True, f"proactive_ratio={proactive_ratio:.2f} exceeds {max_proactive_ratio}"
        
        if self.user_rejection_rate > max_rejection_rate:
            return True, f"rejection_rate={self.user_rejection_rate:.2f} exceeds {max_rejection_rate}"
        
        return False, "assistance_level_appropriate"


@dataclass(frozen=True)
class TemporalSafeguards:
    """Temporal safety limits"""
    max_session_duration_minutes: float = 30.0
    required_break_minutes: float = 10.0
    max_daily_interaction_minutes: float = 180.0
    
    consecutive_session_limit: int = 4
    escalation_cooldown_minutes: float = 60.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== SECTION 3: INPUT SIGNALS =====

@dataclass(frozen=True)
class NeuroSignals:
    """Input signals for neuro-assistive governance"""
    # Primary scores
    assistance_trust_score: float  # [0-1] From assistance model
    interpretation_confidence: float  # [0-1] How well we understood input
    
    # User input
    user_input: str
    interpreted_intent: str
    input_modality: str  # "text", "speech", "gesture"
    
    # Cognitive state
    current_cognitive_state: CognitiveState
    feature_requested: str
    cognitive_load_metrics: Optional[CognitiveLoadMetrics] = None
    
    # Session context
    session_metrics: Optional[SessionMetrics] = None
    
    # Interpretation
    interpretation_result: Optional[InterpretationResult] = None
    
    # Consent
    consent_records: Dict[str, ConsentRecord] = field(default_factory=dict)
    
    # User profile
    impairment_category: ImpairmentCategory = ImpairmentCategory.NONE
    baseline_cognitive_state: CognitiveState = CognitiveState.BASELINE_STABLE
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())


# ===== SECTION 4: OUTPUT STRUCTURES =====

@dataclass(frozen=True)
class AssistanceConstraints:
    """Constraints on assistance delivery"""
    max_suggestions: int = 3
    require_user_confirmation: bool = False
    
    allowed_modalities: List[str] = field(default_factory=lambda: ["text"])
    complexity_limit: Optional[str] = None  # "minimal", "simple", "moderate", "full"
    
    interaction_timeout_seconds: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuroDecisionResult:
    """Neuro-assistive decision result"""
    assistance_authorized: bool
    assistance_level: AssistanceLevel
    decision_outcome: AssistanceOutcome
    
    validated_trust_score: float
    confidence_score: float
    
    recommendation: str
    reasons: List[str]
    
    assistance_constraints: Optional[AssistanceConstraints] = None
    
    requires_break: bool = False
    requires_consent: bool = False
    requires_disambiguation: bool = False
    
    ailee_result: Optional[DecisionResult] = None
    
    risk_level: Optional[str] = None
    safety_flags: Optional[List[str]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuroEvent:
    """Event log for neuro-assistive decisions"""
    timestamp: float
    event_type: str
    
    cognitive_state: CognitiveState
    assistance_level: AssistanceLevel
    decision_outcome: AssistanceOutcome
    
    assistance_trust_score: float
    interpretation_confidence: float
    
    reasons: List[str]
    
    impairment_category: ImpairmentCategory = ImpairmentCategory.NONE
    
    cognitive_load_metrics: Optional[CognitiveLoadMetrics] = None
    session_metrics: Optional[SessionMetrics] = None
    interpretation_result: Optional[InterpretationResult] = None
    
    ailee_decision: Optional[DecisionResult] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== SECTION 5: POLICY CONFIGURATION =====

@dataclass(frozen=True)
class NeuroAssistivePolicy:
    """Domain policy for neuro-assistive governance"""
    impairment_category: ImpairmentCategory = ImpairmentCategory.NONE
    
    # Trust thresholds
    min_assistance_trust_score: float = 0.70
    min_interpretation_confidence: float = 0.70
    min_cognitive_load_threshold: float = 0.70
    
    # Authority limits
    max_assistance_level: AssistanceLevel = AssistanceLevel.FULL_ASSISTANCE
    
    # Throttling
    max_suggestions_per_hour: int = 10
    max_proactive_to_reactive_ratio: float = 0.5
    max_rejection_rate_threshold: float = 0.60
    
    # Temporal safeguards
    temporal_safeguards: TemporalSafeguards = field(default_factory=TemporalSafeguards)
    
    # Interpretation
    require_disambiguation: bool = True
    require_user_confirmation_on_ambiguity: bool = True
    
    # Consent
    consent_expiry_days: int = 90
    require_periodic_reaffirmation: bool = True
    reaffirmation_interval_days: int = 30
    
    # Graceful degradation - RENAMED for clarity
    emergency_load_threshold: float = 0.85  # Load above this triggers emergency mode
    graceful_degradation_enabled: bool = True
    
    # Authority thresholds by cognitive state
    state_authority_ceilings: Dict[CognitiveState, AssistanceLevel] = field(default_factory=lambda: {
        CognitiveState.CRISIS_MODE: AssistanceLevel.EMERGENCY_SIMPLIFICATION,
        CognitiveState.ACUTE_DISTRESS: AssistanceLevel.SIMPLE_PROMPTS,
        CognitiveState.IMPAIRED_CAPACITY: AssistanceLevel.GUIDED_OPTIONS,
        CognitiveState.ELEVATED_LOAD: AssistanceLevel.FULL_ASSISTANCE,
        CognitiveState.BASELINE_STABLE: AssistanceLevel.FULL_ASSISTANCE,
        CognitiveState.RECOVERY_STABLE: AssistanceLevel.GUIDED_OPTIONS,
        CognitiveState.UNKNOWN: AssistanceLevel.PASSIVE_MONITORING,
    })
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== SECTION 6: POLICY EVALUATOR =====

class PolicyEvaluator:
    """Evaluate policy gates for neuro-assistive decisions"""
    
    def __init__(self, policy: NeuroAssistivePolicy):
        self.policy = policy
    
    def check_consent_gates(
        self,
        signals: NeuroSignals,
        current_time: float
    ) -> Tuple[bool, List[str]]:
        """Check consent status"""
        issues = []
        
        feature = signals.feature_requested
        if feature not in signals.consent_records:
            issues.append(f"no_consent_record_for_{feature}")
            return False, issues
        
        consent = signals.consent_records[feature]
        valid, reason = consent.is_valid(current_time)
        
        if not valid:
            issues.append(f"consent_invalid: {reason}")
        
        return valid, issues
    
    def check_interpretation_gates(
        self,
        signals: NeuroSignals
    ) -> Tuple[bool, List[str]]:
        """Check interpretation quality"""
        issues = []
        
        if signals.interpretation_confidence < self.policy.min_interpretation_confidence:
            issues.append(
                f"interpretation_confidence={signals.interpretation_confidence:.2f} "
                f"below {self.policy.min_interpretation_confidence}"
            )
        
        if signals.interpretation_result:
            interp = signals.interpretation_result
            if interp.ambiguity_detected and self.policy.require_disambiguation:
                if not interp.user_confirmed:
                    issues.append("ambiguity_detected_requires_user_confirmation")
        
        return len(issues) == 0, issues
    
    def check_cognitive_load_gates(
        self,
        signals: NeuroSignals
    ) -> Tuple[bool, List[str], bool]:
        """Check cognitive load status"""
        issues = []
        is_emergency = False
        
        if not signals.cognitive_load_metrics:
            return True, [], False
        
        load = signals.cognitive_load_metrics.estimated_load
        
        if load > self.policy.min_cognitive_load_threshold:
            issues.append(f"cognitive_load_elevated ({load:.2f})")
        
        if load > self.policy.emergency_load_threshold:
            is_emergency = True
            issues.append(f"cognitive_load_critical ({load:.2f}) - emergency_simplification_required")
        
        return len(issues) == 0, issues, is_emergency
    
    def check_temporal_gates(
        self,
        signals: NeuroSignals
    ) -> Tuple[bool, List[str]]:
        """Check temporal safety limits"""
        issues = []
        
        if not signals.session_metrics:
            return True, []
        
        session = signals.session_metrics
        safeguards = self.policy.temporal_safeguards
        
        if session.session_duration_minutes > safeguards.max_session_duration_minutes:
            issues.append(
                f"session_duration={session.session_duration_minutes:.1f}min "
                f"exceeds {safeguards.max_session_duration_minutes}min"
            )
        
        if session.time_since_last_break_minutes > safeguards.max_session_duration_minutes:
            issues.append(
                f"time_since_break={session.time_since_last_break_minutes:.1f}min "
                f"requires break"
            )
        
        if session.total_interaction_minutes_today > safeguards.max_daily_interaction_minutes:
            issues.append(
                f"daily_interaction={session.total_interaction_minutes_today:.1f}min "
                f"exceeds {safeguards.max_daily_interaction_minutes}min"
            )
        
        if session.consecutive_sessions_today >= safeguards.consecutive_session_limit:
            issues.append(
                f"consecutive_sessions={session.consecutive_sessions_today} "
                f"exceeds limit of {safeguards.consecutive_session_limit}"
            )
        
        return len(issues) == 0, issues
    
    def check_over_assistance(
        self,
        signals: NeuroSignals
    ) -> Tuple[bool, List[str]]:
        """Check for over-assistance patterns"""
        issues = []
        
        if not signals.session_metrics:
            return True, []
        
        over_assisting, reason = signals.session_metrics.is_over_assisting(
            max_proactive_ratio=self.policy.max_proactive_to_reactive_ratio,
            max_rejection_rate=self.policy.max_rejection_rate_threshold
        )
        
        if over_assisting:
            issues.append(reason)
        
        return not over_assisting, issues


# ===== SECTION 7: COGNITIVE STATE TRACKER =====

class CognitiveStateTracker:
    """Track cognitive state transitions with hysteresis"""
    
    def __init__(self):
        self._current_state: CognitiveState = CognitiveState.UNKNOWN
        self._last_transition_time: Optional[float] = None
        self._min_state_duration_seconds: float = 120.0  # 2 minutes minimum
        self._state_history: List[Tuple[float, CognitiveState, float]] = []
    
    def get_current_state(self) -> CognitiveState:
        """Get current state"""
        return self._current_state
    
    def update_state(
        self,
        signals: NeuroSignals,
        force_transition: bool = False
    ) -> CognitiveState:
        """Update state with hysteresis"""
        
        candidate_state = signals.current_cognitive_state
        
        # If explicit state provided, use it
        if candidate_state != CognitiveState.UNKNOWN:
            return self._apply_hysteresis(candidate_state, signals.timestamp, force_transition)
        
        # Otherwise infer from cognitive load
        if signals.cognitive_load_metrics:
            load = signals.cognitive_load_metrics.estimated_load
            candidate_state = self._infer_state_from_load(load, signals)
            return self._apply_hysteresis(candidate_state, signals.timestamp, force_transition)
        
        return self._current_state
    
    def _infer_state_from_load(
        self,
        load: float,
        signals: NeuroSignals
    ) -> CognitiveState:
        """Infer state from cognitive load"""
        
        if load > 0.90:
            return CognitiveState.CRISIS_MODE
        elif load > 0.75:
            return CognitiveState.ACUTE_DISTRESS
        elif load > 0.60:
            return CognitiveState.ELEVATED_LOAD
        elif load < 0.30:
            return CognitiveState.BASELINE_STABLE
        else:
            return CognitiveState.UNKNOWN
    
    def _apply_hysteresis(
        self,
        candidate_state: CognitiveState,
        timestamp: float,
        force: bool
    ) -> CognitiveState:
        """Apply hysteresis to prevent rapid state transitions"""
        
        if self._last_transition_time is None:
            self._current_state = candidate_state
            self._last_transition_time = timestamp
            return candidate_state
        
        if candidate_state == self._current_state:
            return self._current_state
        
        # Fast transitions for crises
        if force or candidate_state == CognitiveState.CRISIS_MODE:
            self._current_state = candidate_state
            self._last_transition_time = timestamp
            return candidate_state
        
        # Otherwise require minimum duration
        time_in_state = timestamp - self._last_transition_time
        if time_in_state < self._min_state_duration_seconds:
            return self._current_state
        
        # Transition
        self._current_state = candidate_state
        self._last_transition_time = timestamp
        return candidate_state


# ===== SECTION 8: NEURO GOVERNOR =====

class NeuroGovernor:
    """Main orchestrator for neuro-assistive governance"""
    
    def __init__(
        self,
        cfg: Optional[AileeConfig] = None,
        policy: Optional[NeuroAssistivePolicy] = None
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable")
        
        self.policy = policy or NeuroAssistivePolicy()
        self.cfg = cfg or self._default_config()
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        self.policy_evaluator = PolicyEvaluator(self.policy)
        self.state_tracker = CognitiveStateTracker()
        
        self._event_log: List[NeuroEvent] = []
        self._last_event: Optional[NeuroEvent] = None
    
    def _default_config(self) -> AileeConfig:
        """Default AILEE configuration for neuro domain"""
        return AileeConfig(
            accept_threshold=0.75,
            borderline_low=0.60,
            borderline_high=0.75,
            
            w_stability=0.45,
            w_agreement=0.35,
            w_likelihood=0.20,
            
            history_window=100,
            forecast_window=20,
            
            grace_peer_delta=0.15,
            grace_min_peer_agreement_ratio=0.65,
            
            enable_grace=True,
            enable_consensus=True,
            enable_audit_metadata=True,
        )
    
    def evaluate(self, signals: NeuroSignals) -> NeuroDecisionResult:
        """Evaluate assistance request"""
        
        ts = float(signals.timestamp)
        reasons: List[str] = []
        safety_flags: List[str] = []
        
        # 1) Update cognitive state
        current_state = self.state_tracker.update_state(signals)
        
        # 2) Check consent
        consent_ok, consent_issues = self.policy_evaluator.check_consent_gates(signals, ts)
        if not consent_ok:
            reasons.extend(consent_issues)
            decision = self._create_denial_decision(
                signals, reasons, ts, "consent_required"
            )
            self._log_event(ts, signals, decision, reasons)
            return decision
        
        # 3) Check interpretation
        interp_ok, interp_issues = self.policy_evaluator.check_interpretation_gates(signals)
        if not interp_ok:
            reasons.extend(interp_issues)
            safety_flags.append("interpretation_uncertain")
        
        # 4) Check cognitive load
        load_ok, load_issues, is_emergency = self.policy_evaluator.check_cognitive_load_gates(signals)
        if not load_ok:
            reasons.extend(load_issues)
            if is_emergency:
                safety_flags.append("cognitive_overload_critical")
        
        # 5) Check temporal
        temporal_ok, temporal_issues = self.policy_evaluator.check_temporal_gates(signals)
        if not temporal_ok:
            reasons.extend(temporal_issues)
            safety_flags.append("temporal_limits_exceeded")
        
        # 6) Check over-assistance
        assist_ok, assist_issues = self.policy_evaluator.check_over_assistance(signals)
        if not assist_ok:
            reasons.extend(assist_issues)
            safety_flags.append("over_assistance_detected")
        
        # 7) Compute authority ceiling from cognitive state
        max_authority = self.policy.state_authority_ceilings.get(
            current_state,
            AssistanceLevel.PASSIVE_MONITORING
        )
        
        # 8) Build context for AILEE
        ctx = self._build_context(signals, reasons, safety_flags, current_state)
        
        # 9) Run through AILEE pipeline
        peer_values = [signals.interpretation_confidence]
        if signals.cognitive_load_metrics:
            peer_values.append(1.0 - signals.cognitive_load_metrics.estimated_load)
        
        ailee_result = self.pipeline.process(
            raw_value=float(signals.assistance_trust_score),
            raw_confidence=float(signals.interpretation_confidence),
            peer_values=peer_values,
            timestamp=ts,
            context=ctx
        )
        
        # 10) Make decision
        decision = self._make_assistance_decision(
            signals, ailee_result, reasons, safety_flags,
            is_emergency, max_authority, ts
        )
        
        # 11) Log
        self._log_event(ts, signals, decision, reasons)
        
        return decision
    
    def _build_context(
        self,
        signals: NeuroSignals,
        reasons: List[str],
        safety_flags: List[str],
        current_state: CognitiveState
    ) -> Dict[str, Any]:
        """Build context dictionary"""
        ctx = {
            "cognitive_state": current_state.value,
            "impairment_category": signals.impairment_category.value,
            "interpretation_confidence": signals.interpretation_confidence,
            "feature_requested": signals.feature_requested,
            "reasons": reasons[:],
            "safety_flags": safety_flags[:],
        }
        
        if signals.cognitive_load_metrics:
            ctx["cognitive_load"] = signals.cognitive_load_metrics.estimated_load
        
        if signals.session_metrics:
            ctx["session_duration"] = signals.session_metrics.session_duration_minutes
            ctx["rejection_rate"] = signals.session_metrics.user_rejection_rate
        
        ctx.update(signals.context)
        return ctx
    
    def _make_assistance_decision(
        self,
        signals: NeuroSignals,
        ailee_result: DecisionResult,
        reasons: List[str],
        safety_flags: List[str],
        is_emergency: bool,
        max_authority: AssistanceLevel,
        ts: float
    ) -> NeuroDecisionResult:
        """Convert AILEE result to assistance decision"""
        
        validated_score = ailee_result.validated_value
        
        # Determine base authority
        if is_emergency:
            authority = AssistanceLevel.EMERGENCY_SIMPLIFICATION
            outcome = AssistanceOutcome.EMERGENCY_SIMPLIFICATION_ACTIVE
            authorized = True
            recommendation = "emergency_simplification_minimal_interaction"
        
        elif validated_score < 0.60:
            authority = AssistanceLevel.NO_ASSISTANCE
            outcome = AssistanceOutcome.DENIED
            authorized = False
            recommendation = "assistance_denied_insufficient_trust"
        
        elif validated_score < 0.70:
            authority = AssistanceLevel.PASSIVE_MONITORING
            outcome = AssistanceOutcome.OBSERVE_ONLY
            authorized = False
            recommendation = "observe_only_no_assistance"
        
        elif validated_score < 0.80:
            authority = AssistanceLevel.SIMPLE_PROMPTS
            outcome = AssistanceOutcome.PROMPT_APPROVED
            authorized = True
            recommendation = "simple_prompts_approved"
        
        elif validated_score < 0.90:
            authority = AssistanceLevel.GUIDED_OPTIONS
            outcome = AssistanceOutcome.GUIDANCE_APPROVED
            authorized = True
            recommendation = "guided_options_approved"
        
        else:
            authority = AssistanceLevel.FULL_ASSISTANCE
            outcome = AssistanceOutcome.FULL_APPROVED
            authorized = True
            recommendation = "full_assistance_approved"
        
        # Apply authority ceiling
        authority_index = {
            AssistanceLevel.NO_ASSISTANCE: 0,
            AssistanceLevel.PASSIVE_MONITORING: 1,
            AssistanceLevel.SIMPLE_PROMPTS: 2,
            AssistanceLevel.GUIDED_OPTIONS: 3,
            AssistanceLevel.FULL_ASSISTANCE: 4,
            AssistanceLevel.EMERGENCY_SIMPLIFICATION: 5,
        }
        
        if authority_index[authority] > authority_index[max_authority]:
            authority = max_authority
            recommendation = f"{recommendation}_capped_by_cognitive_state"
        
        # Check for required actions
        requires_break = "temporal_limits_exceeded" in safety_flags
        requires_consent = "consent_invalid" in " ".join(reasons)
        requires_disambiguation = "ambiguity_detected" in " ".join(reasons)
        
        # Generate constraints
        constraints = self._generate_constraints(signals, authority, safety_flags)
        
        # Risk level - ENHANCED with cognitive load consideration
        risk_level = self._compute_risk_level(validated_score, signals)
        
        return NeuroDecisionResult(
            assistance_authorized=authorized,
            assistance_level=authority,
            decision_outcome=outcome,
            validated_trust_score=validated_score,
            confidence_score=ailee_result.confidence_score,
            recommendation=recommendation,
            reasons=reasons[:],
            assistance_constraints=constraints,
            requires_break=requires_break,
            requires_consent=requires_consent,
            requires_disambiguation=requires_disambiguation,
            ailee_result=ailee_result,
            risk_level=risk_level,
            safety_flags=safety_flags[:],
            metadata={
                "timestamp": ts,
                "cognitive_state": signals.current_cognitive_state.value,
                "impairment_category": signals.impairment_category.value,
                "authority_ceiling": max_authority.value,
            }
        )
    
    def _compute_risk_level(
        self,
        validated_score: float,
        signals: NeuroSignals
    ) -> str:
        """
        Compute risk level considering both trust score and cognitive load.
        Multi-axis risk assessment for comprehensive safety.
        """
        # Trust-based risk
        if validated_score >= 0.85:
            trust_risk = "low"
        elif validated_score >= 0.75:
            trust_risk = "moderate"
        elif validated_score >= 0.60:
            trust_risk = "elevated"
        else:
            trust_risk = "high"
        
        # Cognitive load-based risk
        load_risk = "low"
        if signals.cognitive_load_metrics:
            load = signals.cognitive_load_metrics.estimated_load
            if load >= 0.85:
                load_risk = "high"
            elif load >= 0.70:
                load_risk = "elevated"
            elif load >= 0.50:
                load_risk = "moderate"
        
        # Combined risk (take the higher of the two)
        risk_levels = {"low": 0, "moderate": 1, "elevated": 2, "high": 3}
        trust_level = risk_levels[trust_risk]
        load_level = risk_levels[load_risk]
        
        combined_level = max(trust_level, load_level)
        
        for risk_name, risk_value in risk_levels.items():
            if risk_value == combined_level:
                return risk_name
        
        return "moderate"  # Fallback
    
    def _generate_constraints(
        self,
        signals: NeuroSignals,
        authority: AssistanceLevel,
        safety_flags: List[str]
    ) -> AssistanceConstraints:
        """Generate assistance constraints based on authority level"""
        
        if authority == AssistanceLevel.EMERGENCY_SIMPLIFICATION:
            constraints = AssistanceConstraints(
                max_suggestions=1,
                require_user_confirmation=False,
                allowed_modalities=["text"],
                complexity_limit="minimal",
                interaction_timeout_seconds=10.0
            )
        
        elif authority == AssistanceLevel.SIMPLE_PROMPTS:
            constraints = AssistanceConstraints(
                max_suggestions=2,
                require_user_confirmation=True,
                complexity_limit="simple"
            )
        
        elif authority == AssistanceLevel.GUIDED_OPTIONS:
            constraints = AssistanceConstraints(
                max_suggestions=3,
                require_user_confirmation=False,
                complexity_limit="moderate"
            )
        
        else:
            # FULL_ASSISTANCE or default
            constraints = AssistanceConstraints(
                max_suggestions=5,
                require_user_confirmation=False,
                complexity_limit="full"
            )
        
        # Override if interpretation uncertain
        if "interpretation_uncertain" in safety_flags:
            # Create new constraints with updated confirmation requirement
            constraints = AssistanceConstraints(
                max_suggestions=constraints.max_suggestions,
                require_user_confirmation=True,
                allowed_modalities=constraints.allowed_modalities,
                complexity_limit=constraints.complexity_limit,
                interaction_timeout_seconds=constraints.interaction_timeout_seconds
            )
        
        return constraints
    
    def _create_denial_decision(
        self,
        signals: NeuroSignals,
        reasons: List[str],
        ts: float,
        denial_type: str
    ) -> NeuroDecisionResult:
        """Create denial decision"""
        return NeuroDecisionResult(
            assistance_authorized=False,
            assistance_level=AssistanceLevel.NO_ASSISTANCE,
            decision_outcome=AssistanceOutcome.DENIED,
            validated_trust_score=0.0,
            confidence_score=1.0,
            recommendation=f"assistance_denied_{denial_type}",
            reasons=reasons[:],
            risk_level="high",
            safety_flags=[denial_type],
            metadata={"timestamp": ts}
        )
    
    def _log_event(
        self,
        ts: float,
        signals: NeuroSignals,
        decision: NeuroDecisionResult,
        reasons: List[str]
    ):
        """Log decision event"""
        
        event_type_map = {
            AssistanceOutcome.DENIED: "assistance_denied",
            AssistanceOutcome.OBSERVE_ONLY: "observation_only",
            AssistanceOutcome.PROMPT_APPROVED: "prompt_approved",
            AssistanceOutcome.GUIDANCE_APPROVED: "guidance_approved",
            AssistanceOutcome.FULL_APPROVED: "full_assistance_approved",
            AssistanceOutcome.EMERGENCY_SIMPLIFICATION_ACTIVE: "emergency_simplification",
            AssistanceOutcome.CONSENT_REQUIRED: "consent_required",
            AssistanceOutcome.BREAK_REQUIRED: "break_required",
        }
        
        event = NeuroEvent(
            timestamp=ts,
            event_type=event_type_map.get(decision.decision_outcome, "unknown"),
            cognitive_state=signals.current_cognitive_state,
            assistance_level=decision.assistance_level,
            decision_outcome=decision.decision_outcome,
            assistance_trust_score=decision.validated_trust_score,
            interpretation_confidence=signals.interpretation_confidence,
            reasons=reasons[:],
            impairment_category=signals.impairment_category,
            cognitive_load_metrics=signals.cognitive_load_metrics,
            session_metrics=signals.session_metrics,
            interpretation_result=signals.interpretation_result,
            ailee_decision=decision.ailee_result,
            metadata=decision.metadata
        )
        
        self._event_log.append(event)
        self._last_event = event
    
    # -------------------------
    # Public API
    # -------------------------
    
    def explain_decision(self, decision: NeuroDecisionResult) -> str:
        """Generate plain-language explanation"""
        lines = []
        
        lines.append("=" * 70)
        lines.append("NEURO-ASSISTIVE DECISION EXPLANATION")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"DECISION: {decision.decision_outcome.value}")
        lines.append(f"Assistance Level: {decision.assistance_level.value}")
        lines.append(f"Assistance Authorized: {'YES' if decision.assistance_authorized else 'NO'}")
        lines.append(f"Risk Level: {decision.risk_level.upper()}")
        lines.append("")
        
        lines.append(f"Validated Trust Score: {decision.validated_trust_score:.2f} / 1.00")
        lines.append(f"Decision Confidence: {decision.confidence_score:.2f} / 1.00")
        lines.append("")
        
        if decision.ailee_result:
            lines.append("WHAT MATTERED MOST:")
            lines.append("-" * 70)
            lines.append(f"• AILEE Pipeline Status: {decision.ailee_result.status.value}")
            if decision.ailee_result.grace_applied:
                lines.append("  → Grace conditions applied (leniency given)")
            if decision.ailee_result.consensus_status:
                lines.append(f"  → Consensus: {decision.ailee_result.consensus_status}")
            lines.append("")
        
        if not decision.assistance_authorized:
            lines.append("WHAT BLOCKED ASSISTANCE:")
            lines.append("-" * 70)
            for reason in decision.reasons:
                lines.append(f"• {reason}")
            lines.append("")
        
        if decision.assistance_constraints:
            lines.append("ASSISTANCE CONSTRAINTS:")
            lines.append("-" * 70)
            c = decision.assistance_constraints
            lines.append(f"• Max Suggestions: {c.max_suggestions}")
            lines.append(f"• Requires User Confirmation: {c.require_user_confirmation}")
            if c.complexity_limit:
                lines.append(f"• Complexity Limit: {c.complexity_limit}")
            if c.interaction_timeout_seconds:
                lines.append(f"• Interaction Timeout: {c.interaction_timeout_seconds}s")
            lines.append("")
        
        if decision.requires_break:
            lines.append("⚠️  BREAK REQUIRED - Temporal safety limits reached")
            lines.append("    Please take a break before continuing interaction.")
            lines.append("")
        
        if decision.requires_disambiguation:
            lines.append("⚠️  USER CONFIRMATION REQUIRED - Input ambiguous")
            lines.append("    Please clarify your intent before proceeding.")
            lines.append("")
        
        if decision.requires_consent:
            lines.append("⚠️  CONSENT REQUIRED - Feature access needs authorization")
            lines.append("    Please grant consent for this feature.")
            lines.append("")
        
        lines.append("WHAT WOULD CHANGE THIS DECISION:")
        lines.append("-" * 70)
        if decision.validated_trust_score < 0.80 and not decision.assistance_authorized:
            gap = 0.80 - decision.validated_trust_score
            lines.append(f"• Increase trust score by {gap:.2f} to enable basic assistance")
        
        if decision.safety_flags:
            lines.append("• Address safety concerns:")
            for flag in decision.safety_flags:
                if "cognitive_overload" in flag:
                    lines.append("  → Reduce cognitive load (take a break, simplify task)")
                elif "interpretation_uncertain" in flag:
                    lines.append("  → Provide clearer input or confirm interpretation")
                elif "temporal_limits" in flag:
                    lines.append("  → Take required break before resuming")
                elif "over_assistance" in flag:
                    lines.append("  → Reduce reliance on system suggestions")
        lines.append("")
        
        lines.append("RECOMMENDED ACTION:")
        lines.append("-" * 70)
        lines.append(f"→ {decision.recommendation.replace('_', ' ').title()}")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_last_event(self) -> Optional[NeuroEvent]:
        """Get last event"""
        return self._last_event
    
    def export_events(self, since_ts: Optional[float] = None) -> List[NeuroEvent]:
        """Export events for audit/compliance"""
        if since_ts is None:
            return self._event_log[:]
        return [e for e in self._event_log if e.timestamp >= since_ts]
    
    def get_cognitive_trend(self) -> str:
        """Get cognitive state trend over recent history"""
        if len(self._event_log) < 5:
            return "insufficient_data"
        
        recent_events = self._event_log[-10:]
        
        # Count emergency/distress events
        crisis_count = sum(
            1 for e in recent_events 
            if e.cognitive_state in (CognitiveState.CRISIS_MODE, CognitiveState.ACUTE_DISTRESS)
        )
        
        if crisis_count >= 3:
            return "deteriorating"
        
        # Check load trends if available
        loads = [
            e.cognitive_load_metrics.estimated_load 
            for e in recent_events 
            if e.cognitive_load_metrics
        ]
        
        if len(loads) >= 5:
            recent_avg = sum(loads[-3:]) / 3
            older_avg = sum(loads[:3]) / 3
            
            if recent_avg > older_avg + 0.10:
                return "increasing_load"
            elif recent_avg < older_avg - 0.10:
                return "decreasing_load"
        
        return "stable"
    
    def get_assistance_history(self) -> Dict[str, Any]:
        """Get assistance decision history"""
        approved = [
            e for e in self._event_log
            if e.decision_outcome in (
                AssistanceOutcome.PROMPT_APPROVED,
                AssistanceOutcome.GUIDANCE_APPROVED,
                AssistanceOutcome.FULL_APPROVED
            )
        ]
        
        denied = [
            e for e in self._event_log
            if e.decision_outcome == AssistanceOutcome.DENIED
        ]
        
        emergency = [
            e for e in self._event_log
            if e.decision_outcome == AssistanceOutcome.EMERGENCY_SIMPLIFICATION_ACTIVE
        ]
        
        return {
            "total_requests": len(self._event_log),
            "approved_count": len(approved),
            "denied_count": len(denied),
            "emergency_count": len(emergency),
            "approval_rate": len(approved) / len(self._event_log) if self._event_log else 0.0,
            "average_trust_score": (
                statistics.fmean([e.assistance_trust_score for e in approved]) 
                if approved else 0.0
            )
        }


# ===== CONVENIENCE FUNCTIONS =====

def create_neuro_governor(
    impairment_category: ImpairmentCategory = ImpairmentCategory.NONE,
    **policy_overrides
) -> NeuroGovernor:
    """Factory for creating neuro governor with sensible defaults"""
    policy_kwargs = {"impairment_category": impairment_category}
    
    # Adjust defaults based on impairment category
    if impairment_category in (ImpairmentCategory.TBI_STROKE, ImpairmentCategory.NEURODEGENERATIVE):
        policy_kwargs.update({
            "min_assistance_trust_score": 0.75,
            "min_interpretation_confidence": 0.75,
            "emergency_load_threshold": 0.80,
            "temporal_safeguards": TemporalSafeguards(
                max_session_duration_minutes=20.0,
                required_break_minutes=15.0
            )
        })
    
    elif impairment_category == ImpairmentCategory.APHASIA:
        policy_kwargs.update({
            "min_interpretation_confidence": 0.60,  # Allow lower confidence
            "require_disambiguation": True,
            "require_user_confirmation_on_ambiguity": True
        })
    
    policy_kwargs.update(policy_overrides)
    policy = NeuroAssistivePolicy(**policy_kwargs)
    
    return NeuroGovernor(policy=policy)


def validate_neuro_signals(signals: NeuroSignals) -> Tuple[bool, List[str]]:
    """Validate neuro signals structure"""
    issues: List[str] = []
    
    if not (0.0 <= signals.assistance_trust_score <= 1.0):
        issues.append(f"assistance_trust_score={signals.assistance_trust_score} outside [0, 1]")
    
    if not (0.0 <= signals.interpretation_confidence <= 1.0):
        issues.append(f"interpretation_confidence={signals.interpretation_confidence} outside [0, 1]")
    
    if signals.cognitive_load_metrics:
        if not (0.0 <= signals.cognitive_load_metrics.estimated_load <= 1.0):
            issues.append("cognitive_load outside [0, 1]")
        if not (0.0 <= signals.cognitive_load_metrics.load_confidence <= 1.0):
            issues.append("load_confidence outside [0, 1]")
    
    if signals.session_metrics:
        if not (0.0 <= signals.session_metrics.user_acceptance_rate <= 1.0):
            issues.append("user_acceptance_rate outside [0, 1]")
        if not (0.0 <= signals.session_metrics.user_rejection_rate <= 1.0):
            issues.append("user_rejection_rate outside [0, 1]")
    
    return len(issues) == 0, issues


# ===== MODULE EXPORTS =====

__all__ = [
    # Enums
    "CognitiveState",
    "AssistanceLevel",
    "AssistanceOutcome",
    "ImpairmentCategory",
    "ConsentStatus",
    
    # Data structures
    "ConsentRecord",
    "InterpretationResult",
    "CognitiveLoadMetrics",
    "SessionMetrics",
    "TemporalSafeguards",
    "NeuroSignals",
    "AssistanceConstraints",
    "NeuroDecisionResult",
    "NeuroEvent",
    
    # Configuration
    "NeuroAssistivePolicy",
    
    # Components
    "PolicyEvaluator",
    "CognitiveStateTracker",
    "NeuroGovernor",
    
    # Utilities
    "create_neuro_governor",
    "validate_neuro_signals",
]
