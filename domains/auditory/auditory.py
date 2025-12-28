"""
AILEE Trust Layer — AUDITORY Domain
Version: 1.0.1 - Production Grade (CORRECTED)

All critical syntax errors fixed:
- All class methods properly indented
- AuditoryUncertaintyCalculator properly structured
- compute_precautionary_penalty moved back into AuditoryPolicyEvaluator
- check_environmental_gates has proper return path
- explain_decision moved into AuditoryGovernor
- __all__ export list added
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple
import statistics
import time
import math


# ---- Core imports ----
try:
    from ailee_trust_pipeline_v1 import (
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


# ===== SEVERITY WEIGHTING FOR FLAGS =====

AUDITORY_FLAG_SEVERITY: Dict[str, float] = {
    "hearing_damage_risk": 0.15,
    "feedback_detected": 0.12,
    "hardware_fault_critical": 0.10,
    "excessive_discomfort": 0.08,
    "aggregate_uncertainty_excessive": 0.08,
    "clipping_detected": 0.07,
    "latency_excessive": 0.06,
    "hardware_degraded": 0.06,
    "speech_unintelligible": 0.05,
    "noisy_environment_restriction": 0.05,
    "battery_critical": 0.04,
    "microphone_degraded": 0.03,
    "noise_reduction_poor": 0.03,
}


# -----------------------------
# Output Authorization Levels
# -----------------------------

class OutputAuthorizationLevel(IntEnum):
    """Discrete output authorization levels for auditory enhancement."""
    NO_OUTPUT = 0
    DIAGNOSTIC_ONLY = 1
    SAFETY_LIMITED = 2
    COMFORT_OPTIMIZED = 3
    FULL_ENHANCEMENT = 4


class ListeningMode(str, Enum):
    """Listening modes for hearing systems"""
    QUIET = "QUIET"
    SPEECH_FOCUS = "SPEECH_FOCUS"
    NOISY = "NOISY"
    MUSIC = "MUSIC"
    OUTDOOR_WINDY = "OUTDOOR_WINDY"
    EMERGENCY_ALERTS = "EMERGENCY_ALERTS"
    TELECOIL = "TELECOIL"
    UNKNOWN = "UNKNOWN"


class UserSafetyProfile(str, Enum):
    """User safety profiles for output constraints"""
    PEDIATRIC = "PEDIATRIC"
    STANDARD = "STANDARD"
    TINNITUS_RISK = "TINNITUS_RISK"
    PROFESSIONAL = "PROFESSIONAL"
    UNKNOWN = "UNKNOWN"


class DecisionOutcome(str, Enum):
    """Auditory governance decision outcomes"""
    AUTHORIZED = "AUTHORIZED"
    LIMITED = "LIMITED"
    DIAGNOSTIC_FALLBACK = "DIAGNOSTIC_FALLBACK"
    SUPPRESSED = "SUPPRESSED"
    HARDWARE_FAULT = "HARDWARE_FAULT"


class RegulatoryGateResult(str, Enum):
    """Regulatory compliance results"""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


# -----------------------------
# Hearing Profile & Safety
# -----------------------------

@dataclass(frozen=True)
class HearingProfile:
    """User hearing profile and safety limits."""
    max_safe_output_db: float
    preferred_output_db: float
    frequency_loss_profile: Optional[Dict[str, float]] = None
    tinnitus_history: bool = False
    noise_induced_loss: bool = False
    age_years: Optional[int] = None
    notes: Optional[str] = None
    last_assessment_date: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_dynamic_safety_margin(self) -> float:
        """Compute safety margin based on risk factors."""
        margin = 0.0
        if self.tinnitus_history:
            margin += 5.0
        if self.noise_induced_loss:
            margin += 3.0
        if self.age_years is not None:
            if self.age_years < 18:
                margin += 10.0
            elif self.age_years > 65:
                margin += 2.0
        return margin


# -----------------------------
# Environmental Metrics
# -----------------------------

@dataclass(frozen=True)
class EnvironmentMetrics:
    """Environmental acoustic metrics."""
    ambient_noise_db: Optional[float] = None
    snr_db: Optional[float] = None
    reverberation_time_s: Optional[float] = None
    transient_noise_score: Optional[float] = None
    wind_level: Optional[float] = None
    scene_confidence: Optional[float] = None
    scene_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_challenging_environment(self) -> Tuple[bool, List[str]]:
        """Check if environment is acoustically challenging"""
        issues: List[str] = []
        
        if self.ambient_noise_db is not None:
            if self.ambient_noise_db > 80.0:
                issues.append(f"high_ambient_noise ({self.ambient_noise_db:.1f} dB)")
        
        if self.snr_db is not None:
            if self.snr_db < 5.0:
                issues.append(f"poor_snr ({self.snr_db:.1f} dB)")
        
        if self.reverberation_time_s is not None:
            if self.reverberation_time_s > 1.0:
                issues.append(f"high_reverberation ({self.reverberation_time_s:.2f}s)")
        
        if self.transient_noise_score is not None:
            if self.transient_noise_score > 0.60:
                issues.append("frequent_transients")
        
        return len(issues) > 0, issues


# -----------------------------
# Enhancement Metrics
# -----------------------------

@dataclass(frozen=True)
class EnhancementMetrics:
    """Model-driven enhancement quality metrics."""
    speech_intelligibility_score: Optional[float] = None
    noise_reduction_score: Optional[float] = None
    spectral_balance_score: Optional[float] = None
    enhancement_latency_ms: Optional[float] = None
    ai_confidence: Optional[float] = None
    beamforming_active: bool = False
    noise_suppression_db: Optional[float] = None
    compression_ratio: Optional[float] = None
    artifacts_detected: bool = False
    distortion_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_quality_acceptable(
        self,
        min_intelligibility: float = 0.65,
        min_noise_reduction: float = 0.55,
        max_latency_ms: float = 20.0
    ) -> Tuple[bool, List[str]]:
        """Check if enhancement quality meets thresholds"""
        issues: List[str] = []
        
        if self.speech_intelligibility_score is not None:
            if self.speech_intelligibility_score < min_intelligibility:
                issues.append(
                    f"speech_intelligibility={self.speech_intelligibility_score:.2f} "
                    f"< {min_intelligibility}"
                )
        
        if self.noise_reduction_score is not None:
            if self.noise_reduction_score < min_noise_reduction:
                issues.append(
                    f"noise_reduction={self.noise_reduction_score:.2f} "
                    f"< {min_noise_reduction}"
                )
        
        if self.enhancement_latency_ms is not None:
            if self.enhancement_latency_ms > max_latency_ms:
                issues.append(
                    f"latency={self.enhancement_latency_ms:.1f}ms > {max_latency_ms}ms"
                )
        
        if self.artifacts_detected:
            issues.append("artifacts_detected_in_output")
        
        return len(issues) == 0, issues


# -----------------------------
# Comfort Metrics
# -----------------------------

@dataclass(frozen=True)
class ComfortMetrics:
    """User comfort and fatigue indicators."""
    perceived_loudness_db: Optional[float] = None
    discomfort_score: Optional[float] = None
    fatigue_risk_score: Optional[float] = None
    volume_adjustments_count: Optional[int] = None
    device_removal_events: Optional[int] = None
    continuous_usage_hours: Optional[float] = None
    time_since_last_break_minutes: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_comfort_acceptable(
        self,
        max_discomfort: float = 0.35,
        max_fatigue: float = 0.60
    ) -> Tuple[bool, List[str]]:
        """Check if comfort metrics are within acceptable range"""
        issues: List[str] = []
        
        if self.discomfort_score is not None:
            if self.discomfort_score > max_discomfort:
                issues.append(
                    f"discomfort={self.discomfort_score:.2f} exceeds {max_discomfort}"
                )
        
        if self.fatigue_risk_score is not None:
            if self.fatigue_risk_score > max_fatigue:
                issues.append(
                    f"fatigue_risk={self.fatigue_risk_score:.2f} exceeds {max_fatigue}"
                )
        
        if self.continuous_usage_hours is not None:
            if self.continuous_usage_hours > 8.0:
                issues.append(
                    f"continuous_usage={self.continuous_usage_hours:.1f}hrs > 8hrs"
                )
        
        return len(issues) == 0, issues


# -----------------------------
# Device Health
# -----------------------------

@dataclass(frozen=True)
class DeviceHealth:
    """Device status and safety indicators."""
    mic_health_score: Optional[float] = None
    speaker_health_score: Optional[float] = None
    battery_level: Optional[float] = None
    temperature_c: Optional[float] = None
    hardware_faults: Tuple[str, ...] = ()
    feedback_detected: bool = False
    occlusion_detected: bool = False
    clipping_detected: bool = False
    last_calibration_timestamp: Optional[float] = None
    calibration_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_health_issues(
        self,
        min_mic_health: float = 0.75,
        min_battery: float = 0.10
    ) -> Tuple[bool, List[str]]:
        """Check for device health issues"""
        issues: List[str] = []
        
        if self.hardware_faults:
            issues.extend([f"hardware_fault: {f}" for f in self.hardware_faults])
        
        if self.feedback_detected:
            issues.append("acoustic_feedback_detected")
        
        if self.clipping_detected:
            issues.append("output_clipping_detected")
        
        if self.occlusion_detected:
            issues.append("ear_canal_occlusion")
        
        if self.mic_health_score is not None:
            if self.mic_health_score < min_mic_health:
                issues.append(f"mic_health={self.mic_health_score:.2f} < {min_mic_health}")
        
        if self.battery_level is not None:
            if self.battery_level < min_battery:
                issues.append(f"battery_critical={self.battery_level:.2%}")
        
        if not self.calibration_valid:
            issues.append("calibration_expired")
        
        return len(issues) > 0, issues


# -----------------------------
# Aggregate Uncertainty
# -----------------------------

@dataclass(frozen=True)
class AuditoryUncertainty:
    """Explicit aggregation of uncertainty sources in auditory domain."""
    aggregate_uncertainty_score: float
    enhancement_uncertainty: float
    environmental_uncertainty: float
    device_uncertainty: float
    comfort_uncertainty: float
    dominant_uncertainty_source: str
    uncertainty_sources: Dict[str, float] = field(default_factory=dict)
    
    def is_uncertainty_acceptable(self, max_aggregate: float = 0.35) -> Tuple[bool, str]:
        """Check if aggregate uncertainty is within acceptable bounds"""
        if self.aggregate_uncertainty_score > max_aggregate:
            return False, (
                f"aggregate_uncertainty={self.aggregate_uncertainty_score:.2f} "
                f"exceeds {max_aggregate:.2f} (source: {self.dominant_uncertainty_source})"
            )
        return True, "uncertainty_acceptable"


# -----------------------------
# Decision Delta Tracking
# -----------------------------

@dataclass(frozen=True)
class AuditoryDecisionDelta:
    """Change tracking since last decision."""
    trust_score_delta: float
    output_level_delta_db: float
    comfort_delta: float
    enhancement_quality_delta: float
    authorization_level_changed: bool
    listening_mode_changed: bool
    time_since_last_decision_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Domain Inputs
# -----------------------------

@dataclass(frozen=True)
class AuditorySignals:
    """Governance signals for auditory enhancement assessment."""
    proposed_action_trust_score: float
    desired_level: OutputAuthorizationLevel
    listening_mode: ListeningMode
    environment: Optional[EnvironmentMetrics] = None
    enhancement: Optional[EnhancementMetrics] = None
    comfort: Optional[ComfortMetrics] = None
    device_health: Optional[DeviceHealth] = None
    hearing_profile: Optional[HearingProfile] = None
    peer_enhancement_scores: Tuple[float, ...] = ()
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())


# -----------------------------
# Domain Configuration
# -----------------------------

@dataclass(frozen=True)
class AuditoryGovernancePolicy:
    """Domain policy for auditory enhancement governance."""
    max_allowed_level: OutputAuthorizationLevel = OutputAuthorizationLevel.COMFORT_OPTIMIZED
    max_output_db_spl: float = 100.0
    max_continuous_output_db: float = 85.0
    user_safety_profile: UserSafetyProfile = UserSafetyProfile.STANDARD
    min_speech_intelligibility: float = 0.65
    min_noise_reduction: float = 0.55
    min_enhancement_confidence: float = 0.70
    max_latency_ms: float = 20.0
    max_discomfort_score: float = 0.35
    max_fatigue_risk: float = 0.60
    min_mic_health_score: float = 0.75
    min_battery_level: float = 0.10
    allow_full_in_noisy: bool = False
    require_calibration: bool = True
    enable_predictive_warnings: bool = True
    enable_automatic_volume_limiting: bool = True
    max_event_log_size: int = 5000
    metadata: Dict[str, Any] = field(default_factory=dict)


def default_auditory_config() -> AileeConfig:
    """Safe defaults for auditory governance pipeline configuration."""
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    return AileeConfig(
        accept_threshold=0.85,
        borderline_low=0.65,
        borderline_high=0.85,
        w_stability=0.40,
        w_agreement=0.35,
        w_likelihood=0.25,
        history_window=80,
        forecast_window=12,
        grace_peer_delta=0.20,
        grace_min_peer_agreement_ratio=0.60,
        grace_forecast_epsilon=0.18,
        grace_max_abs_z=2.8,
        consensus_quorum=2,
        consensus_delta=0.18,
        consensus_pass_ratio=0.70,
        fallback_mode="last_good",
        fallback_clamp_min=0.0,
        fallback_clamp_max=1.0,
        hard_min=0.0,
        hard_max=1.0,
        enable_grace=True,
        enable_consensus=True,
        enable_audit_metadata=True,
    )


# -----------------------------
# Result Structure
# -----------------------------

@dataclass(frozen=True)
class AuditoryDecision:
    """Auditory governance decision result."""
    authorized_level: OutputAuthorizationLevel
    decision_outcome: DecisionOutcome
    validated_trust_score: float
    confidence_score: float
    output_db_cap: float
    enhancement_constraints: Optional[Dict[str, Any]] = None
    recommendation: str = ""
    reasons: List[str] = field(default_factory=list)
    ailee_result: Optional[DecisionResult] = None
    safety_margin_db: Optional[float] = None
    precautionary_flags: Optional[List[str]] = None
    warning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Auditory Events
# -----------------------------

@dataclass(frozen=True)
class AuditoryEvent:
    """Structured event for medical device compliance logging."""
    timestamp: float
    event_type: str
    listening_mode: ListeningMode
    desired_level: OutputAuthorizationLevel
    authorized_level: OutputAuthorizationLevel
    decision_outcome: DecisionOutcome
    proposed_action_trust_score: float
    output_db_cap: float
    validated_trust_score: float
    confidence_score: float
    reasons: List[str]
    environment_metrics: Optional[EnvironmentMetrics] = None
    enhancement_metrics: Optional[EnhancementMetrics] = None
    comfort_metrics: Optional[ComfortMetrics] = None
    device_health: Optional[DeviceHealth] = None
    aggregate_uncertainty: Optional[AuditoryUncertainty] = None
    ailee_decision: Optional[DecisionResult] = None
    warning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== POLICY EVALUATOR =====

class AuditoryPolicyEvaluator:
    """
    Formalized policy-derived gate checks for auditory governance.
    ADVISORY ROLE: Evaluates constraints and recommends ceilings.
    """
    
    def __init__(self, policy: AuditoryGovernancePolicy):
        self.policy = policy
    
    def check_device_health_gates(
        self,
        signals: AuditorySignals
    ) -> Tuple[bool, List[str], OutputAuthorizationLevel]:
        """Check device health constraints and recommend ceiling."""
        issues = []
        max_level = OutputAuthorizationLevel.FULL_ENHANCEMENT
        
        if not signals.device_health:
            return True, [], max_level
        
        device = signals.device_health
        
        has_issues, health_issues = device.get_health_issues(
            min_mic_health=self.policy.min_mic_health_score,
            min_battery=self.policy.min_battery_level
        )
        
        if has_issues:
            issues.extend(health_issues)
            
            if device.hardware_faults or device.feedback_detected:
                max_level = OutputAuthorizationLevel.DIAGNOSTIC_ONLY
            elif device.clipping_detected or (
                device.mic_health_score is not None and device.mic_health_score < 0.60
            ):
                max_level = OutputAuthorizationLevel.SAFETY_LIMITED
        
        passed = len(issues) == 0
        return passed, issues, max_level
    
    def check_comfort_gates(
        self,
        signals: AuditorySignals
    ) -> Tuple[bool, List[str]]:
        """Check user comfort constraints."""
        issues = []
        
        if not signals.comfort:
            return True, []
        
        comfort_ok, comfort_issues = signals.comfort.is_comfort_acceptable(
            max_discomfort=self.policy.max_discomfort_score,
            max_fatigue=self.policy.max_fatigue_risk
        )
        
        if not comfort_ok:
            issues.extend(comfort_issues)
        
        return len(issues) == 0, issues
    
    def check_quality_gates(
        self,
        signals: AuditorySignals
    ) -> Tuple[bool, List[str]]:
        """Check enhancement quality constraints."""
        issues = []
        
        if not signals.enhancement:
            return True, []
        
        quality_ok, quality_issues = signals.enhancement.is_quality_acceptable(
            min_intelligibility=self.policy.min_speech_intelligibility,
            min_noise_reduction=self.policy.min_noise_reduction,
            max_latency_ms=self.policy.max_latency_ms
        )
        
        if not quality_ok:
            issues.extend(quality_issues)
        
        return len(issues) == 0, issues
    
    def check_environmental_gates(
        self,
        signals: AuditorySignals
    ) -> Tuple[bool, List[str]]:
        """Check environmental acoustic constraints."""
        issues = []
        
        if not signals.environment:
            return True, []
        
        challenging, env_issues = signals.environment.is_challenging_environment()
        
        if challenging:
            issues.extend(env_issues)
        
        # FIX ERROR 5: Added missing return path
        return len(issues) == 0, issues
    
    def compute_precautionary_penalty(
        self,
        precautionary_flags: List[str]
    ) -> Tuple[float, str]:
        """Compute severity-weighted penalty from precautionary flags."""
        if not precautionary_flags:
            return 0.0, "no_flags"
        
        total_severity = sum(
            AUDITORY_FLAG_SEVERITY.get(flag, 0.03)
            for flag in precautionary_flags
        )
        
        capped_penalty = min(0.25, total_severity)
        
        severities = [(AUDITORY_FLAG_SEVERITY.get(f, 0.03), f) for f in precautionary_flags]
        most_severe_score, most_severe_flag = max(severities, key=lambda x: x[0])
        
        explanation = (
            f"{len(precautionary_flags)} flags, "
            f"most_severe={most_severe_flag} ({most_severe_score:.2f}), "
            f"total_penalty={capped_penalty:.2f}"
        )
        
        return capped_penalty, explanation


# ===== UNCERTAINTY CALCULATOR =====

class AuditoryUncertaintyCalculator:
    """
    Explicit uncertainty aggregation for auditory domain.
    COMPUTATION ROLE: Pure calculation, no policy decisions.
    """
    
    @staticmethod
    def compute_aggregate_uncertainty(
        signals: AuditorySignals
    ) -> AuditoryUncertainty:
        """Aggregate uncertainty from all auditory sources."""
        components = {}
        
        # 1. Enhancement uncertainty
        enhancement_unc = 0.0
        if signals.enhancement:
            if signals.enhancement.ai_confidence is not None:
                enhancement_unc = 1.0 - signals.enhancement.ai_confidence
            else:
                enhancement_unc = 0.3
            
            if signals.enhancement.artifacts_detected:
                enhancement_unc = min(1.0, enhancement_unc + 0.2)
        else:
            enhancement_unc = 0.5
        
        components["enhancement"] = enhancement_unc
        
        # 2. Environmental uncertainty
        environmental_unc = 0.0
        if signals.environment:
            if signals.environment.ambient_noise_db is not None and signals.environment.ambient_noise_db > 75.0:
                environmental_unc += 0.2
            
            if signals.environment.snr_db is not None and signals.environment.snr_db < 5.0:
                environmental_unc += 0.2
            
            if signals.environment.scene_confidence is not None:
                environmental_unc += (1.0 - signals.environment.scene_confidence) * 0.3
            else:
                environmental_unc += 0.2
        else:
            environmental_unc = 0.4
        
        components["environmental"] = min(1.0, environmental_unc)
        
        # 3. Device uncertainty
        device_unc = 0.0
        if signals.device_health:
            device = signals.device_health
            
            if device.mic_health_score is not None:
                device_unc += (1.0 - device.mic_health_score) * 0.5
            
            if device.hardware_faults:
                device_unc += 0.3
            
            if not device.calibration_valid:
                device_unc += 0.2
        else:
            device_unc = 0.3
        
        components["device"] = min(1.0, device_unc)
        
        # 4. Comfort uncertainty
        comfort_unc = 0.0
        if signals.comfort:
            if signals.comfort.discomfort_score is None:
                comfort_unc = 0.25
            else:
                comfort_unc = 0.1
        else:
            comfort_unc = 0.35
        
        components["comfort"] = comfort_unc
        
        # Aggregate using weighted average
        weights = {
            "enhancement": 0.40,
            "environmental": 0.30,
            "device": 0.20,
            "comfort": 0.10,
        }
        
        aggregate = sum(components[k] * weights[k] for k in components.keys())
        
        dominant_source = max(components.items(), key=lambda x: x[1])[0]
        
        return AuditoryUncertainty(
            aggregate_uncertainty_score=aggregate,
            enhancement_uncertainty=components["enhancement"],
            environmental_uncertainty=components["environmental"],
            device_uncertainty=components["device"],
            comfort_uncertainty=components["comfort"],
            dominant_uncertainty_source=dominant_source,
            uncertainty_sources=components,
        )


# ===== AUDITORY GOVERNOR =====

class AuditoryGovernor:
    """
    Production-grade governance controller for auditory enhancement systems.
    AUTHORITATIVE ROLE: Makes final authorization decisions.
    """
    
    def __init__(
        self,
        cfg: Optional[AileeConfig] = None,
        policy: Optional[AuditoryGovernancePolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable")
        
        self.policy = policy or AuditoryGovernancePolicy()
        self.cfg = cfg or default_auditory_config()
        
        self.cfg.hard_min = 0.0
        self.cfg.hard_max = 1.0
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        self.policy_evaluator = AuditoryPolicyEvaluator(self.policy)
        
        # State tracking
        self._last_decision: Optional[OutputAuthorizationLevel] = None
        self._last_trust_score: Optional[float] = None
        self._last_output_db: Optional[float] = None
        self._last_comfort_score: Optional[float] = None
        self._last_enhancement_quality: Optional[float] = None
        self._last_uncertainty: Optional[float] = None
        self._last_decision_time: Optional[float] = None
        self._last_listening_mode: Optional[ListeningMode] = None
        
        # Event logging
        self._event_log: List[AuditoryEvent] = []
        self._last_event: Optional[AuditoryEvent] = None
    
    def evaluate(self, signals: AuditorySignals) -> AuditoryDecision:
        """
        Evaluate auditory enhancement proposal and produce governance decision.
        AUTHORITATIVE DECISION: This method has sole decision authority.
        """
        ts = float(signals.timestamp)
        reasons: List[str] = []
        precautionary_flags: List[str] = []
        
        # 0) Compute aggregate uncertainty FIRST
        aggregate_unc = AuditoryUncertaintyCalculator.compute_aggregate_uncertainty(signals)
        
        unc_ok, unc_reason = aggregate_unc.is_uncertainty_acceptable(max_aggregate=0.35)
        if not unc_ok:
            reasons.append(unc_reason)
            precautionary_flags.append("aggregate_uncertainty_excessive")
        
        # 1) Device health gate
        device_ok, device_issues, max_device_level = \
            self.policy_evaluator.check_device_health_gates(signals)
        
        if not device_ok:
            reasons.extend([f"Device: {issue}" for issue in device_issues])
            if max_device_level == OutputAuthorizationLevel.DIAGNOSTIC_ONLY:
                precautionary_flags.append("hardware_fault_critical")
            elif max_device_level == OutputAuthorizationLevel.SAFETY_LIMITED:
                precautionary_flags.append("hardware_degraded")
        
        # 2) Comfort gate
        comfort_ok, comfort_issues = self.policy_evaluator.check_comfort_gates(signals)
        if not comfort_ok:
            reasons.extend([f"Comfort: {issue}" for issue in comfort_issues])
            precautionary_flags.append("excessive_discomfort")
        
        # 3) Quality gate
        quality_ok, quality_issues = self.policy_evaluator.check_quality_gates(signals)
        if not quality_ok:
            reasons.extend([f"Quality: {issue}" for issue in quality_issues])
            
            if any("speech_intelligibility" in issue for issue in quality_issues):
                precautionary_flags.append("speech_unintelligible")
            if any("noise_reduction" in issue for issue in quality_issues):
                precautionary_flags.append("noise_reduction_poor")
            if any("latency" in issue for issue in quality_issues):
                precautionary_flags.append("latency_excessive")
        
        # 4) Environmental assessment
        env_ok, env_issues = self.policy_evaluator.check_environmental_gates(signals)
        if not env_ok:
            reasons.extend([f"Environment: {issue}" for issue in env_issues])
        
        # 5) Mode-specific restrictions
        if signals.listening_mode == ListeningMode.NOISY and not self.policy.allow_full_in_noisy:
            if signals.desired_level == OutputAuthorizationLevel.FULL_ENHANCEMENT:
                reasons.append("Noisy mode limits full enhancement")
                precautionary_flags.append("noisy_environment_restriction")
        
        # 6) Policy maximum level
        policy_max_level = self.policy.max_allowed_level
        if signals.desired_level > policy_max_level:
            reasons.append(f"Policy cap to {int(policy_max_level)}")
        
        # 7) Compute severity-weighted precautionary penalty
        precautionary_penalty, penalty_explanation = \
            self.policy_evaluator.compute_precautionary_penalty(precautionary_flags)
        
        # 8) Extract peer values
        peer_values = list(signals.peer_enhancement_scores)
        
        # 9) Build context for AILEE pipeline
        ctx = self._build_context(signals, reasons, precautionary_flags, aggregate_unc)
        
        # 10) Apply penalties
        base_score = signals.proposed_action_trust_score
        score_after_precaution = base_score * (1.0 - precautionary_penalty)
        uncertainty_penalty = aggregate_unc.aggregate_uncertainty_score * 0.20
        final_adjusted_score = score_after_precaution * (1.0 - uncertainty_penalty)
        
        ctx["precautionary_penalty"] = precautionary_penalty
        ctx["penalty_explanation"] = penalty_explanation
        ctx["uncertainty_penalty"] = uncertainty_penalty
        ctx["final_adjustment"] = base_score - final_adjusted_score
        
        # 11) AILEE pipeline
        ailee_result = self.pipeline.process(
            raw_value=float(final_adjusted_score),
            raw_confidence=float(1.0 - aggregate_unc.aggregate_uncertainty_score),
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        # 12) Determine maximum allowed level from uncertainty
        max_uncertainty_level = self._compute_level_ceiling_from_uncertainty(aggregate_unc)
        
        # 13) Make decision with all constraints
        decision = self._make_auditory_decision(
            signals,
            ailee_result,
            reasons,
            precautionary_flags,
            aggregate_unc,
            ts,
            max_device_level,
            max_uncertainty_level,
            policy_max_level
        )
        
        # 14) Compute decision delta
        decision_delta = self._compute_decision_delta(signals, decision, aggregate_unc, ts)
        if decision_delta:
            decision.metadata["decision_delta"] = decision_delta
        
        # 15) Generate predictive warning
        warning = None
        if self.policy.enable_predictive_warnings:
            warning = self._predict_warning(signals, decision, aggregate_unc)
        
        # 16) Log and track
        self._log_and_track(ts, signals, decision, reasons, aggregate_unc, warning)
        
        return decision
    
    def _compute_level_ceiling_from_uncertainty(
        self,
        aggregate_unc: AuditoryUncertainty
    ) -> OutputAuthorizationLevel:
        """Compute maximum allowed level based on uncertainty."""
        unc_score = aggregate_unc.aggregate_uncertainty_score
        
        if unc_score > 0.50:
            return OutputAuthorizationLevel.DIAGNOSTIC_ONLY
        if unc_score > 0.35:
            return OutputAuthorizationLevel.SAFETY_LIMITED
        if unc_score > 0.25:
            return OutputAuthorizationLevel.COMFORT_OPTIMIZED
        
        return OutputAuthorizationLevel.FULL_ENHANCEMENT
    
    def _build_context(
        self,
        signals: AuditorySignals,
        reasons: List[str],
        precautionary_flags: List[str],
        aggregate_unc: AuditoryUncertainty
    ) -> Dict[str, Any]:
        """Build context dictionary for AILEE pipeline"""
        ctx = {
            "domain": "auditory",
            "listening_mode": signals.listening_mode.value,
            "desired_level": int(signals.desired_level),
            "aggregate_uncertainty": aggregate_unc.aggregate_uncertainty_score,
            "dominant_uncertainty_source": aggregate_unc.dominant_uncertainty_source,
            "reasons": reasons[:],
            "precautionary_flags": precautionary_flags[:],
        }
        
        if signals.environment:
            if signals.environment.ambient_noise_db is not None:
                ctx["ambient_noise_db"] = signals.environment.ambient_noise_db
            if signals.environment.snr_db is not None:
                ctx["snr_db"] = signals.environment.snr_db
        
        if signals.enhancement:
            if signals.enhancement.speech_intelligibility_score is not None:
                ctx["speech_intelligibility"] = signals.enhancement.speech_intelligibility_score
            if signals.enhancement.enhancement_latency_ms is not None:
                ctx["latency_ms"] = signals.enhancement.enhancement_latency_ms
        
        if signals.comfort:
            if signals.comfort.discomfort_score is not None:
                ctx["discomfort"] = signals.comfort.discomfort_score
        
        ctx.update(signals.context)
        return ctx
    
    def _make_auditory_decision(
        self,
        signals: AuditorySignals,
        ailee_result: DecisionResult,
        reasons: List[str],
        precautionary_flags: List[str],
        aggregate_unc: AuditoryUncertainty,
        ts: float,
        max_device_level: OutputAuthorizationLevel,
        max_uncertainty_level: OutputAuthorizationLevel,
        policy_max_level: OutputAuthorizationLevel
    ) -> AuditoryDecision:
        """Convert AILEE result to auditory-specific decision"""
        
        validated_score = ailee_result.validated_value
        
        # Determine base level from score
        if validated_score >= 0.85:
            score_level = OutputAuthorizationLevel.FULL_ENHANCEMENT
        elif validated_score >= 0.70:
            score_level = OutputAuthorizationLevel.COMFORT_OPTIMIZED
        elif validated_score >= 0.50:
            score_level = OutputAuthorizationLevel.SAFETY_LIMITED
        elif validated_score >= 0.25:
            score_level = OutputAuthorizationLevel.DIAGNOSTIC_ONLY
        else:
            score_level = OutputAuthorizationLevel.NO_OUTPUT
        
        # Apply multiple ceilings
        authorized_level = min(
            score_level,
            max_device_level,
            max_uncertainty_level,
            policy_max_level,
            signals.desired_level
        )
        
        # Determine outcome
        if authorized_level == OutputAuthorizationLevel.NO_OUTPUT:
            outcome = DecisionOutcome.SUPPRESSED
            recommendation = "suppress_enhancement_quality_insufficient"
        elif authorized_level == OutputAuthorizationLevel.DIAGNOSTIC_ONLY:
            outcome = DecisionOutcome.DIAGNOSTIC_FALLBACK
            recommendation = "diagnostic_mode_only"
        elif authorized_level < signals.desired_level:
            outcome = DecisionOutcome.LIMITED
            recommendation = f"limited_to_{authorized_level.name.lower()}"
        else:
            outcome = DecisionOutcome.AUTHORIZED
            recommendation = f"authorized_{authorized_level.name.lower()}"
        
        # Add ceiling explanations
        metadata_flags = {}
        if max_device_level < score_level:
            reasons.append(f"Device health limited to {max_device_level.name}")
            metadata_flags["device_ceiling_applied"] = True
        
        if max_uncertainty_level < score_level:
            reasons.append(f"Uncertainty limited to {max_uncertainty_level.name}")
            metadata_flags["uncertainty_ceiling_applied"] = True
        
        # Compute output dB cap
        output_db_cap = self._compute_output_db_cap(signals, authorized_level)
        
        # Compute safety margin
        safety_margin = None
        if signals.hearing_profile:
            max_safe = signals.hearing_profile.max_safe_output_db
            safety_margin_value = signals.hearing_profile.get_dynamic_safety_margin()
            safety_margin = max_safe - safety_margin_value - output_db_cap
        
        # Generate enhancement constraints
        enhancement_constraints = None
        if authorized_level >= OutputAuthorizationLevel.SAFETY_LIMITED:
            enhancement_constraints = self._generate_enhancement_constraints(
                signals, precautionary_flags, authorized_level
            )
        
        return AuditoryDecision(
            authorized_level=authorized_level,
            decision_outcome=outcome,
            validated_trust_score=validated_score,
            confidence_score=ailee_result.confidence_score,
            output_db_cap=output_db_cap,
            enhancement_constraints=enhancement_constraints,
            recommendation=recommendation,
            reasons=reasons[:],
            ailee_result=ailee_result,
            safety_margin_db=safety_margin,
            precautionary_flags=precautionary_flags[:],
            metadata={
                "timestamp": ts,
                "listening_mode": signals.listening_mode.value,
                "aggregate_uncertainty": aggregate_unc.aggregate_uncertainty_score,
                "dominant_uncertainty_source": aggregate_unc.dominant_uncertainty_source,
                "max_device_level": int(max_device_level),
                "max_uncertainty_level": int(max_uncertainty_level),
                **metadata_flags,
            }
        )
    
    def _compute_output_db_cap(
        self,
        signals: AuditorySignals,
        authorized_level: OutputAuthorizationLevel
    ) -> float:
        """Compute safe output dB SPL cap."""
        default_output = 70.0
        if signals.hearing_profile:
            default_output = signals.hearing_profile.preferred_output_db
        
        # Ambient noise compensation
        ambient_boost = 0.0
        if signals.environment and signals.environment.ambient_noise_db is not None:
            ambient_noise = signals.environment.ambient_noise_db
            if ambient_noise >= 80.0:
                ambient_boost = 8.0
            elif ambient_noise >= 70.0:
                ambient_boost = 4.0
            elif ambient_noise >= 60.0:
                ambient_boost = 2.0
        
        # Level-based reduction
        level_reduction = 0.0
        if authorized_level == OutputAuthorizationLevel.DIAGNOSTIC_ONLY:
            level_reduction = 15.0
        elif authorized_level == OutputAuthorizationLevel.SAFETY_LIMITED:
            level_reduction = 8.0
        elif authorized_level == OutputAuthorizationLevel.COMFORT_OPTIMIZED:
            level_reduction = 3.0
        
        output_target = default_output + ambient_boost - level_reduction
        
        # Safety ceiling
        max_safe = self.policy.max_output_db_spl
        
        if signals.hearing_profile:
            user_max = signals.hearing_profile.max_safe_output_db
            safety_margin = signals.hearing_profile.get_dynamic_safety_margin()
            max_safe = min(max_safe, user_max - safety_margin)
        
        # Apply safety profile adjustments
        if self.policy.user_safety_profile == UserSafetyProfile.PEDIATRIC:
            max_safe = min(max_safe, 85.0)
        elif self.policy.user_safety_profile == UserSafetyProfile.TINNITUS_RISK:
            max_safe = min(max_safe, 90.0)
        
        return max(0.0, min(max_safe, output_target))
    
    def _generate_enhancement_constraints(
        self,
        signals: AuditorySignals,
        precautionary_flags: List[str],
        authorized_level: OutputAuthorizationLevel
    ) -> Dict[str, Any]:
        """Generate constraints for enhancement execution"""
        constraints = {
            "monitoring_required": True,
            "max_output_db": self._compute_output_db_cap(signals, authorized_level),
        }
        
        if signals.enhancement and signals.enhancement.enhancement_latency_ms is not None:
            constraints["max_latency_ms"] = self.policy.max_latency_ms
        
        if self.policy.enable_automatic_volume_limiting:
            constraints["automatic_volume_limiting"] = True
            constraints["volume_limiter_attack_ms"] = 5.0
            constraints["volume_limiter_release_ms"] = 100.0
        
        if signals.listening_mode == ListeningMode.MUSIC:
            constraints["preserve_dynamic_range"] = True
            constraints["limit_compression_ratio"] = 3.0
        elif signals.listening_mode == ListeningMode.SPEECH_FOCUS:
            constraints["prioritize_speech_bands"] = True
            constraints["speech_frequency_range_hz"] = (500, 4000)
        
        if len(precautionary_flags) > 0:
            constraints["precautionary_measures"] = precautionary_flags[:]
            constraints["enhanced_monitoring"] = True
        
        if signals.device_health and signals.device_health.feedback_detected:
            constraints["feedback_cancellation_aggressive"] = True
            constraints["gain_reduction_db"] = 6.0
        
        return constraints
    
    def _predict_warning(
        self,
        signals: AuditorySignals,
        decision: AuditoryDecision,
        aggregate_unc: AuditoryUncertainty
    ) -> Optional[str]:
        """Generate predictive warning if risks detected."""
        
        if signals.device_health and signals.device_health.feedback_detected:
            return "Feedback detected: consider repositioning device"
        
        if signals.comfort and signals.comfort.fatigue_risk_score is not None:
            if signals.comfort.fatigue_risk_score > 0.60:
                return "Listening fatigue rising: suggest break within 30 minutes"
        
        if decision.validated_trust_score < 0.50:
            return "Enhancement quality declining: may need recalibration"
        
        if signals.device_health and signals.device_health.battery_level is not None:
            if signals.device_health.battery_level < 0.15:
                return "Battery low: enhanced features may be reduced soon"
        
        if aggregate_unc.aggregate_uncertainty_score > 0.40:
            return f"High uncertainty in {aggregate_unc.dominant_uncertainty_source}: verify measurements"
        
        return None
    
    def _compute_decision_delta(
        self,
        signals: AuditorySignals,
        decision: AuditoryDecision,
        aggregate_unc: AuditoryUncertainty,
        ts: float
    ) -> Optional[AuditoryDecisionDelta]:
        """Compute change since last decision"""
        
        if None in (
            self._last_trust_score,
            self._last_output_db,
            self._last_enhancement_quality,
            self._last_uncertainty,
        ):
            return None
        
        trust_delta = decision.validated_trust_score - self._last_trust_score
        output_delta = decision.output_db_cap - self._last_output_db
        
        current_quality = signals.proposed_action_trust_score
        quality_delta = current_quality - self._last_enhancement_quality
        
        comfort_delta = 0.0
        if signals.comfort and signals.comfort.discomfort_score is not None:
            if self._last_comfort_score is not None:
                comfort_delta = signals.comfort.discomfort_score - self._last_comfort_score
        
        authorization_changed = (decision.authorized_level != self._last_decision)
        mode_changed = (signals.listening_mode != self._last_listening_mode)
        time_delta = ts - (self._last_decision_time or ts)
        
        return AuditoryDecisionDelta(
            trust_score_delta=trust_delta,
            output_level_delta_db=output_delta,
            comfort_delta=comfort_delta,
            enhancement_quality_delta=quality_delta,
            authorization_level_changed=authorization_changed,
            listening_mode_changed=mode_changed,
            time_since_last_decision_seconds=time_delta,
        )
    
    def _log_and_track(
        self,
        ts: float,
        signals: AuditorySignals,
        decision: AuditoryDecision,
        reasons: List[str],
        aggregate_unc: AuditoryUncertainty,
        warning: Optional[str]
    ):
        """Combined logging and state tracking"""
        
        event_type_map = {
            DecisionOutcome.AUTHORIZED: "enhancement_authorized",
            DecisionOutcome.LIMITED: "enhancement_limited",
            DecisionOutcome.DIAGNOSTIC_FALLBACK: "diagnostic_fallback",
            DecisionOutcome.SUPPRESSED: "enhancement_suppressed",
            DecisionOutcome.HARDWARE_FAULT: "hardware_fault",
        }
        
        event = AuditoryEvent(
            timestamp=ts,
            event_type=event_type_map.get(decision.decision_outcome, "unknown"),
            listening_mode=signals.listening_mode,
            desired_level=signals.desired_level,
            authorized_level=decision.authorized_level,
            decision_outcome=decision.decision_outcome,
            proposed_action_trust_score=signals.proposed_action_trust_score,
            output_db_cap=decision.output_db_cap,
            validated_trust_score=decision.validated_trust_score,
            confidence_score=decision.confidence_score,
            reasons=reasons[:],
            environment_metrics=signals.environment,
            enhancement_metrics=signals.enhancement,
            comfort_metrics=signals.comfort,
            device_health=signals.device_health,
            aggregate_uncertainty=aggregate_unc,
            ailee_decision=decision.ailee_result,
            warning=warning,
            metadata=decision.metadata,
        )
        
        self._event_log.append(event)
        
        if len(self._event_log) > self.policy.max_event_log_size:
            self._event_log = self._event_log[-self.policy.max_event_log_size:]
        
        self._last_event = event
        
        # Update state tracking
        self._last_decision = decision.authorized_level
        self._last_trust_score = decision.validated_trust_score
        self._last_output_db = decision.output_db_cap
        self._last_enhancement_quality = signals.proposed_action_trust_score
        self._last_uncertainty = aggregate_unc.aggregate_uncertainty_score
        self._last_decision_time = ts
        self._last_listening_mode = signals.listening_mode
        
        if signals.comfort and signals.comfort.discomfort_score is not None:
            self._last_comfort_score = signals.comfort.discomfort_score
    
    def explain_decision(self, decision: AuditoryDecision) -> str:
        """Generate plain-language explanation of decision"""
        lines = []
        
        lines.append("=" * 70)
        lines.append("AUDITORY GOVERNANCE DECISION EXPLANATION")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"DECISION: {decision.decision_outcome.value}")
        lines.append(f"Authorized Level: {decision.authorized_level.name}")
        lines.append(f"Output Cap: {decision.output_db_cap:.1f} dB SPL")
        if decision.safety_margin_db is not None:
            lines.append(f"Safety Margin: {decision.safety_margin_db:.1f} dB")
        lines.append("")
        
        lines.append(f"Validated Trust Score: {decision.validated_trust_score:.2f} / 1.00")
        lines.append(f"Decision Confidence: {decision.confidence_score:.2f} / 1.00")
        lines.append("")
        
        lines.append("WHAT MATTERED MOST:")
        lines.append("-" * 70)
        
        if decision.ailee_result:
            lines.append(f"• AILEE Pipeline Status: {decision.ailee_result.status.value}")
            if decision.ailee_result.grace_applied:
                lines.append("  → Grace conditions applied")
            if decision.ailee_result.consensus_status:
                lines.append(f"  → Consensus: {decision.ailee_result.consensus_status}")
        
        if decision.metadata.get("aggregate_uncertainty") is not None:
            unc = decision.metadata["aggregate_uncertainty"]
            source = decision.metadata.get("dominant_uncertainty_source", "unknown")
            lines.append(f"• Aggregate Uncertainty: {unc:.2f} (primary: {source})")
        
        if decision.metadata.get("uncertainty_ceiling_applied"):
            lines.append("• ⚠️  Uncertainty limited output level")
        
        if decision.metadata.get("device_ceiling_applied"):
            lines.append("• ⚠️  Device health limited output level")
        
        lines.append("")
        
        if decision.reasons:
            lines.append("LIMITING FACTORS:")
            lines.append("-" * 70)
            for reason in decision.reasons:
                lines.append(f"• {reason}")
            lines.append("")
        
        if decision.precautionary_flags:
            lines.append("PRECAUTIONARY FLAGS:")
            lines.append("-" * 70)
            for flag in decision.precautionary_flags:
                severity = AUDITORY_FLAG_SEVERITY.get(flag, 0.03)
                lines.append(f"• {flag} (severity: {severity:.2f})")
            lines.append("")
        
        if decision.enhancement_constraints:
            lines.append("ENHANCEMENT CONSTRAINTS:")
            lines.append("-" * 70)
            for key, value in decision.enhancement_constraints.items():
                lines.append(f"• {key}: {value}")
            lines.append("")
        
        if decision.warning:
            lines.append("⚠️  WARNING:")
            lines.append("-" * 70)
            lines.append(f"→ {decision.warning}")
            lines.append("")
        
        lines.append("RECOMMENDED ACTION:")
        lines.append("-" * 70)
        lines.append(f"→ {decision.recommendation.replace('_', ' ').title()}")
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_last_event(self) -> Optional[AuditoryEvent]:
        """Get most recent auditory governance event"""
        return self._last_event
    
    def export_events(self, since_ts: Optional[float] = None) -> List[AuditoryEvent]:
        """Export events for medical device compliance reporting"""
        if since_ts is None:
            return self._event_log[:]
        return [e for e in self._event_log if e.timestamp >= since_ts]
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for clinical monitoring"""
        if not self._event_log:
            return {"status": "no_history"}
        
        total_events = len(self._event_log)
        
        level_counts = {}
        for event in self._event_log:
            level_name = event.authorized_level.name
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
        
        warning_count = sum(1 for e in self._event_log if e.warning is not None)
        
        output_levels = [e.output_db_cap for e in self._event_log]
        avg_output = statistics.mean(output_levels) if output_levels else 0.0
        
        trust_scores = [e.validated_trust_score for e in self._event_log]
        avg_trust = statistics.mean(trust_scores) if trust_scores else 0.0
        
        return {
            "total_events": total_events,
            "authorization_level_distribution": level_counts,
            "warning_count": warning_count,
            "average_output_db": avg_output,
            "average_trust_score": avg_trust,
            "last_event_timestamp": self._event_log[-1].timestamp if self._event_log else None,
        }
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get safety summary for clinical review"""
        recent_events = self._event_log[-100:] if len(self._event_log) > 100 else self._event_log
        
        if not recent_events:
            return {"status": "no_recent_history"}
        
        high_output_count = sum(1 for e in recent_events if e.output_db_cap > 90.0)
        discomfort_events = sum(
            1 for e in recent_events 
            if e.comfort_metrics and e.comfort_metrics.discomfort_score is not None and e.comfort_metrics.discomfort_score > 0.35
        )
        hardware_issues = sum(
            1 for e in recent_events
            if e.device_health and (e.device_health.feedback_detected or e.device_health.hardware_faults)
        )
        
        return {
            "recent_event_count": len(recent_events),
            "high_output_events": high_output_count,
            "discomfort_events": discomfort_events,
            "hardware_issue_events": hardware_issues,
            "safety_concerns": high_output_count > 10 or discomfort_events > 20 or hardware_issues > 5,
        }


# -----------------------------
# Convenience Functions
# -----------------------------

def create_auditory_governor(
    user_safety_profile: UserSafetyProfile = UserSafetyProfile.STANDARD,
    max_output_db_spl: float = 100.0,
    max_allowed_level: OutputAuthorizationLevel = OutputAuthorizationLevel.COMFORT_OPTIMIZED,
    **policy_overrides
) -> AuditoryGovernor:
    """
    Convenience factory for creating auditory governor with common configurations.
    
    Args:
        user_safety_profile: User safety category
        max_output_db_spl: Absolute maximum output in dB SPL
        max_allowed_level: Maximum authorization level allowed by policy
        **policy_overrides: Additional policy parameters
    
    Returns:
        Configured AuditoryGovernor instance
    
    Example:
        governor = create_auditory_governor(
            user_safety_profile=UserSafetyProfile.PEDIATRIC,
            max_output_db_spl=85.0,
            max_allowed_level=OutputAuthorizationLevel.COMFORT_OPTIMIZED,
            enable_automatic_volume_limiting=True,
        )
    """
    policy_kwargs = {
        "user_safety_profile": user_safety_profile,
        "max_output_db_spl": max_output_db_spl,
        "max_allowed_level": max_allowed_level,
    }
    
    # Profile-specific defaults
    if user_safety_profile == UserSafetyProfile.PEDIATRIC:
        policy_kwargs.update({
            "max_output_db_spl": min(max_output_db_spl, 85.0),
            "max_continuous_output_db": 75.0,
            "max_allowed_level": min(max_allowed_level, OutputAuthorizationLevel.COMFORT_OPTIMIZED),
            "enable_automatic_volume_limiting": True,
        })
    elif user_safety_profile == UserSafetyProfile.TINNITUS_RISK:
        policy_kwargs.update({
            "max_output_db_spl": min(max_output_db_spl, 90.0),
            "max_discomfort_score": 0.25,
            "enable_predictive_warnings": True,
        })
    elif user_safety_profile == UserSafetyProfile.PROFESSIONAL:
        policy_kwargs.update({
            "min_speech_intelligibility": 0.75,
            "min_noise_reduction": 0.65,
            "max_latency_ms": 15.0,
        })
    
    policy_kwargs.update(policy_overrides)
    policy = AuditoryGovernancePolicy(**policy_kwargs)
    
    cfg = default_auditory_config()
    
    return AuditoryGovernor(cfg=cfg, policy=policy)


def validate_auditory_signals(signals: AuditorySignals) -> Tuple[bool, List[str]]:
    """
    Pre-flight validation of auditory signals structure.
    
    Args:
        signals: AuditorySignals to validate
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues: List[str] = []
    
    # Check score range
    if not (0.0 <= signals.proposed_action_trust_score <= 1.0):
        issues.append(f"proposed_action_trust_score={signals.proposed_action_trust_score} outside [0.0, 1.0]")
    
    # Check peer enhancement scores
    for i, score in enumerate(signals.peer_enhancement_scores):
        if not (0.0 <= score <= 1.0):
            issues.append(f"peer_enhancement_scores[{i}]={score} outside [0.0, 1.0]")
    
    # Validate enhancement metrics if present
    if signals.enhancement:
        enhancement = signals.enhancement
        if enhancement.speech_intelligibility_score is not None:
            if not (0.0 <= enhancement.speech_intelligibility_score <= 1.0):
                issues.append(f"speech_intelligibility_score outside [0.0, 1.0]")
        
        if enhancement.noise_reduction_score is not None:
            if not (0.0 <= enhancement.noise_reduction_score <= 1.0):
                issues.append(f"noise_reduction_score outside [0.0, 1.0]")
        
        if enhancement.enhancement_latency_ms is not None:
            if enhancement.enhancement_latency_ms < 0:
                issues.append(f"enhancement_latency_ms cannot be negative")
    
    # Validate comfort metrics if present
    if signals.comfort:
        comfort = signals.comfort
        if comfort.discomfort_score is not None:
            if not (0.0 <= comfort.discomfort_score <= 1.0):
                issues.append(f"discomfort_score outside [0.0, 1.0]")
        
        if comfort.fatigue_risk_score is not None:
            if not (0.0 <= comfort.fatigue_risk_score <= 1.0):
                issues.append(f"fatigue_risk_score outside [0.0, 1.0]")
    
    # Validate device health if present
    if signals.device_health:
        device = signals.device_health
        if device.mic_health_score is not None:
            if not (0.0 <= device.mic_health_score <= 1.0):
                issues.append(f"mic_health_score outside [0.0, 1.0]")
        
        if device.battery_level is not None:
            if not (0.0 <= device.battery_level <= 1.0):
                issues.append(f"battery_level outside [0.0, 1.0]")
    
    # Validate hearing profile if present
    if signals.hearing_profile:
        profile = signals.hearing_profile
        if profile.max_safe_output_db < profile.preferred_output_db:
            issues.append("max_safe_output_db must be >= preferred_output_db")
        
        if profile.max_safe_output_db > 120.0:
            issues.append("max_safe_output_db exceeds safe limits (>120 dB)")
    
    return len(issues) == 0, issues


# -----------------------------
# Module Exports
# -----------------------------

__all__ = [
    "OutputAuthorizationLevel",
    "ListeningMode",
    "UserSafetyProfile",
    "DecisionOutcome",
    "RegulatoryGateResult",
    "HearingProfile",
    "EnvironmentMetrics",
    "EnhancementMetrics",
    "ComfortMetrics",
    "DeviceHealth",
    "AuditoryUncertainty",
    "AuditoryDecisionDelta",
    "AuditorySignals",
    "AuditoryGovernancePolicy",
    "AuditoryDecision",
    "AuditoryEvent",
    "AuditoryPolicyEvaluator",
    "AuditoryUncertaintyCalculator",
    "AuditoryGovernor",
    "default_auditory_config",
    "create_auditory_governor",
    "validate_auditory_signals",
    "AUDITORY_FLAG_SEVERITY",
]
