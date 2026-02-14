"""
AILEE Trust Layer — Power Grid Governance Domain
Version: 2.0.0 - Production Grade

Governance-only authorization layer for power grid and energy systems.

This module determines the maximum allowed operational authority level
for AI-assisted grid control based on safety, stability, consensus,
operator readiness, and system health.

Primary governed signal:
    Grid Authority Level (scalar, 0..3)

Level semantics:
    0 = MANUAL_ONLY         Human operators only
    1 = ASSISTED_OPERATION  Advisory AI, human confirms all actions
    2 = CONSTRAINED_AUTONOMY AI acts within boundaries, human supervises
    3 = FULL_AUTONOMY       AI has full authority within operational envelope

Key properties:
    • Deterministic decisions (no randomness)
    • Auditable, reason-rich outputs with black box logging
    • Explicit downgrade/fallback behavior with degradation planning
    • Multi-oracle consensus via peer recommended levels
    • Stability-oriented (prevents mode thrash using hysteresis + Grace checks)
    • Operational domain aware (grid stress, weather, islanding)
    • Safety monitor integration (frequency, voltage, reserves)
    • Operator state monitoring (readiness, fatigue, handoff capability)
    • Multi-timescale confidence tracking
    • Predictive downgrade warnings
    • Scenario-based policy adaptation

INTEGRATION EXAMPLE:

    # Setup (once)
    policy = GridGovernancePolicy(
        max_allowed_level=GridAuthorityLevel.CONSTRAINED_AUTONOMY,
        min_operator_readiness=0.65,
    )
    governor = GridGovernor(policy=policy)
    
    # Per-cycle evaluation (1-10Hz typical for grid systems)
    while running:
        signals = GridSignals(
            proposed_level=grid_ai.get_desired_level(),
            model_confidence=grid_ai.get_confidence(),
            peer_recommended_levels=tuple(peer.get_level() for peer in peers),
            system_health=GridSystemHealth(...),
            operator_state=GridOperatorState(...),
            operational_domain=GridOperationalDomain(...),
            safety_monitors=GridSafetyMonitors(...),
        )
        
        authorized_level, decision = governor.evaluate(signals)
        
        # Use authorized_level as authorization CEILING
        grid_ai.set_authorization_ceiling(authorized_level)
        
        # ALERT if fallback was used
        if decision.used_fallback:
            scada.show_alarm("AI authority degraded", decision.reasons)
        
        # LOG for compliance (NERC CIP, etc.)
        event_log.record_governance_event(governor.get_last_event())

⚠️  This module DOES NOT execute control actions.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core AILEE Trust Pipeline Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    from ...ailee_trust_pipeline_v1 import (
        AileeConfig,
        AileeTrustPipeline,
        DecisionResult,
        SafetyStatus,
    )
except Exception:  # pragma: no cover
    AileeTrustPipeline = None  # type: ignore
    AileeConfig = None  # type: ignore
    DecisionResult = None  # type: ignore
    SafetyStatus = None  # type: ignore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Authority Levels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GridAuthorityLevel(IntEnum):
    """
    Discrete grid authority authorization levels.
    
    These levels represent increasing degrees of AI autonomy in grid operations,
    with each level having specific safety requirements and operator oversight.
    """
    MANUAL_ONLY = 0
    ASSISTED_OPERATION = 1
    CONSTRAINED_AUTONOMY = 2
    FULL_AUTONOMY = 3


def _clamp_int(x: int, lo: int, hi: int) -> int:
    """Clamp integer to inclusive range [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def _level_from_float(x: float) -> GridAuthorityLevel:
    """
    Quantize continuous value to nearest discrete authority level.
    
    Args:
        x: Continuous value to quantize
    
    Returns:
        Authority level clamped to valid range [0..3]
    """
    ix = int(round(float(x)))
    ix = _clamp_int(
        ix,
        int(GridAuthorityLevel.MANUAL_ONLY),
        int(GridAuthorityLevel.FULL_AUTONOMY)
    )
    return GridAuthorityLevel(ix)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Safety Monitor Signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GridSafetyMonitors:
    """
    External safety monitor inputs for grid stability.
    
    These should come from independent monitoring systems (PMUs, SCADA, etc.)
    and represent the ground truth about current grid conditions.
    """
    # Grid stability metrics
    frequency_deviation_hz: Optional[float] = None
    voltage_stability_index: Optional[float] = None
    angle_stability_margin: Optional[float] = None
    reserve_margin_mw: Optional[float] = None
    
    # Protection system status
    protection_systems_healthy: bool = True
    relay_failures: int = 0
    
    # Disturbance detection
    oscillation_detected: bool = False
    islanding_risk_score: Optional[float] = None
    
    def is_safe_for_level(self, level: GridAuthorityLevel) -> Tuple[bool, List[str]]:
        """
        Determine if current safety conditions permit the requested authority level.
        
        Args:
            level: Requested authority level
        
        Returns:
            (is_safe, list of issues) - Empty list indicates safe conditions
        """
        issues: List[str] = []
        
        if not self.protection_systems_healthy:
            issues.append("protection_systems_unhealthy")
        
        if level >= GridAuthorityLevel.ASSISTED_OPERATION:
            if self.relay_failures > 0:
                issues.append(f"relay_failures={self.relay_failures}")
        
        if level >= GridAuthorityLevel.CONSTRAINED_AUTONOMY:
            if self.frequency_deviation_hz is not None:
                if abs(self.frequency_deviation_hz) > 0.2:
                    issues.append(
                        f"frequency_deviation={self.frequency_deviation_hz:.3f}Hz exceeds ±0.2Hz"
                    )
            
            if self.voltage_stability_index is not None:
                if self.voltage_stability_index < 0.70:
                    issues.append(
                        f"voltage_stability={self.voltage_stability_index:.2f} below 0.70"
                    )
            
            if self.reserve_margin_mw is not None:
                if self.reserve_margin_mw < 100.0:
                    issues.append(
                        f"reserve_margin={self.reserve_margin_mw:.0f}MW below 100MW"
                    )
            
            if self.oscillation_detected:
                issues.append("power_oscillation_detected")
        
        if level >= GridAuthorityLevel.FULL_AUTONOMY:
            if self.frequency_deviation_hz is not None:
                if abs(self.frequency_deviation_hz) > 0.05:
                    issues.append(
                        f"frequency_deviation={self.frequency_deviation_hz:.3f}Hz "
                        f"exceeds ±0.05Hz for full autonomy"
                    )
            
            if self.voltage_stability_index is not None:
                if self.voltage_stability_index < 0.85:
                    issues.append(
                        f"voltage_stability={self.voltage_stability_index:.2f} "
                        f"below 0.85 for full autonomy"
                    )
            
            if self.islanding_risk_score is not None:
                if self.islanding_risk_score > 0.20:
                    issues.append(
                        f"islanding_risk={self.islanding_risk_score:.2f} "
                        f"exceeds 0.20 for full autonomy"
                    )
            
            if self.angle_stability_margin is not None:
                if self.angle_stability_margin < 15.0:
                    issues.append(
                        f"angle_stability_margin={self.angle_stability_margin:.1f}° "
                        f"below 15° for full autonomy"
                    )
        
        return len(issues) == 0, issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Operational Domain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GridOperationalDomain:
    """
    Operational constraints defining when/where AI authority is permitted.
    """
    islanded: bool = False
    interconnection_healthy: bool = True
    extreme_weather: bool = False
    weather_severity: str = "normal"
    grid_stress_level: str = "normal"
    peak_demand: bool = False
    load_forecast_uncertainty: str = "low"
    energy_market_volatile: bool = False
    planned_maintenance: bool = False
    critical_equipment_degraded: bool = False
    
    def max_safe_level(self) -> GridAuthorityLevel:
        """Returns maximum authority level permitted by operating conditions."""
        if self.grid_stress_level == "emergency":
            return GridAuthorityLevel.MANUAL_ONLY
        
        if self.critical_equipment_degraded:
            return GridAuthorityLevel.MANUAL_ONLY
        
        if self.weather_severity in ("severe", "extreme"):
            return GridAuthorityLevel.ASSISTED_OPERATION
        
        if self.grid_stress_level == "high":
            return GridAuthorityLevel.ASSISTED_OPERATION
        
        if self.islanded or not self.interconnection_healthy:
            return GridAuthorityLevel.CONSTRAINED_AUTONOMY
        
        if self.load_forecast_uncertainty == "high":
            return GridAuthorityLevel.CONSTRAINED_AUTONOMY
        
        if self.energy_market_volatile and self.peak_demand:
            return GridAuthorityLevel.CONSTRAINED_AUTONOMY
        
        if self.grid_stress_level == "elevated":
            return GridAuthorityLevel.CONSTRAINED_AUTONOMY
        
        return GridAuthorityLevel.FULL_AUTONOMY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Operator State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GridOperatorState:
    """Human operator monitoring for grid control rooms."""
    readiness: float
    on_duty: bool = True
    shift_hours_elapsed: Optional[float] = None
    attention_level: Optional[float] = None
    fatigue_detected: bool = False
    distraction_detected: bool = False
    response_time_ms: Optional[float] = None
    last_manual_action_ts: Optional[float] = None
    recent_overrides: Tuple[Tuple[float, str], ...] = ()
    
    def is_ready_for_handoff(self, current_ts: float) -> Tuple[bool, str]:
        """Determine if operator can safely receive control authority."""
        if not self.on_duty:
            return False, "operator_not_on_duty"
        
        if self.readiness < 0.70:
            return False, f"readiness={self.readiness:.2f} below 0.70"
        
        if self.fatigue_detected:
            return False, "fatigue_detected"
        
        if self.distraction_detected:
            return False, "distraction_detected"
        
        if self.attention_level is not None and self.attention_level < 0.50:
            return False, f"attention={self.attention_level:.2f} below 0.50"
        
        if self.response_time_ms is not None and self.response_time_ms > 5000.0:
            return False, f"response_time={self.response_time_ms:.0f}ms exceeds 5000ms"
        
        if self.last_manual_action_ts is not None:
            time_since_action = current_ts - self.last_manual_action_ts
            if time_since_action > 1800.0:
                return False, f"no_manual_action_for={time_since_action:.0f}s"
        
        if self.shift_hours_elapsed is not None and self.shift_hours_elapsed > 10.0:
            return False, f"shift_hours={self.shift_hours_elapsed:.1f} exceeds 10.0"
        
        return True, "operator_ready"
    
    def requires_attention_alert(self) -> bool:
        """Should we alert operator to increase attention?"""
        if self.fatigue_detected or self.distraction_detected:
            return True
        if self.attention_level is not None and self.attention_level < 0.30:
            return True
        if self.shift_hours_elapsed is not None and self.shift_hours_elapsed > 8.0:
            return True
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System Health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GridSystemHealth:
    """System health metrics for AI/control system infrastructure."""
    compute_latency_ms: Optional[float] = None
    communication_latency_ms: Optional[float] = None
    data_quality_score: Optional[float] = None
    critical_alarms: int = 0
    warning_alarms: int = 0
    stale_measurements: int = 0
    pmu_health_score: Optional[float] = None
    scada_health_score: Optional[float] = None
    forecast_model_accuracy: Optional[float] = None
    state_estimator_converged: bool = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain Input Signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GridSignals:
    """Primary input structure for governance decisions."""
    proposed_level: GridAuthorityLevel
    model_confidence: float
    peer_recommended_levels: Tuple[GridAuthorityLevel, ...] = ()
    safety_monitors: Optional[GridSafetyMonitors] = None
    operational_domain: Optional[GridOperationalDomain] = None
    operator_state: Optional[GridOperatorState] = None
    system_health: Optional[GridSystemHealth] = None
    current_scenario: Optional[str] = None
    timestamp: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Confidence Tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConfidenceTracker:
    """Track confidence trends at multiple timescales for early warning."""
    
    def __init__(self):
        self.short_term: List[Tuple[float, float]] = []
        self.medium_term: List[Tuple[float, float]] = []
        self.long_term: List[Tuple[float, float]] = []
    
    def update(self, ts: float, confidence: float) -> None:
        """Add new confidence sample."""
        self.short_term.append((ts, confidence))
        self.medium_term.append((ts, confidence))
        self.long_term.append((ts, confidence))
        
        self.short_term = [(t, c) for t, c in self.short_term if ts - t <= 60.0]
        self.medium_term = [(t, c) for t, c in self.medium_term if ts - t <= 300.0]
        self.long_term = [(t, c) for t, c in self.long_term if ts - t <= 1800.0]
    
    def is_confidence_declining(self) -> Tuple[bool, str]:
        """Detect confidence erosion as early warning signal."""
        if len(self.short_term) < 5 or len(self.medium_term) < 10:
            return False, "insufficient_history"
        
        short_avg = statistics.fmean([c for _, c in self.short_term])
        medium_avg = statistics.fmean([c for _, c in self.medium_term])
        
        if short_avg < medium_avg - 0.10:
            return True, f"confidence_decline: short={short_avg:.2f} vs medium={medium_avg:.2f}"
        
        if len(self.short_term) >= 5:
            short_values = [c for _, c in self.short_term]
            try:
                variance = statistics.pvariance(short_values)
                if variance > 0.04:
                    return True, f"high_confidence_volatility: variance={variance:.3f}"
            except statistics.StatisticsError:
                pass
        
        return False, ""
    
    def get_trend(self) -> str:
        """Get overall trend: improving, stable, declining."""
        if len(self.short_term) < 5 or len(self.long_term) < 20:
            return "unknown"
        
        short_avg = statistics.fmean([c for _, c in self.short_term])
        long_avg = statistics.fmean([c for _, c in self.long_term])
        
        diff = short_avg - long_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GovernanceEvent:
    """Structured event for compliance logging and post-incident analysis."""
    timestamp: float
    event_type: str
    from_level: GridAuthorityLevel
    to_level: GridAuthorityLevel
    confidence: float
    reasons: List[str]
    metadata: Dict[str, Any]
    safety_status: Optional[str] = None
    grace_status: Optional[str] = None
    consensus_status: Optional[str] = None
    used_fallback: bool = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scenario-Based Policies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ScenarioPolicy:
    """Context-aware policy adjustments based on grid operating scenario."""
    
    SCENARIOS: Dict[str, Dict[str, Any]] = {
        "normal_operations": {
            "max_level": GridAuthorityLevel.FULL_AUTONOMY,
            "min_confidence": 0.90,
            "min_peer_agreement": 0.70,
        },
        "peak_load": {
            "max_level": GridAuthorityLevel.CONSTRAINED_AUTONOMY,
            "min_confidence": 0.92,
            "min_peer_agreement": 0.75,
        },
        "unknown": {
            "max_level": GridAuthorityLevel.ASSISTED_OPERATION,
            "min_confidence": 0.95,
            "min_peer_agreement": 0.90,
        },
    }
    
    @classmethod
    def get_max_level_for_scenario(cls, scenario: Optional[str]) -> GridAuthorityLevel:
        """Get maximum authority level for scenario."""
        if scenario is None:
            scenario = "unknown"
        policy = cls.SCENARIOS.get(scenario, cls.SCENARIOS["unknown"])
        return policy["max_level"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance Policy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GridGovernancePolicy:
    """Domain policy for grid governance."""
    min_operator_readiness: float = 0.60
    min_operator_readiness_for_full: float = 0.75
    max_compute_latency_ms: float = 100.0
    max_communication_latency_ms: float = 200.0
    max_critical_alarms: int = 0
    max_stale_measurements: int = 5
    min_seconds_between_escalations: float = 15.0
    min_seconds_between_downgrades: float = 0.0
    max_allowed_level: GridAuthorityLevel = GridAuthorityLevel.FULL_AUTONOMY
    max_event_log_size: int = 1000
    enable_predictive_warnings: bool = True
    confidence_decline_warning_enabled: bool = True
    enable_scenario_policies: bool = True


def default_grid_config() -> "AileeConfig":
    """Safe defaults for grid governance pipeline configuration."""
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    return AileeConfig(
        accept_threshold=0.90,
        borderline_low=0.75,
        borderline_high=0.90,
        w_stability=0.55,
        w_agreement=0.30,
        w_likelihood=0.15,
        history_window=120,
        forecast_window=20,
        grace_peer_delta=1.0,
        grace_min_peer_agreement_ratio=0.60,
        grace_forecast_epsilon=0.30,
        grace_max_abs_z=3.0,
        consensus_quorum=3,
        consensus_delta=1.0,
        consensus_pass_ratio=0.60,
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
# Grid Governor (Main Controller)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GridGovernor:
    """Production-grade governance controller for AI-assisted grid operations."""
    
    def __init__(
        self,
        cfg: Optional["AileeConfig"] = None,
        policy: Optional[GridGovernancePolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable")
        
        self.cfg = cfg or default_grid_config()
        self.policy = policy or GridGovernancePolicy()
        
        self.cfg.hard_max = float(int(self.policy.max_allowed_level))
        self.cfg.fallback_clamp_max = float(int(self.policy.max_allowed_level))
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        
        self._last_level: GridAuthorityLevel = GridAuthorityLevel.MANUAL_ONLY
        self._last_change_ts: float = 0.0
        self._last_operator_state: Optional[GridOperatorState] = None
        self._confidence_tracker = ConfidenceTracker()
        self._event_log: List[GovernanceEvent] = []
        self._last_event: Optional[GovernanceEvent] = None
        self._last_warning: Optional[Tuple[GridAuthorityLevel, float, str]] = None
    
    def evaluate(self, signals: GridSignals) -> Tuple[GridAuthorityLevel, "DecisionResult"]:
        """
        Evaluate authority authorization decision.
        
        This is the primary entry point. Call at your system rate (1-10Hz typical).
        
        Returns:
            (trusted_level, decision_result)
        """
        ts = float(signals.timestamp if signals.timestamp is not None else time.time())
        
        self._last_operator_state = signals.operator_state
        self._confidence_tracker.update(ts, signals.model_confidence)
        
        gated_level, gate_reasons = self._apply_governance_gates(signals, ts)
        hysteresis_level, hysteresis_reasons = self._apply_hysteresis(ts, gated_level)
        
        if self.policy.enable_predictive_warnings:
            warning = self._predict_required_downgrade(signals, ts)
            if warning is not None:
                self._last_warning = warning
        
        peer_values = [float(int(lvl)) for lvl in signals.peer_recommended_levels]
        
        ctx = {
            "domain": "power_grid",
            "signal": "grid_authority_level",
            "proposed_level": int(signals.proposed_level),
            "gated_level": int(gated_level),
            "gate_reasons": gate_reasons,
            "hysteresis_reasons": hysteresis_reasons,
            "scenario": signals.current_scenario,
            "confidence_trend": self._confidence_tracker.get_trend(),
        }
        
        res = self.pipeline.process(
            raw_value=float(int(hysteresis_level)),
            raw_confidence=float(signals.model_confidence),
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        trusted_level = _level_from_float(res.value)
        
        self._log_governance_event(
            ts,
            signals,
            trusted_level,
            res,
            gate_reasons,
            hysteresis_reasons,
        )
        
        self._commit_state(ts, trusted_level, res)
        
        return trusted_level, res
    
    def get_last_event(self) -> Optional[GovernanceEvent]:
        """Get most recent governance event for logging/monitoring."""
        return self._last_event
    
    def get_event_log(self) -> List[GovernanceEvent]:
        """Get full event log for compliance export."""
        return list(self._event_log)
    
    def get_confidence_trend(self) -> str:
        """Get current confidence trend for monitoring dashboards."""
        return self._confidence_tracker.get_trend()
    
    def _apply_governance_gates(
        self, signals: GridSignals, ts: float
    ) -> Tuple[GridAuthorityLevel, List[str]]:
        """Apply deterministic governance gates that override AI proposals."""
        reasons: List[str] = []
        level = signals.proposed_level
        
        # Deployment cap
        if level > self.policy.max_allowed_level:
            reasons.append(f"Policy cap to {int(self.policy.max_allowed_level)}")
            level = self.policy.max_allowed_level
        
        # Operational domain
        if signals.operational_domain is not None:
            max_level = signals.operational_domain.max_safe_level()
            if level > max_level:
                reasons.append(f"Operational domain limits to {int(max_level)}")
                level = max_level
        
        # Safety monitors
        if signals.safety_monitors is not None:
            safe, issues = signals.safety_monitors.is_safe_for_level(level)
            if not safe:
                for test_level in reversed(list(GridAuthorityLevel)):
                    if test_level < level:
                        ok, _ = signals.safety_monitors.is_safe_for_level(test_level)
                        if ok:
                            reasons.append(
                                f"Safety monitors block {int(level)}, cap to {int(test_level)}: {issues}"
                            )
                            level = test_level
                            break
                else:
                    reasons.append(f"Safety monitors force MANUAL_ONLY: {issues}")
                    level = GridAuthorityLevel.MANUAL_ONLY
        
        # Scenario policy
        if self.policy.enable_scenario_policies and signals.current_scenario:
            scenario_max = ScenarioPolicy.get_max_level_for_scenario(signals.current_scenario)
            if level > scenario_max:
                reasons.append(f"Scenario '{signals.current_scenario}' caps to {int(scenario_max)}")
                level = scenario_max
        
        # Operator readiness
        if signals.operator_state is not None:
            if signals.operator_state.requires_attention_alert():
                reasons.append("Operator attention alert required")
            
            if level >= GridAuthorityLevel.FULL_AUTONOMY:
                if signals.operator_state.readiness < self.policy.min_operator_readiness_for_full:
                    reasons.append(
                        f"Operator readiness={signals.operator_state.readiness:.2f} "
                        f"insufficient for full autonomy"
                    )
                    level = GridAuthorityLevel.CONSTRAINED_AUTONOMY
            
            if level >= GridAuthorityLevel.CONSTRAINED_AUTONOMY:
                if signals.operator_state.readiness < self.policy.min_operator_readiness:
                    reasons.append(
                        f"Operator readiness={signals.operator_state.readiness:.2f} "
                        f"insufficient for autonomy"
                    )
                    level = GridAuthorityLevel.ASSISTED_OPERATION
        
        # System health
        if signals.system_health is not None:
            h = signals.system_health
            
            if h.critical_alarms > self.policy.max_critical_alarms:
                reasons.append(f"Critical alarms={h.critical_alarms} exceed threshold → MANUAL_ONLY")
                return GridAuthorityLevel.MANUAL_ONLY, reasons
            
            if h.stale_measurements > self.policy.max_stale_measurements:
                reasons.append(
                    f"Stale measurements={h.stale_measurements} exceed threshold → cap to ASSISTED"
                )
                level = min(level, GridAuthorityLevel.ASSISTED_OPERATION)
            
            if h.compute_latency_ms is not None:
                if h.compute_latency_ms > self.policy.max_compute_latency_ms:
                    reasons.append(
                        f"Compute latency={h.compute_latency_ms:.0f}ms too high → downgrade"
                    )
                    level = min(level, GridAuthorityLevel.CONSTRAINED_AUTONOMY)
            
            if h.communication_latency_ms is not None:
                if h.communication_latency_ms > self.policy.max_communication_latency_ms:
                    reasons.append(
                        f"Communication latency={h.communication_latency_ms:.0f}ms too high → downgrade"
                    )
                    level = min(level, GridAuthorityLevel.CONSTRAINED_AUTONOMY)
        
        return level, reasons
    
    def _apply_hysteresis(
        self, ts: float, requested: GridAuthorityLevel
    ) -> Tuple[GridAuthorityLevel, List[str]]:
        """Apply hysteresis to prevent mode thrashing."""
        reasons: List[str] = []
        current = self._last_level
        
        if requested == current:
            return requested, reasons
        
        dt = ts - self._last_change_ts
        
        if requested > current:
            if dt < self.policy.min_seconds_between_escalations:
                reasons.append(
                    f"Escalation blocked (dt={dt:.1f}s < "
                    f"{self.policy.min_seconds_between_escalations:.1f}s)"
                )
                return current, reasons
        
        return requested, reasons
    
    def _predict_required_downgrade(
        self, signals: GridSignals, ts: float
    ) -> Optional[Tuple[GridAuthorityLevel, float, str]]:
        """Predict if downgrade will be needed soon based on trends."""
        if self.policy.confidence_decline_warning_enabled:
            declining, reason = self._confidence_tracker.is_confidence_declining()
            if declining:
                return (GridAuthorityLevel.ASSISTED_OPERATION, ts + 30.0, reason)
        
        if signals.operational_domain is not None:
            if signals.operational_domain.grid_stress_level == "elevated":
                return (
                    GridAuthorityLevel.CONSTRAINED_AUTONOMY,
                    ts + 60.0,
                    "Grid stress increasing",
                )
        
        return None
    
    def _log_governance_event(
        self,
        ts: float,
        signals: GridSignals,
        trusted_level: GridAuthorityLevel,
        res: DecisionResult,
        gate_reasons: List[str],
        hysteresis_reasons: List[str],
    ) -> None:
        """Log governance event for compliance and audit trail."""
        event_type = "evaluation"
        if trusted_level != self._last_level:
            event_type = "level_change"
        elif res.used_fallback:
            event_type = "fallback_used"
        elif gate_reasons:
            event_type = "gate_applied"
        
        event = GovernanceEvent(
            timestamp=ts,
            event_type=event_type,
            from_level=self._last_level,
            to_level=trusted_level,
            confidence=signals.model_confidence,
            reasons=res.reasons + gate_reasons + hysteresis_reasons,
            metadata=dict(res.metadata),
            safety_status=str(res.safety_status),
            grace_status=str(res.grace_status),
            consensus_status=str(res.consensus_status),
            used_fallback=res.used_fallback,
        )
        
        self._event_log.append(event)
        self._last_event = event
        
        if len(self._event_log) > self.policy.max_event_log_size:
            self._event_log = self._event_log[-self.policy.max_event_log_size:]
    
    def _commit_state(
        self, ts: float, trusted_level: GridAuthorityLevel, res: DecisionResult
    ) -> None:
        """Commit governance state after successful evaluation."""
        accepted = not res.used_fallback
        if SafetyStatus is not None:
            accepted = accepted and res.safety_status in (
                SafetyStatus.ACCEPTED,
                SafetyStatus.BORDERLINE,
            )
        
        if accepted and trusted_level != self._last_level:
            self._last_level = trusted_level
            self._last_change_ts = ts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_default_governor(
    max_level: GridAuthorityLevel = GridAuthorityLevel.CONSTRAINED_AUTONOMY,
    min_operator_readiness: float = 0.65,
) -> GridGovernor:
    """Create a grid governor with safe, production-ready defaults."""
    policy = GridGovernancePolicy(
        max_allowed_level=max_level,
        min_operator_readiness=min_operator_readiness,
    )
    cfg = default_grid_config()
    return GridGovernor(cfg=cfg, policy=policy)


def create_example_signals() -> GridSignals:
    """Create example signals for testing."""
    return GridSignals(
        proposed_level=GridAuthorityLevel.CONSTRAINED_AUTONOMY,
        model_confidence=0.92,
        peer_recommended_levels=(
            GridAuthorityLevel.CONSTRAINED_AUTONOMY,
            GridAuthorityLevel.CONSTRAINED_AUTONOMY,
            GridAuthorityLevel.FULL_AUTONOMY,
        ),
        safety_monitors=GridSafetyMonitors(
            frequency_deviation_hz=0.03,
            voltage_stability_index=0.92,
            angle_stability_margin=25.0,
            reserve_margin_mw=450.0,
        ),
        operational_domain=GridOperationalDomain(),
        operator_state=GridOperatorState(readiness=0.85, shift_hours_elapsed=3.5),
        system_health=GridSystemHealth(compute_latency_ms=45.0),
        current_scenario="normal_operations",
        timestamp=time.time(),
    )


__version__ = "2.0.0"
__all__ = [
    "GridAuthorityLevel",
    "GridSignals",
    "GridSafetyMonitors",
    "GridOperationalDomain",
    "GridOperatorState",
    "GridSystemHealth",
    "GovernanceEvent",
    "GridGovernancePolicy",
    "ScenarioPolicy",
    "GridGovernor",
    "ConfidenceTracker",
    "default_grid_config",
    "create_default_governor",
    "create_example_signals",
]


if __name__ == "__main__":
    print("=" * 80)
    print("AILEE Grid Governance System - Demo")
    print("=" * 80)
    print()
    
    print("Creating governor...")
    governor = create_default_governor()
    print(f"✓ Governor created")
    print()
    
    print("Testing with example signals...")
    signals = create_example_signals()
    level, decision = governor.evaluate(signals)
    
    print(f"  Proposed: {signals.proposed_level.name}")
    print(f"  Authorized: {level.name}")
    print(f"  Confidence: {signals.model_confidence:.2f}")
    print(f"  Used fallback: {decision.used_fallback}")
    print()
    
    print("=" * 80)
    print("Demo complete.")
    print("=" * 80)
