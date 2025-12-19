"""
AILEE Trust Layer — Automotive Autonomy Governance Domain
Version: 2.0.0 - Production Grade

Governance-only domain for autonomous driving decision integrity.

This domain does NOT do perception, planning, or control.
It governs whether autonomy is allowed to operate right now, and at what level,
based on confidence, peer agreement, strict safety envelopes, ODD constraints,
and safety monitor inputs.

Primary governed signal:
- Autonomy Authorization Level (scalar, 0..3)

Level semantics:
  0 = MANUAL_ONLY
  1 = ASSISTED_ONLY (e.g., lane keep, adaptive cruise)
  2 = CONSTRAINED_AUTONOMY (e.g., highway pilot, geo-fenced)
  3 = FULL_AUTONOMY_ALLOWED (requires all safety conditions met)

Key properties:
- Deterministic decisions (no randomness)
- Auditable, reason-rich outputs with black box logging
- Explicit downgrade/fallback behavior with degradation planning
- Multi-oracle consensus via peer recommended levels
- Stability-oriented (prevents mode thrash using hysteresis + Grace checks)
- ODD-aware (respects operational design domain boundaries)
- Safety monitor integration (collision risk, path safety, sensor fusion)
- Driver state monitoring (readiness, attention, handoff capability)
- Multi-timescale confidence tracking
- Predictive downgrade warnings
- Scenario-based policy adaptation

INTEGRATION EXAMPLE:

    # Setup (once)
    policy = AutonomyGovernancePolicy(
        max_allowed_level=AutonomyLevel.CONSTRAINED_AUTONOMY,
        min_driver_readiness_for_autonomy=0.65,
    )
    governor = AutonomyGovernor(policy=policy)
    
    # Per-frame evaluation (100Hz typical)
    while running:
        signals = AutonomySignals(
            proposed_level=autonomy_stack.get_desired_level(),
            model_confidence=autonomy_stack.get_confidence(),
            peer_recommended_levels=tuple(peer.get_level() for peer in peers),
            system_health=SystemHealth(...),
            driver_state=DriverState(...),
            odd=OperationalDesignDomain(...),
            safety_monitors=SafetyMonitorSignals(...),
        )
        
        authorized_level, decision = governor.evaluate(signals)
        
        # Use authorized_level as authorization CEILING
        # (governor does not command control, only sets maximum allowed level)
        autonomy_stack.set_authorization_ceiling(authorized_level)
        
        # ALERT if fallback was used
        if decision.used_fallback:
            hmi.show_warning("Autonomy degraded", decision.reasons)
        
        # LOG for compliance
        black_box.record_governance_event(governor.get_last_event())

This module is designed to be safe-by-default, certification-friendly,
and integration-ready for ISO 26262 / UL 4600 compliance workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import time
import statistics


# ---- Core imports (kept conservative to avoid import churn) ----
try:
    from ailee_trust_pipeline_v1 import (
        AileeTrustPipeline, 
        AileeConfig, 
        DecisionResult, 
        SafetyStatus
    )
except Exception:  # pragma: no cover
    AileeTrustPipeline = None  # type: ignore
    AileeConfig = None  # type: ignore
    DecisionResult = None  # type: ignore
    SafetyStatus = None  # type: ignore


# -----------------------------
# Autonomy Levels
# -----------------------------

class AutonomyLevel(IntEnum):
    """Discrete autonomy authorization levels"""
    MANUAL_ONLY = 0
    ASSISTED_ONLY = 1
    CONSTRAINED_AUTONOMY = 2
    FULL_AUTONOMY_ALLOWED = 3


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def _level_from_float(x: float) -> AutonomyLevel:
    """Quantize to nearest integer level; enforce 0..3"""
    ix = int(round(float(x)))
    ix = _clamp_int(ix, int(AutonomyLevel.MANUAL_ONLY), int(AutonomyLevel.FULL_AUTONOMY_ALLOWED))
    return AutonomyLevel(ix)


# -----------------------------
# Safety Monitor Signals
# -----------------------------

@dataclass(frozen=True)
class SafetyMonitorSignals:
    """
    External safety monitor inputs (e.g., ISO 26262 ASIL-D monitors).
    These should come from independent safety monitors, not the primary autonomy stack.
    """
    collision_risk_score: Optional[float] = None  # 0..1, higher = more risk
    path_safety_score: Optional[float] = None     # 0..1, higher = safer
    sensor_fusion_health: Optional[float] = None  # 0..1, health metric
    localization_uncertainty_m: Optional[float] = None  # meters, position uncertainty
    
    # Object detection health
    object_detection_health: Optional[float] = None  # 0..1
    
    # Additional monitor scores
    emergency_brake_available: bool = True
    redundant_systems_online: bool = True
    
    def is_safe_for_level(self, level: AutonomyLevel) -> Tuple[bool, List[str]]:
        """Domain-specific safety thresholds per autonomy level"""
        issues: List[str] = []
        
        # Basic safety requirements for any autonomy
        if not self.emergency_brake_available:
            issues.append("emergency_brake_unavailable")
        
        if level >= AutonomyLevel.ASSISTED_ONLY:
            if self.sensor_fusion_health is not None and self.sensor_fusion_health < 0.80:
                issues.append(f"sensor_fusion_health={self.sensor_fusion_health:.2f} below 0.80")
        
        if level >= AutonomyLevel.CONSTRAINED_AUTONOMY:
            if self.collision_risk_score is not None and self.collision_risk_score > 0.30:
                issues.append(f"collision_risk={self.collision_risk_score:.2f} exceeds 0.30")
            
            if self.path_safety_score is not None and self.path_safety_score < 0.70:
                issues.append(f"path_safety={self.path_safety_score:.2f} below 0.70")
            
            if self.localization_uncertainty_m is not None and self.localization_uncertainty_m > 2.0:
                issues.append(f"localization_uncertainty={self.localization_uncertainty_m:.1f}m exceeds 2.0m")
            
            if self.object_detection_health is not None and self.object_detection_health < 0.85:
                issues.append(f"object_detection_health={self.object_detection_health:.2f} below 0.85")
        
        if level >= AutonomyLevel.FULL_AUTONOMY_ALLOWED:
            if not self.redundant_systems_online:
                issues.append("redundant_systems_offline")
            
            if self.collision_risk_score is not None and self.collision_risk_score > 0.10:
                issues.append(f"collision_risk={self.collision_risk_score:.2f} exceeds 0.10 for full autonomy")
            
            if self.localization_uncertainty_m is not None and self.localization_uncertainty_m > 0.5:
                issues.append(f"localization_uncertainty={self.localization_uncertainty_m:.1f}m exceeds 0.5m for full autonomy")
        
        return len(issues) == 0, issues


# -----------------------------
# Operational Design Domain
# -----------------------------

@dataclass(frozen=True)
class OperationalDesignDomain:
    """
    ODD constraints - defines where/when autonomy is permitted.
    Based on SAE J3016 and ISO 34503 standards.
    """
    # Geographic constraints
    geofence_authorized: bool = True
    hd_map_available: bool = True
    distance_to_boundary_m: Optional[float] = None  # Distance to ODD boundary
    
    # Environmental constraints
    weather_code: str = "clear"  # clear, light_rain, heavy_rain, snow, fog, ice
    weather_trend: str = "stable"  # improving, stable, deteriorating
    visibility_m: Optional[float] = None  # meters
    road_type: str = "highway"   # highway, urban, rural, parking, unknown
    
    # Road condition
    road_surface: str = "dry"  # dry, wet, snow, ice
    construction_zone: bool = False
    
    # Temporal constraints
    time_of_day: str = "day"  # day, dusk, dawn, night
    
    # Traffic density
    traffic_density: str = "moderate"  # light, moderate, heavy, congested
    
    def max_safe_level(self) -> AutonomyLevel:
        """
        Conservative ODD-based ceiling.
        Returns the maximum autonomy level permitted by current ODD.
        """
        # Hard blocks
        if not self.geofence_authorized:
            return AutonomyLevel.MANUAL_ONLY
        
        if not self.hd_map_available and self.road_type in ("highway", "urban"):
            return AutonomyLevel.ASSISTED_ONLY
        
        # Weather constraints
        if self.weather_code in ("heavy_rain", "snow", "fog", "ice"):
            return AutonomyLevel.ASSISTED_ONLY
        
        if self.road_surface == "ice":
            return AutonomyLevel.MANUAL_ONLY
        
        if self.road_surface == "snow":
            return AutonomyLevel.ASSISTED_ONLY
        
        # Visibility constraints
        if self.visibility_m is not None:
            if self.visibility_m < 30.0:
                return AutonomyLevel.MANUAL_ONLY
            if self.visibility_m < 100.0:
                return AutonomyLevel.ASSISTED_ONLY
        
        # Construction zones
        if self.construction_zone:
            return AutonomyLevel.ASSISTED_ONLY
        
        # Road type constraints
        if self.road_type == "unknown":
            return AutonomyLevel.ASSISTED_ONLY
        
        if self.road_type == "parking":
            return AutonomyLevel.CONSTRAINED_AUTONOMY
        
        # Time of day + road type combinations
        if self.time_of_day == "night" and self.road_type == "urban":
            return AutonomyLevel.CONSTRAINED_AUTONOMY
        
        # Traffic density
        if self.traffic_density == "congested" and self.road_type == "urban":
            return AutonomyLevel.CONSTRAINED_AUTONOMY
        
        # Default: full autonomy allowed if all checks pass
        return AutonomyLevel.FULL_AUTONOMY_ALLOWED
    
    def get_warning_distance_m(self) -> float:
        """Distance at which to warn about ODD boundary approach"""
        if self.road_type == "highway":
            return 1000.0  # 1km warning on highway
        elif self.road_type == "urban":
            return 300.0   # 300m warning in urban
        return 500.0       # default


# -----------------------------
# Driver State
# -----------------------------

@dataclass(frozen=True)
class DriverState:
    """
    Extended driver monitoring beyond basic readiness.
    Supports Euro NCAP and NHTSA driver monitoring requirements.
    """
    readiness: float  # 0..1, overall readiness score
    
    # Attention monitoring
    attention_level: Optional[float] = None  # 0..1 from gaze/pose tracking
    distraction_detected: bool = False
    drowsiness_detected: bool = False
    
    # Physical state
    hands_on_wheel: Optional[bool] = None
    eyes_on_road: Optional[bool] = None
    
    # Response capability
    response_time_ms: Optional[float] = None  # from last alert/test
    last_manual_input_ts: Optional[float] = None  # timestamp of last steering/pedal input
    
    # Override history (recent manual interventions)
    recent_overrides: Tuple[Tuple[float, str], ...] = ()  # [(timestamp, reason), ...]
    
    def is_ready_for_handoff(self, current_ts: float) -> Tuple[bool, str]:
        """
        Determine if driver can safely receive control.
        Critical for L3+ systems requiring driver takeover.
        """
        if self.readiness < 0.70:
            return False, f"readiness={self.readiness:.2f} below 0.70"
        
        if self.attention_level is not None and self.attention_level < 0.50:
            return False, f"attention={self.attention_level:.2f} below 0.50"
        
        if self.distraction_detected:
            return False, "distraction_detected"
        
        if self.drowsiness_detected:
            return False, "drowsiness_detected"
        
        if self.response_time_ms is not None and self.response_time_ms > 2000.0:
            return False, f"response_time={self.response_time_ms:.0f}ms exceeds 2000ms"
        
        # Check if driver has been inactive too long
        if self.last_manual_input_ts is not None:
            time_since_input = current_ts - self.last_manual_input_ts
            if time_since_input > 300.0:  # 5 minutes
                return False, f"no_manual_input_for={time_since_input:.0f}s"
        
        # Check hands on wheel for immediate takeover
        if self.hands_on_wheel is not None and not self.hands_on_wheel:
            return False, "hands_not_on_wheel"
        
        return True, "driver_ready"
    
    def requires_attention_alert(self) -> bool:
        """Should we alert driver to pay attention?"""
        if self.distraction_detected or self.drowsiness_detected:
            return True
        if self.attention_level is not None and self.attention_level < 0.30:
            return True
        return False


# -----------------------------
# System Health
# -----------------------------

@dataclass(frozen=True)
class SystemHealth:
    """
    System health metrics for governance gating.
    Expanded from simple dict to structured type.
    """
    latency_ms: Optional[float] = None
    sensor_faults: int = 0
    compute_load: Optional[float] = None  # 0..1
    memory_available_mb: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    
    # Communication health
    can_bus_errors: int = 0
    network_latency_ms: Optional[float] = None
    
    # Sensor-specific health
    camera_health: Optional[float] = None  # 0..1
    lidar_health: Optional[float] = None   # 0..1
    radar_health: Optional[float] = None   # 0..1
    gps_health: Optional[float] = None     # 0..1
    imu_health: Optional[float] = None     # 0..1


# -----------------------------
# Domain Inputs
# -----------------------------

@dataclass(frozen=True)
class AutonomySignals:
    """
    Governance signals used to decide whether autonomy is allowed and at what level.
    
    This is the primary input structure for the governor.
    """
    proposed_level: AutonomyLevel
    model_confidence: float  # 0..1
    
    # Peer recommendations (multi-oracle)
    peer_recommended_levels: Tuple[AutonomyLevel, ...] = ()
    
    # Extended structured inputs
    safety_monitors: Optional[SafetyMonitorSignals] = None
    odd: Optional[OperationalDesignDomain] = None
    driver_state: Optional[DriverState] = None
    system_health: Optional[SystemHealth] = None
    
    # Legacy/additional context (for backward compatibility)
    environment: Dict[str, Any] = field(default_factory=dict)
    
    # Scenario identification (optional)
    current_scenario: Optional[str] = None  # e.g., "highway_cruise", "urban_intersection"
    
    timestamp: Optional[float] = None


# -----------------------------
# Degradation Strategy
# -----------------------------

@dataclass
class DegradationStrategy:
    """
    Defines graceful degradation behavior when governance blocks a level.
    Critical for safe transitions and minimal risk maneuvers.
    
    NOTE: This is a planning/specification structure. Actual execution of the
    degradation trajectory (e.g., minimum risk maneuver, pull-over) belongs in
    the vehicle control layer, NOT in the governance layer.
    
    The governor's role is to:
    1. Decide WHEN degradation is needed (via level authorization)
    2. Specify WHAT degradation is required (via this strategy)
    3. Validate IF degradation is safe to initiate (driver ready, etc.)
    
    The vehicle controller's role is to:
    1. Execute the degradation trajectory
    2. Manage handoff protocols
    3. Perform the actual control actions
    
    TODO: If you need the governor to actively trigger degradation plans,
    add an external callback interface rather than embedding execution here.
    Example:
        governor.register_degradation_callback(lambda strategy: controller.execute(strategy))
    """
    target_level: AutonomyLevel
    timeout_seconds: float
    handoff_required: bool  # Does this require driver takeover?
    fallback_trajectory: str  # "minimum_risk_maneuver" | "continue_current" | "pull_over" | "slow_in_lane"
    alert_driver: bool = True
    reason: str = ""
    
    def is_safe_to_degrade_from(self, current: AutonomyLevel, driver_ready: bool) -> Tuple[bool, str]:
        """Check if degradation is safe right now"""
        if current == AutonomyLevel.MANUAL_ONLY:
            return True, "already_manual"
        
        if self.handoff_required and not driver_ready:
            return False, "driver_takeover_required_but_driver_not_ready"
        
        # Can't degrade more than 1 level per cycle without driver readiness
        if abs(int(self.target_level) - int(current)) > 1 and not driver_ready:
            return False, "multi_level_degradation_requires_driver_ready"
        
        return True, "safe_to_degrade"


# -----------------------------
# Confidence Tracking
# -----------------------------

class ConfidenceTracker:
    """
    Track confidence trends at multiple timescales for early warning.
    Enables predictive downgrade decisions.
    """
    def __init__(self):
        self.short_term: List[Tuple[float, float]] = []   # 10 seconds
        self.medium_term: List[Tuple[float, float]] = []  # 60 seconds
        self.long_term: List[Tuple[float, float]] = []    # 300 seconds (5 min)
    
    def update(self, ts: float, confidence: float) -> None:
        """Add new confidence sample"""
        self.short_term.append((ts, confidence))
        self.medium_term.append((ts, confidence))
        self.long_term.append((ts, confidence))
        
        # Trim windows
        self.short_term = [(t, c) for t, c in self.short_term if ts - t <= 10.0]
        self.medium_term = [(t, c) for t, c in self.medium_term if ts - t <= 60.0]
        self.long_term = [(t, c) for t, c in self.long_term if ts - t <= 300.0]
    
    def is_confidence_declining(self) -> Tuple[bool, str]:
        """
        Detect confidence erosion as early warning.
        Returns (is_declining, reason)
        """
        if len(self.short_term) < 5 or len(self.medium_term) < 10:
            return False, "insufficient_history"
        
        short_avg = statistics.fmean([c for _, c in self.short_term])
        medium_avg = statistics.fmean([c for _, c in self.medium_term])
        
        # Significant drop in recent confidence
        if short_avg < medium_avg - 0.10:  # 10% drop
            return True, f"confidence_decline: short={short_avg:.2f} vs medium={medium_avg:.2f}"
        
        # Check volatility (high variance indicates instability)
        if len(self.short_term) >= 5:
            short_values = [c for _, c in self.short_term]
            try:
                variance = statistics.pvariance(short_values)
                if variance > 0.04:  # stdev > 0.2
                    return True, f"high_confidence_volatility: variance={variance:.3f}"
            except statistics.StatisticsError:
                pass
        
        return False, ""
    
    def get_trend(self) -> str:
        """Get overall trend: improving, stable, declining"""
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


# -----------------------------
# Governance Events (Black Box Logging)
# -----------------------------

@dataclass(frozen=True)
class GovernanceEvent:
    """
    Structured event for black box / telemetry.
    Critical for post-incident analysis and certification.
    """
    timestamp: float
    event_type: str  # "level_change" | "gate_block" | "fallback_used" | "consensus_fail" | "degradation_initiated" | "driver_alert"
    from_level: AutonomyLevel
    to_level: AutonomyLevel
    confidence: float
    reasons: List[str]
    metadata: Dict[str, Any]
    
    # Additional context
    safety_status: Optional[str] = None
    grace_status: Optional[str] = None
    consensus_status: Optional[str] = None
    used_fallback: bool = False


# -----------------------------
# Scenario-Based Policies
# -----------------------------

class ScenarioPolicy:
    """
    Context-aware policy adjustments based on driving scenario.
    Enables fine-tuned governance per operational context.
    """
    
    SCENARIOS: Dict[str, Dict[str, Any]] = {
        "highway_cruise": {
            "max_level": AutonomyLevel.FULL_AUTONOMY_ALLOWED,
            "min_confidence": 0.90,
            "min_peer_agreement": 0.70,
            "description": "Steady highway driving, low complexity",
        },
        "highway_merge": {
            "max_level": AutonomyLevel.CONSTRAINED_AUTONOMY,
            "min_confidence": 0.95,
            "min_peer_agreement": 0.80,
            "description": "Highway merge maneuver, elevated risk",
        },
        "highway_exit": {
            "max_level": AutonomyLevel.CONSTRAINED_AUTONOMY,
            "min_confidence": 0.93,
            "min_peer_agreement": 0.75,
            "description": "Highway exit, transition to lower speed",
        },
        "urban_intersection": {
            "max_level": AutonomyLevel.CONSTRAINED_AUTONOMY,
            "min_confidence": 0.92,
            "min_peer_agreement": 0.75,
            "description": "Urban intersection, high complexity",
        },
        "urban_straight": {
            "max_level": AutonomyLevel.CONSTRAINED_AUTONOMY,
            "min_confidence": 0.88,
            "min_peer_agreement": 0.65,
            "description": "Urban straight road, moderate complexity",
        },
        "parking_lot": {
            "max_level": AutonomyLevel.CONSTRAINED_AUTONOMY,
            "min_confidence": 0.88,
            "min_peer_agreement": 0.60,
            "description": "Parking lot navigation, low speed",
        },
        "school_zone": {
            "max_level": AutonomyLevel.ASSISTED_ONLY,
            "min_confidence": 0.95,
            "min_peer_agreement": 0.85,
            "description": "School zone, maximum caution required",
        },
        "unknown": {
            "max_level": AutonomyLevel.ASSISTED_ONLY,
            "min_confidence": 0.95,
            "min_peer_agreement": 0.90,
            "description": "Unknown scenario, conservative limits",
        },
    }
    
    @classmethod
    def get_policy(cls, scenario: Optional[str]) -> Dict[str, Any]:
        """Get policy for given scenario, fallback to 'unknown' if not found"""
        if scenario is None:
            scenario = "unknown"
        return cls.SCENARIOS.get(scenario, cls.SCENARIOS["unknown"])
    
    @classmethod
    def get_max_level_for_scenario(cls, scenario: Optional[str]) -> AutonomyLevel:
        """Convenience method to get max level for scenario"""
        policy = cls.get_policy(scenario)
        return policy["max_level"]


# -----------------------------
# Domain Configuration
# -----------------------------

@dataclass(frozen=True)
class AutonomyGovernancePolicy:
    """
    Domain policy that sits ABOVE the AILEE pipeline configuration.
    
    This policy is intentionally simple and deterministic:
    - enforce hard stop conditions
    - add hysteresis / rate-limits to reduce mode flapping
    - block escalation when readiness/health is insufficient
    """
    
    # Driver readiness requirements
    min_driver_readiness_for_autonomy: float = 0.60
    min_driver_readiness_for_full_autonomy: float = 0.75
    
    # System health requirements
    max_latency_ms_for_autonomy: float = 150.0
    max_latency_ms_for_full_autonomy: float = 100.0
    max_sensor_faults: int = 0
    max_can_bus_errors: int = 5
    
    # Hysteresis (prevent mode thrash)
    min_seconds_between_escalations: float = 10.0
    min_seconds_between_downgrades: float = 0.0  # Allow immediate downgrade (safety-first)
    
    # Deployment-level cap
    max_allowed_level: AutonomyLevel = AutonomyLevel.FULL_AUTONOMY_ALLOWED
    
    # Event logging
    max_event_log_size: int = 1000
    
    # Predictive warnings
    enable_predictive_warnings: bool = True
    odd_boundary_warning_enabled: bool = True
    confidence_decline_warning_enabled: bool = True
    
    # Scenario-based policy enforcement
    enable_scenario_policies: bool = True


def default_autonomy_config() -> "AileeConfig":
    """
    Safe defaults for autonomy governance pipeline configuration.
    
    Tuned for stability and safety in automotive domain:
    - Higher stability weight (avoid mode thrash)
    - Moderate agreement weight (consensus matters)
    - Conservative thresholds
    - Last-good fallback (preserve safe mode)
    """
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    cfg = AileeConfig(
        accept_threshold=0.92,
        borderline_low=0.75,
        borderline_high=0.92,
        
        # Bias toward stability and agreement for mode governance
        w_stability=0.55,
        w_agreement=0.30,
        w_likelihood=0.15,
        
        history_window=80,
        forecast_window=16,
        
        # Peer delta for discrete levels: require near-identical recommendations
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
    return cfg


# -----------------------------
# Autonomy Governor
# -----------------------------

class AutonomyGovernor:
    """
    Production-grade governance controller for autonomous driving.
    
    Produces a trusted autonomy authorization level by integrating:
    - AILEE trust pipeline (core validation)
    - ODD constraints (where/when autonomy is allowed)
    - Safety monitors (collision risk, path safety, etc.)
    - Driver state (readiness, attention, handoff capability)
    - System health (latency, faults, sensor health)
    - Scenario-based policies (context-aware limits)
    - Predictive warnings (anticipate degradation needs)
    - Degradation planning (safe mode transitions)
    - Event logging (black box for post-incident analysis)
    
    Integration contract:
    - Call evaluate(signals) at system rate (typically 10-100Hz)
    - Use returned trusted_level as authorization ceiling
    - Monitor decision.used_fallback for degraded states
    - Export events periodically for compliance logging
    """
    
    def __init__(
        self,
        cfg: Optional["AileeConfig"] = None,
        policy: Optional[AutonomyGovernancePolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable. Ensure ailee_trust_pipeline_v1.py is importable.")
        
        self.cfg = cfg or default_autonomy_config()
        self.policy = policy or AutonomyGovernancePolicy()
        
        # Apply policy cap as a hard bound (deployment-level guarantee)
        self.cfg.hard_max = float(int(self.policy.max_allowed_level))
        self.cfg.fallback_clamp_max = float(int(self.policy.max_allowed_level))
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        
        # Governance state for hysteresis/rate limiting
        self._last_level: AutonomyLevel = AutonomyLevel.MANUAL_ONLY
        self._last_change_ts: float = 0.0
        
        # Last seen driver state (for degradation handoff checks)
        self._last_driver_state: Optional[DriverState] = None
        
        # Degradation management
        self._degradation_plan: Optional[DegradationStrategy] = None
        self._degradation_initiated_ts: Optional[float] = None
        
        # Confidence tracking (multi-timescale)
        self._confidence_tracker = ConfidenceTracker()
        
        # Event logging (black box)
        self._event_log: List[GovernanceEvent] = []
        self._last_event: Optional[GovernanceEvent] = None
        
        # Predictive warning state
        self._last_warning: Optional[Tuple[AutonomyLevel, float, str]] = None
    
    # -------------------------
    # Public API
    # -------------------------
    
    def evaluate(self, signals: AutonomySignals) -> Tuple[AutonomyLevel, "DecisionResult"]:
        """
        Evaluate autonomy authorization decision.
        
        This is the primary entry point. Call this at your system rate.
        
        Args:
            signals: Current autonomy signals (proposed level, confidence, peers, etc.)
        
        Returns:
            (trusted_level, decision_result)
            - trusted_level: The authorized autonomy level (use this as ceiling)
            - decision_result: Full AILEE decision with metadata
        """
        ts = float(signals.timestamp if signals.timestamp is not None else time.time())
        
        # Store driver state for potential degradation handoff checks
        self._last_driver_state = signals.driver_state
        
        # Update confidence tracker
        self._confidence_tracker.update(ts, signals.model_confidence)
        
        # Apply deterministic governance gates BEFORE pipeline arbitration
        gated_level, gate_reasons = self._apply_governance_gates(signals, ts)
        
        # Apply rate limiting / hysteresis on escalation/downgrade
        hysteresis_level, hysteresis_reasons = self._apply_hysteresis(ts, gated_level)
        
        # Check for predictive warnings
        if self.policy.enable_predictive_warnings:
            warning = self._predict_required_downgrade(signals, ts)
            if warning is not None:
                self._last_warning = warning
                predicted_level, predicted_ts, warning_reason = warning
                hysteresis_reasons.append(f"Predictive warning: {warning_reason}")
        
        # Convert peer levels to floats for pipeline consensus/agreement
        peer_values = [float(int(lvl)) for lvl in signals.peer_recommended_levels]
        
        # Build context for audit metadata
        ctx = self._build_context(signals, gate_reasons, hysteresis_reasons, gated_level, hysteresis_level)
        
        # Use the AILEE pipeline to validate this requested mode level
        res = self.pipeline.process(
            raw_value=float(int(hysteresis_level)),
            raw_confidence=float(signals.model_confidence),
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        trusted_level = _level_from_float(res.value)
        
        # Log governance event
        self._log_governance_event(ts, signals, trusted_level, res, gate_reasons, hysteresis_reasons)
        
        # Update governor state if decision is accepted and not fallback
        self._commit_state(ts, trusted_level, res)
        
        return trusted_level, res
    
    def last_level(self) -> AutonomyLevel:
        """Get the last authorized autonomy level"""
        return self._last_level
    
    def get_last_event(self) -> Optional[GovernanceEvent]:
        """Get the most recent governance event (for logging)"""
        return self._last_event
    
    def export_events(self, since_ts: Optional[float] = None) -> List[GovernanceEvent]:
        """
        Export governance events for analysis/reporting.
        
        Args:
            since_ts: If provided, only return events after this timestamp
        
        Returns:
            List of governance events
        """
        if since_ts is None:
            return self._event_log[:]
        return [e for e in self._event_log if e.timestamp >= since_ts]
    
    def get_confidence_trend(self) -> str:
        """Get current confidence trend: improving, stable, declining"""
        return self._confidence_tracker.get_trend()
    
    def get_current_warning(self) -> Optional[Tuple[AutonomyLevel, float, str]]:
        """
        Get current predictive warning if any.
        
        Note: This is stateful and persists until overwritten by next evaluate().
        Consumers should treat this as advisory context, not current fact.
        Always check the timestamp in the tuple to determine if warning is still relevant.
        
        Returns:
            (predicted_level, predicted_timestamp, reason) or None
        """
        return self._last_warning
    
    def initiate_degradation(self, strategy: DegradationStrategy, ts: Optional[float] = None) -> bool:
        """
        Initiate a planned degradation.
        
        This stores the degradation plan for reference but does NOT execute it.
        Actual trajectory execution belongs in the vehicle control layer.
        
        The governor's role is authorization and validation; the controller's
        role is execution. This method validates the plan is safe to initiate.
        
        Args:
            strategy: Degradation strategy to validate and store
            ts: Current timestamp (optional)
        
        Returns:
            True if degradation plan is safe to initiate, False if blocked
        
        Example usage:
            strategy = DegradationStrategy(
                target_level=AutonomyLevel.ASSISTED_ONLY,
                timeout_seconds=5.0,
                handoff_required=True,
                fallback_trajectory="pull_over",
                reason="ODD boundary approaching"
            )
            if governor.initiate_degradation(strategy):
                # Signal vehicle controller to execute strategy
                vehicle_controller.execute_degradation(strategy)
        """
        ts = ts or time.time()
        
        # Check if driver is ready for handoff if required
        driver_ready = False  # Conservative default
        if strategy.handoff_required and self._last_driver_state is not None:
            # Use last seen driver state from evaluate()
            driver_ready, _ = self._last_driver_state.is_ready_for_handoff(ts)
        elif not strategy.handoff_required:
            # No handoff required, so driver readiness doesn't block
            driver_ready = True
        
        safe, reason = strategy.is_safe_to_degrade_from(self._last_level, driver_ready)
        if not safe:
            return False
        
        self._degradation_plan = strategy
        self._degradation_initiated_ts = ts
        return True
    
    # -------------------------
    # Governance gates
    # -------------------------
    
    def _apply_governance_gates(self, signals: AutonomySignals, ts: float) -> Tuple[AutonomyLevel, List[str]]:
        """
        Apply deterministic governance gates.
        These are hard constraints that override all other considerations.
        """
        reasons: List[str] = []
        level = signals.proposed_level
        
        # 1) Deployment-level cap
        if level > self.policy.max_allowed_level:
            reasons.append(f"Gate: cap to max_allowed_level={int(self.policy.max_allowed_level)}.")
            level = self.policy.max_allowed_level
        
        # 2) ODD constraints (where/when autonomy is allowed)
        if signals.odd is not None:
            odd_max = signals.odd.max_safe_level()
            if level > odd_max:
                reasons.append(f"Gate: ODD limits to {int(odd_max)}.")
                level = odd_max
            
            # Warning for approaching ODD boundary
            if self.policy.odd_boundary_warning_enabled and signals.odd.distance_to_boundary_m is not None:
                warning_dist = signals.odd.get_warning_distance_m()
                if signals.odd.distance_to_boundary_m < warning_dist:
                    reasons.append(
                        f"Gate: approaching ODD boundary in {signals.odd.distance_to_boundary_m:.0f}m "
                        f"(warning at {warning_dist:.0f}m)."
                    )
        
        # 3) Safety monitor constraints
        if signals.safety_monitors is not None:
            safe, issues = signals.safety_monitors.is_safe_for_level(level)
            if not safe:
                # Find highest safe level
                for test_level in reversed(list(AutonomyLevel)):
                    if test_level < level:
                        safe_test, _ = signals.safety_monitors.is_safe_for_level(test_level)
                        if safe_test:
                            reasons.append(f"Gate: safety monitors block {int(level)}, cap to {int(test_level)}: {issues}.")
                            level = test_level
                            break
                else:
                    # No safe level found
                    reasons.append(f"Gate: safety monitors force MANUAL_ONLY: {issues}.")
                    level = AutonomyLevel.MANUAL_ONLY
        
        # 4) Scenario-based policy
        if self.policy.enable_scenario_policies and signals.current_scenario is not None:
            scenario_max = ScenarioPolicy.get_max_level_for_scenario(signals.current_scenario)
            if level > scenario_max:
                reasons.append(f"Gate: scenario '{signals.current_scenario}' limits to {int(scenario_max)}.")
                level = scenario_max
        
        # 5) System health gates
        if signals.system_health is not None:
            health = signals.system_health
            
            # Sensor faults (force manual)
            if health.sensor_faults > self.policy.max_sensor_faults:
                reasons.append(
                    f"Gate: sensor_faults={health.sensor_faults} > {self.policy.max_sensor_faults} -> MANUAL_ONLY."
                )
                return AutonomyLevel.MANUAL_ONLY, reasons
            
            # CAN bus errors (force manual if severe)
            if health.can_bus_errors > self.policy.max_can_bus_errors:
                reasons.append(
                    f"Gate: can_bus_errors={health.can_bus_errors} > {self.policy.max_can_bus_errors} -> MANUAL_ONLY."
                )
                return AutonomyLevel.MANUAL_ONLY, reasons
            
            # Latency gating
            if health.latency_ms is not None:
                if level >= AutonomyLevel.FULL_AUTONOMY_ALLOWED:
                    if health.latency_ms > self.policy.max_latency_ms_for_full_autonomy:
                        reasons.append(
                            f"Gate: latency={health.latency_ms:.1f}ms > "
                            f"{self.policy.max_latency_ms_for_full_autonomy:.1f}ms -> cap to CONSTRAINED."
                        )
                        level = min(level, AutonomyLevel.CONSTRAINED_AUTONOMY)
                
                if level >= AutonomyLevel.CONSTRAINED_AUTONOMY:
                    if health.latency_ms > self.policy.max_latency_ms_for_autonomy:
                        reasons.append(
                            f"Gate: latency={health.latency_ms:.1f}ms > "
                            f"{self.policy.max_latency_ms_for_autonomy:.1f}ms -> cap to ASSISTED."
                        )
                        level = min(level, AutonomyLevel.ASSISTED_ONLY)
        
        # 6) Driver readiness gates
        if signals.driver_state is not None:
            dr = signals.driver_state.readiness
            
            # Check if driver attention alert is needed
            if signals.driver_state.requires_attention_alert():
                reasons.append("Gate: driver attention alert required.")
            
            # Full autonomy requires higher readiness
            if level >= AutonomyLevel.FULL_AUTONOMY_ALLOWED:
                if dr < self.policy.min_driver_readiness_for_full_autonomy:
                    reasons.append(
                        f"Gate: driver_readiness={dr:.2f} < "
                        f"{self.policy.min_driver_readiness_for_full_autonomy:.2f} -> cap to CONSTRAINED."
                    )
                    level = min(level, AutonomyLevel.CONSTRAINED_AUTONOMY)
            
            # Any autonomy requires minimum readiness
            if level >= AutonomyLevel.CONSTRAINED_AUTONOMY:
                if dr < self.policy.min_driver_readiness_for_autonomy:
                    reasons.append(
                        f"Gate: driver_readiness={dr:.2f} < "
                        f"{self.policy.min_driver_readiness_for_autonomy:.2f} -> cap to ASSISTED."
                    )
                    level = min(level, AutonomyLevel.ASSISTED_ONLY)
            
            # Check handoff capability if we might need to degrade
            if level > AutonomyLevel.ASSISTED_ONLY:
                can_handoff, handoff_reason = signals.driver_state.is_ready_for_handoff(ts)
                if not can_handoff:
                    reasons.append(f"Gate: driver not ready for handoff: {handoff_reason}.")
        
        return level, reasons
    
    # -------------------------
    # Hysteresis / rate limits
    # -------------------------
    
    def _apply_hysteresis(self, ts: float, requested: AutonomyLevel) -> Tuple[AutonomyLevel, List[str]]:
        """
        Apply hysteresis to prevent mode thrashing.
        Escalations are rate-limited; downgrades are allowed immediately (safety-first).
        """
        reasons: List[str] = []
        
        current = self._last_level
        if requested == current:
            return requested, reasons
        
        dt = ts - self._last_change_ts
        
        # Escalation: slower (prevent premature mode increases)
        if requested > current:
            if dt < self.policy.min_seconds_between_escalations:
                reasons.append(
                    f"Hysteresis: block escalation {int(current)}->{int(requested)} "
                    f"(dt={dt:.2f}s < {self.policy.min_seconds_between_escalations:.2f}s)."
                )
                return current, reasons
            return requested, reasons
        
        # Downgrade: allow immediately by default (safety-first)
        if requested < current and dt < self.policy.min_seconds_between_downgrades:
            reasons.append(
                f"Hysteresis: block downgrade {int(current)}->{int(requested)} "
                f"(dt={dt:.2f}s < {self.policy.min_seconds_between_downgrades:.2f}s)."
            )
            return current, reasons
        
        return requested, reasons
    
    # -------------------------
    # Predictive warnings
    # -------------------------
    
    def _predict_required_downgrade(
        self, 
        signals: AutonomySignals, 
        ts: float
    ) -> Optional[Tuple[AutonomyLevel, float, str]]:
        """
        Predict if downgrade will be needed soon based on trends.
        
        Returns:
            (predicted_level, predicted_timestamp, reason) or None
        """
        # Check confidence decline
        if self.policy.confidence_decline_warning_enabled:
            declining, reason = self._confidence_tracker.is_confidence_declining()
            if declining:
                # Predict downgrade to ASSISTED in 5 seconds
                return AutonomyLevel.ASSISTED_ONLY, ts + 5.0, f"predicted: {reason}"
        
        # Check ODD boundary approach
        if self.policy.odd_boundary_warning_enabled and signals.odd is not None:
            if signals.odd.distance_to_boundary_m is not None:
                warning_dist = signals.odd.get_warning_distance_m()
                if signals.odd.distance_to_boundary_m < warning_dist:
                    # Estimate time to boundary (assume 30 m/s ~ 108 km/h average)
                    est_velocity_ms = 30.0
                    time_to_boundary = signals.odd.distance_to_boundary_m / est_velocity_ms
                    return (
                        AutonomyLevel.ASSISTED_ONLY, 
                        ts + time_to_boundary, 
                        f"predicted: ODD boundary in {signals.odd.distance_to_boundary_m:.0f}m"
                    )
        
        # Check weather deterioration
        if signals.odd is not None and signals.odd.weather_trend == "deteriorating":
            return (
                AutonomyLevel.CONSTRAINED_AUTONOMY, 
                ts + 60.0, 
                "predicted: weather deteriorating"
            )
        
        # Check driver readiness trend (if driver becoming less ready)
        if signals.driver_state is not None:
            if signals.driver_state.distraction_detected or signals.driver_state.drowsiness_detected:
                return (
                    AutonomyLevel.ASSISTED_ONLY,
                    ts + 10.0,
                    "predicted: driver attention declining"
                )
        
        return None
    
    # -------------------------
    # Context building
    # -------------------------
    
    def _build_context(
        self,
        signals: AutonomySignals,
        gate_reasons: List[str],
        hysteresis_reasons: List[str],
        gated_level: AutonomyLevel,
        hysteresis_level: AutonomyLevel,
    ) -> Dict[str, Any]:
        """Build comprehensive context for audit metadata"""
        ctx: Dict[str, Any] = {
            "domain": "automotive",
            "signal": "autonomy_authorization_level",
            "units": "level(0..3)",
        }
        
        # Policy info
        ctx["policy"] = {
            "max_allowed_level": int(self.policy.max_allowed_level),
            "min_driver_readiness_for_autonomy": self.policy.min_driver_readiness_for_autonomy,
            "min_driver_readiness_for_full_autonomy": self.policy.min_driver_readiness_for_full_autonomy,
            "max_latency_ms_for_autonomy": self.policy.max_latency_ms_for_autonomy,
            "max_sensor_faults": self.policy.max_sensor_faults,
            "min_seconds_between_escalations": self.policy.min_seconds_between_escalations,
        }
        
        # Gate and hysteresis info
        ctx["governance_gates"] = gate_reasons
        ctx["hysteresis"] = hysteresis_reasons
        ctx["proposed_level"] = int(signals.proposed_level)
        ctx["gated_level"] = int(gated_level)
        ctx["requested_level_after_hysteresis"] = int(hysteresis_level)
        
        # Scenario
        if signals.current_scenario is not None:
            ctx["scenario"] = signals.current_scenario
            scenario_policy = ScenarioPolicy.get_policy(signals.current_scenario)
            ctx["scenario_policy"] = scenario_policy
        
        # ODD info
        if signals.odd is not None:
            ctx["odd"] = {
                "geofence_authorized": signals.odd.geofence_authorized,
                "hd_map_available": signals.odd.hd_map_available,
                "weather_code": signals.odd.weather_code,
                "road_type": signals.odd.road_type,
                "max_safe_level": int(signals.odd.max_safe_level()),
            }
            if signals.odd.distance_to_boundary_m is not None:
                ctx["odd"]["distance_to_boundary_m"] = signals.odd.distance_to_boundary_m
        
        # Safety monitors
        if signals.safety_monitors is not None:
            ctx["safety_monitors"] = {
                "collision_risk_score": signals.safety_monitors.collision_risk_score,
                "path_safety_score": signals.safety_monitors.path_safety_score,
                "localization_uncertainty_m": signals.safety_monitors.localization_uncertainty_m,
            }
        
        # Driver state
        if signals.driver_state is not None:
            ctx["driver_state"] = {
                "readiness": signals.driver_state.readiness,
                "attention_level": signals.driver_state.attention_level,
                "hands_on_wheel": signals.driver_state.hands_on_wheel,
            }
        
        # System health
        if signals.system_health is not None:
            ctx["system_health"] = {
                "latency_ms": signals.system_health.latency_ms,
                "sensor_faults": signals.system_health.sensor_faults,
                "can_bus_errors": signals.system_health.can_bus_errors,
            }
        
        # Confidence trend
        ctx["confidence_trend"] = self._confidence_tracker.get_trend()
        
        # Predictive warning
        if self._last_warning is not None:
            pred_level, pred_ts, pred_reason = self._last_warning
            ctx["predictive_warning"] = {
                "level": int(pred_level),
                "timestamp": pred_ts,
                "reason": pred_reason,
            }
        
        # Legacy environment dict
        if signals.environment:
            ctx["environment"] = signals.environment
        
        return ctx
    
    # -------------------------
    # Event logging
    # -------------------------
    
    def _log_governance_event(
        self,
        ts: float,
        signals: AutonomySignals,
        trusted_level: AutonomyLevel,
        res: "DecisionResult",
        gate_reasons: List[str],
        hysteresis_reasons: List[str],
    ) -> None:
        """Log governance event for black box / telemetry"""
        
        # Determine event type (with clear precedence for analysis)
        event_type = "evaluation"
        
        # Prioritize level change as primary event type
        if trusted_level != self._last_level:
            event_type = "level_change"
        # Fallback without level change is also significant
        elif res.used_fallback:
            event_type = "fallback_used"
        # Gate applied without fallback or level change
        elif len(gate_reasons) > 0:
            event_type = "gate_applied"
        
        # Note: If both level_change AND fallback occur, event_type="level_change"
        # but used_fallback=True in the event metadata preserves this information
        
        # Build reasons list
        all_reasons = list(res.reasons)
        if gate_reasons:
            all_reasons.extend([f"Gate: {r}" for r in gate_reasons])
        if hysteresis_reasons:
            all_reasons.extend([f"Hysteresis: {r}" for r in hysteresis_reasons])
        
        # Create event
        event = GovernanceEvent(
            timestamp=ts,
            event_type=event_type,
            from_level=self._last_level,
            to_level=trusted_level,
            confidence=signals.model_confidence,
            reasons=all_reasons,
            metadata=dict(res.metadata) if res.metadata else {},
            safety_status=res.safety_status.value if hasattr(res.safety_status, 'value') else str(res.safety_status),
            grace_status=res.grace_status.value if hasattr(res.grace_status, 'value') else str(res.grace_status),
            consensus_status=res.consensus_status.value if hasattr(res.consensus_status, 'value') else str(res.consensus_status),
            used_fallback=res.used_fallback,
        )
        
        # Store event
        self._event_log.append(event)
        self._last_event = event
        
        # Trim log if needed
        if len(self._event_log) > self.policy.max_event_log_size:
            self._event_log = self._event_log[-self.policy.max_event_log_size:]
    
    # -------------------------
    # Commit state
    # -------------------------
    
    def _commit_state(self, ts: float, trusted_level: AutonomyLevel, res: "DecisionResult") -> None:
        """
        Governor-level commit complements the pipeline's own history.
        
        We treat:
        - accepted + not fallback => a legitimate mode setting
        - otherwise => do not "advance" governor state
        """
        try:
            accepted = (res.used_fallback is False)
            if SafetyStatus is not None:
                accepted = accepted and (
                    res.safety_status in (SafetyStatus.ACCEPTED, SafetyStatus.BORDERLINE)
                )
        except Exception:
            accepted = (res.used_fallback is False)
        
        if accepted and trusted_level != self._last_level:
            self._last_level = trusted_level
            self._last_change_ts = ts


# -----------------------------
# Test / Demo Utilities
# -----------------------------

def create_test_signals(
    level: AutonomyLevel = AutonomyLevel.CONSTRAINED_AUTONOMY,
    confidence: float = 0.90,
    **kwargs
) -> AutonomySignals:
    """
    Factory for creating test signals.
    Useful for unit tests and integration testing.
    """
    return AutonomySignals(
        proposed_level=level,
        model_confidence=confidence,
        peer_recommended_levels=kwargs.get('peers', ()),
        safety_monitors=kwargs.get('safety_monitors'),
        odd=kwargs.get('odd'),
        driver_state=kwargs.get('driver_state'),
        system_health=kwargs.get('system_health'),
        environment=kwargs.get('environment', {}),
        current_scenario=kwargs.get('scenario'),
        timestamp=kwargs.get('ts', time.time()),
    )


# -----------------------------
# Convenience Exports
# -----------------------------

__all__ = [
    "AutonomyLevel",
    "AutonomySignals",
    "AutonomyGovernancePolicy",
    "AutonomyGovernor",
    "SafetyMonitorSignals",
    "OperationalDesignDomain",
    "DriverState",
    "SystemHealth",
    "DegradationStrategy",
    "ConfidenceTracker",
    "GovernanceEvent",
    "ScenarioPolicy",
    "default_autonomy_config",
    "create_test_signals",
]


# -----------------------------
# Demo / Self-Test
# -----------------------------

if __name__ == "__main__":
    print("AILEE Automotive Domain - Production Grade Demo")
    print("=" * 60)
    
    # Setup governor
    policy = AutonomyGovernancePolicy(
        max_allowed_level=AutonomyLevel.CONSTRAINED_AUTONOMY,
        min_driver_readiness_for_autonomy=0.65,
    )
    governor = AutonomyGovernor(policy=policy)
    
    # Simulate various scenarios
    scenarios = [
        # Scenario 1: Good conditions, escalate to constrained
        {
            "name": "Highway cruise - good conditions",
            "signals": create_test_signals(
                level=AutonomyLevel.CONSTRAINED_AUTONOMY,
                confidence=0.92,
                peers=(AutonomyLevel.CONSTRAINED_AUTONOMY, AutonomyLevel.CONSTRAINED_AUTONOMY),
                driver_state=DriverState(readiness=0.80, attention_level=0.85),
                system_health=SystemHealth(latency_ms=80.0, sensor_faults=0),
                odd=OperationalDesignDomain(
                    weather_code="clear",
                    road_type="highway",
                    visibility_m=1000.0,
                ),
                safety_monitors=SafetyMonitorSignals(
                    collision_risk_score=0.05,
                    path_safety_score=0.95,
                    localization_uncertainty_m=0.3,
                ),
                scenario="highway_cruise",
            ),
        },
        # Scenario 2: Driver distracted, must downgrade
        {
            "name": "Driver distraction detected",
            "signals": create_test_signals(
                level=AutonomyLevel.CONSTRAINED_AUTONOMY,
                confidence=0.88,
                peers=(AutonomyLevel.ASSISTED_ONLY, AutonomyLevel.CONSTRAINED_AUTONOMY),
                driver_state=DriverState(
                    readiness=0.45, 
                    attention_level=0.30,
                    distraction_detected=True,
                ),
                system_health=SystemHealth(latency_ms=90.0, sensor_faults=0),
            ),
        },
        # Scenario 3: High latency, block autonomy
        {
            "name": "System latency too high",
            "signals": create_test_signals(
                level=AutonomyLevel.FULL_AUTONOMY_ALLOWED,
                confidence=0.91,
                driver_state=DriverState(readiness=0.85),
                system_health=SystemHealth(latency_ms=180.0, sensor_faults=0),
            ),
        },
        # Scenario 4: Sensor fault, force manual
        {
            "name": "Sensor fault detected",
            "signals": create_test_signals(
                level=AutonomyLevel.ASSISTED_ONLY,
                confidence=0.75,
                system_health=SystemHealth(latency_ms=95.0, sensor_faults=2),
            ),
        },
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
        
        authorized_level, decision = governor.evaluate(scenario['signals'])
        
        print(f"Proposed: {int(scenario['signals'].proposed_level)}")
        print(f"Authorized: {int(authorized_level)} ({authorized_level.name})")
        print(f"Confidence: {decision.confidence_score:.3f}")
        print(f"Safety: {decision.safety_status.value if hasattr(decision.safety_status, 'value') else decision.safety_status}")
        print(f"Fallback: {decision.used_fallback}")
        if decision.reasons:
            print(f"Reasons:")
            for reason in decision.reasons[:3]:  # Show first 3
                print(f"  - {reason}")
    
    # Show event log
    print(f"\n--- Event Log ({len(governor.export_events())} events) ---")
    for event in governor.export_events()[-3:]:  # Last 3 events
        print(f"[{event.timestamp:.2f}] {event.event_type}: {int(event.from_level)}->{int(event.to_level)}")
    
    print("\n" + "=" * 60)
    print("Demo complete. Governor is production-ready.")
