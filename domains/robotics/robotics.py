"""
AILEE Trust Layer — ROBOTICS Domain
Version: 1.0.0 - Production Grade

Robotics-focused governance domain for autonomous systems, manipulation,
navigation, and human-robot interaction.

This domain does NOT implement control algorithms, path planning, or kinematics.
It governs whether robotic actions and decisions are trustworthy based on:
- Safety metrics (collision risk, force limits, workspace boundaries)
- Performance metrics (accuracy, repeatability, efficiency)
- Multi-sensor consensus
- Uncertainty quantification
- Human safety considerations

Primary governed signals:
- Action safety scores
- Trajectory feasibility
- Grasp quality metrics
- Navigation confidence
- Human proximity awareness
- System health indicators

Key properties:
- Real-time safety validation
- Multi-sensor fusion validation
- Human-aware decision making
- Fail-safe fallback mechanisms
- Auditable action logging
- Adaptive behavior tuning
- Hardware constraint enforcement

INTEGRATION EXAMPLE:

    # Setup (once)
    policy = RoboticsGovernancePolicy(
        robot_type=RobotType.MANIPULATOR,
        max_tcp_velocity_m_s=0.5,
        min_obstacle_clearance_m=0.15,
        human_safety_zone_m=1.0,
    )
    governor = RoboticsGovernor(policy=policy)
    
    # Per-action evaluation
    while robot_active:
        signals = RoboticsSignals(
            action_safety_score=compute_collision_risk(planned_trajectory),
            model_confidence=policy_network.get_confidence(),
            sensor_readings=(camera_data, lidar_data, force_sensor),
            workspace_state=WorkspaceState(
                human_detected=True,
                human_distance_m=1.2,
                obstacles=detected_obstacles,
            ),
            performance_metrics=PerformanceMetrics(
                position_error_m=0.003,
                velocity_m_s=0.35,
                computation_time_ms=12.0,
            ),
        )
        
        decision = governor.evaluate(signals)
        
        if decision.action_safe:
            execute_action(planned_trajectory)
        elif decision.recommendation == "reduce_velocity":
            execute_action(planned_trajectory, velocity_scale=0.5)
        else:
            trigger_emergency_stop()
        
        # Log for safety compliance
        safety_system.record_robotics_event(governor.get_last_event())

This module is designed for industrial robotics, autonomous vehicles,
service robots, collaborative robots (cobots), and research platforms.

SAFETY SCORE CONVENTIONS:
========================
All safety metrics in this domain follow consistent semantics:

- action_safety_score: [0.0, 1.0] where 1.0 = fully safe, 0.0 = prohibited
- collision_risk_score: [0.0, 1.0] where 1.0 = collision imminent, 0.0 = clear
- Confidence scores: [0.0, 1.0] where 1.0 = certain, 0.0 = no confidence

CRITICAL: Integrators must ensure their planners/controllers output safety scores
in this convention. Inverted semantics (0=safe) will cause dangerous misinterpretation.

For collision risk specifically: we store risk (higher=worse) but evaluate against
max thresholds. This matches standard robotics practice (ISO 10218, ISO/TS 15066).

DESIGN NOTE: This governor implements binary safety decisions (execute/stop).
For systems requiring graceful degradation (e.g., reduced capability modes),
integrators should implement degradation logic in their control layer using
decision.recommendation and AdaptiveStrategy as guidance.

ARCHITECTURAL NOTE:
This module is deterministic, synchronous, and side-effect-free
except for internal event logging. It is safe to call from real-time
or near-real-time robotics loops provided evaluation latency budgets
are respected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import math
import statistics
import time


# ---- Core imports ----
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


# ---- Forward references for type checking ----
if TYPE_CHECKING:
    from typing import TYPE_CHECKING as _TYPE_CHECKING


# -----------------------------
# Robot Types and Categories
# -----------------------------

"""
ENUM DESIGN PHILOSOPHY:
======================
The enums below (RobotType, OperationalMode, ActionType) provide standard
categories for common robotics applications. They are intentionally descriptive
rather than exhaustive.

Integrators may:
- Use metadata fields for vendor-specific categorizations
- Map custom types to the nearest standard category
- Extend via string values in context dictionaries

We resist expanding these enums indefinitely to maintain API stability.
If your robot type isn't listed, use RobotType.UNKNOWN and populate
signals.context with detailed categorization.
"""

class RobotType(str, Enum):
    """Robot categories with different safety requirements"""
    MANIPULATOR = "MANIPULATOR"  # Industrial arms
    MOBILE_ROBOT = "MOBILE_ROBOT"  # AGVs, AMRs
    HUMANOID = "HUMANOID"
    QUADRUPED = "QUADRUPED"
    AERIAL = "AERIAL"  # Drones, UAVs
    UNDERWATER = "UNDERWATER"  # ROVs, AUVs
    COLLABORATIVE = "COLLABORATIVE"  # Cobots
    SURGICAL = "SURGICAL"  # Medical robots
    SERVICE = "SERVICE"  # Delivery, cleaning, etc.
    AUTONOMOUS_VEHICLE = "AUTONOMOUS_VEHICLE"
    EXOSKELETON = "EXOSKELETON"
    UNKNOWN = "UNKNOWN"


class OperationalMode(str, Enum):
    """Robot operational modes affecting safety requirements"""
    AUTONOMOUS = "AUTONOMOUS"
    SEMI_AUTONOMOUS = "SEMI_AUTONOMOUS"
    TELEOPERATED = "TELEOPERATED"
    COLLABORATIVE = "COLLABORATIVE"  # Working alongside humans
    TEACHING = "TEACHING"  # Programming by demonstration
    MAINTENANCE = "MAINTENANCE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class ActionType(str, Enum):
    """Types of robotic actions"""
    MOTION = "MOTION"  # General movement
    MANIPULATION = "MANIPULATION"  # Grasping, placing
    NAVIGATION = "NAVIGATION"  # Path following
    INTERACTION = "INTERACTION"  # Human or object interaction
    PERCEPTION = "PERCEPTION"  # Sensing action
    COMMUNICATION = "COMMUNICATION"  # HRI
    SYSTEM_CHANGE = "SYSTEM_CHANGE"  # Mode change, config update


# -----------------------------
# Safety Decision Status
# -----------------------------

class SafetyDecision(str, Enum):
    """Robotics safety assessment decision"""
    EXECUTE = "EXECUTE"
    EXECUTE_REDUCED_SPEED = "EXECUTE_REDUCED_SPEED"
    EXECUTE_WITH_MONITORING = "EXECUTE_WITH_MONITORING"
    DEFER_TO_HUMAN = "DEFER_TO_HUMAN"
    REJECT_UNSAFE = "REJECT_UNSAFE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    FALLBACK_BEHAVIOR = "FALLBACK_BEHAVIOR"


# -----------------------------
# Workspace State
# -----------------------------

@dataclass(frozen=True)
class WorkspaceState:
    """
    Current state of the robot's workspace.
    Critical for collision avoidance and human safety.
    """
    # Human presence
    human_detected: bool = False
    human_distance_m: Optional[float] = None  # Closest human distance
    human_velocity_m_s: Optional[float] = None
    human_in_safety_zone: bool = False
    
    # Obstacles
    obstacles: Tuple[Dict[str, Any], ...] = ()  # List of detected obstacles
    closest_obstacle_distance_m: Optional[float] = None
    dynamic_obstacles_present: bool = False
    
    # Workspace boundaries
    within_workspace: bool = True
    workspace_boundary_distance_m: Optional[float] = None
    
    # Environmental conditions
    lighting_adequate: bool = True
    floor_condition: Optional[str] = None  # "dry", "wet", "uneven"
    temperature_c: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_safe_for_operation(self, safety_zone_m: float) -> Tuple[bool, List[str]]:
        """Check if workspace state allows safe operation"""
        issues: List[str] = []
        
        if self.human_detected and self.human_in_safety_zone:
            issues.append(f"human_in_safety_zone (distance={self.human_distance_m:.2f}m)")
        
        if self.closest_obstacle_distance_m is not None:
            if self.closest_obstacle_distance_m < 0.05:  # 5cm critical
                issues.append(f"obstacle_too_close ({self.closest_obstacle_distance_m:.3f}m)")
        
        if not self.within_workspace:
            issues.append("outside_workspace_boundaries")
        
        if not self.lighting_adequate:
            issues.append("inadequate_lighting")
        
        return len(issues) == 0, issues


# -----------------------------
# Performance Metrics
# -----------------------------

@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Robot performance and accuracy metrics.
    Used for quality assessment and degradation detection.
    """
    # Accuracy metrics
    position_error_m: Optional[float] = None
    orientation_error_rad: Optional[float] = None
    trajectory_tracking_error_m: Optional[float] = None
    
    # Velocity and dynamics
    velocity_m_s: Optional[float] = None
    acceleration_m_s2: Optional[float] = None
    jerk_m_s3: Optional[float] = None  # Rate of acceleration change
    
    # Force/torque (for manipulation)
    applied_force_n: Optional[float] = None
    applied_torque_nm: Optional[float] = None
    contact_force_n: Optional[float] = None
    
    # Timing
    computation_time_ms: Optional[float] = None
    control_loop_frequency_hz: Optional[float] = None
    latency_ms: Optional[float] = None
    
    # Energy
    power_consumption_w: Optional[float] = None
    battery_level_percent: Optional[float] = None
    
    # Quality indicators
    repeatability_score: Optional[float] = None  # 0..1
    path_smoothness_score: Optional[float] = None  # 0..1
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check_performance_limits(
        self,
        max_position_error_m: float = 0.01,
        max_velocity_m_s: float = 1.0,
        max_force_n: float = 100.0,
    ) -> Tuple[bool, List[str]]:
        """Check if performance is within acceptable limits"""
        issues: List[str] = []
        
        if self.position_error_m is not None:
            if self.position_error_m > max_position_error_m:
                issues.append(f"position_error={self.position_error_m:.4f}m exceeds {max_position_error_m}m")
        
        if self.velocity_m_s is not None:
            if self.velocity_m_s > max_velocity_m_s:
                issues.append(f"velocity={self.velocity_m_s:.2f}m/s exceeds {max_velocity_m_s}m/s")
        
        if self.applied_force_n is not None:
            if self.applied_force_n > max_force_n:
                issues.append(f"force={self.applied_force_n:.1f}N exceeds {max_force_n}N")
        
        if self.battery_level_percent is not None:
            if self.battery_level_percent < 15.0:
                issues.append(f"low_battery={self.battery_level_percent:.1f}%")
        
        return len(issues) == 0, issues


# -----------------------------
# Sensor Data
# -----------------------------

@dataclass(frozen=True)
class SensorReading:
    """Individual sensor reading with confidence
    
    INTEGRATION NOTE: Implementations should validate sensor timestamp freshness
    before evaluation. Stale sensor data (>100ms for dynamic environments) may
    compromise safety assessments. This module does not enforce staleness checks
    to avoid coupling to specific timing architectures.
    """
    sensor_type: str  # "camera", "lidar", "force", "imu", etc.
    value: float  # Primary measured value
    confidence: Optional[float] = None  # 0..1
    quality_score: Optional[float] = None  # 0..1
    timestamp: Optional[float] = None  # Unix timestamp; freshness checked by integrator
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Uncertainty Quantification
# -----------------------------

@dataclass(frozen=True)
class UncertaintyEstimate:
    """
    Quantify uncertainty in robot's perception, planning, and control.
    Critical for safe decision-making under uncertainty.
    """
    # Perception uncertainty
    perception_confidence: Optional[float] = None  # 0..1
    localization_uncertainty_m: Optional[float] = None
    object_detection_confidence: Optional[float] = None
    
    # Planning uncertainty
    trajectory_confidence: Optional[float] = None
    collision_risk_score: Optional[float] = None  # 0..1, higher = more risk
    path_feasibility_score: Optional[float] = None  # 0..1
    
    # Control uncertainty
    control_accuracy_confidence: Optional[float] = None
    model_prediction_error: Optional[float] = None
    
    # Epistemic vs aleatoric
    epistemic_uncertainty: Optional[float] = None  # Model uncertainty
    aleatoric_uncertainty: Optional[float] = None  # Data/sensor noise
    
    def is_uncertainty_acceptable(self, max_collision_risk: float = 0.10) -> Tuple[bool, str]:
        """Check if uncertainty is within acceptable bounds"""
        if self.collision_risk_score is not None:
            if self.collision_risk_score > max_collision_risk:
                return False, f"collision_risk={self.collision_risk_score:.3f} exceeds {max_collision_risk:.3f}"
        
        if self.perception_confidence is not None:
            if self.perception_confidence < 0.60:
                return False, f"perception_confidence={self.perception_confidence:.2f} too low"
        
        return True, "uncertainty_acceptable"
    
    def get_dominant_uncertainty_source(self) -> Optional[str]:
        """Identify primary uncertainty contributor for diagnostics
        
        Returns:
            "perception" | "planning" | "control" | "localization" | None
            
        Useful for adaptive strategies and debugging poor safety scores.
        """
        sources = []
        
        if self.perception_confidence is not None and self.perception_confidence < 0.7:
            sources.append(("perception", 1.0 - self.perception_confidence))
        
        if self.collision_risk_score is not None and self.collision_risk_score > 0.1:
            sources.append(("planning", self.collision_risk_score))
        
        if self.localization_uncertainty_m is not None and self.localization_uncertainty_m > 0.05:
            sources.append(("localization", self.localization_uncertainty_m))
        
        if self.control_accuracy_confidence is not None and self.control_accuracy_confidence < 0.7:
            sources.append(("control", 1.0 - self.control_accuracy_confidence))
        
        if not sources:
            return None
        
        return max(sources, key=lambda x: x[1])[0]


# -----------------------------
# Multi-Sensor Validation Results
# -----------------------------

@dataclass(frozen=True)
class ValidationResult:
    """Single validation result from a sensor or model"""
    source: str  # "lidar", "camera_rgb", "force_sensor", "backup_planner"
    safety_score: float  # Primary safety metric (higher = safer)
    confidence: Optional[float] = None
    computation_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Domain Inputs
# -----------------------------

@dataclass(frozen=True)
class RoboticsSignals:
    """
    Governance signals for robotics safety assessment.
    Primary input structure for the robotics governor.
    """
    # Primary safety metric (higher = safer, e.g., clearance, grasp quality)
    action_safety_score: float
    model_confidence: Optional[float] = None  # 0..1 from policy/planner
    
    # Multi-sensor/multi-model validation
    sensor_readings: Tuple[SensorReading, ...] = ()
    validation_results: Tuple[ValidationResult, ...] = ()
    
    # Context
    workspace_state: Optional[WorkspaceState] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    uncertainty_estimate: Optional[UncertaintyEstimate] = None
    
    # Robot state
    robot_type: RobotType = RobotType.UNKNOWN
    operational_mode: OperationalMode = OperationalMode.AUTONOMOUS
    action_type: ActionType = ActionType.MOTION
    action_id: Optional[str] = None
    
    # Metadata
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Automatically set timestamp if not provided"""
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())


# -----------------------------
# Adaptive Behavior Strategy
# -----------------------------

@dataclass
class AdaptiveStrategy:
    """
    Recommendations for adaptive behavior adjustment.
    The governor produces these; the robot controller decides execution.
    """
    action: str  # "continue" | "reduce_speed" | "increase_caution" | "stop" | "defer"
    
    # Parameter adjustments (multiplicative factors)
    velocity_scale: float = 1.0  # <1 = slower, >1 = faster
    force_limit_scale: float = 1.0
    monitoring_frequency_scale: float = 1.0  # >1 = more frequent checks
    
    # Behavioral changes
    require_human_confirmation: bool = False
    enable_emergency_stop: bool = False
    switch_to_fallback: bool = False
    
    # Suggested absolute limits
    suggested_max_velocity_m_s: Optional[float] = None
    suggested_safety_zone_m: Optional[float] = None
    
    reason: str = ""
    
    def should_proceed(self) -> bool:
        """Check if action should proceed"""
        return not self.enable_emergency_stop and self.action not in ("stop", "defer")


# -----------------------------
# Robotics Events (Safety Logging)
# -----------------------------

@dataclass(frozen=True)
class RoboticsEvent:
    """
    Structured event for safety and compliance logging.
    Critical for incident investigation and certification.
    """
    timestamp: float
    event_type: str  # "action_executed" | "action_rejected" | "emergency_stop" | "human_intervention"
    robot_type: RobotType
    operational_mode: OperationalMode
    action_type: ActionType
    
    safety_score: float
    decision: SafetyDecision
    reasons: List[str]
    
    # Context
    workspace_state: Optional[WorkspaceState] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    uncertainty_estimate: Optional[UncertaintyEstimate] = None
    
    # Pipeline results
    ailee_decision: Optional[DecisionResult] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Domain Configuration
# -----------------------------

@dataclass(frozen=True)
class RoboticsGovernancePolicy:
    """
    Domain policy for robotics safety governance.
    Defines safety thresholds, operational limits, and risk tolerance.
    """
    # Robot configuration
    robot_type: RobotType = RobotType.UNKNOWN
    operational_mode: OperationalMode = OperationalMode.AUTONOMOUS
    
    # Safety thresholds
    min_safety_score: float = 0.70  # Minimum acceptable safety score
    max_collision_risk: float = 0.10  # Maximum acceptable collision risk
    min_obstacle_clearance_m: float = 0.10
    human_safety_zone_m: float = 1.5  # Slow down if human within this distance
    human_critical_zone_m: float = 0.5  # Stop if human within this distance
    
    # Performance limits
    max_tcp_velocity_m_s: Optional[float] = None  # Tool center point
    max_joint_velocity_rad_s: Optional[float] = None
    max_acceleration_m_s2: Optional[float] = None
    max_force_n: Optional[float] = None
    max_torque_nm: Optional[float] = None
    
    # Quality requirements
    max_position_error_m: float = 0.01
    min_repeatability_score: float = 0.90
    min_control_frequency_hz: float = 100.0
    
    # Uncertainty tolerance
    max_localization_uncertainty_m: float = 0.05
    min_perception_confidence: float = 0.70
    
    # Multi-sensor validation
    require_sensor_consensus: bool = True
    min_sensor_agreement_ratio: float = 0.75
    
    # Adaptive behavior
    enable_adaptive_behavior: bool = True
    velocity_reduction_threshold: float = 0.80  # Reduce speed if safety score below this
    
    # Event logging
    max_event_log_size: int = 10000  # Large by default for regulatory traceability
                                      # (ISO 10218-1 §5.10.6, ANSI/RIA R15.06).
                                      # Production systems should offload to external
                                      # logging infrastructure for long-term retention.
    
    # Fail-safe behavior
    emergency_stop_on_human_detection: bool = False  # For non-collaborative robots
    require_human_confirmation_threshold: float = 0.60  # Defer to human if below


def default_robotics_config(robot_type: RobotType) -> "AileeConfig":
    """
    Safe defaults for robotics governance pipeline configuration.
    Tuned per robot type for appropriate safety sensitivity.
    """
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    # Base configuration with conservative safety settings
    cfg = AileeConfig(
        accept_threshold=0.85,
        borderline_low=0.70,
        borderline_high=0.85,
        
        # Weights for robotics: prioritize stability and agreement heavily
        w_stability=0.45,
        w_agreement=0.40,
        w_likelihood=0.15,
        
        history_window=100,
        forecast_window=15,
        
        grace_peer_delta=0.10,  # Safety score tolerance
        grace_min_peer_agreement_ratio=0.70,
        grace_forecast_epsilon=0.15,
        grace_max_abs_z=2.0,  # Stricter for safety
        
        consensus_quorum=2,
        consensus_delta=0.15,
        consensus_pass_ratio=0.75,
        
        fallback_mode="last_good",
        
        enable_grace=True,
        enable_consensus=True,
        enable_audit_metadata=True,
    )
    
    # Robot-type-specific adjustments
    if robot_type == RobotType.COLLABORATIVE:
        # Cobots: most stringent safety requirements
        cfg.accept_threshold = 0.90
        cfg.grace_peer_delta = 0.08
        cfg.consensus_delta = 0.12
        cfg.grace_max_abs_z = 1.5
    
    elif robot_type == RobotType.SURGICAL:
        # Surgical: highest precision and safety
        cfg.accept_threshold = 0.95
        cfg.grace_peer_delta = 0.05
        cfg.w_stability = 0.50
    
    elif robot_type in (RobotType.MANIPULATOR, RobotType.MOBILE_ROBOT):
        # Industrial: balance safety and productivity
        cfg.accept_threshold = 0.82
        cfg.grace_peer_delta = 0.12
    
    elif robot_type == RobotType.AUTONOMOUS_VEHICLE:
        # Autonomous vehicles: emphasis on multi-sensor agreement
        cfg.w_agreement = 0.50
        cfg.w_stability = 0.35
        cfg.consensus_quorum = 3
    
    elif robot_type in (RobotType.SERVICE, RobotType.AERIAL):
        # Service robots and drones: moderate requirements
        cfg.accept_threshold = 0.80
        cfg.grace_peer_delta = 0.15
    
    return cfg


# -----------------------------
# Result Structure
# -----------------------------

@dataclass(frozen=True)
class RoboticsDecisionResult:
    """
    Robotics governance decision result.
    Extends AILEE decision with robotics-specific context.
    
    FRAMEWORK COHERENCE NOTE:
    - action_safe (robotics) ≈ quality_acceptable (imaging) ≈ prediction_trustworthy (NLP)
    - validated_safety_score ≈ validated_metric (cross-domain)
    - SafetyDecision ≈ domain-specific decision enumeration
    
    Each domain uses terminology natural to its practitioners while maintaining
    structural alignment for multi-domain governance pipelines.
    """
    action_safe: bool
    decision: SafetyDecision
    validated_safety_score: float
    confidence_score: float
    recommendation: str
    reasons: List[str]
    ailee_result: Optional[DecisionResult] = None
    risk_level: Optional[str] = None  # "low" | "medium" | "high" | "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Robotics Governor
# -----------------------------

class RoboticsGovernor:
    """
    Production-grade governance controller for robotic systems.
    
    Validates robotic actions by integrating:
    - AILEE trust pipeline (core validation)
    - Multi-sensor consensus
    - Workspace state monitoring
    - Human safety enforcement
    - Performance limit checking
    - Uncertainty quantification
    - Adaptive behavior management
    - Fail-safe mechanisms
    
    Integration contract:
    - Call evaluate(signals) before executing each action
    - Use decision.action_safe to proceed or abort
    - Monitor decision.recommendation for behavior adaptation
    - Export events for safety audits and certification
    """
    
    def __init__(
        self,
        cfg: Optional["AileeConfig"] = None,
        policy: Optional[RoboticsGovernancePolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable. Ensure ailee_trust_pipeline_v1.py is importable.")
        
        self.policy = policy or RoboticsGovernancePolicy()
        self.cfg = cfg or default_robotics_config(self.policy.robot_type)
        
        # Apply policy constraints to config
        self.cfg.hard_min = 0.0
        self.cfg.hard_max = 1.0
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        
        # Tracking state
        self._last_safety_score: Optional[float] = None
        self._last_decision: Optional[SafetyDecision] = None
        self._emergency_stop_triggered: bool = False
        
        # Event logging
        self._event_log: List[RoboticsEvent] = []
        self._last_event: Optional[RoboticsEvent] = None
        
        # Performance tracking
        self._performance_history: List[Tuple[float, float]] = []  # [(timestamp, safety_score)]
        
        # Human interaction tracking
        self._human_proximity_history: List[Tuple[float, float]] = []  # [(timestamp, distance)]
    
    # -------------------------
    # Public API
    # -------------------------
    
    def evaluate(self, signals: RoboticsSignals) -> "RoboticsDecisionResult":
        """
        Evaluate action safety and produce governance decision.
        
        Args:
            signals: Current robotics signals (safety scores, sensors, workspace)
        
        Returns:
            RoboticsDecisionResult with safety assessment and recommendations
        """
        ts = float(signals.timestamp)
        reasons: List[str] = []
        
        # 1) Emergency checks (immediate abort conditions)
        if self._emergency_stop_triggered:
            reasons.append("Emergency stop active - manual reset required")
            decision = self._create_emergency_stop_decision(signals, reasons, ts)
            self._log_event(ts, signals, decision, reasons)
            return decision
        
        # 2) Workspace safety check (hard constraints)
        if signals.workspace_state is not None:
            workspace_safe, workspace_issues = signals.workspace_state.is_safe_for_operation(
                self.policy.human_safety_zone_m
            )
            
            # Critical: Human in critical zone
            if signals.workspace_state.human_detected:
                if signals.workspace_state.human_distance_m is not None:
                    self._human_proximity_history.append((ts, signals.workspace_state.human_distance_m))
                    
                    if signals.workspace_state.human_distance_m < self.policy.human_critical_zone_m:
                        reasons.append(
                            f"Human at {signals.workspace_state.human_distance_m:.2f}m "
                            f"(critical zone: {self.policy.human_critical_zone_m}m)"
                        )
                        if self.policy.emergency_stop_on_human_detection:
                            self._emergency_stop_triggered = True
                            decision = self._create_emergency_stop_decision(signals, reasons, ts)
                            self._log_event(ts, signals, decision, reasons)
                            return decision
            
            if not workspace_safe:
                reasons.extend([f"Workspace: {issue}" for issue in workspace_issues])
        
        # 3) Performance limits check
        if signals.performance_metrics is not None:
            perf_ok, perf_issues = signals.performance_metrics.check_performance_limits(
                max_position_error_m=self.policy.max_position_error_m,
                max_velocity_m_s=self.policy.max_tcp_velocity_m_s or float('inf'),
                max_force_n=self.policy.max_force_n or float('inf'),
            )
            if not perf_ok:
                reasons.extend([f"Performance: {issue}" for issue in perf_issues])
        
        # 4) Uncertainty check
        if signals.uncertainty_estimate is not None:
            unc_ok, unc_reason = signals.uncertainty_estimate.is_uncertainty_acceptable(
                self.policy.max_collision_risk
            )
            if not unc_ok:
                reasons.append(f"Uncertainty: {unc_reason}")
        
        # 5) Multi-sensor consensus check
        if self.policy.require_sensor_consensus and len(signals.validation_results) >= 2:
            consensus_ok, consensus_reason = self._check_sensor_consensus(signals)
            if not consensus_ok:
                reasons.append(f"Consensus: {consensus_reason}")
        
        # 6) Extract peer safety scores
        peer_values = [v.safety_score for v in signals.validation_results]
        
        # 7) Build context for AILEE pipeline
        ctx = self._build_context(signals, reasons)
        
        # 8) Use AILEE pipeline to validate safety score
        ailee_result = self.pipeline.process(
            raw_value=float(signals.action_safety_score),
            raw_confidence=float(signals.model_confidence) if signals.model_confidence is not None else None,
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        # 9) Make robotics-specific decision
        decision = self._make_robotics_decision(signals, ailee_result, reasons, ts)
        
        # 10) Log event
        self._log_event(ts, signals, decision, reasons)
        
        # 11) Update state
        self._commit_state(ts, signals, decision, ailee_result)
        
        return decision
    
    def reset_emergency_stop(self) -> bool:
        """
        Reset emergency stop state.
        Should only be called after ensuring workspace is safe.
        
        Returns:
            True if reset successful
        """
        if self._emergency_stop_triggered:
            self._emergency_stop_triggered = False
            return True
        return False
    
    def get_last_event(self) -> Optional[RoboticsEvent]:
        """Get most recent robotics event"""
        return self._last_event
    
    def export_events(self, since_ts: Optional[float] = None) -> List[RoboticsEvent]:
        """Export robotics events for safety audits"""
        if since_ts is None:
            return self._event_log[:]
        return [e for e in self._event_log if e.timestamp >= since_ts]
    
    def get_safety_trend(self) -> str:
        """Get safety performance trend: improving, stable, degrading"""
        if len(self._performance_history) < 20:
            return "insufficient_data"
        
        recent = [score for _, score in self._performance_history[-10:]]
        older = [score for _, score in self._performance_history[-20:-10]]
        
        recent_avg = statistics.fmean(recent)
        older_avg = statistics.fmean(older)
        
        diff = recent_avg - older_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        return "stable"
    
    def get_human_interaction_stats(self) -> Dict[str, Any]:
        """Get statistics on human proximity over time"""
        if not self._human_proximity_history:
            return {"status": "no_human_interactions"}
        
        recent = [d for _, d in self._human_proximity_history[-50:]]
        
        return {
            "min_distance_m": min(recent),
            "avg_distance_m": statistics.fmean(recent),
            "max_distance_m": max(recent),
            "interaction_count": len(self._human_proximity_history),
            "critical_zone_entries": sum(1 for d in recent if d < self.policy.human_critical_zone_m),
        }
    
    def suggest_adaptive_strategy(
        self,
        signals: RoboticsSignals,
        decision: "RoboticsDecisionResult"
    ) -> AdaptiveStrategy:
        """
        Suggest adaptive behavior strategy based on current safety assessment.
        This is advisory only; the robot controller decides whether to execute.
        """
        if not self.policy.enable_adaptive_behavior:
            return AdaptiveStrategy(action="continue", reason="adaptive_behavior_disabled")
        
        # Emergency conditions
        if decision.decision == SafetyDecision.EMERGENCY_STOP:
            return AdaptiveStrategy(
                action="stop",
                enable_emergency_stop=True,
                velocity_scale=0.0,
                reason="emergency_stop_required"
            )
        
        # Human proximity adaptation
        if signals.workspace_state is not None and signals.workspace_state.human_detected:
            if signals.workspace_state.human_distance_m is not None:
                dist = signals.workspace_state.human_distance_m
                
                # Critical zone: stop
                if dist < self.policy.human_critical_zone_m:
                    return AdaptiveStrategy(
                        action="stop",
                        enable_emergency_stop=True,
                        velocity_scale=0.0,
                        reason=f"human_in_critical_zone ({dist:.2f}m)"
                    )
                
                # Safety zone: reduce speed
                if dist < self.policy.human_safety_zone_m:
                    # Scale velocity inversely with distance
                    scale = dist / self.policy.human_safety_zone_m
                    scale = max(0.3, min(0.7, scale))  # Clamp to 30-70%
                    
                    return AdaptiveStrategy(
                        action="reduce_speed",
                        velocity_scale=scale,
                        monitoring_frequency_scale=2.0,  # Double monitoring frequency
                        suggested_max_velocity_m_s=self.policy.max_tcp_velocity_m_s * scale if self.policy.max_tcp_velocity_m_s else None,
                        reason=f"human_proximity ({dist:.2f}m, safety_zone={self.policy.human_safety_zone_m}m)"
                    )
        
        # Safety score based adaptation
        if decision.validated_safety_score < self.policy.require_human_confirmation_threshold:
            return AdaptiveStrategy(
                action="defer",
                require_human_confirmation=True,
                reason=f"safety_score={decision.validated_safety_score:.2f} below confirmation threshold"
            )
        
        if decision.validated_safety_score < self.policy.velocity_reduction_threshold:
            # Reduce velocity proportionally
            ratio = decision.validated_safety_score / self.policy.velocity_reduction_threshold
            velocity_scale = max(0.5, ratio)
            
            return AdaptiveStrategy(
                action="reduce_speed",
                velocity_scale=velocity_scale,
                monitoring_frequency_scale=1.5,
                reason=f"safety_score={decision.validated_safety_score:.2f} below threshold"
            )
        
        # High uncertainty adaptation
        if signals.uncertainty_estimate is not None:
            if signals.uncertainty_estimate.collision_risk_score is not None:
                if signals.uncertainty_estimate.collision_risk_score > self.policy.max_collision_risk * 0.8:
                    return AdaptiveStrategy(
                        action="increase_caution",
                        velocity_scale=0.7,
                        force_limit_scale=0.8,
                        monitoring_frequency_scale=2.0,
                        reason=f"high_collision_risk={signals.uncertainty_estimate.collision_risk_score:.2f}"
                    )
        
        # Performance degradation adaptation
        if signals.performance_metrics is not None:
            if signals.performance_metrics.position_error_m is not None:
                if signals.performance_metrics.position_error_m > self.policy.max_position_error_m * 0.8:
                    return AdaptiveStrategy(
                        action="increase_caution",
                        velocity_scale=0.8,
                        reason=f"position_error={signals.performance_metrics.position_error_m:.4f}m approaching limit"
                    )
        
        # All clear: continue normal operation
        if decision.action_safe:
            return AdaptiveStrategy(action="continue", reason="all_safety_checks_passed")
        
        # Fallback: increase caution
        return AdaptiveStrategy(
            action="increase_caution",
            velocity_scale=0.7,
            monitoring_frequency_scale=1.5,
            reason="safety_concern_detected"
        )
    
    # -------------------------
    # Internal Helpers
    # -------------------------
    
    def _check_sensor_consensus(self, signals: RoboticsSignals) -> Tuple[bool, str]:
        """Validate multi-sensor agreement if policy requires it"""
        if not self.policy.require_sensor_consensus:
            return True, "consensus_not_required"
        
        if len(signals.validation_results) < 2:
            return True, "insufficient_peers_for_consensus"
        
        scores = [v.safety_score for v in signals.validation_results]
        
        # Check agreement using policy threshold
        score_range = max(scores) - min(scores)
        avg_score = statistics.fmean(scores)
        
        if score_range > 0.20:  # 20% disagreement threshold
            return False, f"sensor_disagreement (range={score_range:.3f})"
        
        # Check proportion in agreement
        in_agreement = sum(1 for s in scores if abs(s - avg_score) < 0.10)
        agreement_ratio = in_agreement / len(scores)
        
        if agreement_ratio < self.policy.min_sensor_agreement_ratio:
            return False, f"insufficient_agreement (ratio={agreement_ratio:.2f})"
        
        return True, "sensor_consensus_achieved"
    
    def _build_context(
        self,
        signals: RoboticsSignals,
        reasons: List[str]
    ) -> Dict[str, Any]:
        """Build context dictionary for AILEE pipeline"""
        ctx = {
            "robot_type": signals.robot_type.value,
            "operational_mode": signals.operational_mode.value,
            "action_type": signals.action_type.value,
            "reasons": reasons[:],
        }
        
        if signals.action_id:
            ctx["action_id"] = signals.action_id
        
        if signals.workspace_state:
            ctx["human_detected"] = signals.workspace_state.human_detected
            if signals.workspace_state.human_distance_m is not None:
                ctx["human_distance_m"] = signals.workspace_state.human_distance_m
        
        if signals.uncertainty_estimate:
            if signals.uncertainty_estimate.collision_risk_score is not None:
                ctx["collision_risk"] = signals.uncertainty_estimate.collision_risk_score
        
        ctx.update(signals.context)
        return ctx
    
    def _make_robotics_decision(
        self,
        signals: RoboticsSignals,
        ailee_result: DecisionResult,
        reasons: List[str],
        ts: float
    ) -> RoboticsDecisionResult:
        """Convert AILEE result to robotics decision"""
        
        # Determine safety decision based on AILEE status and reasons
        if ailee_result.status == SafetyStatus.SAFE:
            if not reasons:
                safety_decision = SafetyDecision.EXECUTE
                action_safe = True
                recommendation = "execute_action"
            else:
                # Warnings present but AILEE approved
                safety_decision = SafetyDecision.EXECUTE_WITH_MONITORING
                action_safe = True
                recommendation = "execute_with_increased_monitoring"
        
        elif ailee_result.status == SafetyStatus.BORDERLINE:
            if signals.workspace_state and signals.workspace_state.human_detected:
                safety_decision = SafetyDecision.EXECUTE_REDUCED_SPEED
                action_safe = True
                recommendation = "reduce_velocity"
            else:
                safety_decision = SafetyDecision.DEFER_TO_HUMAN
                action_safe = False
                recommendation = "require_human_confirmation"
        
        else:  # UNSAFE or FALLBACK
            if any("human" in r.lower() for r in reasons):
                safety_decision = SafetyDecision.EMERGENCY_STOP
                action_safe = False
                recommendation = "emergency_stop"
            else:
                safety_decision = SafetyDecision.REJECT_UNSAFE
                action_safe = False
                recommendation = "reject_action"
        
        # Determine risk level
        score = ailee_result.validated_value
        if score >= 0.90:
            risk_level = "low"
        elif score >= 0.75:
            risk_level = "medium"
        elif score >= 0.60:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        return RoboticsDecisionResult(
            action_safe=action_safe,
            decision=safety_decision,
            validated_safety_score=score,
            confidence_score=ailee_result.confidence_score,
            recommendation=recommendation,
            reasons=reasons[:],
            ailee_result=ailee_result,
            risk_level=risk_level,
            metadata={
                "timestamp": ts,
                "robot_type": signals.robot_type.value,
                "action_type": signals.action_type.value,
            }
        )
    
    def _create_emergency_stop_decision(
        self,
        signals: RoboticsSignals,
        reasons: List[str],
        ts: float
    ) -> RoboticsDecisionResult:
        """Create emergency stop decision"""
        return RoboticsDecisionResult(
            action_safe=False,
            decision=SafetyDecision.EMERGENCY_STOP,
            validated_safety_score=0.0,
            confidence_score=1.0,
            recommendation="emergency_stop",
            reasons=reasons[:],
            ailee_result=None,
            risk_level="critical",
            metadata={
                "timestamp": ts,
                "robot_type": signals.robot_type.value,
                "emergency_stop_triggered": True,
            }
        )
    
    def _log_event(
        self,
        ts: float,
        signals: RoboticsSignals,
        decision: RoboticsDecisionResult,
        reasons: List[str]
    ):
        """Log robotics event"""
        event = RoboticsEvent(
            timestamp=ts,
            event_type="action_executed" if decision.action_safe else "action_rejected",
            robot_type=signals.robot_type,
            operational_mode=signals.operational_mode,
            action_type=signals.action_type,
            safety_score=decision.validated_safety_score,
            decision=decision.decision,
            reasons=reasons[:],
            workspace_state=signals.workspace_state,
            performance_metrics=signals.performance_metrics,
            uncertainty_estimate=signals.uncertainty_estimate,
            ailee_decision=decision.ailee_result,
            metadata=decision.metadata,
        )
        
        self._event_log.append(event)
        
        # Trim log if needed
        if len(self._event_log) > self.policy.max_event_log_size:
            self._event_log = self._event_log[-self.policy.max_event_log_size:]
        
        self._last_event = event
    
    def _commit_state(
        self,
        ts: float,
        signals: RoboticsSignals,
        decision: RoboticsDecisionResult,
        ailee_result: DecisionResult
    ):
        """Update internal state tracking"""
        self._last_safety_score = decision.validated_safety_score
        self._last_decision = decision.decision
        
        self._performance_history.append((ts, decision.validated_safety_score))
        
        # Trim history
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]
        
        if len(self._human_proximity_history) > 500:
            self._human_proximity_history = self._human_proximity_history[-500:]


# -----------------------------
# Convenience Functions
# -----------------------------

def create_robotics_governor(
    robot_type: RobotType = RobotType.UNKNOWN,
    collaborative: bool = False,
    **policy_overrides
) -> RoboticsGovernor:
    """
    Convenience factory for creating a robotics governor with common configurations.
    
    Args:
        robot_type: Type of robot
        collaborative: If True, use collaborative robot safety settings
        **policy_overrides: Additional policy parameters to override
    
    Returns:
        Configured RoboticsGovernor instance
    
    Example:
        governor = create_robotics_governor(
            robot_type=RobotType.MANIPULATOR,
            collaborative=True,
            max_tcp_velocity_m_s=0.5,
            human_safety_zone_m=1.2
        )
    """
    policy_kwargs = {"robot_type": robot_type}
    
    if collaborative:
        policy_kwargs.update({
            "human_safety_zone_m": 1.5,
            "human_critical_zone_m": 0.5,
            "min_safety_score": 0.85,
            "require_sensor_consensus": True,
            "emergency_stop_on_human_detection": False,
        })
    
    policy_kwargs.update(policy_overrides)
    policy = RoboticsGovernancePolicy(**policy_kwargs)
    
    cfg = default_robotics_config(robot_type)
    
    return RoboticsGovernor(cfg=cfg, policy=policy)


def validate_robotics_signals(signals: RoboticsSignals) -> Tuple[bool, List[str]]:
    """
    Pre-flight validation of robotics signals structure.
    Checks for common integration errors before evaluation.
    
    Args:
        signals: RoboticsSignals to validate
    
    Returns:
        (is_valid, list_of_issues)
    
    Example:
        valid, issues = validate_robotics_signals(signals)
        if not valid:
            logger.error(f"Invalid signals: {issues}")
            return
        
        decision = governor.evaluate(signals)
    """
    issues: List[str] = []
    
    # Check safety score range
    if not (0.0 <= signals.action_safety_score <= 1.0):
        issues.append(f"action_safety_score={signals.action_safety_score} outside [0.0, 1.0]")
    
    # Check confidence if provided
    if signals.model_confidence is not None:
        if not (0.0 <= signals.model_confidence <= 1.0):
            issues.append(f"model_confidence={signals.model_confidence} outside [0.0, 1.0]")
    
    # Check validation results
    for i, vr in enumerate(signals.validation_results):
        if not (0.0 <= vr.safety_score <= 1.0):
            issues.append(f"validation_results[{i}].safety_score={vr.safety_score} outside [0.0, 1.0]")
        if vr.confidence is not None and not (0.0 <= vr.confidence <= 1.0):
            issues.append(f"validation_results[{i}].confidence={vr.confidence} outside [0.0, 1.0]")
    
    # Check sensor readings
    for i, sr in enumerate(signals.sensor_readings):
        if sr.confidence is not None and not (0.0 <= sr.confidence <= 1.0):
            issues.append(f"sensor_readings[{i}].confidence={sr.confidence} outside [0.0, 1.0]")
        if sr.quality_score is not None and not (0.0 <= sr.quality_score <= 1.0):
            issues.append(f"sensor_readings[{i}].quality_score={sr.quality_score} outside [0.0, 1.0]")
    
    # Check uncertainty estimate
    if signals.uncertainty_estimate is not None:
        ue = signals.uncertainty_estimate
        if ue.collision_risk_score is not None and not (0.0 <= ue.collision_risk_score <= 1.0):
            issues.append(f"uncertainty_estimate.collision_risk_score={ue.collision_risk_score} outside [0.0, 1.0]")
        if ue.perception_confidence is not None and not (0.0 <= ue.perception_confidence <= 1.0):
            issues.append(f"uncertainty_estimate.perception_confidence={ue.perception_confidence} outside [0.0, 1.0]")
    
    # Check workspace state consistency
    if signals.workspace_state is not None:
        ws = signals.workspace_state
        if ws.human_detected and ws.human_distance_m is None:
            issues.append("workspace_state: human_detected=True but human_distance_m=None")
        if ws.human_distance_m is not None and ws.human_distance_m < 0:
            issues.append(f"workspace_state.human_distance_m={ws.human_distance_m} is negative")
    
    # Check performance metrics
    if signals.performance_metrics is not None:
        pm = signals.performance_metrics
        if pm.velocity_m_s is not None and pm.velocity_m_s < 0:
            issues.append(f"performance_metrics.velocity_m_s={pm.velocity_m_s} is negative")
        if pm.battery_level_percent is not None and not (0.0 <= pm.battery_level_percent <= 100.0):
            issues.append(f"performance_metrics.battery_level_percent={pm.battery_level_percent} outside [0, 100]")
    
    return len(issues) == 0, issues


# -----------------------------
# Module Exports
# -----------------------------

__all__ = [
    # Enums
    "RobotType",
    "OperationalMode",
    "ActionType",
    "SafetyDecision",
    
    # Data structures
    "WorkspaceState",
    "PerformanceMetrics",
    "SensorReading",
    "UncertaintyEstimate",
    "ValidationResult",
    "RoboticsSignals",
    "AdaptiveStrategy",
    "RoboticsEvent",
    
    # Configuration
    "RoboticsGovernancePolicy",
    "RoboticsDecisionResult",
    
    # Main governor
    "RoboticsGovernor",
    
    # Utilities
    "default_robotics_config",
    "create_robotics_governor",
    "validate_robotics_signals",
]
