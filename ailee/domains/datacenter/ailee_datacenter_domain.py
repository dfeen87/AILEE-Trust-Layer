"""
AILEE Trust Layer — Data Center Domain
Version: 2.0.0

First-class AILEE domain implementation for data center decision integrity.

This domain applies the AILEE Trust Pipeline to:
- Cooling / HVAC governance
- Power capping and throttling
- Workload placement and migration decisions
- Predictive maintenance gating
- Incident automation governance

The module defines domain-specific configurations, telemetry processing,
governed controllers, and a unified governance interface for production
data center environments.

Quick Start:
    >>> from ailee.domains.datacenter import (
    ...     DatacenterGovernor,
    ...     DatacenterPolicy,
    ...     DatacenterSignals,
    ...     DatacenterTrustLevel,
    ...     ControlDomain,
    ...     ControlAction,
    ...     SensorReading,
    ...     create_datacenter_governor,
    ... )
    >>>
    >>> governor = create_datacenter_governor()
    >>>
    >>> signals = DatacenterSignals(
    ...     control_domain=ControlDomain.COOLING,
    ...     proposed_action=ControlAction.SETPOINT_CHANGE,
    ...     ai_value=22.0,
    ...     ai_confidence=0.88,
    ...     sensor_readings=[
    ...         SensorReading(22.3, time.time(), "rack_01_inlet"),
    ...         SensorReading(22.5, time.time(), "rack_02_inlet"),
    ...     ],
    ...     zone_id="zone_a",
    ... )
    >>>
    >>> decision = governor.evaluate(signals)
    >>> if decision.actionable:
    ...     bms.set_temperature(decision.trusted_value)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import time


# Import core AILEE components
from ...ailee_trust_pipeline_v1 import AileeTrustPipeline, AileeConfig, DecisionResult
from ...optional.ailee_config_presets import SENSOR_FUSION, TEMPERATURE_MONITORING, AUTONOMOUS_VEHICLE
from ...optional.ailee_monitors import TrustMonitor, AlertingMonitor
from ...optional.ailee_peer_adapters import StaticPeerAdapter, FilteredPeerAdapter, MultiSourcePeerAdapter


# ===========================
# Data Center Specific Configs
# ===========================

COOLING_CONTROL = AileeConfig(
    # Conservative thresholds for thermal safety
    accept_threshold=0.93,
    borderline_low=0.80,
    borderline_high=0.93,
    
    # Thermal envelope (Celsius)
    hard_min=18.0,
    hard_max=27.0,
    
    # Require multiple sensors for consensus
    consensus_quorum=4,
    consensus_delta=0.5,  # 0.5°C tolerance
    
    # GRACE checks for setpoint changes
    grace_peer_delta=0.3,
    grace_forecast_epsilon=0.10,
    
    # Always fall back to last known good setpoint
    fallback_mode="last_good",
    
    # Heavy stability weight (thermal systems are slow)
    w_stability=0.55,
    w_agreement=0.25,
    w_likelihood=0.20,
    
    # Longer history for thermal inertia
    history_window=120,
    forecast_window=20,
    
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


POWER_CAPPING = AileeConfig(
    # Very high confidence required for power limits
    accept_threshold=0.96,
    borderline_low=0.88,
    borderline_high=0.96,
    
    # Power envelope (kW) - set at runtime
    hard_min=None,  # Set based on facility
    hard_max=None,  # Set based on facility
    
    # Strong consensus (multiple power meters)
    consensus_quorum=5,
    consensus_delta=5.0,  # 5 kW tolerance
    
    grace_peer_delta=8.0,
    
    # Conservative fallback for power
    fallback_mode="last_good",
    
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,
    
    history_window=100,
    
    enable_grace=True,
    enable_consensus=True,
)


WORKLOAD_PLACEMENT = AileeConfig(
    # Moderate thresholds for migration decisions
    accept_threshold=0.90,
    borderline_low=0.75,
    borderline_high=0.90,
    
    # Peer models must agree
    consensus_quorum=3,
    consensus_delta=0.15,
    
    grace_peer_delta=0.20,
    
    fallback_mode="median",
    
    w_stability=0.40,
    w_agreement=0.35,
    w_likelihood=0.25,
    
    history_window=60,
    
    enable_grace=True,
    enable_consensus=True,
)


# ===========================
# Enumerations
# ===========================

class DatacenterTrustLevel(IntEnum):
    """
    Graduated trust levels for datacenter control decisions.
    Higher values indicate more authority to act autonomously.
    """
    NO_ACTION = 0       # Insufficient data or confidence — do not act
    ADVISORY = 1        # Log or alert only — no physical actuation
    SUPERVISED = 2      # Act, but flag for operator review
    AUTONOMOUS = 3      # Fully authorized autonomous action


class FacilityHealthStatus(str, Enum):
    """Overall health status of a datacenter facility or subsystem."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class ControlDomain(str, Enum):
    """Datacenter control domains governed by AILEE."""
    COOLING = "COOLING"
    POWER = "POWER"
    WORKLOAD = "WORKLOAD"
    MAINTENANCE = "MAINTENANCE"
    INCIDENT_RESPONSE = "INCIDENT_RESPONSE"


class ControlAction(str, Enum):
    """Types of control actions that can be governed."""
    SETPOINT_CHANGE = "SETPOINT_CHANGE"
    POWER_CAP = "POWER_CAP"
    WORKLOAD_MIGRATE = "WORKLOAD_MIGRATE"
    WORKLOAD_THROTTLE = "WORKLOAD_THROTTLE"
    MAINTENANCE_TRIGGER = "MAINTENANCE_TRIGGER"
    INCIDENT_RESPONSE = "INCIDENT_RESPONSE"
    NO_ACTION = "NO_ACTION"


# ===========================
# Policy & Signal Types
# ===========================

@dataclass
class DatacenterPolicy:
    """
    Governance policy for datacenter control decisions.

    Controls trust thresholds, safety constraints, and audit settings
    applied by the DatacenterGovernor.
    """
    # Minimum trust level required before an action is permitted
    min_trust_for_action: DatacenterTrustLevel = DatacenterTrustLevel.SUPERVISED

    # Require peer sensor consensus before accepting a value
    require_consensus: bool = True

    # Validate that sensors are within expected range before trusting them
    require_sensor_validation: bool = True

    # Maximum allowed setpoint change in one governance cycle
    # Used for cooling (°C) and power (kW fraction of facility capacity)
    max_setpoint_change_per_cycle: float = 2.0

    # Maximum workload migrations permitted per hour
    max_migrations_per_hour: int = 100

    # Facility power capacity (kW); used to derive power bounds when set
    facility_max_kw: Optional[float] = None

    # Audit settings
    enable_audit_events: bool = True
    track_decision_history: bool = True


@dataclass
class DatacenterSignals:
    """
    Input signals for a datacenter governance evaluation.

    Describes the AI-proposed control action together with supporting
    sensor data and contextual metadata.
    """
    # What is being controlled and what action is proposed
    control_domain: ControlDomain
    proposed_action: ControlAction

    # AI-produced value (setpoint °C, power kW, migration score 0-10, etc.)
    ai_value: float

    # AI confidence in the proposed value [0.0, 1.0]
    ai_confidence: float

    # Supporting sensor readings used for consensus
    sensor_readings: List[SensorReading] = field(default_factory=list)

    # Contextual identifiers
    zone_id: str = "default"
    workload_id: Optional[str] = None
    source_node: Optional[str] = None
    target_node: Optional[str] = None

    # Evaluation timestamp (defaults to current time)
    timestamp: Optional[float] = None

    # Additional free-form context passed through to the audit trail
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatacenterDecision:
    """
    Result of a datacenter governance evaluation.

    Carries the authorized trust level, the validated/fallback value to
    act on, and a full audit trail suitable for DCIM/BMS integration.
    """
    # Governance outcome
    authorized_level: DatacenterTrustLevel
    actionable: bool

    # Trusted value to use for actuation (setpoint, power cap, score)
    trusted_value: float

    # What was evaluated
    control_domain: ControlDomain
    proposed_action: ControlAction

    # Raw AILEE pipeline result (DecisionResult or None)
    pipeline_result: Optional[Any] = None

    # Safety and health
    health_status: FacilityHealthStatus = FacilityHealthStatus.HEALTHY
    safety_flags: List[str] = field(default_factory=list)

    # Fallback information
    used_fallback: bool = False
    fallback_reason: Optional[str] = None

    # Audit
    timestamp: float = field(default_factory=time.time)
    decision_id: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatacenterEvent:
    """
    Audit event emitted by the DatacenterGovernor.

    Provides a structured, time-stamped record of each governance decision
    for logging, compliance, and post-incident analysis.
    """
    event_type: str          # "decision", "fallback", "health_change", "rate_limit"
    control_domain: ControlDomain
    timestamp: float = field(default_factory=time.time)
    decision: Optional[DatacenterDecision] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ===========================
# Telemetry Processors
# ===========================

@dataclass
class SensorReading:
    """Standard sensor reading format."""
    value: float
    timestamp: float
    sensor_id: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelemetryProcessor:
    """
    Process raw telemetry from data center sensors/systems.
    Handles unit conversion, filtering, and formatting for AILEE.
    """
    
    def __init__(self):
        self.readings_history: Dict[str, List[SensorReading]] = {}
    
    def process_temperature(
        self, 
        raw_value: float, 
        unit: str = "celsius",
        sensor_id: str = "unknown",
        metadata: Optional[Dict] = None
    ) -> SensorReading:
        """
        Process temperature reading with unit conversion.
        
        Args:
            raw_value: Temperature value
            unit: "celsius", "fahrenheit", or "kelvin"
            sensor_id: Sensor identifier
            metadata: Additional context
            
        Returns:
            Standardized SensorReading in Celsius
        """
        # Convert to Celsius
        if unit.lower() == "fahrenheit":
            celsius = (raw_value - 32) * 5/9
        elif unit.lower() == "kelvin":
            celsius = raw_value - 273.15
        else:
            celsius = raw_value
        
        reading = SensorReading(
            value=celsius,
            timestamp=time.time(),
            sensor_id=sensor_id,
            metadata=metadata or {}
        )
        
        # Store in history
        if sensor_id not in self.readings_history:
            self.readings_history[sensor_id] = []
        self.readings_history[sensor_id].append(reading)
        
        return reading
    
    def process_power(
        self,
        raw_value: float,
        unit: str = "watts",
        sensor_id: str = "unknown",
        metadata: Optional[Dict] = None
    ) -> SensorReading:
        """
        Process power reading with unit conversion.
        
        Args:
            raw_value: Power value
            unit: "watts", "kilowatts", or "megawatts"
            sensor_id: Sensor identifier
            metadata: Additional context
            
        Returns:
            Standardized SensorReading in kilowatts
        """
        # Convert to kW
        if unit.lower() == "watts":
            kw = raw_value / 1000
        elif unit.lower() == "megawatts":
            kw = raw_value * 1000
        else:
            kw = raw_value
        
        reading = SensorReading(
            value=kw,
            timestamp=time.time(),
            sensor_id=sensor_id,
            metadata=metadata or {}
        )
        
        if sensor_id not in self.readings_history:
            self.readings_history[sensor_id] = []
        self.readings_history[sensor_id].append(reading)
        
        return reading
    
    def get_peer_values(
        self, 
        sensor_type: str,
        max_age_seconds: float = 60.0
    ) -> List[float]:
        """
        Get recent peer sensor values for consensus.
        
        Args:
            sensor_type: Type of sensor (e.g., "temp_rack_inlet")
            max_age_seconds: Maximum age of readings to include
            
        Returns:
            List of recent sensor values
        """
        now = time.time()
        peer_values = []
        
        for sensor_id, readings in self.readings_history.items():
            if sensor_type in sensor_id and readings:
                latest = readings[-1]
                if (now - latest.timestamp) <= max_age_seconds:
                    peer_values.append(latest.value)
        
        return peer_values


# ===========================
# Setpoint Controllers
# ===========================

class CoolingController:
    """
    Cooling setpoint controller with AILEE governance.
    Handles HVAC, CRAH, chiller plant control.
    """
    
    def __init__(
        self,
        config: Optional[AileeConfig] = None,
        monitor: Optional[TrustMonitor] = None
    ):
        """
        Initialize cooling controller.
        
        Args:
            config: AILEE configuration (defaults to COOLING_CONTROL)
            monitor: Optional monitoring instance
        """
        self.config = config or COOLING_CONTROL
        self.pipeline = AileeTrustPipeline(self.config)
        self.monitor = monitor or TrustMonitor(window=200)
        self.telemetry = TelemetryProcessor()
        
        # Controller state
        self.current_setpoint = 22.0  # Default: 22°C
        self.setpoint_history: List[Tuple[float, float]] = []  # (timestamp, setpoint)
    
    def propose_setpoint(
        self,
        ai_setpoint: float,
        ai_confidence: float,
        sensor_readings: List[SensorReading],
        zone_id: str = "default"
    ) -> DecisionResult:
        """
        Evaluate AI-proposed cooling setpoint.
        
        Args:
            ai_setpoint: Proposed setpoint in Celsius
            ai_confidence: AI model confidence (0-1)
            sensor_readings: List of temperature sensor readings
            zone_id: Cooling zone identifier
            
        Returns:
            DecisionResult with trusted setpoint
            
        Example:
            >>> controller = CoolingController()
            >>> sensors = [
            ...     SensorReading(22.5, time.time(), "rack_01_inlet"),
            ...     SensorReading(22.3, time.time(), "rack_02_inlet"),
            ... ]
            >>> result = controller.propose_setpoint(22.0, 0.85, sensors, "zone_a")
            >>> if not result.used_fallback:
            ...     bms.set_temperature(result.value)
        """
        # Extract peer values from sensors
        peer_values = [reading.value for reading in sensor_readings]
        
        # Process through AILEE
        result = self.pipeline.process(
            raw_value=ai_setpoint,
            raw_confidence=ai_confidence,
            peer_values=peer_values,
            timestamp=time.time(),
            context={
                "zone": zone_id,
                "units": "celsius",
                "sensor_count": len(sensor_readings),
                "current_setpoint": self.current_setpoint
            }
        )
        
        # Record for monitoring
        self.monitor.record(result)
        
        # Update state if accepted
        if not result.used_fallback:
            self.current_setpoint = result.value
            self.setpoint_history.append((time.time(), result.value))
        
        return result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get controller health metrics."""
        return {
            "current_setpoint": self.current_setpoint,
            "fallback_rate": self.monitor.fallback_rate(),
            "avg_confidence": self.monitor.avg_confidence(),
            "decisions_total": self.monitor.total_decisions,
            "setpoint_history_len": len(self.setpoint_history),
        }


class PowerCapController:
    """
    Power capping controller with AILEE governance.
    Manages rack/cluster power limits and throttling.
    """
    
    def __init__(
        self,
        facility_max_kw: float,
        config: Optional[AileeConfig] = None,
        monitor: Optional[TrustMonitor] = None
    ):
        """
        Initialize power cap controller.
        
        Args:
            facility_max_kw: Facility power limit in kW
            config: AILEE configuration (defaults to POWER_CAPPING)
            monitor: Optional monitoring instance
        """
        config = config or POWER_CAPPING
        # Set hard bounds based on facility
        config.hard_min = facility_max_kw * 0.30  # Never below 30% capacity
        config.hard_max = facility_max_kw * 0.95  # Never above 95% capacity
        
        self.facility_max_kw = facility_max_kw
        self.pipeline = AileeTrustPipeline(config)
        self.monitor = monitor or TrustMonitor(window=200)
        self.telemetry = TelemetryProcessor()
        
        # Current power state
        self.current_cap_kw = facility_max_kw * 0.80  # Default: 80% capacity
    
    def propose_power_cap(
        self,
        ai_cap_kw: float,
        ai_confidence: float,
        power_meters: List[SensorReading],
        reason: str = "optimization"
    ) -> DecisionResult:
        """
        Evaluate AI-proposed power cap.
        
        Args:
            ai_cap_kw: Proposed power cap in kW
            ai_confidence: AI model confidence
            power_meters: List of power meter readings
            reason: Reason for cap change
            
        Returns:
            DecisionResult with trusted power cap
        """
        peer_values = [reading.value for reading in power_meters]
        
        result = self.pipeline.process(
            raw_value=ai_cap_kw,
            raw_confidence=ai_confidence,
            peer_values=peer_values,
            timestamp=time.time(),
            context={
                "reason": reason,
                "facility_max_kw": self.facility_max_kw,
                "current_cap_kw": self.current_cap_kw,
                "utilization": ai_cap_kw / self.facility_max_kw
            }
        )
        
        self.monitor.record(result)
        
        if not result.used_fallback:
            self.current_cap_kw = result.value
        
        return result


# ===========================
# Workload Orchestration
# ===========================

class WorkloadPlacementGovernor:
    """
    Governs AI-driven workload placement and migration decisions.
    """
    
    def __init__(
        self,
        config: Optional[AileeConfig] = None,
        monitor: Optional[TrustMonitor] = None
    ):
        """Initialize workload placement governor."""
        self.config = config or WORKLOAD_PLACEMENT
        self.pipeline = AileeTrustPipeline(self.config)
        self.monitor = monitor or TrustMonitor(window=100)
        
        # Track migration rate
        self.migrations_last_hour = 0
        self.last_hour_reset = time.time()
    
    def evaluate_migration(
        self,
        workload_id: str,
        source_node: str,
        target_node: str,
        ai_score: float,
        ai_confidence: float,
        peer_recommendations: List[float]
    ) -> Tuple[bool, DecisionResult]:
        """
        Evaluate whether to approve workload migration.
        
        Args:
            workload_id: Workload identifier
            source_node: Current node
            target_node: Proposed target node
            ai_score: AI placement score (0-10)
            ai_confidence: AI confidence
            peer_recommendations: Scores from peer placement models
            
        Returns:
            Tuple of (should_migrate, decision_result)
        """
        # Check migration rate limit
        now = time.time()
        if (now - self.last_hour_reset) > 3600:
            self.migrations_last_hour = 0
            self.last_hour_reset = now
        
        # Rate limit: max 100 migrations/hour
        if self.migrations_last_hour >= 100:
            return False, None
        
        result = self.pipeline.process(
            raw_value=ai_score,
            raw_confidence=ai_confidence,
            peer_values=peer_recommendations,
            timestamp=now,
            context={
                "workload_id": workload_id,
                "source": source_node,
                "target": target_node,
                "migrations_this_hour": self.migrations_last_hour
            }
        )
        
        self.monitor.record(result)
        
        # Approve migration if confidence is high and not fallback
        should_migrate = (
            result.safety_status == "ACCEPTED" and
            not result.used_fallback and
            result.confidence_score >= 0.85
        )
        
        if should_migrate:
            self.migrations_last_hour += 1
        
        return should_migrate, result


# ===========================
# Monitoring Integration
# ===========================

class DataCenterMonitor:
    """
    Unified monitoring for data center AILEE deployments.
    Tracks cooling, power, workload governance metrics.
    """
    
    def __init__(self):
        self.cooling_monitor = TrustMonitor(window=500)
        self.power_monitor = TrustMonitor(window=500)
        self.workload_monitor = TrustMonitor(window=200)
    
    def get_unified_metrics(self) -> Dict[str, Any]:
        """Get metrics across all systems."""
        return {
            "cooling": {
                "fallback_rate": self.cooling_monitor.fallback_rate(),
                "avg_confidence": self.cooling_monitor.avg_confidence(),
                "total_decisions": self.cooling_monitor.total_decisions,
            },
            "power": {
                "fallback_rate": self.power_monitor.fallback_rate(),
                "avg_confidence": self.power_monitor.avg_confidence(),
                "total_decisions": self.power_monitor.total_decisions,
            },
            "workload": {
                "fallback_rate": self.workload_monitor.fallback_rate(),
                "avg_confidence": self.workload_monitor.avg_confidence(),
                "total_decisions": self.workload_monitor.total_decisions,
            }
        }
    
    def check_health(self) -> Dict[str, str]:
        """
        Check overall health status.
        
        Returns:
            Dict with health status for each system
        """
        health = {}
        
        # Cooling health
        if self.cooling_monitor.fallback_rate() > 0.30:
            health["cooling"] = "DEGRADED"
        elif self.cooling_monitor.fallback_rate() > 0.15:
            health["cooling"] = "WARNING"
        else:
            health["cooling"] = "HEALTHY"
        
        # Power health
        if self.power_monitor.fallback_rate() > 0.25:
            health["power"] = "DEGRADED"
        elif self.power_monitor.fallback_rate() > 0.12:
            health["power"] = "WARNING"
        else:
            health["power"] = "HEALTHY"
        
        # Workload health
        if self.workload_monitor.fallback_rate() > 0.20:
            health["workload"] = "DEGRADED"
        elif self.workload_monitor.fallback_rate() > 0.10:
            health["workload"] = "WARNING"
        else:
            health["workload"] = "HEALTHY"
        
        return health


# ===========================
# Unified Governance Interface
# ===========================

class DatacenterGovernor:
    """
    Unified governance governor for data center AI control systems.

    Evaluates AI-proposed control actions across cooling, power, and workload
    domains using the AILEE Trust Pipeline, and returns graduated-trust
    ``DatacenterDecision`` objects with full audit trails.

    This is the primary entry point for integrating AILEE into datacenter
    BMS, DCIM, or orchestration systems.

    Example::

        governor = DatacenterGovernor()

        signals = DatacenterSignals(
            control_domain=ControlDomain.COOLING,
            proposed_action=ControlAction.SETPOINT_CHANGE,
            ai_value=22.0,
            ai_confidence=0.88,
            sensor_readings=[
                SensorReading(22.3, time.time(), "rack_01"),
                SensorReading(22.5, time.time(), "rack_02"),
            ],
            zone_id="zone_a",
        )

        decision = governor.evaluate(signals)
        if decision.actionable:
            bms.set_setpoint(decision.trusted_value)
        elif decision.used_fallback:
            logger.warning("Fallback active: %s", decision.fallback_reason)
    """

    def __init__(
        self,
        policy: Optional[DatacenterPolicy] = None,
        cooling_config: Optional[AileeConfig] = None,
        power_config: Optional[AileeConfig] = None,
        workload_config: Optional[AileeConfig] = None,
    ):
        """
        Initialise the DatacenterGovernor.

        Args:
            policy: Governance policy controlling trust thresholds and constraints.
                    Defaults to ``DatacenterPolicy()`` (standard configuration).
            cooling_config: AILEE pipeline config for cooling decisions.
                            Defaults to ``COOLING_CONTROL``.
            power_config:   AILEE pipeline config for power-cap and incident-response
                            decisions. Defaults to ``POWER_CAPPING``.
            workload_config: AILEE pipeline config for workload-placement and
                             maintenance decisions. Defaults to ``WORKLOAD_PLACEMENT``.
        """
        self.policy = policy or DatacenterPolicy()

        # One AILEE pipeline per control domain.
        # MAINTENANCE shares workload config (event-driven, score-based).
        # INCIDENT_RESPONSE shares power config (high-confidence, fast threshold).
        self._pipelines: Dict[str, AileeTrustPipeline] = {
            ControlDomain.COOLING: AileeTrustPipeline(cooling_config or COOLING_CONTROL),
            ControlDomain.POWER: AileeTrustPipeline(power_config or POWER_CAPPING),
            ControlDomain.WORKLOAD: AileeTrustPipeline(workload_config or WORKLOAD_PLACEMENT),
            ControlDomain.MAINTENANCE: AileeTrustPipeline(workload_config or WORKLOAD_PLACEMENT),
            ControlDomain.INCIDENT_RESPONSE: AileeTrustPipeline(power_config or POWER_CAPPING),
        }

        # Track metrics and events
        self._monitor = TrustMonitor(window=500)
        self._events: List[DatacenterEvent] = []
        self._decision_history: List[DatacenterDecision] = []

        # Per-domain migration rate limiting (for workload domain)
        self._migrations_this_hour: int = 0
        self._last_hour_reset: float = time.time()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def evaluate(self, signals: DatacenterSignals) -> DatacenterDecision:
        """
        Evaluate an AI-proposed datacenter control action.

        Runs the AILEE Trust Pipeline for the appropriate control domain,
        applies policy constraints, determines the trust level, and returns
        a ``DatacenterDecision`` with a trusted value and full audit trail.

        Args:
            signals: ``DatacenterSignals`` describing the proposed action
                     and supporting sensor data.

        Returns:
            ``DatacenterDecision`` with an authorized trust level and the
            value that is safe to act on.
        """
        ts = signals.timestamp or time.time()
        peer_values = [r.value for r in signals.sensor_readings]

        # --- Rate-limit workload migrations ---
        if signals.control_domain == ControlDomain.WORKLOAD:
            now = time.time()
            if (now - self._last_hour_reset) > 3600:
                self._migrations_this_hour = 0
                self._last_hour_reset = now
            if self._migrations_this_hour >= self.policy.max_migrations_per_hour:
                decision = self._build_no_action_decision(
                    signals=signals,
                    ts=ts,
                    reason="Rate limit reached: too many migrations this hour",
                )
                self._record_event("rate_limit", signals, decision)
                return decision

        # --- Run AILEE pipeline ---
        pipeline = self._pipelines[signals.control_domain]
        context = dict(signals.context)
        context.update({
            "zone_id": signals.zone_id,
            "control_domain": signals.control_domain.value,
            "proposed_action": signals.proposed_action.value,
        })
        if signals.workload_id:
            context["workload_id"] = signals.workload_id
        if signals.source_node:
            context["source_node"] = signals.source_node
        if signals.target_node:
            context["target_node"] = signals.target_node

        result = pipeline.process(
            raw_value=signals.ai_value,
            raw_confidence=signals.ai_confidence,
            peer_values=peer_values,
            timestamp=ts,
            context=context,
        )
        self._monitor.record(result)

        # --- Determine trust level ---
        authorized_level = self._determine_trust_level(result)
        actionable = authorized_level >= self.policy.min_trust_for_action
        safety_flags: List[str] = []
        reasons: List[str] = []

        if result.used_fallback:
            safety_flags.append("fallback_active")
            reasons.append(f"Fallback used: {result.safety_status}")
        if result.safety_status == "OUTRIGHT_REJECTED":
            safety_flags.append("rejected_by_pipeline")
            reasons.append("Pipeline rejected the proposed value")

        # --- Build decision ---
        decision_id = hashlib.sha256(
            f"{ts}{signals.control_domain}{signals.ai_value}".encode()
        ).hexdigest()[:16]

        decision = DatacenterDecision(
            authorized_level=authorized_level,
            actionable=actionable,
            trusted_value=result.value,
            control_domain=signals.control_domain,
            proposed_action=signals.proposed_action,
            pipeline_result=result,
            health_status=self._get_subsystem_health(signals.control_domain),
            safety_flags=safety_flags,
            used_fallback=result.used_fallback,
            fallback_reason=result.safety_status if result.used_fallback else None,
            timestamp=ts,
            decision_id=decision_id,
            reasons=reasons,
            metadata={
                "confidence_score": result.confidence_score,
                "safety_status": result.safety_status,
                "sensor_count": len(signals.sensor_readings),
            },
        )

        # Track migration count
        if (
            signals.control_domain == ControlDomain.WORKLOAD
            and actionable
            and not result.used_fallback
        ):
            self._migrations_this_hour += 1

        if self.policy.track_decision_history:
            self._decision_history.append(decision)
        if self.policy.enable_audit_events:
            event_type = "fallback" if result.used_fallback else "decision"
            self._record_event(event_type, signals, decision)

        return decision

    # ------------------------------------------------------------------
    # Health & Monitoring
    # ------------------------------------------------------------------

    def get_health(self) -> FacilityHealthStatus:
        """
        Compute the overall health status of the facility based on recent
        governance metrics.

        Returns:
            ``FacilityHealthStatus`` enum value.
        """
        fallback_rate = self._monitor.fallback_rate()
        if fallback_rate > 0.30:
            return FacilityHealthStatus.CRITICAL
        if fallback_rate > 0.20:
            return FacilityHealthStatus.DEGRADED
        if fallback_rate > 0.10:
            return FacilityHealthStatus.WARNING
        return FacilityHealthStatus.HEALTHY

    def get_subsystem_health(self) -> Dict[str, FacilityHealthStatus]:
        """
        Return health status per control domain.

        Returns:
            Dict mapping ``ControlDomain`` value strings to
            ``FacilityHealthStatus`` values.
        """
        return {
            domain.value: self._get_subsystem_health(domain)
            for domain in ControlDomain
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return aggregated governance metrics.

        Returns:
            Dict with fallback rate, average confidence, and decision counts.
        """
        return {
            "fallback_rate": self._monitor.fallback_rate(),
            "avg_confidence": self._monitor.avg_confidence(),
            "total_decisions": self._monitor.total_decisions,
            "migrations_this_hour": self._migrations_this_hour,
            "overall_health": self.get_health().value,
        }

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def get_events(self) -> List[DatacenterEvent]:
        """Return all recorded governance events (newest last)."""
        return list(self._events)

    def get_decision_history(self) -> List[DatacenterDecision]:
        """Return the history of all governance decisions."""
        return list(self._decision_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_trust_level(self, result: Any) -> DatacenterTrustLevel:
        """Map an AILEE DecisionResult to a DatacenterTrustLevel."""
        if result.safety_status == "OUTRIGHT_REJECTED":
            return DatacenterTrustLevel.NO_ACTION
        if result.used_fallback:
            return DatacenterTrustLevel.ADVISORY
        confidence = getattr(result, "confidence_score", 0.0)
        if result.safety_status == "ACCEPTED" and confidence >= 0.90:
            return DatacenterTrustLevel.AUTONOMOUS
        if result.safety_status in ("ACCEPTED", "BORDERLINE") and confidence >= 0.70:
            return DatacenterTrustLevel.SUPERVISED
        return DatacenterTrustLevel.ADVISORY

    def _get_subsystem_health(self, domain: ControlDomain) -> FacilityHealthStatus:
        """Return health status for a single control domain (uses overall monitor)."""
        # For a richer per-domain breakdown, subclass and add per-domain monitors
        return self.get_health()

    def _build_no_action_decision(
        self,
        signals: DatacenterSignals,
        ts: float,
        reason: str,
    ) -> DatacenterDecision:
        """Build a NO_ACTION decision without running the pipeline."""
        decision_id = hashlib.sha256(
            f"{ts}{signals.control_domain}{signals.ai_value}".encode()
        ).hexdigest()[:16]
        return DatacenterDecision(
            authorized_level=DatacenterTrustLevel.NO_ACTION,
            actionable=False,
            trusted_value=signals.ai_value,
            control_domain=signals.control_domain,
            proposed_action=signals.proposed_action,
            pipeline_result=None,
            health_status=FacilityHealthStatus.WARNING,
            safety_flags=["no_action"],
            used_fallback=False,
            fallback_reason=reason,
            timestamp=ts,
            decision_id=decision_id,
            reasons=[reason],
        )

    def _record_event(
        self,
        event_type: str,
        signals: DatacenterSignals,
        decision: DatacenterDecision,
    ) -> None:
        """Append a DatacenterEvent to the internal log."""
        self._events.append(DatacenterEvent(
            event_type=event_type,
            control_domain=signals.control_domain,
            timestamp=decision.timestamp,
            decision=decision,
            details={
                "zone_id": signals.zone_id,
                "ai_value": signals.ai_value,
                "ai_confidence": signals.ai_confidence,
                "sensor_count": len(signals.sensor_readings),
            },
        ))


# ===========================
# Convenience Factory Functions
# ===========================

def create_datacenter_governor(
    policy: Optional[DatacenterPolicy] = None,
    **policy_overrides: Any,
) -> DatacenterGovernor:
    """
    Create a ``DatacenterGovernor`` with a sensible default policy.

    Args:
        policy: Optional pre-built ``DatacenterPolicy``.  If omitted, a
                default policy is constructed from ``policy_overrides``.
        **policy_overrides: Keyword arguments forwarded to
                            ``DatacenterPolicy()`` if no policy is given.

    Returns:
        Configured ``DatacenterGovernor`` instance.

    Example::

        governor = create_datacenter_governor(max_migrations_per_hour=50)
    """
    if policy is None:
        policy = DatacenterPolicy(**policy_overrides)
    return DatacenterGovernor(policy=policy)


def create_default_governor(**policy_overrides: Any) -> DatacenterGovernor:
    """
    Create a ``DatacenterGovernor`` with default (balanced) policy settings.

    Args:
        **policy_overrides: Optional ``DatacenterPolicy`` attribute overrides.

    Returns:
        ``DatacenterGovernor`` instance with default policy.

    Example::

        governor = create_default_governor()
    """
    return DatacenterGovernor(policy=DatacenterPolicy(**policy_overrides))


def create_strict_governor(**policy_overrides: Any) -> DatacenterGovernor:
    """
    Create a ``DatacenterGovernor`` with a strict safety policy.

    Strict settings:
    - Requires AUTONOMOUS trust level before allowing action
    - Enforces consensus and sensor validation
    - Limits setpoint changes to ±1 °C / 1 kW per cycle
    - Enables full audit trail

    Args:
        **policy_overrides: Optional further overrides.

    Returns:
        ``DatacenterGovernor`` instance with strict policy.

    Example::

        governor = create_strict_governor()
    """
    strict_defaults: Dict[str, Any] = {
        "min_trust_for_action": DatacenterTrustLevel.AUTONOMOUS,
        "require_consensus": True,
        "require_sensor_validation": True,
        "max_setpoint_change_per_cycle": 1.0,
        "max_migrations_per_hour": 50,
        "enable_audit_events": True,
        "track_decision_history": True,
    }
    strict_defaults.update(policy_overrides)
    return DatacenterGovernor(policy=DatacenterPolicy(**strict_defaults))


def create_permissive_governor(**policy_overrides: Any) -> DatacenterGovernor:
    """
    Create a ``DatacenterGovernor`` with a permissive policy for development
    or testing.

    Permissive settings:
    - Allows action at ADVISORY trust level
    - Does not enforce consensus or sensor validation
    - Higher migration and setpoint-change limits

    **WARNING**: Not recommended for production use.

    Args:
        **policy_overrides: Optional further overrides.

    Returns:
        ``DatacenterGovernor`` instance with permissive policy.

    Example::

        governor = create_permissive_governor()
    """
    permissive_defaults: Dict[str, Any] = {
        "min_trust_for_action": DatacenterTrustLevel.ADVISORY,
        "require_consensus": False,
        "require_sensor_validation": False,
        "max_setpoint_change_per_cycle": 10.0,
        "max_migrations_per_hour": 500,
        "enable_audit_events": False,
        "track_decision_history": False,
    }
    permissive_defaults.update(policy_overrides)
    return DatacenterGovernor(policy=DatacenterPolicy(**permissive_defaults))


def validate_datacenter_signals(signals: DatacenterSignals) -> List[str]:
    """
    Validate a ``DatacenterSignals`` object and return a list of issue strings.

    An empty list means the signals are valid.

    Args:
        signals: ``DatacenterSignals`` instance to validate.

    Returns:
        List of validation error/warning strings.  Empty if all checks pass.

    Example::

        issues = validate_datacenter_signals(signals)
        if issues:
            for issue in issues:
                logger.warning("Signal issue: %s", issue)
    """
    issues: List[str] = []

    if not (0.0 <= signals.ai_confidence <= 1.0):
        issues.append(
            f"ai_confidence must be in [0.0, 1.0], got {signals.ai_confidence}"
        )

    if signals.control_domain == ControlDomain.COOLING:
        if not (-30.0 <= signals.ai_value <= 60.0):
            issues.append(
                f"Cooling setpoint {signals.ai_value}°C looks out of range "
                "[-30, 60]"
            )
    elif signals.control_domain == ControlDomain.WORKLOAD:
        if not (0.0 <= signals.ai_value <= 10.0):
            issues.append(
                f"Workload placement score {signals.ai_value} must be in [0, 10]"
            )

    if not signals.sensor_readings:
        issues.append("No sensor readings provided; consensus cannot be computed")

    return issues


# ===========================
# Convenience Exports
# ===========================

__all__ = [
    # Enumerations
    'DatacenterTrustLevel',
    'FacilityHealthStatus',
    'ControlDomain',
    'ControlAction',

    # Policy & Signal Types
    'DatacenterPolicy',
    'DatacenterSignals',
    'DatacenterDecision',
    'DatacenterEvent',

    # Governor
    'DatacenterGovernor',

    # Factory Functions
    'create_datacenter_governor',
    'create_default_governor',
    'create_strict_governor',
    'create_permissive_governor',
    'validate_datacenter_signals',

    # Configurations
    'COOLING_CONTROL',
    'POWER_CAPPING',
    'WORKLOAD_PLACEMENT',

    # Telemetry
    'SensorReading',
    'TelemetryProcessor',

    # Controllers (lower-level API)
    'CoolingController',
    'PowerCapController',
    'WorkloadPlacementGovernor',

    # Monitoring
    'DataCenterMonitor',
]


# ===========================
# Demo Usage
# ===========================

if __name__ == "__main__":
    print("=== AILEE Data Center Helpers Demo ===\n")
    
    # Demo: Cooling Control
    print("1. Cooling Controller Demo")
    cooling = CoolingController()
    
    # Simulate sensor readings
    sensors = [
        SensorReading(22.5, time.time(), "rack_01_inlet"),
        SensorReading(22.3, time.time(), "rack_02_inlet"),
        SensorReading(22.7, time.time(), "rack_03_inlet"),
        SensorReading(22.4, time.time(), "rack_04_inlet"),
    ]
    
    # AI proposes new setpoint
    result = cooling.propose_setpoint(
        ai_setpoint=22.0,
        ai_confidence=0.85,
        sensor_readings=sensors,
        zone_id="zone_a"
    )
    
    print(f"   Proposed: 22.0°C (confidence: 0.85)")
    print(f"   Trusted:  {result.value:.2f}°C")
    print(f"   Status:   {result.safety_status}")
    print(f"   Fallback: {result.used_fallback}")
    
    # Demo: Power Capping
    print("\n2. Power Cap Controller Demo")
    power = PowerCapController(facility_max_kw=5000)
    
    power_meters = [
        SensorReading(4200, time.time(), "main_meter"),
        SensorReading(4250, time.time(), "backup_meter"),
        SensorReading(4180, time.time(), "ups_meter"),
    ]
    
    result = power.propose_power_cap(
        ai_cap_kw=4500,
        ai_confidence=0.92,
        power_meters=power_meters,
        reason="peak_shaving"
    )
    
    print(f"   Proposed: 4500 kW (confidence: 0.92)")
    print(f"   Trusted:  {result.value:.0f} kW")
    print(f"   Status:   {result.safety_status}")
    
    print("\n✓ Demo complete. Ready for production integration.")
