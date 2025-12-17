"""
AILEE Trust Layer - Data Center Helpers
Version: 1.1.1

Specialized utilities and pre-configured pipelines for data center operations:
- Cooling/HVAC control
- Power management
- Workload placement
- Predictive maintenance
- Incident automation

This module provides production-ready helpers that integrate AILEE Trust Layer
with common data center systems (BMS, DCIM, SCADA, orchestrators).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time


# Import core AILEE components
try:
    from ailee_trust_pipeline_v1 import AileeTrustPipeline, AileeConfig, DecisionResult
    from ailee_config_presets import SENSOR_FUSION, TEMPERATURE_MONITORING, AUTONOMOUS_VEHICLE
    from ailee_monitors import TrustMonitor, AlertingMonitor
    from ailee_peer_adapters import StaticPeerAdapter, FilteredPeerAdapter, MultiSourcePeerAdapter
except ImportError:
    # Allow import even if not all modules available
    pass


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
# Convenience Exports
# ===========================

__all__ = [
    # Configurations
    'COOLING_CONTROL',
    'POWER_CAPPING',
    'WORKLOAD_PLACEMENT',
    
    # Telemetry
    'SensorReading',
    'TelemetryProcessor',
    
    # Controllers
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
