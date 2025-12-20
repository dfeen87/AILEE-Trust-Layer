"""
AILEE Trust Layer — Telecommunications Governance Domain
Version: 2.0.0 - Production Grade

Telecommunications-focused governance domain for communication systems operating
under uncertainty, latency, bandwidth constraints, and energy limits.

This domain does NOT implement networking protocols, routing algorithms, or PHY controls.
It governs whether received communications are trustworthy enough to act upon based on:
- Link quality metrics (latency, jitter, packet loss)
- Message freshness and currency
- Redundant path consensus
- Network operational domain constraints
- System health indicators

Primary governed signals:
- Communication trust level (scalar, 0..3)
- Link quality scores
- Message age validation
- Multi-path consensus

Trust level semantics:
    0 = NO_TRUST              Do not act on received data
    1 = ADVISORY_TRUST        Use data for advisory purposes only
    2 = CONSTRAINED_TRUST     Act on data within safety boundaries
    3 = FULL_TRUST            Full confidence in communication link

Key properties:
- Deterministic trust decisions (no randomness)
- Auditable, reason-rich outputs with black box logging
- Message freshness enforcement
- Multi-path validation support
- Stability-oriented (prevents trust thrashing using hysteresis)
- Network domain aware (connectivity mode, security, interference)
- System health integration (latency, resources, errors)
- Predictive degradation warnings
- Energy-efficient communication validation

INTEGRATION EXAMPLE:

    # Setup (once)
    policy = TelecomGovernancePolicy(
        max_allowed_level=CommunicationTrustLevel.CONSTRAINED_TRUST,
        max_message_age_ms=500.0,
    )
    governor = TelecomGovernor(policy=policy)
    
    # Per-message evaluation
    while system_active:
        message = network.receive_message()
        
        signals = TelecomSignals(
            desired_level=application.get_required_trust(),
            message_confidence=message.get_confidence(),
            redundant_path_values=(alt_msg1.value, alt_msg2.value),
            link_quality=LinkQualitySignals(
                latency_ms=network.measure_latency(),
                jitter_ms=network.measure_jitter(),
                packet_loss_rate=network.get_loss_rate(),
            ),
            network_domain=NetworkOperationalDomain(
                connectivity_mode="online",
                security_posture="secured",
            ),
            system_health=SystemHealth(
                processing_latency_ms=system.get_latency(),
                network_interface_errors=system.get_errors(),
            ),
            message_age_ms=time.time() * 1000 - message.timestamp,
        )
        
        authorized_level, decision = governor.evaluate(signals)
        
        if authorized_level >= application.minimum_trust_required():
            application.process_message(message)
        else:
            application.use_fallback_strategy()
        
        # Log for compliance
        audit_log.record(governor.get_last_event())

⚠️  This module DOES NOT control data transport.
    It determines whether to ACT ON received information.
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
    from ailee_trust_pipeline_v1 import (
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
# Communication Trust Levels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CommunicationTrustLevel(IntEnum):
    """
    Discrete communication trust authorization levels.
    
    These levels represent increasing confidence in received communications,
    with each level having specific requirements and operator oversight.
    """
    NO_TRUST = 0
    ADVISORY_TRUST = 1
    CONSTRAINED_TRUST = 2
    FULL_TRUST = 3


# Alias for cross-domain consistency
TrustLevel = CommunicationTrustLevel


def _clamp_int(x: int, lo: int, hi: int) -> int:
    """Clamp integer to inclusive range [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def _level_from_float(x: float) -> CommunicationTrustLevel:
    """
    Quantize continuous value to nearest discrete trust level.
    
    Args:
        x: Continuous value to quantize
    
    Returns:
        Trust level clamped to valid range [0..3]
    """
    ix = int(round(float(x)))
    ix = _clamp_int(
        ix,
        int(CommunicationTrustLevel.NO_TRUST),
        int(CommunicationTrustLevel.FULL_TRUST)
    )
    return CommunicationTrustLevel(ix)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Link Quality Signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class LinkQualitySignals:
    """
    Network link quality metrics for trust assessment.
    
    These should come from network monitoring systems and represent
    current communication channel health.
    """
    # Latency metrics
    latency_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    
    # Reliability metrics
    packet_loss_rate: Optional[float] = None  # 0..1
    bit_error_rate: Optional[float] = None
    retry_count: int = 0
    
    # Signal strength (if applicable)
    signal_strength_dbm: Optional[float] = None
    snr_db: Optional[float] = None
    
    # Bandwidth
    bandwidth_available_mbps: Optional[float] = None
    congestion_level: Optional[float] = None  # 0..1
    
    # Stability
    link_stability_score: Optional[float] = None  # 0..1
    path_mtu: Optional[int] = None
    
    def is_safe_for_level(self, level: CommunicationTrustLevel) -> Tuple[bool, List[str]]:
        """
        Determine if current link quality permits the requested trust level.
        
        Args:
            level: Requested trust level
        
        Returns:
            (is_safe, list of issues) - Empty list indicates acceptable quality
        """
        issues: List[str] = []
        
        if level >= CommunicationTrustLevel.ADVISORY_TRUST:
            if self.packet_loss_rate is not None and self.packet_loss_rate > 0.10:
                issues.append(f"packet_loss={self.packet_loss_rate:.3f} exceeds 0.10")
        
        if level >= CommunicationTrustLevel.CONSTRAINED_TRUST:
            if self.latency_ms is not None and self.latency_ms > 200.0:
                issues.append(f"latency={self.latency_ms:.1f}ms exceeds 200ms")
            
            if self.jitter_ms is not None and self.jitter_ms > 20.0:
                issues.append(f"jitter={self.jitter_ms:.1f}ms exceeds 20ms")
            
            if self.packet_loss_rate is not None and self.packet_loss_rate > 0.05:
                issues.append(f"packet_loss={self.packet_loss_rate:.3f} exceeds 0.05")
            
            if self.link_stability_score is not None and self.link_stability_score < 0.70:
                issues.append(f"link_stability={self.link_stability_score:.2f} below 0.70")
        
        if level >= CommunicationTrustLevel.FULL_TRUST:
            if self.latency_ms is not None and self.latency_ms > 50.0:
                issues.append(f"latency={self.latency_ms:.1f}ms exceeds 50ms for full trust")
            
            if self.jitter_ms is not None and self.jitter_ms > 5.0:
                issues.append(f"jitter={self.jitter_ms:.1f}ms exceeds 5ms for full trust")
            
            if self.packet_loss_rate is not None and self.packet_loss_rate > 0.005:
                issues.append(f"packet_loss={self.packet_loss_rate:.3f} exceeds 0.005 for full trust")
            
            if self.link_stability_score is not None and self.link_stability_score < 0.85:
                issues.append(f"link_stability={self.link_stability_score:.2f} below 0.85 for full trust")
        
        return len(issues) == 0, issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Network Operational Domain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class NetworkOperationalDomain:
    """
    Network operating conditions that constrain communication trust.
    Defines environmental and operational boundaries.
    """
    network_type: str = "ethernet"  # ethernet | wifi | 5g | satellite | mesh | tsn
    connectivity_mode: str = "online"  # online | degraded | intermittent | offline
    
    # Environmental factors
    interference_level: str = "low"  # low | moderate | high | severe
    mobility_state: str = "stationary"  # stationary | low_mobility | high_mobility
    weather_impact: str = "none"  # none | rain | fog | severe
    
    # Operational state
    emergency_mode: bool = False
    congestion_severity: str = "normal"  # normal | elevated | high | critical
    security_posture: str = "secured"  # secured | degraded | compromised
    
    # Spectrum/resources
    spectrum_availability: Optional[float] = None  # 0..1
    physical_obstruction: bool = False
    
    def max_safe_level(self) -> CommunicationTrustLevel:
        """Returns maximum trust level permitted by operating conditions."""
        # Hard blocks
        if self.connectivity_mode == "offline":
            return CommunicationTrustLevel.NO_TRUST
        
        if self.security_posture == "compromised":
            return CommunicationTrustLevel.NO_TRUST
        
        # Degraded conditions
        if self.interference_level in ("high", "severe"):
            return CommunicationTrustLevel.ADVISORY_TRUST
        
        if self.congestion_severity == "critical":
            return CommunicationTrustLevel.ADVISORY_TRUST
        
        if self.connectivity_mode == "intermittent":
            return CommunicationTrustLevel.CONSTRAINED_TRUST
        
        if self.congestion_severity in ("elevated", "high"):
            return CommunicationTrustLevel.CONSTRAINED_TRUST
        
        if self.connectivity_mode == "degraded":
            return CommunicationTrustLevel.CONSTRAINED_TRUST
        
        if self.security_posture == "degraded":
            return CommunicationTrustLevel.CONSTRAINED_TRUST
        
        # Mobility considerations
        if self.mobility_state == "high_mobility" and self.network_type not in ("5g", "satellite"):
            return CommunicationTrustLevel.CONSTRAINED_TRUST
        
        # Default: full trust allowed if all checks pass
        return CommunicationTrustLevel.FULL_TRUST


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System Health
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class SystemHealth:
    """Communication system infrastructure health metrics."""
    processing_latency_ms: Optional[float] = None
    buffer_utilization: Optional[float] = None  # 0..1
    cpu_load: Optional[float] = None  # 0..1
    memory_available_mb: Optional[float] = None
    
    # Network interface health
    network_interface_errors: int = 0
    routing_table_stable: bool = True
    dns_resolution_time_ms: Optional[float] = None
    
    # Security overhead
    encryption_overhead_ms: Optional[float] = None
    
    # Time synchronization
    clock_sync_offset_ms: Optional[float] = None
    
    # Storage for buffering
    storage_available_gb: Optional[float] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Redundancy State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class RedundancyState:
    """Multi-path communication redundancy monitoring."""
    active_paths: int = 1
    path_agreement_score: Optional[float] = None  # 0..1
    primary_path_health: Optional[float] = None  # 0..1
    backup_path_health: Optional[float] = None  # 0..1
    failover_ready: bool = False
    diversity_score: Optional[float] = None  # 0..1, path independence
    cross_validation_passed: bool = True
    synchronization_offset_ms: Optional[float] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain Input Signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class TelecomSignals:
    """Primary input structure for telecommunications governance decisions."""
    desired_level: CommunicationTrustLevel
    message_confidence: Optional[float] = None  # 0..1
    
    # Multi-path validation
    redundant_path_values: Tuple[float, ...] = ()
    
    # Link quality
    link_quality: Optional[LinkQualitySignals] = None
    network_domain: Optional[NetworkOperationalDomain] = None
    system_health: Optional[SystemHealth] = None
    redundancy_state: Optional[RedundancyState] = None
    
    # Message metadata
    message_age_ms: Optional[float] = None
    message_id: Optional[str] = None
    
    # Scenario identification
    current_scenario: Optional[str] = None  # e.g., "normal_operation", "high_mobility"
    
    timestamp: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Confidence Tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConfidenceTracker:
    """Track link quality trends at multiple timescales for early warning."""
    
    def __init__(self):
        self.short_term: List[Tuple[float, float]] = []   # 30 seconds
        self.medium_term: List[Tuple[float, float]] = []  # 3 minutes
        self.long_term: List[Tuple[float, float]] = []    # 10 minutes
    
    def update(self, ts: float, quality: float) -> None:
        """Add new quality sample."""
        self.short_term.append((ts, quality))
        self.medium_term.append((ts, quality))
        self.long_term.append((ts, quality))
        
        # Trim windows
        self.short_term = [(t, q) for t, q in self.short_term if ts - t <= 30.0]
        self.medium_term = [(t, q) for t, q in self.medium_term if ts - t <= 180.0]
        self.long_term = [(t, q) for t, q in self.long_term if ts - t <= 600.0]
    
    def is_quality_declining(self) -> Tuple[bool, str]:
        """Detect quality erosion as early warning signal."""
        if len(self.short_term) < 5 or len(self.medium_term) < 10:
            return False, "insufficient_history"
        
        short_avg = statistics.fmean([q for _, q in self.short_term])
        medium_avg = statistics.fmean([q for _, q in self.medium_term])
        
        if short_avg < medium_avg - 0.15:  # 15% drop
            return True, f"quality_decline: short={short_avg:.2f} vs medium={medium_avg:.2f}"
        
        if len(self.short_term) >= 5:
            short_values = [q for _, q in self.short_term]
            try:
                variance = statistics.pvariance(short_values)
                if variance > 0.05:
                    return True, f"high_quality_volatility: variance={variance:.3f}"
            except statistics.StatisticsError:
                pass
        
        return False, ""
    
    def get_trend(self) -> str:
        """Get overall trend: improving, stable, declining."""
        if len(self.short_term) < 5 or len(self.long_term) < 20:
            return "unknown"
        
        short_avg = statistics.fmean([q for _, q in self.short_term])
        long_avg = statistics.fmean([q for _, q in self.long_term])
        
        diff = short_avg - long_avg
        if diff > 0.10:
            return "improving"
        elif diff < -0.10:
            return "declining"
        return "stable"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GovernanceEvent:
    """Structured event for compliance logging and post-incident analysis."""
    timestamp: float
    event_type: str  # "level_change" | "gate_applied" | "fallback_used" | "freshness_violation"
    from_level: CommunicationTrustLevel
    to_level: CommunicationTrustLevel
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
    """Context-aware policy adjustments based on operational scenario."""
    
    SCENARIOS: Dict[str, Dict[str, Any]] = {
        "normal_operation": {
            "max_level": CommunicationTrustLevel.FULL_TRUST,
            "min_confidence": 0.85,
            "max_latency_ms": 100.0,
            "max_loss_rate": 0.01,
        },
        "high_mobility": {
            "max_level": CommunicationTrustLevel.CONSTRAINED_TRUST,
            "min_confidence": 0.88,
            "max_latency_ms": 200.0,
            "max_loss_rate": 0.05,
        },
        "emergency_mode": {
            "max_level": CommunicationTrustLevel.CONSTRAINED_TRUST,
            "min_confidence": 0.80,  # More tolerant
            "max_latency_ms": 500.0,
            "max_loss_rate": 0.10,
            "require_redundancy": True,
        },
        "degraded_network": {
            "max_level": CommunicationTrustLevel.ADVISORY_TRUST,
            "min_confidence": 0.90,  # Stricter
            "max_latency_ms": 300.0,
            "max_loss_rate": 0.08,
        },
        "satellite_link": {
            "max_level": CommunicationTrustLevel.CONSTRAINED_TRUST,
            "min_confidence": 0.85,
            "max_latency_ms": 600.0,  # Higher tolerance
            "max_loss_rate": 0.03,
        },
        "unknown": {
            "max_level": CommunicationTrustLevel.ADVISORY_TRUST,
            "min_confidence": 0.95,
            "max_latency_ms": 150.0,
            "max_loss_rate": 0.02,
        },
    }
    
    @classmethod
    def get_max_level_for_scenario(cls, scenario: Optional[str]) -> CommunicationTrustLevel:
        """Get maximum trust level for scenario."""
        if scenario is None:
            scenario = "unknown"
        policy = cls.SCENARIOS.get(scenario, cls.SCENARIOS["unknown"])
        return policy["max_level"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance Policy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class TelecomGovernancePolicy:
    """Domain policy for telecommunications governance."""
    # Message freshness
    max_message_age_ms: float = 1000.0
    
    # Link quality thresholds
    max_latency_ms_for_trust: float = 200.0
    max_latency_ms_for_full_trust: float = 50.0
    max_jitter_ms: float = 20.0
    max_packet_loss_rate: float = 0.05
    
    # System health thresholds
    max_network_interface_errors: int = 0
    max_buffer_utilization: float = 0.90
    max_processing_latency_ms: float = 100.0
    max_clock_sync_offset_ms: float = 50.0
    
    # Hysteresis
    min_seconds_between_escalations: float = 10.0
    min_seconds_between_downgrades: float = 0.0  # Immediate downgrades allowed
    min_seconds_at_full_trust: float = 5.0  # Minimum dwell time at FULL_TRUST (noisy links)
    
    # Deployment cap
    max_allowed_level: CommunicationTrustLevel = CommunicationTrustLevel.FULL_TRUST
    
    # Event logging
    max_event_log_size: int = 1000
    
    # Predictive warnings
    enable_predictive_warnings: bool = True
    quality_decline_warning_enabled: bool = True
    
    # Scenario policies
    enable_scenario_policies: bool = True


def default_telecom_config() -> "AileeConfig":
    """Safe defaults for telecommunications governance pipeline configuration."""
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    return AileeConfig(
        accept_threshold=0.88,
        borderline_low=0.75,
        borderline_high=0.88,
        w_stability=0.50,
        w_agreement=0.35,
        w_likelihood=0.15,
        history_window=100,
        forecast_window=15,
        grace_peer_delta=1.0,
        grace_min_peer_agreement_ratio=0.65,
        grace_forecast_epsilon=0.25,
        grace_max_abs_z=2.5,
        consensus_quorum=2,
        consensus_delta=1.0,
        consensus_pass_ratio=0.70,
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
# Telecom Governor (Main Controller)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TelecomGovernor:
    """Production-grade governance controller for telecommunications systems."""
    
    def __init__(
        self,
        cfg: Optional["AileeConfig"] = None,
        policy: Optional[TelecomGovernancePolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable")
        
        self.cfg = cfg or default_telecom_config()
        self.policy = policy or TelecomGovernancePolicy()
        
        self.cfg.hard_max = float(int(self.policy.max_allowed_level))
        self.cfg.fallback_clamp_max = float(int(self.policy.max_allowed_level))
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        
        self._last_level: CommunicationTrustLevel = CommunicationTrustLevel.NO_TRUST
        self._last_change_ts: float = 0.0
        self._last_full_trust_ts: Optional[float] = None  # Track time entered FULL_TRUST
        self._quality_tracker = ConfidenceTracker()
        self._event_log: List[GovernanceEvent] = []
        self._last_event: Optional[GovernanceEvent] = None
        self._last_warning: Optional[Tuple[CommunicationTrustLevel, float, str]] = None
    
    def evaluate(self, signals: TelecomSignals) -> Tuple[CommunicationTrustLevel, "DecisionResult"]:
        """
        Evaluate communication trust authorization decision.
        
        Args:
            signals: Current telecom signals
        
        Returns:
            (authorized_level, decision_result)
        """
        ts = float(signals.timestamp if signals.timestamp is not None else time.time())
        
        # Compute link quality score if available
        link_quality_score = 0.85  # Default
        if signals.link_quality is not None:
            link_quality_score = self._compute_link_quality_score(signals.link_quality)
        
        self._quality_tracker.update(ts, link_quality_score)
        
        gated_level, gate_reasons = self._apply_governance_gates(signals, ts)
        hysteresis_level, hysteresis_reasons = self._apply_hysteresis(ts, gated_level)
        
        if self.policy.enable_predictive_warnings:
            warning = self._predict_required_downgrade(signals, ts)
            if warning is not None:
                self._last_warning = warning
        
        peer_values = list(signals.redundant_path_values)
        
        ctx = {
            "domain": "telecommunications",
            "signal": "communication_trust_level",
            "desired_level": int(signals.desired_level),
            "gated_level": int(gated_level),
            "gate_reasons": gate_reasons,
            "hysteresis_reasons": hysteresis_reasons,
            "scenario": signals.current_scenario,
            "quality_trend": self._quality_tracker.get_trend(),
            "message_age_ms": signals.message_age_ms,
        }
        
        res = self.pipeline.process(
            raw_value=float(int(hysteresis_level)),
            raw_confidence=float(signals.message_confidence) if signals.message_confidence is not None else None,
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        authorized_level = _level_from_float(res.value)
        
        self._log_governance_event(
            ts,
            signals,
            authorized_level,
            res,
            gate_reasons,
            hysteresis_reasons,
        )
        
        self._commit_state(ts, authorized_level, res)
        
        return authorized_level, res
    
    def get_last_event(self) -> Optional[GovernanceEvent]:
        """Get most recent governance event."""
        return self._last_event
    
    def get_event_log(self) -> List[GovernanceEvent]:
        """Get full event log for compliance export."""
        return list(self._event_log)
    
    def get_quality_trend(self) -> str:
        """Get current quality trend for monitoring."""
        return self._quality_tracker.get_trend()
    
    def _compute_link_quality_score(self, link_quality: LinkQualitySignals) -> float:
        """Compute overall link quality score from metrics."""
        score = 1.0
        
        # Latency penalty
        if link_quality.latency_ms is not None:
            if link_quality.latency_ms > 200.0:
                score *= 0.5
            elif link_quality.latency_ms > 100.0:
                score *= 0.7
            elif link_quality.latency_ms > 50.0:
                score *= 0.9
        
        # Packet loss penalty
        if link_quality.packet_loss_rate is not None:
            score *= (1.0 - link_quality.packet_loss_rate)
        
        # Jitter penalty
        if link_quality.jitter_ms is not None:
            if link_quality.jitter_ms > 20.0:
                score *= 0.6
            elif link_quality.jitter_ms > 10.0:
                score *= 0.8
        
        # Stability bonus
        if link_quality.link_stability_score is not None:
            score = (score + link_quality.link_stability_score) / 2.0
        
        return max(0.0, min(1.0, score))
    
    def _apply_governance_gates(
        self, signals: TelecomSignals, ts: float
    ) -> Tuple[CommunicationTrustLevel, List[str]]:
        """Apply deterministic governance gates."""
        reasons: List[str] = []
        level = signals.desired_level
        
        # Deployment cap
        if level > self.policy.max_allowed_level:
            reasons.append(f"Policy cap to {int(self.policy.max_allowed_level)}")
            level = self.policy.max_allowed_level
        
        # Message freshness check
        if signals.message_age_ms is not None:
            if signals.message_age_ms > self.policy.max_message_age_ms:
                reasons.append(
                    f"Message age {signals.message_age_ms:.0f}ms exceeds "
                    f"{self.policy.max_message_age_ms:.0f}ms"
                )
                level = min(level, CommunicationTrustLevel.ADVISORY_TRUST)
        
        # Network domain
        if signals.network_domain is not None:
            max_level = signals.network_domain.max_safe_level()
            if level > max_level:
                reasons.append(f"Network domain limits to {int(max_level)}")
                level = max_level
        
        # Link quality
        if signals.link_quality is not None:
            safe, issues = signals.link_quality.is_safe_for_level(level)
            if not safe:
                for test_level in reversed(list(CommunicationTrustLevel)):
                    if test_level < level:
                        ok, _ = signals.link_quality.is_safe_for_level(test_level)
                        if ok:
                            reasons.append(
                                f"Link quality blocks {int(level)}, cap to {int(test_level)}: {issues}"
                            )
                            level = test_level
                            break
                else:
                    reasons.append(f"Link quality forces NO_TRUST: {issues}")
                    level = CommunicationTrustLevel.NO_TRUST
        
        # Scenario policy
        if self.policy.enable_scenario_policies and signals.current_scenario:
            scenario_max = ScenarioPolicy.get_max_level_for_scenario(signals.current_scenario)
            if level > scenario_max:
                reasons.append(f"Scenario '{signals.current_scenario}' caps to {int(scenario_max)}")
                level = scenario_max
        
        # System health
        if signals.system_health is not None:
            h = signals.system_health
            
            if h.network_interface_errors > self.policy.max_network_interface_errors:
                reasons.append(
                    f"Network errors={h.network_interface_errors} → NO_TRUST"
                )
                return CommunicationTrustLevel.NO_TRUST, reasons
            
            if h.buffer_utilization is not None:
                if h.buffer_utilization > self.policy.max_buffer_utilization:
                    reasons.append(
                        f"Buffer utilization={h.buffer_utilization:.2f} → cap to ADVISORY"
                    )
                    level = min(level, CommunicationTrustLevel.ADVISORY_TRUST)
            
            if h.processing_latency_ms is not None:
                if h.processing_latency_ms > self.policy.max_processing_latency_ms:
                    reasons.append(
                        f"Processing latency={h.processing_latency_ms:.0f}ms too high → downgrade"
                    )
                    level = min(level, CommunicationTrustLevel.CONSTRAINED_TRUST)
            
            if h.clock_sync_offset_ms is not None:
                if h.clock_sync_offset_ms > self.policy.max_clock_sync_offset_ms:
                    reasons.append(
                        f"Clock sync offset={h.clock_sync_offset_ms:.0f}ms → downgrade"
                    )
                    level = min(level, CommunicationTrustLevel.CONSTRAINED_TRUST)
        
        # Redundancy requirements
        if level >= CommunicationTrustLevel.FULL_TRUST:
            if signals.redundancy_state is not None:
                if signals.redundancy_state.active_paths < 2:
                    reasons.append("Full trust requires ≥2 active paths")
                    level = CommunicationTrustLevel.CONSTRAINED_TRUST
                elif signals.redundancy_state.path_agreement_score is not None:
                    if signals.redundancy_state.path_agreement_score < 0.90:
                        reasons.append(
                            f"Path agreement={signals.redundancy_state.path_agreement_score:.2f} "
                            "insufficient for full trust"
                        )
                        level = CommunicationTrustLevel.CONSTRAINED_TRUST
        
        return level, reasons
    
    def _apply_hysteresis(
        self, ts: float, requested: CommunicationTrustLevel
    ) -> Tuple[CommunicationTrustLevel, List[str]]:
        """Apply hysteresis to prevent trust thrashing."""
        reasons: List[str] = []
        current = self._last_level
        
        if requested == current:
            return requested, reasons
        
        dt = ts - self._last_change_ts
        
        # Escalations are rate-limited
        if requested > current:
            if dt < self.policy.min_seconds_between_escalations:
                reasons.append(
                    f"Escalation blocked (dt={dt:.1f}s < "
                    f"{self.policy.min_seconds_between_escalations:.1f}s)"
                )
                return current, reasons
        
        # Downgrades from FULL_TRUST: enforce minimum dwell time in noisy environments
        if current == CommunicationTrustLevel.FULL_TRUST and requested < current:
            if self._last_full_trust_ts is not None:
                time_at_full_trust = ts - self._last_full_trust_ts
                if time_at_full_trust < self.policy.min_seconds_at_full_trust:
                    reasons.append(
                        f"FULL_TRUST dwell time {time_at_full_trust:.1f}s < "
                        f"{self.policy.min_seconds_at_full_trust:.1f}s (prevents noise-induced thrashing)"
                    )
                    return current, reasons
        
        # All other downgrades: immediate (safety-first)
        if requested < current and dt < self.policy.min_seconds_between_downgrades:
            reasons.append(
                f"Downgrade blocked (dt={dt:.1f}s < "
                f"{self.policy.min_seconds_between_downgrades:.1f}s)"
            )
            return current, reasons
        
        return requested, reasons
    
    def _predict_required_downgrade(
        self, signals: TelecomSignals, ts: float
    ) -> Optional[Tuple[CommunicationTrustLevel, float, str]]:
        """Predict if downgrade will be needed soon."""
        if self.policy.quality_decline_warning_enabled:
            declining, reason = self._quality_tracker.is_quality_declining()
            if declining:
                return (
                    CommunicationTrustLevel.ADVISORY_TRUST,
                    ts + 15.0,
                    reason
                )
        
        if signals.network_domain is not None:
            if signals.network_domain.congestion_severity == "elevated":
                return (
                    CommunicationTrustLevel.CONSTRAINED_TRUST,
                    ts + 30.0,
                    "Network congestion increasing",
                )
        
        return None
    
    def _log_governance_event(
        self,
        ts: float,
        signals: TelecomSignals,
        authorized_level: CommunicationTrustLevel,
        res: DecisionResult,
        gate_reasons: List[str],
        hysteresis_reasons: List[str],
    ) -> None:
        """Log governance event."""
        event_type = "evaluation"
        if authorized_level != self._last_level:
            event_type = "level_change"
        elif res.used_fallback:
            event_type = "fallback_used"
        elif gate_reasons:
            event_type = "gate_applied"
        
        if signals.message_age_ms is not None:
            if signals.message_age_ms > self.policy.max_message_age_ms:
                event_type = "freshness_violation"
        
        event = GovernanceEvent(
            timestamp=ts,
            event_type=event_type,
            from_level=self._last_level,
            to_level=authorized_level,
            confidence=signals.message_confidence or 0.0,
            reasons=res.reasons + gate_reasons + hysteresis_reasons,
            metadata=dict(res.metadata) if res.metadata else {},
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
        self, ts: float, authorized_level: CommunicationTrustLevel, res: DecisionResult
    ) -> None:
        """Commit governance state."""
        accepted = not res.used_fallback
        if SafetyStatus is not None:
            accepted = accepted and res.safety_status in (
                SafetyStatus.ACCEPTED,
                SafetyStatus.BORDERLINE,
            )
        
        if accepted and authorized_level != self._last_level:
            # Store previous level before updating
            prev_level = self._last_level
            
            self._last_level = authorized_level
            self._last_change_ts = ts
            
            # Track entry to FULL_TRUST for dwell time enforcement
            if authorized_level == CommunicationTrustLevel.FULL_TRUST:
                self._last_full_trust_ts = ts
            elif prev_level == CommunicationTrustLevel.FULL_TRUST:
                # Exiting FULL_TRUST, clear timestamp
                self._last_full_trust_ts = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_default_governor(
    max_level: CommunicationTrustLevel = CommunicationTrustLevel.CONSTRAINED_TRUST,
    max_message_age_ms: float = 500.0,
) -> TelecomGovernor:
    """Create a telecom governor with safe, production-ready defaults."""
    policy = TelecomGovernancePolicy(
        max_allowed_level=max_level,
        max_message_age_ms=max_message_age_ms,
    )
    cfg = default_telecom_config()
    return TelecomGovernor(cfg=cfg, policy=policy)


def create_example_signals() -> TelecomSignals:
    """Create example signals for testing."""
    return TelecomSignals(
        desired_level=CommunicationTrustLevel.CONSTRAINED_TRUST,
        message_confidence=0.88,
        redundant_path_values=(0.85, 0.87, 0.89),
        link_quality=LinkQualitySignals(
            latency_ms=45.0,
            jitter_ms=8.0,
            packet_loss_rate=0.02,
            link_stability_score=0.92,
        ),
        network_domain=NetworkOperationalDomain(),
        system_health=SystemHealth(
            processing_latency_ms=15.0,
            network_interface_errors=0,
        ),
        redundancy_state=RedundancyState(
            active_paths=3,
            path_agreement_score=0.95,
            failover_ready=True,
        ),
        message_age_ms=145.0,
        current_scenario="normal_operation",
        timestamp=time.time(),
    )


def create_degraded_signals(
    connectivity_mode: str = "degraded",
    latency_ms: float = 250.0,
    packet_loss_rate: float = 0.06,
) -> TelecomSignals:
    """Create degraded network scenario signals for testing."""
    return TelecomSignals(
        desired_level=CommunicationTrustLevel.FULL_TRUST,
        message_confidence=0.75,
        link_quality=LinkQualitySignals(
            latency_ms=latency_ms,
            jitter_ms=25.0,
            packet_loss_rate=packet_loss_rate,
            link_stability_score=0.65,
        ),
        network_domain=NetworkOperationalDomain(
            connectivity_mode=connectivity_mode,
            congestion_severity="elevated",
        ),
        system_health=SystemHealth(processing_latency_ms=85.0),
        message_age_ms=450.0,
        timestamp=time.time(),
    )


def validate_signals(signals: TelecomSignals) -> Tuple[bool, List[str]]:
    """Validate telecom signals structure."""
    errors: List[str] = []
    
    # Check desired_level type and bounds
    if not isinstance(signals.desired_level, CommunicationTrustLevel):
        errors.append("desired_level must be CommunicationTrustLevel enum")
    else:
        try:
            # Verify it's a valid enum value
            _ = int(signals.desired_level)
        except (ValueError, TypeError):
            errors.append(f"desired_level has invalid value: {signals.desired_level}")
    
    # Check message_confidence bounds
    if signals.message_confidence is not None:
        if not (0.0 <= signals.message_confidence <= 1.0):
            errors.append(f"message_confidence={signals.message_confidence} outside [0, 1]")
    
    # Check message_age_ms
    if signals.message_age_ms is not None:
        if signals.message_age_ms < 0:
            errors.append(f"message_age_ms={signals.message_age_ms} is negative")
    
    # Check link_quality scores if present
    if signals.link_quality is not None:
        lq = signals.link_quality
        if lq.packet_loss_rate is not None and not (0.0 <= lq.packet_loss_rate <= 1.0):
            errors.append(f"link_quality.packet_loss_rate={lq.packet_loss_rate} outside [0, 1]")
        if lq.congestion_level is not None and not (0.0 <= lq.congestion_level <= 1.0):
            errors.append(f"link_quality.congestion_level={lq.congestion_level} outside [0, 1]")
        if lq.link_stability_score is not None and not (0.0 <= lq.link_stability_score <= 1.0):
            errors.append(f"link_quality.link_stability_score={lq.link_stability_score} outside [0, 1]")
    
    # Check redundancy_state scores if present
    if signals.redundancy_state is not None:
        rs = signals.redundancy_state
        if rs.path_agreement_score is not None and not (0.0 <= rs.path_agreement_score <= 1.0):
            errors.append(f"redundancy_state.path_agreement_score={rs.path_agreement_score} outside [0, 1]")
        if rs.primary_path_health is not None and not (0.0 <= rs.primary_path_health <= 1.0):
            errors.append(f"redundancy_state.primary_path_health={rs.primary_path_health} outside [0, 1]")
    
    # Check system_health scores if present
    if signals.system_health is not None:
        sh = signals.system_health
        if sh.buffer_utilization is not None and not (0.0 <= sh.buffer_utilization <= 1.0):
            errors.append(f"system_health.buffer_utilization={sh.buffer_utilization} outside [0, 1]")
        if sh.cpu_load is not None and not (0.0 <= sh.cpu_load <= 1.0):
            errors.append(f"system_health.cpu_load={sh.cpu_load} outside [0, 1]")
    
    return len(errors) == 0, errors


def export_events_to_dict(events: List[GovernanceEvent]) -> List[Dict[str, Any]]:
    """Export events to JSON-serializable format."""
    return [
        {
            "timestamp": e.timestamp,
            "event_type": e.event_type,
            "from_level": int(e.from_level),
            "to_level": int(e.to_level),
            "confidence": e.confidence,
            "reasons": e.reasons,
            "metadata": e.metadata,
            "safety_status": e.safety_status,
            "used_fallback": e.used_fallback,
        }
        for e in events
    ]


__version__ = "2.0.0"
__all__ = [
    "CommunicationTrustLevel",
    "TrustLevel",  # Alias for cross-domain consistency
    "TelecomSignals",
    "LinkQualitySignals",
    "NetworkOperationalDomain",
    "SystemHealth",
    "RedundancyState",
    "GovernanceEvent",
    "TelecomGovernancePolicy",
    "ScenarioPolicy",
    "TelecomGovernor",
    "ConfidenceTracker",
    "default_telecom_config",
    "create_default_governor",
    "create_example_signals",
    "create_degraded_signals",
    "validate_signals",
    "export_events_to_dict",
]


if __name__ == "__main__":
    print("=" * 80)
    print("AILEE Telecommunications Governance System - Demo")
    print("=" * 80)
    print()
    
    print("Creating governor...")
    governor = create_default_governor()
    print(f"✓ Governor created")
    print()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Good conditions - normal operation",
            "signals": create_example_signals(),
        },
        {
            "name": "Degraded network - high latency",
            "signals": create_degraded_signals(
                connectivity_mode="degraded",
                latency_ms=350.0,
                packet_loss_rate=0.08,
            ),
        },
        {
            "name": "Intermittent connectivity",
            "signals": TelecomSignals(
                desired_level=CommunicationTrustLevel.FULL_TRUST,
                message_confidence=0.70,
                link_quality=LinkQualitySignals(
                    latency_ms=180.0,
                    packet_loss_rate=0.12,
                    link_stability_score=0.50,
                ),
                network_domain=NetworkOperationalDomain(
                    connectivity_mode="intermittent",
                ),
                message_age_ms=800.0,
                timestamp=time.time(),
            ),
        },
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"--- Scenario {i+1}: {scenario['name']} ---")
        
        signals = scenario['signals']
        level, decision = governor.evaluate(signals)
        
        print(f"  Desired: {signals.desired_level.name}")
        print(f"  Authorized: {level.name}")
        print(f"  Confidence: {decision.confidence_score:.2f}")
        print(f"  Used fallback: {decision.used_fallback}")
        if decision.reasons:
            print(f"  Reasons:")
            for reason in decision.reasons[:3]:
                print(f"    - {reason}")
        print()
    
    # Show event log
    print(f"--- Event Log ({len(governor.get_event_log())} events) ---")
    for event in governor.get_event_log()[-3:]:
        print(f"[{event.timestamp:.2f}] {event.event_type}: {event.from_level.name}→{event.to_level.name}")
    
    print()
    print("=" * 80)
    print("Demo complete.")
    print("=" * 80)
