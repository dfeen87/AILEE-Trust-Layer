"""
AILEE Trust Layer — Crypto Mining Domain
Version: 2.0.0

First-class AILEE domain implementation for crypto mining decision integrity.

This domain applies the AILEE Trust Pipeline to:
- Hash rate optimization and tuning
- Thermal management and throttling
- Power consumption governance
- Mining pool selection and switching
- Hardware maintenance gating

The module defines domain-specific configurations, telemetry processing,
governed controllers, and a unified governance interface for production
crypto mining operations.

Quick Start:
    >>> from ailee.domains.crypto_mining import (
    ...     MiningGovernor,
    ...     MiningPolicy,
    ...     MiningSignals,
    ...     CryptoMiningTrustLevel,
    ...     MiningDomain,
    ...     MiningAction,
    ...     HardwareReading,
    ...     create_mining_governor,
    ... )
    >>>
    >>> governor = create_mining_governor()
    >>>
    >>> signals = MiningSignals(
    ...     mining_domain=MiningDomain.HASH_RATE,
    ...     proposed_action=MiningAction.TUNE_HASH_RATE,
    ...     ai_value=95.0,
    ...     ai_confidence=0.88,
    ...     hardware_readings=[
    ...         HardwareReading(93.5, time.time(), "rig_01"),
    ...         HardwareReading(94.2, time.time(), "rig_02"),
    ...     ],
    ...     rig_id="rig_01",
    ... )
    >>>
    >>> decision = governor.evaluate(signals)
    >>> if decision.actionable:
    ...     miner.set_hash_rate(decision.trusted_value)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import time


_SECONDS_PER_HOUR: int = 3600


# Import core AILEE components
from ...ailee_trust_pipeline_v1 import AileeTrustPipeline, AileeConfig, DecisionResult
from ...optional.ailee_config_presets import SENSOR_FUSION, TEMPERATURE_MONITORING
from ...optional.ailee_monitors import TrustMonitor


# ===========================
# Crypto Mining Specific Configs
# ===========================

HASH_RATE_OPTIMIZATION = AileeConfig(
    # Moderate thresholds: hash rate changes have manageable risk
    accept_threshold=0.90,
    borderline_low=0.75,
    borderline_high=0.90,

    # Hash rate (MH/s) — no hard bounds enforced here; set per-rig via policy
    hard_min=None,
    hard_max=None,

    # Require agreement from multiple rig sensors / peer models
    consensus_quorum=3,
    consensus_delta=2.0,  # 2 MH/s tolerance

    grace_peer_delta=3.0,
    grace_forecast_epsilon=0.10,

    fallback_mode="last_good",

    # Moderate stability weight: hash rate systems respond quickly
    w_stability=0.40,
    w_agreement=0.35,
    w_likelihood=0.25,

    history_window=60,
    forecast_window=10,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


THERMAL_PROTECTION = AileeConfig(
    # High confidence required for thermal safety — hardware damage risk
    accept_threshold=0.95,
    borderline_low=0.85,
    borderline_high=0.95,

    # Temperature (°C): enforce hard bounds
    hard_min=0.0,
    hard_max=95.0,  # Never allow above 95 °C

    # Require many sensors in consensus
    consensus_quorum=4,
    consensus_delta=2.0,  # 2 °C tolerance

    grace_peer_delta=1.5,
    grace_forecast_epsilon=0.08,

    fallback_mode="last_good",

    # Heavy stability weight: thermal systems have inertia
    w_stability=0.55,
    w_agreement=0.25,
    w_likelihood=0.20,

    history_window=120,
    forecast_window=20,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


POOL_SWITCHING = AileeConfig(
    # Moderate thresholds for pool selection (business-level decisions)
    accept_threshold=0.88,
    borderline_low=0.72,
    borderline_high=0.88,

    # Pool latency score [0, 1]: no numeric hard bounds
    hard_min=0.0,
    hard_max=1.0,

    # Multiple profitability / latency estimates for consensus
    consensus_quorum=3,
    consensus_delta=0.05,

    grace_peer_delta=0.08,
    grace_forecast_epsilon=0.12,

    fallback_mode="last_good",

    w_stability=0.35,
    w_agreement=0.40,
    w_likelihood=0.25,

    history_window=50,
    forecast_window=10,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


POWER_MANAGEMENT = AileeConfig(
    # High confidence required for power changes
    accept_threshold=0.93,
    borderline_low=0.80,
    borderline_high=0.93,

    # Power (watts) — bounds set per-rig via policy
    hard_min=None,
    hard_max=None,

    consensus_quorum=3,
    consensus_delta=10.0,  # 10 W tolerance

    grace_peer_delta=15.0,
    grace_forecast_epsilon=0.08,

    fallback_mode="last_good",

    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,

    history_window=80,
    forecast_window=15,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


# ===========================
# Enumerations
# ===========================

class CryptoMiningTrustLevel(IntEnum):
    """
    Graduated trust levels for crypto mining control decisions.
    Higher values indicate more authority to act autonomously.
    """
    NO_ACTION = 0    # Insufficient data or confidence — do not act
    ADVISORY = 1     # Log or alert only — no hardware change
    SUPERVISED = 2   # Act, but flag for operator review
    AUTONOMOUS = 3   # Fully authorized autonomous action


class MiningOperationStatus(str, Enum):
    """Overall health/operational status of a mining rig or fleet."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class MiningDomain(str, Enum):
    """Crypto mining control domains governed by AILEE."""
    HASH_RATE = "HASH_RATE"
    THERMAL = "THERMAL"
    POWER = "POWER"
    POOL_MANAGEMENT = "POOL_MANAGEMENT"
    HARDWARE = "HARDWARE"


class MiningAction(str, Enum):
    """Types of mining control actions that can be governed."""
    TUNE_HASH_RATE = "TUNE_HASH_RATE"
    ADJUST_POWER_LIMIT = "ADJUST_POWER_LIMIT"
    SWITCH_POOL = "SWITCH_POOL"
    THERMAL_THROTTLE = "THERMAL_THROTTLE"
    HARDWARE_RESTART = "HARDWARE_RESTART"
    NO_ACTION = "NO_ACTION"


# ===========================
# Policy & Signal Types
# ===========================

@dataclass
class MiningPolicy:
    """
    Governance policy for crypto mining control decisions.

    Controls trust thresholds, safety constraints, and audit settings
    applied by the MiningGovernor.
    """
    # Minimum trust level required before an action is permitted
    min_trust_for_action: CryptoMiningTrustLevel = CryptoMiningTrustLevel.SUPERVISED

    # Require peer hardware sensor consensus before accepting a value
    require_consensus: bool = True

    # Validate hardware readings are within expected range before trusting them
    require_hardware_validation: bool = True

    # Maximum allowed hash rate change per governance cycle (percent of current)
    max_hash_rate_change_pct: float = 10.0

    # Maximum pool switches permitted per hour (prevents pool-thrashing)
    max_pool_switches_per_hour: int = 5

    # Temperature (°C) at or above which thermal throttling is always required
    thermal_throttle_temp_c: float = 85.0

    # Per-rig power limit (watts); used to derive hard bounds when set
    max_power_watts: Optional[float] = None

    # Audit settings
    enable_audit_events: bool = True
    track_decision_history: bool = True


@dataclass
class HardwareReading:
    """Standard hardware sensor reading format for mining rigs."""
    value: float
    timestamp: float
    hardware_id: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MiningSignals:
    """
    Input signals for a crypto mining governance evaluation.

    Describes the AI-proposed control action together with supporting
    hardware telemetry and contextual metadata.
    """
    # What is being controlled and what action is proposed
    mining_domain: MiningDomain
    proposed_action: MiningAction

    # AI-produced value (hash rate MH/s, temperature °C, power W, pool score 0-1)
    ai_value: float

    # AI confidence in the proposed value [0.0, 1.0]
    ai_confidence: float

    # Supporting hardware readings used for consensus
    hardware_readings: List[HardwareReading] = field(default_factory=list)

    # Contextual identifiers
    rig_id: str = "default"
    pool_url: Optional[str] = None
    current_pool_url: Optional[str] = None

    # Evaluation timestamp (defaults to current time)
    timestamp: Optional[float] = None

    # Additional free-form context passed through to the audit trail
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MiningDecision:
    """
    Result of a crypto mining governance evaluation.

    Carries the authorized trust level, the validated/fallback value to
    act on, and a full audit trail suitable for mining management system
    integration.
    """
    # Governance outcome
    authorized_level: CryptoMiningTrustLevel
    actionable: bool

    # Trusted value to use for actuation (hash rate, temperature threshold, power cap, pool score)
    trusted_value: float

    # What was evaluated
    mining_domain: MiningDomain
    proposed_action: MiningAction

    # Raw AILEE pipeline result (DecisionResult or None)
    pipeline_result: Optional[Any] = None

    # Safety and health
    operation_status: MiningOperationStatus = MiningOperationStatus.HEALTHY
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
class MiningEvent:
    """
    Audit event emitted by the MiningGovernor.

    Provides a structured, time-stamped record of each governance decision
    for logging, compliance, and post-incident analysis.
    """
    event_type: str          # "decision", "fallback", "health_change", "rate_limit", "thermal_override"
    mining_domain: MiningDomain
    timestamp: float = field(default_factory=time.time)
    decision: Optional[MiningDecision] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ===========================
# Unified Governance Interface
# ===========================

class MiningGovernor:
    """
    Unified governance governor for crypto mining AI control systems.

    Evaluates AI-proposed control actions across hash rate, thermal,
    power, and pool management domains using the AILEE Trust Pipeline,
    and returns graduated-trust ``MiningDecision`` objects with full
    audit trails.

    This is the primary entry point for integrating AILEE into mining
    management systems, fleet controllers, or auto-tuning pipelines.

    Example::

        governor = MiningGovernor()

        signals = MiningSignals(
            mining_domain=MiningDomain.HASH_RATE,
            proposed_action=MiningAction.TUNE_HASH_RATE,
            ai_value=95.0,
            ai_confidence=0.88,
            hardware_readings=[
                HardwareReading(93.5, time.time(), "rig_01"),
                HardwareReading(94.2, time.time(), "rig_02"),
            ],
            rig_id="rig_01",
        )

        decision = governor.evaluate(signals)
        if decision.actionable:
            miner.set_hash_rate(decision.trusted_value)
        elif decision.used_fallback:
            logger.warning("Fallback active: %s", decision.fallback_reason)
    """

    def __init__(
        self,
        policy: Optional[MiningPolicy] = None,
        hash_rate_config: Optional[AileeConfig] = None,
        thermal_config: Optional[AileeConfig] = None,
        power_config: Optional[AileeConfig] = None,
        pool_config: Optional[AileeConfig] = None,
    ):
        """
        Initialize the MiningGovernor.

        Args:
            policy: Governance policy controlling trust thresholds and constraints.
                    Defaults to ``MiningPolicy()`` (standard configuration).
            hash_rate_config: AILEE pipeline config for hash rate decisions.
                              Defaults to ``HASH_RATE_OPTIMIZATION``.
            thermal_config:   AILEE pipeline config for thermal management decisions.
                              Defaults to ``THERMAL_PROTECTION``.
            power_config:     AILEE pipeline config for power management decisions.
                              Defaults to ``POWER_MANAGEMENT``.
            pool_config:      AILEE pipeline config for pool switching decisions.
                              Defaults to ``POOL_SWITCHING``.
        """
        self.policy = policy or MiningPolicy()

        # One AILEE pipeline per mining domain.
        # HARDWARE shares thermal config (sensor-based, safety-critical).
        self._pipelines: Dict[str, AileeTrustPipeline] = {
            MiningDomain.HASH_RATE: AileeTrustPipeline(
                hash_rate_config or HASH_RATE_OPTIMIZATION
            ),
            MiningDomain.THERMAL: AileeTrustPipeline(
                thermal_config or THERMAL_PROTECTION
            ),
            MiningDomain.POWER: AileeTrustPipeline(
                power_config or POWER_MANAGEMENT
            ),
            MiningDomain.POOL_MANAGEMENT: AileeTrustPipeline(
                pool_config or POOL_SWITCHING
            ),
            MiningDomain.HARDWARE: AileeTrustPipeline(
                thermal_config or THERMAL_PROTECTION
            ),
        }

        # Track metrics and events
        self._monitor = TrustMonitor(window=500)
        self._events: List[MiningEvent] = []
        self._decision_history: List[MiningDecision] = []

        # Pool-switch rate limiting
        self._pool_switches_this_hour: int = 0
        self._last_hour_reset: float = time.time()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def evaluate(self, signals: MiningSignals) -> MiningDecision:
        """
        Evaluate an AI-proposed crypto mining control action.

        Runs the AILEE Trust Pipeline for the appropriate mining domain,
        applies policy constraints (including thermal safety overrides and
        pool-switch rate limiting), determines the trust level, and returns
        a ``MiningDecision`` with a trusted value and full audit trail.

        Args:
            signals: ``MiningSignals`` describing the proposed action
                     and supporting hardware telemetry.

        Returns:
            ``MiningDecision`` with an authorized trust level and the
            value that is safe to act on.
        """
        ts = signals.timestamp or time.time()
        peer_values = [r.value for r in signals.hardware_readings]

        # --- Thermal safety override: always throttle if temp is critical ---
        if signals.mining_domain == MiningDomain.THERMAL and peer_values:
            max_observed_temp = max(peer_values)
            if max_observed_temp >= self.policy.thermal_throttle_temp_c:
                decision = self._build_no_action_decision(
                    signals=signals,
                    ts=ts,
                    reason=(
                        f"Thermal override: observed temperature {max_observed_temp:.1f}°C "
                        f"exceeds throttle threshold {self.policy.thermal_throttle_temp_c:.1f}°C"
                    ),
                    event_type="thermal_override",
                    operation_status=MiningOperationStatus.CRITICAL,
                )
                self._record_event("thermal_override", signals, decision)
                return decision

        # --- Rate-limit pool switches ---
        if signals.mining_domain == MiningDomain.POOL_MANAGEMENT:
            now = time.time()
            if (now - self._last_hour_reset) > _SECONDS_PER_HOUR:
                self._pool_switches_this_hour = 0
                self._last_hour_reset = now
            if self._pool_switches_this_hour >= self.policy.max_pool_switches_per_hour:
                decision = self._build_no_action_decision(
                    signals=signals,
                    ts=ts,
                    reason="Rate limit reached: too many pool switches this hour",
                )
                self._record_event("rate_limit", signals, decision)
                return decision

        # --- Run AILEE pipeline ---
        pipeline = self._pipelines[signals.mining_domain]
        context = dict(signals.context)
        context.update({
            "rig_id": signals.rig_id,
            "mining_domain": signals.mining_domain.value,
            "proposed_action": signals.proposed_action.value,
        })
        if signals.pool_url:
            context["pool_url"] = signals.pool_url
        if signals.current_pool_url:
            context["current_pool_url"] = signals.current_pool_url

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
            f"{ts}{signals.mining_domain}{signals.ai_value}".encode()
        ).hexdigest()[:16]

        decision = MiningDecision(
            authorized_level=authorized_level,
            actionable=actionable,
            trusted_value=result.value,
            mining_domain=signals.mining_domain,
            proposed_action=signals.proposed_action,
            pipeline_result=result,
            operation_status=self._get_subsystem_health(signals.mining_domain),
            safety_flags=safety_flags,
            used_fallback=result.used_fallback,
            fallback_reason=result.safety_status if result.used_fallback else None,
            timestamp=ts,
            decision_id=decision_id,
            reasons=reasons,
            metadata={
                "confidence_score": result.confidence_score,
                "safety_status": result.safety_status,
                "hardware_reading_count": len(signals.hardware_readings),
            },
        )

        # Track pool switch count
        if (
            signals.mining_domain == MiningDomain.POOL_MANAGEMENT
            and actionable
            and not result.used_fallback
        ):
            self._pool_switches_this_hour += 1

        if self.policy.track_decision_history:
            self._decision_history.append(decision)
        if self.policy.enable_audit_events:
            event_type = "fallback" if result.used_fallback else "decision"
            self._record_event(event_type, signals, decision)

        return decision

    # ------------------------------------------------------------------
    # Health & Monitoring
    # ------------------------------------------------------------------

    def get_health(self) -> MiningOperationStatus:
        """
        Compute the overall operational health status based on recent
        governance metrics.

        Returns:
            ``MiningOperationStatus`` enum value.
        """
        fallback_rate = self._monitor.fallback_rate()
        if fallback_rate > 0.30:
            return MiningOperationStatus.CRITICAL
        if fallback_rate > 0.20:
            return MiningOperationStatus.DEGRADED
        if fallback_rate > 0.10:
            return MiningOperationStatus.WARNING
        return MiningOperationStatus.HEALTHY

    def get_subsystem_health(self) -> Dict[str, MiningOperationStatus]:
        """
        Return health status per mining domain.

        Returns:
            Dict mapping ``MiningDomain`` value strings to
            ``MiningOperationStatus`` values.
        """
        return {
            domain.value: self._get_subsystem_health(domain)
            for domain in MiningDomain
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
            "pool_switches_this_hour": self._pool_switches_this_hour,
            "overall_health": self.get_health().value,
        }

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def get_events(self) -> List[MiningEvent]:
        """Return all recorded governance events (newest last)."""
        return list(self._events)

    def get_decision_history(self) -> List[MiningDecision]:
        """Return the history of all governance decisions."""
        return list(self._decision_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_trust_level(self, result: Any) -> CryptoMiningTrustLevel:
        """Map an AILEE DecisionResult to a CryptoMiningTrustLevel."""
        if result.safety_status == "OUTRIGHT_REJECTED":
            return CryptoMiningTrustLevel.NO_ACTION
        if result.used_fallback:
            return CryptoMiningTrustLevel.ADVISORY
        confidence = getattr(result, "confidence_score", 0.0)
        if result.safety_status == "ACCEPTED" and confidence >= 0.90:
            return CryptoMiningTrustLevel.AUTONOMOUS
        if result.safety_status in ("ACCEPTED", "BORDERLINE") and confidence >= 0.70:
            return CryptoMiningTrustLevel.SUPERVISED
        return CryptoMiningTrustLevel.ADVISORY

    def _get_subsystem_health(self, domain: MiningDomain) -> MiningOperationStatus:
        """Return health status for a single mining domain (uses overall monitor)."""
        return self.get_health()

    def _build_no_action_decision(
        self,
        signals: MiningSignals,
        ts: float,
        reason: str,
        event_type: str = "rate_limit",
        operation_status: MiningOperationStatus = MiningOperationStatus.WARNING,
    ) -> MiningDecision:
        """Build a NO_ACTION decision without running the pipeline."""
        decision_id = hashlib.sha256(
            f"{ts}{signals.mining_domain}{signals.ai_value}".encode()
        ).hexdigest()[:16]
        return MiningDecision(
            authorized_level=CryptoMiningTrustLevel.NO_ACTION,
            actionable=False,
            trusted_value=signals.ai_value,
            mining_domain=signals.mining_domain,
            proposed_action=signals.proposed_action,
            pipeline_result=None,
            operation_status=operation_status,
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
        signals: MiningSignals,
        decision: MiningDecision,
    ) -> None:
        """Append a MiningEvent to the internal log."""
        self._events.append(MiningEvent(
            event_type=event_type,
            mining_domain=signals.mining_domain,
            timestamp=decision.timestamp,
            decision=decision,
            details={
                "rig_id": signals.rig_id,
                "ai_value": signals.ai_value,
                "ai_confidence": signals.ai_confidence,
                "hardware_reading_count": len(signals.hardware_readings),
            },
        ))


# ===========================
# Convenience Factory Functions
# ===========================

def create_mining_governor(
    policy: Optional[MiningPolicy] = None,
    **policy_overrides: Any,
) -> MiningGovernor:
    """
    Create a ``MiningGovernor`` with a sensible default policy.

    Args:
        policy: Optional pre-built ``MiningPolicy``.  If omitted, a
                default policy is constructed from ``policy_overrides``.
        **policy_overrides: Keyword arguments forwarded to
                            ``MiningPolicy()`` if no policy is given.

    Returns:
        Configured ``MiningGovernor`` instance.

    Example::

        governor = create_mining_governor(max_pool_switches_per_hour=3)
    """
    if policy is None:
        policy = MiningPolicy(**policy_overrides)
    return MiningGovernor(policy=policy)


def create_default_governor(**policy_overrides: Any) -> MiningGovernor:
    """
    Create a ``MiningGovernor`` with default (balanced) policy settings.

    Args:
        **policy_overrides: Optional ``MiningPolicy`` attribute overrides.

    Returns:
        ``MiningGovernor`` instance with default policy.

    Example::

        governor = create_default_governor()
    """
    return MiningGovernor(policy=MiningPolicy(**policy_overrides))


def create_strict_governor(**policy_overrides: Any) -> MiningGovernor:
    """
    Create a ``MiningGovernor`` with a strict safety policy.

    Strict settings:
    - Requires AUTONOMOUS trust level before allowing action
    - Enforces consensus and hardware validation
    - Limits hash rate changes to ±5% per cycle
    - Limits pool switches to 2 per hour
    - Lower thermal throttle threshold (80 °C)
    - Enables full audit trail

    Args:
        **policy_overrides: Optional further overrides.

    Returns:
        ``MiningGovernor`` instance with strict policy.

    Example::

        governor = create_strict_governor()
    """
    strict_defaults: Dict[str, Any] = {
        "min_trust_for_action": CryptoMiningTrustLevel.AUTONOMOUS,
        "require_consensus": True,
        "require_hardware_validation": True,
        "max_hash_rate_change_pct": 5.0,
        "max_pool_switches_per_hour": 2,
        "thermal_throttle_temp_c": 80.0,
        "enable_audit_events": True,
        "track_decision_history": True,
    }
    strict_defaults.update(policy_overrides)
    return MiningGovernor(policy=MiningPolicy(**strict_defaults))


def create_permissive_governor(**policy_overrides: Any) -> MiningGovernor:
    """
    Create a ``MiningGovernor`` with a permissive policy for development
    or testing.

    Permissive settings:
    - Allows action at ADVISORY trust level
    - Does not enforce consensus or hardware validation
    - Higher pool-switch and hash-rate-change limits

    **WARNING**: Not recommended for production use.

    Args:
        **policy_overrides: Optional further overrides.

    Returns:
        ``MiningGovernor`` instance with permissive policy.

    Example::

        governor = create_permissive_governor()
    """
    permissive_defaults: Dict[str, Any] = {
        "min_trust_for_action": CryptoMiningTrustLevel.ADVISORY,
        "require_consensus": False,
        "require_hardware_validation": False,
        "max_hash_rate_change_pct": 50.0,
        "max_pool_switches_per_hour": 100,
        "enable_audit_events": False,
        "track_decision_history": False,
    }
    permissive_defaults.update(policy_overrides)
    return MiningGovernor(policy=MiningPolicy(**permissive_defaults))


def validate_mining_signals(signals: MiningSignals) -> List[str]:
    """
    Validate a ``MiningSignals`` object and return a list of issue strings.

    An empty list means the signals are valid.

    Args:
        signals: ``MiningSignals`` instance to validate.

    Returns:
        List of validation error/warning strings.  Empty if all checks pass.

    Example::

        issues = validate_mining_signals(signals)
        if issues:
            for issue in issues:
                logger.warning("Signal issue: %s", issue)
    """
    issues: List[str] = []

    if not (0.0 <= signals.ai_confidence <= 1.0):
        issues.append(
            f"ai_confidence must be in [0.0, 1.0], got {signals.ai_confidence}"
        )

    if signals.mining_domain == MiningDomain.THERMAL:
        if not (-10.0 <= signals.ai_value <= 120.0):
            issues.append(
                f"Thermal value {signals.ai_value}°C looks out of range [-10, 120]"
            )
    elif signals.mining_domain == MiningDomain.POOL_MANAGEMENT:
        if not (0.0 <= signals.ai_value <= 1.0):
            issues.append(
                f"Pool score {signals.ai_value} must be in [0.0, 1.0]"
            )
    elif signals.mining_domain == MiningDomain.HASH_RATE:
        if signals.ai_value < 0.0:
            issues.append(
                f"Hash rate {signals.ai_value} MH/s must be non-negative"
            )
    elif signals.mining_domain == MiningDomain.POWER:
        if signals.ai_value < 0.0:
            issues.append(
                f"Power value {signals.ai_value} W must be non-negative"
            )

    if not signals.hardware_readings:
        issues.append(
            "No hardware readings provided; consensus cannot be computed"
        )

    return issues


# ===========================
# Convenience Exports
# ===========================

__all__ = [
    # Enumerations
    'CryptoMiningTrustLevel',
    'MiningOperationStatus',
    'MiningDomain',
    'MiningAction',

    # Policy & Signal Types
    'MiningPolicy',
    'HardwareReading',
    'MiningSignals',
    'MiningDecision',
    'MiningEvent',

    # Governor
    'MiningGovernor',

    # Factory Functions
    'create_mining_governor',
    'create_default_governor',
    'create_strict_governor',
    'create_permissive_governor',
    'validate_mining_signals',

    # Configurations
    'HASH_RATE_OPTIMIZATION',
    'THERMAL_PROTECTION',
    'POOL_SWITCHING',
    'POWER_MANAGEMENT',
]
