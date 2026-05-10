"""
AILEE Trust Layer — Light Transition Domain
Version: 4.7.1

Deterministic governance for optical, photonic, laser, fiber, free-space, and
light-transition signaling systems.

This domain does not create a faster-than-light transport and does not modify
physical propagation limits. It applies AILEE-Trust-Layer semantics to decide
whether light-carried data signals are trustworthy enough to act on, based on
confidence, peer consensus, freshness, photonic link quality, clock discipline,
and physics-bounded propagation claims.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...ailee_trust_pipeline_v1 import AileeConfig, AileeTrustPipeline, DecisionResult, SafetyStatus

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
STANDARD_ATMOSPHERE_PRESSURE_PA = 101_325.0
STANDARD_ATMOSPHERE_TEMPERATURE_K = 288.15
REFRACTIVE_INDEX_JITTER_BUFFER = 0.005


# ===========================
# AILEE Configurations
# ===========================

PHOTONIC_SIGNAL_INTEGRITY = AileeConfig(
    accept_threshold=0.70,
    borderline_low=0.55,
    borderline_high=0.70,
    hard_min=0.0,
    hard_max=1.0,
    consensus_quorum=3,
    consensus_delta=0.08,
    grace_peer_delta=0.10,
    grace_forecast_epsilon=0.10,
    fallback_mode="last_good",
    w_stability=0.45,
    w_agreement=0.35,
    w_likelihood=0.20,
    history_window=120,
    forecast_window=12,
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)

LIGHT_CLOCK_SYNCHRONIZATION = AileeConfig(
    accept_threshold=0.74,
    borderline_low=0.60,
    borderline_high=0.74,
    hard_min=0.0,
    hard_max=1.0,
    consensus_quorum=4,
    consensus_delta=0.05,
    grace_peer_delta=0.06,
    fallback_mode="last_good",
    w_stability=0.55,
    w_agreement=0.30,
    w_likelihood=0.15,
    history_window=160,
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)

OPTICAL_PATH_RELIABILITY = AileeConfig(
    accept_threshold=0.68,
    borderline_low=0.52,
    borderline_high=0.68,
    hard_min=0.0,
    hard_max=1.0,
    consensus_quorum=2,
    consensus_delta=0.12,
    grace_peer_delta=0.14,
    fallback_mode="median",
    w_stability=0.38,
    w_agreement=0.37,
    w_likelihood=0.25,
    history_window=80,
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


# ===========================
# Enumerations
# ===========================

class LightTransitionTrustLevel(IntEnum):
    """Graduated authority for acting on light-transition data."""
    NO_ACTION = 0
    ADVISORY = 1
    GUARDED = 2
    AUTONOMOUS = 3


class LightTransitionHealthStatus(str, Enum):
    """Overall health status for a light-transition subsystem."""
    OPTIMAL = "OPTIMAL"
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class LightTransitionControlDomain(str, Enum):
    """Light-transition governance domains."""
    PHOTONIC_SIGNAL_INTEGRITY = "PHOTONIC_SIGNAL_INTEGRITY"
    LIGHT_CLOCK_SYNCHRONIZATION = "LIGHT_CLOCK_SYNCHRONIZATION"
    OPTICAL_PATH_RELIABILITY = "OPTICAL_PATH_RELIABILITY"
    DATA_FRAME_ACCEPTANCE = "DATA_FRAME_ACCEPTANCE"
    BEAM_HANDOFF = "BEAM_HANDOFF"


class LightTransitionControlAction(str, Enum):
    """Governed light-transition actions."""
    ACCEPT_FRAME = "ACCEPT_FRAME"
    FORWARD_SIGNAL = "FORWARD_SIGNAL"
    SWITCH_PATH = "SWITCH_PATH"
    RESYNCHRONIZE_CLOCK = "RESYNCHRONIZE_CLOCK"
    REDUCE_RATE = "REDUCE_RATE"
    ENTER_SAFE_MODE = "ENTER_SAFE_MODE"
    NO_ACTION = "NO_ACTION"


class PropagationMedium(str, Enum):
    """Supported light-signal propagation media."""
    VACUUM = "VACUUM"
    FREE_SPACE = "FREE_SPACE"
    FIBER = "FIBER"
    WAVEGUIDE = "WAVEGUIDE"
    INTERCONNECT = "INTERCONNECT"
    UNKNOWN = "UNKNOWN"


REFRACTIVE_INDEX_BY_MEDIUM: Dict[PropagationMedium, float] = {
    PropagationMedium.VACUUM: 1.0000,
    PropagationMedium.FREE_SPACE: 1.0003,
    PropagationMedium.FIBER: 1.467,
    PropagationMedium.WAVEGUIDE: 1.500,
    PropagationMedium.INTERCONNECT: 1.330,
    PropagationMedium.UNKNOWN: 1.0000,
}


# ===========================
# Policy & Signal Types
# ===========================

@dataclass
class LightTransitionPolicy:
    """Policy constraints for light-transition governance."""
    min_trust_for_action: LightTransitionTrustLevel = LightTransitionTrustLevel.GUARDED
    require_consensus: bool = True
    max_signal_age_ns: float = 1_000_000.0
    max_bit_error_rate: float = 1e-9
    min_snr_db: float = 18.0
    max_clock_offset_ps: float = 50.0
    max_propagation_fraction_c: float = 1.0
    min_eye_opening: float = 0.62
    max_dispersion_ps_nm: float = 35.0
    enable_audit_events: bool = True
    track_decision_history: bool = True


@dataclass(frozen=True)
class OpticalReading:
    """Peer or instrument reading for a light-transition signal."""
    value: float
    timestamp: float
    sensor_id: str
    confidence: Optional[float] = None
    wavelength_nm: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LightTransitionSignals:
    """Input payload for a light-transition governance evaluation."""
    control_domain: LightTransitionControlDomain
    proposed_action: LightTransitionControlAction
    ai_value: float
    ai_confidence: float
    optical_readings: List[OpticalReading] = field(default_factory=list)
    medium: PropagationMedium = PropagationMedium.UNKNOWN
    signal_age_ns: Optional[float] = None
    distance_m: Optional[float] = None
    measured_time_of_flight_ns: Optional[float] = None
    bit_error_rate: Optional[float] = None
    snr_db: Optional[float] = None
    clock_offset_ps: Optional[float] = None
    eye_opening: Optional[float] = None
    dispersion_ps_nm: Optional[float] = None
    data_rate_gbps: Optional[float] = None
    wavelength_nm: Optional[float] = None
    environmental_temperature_c: Optional[float] = None
    environmental_pressure_pa: Optional[float] = None
    refractive_index_override: Optional[float] = None
    channel_id: str = "default"
    timestamp: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LightTransitionDecision:
    """Governance result for a light-transition signal."""
    authorized_level: LightTransitionTrustLevel
    actionable: bool
    trusted_value: float
    control_domain: LightTransitionControlDomain
    proposed_action: LightTransitionControlAction
    pipeline_result: Optional[DecisionResult] = None
    health_status: LightTransitionHealthStatus = LightTransitionHealthStatus.OPTIMAL
    safety_flags: List[str] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    decision_id: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LightTransitionEvent:
    """Audit event emitted by the light-transition governor."""
    event_type: str
    control_domain: LightTransitionControlDomain
    timestamp: float = field(default_factory=time.time)
    decision: Optional[LightTransitionDecision] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ===========================
# Helpers
# ===========================

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _config_for_domain(domain: LightTransitionControlDomain) -> AileeConfig:
    if domain == LightTransitionControlDomain.LIGHT_CLOCK_SYNCHRONIZATION:
        return LIGHT_CLOCK_SYNCHRONIZATION
    if domain == LightTransitionControlDomain.OPTICAL_PATH_RELIABILITY:
        return OPTICAL_PATH_RELIABILITY
    return PHOTONIC_SIGNAL_INTEGRITY


def _propagation_fraction_c(distance_m: Optional[float], tof_ns: Optional[float]) -> Optional[float]:
    if distance_m is None or tof_ns is None or tof_ns <= 0.0:
        return None
    measured_velocity = float(distance_m) / (float(tof_ns) * 1e-9)
    return measured_velocity / SPEED_OF_LIGHT_M_PER_S


def _get_refractive_index(
    medium: PropagationMedium,
    temp: Optional[float] = None,
    pressure: Optional[float] = None,
    override_n: Optional[float] = None,
) -> float:
    """Return the deterministic refractive index for a propagation medium."""
    if override_n is not None:
        return float(override_n)

    base_n = REFRACTIVE_INDEX_BY_MEDIUM.get(
        medium,
        REFRACTIVE_INDEX_BY_MEDIUM[PropagationMedium.UNKNOWN],
    )
    if medium != PropagationMedium.FREE_SPACE:
        return base_n
    if temp is None and pressure is None:
        return base_n

    temperature_k = (
        STANDARD_ATMOSPHERE_TEMPERATURE_K
        if temp is None
        else float(temp) + 273.15
    )
    if temperature_k <= 0.0:
        return base_n

    pressure_pa = (
        STANDARD_ATMOSPHERE_PRESSURE_PA
        if pressure is None
        else max(0.0, float(pressure))
    )
    density_ratio = (
        pressure_pa / STANDARD_ATMOSPHERE_PRESSURE_PA
    ) * (STANDARD_ATMOSPHERE_TEMPERATURE_K / temperature_k)
    return 1.0 + ((base_n - 1.0) * density_ratio)


def _get_dynamic_speed_limit(
    medium: PropagationMedium,
    temp: Optional[float] = None,
    pressure: Optional[float] = None,
    override_n: Optional[float] = None,
) -> float:
    """Return the medium-aware maximum propagation fraction of c."""
    refractive_index = _get_refractive_index(
        medium=medium,
        temp=temp,
        pressure=pressure,
        override_n=override_n,
    )
    if refractive_index <= 0.0:
        return 0.0
    return (1.0 / refractive_index) + REFRACTIVE_INDEX_JITTER_BUFFER


def _trust_level_from_result(
    result: DecisionResult,
    flags: Sequence[str],
) -> LightTransitionTrustLevel:
    if flags or result.used_fallback or result.safety_status == SafetyStatus.OUTRIGHT_REJECTED:
        if result.confidence_score >= 0.70 and len(flags) <= 1:
            return LightTransitionTrustLevel.ADVISORY
        return LightTransitionTrustLevel.NO_ACTION
    if result.confidence_score >= 0.90:
        return LightTransitionTrustLevel.AUTONOMOUS
    if result.safety_status == SafetyStatus.ACCEPTED or result.confidence_score >= 0.70:
        return LightTransitionTrustLevel.GUARDED
    if result.confidence_score >= 0.55:
        return LightTransitionTrustLevel.ADVISORY
    return LightTransitionTrustLevel.NO_ACTION


def _health_from_level(
    level: LightTransitionTrustLevel,
    flags: Sequence[str],
    used_fallback: bool,
) -> LightTransitionHealthStatus:
    if level == LightTransitionTrustLevel.NO_ACTION:
        return LightTransitionHealthStatus.CRITICAL if flags else LightTransitionHealthStatus.DEGRADED
    if used_fallback:
        return LightTransitionHealthStatus.WARNING
    if level == LightTransitionTrustLevel.ADVISORY:
        return LightTransitionHealthStatus.WARNING
    if level == LightTransitionTrustLevel.GUARDED:
        return LightTransitionHealthStatus.HEALTHY
    return LightTransitionHealthStatus.OPTIMAL


def validate_light_transition_signals(signals: LightTransitionSignals) -> Tuple[bool, List[str]]:
    """Validate signal structure and normalized ranges."""
    issues: List[str] = []
    if not 0.0 <= signals.ai_value <= 1.0:
        issues.append("ai_value must be in [0.0, 1.0]")
    if not 0.0 <= signals.ai_confidence <= 1.0:
        issues.append("ai_confidence must be in [0.0, 1.0]")
    for reading in signals.optical_readings:
        if not 0.0 <= reading.value <= 1.0:
            issues.append(f"reading {reading.sensor_id} value must be in [0.0, 1.0]")
    if signals.signal_age_ns is not None and signals.signal_age_ns < 0.0:
        issues.append("signal_age_ns must be non-negative")
    if signals.bit_error_rate is not None and signals.bit_error_rate < 0.0:
        issues.append("bit_error_rate must be non-negative")
    if signals.distance_m is not None and signals.distance_m < 0.0:
        issues.append("distance_m must be non-negative")
    if signals.measured_time_of_flight_ns is not None and signals.measured_time_of_flight_ns <= 0.0:
        issues.append("measured_time_of_flight_ns must be positive")
    if signals.environmental_pressure_pa is not None and signals.environmental_pressure_pa < 0.0:
        issues.append("environmental_pressure_pa must be non-negative")
    if signals.refractive_index_override is not None and signals.refractive_index_override <= 0.0:
        issues.append("refractive_index_override must be positive")
    return len(issues) == 0, issues


validate_signals = validate_light_transition_signals


# ===========================
# Governor
# ===========================

class LightTransitionGovernor:
    """AILEE-backed governor for light-carried data signal decisions."""

    def __init__(self, policy: Optional[LightTransitionPolicy] = None):
        self.policy = policy or LightTransitionPolicy()
        self.pipelines: Dict[LightTransitionControlDomain, AileeTrustPipeline] = {
            domain: AileeTrustPipeline(_config_for_domain(domain))
            for domain in LightTransitionControlDomain
        }
        self.events: List[LightTransitionEvent] = []
        self.decision_history: List[LightTransitionDecision] = []
        self.last_decision: Optional[LightTransitionDecision] = None

    def evaluate(self, signals: LightTransitionSignals) -> LightTransitionDecision:
        valid, validation_issues = validate_light_transition_signals(signals)
        peer_values = [r.value for r in signals.optical_readings]
        timestamp = float(signals.timestamp if signals.timestamp is not None else time.time())
        flags = self._safety_flags(signals) + validation_issues
        propagation_fraction_c = _propagation_fraction_c(
            signals.distance_m,
            signals.measured_time_of_flight_ns,
        )
        dynamic_speed_limit = _get_dynamic_speed_limit(
            signals.medium,
            signals.environmental_temperature_c,
            signals.environmental_pressure_pa,
            signals.refractive_index_override,
        )
        refractive_index = _get_refractive_index(
            signals.medium,
            signals.environmental_temperature_c,
            signals.environmental_pressure_pa,
            signals.refractive_index_override,
        )

        if not valid:
            peer_values = []

        pipeline = self.pipelines[signals.control_domain]
        result = pipeline.process(
            raw_value=_clamp01(signals.ai_value),
            raw_confidence=_clamp01(signals.ai_confidence),
            peer_values=peer_values,
            timestamp=timestamp,
            context={
                "domain": "light_transition",
                "control_domain": signals.control_domain.value,
                "proposed_action": signals.proposed_action.value,
                "channel_id": signals.channel_id,
                "medium": signals.medium.value,
                "peer_count": len(peer_values),
                "data_rate_gbps": signals.data_rate_gbps,
                "wavelength_nm": signals.wavelength_nm,
                "environmental_temperature_c": signals.environmental_temperature_c,
                "environmental_pressure_pa": signals.environmental_pressure_pa,
                "refractive_index_override": signals.refractive_index_override,
                "propagation_fraction_c": propagation_fraction_c,
                "dynamic_speed_limit_fraction_c": dynamic_speed_limit,
                "refractive_index": refractive_index,
                **signals.context,
            },
        )

        level = _trust_level_from_result(result, flags)
        actionable = level >= self.policy.min_trust_for_action and not flags
        if self.policy.require_consensus and len(peer_values) < _config_for_domain(signals.control_domain).consensus_quorum:
            actionable = False
            if "insufficient_optical_peer_quorum" not in flags:
                flags.append("insufficient_optical_peer_quorum")
            if level > LightTransitionTrustLevel.ADVISORY:
                level = LightTransitionTrustLevel.ADVISORY
            actionable = False

        health = _health_from_level(level, flags, result.used_fallback)
        decision = LightTransitionDecision(
            authorized_level=level,
            actionable=actionable,
            trusted_value=result.value,
            control_domain=signals.control_domain,
            proposed_action=signals.proposed_action,
            pipeline_result=result,
            health_status=health,
            safety_flags=flags,
            used_fallback=result.used_fallback,
            fallback_reason="; ".join(result.reasons) if result.used_fallback else None,
            timestamp=timestamp,
            decision_id=self._decision_id(signals, timestamp),
            reasons=list(result.reasons),
            metadata={
                "confidence_score": result.confidence_score,
                "safety_status": result.safety_status.value,
                "consensus_status": result.consensus_status.value,
                "grace_status": result.grace_status.value,
                "propagation_fraction_c": propagation_fraction_c,
                "dynamic_speed_limit_fraction_c": dynamic_speed_limit,
                "refractive_index": refractive_index,
            },
        )
        self.last_decision = decision
        if self.policy.track_decision_history:
            self.decision_history.append(decision)
        if self.policy.enable_audit_events:
            self.events.append(
                LightTransitionEvent(
                    event_type="decision",
                    control_domain=signals.control_domain,
                    timestamp=timestamp,
                    decision=decision,
                    details={"channel_id": signals.channel_id, "flags": list(flags)},
                )
            )
        return decision

    def _safety_flags(self, signals: LightTransitionSignals) -> List[str]:
        flags: List[str] = []
        p = self.policy
        if signals.signal_age_ns is not None and signals.signal_age_ns > p.max_signal_age_ns:
            flags.append(f"signal_stale:{signals.signal_age_ns:.1f}ns>{p.max_signal_age_ns:.1f}ns")
        if signals.bit_error_rate is not None and signals.bit_error_rate > p.max_bit_error_rate:
            flags.append(f"bit_error_rate_high:{signals.bit_error_rate:.3g}>{p.max_bit_error_rate:.3g}")
        if signals.snr_db is not None and signals.snr_db < p.min_snr_db:
            flags.append(f"snr_low:{signals.snr_db:.1f}dB<{p.min_snr_db:.1f}dB")
        if signals.clock_offset_ps is not None and abs(signals.clock_offset_ps) > p.max_clock_offset_ps:
            flags.append(f"clock_offset_high:{signals.clock_offset_ps:.1f}ps>{p.max_clock_offset_ps:.1f}ps")
        if signals.eye_opening is not None and signals.eye_opening < p.min_eye_opening:
            flags.append(f"eye_opening_low:{signals.eye_opening:.2f}<{p.min_eye_opening:.2f}")
        if signals.dispersion_ps_nm is not None and abs(signals.dispersion_ps_nm) > p.max_dispersion_ps_nm:
            flags.append(f"dispersion_high:{signals.dispersion_ps_nm:.1f}ps/nm>{p.max_dispersion_ps_nm:.1f}ps/nm")
        fraction_c = _propagation_fraction_c(signals.distance_m, signals.measured_time_of_flight_ns)
        dynamic_limit = _get_dynamic_speed_limit(
            signals.medium,
            signals.environmental_temperature_c,
            signals.environmental_pressure_pa,
            signals.refractive_index_override,
        )
        if fraction_c is not None and fraction_c > dynamic_limit + 1e-9:
            flags.append(
                "physics_bound_violation:"
                f" measured {fraction_c:.6f}c > limit {dynamic_limit:.6f}c"
                f" for {signals.medium.value}"
            )
        return flags

    def _decision_id(self, signals: LightTransitionSignals, timestamp: float) -> str:
        return f"lt-{signals.channel_id}-{signals.control_domain.value}-{int(timestamp * 1_000_000)}"

    def get_health(self) -> LightTransitionHealthStatus:
        if self.last_decision is None:
            return LightTransitionHealthStatus.UNKNOWN
        return self.last_decision.health_status

    def get_subsystem_health(self) -> Dict[str, Any]:
        return {
            domain.value: {
                "history_length": len(pipeline.history),
                "last_good_value": pipeline.last_good_value,
                "last_result_used_fallback": (
                    pipeline.last_result.used_fallback if pipeline.last_result is not None else None
                ),
            }
            for domain, pipeline in self.pipelines.items()
        }

    def get_metrics(self) -> Dict[str, Any]:
        fallback_count = sum(1 for d in self.decision_history if d.used_fallback)
        actionable_count = sum(1 for d in self.decision_history if d.actionable)
        confidence_values = [
            d.metadata.get("confidence_score", 0.0)
            for d in self.decision_history
        ]
        return {
            "decisions_total": len(self.decision_history),
            "events_total": len(self.events),
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / len(self.decision_history) if self.decision_history else 0.0,
            "actionable_count": actionable_count,
            "avg_confidence": statistics.fmean(confidence_values) if confidence_values else 0.0,
        }

    def get_events(self) -> List[LightTransitionEvent]:
        return list(self.events)

    def get_decision_history(self) -> List[LightTransitionDecision]:
        return list(self.decision_history)


# ===========================
# Factories & Compatibility Helpers
# ===========================

def create_light_transition_governor(
    policy: Optional[LightTransitionPolicy] = None,
) -> LightTransitionGovernor:
    return LightTransitionGovernor(policy=policy)


def create_default_governor() -> LightTransitionGovernor:
    return create_light_transition_governor()


def create_strict_governor() -> LightTransitionGovernor:
    return create_light_transition_governor(
        LightTransitionPolicy(
            min_trust_for_action=LightTransitionTrustLevel.AUTONOMOUS,
            require_consensus=True,
            max_signal_age_ns=250_000.0,
            max_bit_error_rate=1e-12,
            min_snr_db=24.0,
            max_clock_offset_ps=20.0,
            min_eye_opening=0.75,
            max_dispersion_ps_nm=18.0,
        )
    )


def create_permissive_governor() -> LightTransitionGovernor:
    return create_light_transition_governor(
        LightTransitionPolicy(
            min_trust_for_action=LightTransitionTrustLevel.ADVISORY,
            require_consensus=False,
            max_signal_age_ns=5_000_000.0,
            max_bit_error_rate=1e-6,
            min_snr_db=10.0,
            max_clock_offset_ps=250.0,
            min_eye_opening=0.35,
            max_dispersion_ps_nm=100.0,
        )
    )


def create_example_signals() -> LightTransitionSignals:
    now = time.time()
    return LightTransitionSignals(
        control_domain=LightTransitionControlDomain.PHOTONIC_SIGNAL_INTEGRITY,
        proposed_action=LightTransitionControlAction.ACCEPT_FRAME,
        ai_value=0.94,
        ai_confidence=0.96,
        optical_readings=[
            OpticalReading(0.93, now, "photodiode_a", confidence=0.95, wavelength_nm=1550.0),
            OpticalReading(0.95, now, "photodiode_b", confidence=0.94, wavelength_nm=1550.0),
            OpticalReading(0.92, now, "tap_monitor_c", confidence=0.93, wavelength_nm=1550.0),
        ],
        medium=PropagationMedium.FIBER,
        signal_age_ns=125_000.0,
        distance_m=20_000.0,
        measured_time_of_flight_ns=100_000.0,
        bit_error_rate=1e-12,
        snr_db=28.0,
        clock_offset_ps=8.0,
        eye_opening=0.82,
        dispersion_ps_nm=12.0,
        data_rate_gbps=400.0,
        wavelength_nm=1550.0,
        channel_id="lambda_1550_a",
    )


def create_degraded_signals() -> LightTransitionSignals:
    now = time.time()
    return LightTransitionSignals(
        control_domain=LightTransitionControlDomain.OPTICAL_PATH_RELIABILITY,
        proposed_action=LightTransitionControlAction.SWITCH_PATH,
        ai_value=0.61,
        ai_confidence=0.64,
        optical_readings=[
            OpticalReading(0.58, now, "photodiode_a"),
            OpticalReading(0.66, now, "photodiode_b"),
        ],
        medium=PropagationMedium.FREE_SPACE,
        signal_age_ns=2_000_000.0,
        bit_error_rate=1e-5,
        snr_db=9.0,
        clock_offset_ps=180.0,
        eye_opening=0.28,
        channel_id="degraded_fso_path",
    )


def default_light_transition_config() -> AileeConfig:
    return PHOTONIC_SIGNAL_INTEGRITY


def export_events_to_dict(events: Sequence[LightTransitionEvent]) -> List[Dict[str, Any]]:
    return [
        {
            "event_type": event.event_type,
            "control_domain": event.control_domain.value,
            "timestamp": event.timestamp,
            "decision_id": event.decision.decision_id if event.decision else None,
            "authorized_level": event.decision.authorized_level.name if event.decision else None,
            "actionable": event.decision.actionable if event.decision else None,
            "details": dict(event.details),
        }
        for event in events
    ]


def get_health(governor: LightTransitionGovernor) -> LightTransitionHealthStatus:
    return governor.get_health()


def get_subsystem_health(governor: LightTransitionGovernor) -> Dict[str, Any]:
    return governor.get_subsystem_health()


def get_metrics(governor: LightTransitionGovernor) -> Dict[str, Any]:
    return governor.get_metrics()


def get_events(governor: LightTransitionGovernor) -> List[LightTransitionEvent]:
    return governor.get_events()


def get_decision_history(governor: LightTransitionGovernor) -> List[LightTransitionDecision]:
    return governor.get_decision_history()


# Short aliases matching existing domain naming conventions.
LightTrustLevel = LightTransitionTrustLevel
LightTransitionConfig = LightTransitionPolicy
