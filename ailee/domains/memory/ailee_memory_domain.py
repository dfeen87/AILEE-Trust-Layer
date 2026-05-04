"""
AILEE Trust Layer — Memory Domain
Version: 4.6.0

First-class AILEE domain implementation for Memory decision integrity.
"""

from __future__ import annotations

import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, IntEnum
from dataclasses import dataclass, field

from ...ailee_trust_pipeline_v1 import AileeTrustPipeline, AileeConfig, DecisionResult, SafetyStatus

# ===========================
# Config Presets
# ===========================

RAM_ALLOCATION = AileeConfig(
    accept_threshold=0.90,
    borderline_low=0.75,
    borderline_high=0.90,
    hard_min=0.0,
    hard_max=0.98,
    consensus_quorum=3,
    consensus_delta=0.03,
    grace_peer_delta=0.05,
    grace_forecast_epsilon=0.08,
    fallback_mode="last_good",
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,
    history_window=120,
    forecast_window=20,
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True
)

HEAP_MONITORING = AileeConfig(
    accept_threshold=0.92,
    borderline_low=0.75,
    borderline_high=0.92,
    hard_min=0.0,
    hard_max=0.95,
    consensus_quorum=3,
    consensus_delta=0.03,
    grace_peer_delta=0.05,
    grace_forecast_epsilon=0.08,
    fallback_mode="last_good",
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,
    history_window=120,
    forecast_window=20,
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True
)

SWAP_MANAGEMENT = AileeConfig(
    accept_threshold=0.95,
    borderline_low=0.75,
    borderline_high=0.95,
    hard_min=0.0,
    hard_max=1.0,
    consensus_quorum=3,
    consensus_delta=0.03,
    grace_peer_delta=0.05,
    grace_forecast_epsilon=0.08,
    fallback_mode="last_good",
    w_stability=0.55,
    w_agreement=0.25,
    w_likelihood=0.20,
    history_window=120,
    forecast_window=20,
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True
)

PROCESS_MEMORY = AileeConfig(
    accept_threshold=0.88,
    borderline_low=0.75,
    borderline_high=0.88,
    hard_min=0.0,
    hard_max=1.0,
    consensus_quorum=3,
    consensus_delta=0.03,
    grace_peer_delta=0.05,
    grace_forecast_epsilon=0.08,
    fallback_mode="median",
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,
    history_window=120,
    forecast_window=20,
    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True
)

# ===========================
# Enumerations
# ===========================

class MemoryTrustLevel(IntEnum):
    NO_ACTION = 0    # Do not act — insufficient confidence or OOM imminent
    ADVISORY = 1     # Log or alert only
    SUPERVISED = 2   # Act, but flag for operator review
    AUTONOMOUS = 3   # Fully authorized autonomous memory management

class MemoryHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"     # utilization trending high
    DEGRADED = "DEGRADED"   # frequent fallbacks or swap active
    CRITICAL = "CRITICAL"   # OOM imminent or override active
    UNKNOWN = "UNKNOWN"

class MemoryDomain(str, Enum):
    RAM_ALLOCATION = "RAM_ALLOCATION"
    HEAP = "HEAP"
    SWAP = "SWAP"
    PROCESS = "PROCESS"

class MemoryAction(str, Enum):
    ENFORCE_ALLOCATION_LIMIT = "ENFORCE_ALLOCATION_LIMIT"
    TRIGGER_GC = "TRIGGER_GC"
    ENABLE_SWAP = "ENABLE_SWAP"
    THROTTLE_PROCESS = "THROTTLE_PROCESS"
    EVICT_CACHE = "EVICT_CACHE"
    KILL_PROCESS = "KILL_PROCESS"      # OOM-kill equivalent — only at AUTONOMOUS level
    NO_ACTION = "NO_ACTION"

# ===========================
# Dataclasses
# ===========================

@dataclass
class MemoryReading:
    value: float          # utilization [0.0, 1.0] or bytes
    timestamp: float
    sensor_id: str        # e.g. "node_01_ram", "heap_jvm_prod"
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryPolicy:
    min_trust_for_action: MemoryTrustLevel = MemoryTrustLevel.SUPERVISED
    require_consensus: bool = True
    require_reading_validation: bool = True
    max_allocation_change_per_cycle: float = 0.10  # 10% utilization delta
    max_oom_kills_per_hour: int = 5
    oom_emergency_threshold: float = 0.97  # triggers pre-pipeline OOM override
    swap_enable_threshold: float = 0.85
    enable_audit_events: bool = True
    track_decision_history: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.oom_emergency_threshold <= 1.0):
            raise ValueError(
                f"oom_emergency_threshold must be in (0.0, 1.0], got {self.oom_emergency_threshold}"
            )
        if not (0.0 < self.swap_enable_threshold <= 1.0):
            raise ValueError(
                f"swap_enable_threshold must be in (0.0, 1.0], got {self.swap_enable_threshold}"
            )
        if self.max_oom_kills_per_hour < 0:
            raise ValueError(
                f"max_oom_kills_per_hour must be >= 0, got {self.max_oom_kills_per_hour}"
            )
        if not (0.0 < self.max_allocation_change_per_cycle <= 1.0):
            raise ValueError(
                f"max_allocation_change_per_cycle must be in (0.0, 1.0], got {self.max_allocation_change_per_cycle}"
            )

@dataclass
class MemorySignals:
    memory_domain: MemoryDomain
    proposed_action: MemoryAction
    ai_value: float               # proposed utilization [0.0, 1.0] or bytes
    ai_confidence: float          # [0.0, 1.0]
    memory_readings: List[MemoryReading] = field(default_factory=list)
    node_id: str = "default"
    process_id: Optional[str] = None
    timestamp: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryDecision:
    authorized_level: MemoryTrustLevel
    actionable: bool
    trusted_value: float
    memory_domain: MemoryDomain
    proposed_action: MemoryAction
    pipeline_result: Optional[Any] = None
    health_status: MemoryHealthStatus = MemoryHealthStatus.HEALTHY
    safety_flags: List[str] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    decision_id: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryEvent:
    event_type: str   # "decision" | "fallback" | "oom_override" | "rate_limit"
    memory_domain: MemoryDomain
    timestamp: float = field(default_factory=time.time)
    decision: Optional[MemoryDecision] = None
    details: Dict[str, Any] = field(default_factory=dict)

# ===========================
# Governor
# ===========================

class MemoryGovernor:
    def __init__(
        self,
        policy: Optional[MemoryPolicy] = None,
        ram_config: Optional[AileeConfig] = None,
        heap_config: Optional[AileeConfig] = None,
        swap_config: Optional[AileeConfig] = None,
        process_config: Optional[AileeConfig] = None,
    ):
        self.policy = policy or MemoryPolicy()
        self._pipelines = {
            MemoryDomain.RAM_ALLOCATION: AileeTrustPipeline(ram_config or RAM_ALLOCATION),
            MemoryDomain.HEAP: AileeTrustPipeline(heap_config or HEAP_MONITORING),
            MemoryDomain.SWAP: AileeTrustPipeline(swap_config or SWAP_MANAGEMENT),
            MemoryDomain.PROCESS: AileeTrustPipeline(process_config or PROCESS_MEMORY),
        }
        self._decision_history: List[MemoryDecision] = []
        self._events: List[MemoryEvent] = []
        self._oom_kills_this_hour = 0
        self._hour_start_time = time.time()
        self._last_health = MemoryHealthStatus.HEALTHY
        self._subsystem_health = {domain: MemoryHealthStatus.HEALTHY for domain in MemoryDomain}

    def evaluate(self, signals: MemorySignals) -> MemoryDecision:
        ts = signals.timestamp or time.time()
        peer_values = [r.value for r in signals.memory_readings]

        # Optional pre-flight validation
        if self.policy.require_reading_validation:
            issues = validate_memory_signals(signals)
            if issues:
                reason = f"Signal validation failed: {'; '.join(issues)}"
                decision = self._build_no_action_decision(
                    signals,
                    reason=reason,
                    flags=["validation_failed"],
                    health=MemoryHealthStatus.WARNING,
                    ts=ts
                )
                self._record_event(MemoryEvent(
                    event_type="validation_failed",
                    memory_domain=signals.memory_domain,
                    timestamp=ts,
                    decision=decision,
                    details={"reason": reason, "issues": issues}
                ))
                if self.policy.track_decision_history:
                    self._decision_history.append(decision)
                self._update_health(signals.memory_domain, decision)
                return decision

        # Reset hourly counter if needed
        if ts - self._hour_start_time >= 3600:
            self._oom_kills_this_hour = 0
            self._hour_start_time = ts

        # OOM emergency override
        if signals.memory_domain == MemoryDomain.RAM_ALLOCATION and peer_values:
            max_peer = max(peer_values)
            if max_peer >= self.policy.oom_emergency_threshold:
                reason = f"OOM override: observed utilization {max_peer:.3f} exceeds threshold {self.policy.oom_emergency_threshold}"
                decision = self._build_no_action_decision(
                    signals,
                    reason=reason,
                    flags=["oom_override"],
                    health=MemoryHealthStatus.CRITICAL,
                    ts=ts
                )
                self._record_event(MemoryEvent(
                    event_type="oom_override",
                    memory_domain=signals.memory_domain,
                    timestamp=ts,
                    decision=decision,
                    details={"reason": reason, "max_peer": max_peer}
                ))
                if self.policy.track_decision_history:
                    self._decision_history.append(decision)
                self._update_health(signals.memory_domain, decision)
                return decision

        # Rate-limit OOM kills
        if signals.proposed_action == MemoryAction.KILL_PROCESS:
            if self._oom_kills_this_hour >= self.policy.max_oom_kills_per_hour:
                reason = f"Rate limit: max OOM kills per hour ({self.policy.max_oom_kills_per_hour}) reached"
                decision = self._build_no_action_decision(
                    signals,
                    reason=reason,
                    flags=["rate_limit"],
                    health=MemoryHealthStatus.WARNING,
                    ts=ts
                )
                self._record_event(MemoryEvent(
                    event_type="rate_limit",
                    memory_domain=signals.memory_domain,
                    timestamp=ts,
                    decision=decision,
                    details={"reason": reason}
                ))
                if self.policy.track_decision_history:
                    self._decision_history.append(decision)
                self._update_health(signals.memory_domain, decision)
                return decision

        pipeline = self._pipelines[signals.memory_domain]
        context = {
            "node_id": signals.node_id,
            "memory_domain": signals.memory_domain.value,
            "proposed_action": signals.proposed_action.value,
        }
        if signals.process_id:
            context["process_id"] = signals.process_id

        result = pipeline.process(
            raw_value=signals.ai_value,
            raw_confidence=signals.ai_confidence,
            peer_values=peer_values,
            timestamp=ts,
            context=context
        )

        trust_level, is_actionable, fallback_used = self._determine_trust_level(result)

        decision_id = hashlib.sha256(f"{ts}{signals.memory_domain}{signals.ai_value}".encode()).hexdigest()[:16]

        flags = []
        if result.safety_status == SafetyStatus.OUTRIGHT_REJECTED:
            flags.append("unsafe_value")
        if fallback_used:
            flags.append("fallback_used")

        health = MemoryHealthStatus.HEALTHY
        if fallback_used:
            health = MemoryHealthStatus.DEGRADED
        elif result.safety_status == SafetyStatus.OUTRIGHT_REJECTED:
            health = MemoryHealthStatus.WARNING

        metadata = {
            "confidence_score": signals.ai_confidence,
            "safety_status": result.safety_status.value,
            "reading_count": len(peer_values)
        }

        decision = MemoryDecision(
            authorized_level=trust_level,
            actionable=is_actionable,
            trusted_value=result.value,
            memory_domain=signals.memory_domain,
            proposed_action=signals.proposed_action,
            pipeline_result=result,
            health_status=health,
            safety_flags=flags,
            used_fallback=fallback_used,
            fallback_reason=result.safety_status.value if fallback_used else None,
            timestamp=ts,
            decision_id=decision_id,
            reasons=result.reasons,
            metadata=metadata
        )

        if is_actionable and signals.proposed_action == MemoryAction.KILL_PROCESS:
            self._oom_kills_this_hour += 1

        if self.policy.track_decision_history:
            self._decision_history.append(decision)

        event_type = "fallback" if fallback_used else "decision"
        self._record_event(MemoryEvent(
            event_type=event_type,
            memory_domain=signals.memory_domain,
            timestamp=ts,
            decision=decision,
            details={"actionable": is_actionable, "trusted_value": result.value}
        ))

        self._update_health(signals.memory_domain, decision)

        return decision

    def _build_no_action_decision(
        self,
        signals: MemorySignals,
        reason: str,
        flags: List[str],
        health: MemoryHealthStatus,
        ts: float
    ) -> MemoryDecision:
        decision_id = hashlib.sha256(f"{ts}{signals.memory_domain}{signals.ai_value}".encode()).hexdigest()[:16]
        return MemoryDecision(
            authorized_level=MemoryTrustLevel.NO_ACTION,
            actionable=False,
            trusted_value=signals.ai_value,
            memory_domain=signals.memory_domain,
            proposed_action=signals.proposed_action,
            health_status=health,
            safety_flags=flags,
            used_fallback=False,
            fallback_reason=reason,
            timestamp=ts,
            decision_id=decision_id,
            reasons=[reason],
            metadata={}
        )

    def _determine_trust_level(self, result: DecisionResult) -> Tuple[MemoryTrustLevel, bool, bool]:
        fallback_used = result.used_fallback
        if result.safety_status == SafetyStatus.OUTRIGHT_REJECTED:
            return MemoryTrustLevel.NO_ACTION, False, False

        if fallback_used:
            level = MemoryTrustLevel.ADVISORY
        elif result.safety_status == "ACCEPTED" and result.confidence_score >= 0.90:
            level = MemoryTrustLevel.AUTONOMOUS
        else:
            level = MemoryTrustLevel.SUPERVISED

        is_actionable = level >= self.policy.min_trust_for_action
        return level, is_actionable, fallback_used

    def _record_event(self, event: MemoryEvent):
        if self.policy.enable_audit_events:
            self._events.append(event)

    def _update_health(self, domain: MemoryDomain, decision: MemoryDecision):
        self._subsystem_health[domain] = decision.health_status
        # Overall health logic
        criticals = sum(1 for h in self._subsystem_health.values() if h == MemoryHealthStatus.CRITICAL)
        degradeds = sum(1 for h in self._subsystem_health.values() if h == MemoryHealthStatus.DEGRADED)
        warnings = sum(1 for h in self._subsystem_health.values() if h == MemoryHealthStatus.WARNING)

        if criticals > 0:
            self._last_health = MemoryHealthStatus.CRITICAL
        elif degradeds > 0:
            self._last_health = MemoryHealthStatus.DEGRADED
        elif warnings > 0:
            self._last_health = MemoryHealthStatus.WARNING
        else:
            self._last_health = MemoryHealthStatus.HEALTHY

    def get_health(self) -> MemoryHealthStatus:
        return self._last_health

    def get_subsystem_health(self) -> Dict[str, MemoryHealthStatus]:
        return {domain.value: status for domain, status in self._subsystem_health.items()}

    def get_metrics(self) -> Dict[str, Any]:
        total = len(self._decision_history)
        fallbacks = sum(1 for d in self._decision_history if d.used_fallback)
        fallback_rate = fallbacks / total if total > 0 else 0.0

        avg_conf = 0.0
        if total > 0:
            conf_sum = sum(d.metadata.get("confidence_score", 0.0) for d in self._decision_history)
            avg_conf = conf_sum / total

        return {
            "fallback_rate": fallback_rate,
            "avg_confidence": avg_conf,
            "total_decisions": total,
            "oom_kills_this_hour": self._oom_kills_this_hour,
            "overall_health": self._last_health.value
        }

    def get_events(self) -> List[MemoryEvent]:
        return self._events

    def get_decision_history(self) -> List[MemoryDecision]:
        return self._decision_history

# ===========================
# Validation
# ===========================

def validate_memory_signals(signals: MemorySignals) -> List[str]:
    """Returns a list of issue strings (empty = valid)."""
    issues = []

    if not (0.0 <= signals.ai_confidence <= 1.0):
        issues.append(f"ai_confidence must be between 0.0 and 1.0, got {signals.ai_confidence}")

    if not signals.memory_readings:
        issues.append("No memory_readings provided (at least one reading/sensor is required)")

    normalized_domains = {MemoryDomain.RAM_ALLOCATION, MemoryDomain.HEAP, MemoryDomain.SWAP, MemoryDomain.PROCESS}
    if signals.memory_domain in normalized_domains:
        if not (0.0 <= signals.ai_value <= 1.0):
            issues.append(f"ai_value for normalized domain {signals.memory_domain.value} must be between 0.0 and 1.0, got {signals.ai_value}")

        for i, r in enumerate(signals.memory_readings):
            if not (0.0 <= r.value <= 1.0):
                issues.append(f"Reading {i} ({r.sensor_id}) value out of range: {r.value}")

    if not signals.node_id:
        issues.append("node_id cannot be empty or None")

    return issues

# ===========================
# Factories
# ===========================

def create_memory_governor(policy: Optional[MemoryPolicy] = None, **policy_overrides) -> MemoryGovernor:
    if policy is None:
        policy = MemoryPolicy()
    for k, v in policy_overrides.items():
        setattr(policy, k, v)
    return MemoryGovernor(policy=policy)

def create_default_governor(**policy_overrides) -> MemoryGovernor:
    return create_memory_governor(None, **policy_overrides)

def create_strict_governor(**policy_overrides) -> MemoryGovernor:
    policy = MemoryPolicy(
        min_trust_for_action=MemoryTrustLevel.AUTONOMOUS,
        require_consensus=True,
        max_allocation_change_per_cycle=0.05,
        max_oom_kills_per_hour=2
    )
    for k, v in policy_overrides.items():
        setattr(policy, k, v)
    return MemoryGovernor(policy=policy)

def create_permissive_governor(**policy_overrides) -> MemoryGovernor:
    policy = MemoryPolicy(
        min_trust_for_action=MemoryTrustLevel.ADVISORY,
        require_consensus=False,
        enable_audit_events=False,
        track_decision_history=False
    )
    for k, v in policy_overrides.items():
        setattr(policy, k, v)
    return MemoryGovernor(policy=policy)
