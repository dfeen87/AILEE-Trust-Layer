
"""
AILEE Trust Layer — Topology Domain
Version: 4.2.0

Governance and trust-scoring framework for Topology systems.
"""

import time
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import hashlib

class TopologyTrustLevel(IntEnum):
    NO_ACTION = 0
    ADVISORY = 1
    SUPERVISED = 2
    AUTONOMOUS = 3

class TopologyHealthStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class TopologyControlDomain(str, Enum):
    DEFAULT = "DEFAULT"

class TopologyControlAction(str, Enum):
    MONITOR = "MONITOR"
    ACT = "ACT"

@dataclass
class TopologyPolicy:
    min_trust_for_action: TopologyTrustLevel = TopologyTrustLevel.SUPERVISED
    require_consensus: bool = False
    enable_audit_events: bool = True
    track_decision_history: bool = True

@dataclass
class TopologySignals:
    control_domain: TopologyControlDomain
    proposed_action: TopologyControlAction
    ai_value: Any
    ai_confidence: float
    zone_id: str = "default"
    sensor_readings: List[Any] = field(default_factory=list)

@dataclass
class TopologyDecision:
    authorized_level: TopologyTrustLevel
    actionable: bool
    trusted_value: Any
    control_domain: TopologyControlDomain
    proposed_action: TopologyControlAction
    health_status: TopologyHealthStatus
    safety_flags: List[str]
    used_fallback: bool
    fallback_reason: str
    timestamp: float
    decision_id: str
    reasons: List[str]
    pipeline_result: Any = None

@dataclass
class TopologyEvent:
    event_type: str
    control_domain: TopologyControlDomain
    timestamp: float
    decision: TopologyDecision
    details: Dict[str, Any] = field(default_factory=dict)

class TopologyGovernor:
    def __init__(self, policy: TopologyPolicy):
        self.policy = policy
        self._events: List[TopologyEvent] = []
        self._history: List[TopologyDecision] = []

    def evaluate(self, signals: TopologySignals) -> TopologyDecision:
        ts = time.time()
        decision_id = hashlib.sha256(f"{ts}{signals.control_domain}".encode()).hexdigest()[:16]

        # Validation
        issues = validate_topology_signals(signals)
        if issues:
            decision = TopologyDecision(
                authorized_level=TopologyTrustLevel.NO_ACTION,
                actionable=False,
                trusted_value=signals.ai_value,
                control_domain=signals.control_domain,
                proposed_action=signals.proposed_action,
                health_status=TopologyHealthStatus.CRITICAL,
                safety_flags=["validation_failed"] + issues,
                used_fallback=True,
                fallback_reason="Invalid signals",
                timestamp=ts,
                decision_id=decision_id,
                reasons=issues,
            )
        else:
            trust_level = TopologyTrustLevel.AUTONOMOUS if signals.ai_confidence > 0.8 else TopologyTrustLevel.ADVISORY
            actionable = trust_level >= self.policy.min_trust_for_action

            decision = TopologyDecision(
                authorized_level=trust_level,
                actionable=actionable,
                trusted_value=signals.ai_value,
                control_domain=signals.control_domain,
                proposed_action=signals.proposed_action,
                health_status=TopologyHealthStatus.OPTIMAL,
                safety_flags=[],
                used_fallback=False,
                fallback_reason="",
                timestamp=ts,
                decision_id=decision_id,
                reasons=["Evaluated"],
            )

        if self.policy.track_decision_history:
            self._history.append(decision)

        if self.policy.enable_audit_events:
            self._events.append(TopologyEvent(
                event_type="evaluation",
                control_domain=signals.control_domain,
                timestamp=ts,
                decision=decision,
            ))

        return decision

    def get_trust_level(self) -> TopologyTrustLevel:
        if not self._history:
            return TopologyTrustLevel.NO_ACTION
        return self._history[-1].authorized_level

    def get_health(self) -> TopologyHealthStatus:
        if not self._history:
            return TopologyHealthStatus.OPTIMAL
        return self._history[-1].health_status

    def get_subsystem_health(self) -> Dict[str, TopologyHealthStatus]:
        return {"default": self.get_health()}

    def get_decision_history(self) -> List[TopologyDecision]:
        return list(self._history)

    def get_events(self) -> List[TopologyEvent]:
        return list(self._events)

    def get_metrics(self) -> Dict[str, float]:
        return {
            "decisions_made": len(self._history),
            "events_logged": len(self._events)
        }

def get_health(governor: 'TopologyGovernor') -> TopologyHealthStatus:
    return governor.get_health()

def get_subsystem_health(governor: 'TopologyGovernor') -> Dict[str, TopologyHealthStatus]:
    return governor.get_subsystem_health()

def get_metrics(governor: 'TopologyGovernor') -> Dict[str, Any]:
    return governor.get_metrics()

def get_events(governor: 'TopologyGovernor') -> List[TopologyEvent]:
    return governor.get_events()

def get_decision_history(governor: 'TopologyGovernor') -> List[TopologyDecision]:
    return governor.get_decision_history()

def create_topology_governor(policy: Optional[TopologyPolicy] = None, **policy_overrides: Any) -> TopologyGovernor:
    if policy is None:
        policy = TopologyPolicy(**policy_overrides)
    return TopologyGovernor(policy=policy)

def create_default_governor(**policy_overrides: Any) -> TopologyGovernor:
    return TopologyGovernor(policy=TopologyPolicy(**policy_overrides))

def create_strict_governor(**policy_overrides: Any) -> TopologyGovernor:
    overrides = {
        "min_trust_for_action": TopologyTrustLevel.AUTONOMOUS,
        "require_consensus": True,
        "enable_audit_events": True,
        "track_decision_history": True,
    }
    overrides.update(policy_overrides)
    return TopologyGovernor(policy=TopologyPolicy(**overrides))

def create_permissive_governor(**policy_overrides: Any) -> TopologyGovernor:
    overrides = {
        "min_trust_for_action": TopologyTrustLevel.ADVISORY,
        "require_consensus": False,
        "enable_audit_events": False,
        "track_decision_history": False,
    }
    overrides.update(policy_overrides)
    return TopologyGovernor(policy=TopologyPolicy(**overrides))

def validate_topology_signals(signals: TopologySignals) -> List[str]:
    issues = []
    if not (0.0 <= signals.ai_confidence <= 1.0):
        issues.append("confidence must be between 0.0 and 1.0")
    return issues
