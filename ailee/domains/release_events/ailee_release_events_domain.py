
"""
AILEE Trust Layer — ReleaseEvents Domain
Version: 4.2.0

Governance and trust-scoring framework for ReleaseEvents systems.
"""

import time
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import hashlib

class ReleaseEventsTrustLevel(IntEnum):
    NO_ACTION = 0
    ADVISORY = 1
    SUPERVISED = 2
    AUTONOMOUS = 3

class ReleaseEventsHealthStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class ReleaseEventsControlDomain(str, Enum):
    DEFAULT = "DEFAULT"

class ReleaseEventsControlAction(str, Enum):
    MONITOR = "MONITOR"
    ACT = "ACT"

@dataclass
class ReleaseEventsPolicy:
    min_trust_for_action: ReleaseEventsTrustLevel = ReleaseEventsTrustLevel.SUPERVISED
    require_consensus: bool = False
    enable_audit_events: bool = True
    track_decision_history: bool = True

@dataclass
class ReleaseEventsSignals:
    control_domain: ReleaseEventsControlDomain
    proposed_action: ReleaseEventsControlAction
    ai_value: Any
    ai_confidence: float
    zone_id: str = "default"
    sensor_readings: List[Any] = field(default_factory=list)

@dataclass
class ReleaseEventsDecision:
    authorized_level: ReleaseEventsTrustLevel
    actionable: bool
    trusted_value: Any
    control_domain: ReleaseEventsControlDomain
    proposed_action: ReleaseEventsControlAction
    health_status: ReleaseEventsHealthStatus
    safety_flags: List[str]
    used_fallback: bool
    fallback_reason: str
    timestamp: float
    decision_id: str
    reasons: List[str]
    pipeline_result: Any = None

@dataclass
class ReleaseEventsEvent:
    event_type: str
    control_domain: ReleaseEventsControlDomain
    timestamp: float
    decision: ReleaseEventsDecision
    details: Dict[str, Any] = field(default_factory=dict)

class ReleaseEventsGovernor:
    def __init__(self, policy: ReleaseEventsPolicy):
        self.policy = policy
        self._events: List[ReleaseEventsEvent] = []
        self._history: List[ReleaseEventsDecision] = []

    def evaluate(self, signals: ReleaseEventsSignals) -> ReleaseEventsDecision:
        ts = time.time()
        decision_id = hashlib.sha256(f"{ts}{signals.control_domain}".encode()).hexdigest()[:16]

        # Validation
        issues = validate_release_events_signals(signals)
        if issues:
            decision = ReleaseEventsDecision(
                authorized_level=ReleaseEventsTrustLevel.NO_ACTION,
                actionable=False,
                trusted_value=signals.ai_value,
                control_domain=signals.control_domain,
                proposed_action=signals.proposed_action,
                health_status=ReleaseEventsHealthStatus.CRITICAL,
                safety_flags=["validation_failed"] + issues,
                used_fallback=True,
                fallback_reason="Invalid signals",
                timestamp=ts,
                decision_id=decision_id,
                reasons=issues,
            )
        else:
            trust_level = ReleaseEventsTrustLevel.AUTONOMOUS if signals.ai_confidence > 0.8 else ReleaseEventsTrustLevel.ADVISORY
            actionable = trust_level >= self.policy.min_trust_for_action

            decision = ReleaseEventsDecision(
                authorized_level=trust_level,
                actionable=actionable,
                trusted_value=signals.ai_value,
                control_domain=signals.control_domain,
                proposed_action=signals.proposed_action,
                health_status=ReleaseEventsHealthStatus.OPTIMAL,
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
            self._events.append(ReleaseEventsEvent(
                event_type="evaluation",
                control_domain=signals.control_domain,
                timestamp=ts,
                decision=decision,
            ))

        return decision

    def get_trust_level(self) -> ReleaseEventsTrustLevel:
        if not self._history:
            return ReleaseEventsTrustLevel.NO_ACTION
        return self._history[-1].authorized_level

    def get_health(self) -> ReleaseEventsHealthStatus:
        if not self._history:
            return ReleaseEventsHealthStatus.OPTIMAL
        return self._history[-1].health_status

    def get_subsystem_health(self) -> Dict[str, ReleaseEventsHealthStatus]:
        return {"default": self.get_health()}

    def get_decision_history(self) -> List[ReleaseEventsDecision]:
        return list(self._history)

    def get_events(self) -> List[ReleaseEventsEvent]:
        return list(self._events)

    def get_metrics(self) -> Dict[str, float]:
        return {
            "decisions_made": len(self._history),
            "events_logged": len(self._events)
        }

def get_health(governor: 'ReleaseEventsGovernor') -> ReleaseEventsHealthStatus:
    return governor.get_health()

def get_subsystem_health(governor: 'ReleaseEventsGovernor') -> Dict[str, ReleaseEventsHealthStatus]:
    return governor.get_subsystem_health()

def get_metrics(governor: 'ReleaseEventsGovernor') -> Dict[str, Any]:
    return governor.get_metrics()

def get_events(governor: 'ReleaseEventsGovernor') -> List[ReleaseEventsEvent]:
    return governor.get_events()

def get_decision_history(governor: 'ReleaseEventsGovernor') -> List[ReleaseEventsDecision]:
    return governor.get_decision_history()

def create_release_events_governor(policy: Optional[ReleaseEventsPolicy] = None, **policy_overrides: Any) -> ReleaseEventsGovernor:
    if policy is None:
        policy = ReleaseEventsPolicy(**policy_overrides)
    return ReleaseEventsGovernor(policy=policy)

def create_default_governor(**policy_overrides: Any) -> ReleaseEventsGovernor:
    return ReleaseEventsGovernor(policy=ReleaseEventsPolicy(**policy_overrides))

def create_strict_governor(**policy_overrides: Any) -> ReleaseEventsGovernor:
    overrides = {
        "min_trust_for_action": ReleaseEventsTrustLevel.AUTONOMOUS,
        "require_consensus": True,
        "enable_audit_events": True,
        "track_decision_history": True,
    }
    overrides.update(policy_overrides)
    return ReleaseEventsGovernor(policy=ReleaseEventsPolicy(**overrides))

def create_permissive_governor(**policy_overrides: Any) -> ReleaseEventsGovernor:
    overrides = {
        "min_trust_for_action": ReleaseEventsTrustLevel.ADVISORY,
        "require_consensus": False,
        "enable_audit_events": False,
        "track_decision_history": False,
    }
    overrides.update(policy_overrides)
    return ReleaseEventsGovernor(policy=ReleaseEventsPolicy(**overrides))

def validate_release_events_signals(signals: ReleaseEventsSignals) -> List[str]:
    issues = []
    if not (0.0 <= signals.ai_confidence <= 1.0):
        issues.append("confidence must be between 0.0 and 1.0")
    return issues
