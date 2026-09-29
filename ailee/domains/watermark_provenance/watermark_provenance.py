# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
AILEE Watermark-Provenance-Governance Domain — v7.0.0

Evaluates, interprets, and contextualizes AI watermark signals (e.g. SynthID-Text,
Claude watermarking) across real workflows. Treats watermark detection as one fragment
of provenance, not a verdict on authorship.
"""

from __future__ import annotations

import time
import hashlib
import math
from collections import deque
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


class WatermarkProvenanceTrustLevel(IntEnum):
    """Graduated trust levels for watermark provenance signals."""
    NO_ACTION = 0        # Unverified, blocked, or high-risk uncorroborated inference
    ADVISORY = 1         # Contextual information provided; no automated action
    SUPERVISED = 2       # Constrained usage requiring human sign-off / review
    AUTONOMOUS = 3       # Fully verified multi-event provenance with high confidence


class WatermarkProvenanceHealthStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class WatermarkProvenanceControlDomain(str, Enum):
    WORKFLOW_EVALUATION = "WORKFLOW_EVALUATION"
    PUBLICATION_GATE = "PUBLICATION_GATE"
    HIGH_STAKES_COMPLIANCE = "HIGH_STAKES_COMPLIANCE"
    GENERAL_PROVENANCE = "GENERAL_PROVENANCE"


class WatermarkProvenanceControlAction(str, Enum):
    MONITOR = "MONITOR"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    REQUIRE_CORROBORATION = "REQUIRE_CORROBORATION"
    AUTHORIZE_PUBLICATION = "AUTHORIZE_PUBLICATION"
    REJECT_BINARY_INFERENCE = "REJECT_BINARY_INFERENCE"


class ProvenanceQualifier(str, Enum):
    """Canonical set of non-binary provenance qualifiers."""
    MODEL_INVOLVED_NOT_AUTHORED = "MODEL_INVOLVED_NOT_AUTHORED"
    MODEL_PRIMARY_DRAFTER = "MODEL_PRIMARY_DRAFTER"
    HUMAN_PRIMARY_DRAFTER_MODEL_EDITOR = "HUMAN_PRIMARY_DRAFTER_MODEL_EDITOR"
    HUMAN_EDITED = "HUMAN_EDITED"
    LIGHTLY_TOUCHED = "LIGHTLY_TOUCHED"
    TRANSLATED = "TRANSLATED"
    STRUCTURALLY_REWRITTEN = "STRUCTURALLY_REWRITTEN"
    UNKNOWN_ROLE_MODEL_INVOLVEMENT = "UNKNOWN_ROLE_MODEL_INVOLVEMENT"


@dataclass
class ProvenanceEventNode:
    """Represents a single node in a C2PA-style custody chain."""
    event_id: str
    stage: str  # e.g., 'G0_generation', 'E1_edit', 'V2_verification', 'A3_authorization', 'P4_publication'
    actor: str  # 'model:claude-3-5-sonnet', 'human:john_doe', 'system:verifier'
    action: str  # 'draft', 'paraphrase', 'edit', 'verify', 'approve'
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatermarkSignalData:
    """Raw or extracted watermark signal information."""
    detector_name: str  # e.g. 'SynthID-Text', 'Claude-Watermark', 'Perplexity-Detect'
    raw_score: float    # Detector output score (0.0 to 1.0)
    presence_detected: bool
    confidence: float   # Detector confidence score (0.0 to 1.0)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatermarkProvenancePolicy:
    """Policy configuration for Watermark-Provenance-Governance."""
    min_trust_for_action: WatermarkProvenanceTrustLevel = WatermarkProvenanceTrustLevel.SUPERVISED
    require_corroboration_for_high_stakes: bool = True
    high_stakes_contexts: Set[str] = field(default_factory=lambda: {
        "disciplinary_decision",
        "hiring_ranking",
        "contractual_enforcement",
        "academic_misconduct",
        "legal_evidence",
    })
    disruption_vectors: Set[str] = field(default_factory=lambda: {
        "paraphrasing",
        "back_translation",
        "regeneration",
        "structural_rewriting",
        "public_removal_tool",
    })
    allow_binary_labels: bool = False
    enable_audit_events: bool = True
    track_decision_history: bool = True
    max_history_size: int = 1000
    max_event_history_size: int = 1000
    max_evidence_items: int = 10000


@dataclass
class WatermarkProvenanceSignals:
    """Input signals for provenance evaluation."""
    control_domain: WatermarkProvenanceControlDomain
    proposed_action: WatermarkProvenanceControlAction
    watermark_signals: List[WatermarkSignalData] = field(default_factory=list)
    custody_chain: List[ProvenanceEventNode] = field(default_factory=list)
    suspected_disruptions: List[str] = field(default_factory=list)
    workflow_context: Dict[str, Any] = field(default_factory=dict)
    attempted_binary_label: Optional[str] = None  # e.g., "AI-generated" or "Human-written"
    is_high_stakes: bool = False
    context_type: str = "general"


@dataclass
class WatermarkProvenanceDecision:
    """Output decision struct for Watermark-Provenance-Governance."""
    authorized_level: WatermarkProvenanceTrustLevel
    actionable: bool
    provenance_confidence: float
    qualifiers: List[ProvenanceQualifier]
    over_interpreted: bool
    control_domain: WatermarkProvenanceControlDomain
    proposed_action: WatermarkProvenanceControlAction
    health_status: WatermarkProvenanceHealthStatus
    safety_flags: List[str]
    used_fallback: bool
    fallback_reason: str
    timestamp: float
    decision_id: str
    reasons: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatermarkProvenanceEvent:
    event_type: str
    control_domain: WatermarkProvenanceControlDomain
    timestamp: float
    decision: WatermarkProvenanceDecision
    details: Dict[str, Any] = field(default_factory=dict)


class WatermarkProvenanceGovernor:
    """
    AILEE Governor for Watermark Provenance.
    Evaluates watermark signals in workflow context, assigns non-binary qualifiers,
    maps attack surface disruptions, enforces high-stakes safeguards, and ensures
    mandatory challenge routes.
    """

    def __init__(self, policy: Optional[WatermarkProvenancePolicy] = None):
        self.policy = policy or WatermarkProvenancePolicy()
        if not isinstance(self.policy.max_history_size, int) or self.policy.max_history_size < 1:
            raise ValueError("max_history_size must be a positive integer")
        if not isinstance(self.policy.max_event_history_size, int) or self.policy.max_event_history_size < 1:
            raise ValueError("max_event_history_size must be a positive integer")
        if not isinstance(self.policy.max_evidence_items, int) or self.policy.max_evidence_items < 1:
            raise ValueError("max_evidence_items must be a positive integer")
        if not isinstance(self.policy.min_trust_for_action, WatermarkProvenanceTrustLevel):
            raise ValueError("min_trust_for_action must be a WatermarkProvenanceTrustLevel")
        if not isinstance(self.policy.high_stakes_contexts, set) or not all(isinstance(v, str) for v in self.policy.high_stakes_contexts):
            raise ValueError("high_stakes_contexts must be a set of strings")
        if not isinstance(self.policy.disruption_vectors, set) or not all(isinstance(v, str) for v in self.policy.disruption_vectors):
            raise ValueError("disruption_vectors must be a set of strings")
        self._events = deque(maxlen=self.policy.max_event_history_size)
        self._history = deque(maxlen=self.policy.max_history_size)

    def evaluate(self, signals: WatermarkProvenanceSignals) -> WatermarkProvenanceDecision:
        ts = time.time()
        decision_id = hashlib.sha256(f"{ts}{signals.control_domain}{signals.context_type}".encode()).hexdigest()[:16]
        reasons: List[str] = []
        safety_flags: List[str] = []
        constraints: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {
            "requires_human_review": True,
            "challenge_available": True,
            "challenge_route": "Any decision based on watermark signals must be reviewable and contestable.",
        }

        # Validation
        issues = validate_watermark_provenance_signals(signals)
        if isinstance(signals.watermark_signals, list) and len(signals.watermark_signals) > self.policy.max_evidence_items:
            issues.append(f"watermark_signals exceeds policy maximum {self.policy.max_evidence_items}")
        if isinstance(signals.custody_chain, list) and len(signals.custody_chain) > self.policy.max_evidence_items:
            issues.append(f"custody_chain exceeds policy maximum {self.policy.max_evidence_items}")
        if issues:
            decision = WatermarkProvenanceDecision(
                authorized_level=WatermarkProvenanceTrustLevel.NO_ACTION,
                actionable=False,
                provenance_confidence=0.0,
                qualifiers=[ProvenanceQualifier.UNKNOWN_ROLE_MODEL_INVOLVEMENT],
                over_interpreted=False,
                control_domain=signals.control_domain,
                proposed_action=signals.proposed_action,
                health_status=WatermarkProvenanceHealthStatus.CRITICAL,
                safety_flags=["VALIDATION_FAILED"] + issues,
                used_fallback=True,
                fallback_reason="Invalid signals: " + "; ".join(issues),
                timestamp=ts,
                decision_id=decision_id,
                reasons=issues,
                constraints=constraints,
                metadata=metadata,
            )
            self._commit_decision(decision, signals)
            return decision

        # 1. Check for attempted binary label over-interpretation
        over_interpreted = False
        if signals.attempted_binary_label and not self.policy.allow_binary_labels:
            over_interpreted = True
            safety_flags.append("OVER_INTERPRETATION_DETECTED")
            reasons.append(
                f"Binary label '{signals.attempted_binary_label}' is rejected. "
                "Watermark presence ≠ AI authorship; watermark absence ≠ human authorship."
            )
            metadata["warning"] = "Binary authorship labels are not supported; provenance_confidence is not authorship."

        # 2. Determine Qualifiers & Base Confidence from Signals & Custody Chain
        qualifiers, base_conf = self._analyze_workflow_and_signals(signals, reasons)

        # 3. Analyze Disruption Vectors / Attack Surface
        disruption_penalty = 0.0
        disruption_detected = False
        for vec in signals.suspected_disruptions:
            if vec in self.policy.disruption_vectors:
                disruption_detected = True
                disruption_penalty += 0.15
                safety_flags.append(f"DISRUPTION_VECTOR_DETECTED_{vec.upper()}")
                reasons.append(f"Attack surface disruption vector detected: '{vec}'.")

        if disruption_detected:
            qualifiers.append(ProvenanceQualifier.STRUCTURALLY_REWRITTEN)
            metadata["attack_surface_risk"] = "Potential watermark disruption or public removal tool usage."

        # Compute Final Provenance Confidence
        provenance_confidence = max(0.0, min(1.0, round(base_conf - disruption_penalty, 4)))

        # Deduplicate qualifiers while keeping list ordering
        qualifiers = list(dict.fromkeys(qualifiers))

        # 4. Evaluate High-Stakes Safeguards & Corroboration
        is_high_stakes = signals.is_high_stakes or (signals.context_type in self.policy.high_stakes_contexts)
        has_corroboration = len(signals.custody_chain) >= 2 or bool(signals.workflow_context.get("human_verification"))

        actionable = True
        authorized_level = WatermarkProvenanceTrustLevel.SUPERVISED

        if is_high_stakes:
            constraints["high_stakes"] = True
            if self.policy.require_corroboration_for_high_stakes and not has_corroboration:
                actionable = False
                authorized_level = WatermarkProvenanceTrustLevel.NO_ACTION
                safety_flags.append("INSUFFICIENT_PROVENANCE_FOR_ACTION")
                safety_flags.append("CORROBORATION_REQUIRED")
                reasons.append(
                    "High-stakes context requires corroboration (multi-event custody log or human verification step)."
                )

        if not (is_high_stakes and self.policy.require_corroboration_for_high_stakes and not has_corroboration):
            if provenance_confidence < 0.4:
                authorized_level = WatermarkProvenanceTrustLevel.NO_ACTION
                actionable = False
                reasons.append("Provenance confidence below baseline threshold (0.40).")
            elif provenance_confidence >= 0.85 and has_corroboration and not is_high_stakes:
                authorized_level = WatermarkProvenanceTrustLevel.AUTONOMOUS
            elif provenance_confidence >= 0.60:
                authorized_level = WatermarkProvenanceTrustLevel.SUPERVISED
            else:
                authorized_level = WatermarkProvenanceTrustLevel.ADVISORY

        if authorized_level < self.policy.min_trust_for_action:
            actionable = False

        health_status = WatermarkProvenanceHealthStatus.OPTIMAL
        if over_interpreted or disruption_detected or (is_high_stakes and not has_corroboration):
            health_status = WatermarkProvenanceHealthStatus.WARNING

        decision = WatermarkProvenanceDecision(
            authorized_level=authorized_level,
            actionable=actionable,
            provenance_confidence=provenance_confidence,
            qualifiers=qualifiers,
            over_interpreted=over_interpreted,
            control_domain=signals.control_domain,
            proposed_action=signals.proposed_action,
            health_status=health_status,
            safety_flags=safety_flags,
            used_fallback=False,
            fallback_reason="",
            timestamp=ts,
            decision_id=decision_id,
            reasons=reasons,
            constraints=constraints,
            metadata=metadata,
        )

        self._commit_decision(decision, signals)
        return decision

    def _analyze_workflow_and_signals(
        self, signals: WatermarkProvenanceSignals, reasons: List[str]
    ) -> tuple[List[ProvenanceQualifier], float]:
        qualifiers: List[ProvenanceQualifier] = []
        base_conf = 0.5

        # Check raw watermark detector scores
        max_detector_score = 0.0
        if signals.watermark_signals:
            max_detector_score = max(s.raw_score * s.confidence for s in signals.watermark_signals)

        # Analyze Custody Chain
        chain_stages = [node.stage.lower() for node in signals.custody_chain]
        chain_actions = [node.action.lower() for node in signals.custody_chain]

        has_model_draft = any("g0" in s or "generation" in s or "draft" in a for s, a in zip(chain_stages, chain_actions))
        has_human_edit = any("e1" in s or "edit" in a or "human" in node.actor.lower() for s, node, a in zip(chain_stages, signals.custody_chain, chain_actions))
        has_translation = any("translate" in a for a in chain_actions)
        is_light_touch = signals.workflow_context.get("touch_level") == "light" or "grammar" in signals.workflow_context.get("operations", [])

        if max_detector_score > 0.6:
            base_conf += 0.3
            if has_human_edit and not has_model_draft:
                qualifiers.append(ProvenanceQualifier.HUMAN_PRIMARY_DRAFTER_MODEL_EDITOR)
                reasons.append("Watermark present in human-drafted content with AI editing.")
            elif is_light_touch:
                qualifiers.append(ProvenanceQualifier.LIGHTLY_TOUCHED)
                reasons.append("Model lightly touched text (e.g. grammar/formatting).")
            elif has_translation:
                qualifiers.append(ProvenanceQualifier.TRANSLATED)
                reasons.append("Model involved in translation transform.")
            elif has_model_draft and has_human_edit:
                qualifiers.append(ProvenanceQualifier.MODEL_INVOLVED_NOT_AUTHORED)
                qualifiers.append(ProvenanceQualifier.HUMAN_EDITED)
                reasons.append("Model generated initial draft, subsequently edited by human.")
            elif has_model_draft:
                qualifiers.append(ProvenanceQualifier.MODEL_PRIMARY_DRAFTER)
                reasons.append("Model identified as primary drafter from custody log.")
            else:
                qualifiers.append(ProvenanceQualifier.MODEL_INVOLVED_NOT_AUTHORED)
                reasons.append("Watermark detected; model involvement confirmed without full authorship claim.")
        else:
            base_conf -= 0.1
            if has_model_draft:
                qualifiers.append(ProvenanceQualifier.MODEL_INVOLVED_NOT_AUTHORED)
                reasons.append("No watermark signal detected, but custody chain confirms model drafting.")
            elif has_human_edit:
                qualifiers.append(ProvenanceQualifier.HUMAN_EDITED)
                reasons.append("No watermark signal detected; human edit confirmed.")
            else:
                qualifiers.append(ProvenanceQualifier.UNKNOWN_ROLE_MODEL_INVOLVEMENT)
                reasons.append("Absence of watermark signal does not confirm human authorship.")

        if len(signals.custody_chain) >= 3:
            base_conf += 0.15

        return qualifiers, base_conf

    def _commit_decision(self, decision: WatermarkProvenanceDecision, signals: WatermarkProvenanceSignals) -> None:
        if self.policy.track_decision_history:
            self._history.append(decision)
        if self.policy.enable_audit_events:
            self._events.append(WatermarkProvenanceEvent(
                event_type="provenance_evaluation",
                control_domain=signals.control_domain,
                timestamp=decision.timestamp,
                decision=decision,
            ))

    def get_trust_level(self) -> WatermarkProvenanceTrustLevel:
        if not self._history:
            return WatermarkProvenanceTrustLevel.NO_ACTION
        return self._history[-1].authorized_level

    def get_health(self) -> WatermarkProvenanceHealthStatus:
        if not self._history:
            return WatermarkProvenanceHealthStatus.OPTIMAL
        return self._history[-1].health_status

    def get_subsystem_health(self) -> Dict[str, WatermarkProvenanceHealthStatus]:
        return {"default": self.get_health()}

    def get_decision_history(self) -> List[WatermarkProvenanceDecision]:
        return list(self._history)

    def get_events(self) -> List[WatermarkProvenanceEvent]:
        return list(self._events)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "decisions_made": len(self._history),
            "events_logged": len(self._events),
        }


# Helper and factory functions
def get_health(governor: WatermarkProvenanceGovernor) -> WatermarkProvenanceHealthStatus:
    return governor.get_health()

def get_subsystem_health(governor: WatermarkProvenanceGovernor) -> Dict[str, WatermarkProvenanceHealthStatus]:
    return governor.get_subsystem_health()

def get_metrics(governor: WatermarkProvenanceGovernor) -> Dict[str, Any]:
    return governor.get_metrics()

def get_events(governor: WatermarkProvenanceGovernor) -> List[WatermarkProvenanceEvent]:
    return governor.get_events()

def get_decision_history(governor: WatermarkProvenanceGovernor) -> List[WatermarkProvenanceDecision]:
    return governor.get_decision_history()

def create_watermark_provenance_governor(policy: Optional[WatermarkProvenancePolicy] = None, **policy_overrides: Any) -> WatermarkProvenanceGovernor:
    if policy is None:
        policy = WatermarkProvenancePolicy(**policy_overrides)
    return WatermarkProvenanceGovernor(policy=policy)

def create_default_governor(**policy_overrides: Any) -> WatermarkProvenanceGovernor:
    return create_watermark_provenance_governor(**policy_overrides)

def create_strict_governor(**policy_overrides: Any) -> WatermarkProvenanceGovernor:
    overrides = {
        "min_trust_for_action": WatermarkProvenanceTrustLevel.SUPERVISED,
        "require_corroboration_for_high_stakes": True,
        "enable_audit_events": True,
        "track_decision_history": True,
    }
    overrides.update(policy_overrides)
    return create_watermark_provenance_governor(**overrides)

def create_permissive_governor(**policy_overrides: Any) -> WatermarkProvenanceGovernor:
    overrides = {
        "min_trust_for_action": WatermarkProvenanceTrustLevel.ADVISORY,
        "require_corroboration_for_high_stakes": False,
        "enable_audit_events": False,
        "track_decision_history": True,
    }
    overrides.update(policy_overrides)
    return create_watermark_provenance_governor(**overrides)

def validate_watermark_provenance_signals(signals: WatermarkProvenanceSignals) -> List[str]:
    issues: List[str] = []
    if not isinstance(signals, WatermarkProvenanceSignals):
        return ["signals must be a WatermarkProvenanceSignals instance"]
    if not isinstance(signals.control_domain, WatermarkProvenanceControlDomain):
        issues.append("control_domain must be a WatermarkProvenanceControlDomain")
    if not isinstance(signals.proposed_action, WatermarkProvenanceControlAction):
        issues.append("proposed_action must be a WatermarkProvenanceControlAction")
    if not isinstance(signals.watermark_signals, list):
        issues.append("watermark_signals must be a list")
        watermark_signals = []
    else:
        watermark_signals = signals.watermark_signals
    if len(watermark_signals) > 10000:
        issues.append("watermark_signals exceeds maximum size 10000")
    for idx, sig in enumerate(watermark_signals[:10000]):
        if not isinstance(sig, WatermarkSignalData):
            issues.append(f"watermark_signals[{idx}] must be WatermarkSignalData")
            continue
        if not isinstance(sig.detector_name, str) or not sig.detector_name.strip():
            issues.append(f"watermark_signals[{idx}].detector_name must be non-empty")
        if not isinstance(sig.raw_score, (int, float)) or isinstance(sig.raw_score, bool) or not math.isfinite(sig.raw_score) or not (0.0 <= sig.raw_score <= 1.0):
            issues.append(f"watermark_signals[{idx}].raw_score must be between 0.0 and 1.0")
        if not isinstance(sig.confidence, (int, float)) or isinstance(sig.confidence, bool) or not math.isfinite(sig.confidence) or not (0.0 <= sig.confidence <= 1.0):
            issues.append(f"watermark_signals[{idx}].confidence must be between 0.0 and 1.0")
        if not isinstance(sig.presence_detected, bool):
            issues.append(f"watermark_signals[{idx}].presence_detected must be boolean")
        if not isinstance(sig.details, dict):
            issues.append(f"watermark_signals[{idx}].details must be a mapping")
    if not isinstance(signals.custody_chain, list):
        issues.append("custody_chain must be a list")
        custody_chain = []
    else:
        custody_chain = signals.custody_chain
    if len(custody_chain) > 10000:
        issues.append("custody_chain exceeds maximum size 10000")
    seen_ids = set()
    previous_timestamp = -math.inf
    for idx, node in enumerate(custody_chain[:10000]):
        if not isinstance(node, ProvenanceEventNode):
            issues.append(f"custody_chain[{idx}] must be ProvenanceEventNode")
            continue
        if not all(isinstance(value, str) and value.strip() for value in (node.event_id, node.stage, node.actor, node.action)):
            issues.append(f"custody_chain[{idx}] identifiers and descriptors must be non-empty")
        if node.event_id in seen_ids:
            issues.append(f"custody_chain[{idx}].event_id must be unique")
        seen_ids.add(node.event_id)
        if not isinstance(node.timestamp, (int, float)) or isinstance(node.timestamp, bool) or not math.isfinite(node.timestamp) or node.timestamp < 0:
            issues.append(f"custody_chain[{idx}].timestamp must be a finite non-negative number")
        elif node.timestamp < previous_timestamp:
            issues.append("custody_chain timestamps must be ordered")
        else:
            previous_timestamp = node.timestamp
        if not isinstance(node.metadata, dict):
            issues.append(f"custody_chain[{idx}].metadata must be a mapping")
    if not isinstance(signals.suspected_disruptions, list) or not all(isinstance(v, str) for v in signals.suspected_disruptions):
        issues.append("suspected_disruptions must be a list of strings")
    if not isinstance(signals.workflow_context, dict):
        issues.append("workflow_context must be a mapping")
    if not isinstance(signals.context_type, str) or not signals.context_type.strip():
        issues.append("context_type must be non-empty")
    return issues
