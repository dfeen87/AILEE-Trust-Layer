"""
AILEE Trust Layer — Topology Domain
Version: 4.2.0

First-class AILEE domain implementation for Trust Layer topology governance.

This domain applies the AILEE Trust Pipeline to:
- Node connectivity and graph integrity
- Trust relationship validation between domains
- Deployment graph health and consistency
- Structural integrity of the component mesh
- Route and path reliability scoring

The module defines domain-specific configurations, signal types, governed
controllers, and a unified governance interface for production topology
environments.

Quick Start:
    >>> from ailee.domains.topology import (
    ...     TopologyGovernor,
    ...     TopologyPolicy,
    ...     TopologySignals,
    ...     TopologyTrustLevel,
    ...     TopologyControlDomain,
    ...     TopologyControlAction,
    ...     TopologyReading,
    ...     create_topology_governor,
    ... )
    >>>
    >>> governor = create_topology_governor()
    >>>
    >>> signals = TopologySignals(
    ...     control_domain=TopologyControlDomain.NODE_CONNECTIVITY,
    ...     proposed_action=TopologyControlAction.REBALANCE,
    ...     ai_value=0.91,
    ...     ai_confidence=0.87,
    ...     topology_readings=[
    ...         TopologyReading(0.92, time.time(), "node_cluster_a"),
    ...         TopologyReading(0.89, time.time(), "node_cluster_b"),
    ...     ],
    ...     zone_id="region_west",
    ... )
    >>>
    >>> decision = governor.evaluate(signals)
    >>> if decision.actionable:
    ...     mesh.rebalance(decision.trusted_value)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

# Import core AILEE components
from ...ailee_trust_pipeline_v1 import AileeTrustPipeline, AileeConfig, DecisionResult
from ...optional.ailee_monitors import TrustMonitor, AlertingMonitor
from ...optional.ailee_peer_adapters import StaticPeerAdapter, FilteredPeerAdapter, MultiSourcePeerAdapter


# ===========================
# Topology-Specific Configs
# ===========================

NODE_CONNECTIVITY = AileeConfig(
    # High confidence required — connectivity decisions affect reachability
    accept_threshold=0.92,
    borderline_low=0.78,
    borderline_high=0.92,

    # Normalized connectivity score [0.0, 1.0]
    hard_min=0.0,
    hard_max=1.0,

    # Require quorum across node probes
    consensus_quorum=4,
    consensus_delta=0.08,  # 8% tolerance on connectivity scores

    grace_peer_delta=0.10,
    grace_forecast_epsilon=0.08,

    # Fall back to last known good topology state
    fallback_mode="last_good",

    # Stability is critical — topology churn is expensive
    w_stability=0.50,
    w_agreement=0.30,
    w_likelihood=0.20,

    history_window=100,
    forecast_window=15,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


TRUST_RELATIONSHIPS = AileeConfig(
    # Very high bar — trust relationship mutations are high-risk
    accept_threshold=0.95,
    borderline_low=0.85,
    borderline_high=0.95,

    # Normalized trust relationship integrity score [0.0, 1.0]
    hard_min=0.0,
    hard_max=1.0,

    # Strong quorum — multiple validators must agree
    consensus_quorum=5,
    consensus_delta=0.05,  # 5% tolerance; trust is binary-leaning

    grace_peer_delta=0.06,

    fallback_mode="last_good",

    # High stability weight — trust relationships must not oscillate
    w_stability=0.60,
    w_agreement=0.25,
    w_likelihood=0.15,

    history_window=150,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


DEPLOYMENT_GRAPH = AileeConfig(
    # Moderate threshold — deployment graph changes are disruptive but reversible
    accept_threshold=0.88,
    borderline_low=0.72,
    borderline_high=0.88,

    # Normalized deployment health score [0.0, 1.0]
    hard_min=0.0,
    hard_max=1.0,

    # Moderate quorum across deployment probes
    consensus_quorum=3,
    consensus_delta=0.12,

    grace_peer_delta=0.15,

    fallback_mode="median",

    w_stability=0.40,
    w_agreement=0.35,
    w_likelihood=0.25,

    history_window=80,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


STRUCTURAL_INTEGRITY = AileeConfig(
    # High threshold — structural integrity is foundational
    accept_threshold=0.93,
    borderline_low=0.80,
    borderline_high=0.93,

    # Normalized structural score [0.0, 1.0]
    hard_min=0.0,
    hard_max=1.0,

    consensus_quorum=4,
    consensus_delta=0.07,

    grace_peer_delta=0.08,

    fallback_mode="last_good",

    # Heaviest stability weight — structural changes are slow and costly
    w_stability=0.55,
    w_agreement=0.28,
    w_likelihood=0.17,

    history_window=120,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


ROUTE_RELIABILITY = AileeConfig(
    # Moderate-high threshold — routes can degrade gracefully
    accept_threshold=0.90,
    borderline_low=0.75,
    borderline_high=0.90,

    # Normalized path reliability score [0.0, 1.0]
    hard_min=0.0,
    hard_max=1.0,

    consensus_quorum=3,
    consensus_delta=0.10,

    grace_peer_delta=0.12,

    fallback_mode="median",

    w_stability=0.42,
    w_agreement=0.33,
    w_likelihood=0.25,

    history_window=60,

    enable_grace=True,
    enable_consensus=True,
    enable_audit_metadata=True,
)


# ===========================
# Enumerations
# ===========================

class TopologyTrustLevel(IntEnum):
    """
    Graduated trust levels for topology governance decisions.
    Higher values indicate more authority to act autonomously.
    """
    NO_ACTION = 0    # Insufficient data or confidence — do not act
    ADVISORY = 1     # Log or alert only — no structural change
    SUPERVISED = 2   # Act, but flag for operator review
    AUTONOMOUS = 3   # Fully authorized autonomous action


class TopologyHealthStatus(str, Enum):
    """Overall health status of a topology domain or subsystem."""
    OPTIMAL = "OPTIMAL"
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class TopologyControlDomain(str, Enum):
    """Topology control domains governed by AILEE."""
    NODE_CONNECTIVITY = "NODE_CONNECTIVITY"
    TRUST_RELATIONSHIPS = "TRUST_RELATIONSHIPS"
    DEPLOYMENT_GRAPH = "DEPLOYMENT_GRAPH"
    STRUCTURAL_INTEGRITY = "STRUCTURAL_INTEGRITY"
    ROUTE_RELIABILITY = "ROUTE_RELIABILITY"


class TopologyControlAction(str, Enum):
    """Types of topology control actions that can be governed."""
    REBALANCE = "REBALANCE"           # Rebalance node connectivity weights
    PROMOTE = "PROMOTE"               # Promote a trust relationship
    REVOKE = "REVOKE"                 # Revoke or demote a trust relationship
    REDEPLOY = "REDEPLOY"             # Trigger a deployment graph update
    REPAIR = "REPAIR"                 # Initiate structural repair
    REROUTE = "REROUTE"               # Switch to an alternate route/path
    ISOLATE = "ISOLATE"               # Isolate a degraded node or domain
    NO_ACTION = "NO_ACTION"


# ===========================
# Policy & Signal Types
# ===========================

@dataclass
class TopologyPolicy:
    """
    Governance policy for topology control decisions.

    Controls trust thresholds, safety constraints, and audit settings
    applied by the TopologyGovernor.
    """
    # Minimum trust level required before a topology action is permitted
    min_trust_for_action: TopologyTrustLevel = TopologyTrustLevel.SUPERVISED

    # Require peer probe consensus before accepting a topology value
    require_consensus: bool = True

    # Minimum acceptable structural integrity score before allowing
    # PROMOTE or REDEPLOY actions (normalized 0.0–1.0)
    min_integrity_score: float = 0.75

    # Maximum number of topology rebalances permitted per hour
    max_rebalances_per_hour: int = 30

    # Maximum number of trust relationship mutations per hour
    max_trust_mutations_per_hour: int = 10

    # Audit settings
    enable_audit_events: bool = True
    track_decision_history: bool = True


@dataclass
class TopologyReading:
    """Standard topology probe reading."""
    value: float                          # Normalized score [0.0, 1.0]
    timestamp: float
    probe_id: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologySignals:
    """
    Input signals for a topology governance evaluation.

    Describes the AI-proposed topology action together with supporting
    probe data and contextual metadata.
    """
    # What is being controlled and what action is proposed
    control_domain: TopologyControlDomain
    proposed_action: TopologyControlAction

    # AI-produced normalized score [0.0, 1.0] representing the proposed
    # topology state (connectivity score, trust integrity, deployment health, etc.)
    ai_value: float

    # AI confidence in the proposed value [0.0, 1.0]
    ai_confidence: float

    # Supporting topology probe readings used for consensus
    topology_readings: List[TopologyReading] = field(default_factory=list)

    # Contextual identifiers
    zone_id: str = "default"
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    domain_pair: Optional[Tuple[str, str]] = None   # For trust relationship signals

    # Evaluation timestamp (defaults to current time)
    timestamp: Optional[float] = None

    # Additional free-form context passed through to the audit trail
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyDecision:
    """
    Result of a topology governance evaluation.

    Carries the authorized trust level, the validated/fallback score to
    act on, and a full audit trail suitable for mesh or orchestration integration.
    """
    # Governance outcome
    authorized_level: TopologyTrustLevel
    actionable: bool

    # Trusted normalized score to use for topology actuation [0.0, 1.0]
    trusted_value: float

    # What was evaluated
    control_domain: TopologyControlDomain
    proposed_action: TopologyControlAction

    # Raw AILEE pipeline result
    pipeline_result: Optional[Any] = None

    # Safety and health
    health_status: TopologyHealthStatus = TopologyHealthStatus.OPTIMAL
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
class TopologyEvent:
    """
    Audit event emitted by the TopologyGovernor.

    Provides a structured, time-stamped record of each governance decision
    for logging, compliance, and post-incident topology analysis.
    """
    event_type: str          # "decision", "fallback", "health_change", "rate_limit"
    control_domain: TopologyControlDomain
    timestamp: float = field(default_factory=time.time)
    decision: Optional[TopologyDecision] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ===========================
# Topology Probe Processors
# ===========================

class TopologyProbeProcessor:
    """
    Process raw topology probe data.
    Handles normalization, filtering, and formatting for AILEE.
    """

    def __init__(self):
        self.probe_history: Dict[str, List[TopologyReading]] = {}

    def process_connectivity_probe(
        self,
        raw_score: float,
        probe_id: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> TopologyReading:
        """
        Process a raw connectivity probe score.

        Args:
            raw_score: Connectivity score, expected in [0.0, 1.0]
            probe_id: Probe or node identifier
            metadata: Additional context

        Returns:
            Normalized TopologyReading
        """
        normalized = max(0.0, min(1.0, raw_score))
        reading = TopologyReading(
            value=normalized,
            timestamp=time.time(),
            probe_id=probe_id,
            metadata=metadata or {},
        )
        self._store(probe_id, reading)
        return reading

    def process_latency_probe(
        self,
        latency_ms: float,
        max_acceptable_ms: float = 200.0,
        probe_id: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> TopologyReading:
        """
        Convert a raw latency reading into a normalized reliability score.

        Score = 1.0 at 0ms, decaying linearly to 0.0 at max_acceptable_ms.

        Args:
            latency_ms: Measured latency in milliseconds
            max_acceptable_ms: Latency ceiling for a score of 0.0
            probe_id: Probe identifier
            metadata: Additional context

        Returns:
            TopologyReading with normalized score in [0.0, 1.0]
        """
        score = max(0.0, 1.0 - (latency_ms / max_acceptable_ms))
        reading = TopologyReading(
            value=score,
            timestamp=time.time(),
            probe_id=probe_id,
            metadata={"latency_ms": latency_ms, "max_ms": max_acceptable_ms, **(metadata or {})},
        )
        self._store(probe_id, reading)
        return reading

    def get_peer_values(
        self,
        probe_type: str,
        max_age_seconds: float = 60.0,
    ) -> List[float]:
        """
        Get recent peer probe values for consensus.

        Args:
            probe_type: Type fragment to match probe IDs (e.g., "cluster")
            max_age_seconds: Maximum age of readings to include

        Returns:
            List of recent normalized scores
        """
        now = time.time()
        peer_values = []
        for probe_id, readings in self.probe_history.items():
            if probe_type in probe_id and readings:
                latest = readings[-1]
                if (now - latest.timestamp) <= max_age_seconds:
                    peer_values.append(latest.value)
        return peer_values

    def _store(self, probe_id: str, reading: TopologyReading) -> None:
        if probe_id not in self.probe_history:
            self.probe_history[probe_id] = []
        self.probe_history[probe_id].append(reading)


# ===========================
# Domain Controllers
# ===========================

class NodeConnectivityController:
    """
    Node connectivity controller with AILEE governance.
    Governs rebalancing and isolation decisions for the node mesh.
    """

    def __init__(
        self,
        config: Optional[AileeConfig] = None,
        monitor: Optional[TrustMonitor] = None,
    ):
        self.config = config or NODE_CONNECTIVITY
        self.pipeline = AileeTrustPipeline(self.config)
        self.monitor = monitor or TrustMonitor(window=200)
        self.probe_processor = TopologyProbeProcessor()
        self.current_connectivity_score: float = 1.0

    def propose_rebalance(
        self,
        ai_score: float,
        ai_confidence: float,
        readings: List[TopologyReading],
        zone_id: str = "default",
    ) -> DecisionResult:
        """
        Evaluate an AI-proposed connectivity rebalance.

        Args:
            ai_score: Proposed post-rebalance connectivity score [0.0, 1.0]
            ai_confidence: AI model confidence [0.0, 1.0]
            readings: Supporting node probe readings
            zone_id: Topology zone identifier

        Returns:
            DecisionResult with trusted connectivity score

        Example:
            >>> controller = NodeConnectivityController()
            >>> readings = [
            ...     TopologyReading(0.91, time.time(), "probe_a"),
            ...     TopologyReading(0.89, time.time(), "probe_b"),
            ... ]
            >>> result = controller.propose_rebalance(0.90, 0.85, readings, "zone_west")
            >>> if not result.used_fallback:
            ...     mesh.rebalance(result.value)
        """
        peer_values = [r.value for r in readings]
        result = self.pipeline.process(
            raw_value=ai_score,
            raw_confidence=ai_confidence,
            peer_values=peer_values,
            timestamp=time.time(),
            context={
                "zone": zone_id,
                "current_connectivity": self.current_connectivity_score,
                "probe_count": len(readings),
            },
        )
        self.monitor.record(result)
        if not result.used_fallback:
            self.current_connectivity_score = result.value
        return result

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "current_connectivity_score": self.current_connectivity_score,
            "fallback_rate": self.monitor.fallback_rate(),
            "avg_confidence": self.monitor.avg_confidence(),
            "decisions_total": self.monitor.total_decisions,
        }


class TrustRelationshipController:
    """
    Trust relationship controller with AILEE governance.
    Governs promote, revoke, and mutation decisions for inter-domain trust.
    """

    def __init__(
        self,
        config: Optional[AileeConfig] = None,
        monitor: Optional[TrustMonitor] = None,
    ):
        self.config = config or TRUST_RELATIONSHIPS
        self.pipeline = AileeTrustPipeline(self.config)
        self.monitor = monitor or TrustMonitor(window=200)
        self.mutations_last_hour: int = 0
        self.last_hour_reset: float = time.time()

    def propose_mutation(
        self,
        domain_pair: Tuple[str, str],
        action: TopologyControlAction,
        ai_integrity_score: float,
        ai_confidence: float,
        peer_scores: List[float],
    ) -> Tuple[bool, Optional[DecisionResult]]:
        """
        Evaluate a proposed trust relationship mutation (promote or revoke).

        Args:
            domain_pair: Tuple of (source_domain, target_domain)
            action: PROMOTE or REVOKE
            ai_integrity_score: Proposed post-mutation integrity score [0.0, 1.0]
            ai_confidence: AI model confidence [0.0, 1.0]
            peer_scores: Integrity scores from peer validators

        Returns:
            Tuple of (should_mutate, decision_result)
        """
        now = time.time()
        if (now - self.last_hour_reset) > 3600:
            self.mutations_last_hour = 0
            self.last_hour_reset = now

        if self.mutations_last_hour >= 10:
            return False, None

        result = self.pipeline.process(
            raw_value=ai_integrity_score,
            raw_confidence=ai_confidence,
            peer_values=peer_scores,
            timestamp=now,
            context={
                "source_domain": domain_pair[0],
                "target_domain": domain_pair[1],
                "action": action.value,
                "mutations_this_hour": self.mutations_last_hour,
            },
        )
        self.monitor.record(result)

        should_mutate = (
            result.safety_status == "ACCEPTED"
            and not result.used_fallback
            and result.confidence_score >= 0.90
        )
        if should_mutate:
            self.mutations_last_hour += 1

        return should_mutate, result


class DeploymentGraphController:
    """
    Deployment graph controller with AILEE governance.
    Governs redeployment and graph update decisions.
    """

    def __init__(
        self,
        config: Optional[AileeConfig] = None,
        monitor: Optional[TrustMonitor] = None,
    ):
        self.config = config or DEPLOYMENT_GRAPH
        self.pipeline = AileeTrustPipeline(self.config)
        self.monitor = monitor or TrustMonitor(window=150)
        self.current_graph_health: float = 1.0

    def propose_redeploy(
        self,
        ai_health_score: float,
        ai_confidence: float,
        readings: List[TopologyReading],
        zone_id: str = "default",
    ) -> DecisionResult:
        """
        Evaluate an AI-proposed deployment graph update.

        Args:
            ai_health_score: Proposed post-redeploy graph health [0.0, 1.0]
            ai_confidence: AI model confidence [0.0, 1.0]
            readings: Supporting deployment probe readings
            zone_id: Deployment zone identifier

        Returns:
            DecisionResult with trusted health score
        """
        peer_values = [r.value for r in readings]
        result = self.pipeline.process(
            raw_value=ai_health_score,
            raw_confidence=ai_confidence,
            peer_values=peer_values,
            timestamp=time.time(),
            context={
                "zone": zone_id,
                "current_graph_health": self.current_graph_health,
                "probe_count": len(readings),
            },
        )
        self.monitor.record(result)
        if not result.used_fallback:
            self.current_graph_health = result.value
        return result


# ===========================
# Monitoring Integration
# ===========================

class TopologyMonitor:
    """
    Unified monitoring for topology AILEE deployments.
    Tracks connectivity, trust, deployment, structural, and route metrics.
    """

    def __init__(self):
        self.connectivity_monitor = TrustMonitor(window=500)
        self.trust_monitor = TrustMonitor(window=500)
        self.deployment_monitor = TrustMonitor(window=300)
        self.structural_monitor = TrustMonitor(window=400)
        self.route_monitor = TrustMonitor(window=300)

    def get_unified_metrics(self) -> Dict[str, Any]:
        """Get governance metrics across all topology domains."""
        return {
            "connectivity": {
                "fallback_rate": self.connectivity_monitor.fallback_rate(),
                "avg_confidence": self.connectivity_monitor.avg_confidence(),
                "total_decisions": self.connectivity_monitor.total_decisions,
            },
            "trust_relationships": {
                "fallback_rate": self.trust_monitor.fallback_rate(),
                "avg_confidence": self.trust_monitor.avg_confidence(),
                "total_decisions": self.trust_monitor.total_decisions,
            },
            "deployment_graph": {
                "fallback_rate": self.deployment_monitor.fallback_rate(),
                "avg_confidence": self.deployment_monitor.avg_confidence(),
                "total_decisions": self.deployment_monitor.total_decisions,
            },
            "structural_integrity": {
                "fallback_rate": self.structural_monitor.fallback_rate(),
                "avg_confidence": self.structural_monitor.avg_confidence(),
                "total_decisions": self.structural_monitor.total_decisions,
            },
            "route_reliability": {
                "fallback_rate": self.route_monitor.fallback_rate(),
                "avg_confidence": self.route_monitor.avg_confidence(),
                "total_decisions": self.route_monitor.total_decisions,
            },
        }

    def check_health(self) -> Dict[str, str]:
        """
        Check health status per topology domain.

        Returns:
            Dict mapping domain name to health string.
        """
        def _status(monitor: TrustMonitor, warn: float, degrade: float) -> str:
            rate = monitor.fallback_rate()
            if rate > degrade:
                return "DEGRADED"
            if rate > warn:
                return "WARNING"
            return "HEALTHY"

        return {
            "connectivity":        _status(self.connectivity_monitor,  0.12, 0.25),
            "trust_relationships": _status(self.trust_monitor,          0.08, 0.18),
            "deployment_graph":    _status(self.deployment_monitor,     0.15, 0.28),
            "structural_integrity":_status(self.structural_monitor,     0.10, 0.22),
            "route_reliability":   _status(self.route_monitor,          0.12, 0.25),
        }


# ===========================
# Unified Governance Interface
# ===========================

class TopologyGovernor:
    """
    Unified governance governor for Trust Layer topology systems.

    Evaluates AI-proposed topology actions across node connectivity, trust
    relationships, deployment graph, structural integrity, and route reliability
    domains using the AILEE Trust Pipeline, and returns graduated-trust
    ``TopologyDecision`` objects with full audit trails.

    This is the primary entry point for integrating AILEE into mesh
    orchestrators, deployment managers, or topology validation systems.

    Example::

        governor = TopologyGovernor()

        signals = TopologySignals(
            control_domain=TopologyControlDomain.NODE_CONNECTIVITY,
            proposed_action=TopologyControlAction.REBALANCE,
            ai_value=0.91,
            ai_confidence=0.87,
            topology_readings=[
                TopologyReading(0.92, time.time(), "probe_cluster_a"),
                TopologyReading(0.89, time.time(), "probe_cluster_b"),
            ],
            zone_id="region_west",
        )

        decision = governor.evaluate(signals)
        if decision.actionable:
            mesh.rebalance(decision.trusted_value)
        elif decision.used_fallback:
            logger.warning("Fallback active: %s", decision.fallback_reason)
    """

    def __init__(
        self,
        policy: Optional[TopologyPolicy] = None,
        connectivity_config: Optional[AileeConfig] = None,
        trust_config: Optional[AileeConfig] = None,
        deployment_config: Optional[AileeConfig] = None,
        structural_config: Optional[AileeConfig] = None,
        route_config: Optional[AileeConfig] = None,
    ):
        """
        Initialise the TopologyGovernor.

        Args:
            policy:              Governance policy. Defaults to ``TopologyPolicy()``.
            connectivity_config: AILEE config for node connectivity decisions.
                                 Defaults to ``NODE_CONNECTIVITY``.
            trust_config:        AILEE config for trust relationship decisions.
                                 Defaults to ``TRUST_RELATIONSHIPS``.
            deployment_config:   AILEE config for deployment graph decisions.
                                 Defaults to ``DEPLOYMENT_GRAPH``.
            structural_config:   AILEE config for structural integrity decisions.
                                 Defaults to ``STRUCTURAL_INTEGRITY``.
            route_config:        AILEE config for route reliability decisions.
                                 Defaults to ``ROUTE_RELIABILITY``.
        """
        self.policy = policy or TopologyPolicy()

        # One AILEE pipeline per control domain
        self._pipelines: Dict[str, AileeTrustPipeline] = {
            TopologyControlDomain.NODE_CONNECTIVITY:   AileeTrustPipeline(connectivity_config or NODE_CONNECTIVITY),
            TopologyControlDomain.TRUST_RELATIONSHIPS: AileeTrustPipeline(trust_config or TRUST_RELATIONSHIPS),
            TopologyControlDomain.DEPLOYMENT_GRAPH:    AileeTrustPipeline(deployment_config or DEPLOYMENT_GRAPH),
            TopologyControlDomain.STRUCTURAL_INTEGRITY:AileeTrustPipeline(structural_config or STRUCTURAL_INTEGRITY),
            TopologyControlDomain.ROUTE_RELIABILITY:   AileeTrustPipeline(route_config or ROUTE_RELIABILITY),
        }

        # Track metrics and events
        self._monitor = TrustMonitor(window=500)
        self._events: List[TopologyEvent] = []
        self._decision_history: List[TopologyDecision] = []

        # Per-domain rate limiting
        self._rebalances_this_hour: int = 0
        self._trust_mutations_this_hour: int = 0
        self._last_hour_reset: float = time.time()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def evaluate(self, signals: TopologySignals) -> TopologyDecision:
        """
        Evaluate an AI-proposed topology control action.

        Runs the AILEE Trust Pipeline for the appropriate topology domain,
        applies policy constraints, determines the trust level, and returns
        a ``TopologyDecision`` with a trusted value and full audit trail.

        Args:
            signals: ``TopologySignals`` describing the proposed action
                     and supporting probe data.

        Returns:
            ``TopologyDecision`` with an authorized trust level and the
            normalized score that is safe to act on.
        """
        ts = signals.timestamp or time.time()
        peer_values = [r.value for r in signals.topology_readings]

        # --- Validate signals ---
        issues = validate_topology_signals(signals)
        if issues:
            decision = self._build_no_action_decision(
                signals=signals,
                ts=ts,
                reason=f"Signal validation failed: {'; '.join(issues)}",
            )
            self._record_event("validation_failed", signals, decision)
            return decision

        # --- Rate limiting ---
        self._refresh_rate_limits()

        if signals.control_domain == TopologyControlDomain.NODE_CONNECTIVITY:
            if self._rebalances_this_hour >= self.policy.max_rebalances_per_hour:
                decision = self._build_no_action_decision(
                    signals=signals,
                    ts=ts,
                    reason="Rate limit reached: too many rebalances this hour",
                )
                self._record_event("rate_limit", signals, decision)
                return decision

        if signals.control_domain == TopologyControlDomain.TRUST_RELATIONSHIPS:
            if self._trust_mutations_this_hour >= self.policy.max_trust_mutations_per_hour:
                decision = self._build_no_action_decision(
                    signals=signals,
                    ts=ts,
                    reason="Rate limit reached: too many trust mutations this hour",
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
        if signals.source_node:
            context["source_node"] = signals.source_node
        if signals.target_node:
            context["target_node"] = signals.target_node
        if signals.domain_pair:
            context["domain_pair"] = list(signals.domain_pair)

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
            reasons.append("Pipeline rejected the proposed topology value")

        # --- Build decision ---
        decision_id = hashlib.sha256(
            f"{ts}{signals.control_domain}{signals.ai_value}".encode()
        ).hexdigest()[:16]

        decision = TopologyDecision(
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
                "probe_count": len(signals.topology_readings),
            },
        )

        # --- Track rate-limited counters ---
        if actionable and not result.used_fallback:
            if signals.control_domain == TopologyControlDomain.NODE_CONNECTIVITY:
                self._rebalances_this_hour += 1
            if signals.control_domain == TopologyControlDomain.TRUST_RELATIONSHIPS:
                self._trust_mutations_this_hour += 1

        if self.policy.track_decision_history:
            self._decision_history.append(decision)
        if self.policy.enable_audit_events:
            event_type = "fallback" if result.used_fallback else "decision"
            self._record_event(event_type, signals, decision)

        return decision

    # ------------------------------------------------------------------
    # Health & Monitoring
    # ------------------------------------------------------------------

    def get_health(self) -> TopologyHealthStatus:
        """
        Compute the overall health status of the topology layer based on
        recent governance metrics.

        Returns:
            ``TopologyHealthStatus`` enum value.
        """
        fallback_rate = self._monitor.fallback_rate()
        if fallback_rate > 0.30:
            return TopologyHealthStatus.CRITICAL
        if fallback_rate > 0.20:
            return TopologyHealthStatus.DEGRADED
        if fallback_rate > 0.10:
            return TopologyHealthStatus.WARNING
        return TopologyHealthStatus.OPTIMAL

    def get_subsystem_health(self) -> Dict[str, TopologyHealthStatus]:
        """
        Return health status per topology control domain.

        Returns:
            Dict mapping ``TopologyControlDomain`` value strings to
            ``TopologyHealthStatus`` values.
        """
        return {
            domain.value: self._get_subsystem_health(domain)
            for domain in TopologyControlDomain
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return aggregated governance metrics.

        Returns:
            Dict with fallback rate, average confidence, decision counts,
            and rate-limited action counters.
        """
        return {
            "fallback_rate": self._monitor.fallback_rate(),
            "avg_confidence": self._monitor.avg_confidence(),
            "total_decisions": self._monitor.total_decisions,
            "rebalances_this_hour": self._rebalances_this_hour,
            "trust_mutations_this_hour": self._trust_mutations_this_hour,
            "overall_health": self.get_health().value,
        }

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def get_events(self) -> List[TopologyEvent]:
        """Return all recorded governance events (newest last)."""
        return list(self._events)

    def get_decision_history(self) -> List[TopologyDecision]:
        """Return the history of all governance decisions."""
        return list(self._decision_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_trust_level(self, result: Any) -> TopologyTrustLevel:
        """Map an AILEE DecisionResult to a TopologyTrustLevel."""
        if result.safety_status == "OUTRIGHT_REJECTED":
            return TopologyTrustLevel.NO_ACTION
        if result.used_fallback:
            return TopologyTrustLevel.ADVISORY
        confidence = getattr(result, "confidence_score", 0.0)
        if result.safety_status == "ACCEPTED" and confidence >= 0.90:
            return TopologyTrustLevel.AUTONOMOUS
        if result.safety_status in ("ACCEPTED", "BORDERLINE") and confidence >= 0.70:
            return TopologyTrustLevel.SUPERVISED
        return TopologyTrustLevel.ADVISORY

    def _get_subsystem_health(self, domain: TopologyControlDomain) -> TopologyHealthStatus:
        """Return health status for a single topology domain (uses overall monitor)."""
        return self.get_health()

    def _refresh_rate_limits(self) -> None:
        """Reset hourly counters if the window has elapsed."""
        now = time.time()
        if (now - self._last_hour_reset) > 3600:
            self._rebalances_this_hour = 0
            self._trust_mutations_this_hour = 0
            self._last_hour_reset = now

    def _build_no_action_decision(
        self,
        signals: TopologySignals,
        ts: float,
        reason: str,
    ) -> TopologyDecision:
        """Build a NO_ACTION decision without running the pipeline."""
        decision_id = hashlib.sha256(
            f"{ts}{signals.control_domain}{signals.ai_value}".encode()
        ).hexdigest()[:16]
        return TopologyDecision(
            authorized_level=TopologyTrustLevel.NO_ACTION,
            actionable=False,
            trusted_value=signals.ai_value,
            control_domain=signals.control_domain,
            proposed_action=signals.proposed_action,
            pipeline_result=None,
            health_status=TopologyHealthStatus.WARNING,
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
        signals: TopologySignals,
        decision: TopologyDecision,
    ) -> None:
        """Append a TopologyEvent to the internal log."""
        self._events.append(TopologyEvent(
            event_type=event_type,
            control_domain=signals.control_domain,
            timestamp=decision.timestamp,
            decision=decision,
            details={
                "zone_id": signals.zone_id,
                "ai_value": signals.ai_value,
                "ai_confidence": signals.ai_confidence,
                "probe_count": len(signals.topology_readings),
            },
        ))


# ===========================
# Convenience Factory Functions
# ===========================

def create_topology_governor(
    policy: Optional[TopologyPolicy] = None,
    **policy_overrides: Any,
) -> TopologyGovernor:
    """
    Create a ``TopologyGovernor`` with a sensible default policy.

    Args:
        policy: Optional pre-built ``TopologyPolicy``. If omitted, a default
                policy is constructed from ``policy_overrides``.
        **policy_overrides: Keyword arguments forwarded to ``TopologyPolicy()``.

    Returns:
        Configured ``TopologyGovernor`` instance.

    Example::

        governor = create_topology_governor(max_rebalances_per_hour=20)
    """
    if policy is None:
        policy = TopologyPolicy(**policy_overrides)
    return TopologyGovernor(policy=policy)


def create_default_governor(**policy_overrides: Any) -> TopologyGovernor:
    """
    Create a ``TopologyGovernor`` with default (balanced) policy settings.

    Args:
        **policy_overrides: Optional ``TopologyPolicy`` attribute overrides.

    Returns:
        ``TopologyGovernor`` instance with default policy.

    Example::

        governor = create_default_governor()
    """
    return TopologyGovernor(policy=TopologyPolicy(**policy_overrides))


def create_strict_governor(**policy_overrides: Any) -> TopologyGovernor:
    """
    Create a ``TopologyGovernor`` with a strict safety policy.

    Strict settings:
    - Requires AUTONOMOUS trust level before allowing action
    - Enforces consensus
    - High minimum integrity score
    - Tighter rate limits on rebalances and trust mutations
    - Full audit trail enabled

    Args:
        **policy_overrides: Optional further overrides.

    Returns:
        ``TopologyGovernor`` instance with strict policy.

    Example::

        governor = create_strict_governor()
    """
    strict_defaults: Dict[str, Any] = {
        "min_trust_for_action": TopologyTrustLevel.AUTONOMOUS,
        "require_consensus": True,
        "min_integrity_score": 0.90,
        "max_rebalances_per_hour": 10,
        "max_trust_mutations_per_hour": 3,
        "enable_audit_events": True,
        "track_decision_history": True,
    }
    strict_defaults.update(policy_overrides)
    return TopologyGovernor(policy=TopologyPolicy(**strict_defaults))


def create_permissive_governor(**policy_overrides: Any) -> TopologyGovernor:
    """
    Create a ``TopologyGovernor`` with a permissive policy for development
    or testing.

    Permissive settings:
    - Allows action at ADVISORY trust level
    - Does not enforce consensus
    - Low minimum integrity score
    - Higher rate limits

    **WARNING**: Not recommended for production use.

    Args:
        **policy_overrides: Optional further overrides.

    Returns:
        ``TopologyGovernor`` instance with permissive policy.

    Example::

        governor = create_permissive_governor()
    """
    permissive_defaults: Dict[str, Any] = {
        "min_trust_for_action": TopologyTrustLevel.ADVISORY,
        "require_consensus": False,
        "min_integrity_score": 0.50,
        "max_rebalances_per_hour": 200,
        "max_trust_mutations_per_hour": 50,
        "enable_audit_events": False,
        "track_decision_history": False,
    }
    permissive_defaults.update(policy_overrides)
    return TopologyGovernor(policy=TopologyPolicy(**permissive_defaults))


def validate_topology_signals(signals: TopologySignals) -> List[str]:
    """
    Validate a ``TopologySignals`` object and return a list of issue strings.

    An empty list means the signals are valid.

    Args:
        signals: ``TopologySignals`` instance to validate.

    Returns:
        List of validation error/warning strings. Empty if all checks pass.

    Example::

        issues = validate_topology_signals(signals)
        if issues:
            for issue in issues:
                logger.warning("Topology signal issue: %s", issue)
    """
    issues: List[str] = []

    if not (0.0 <= signals.ai_confidence <= 1.0):
        issues.append(
            f"ai_confidence must be in [0.0, 1.0], got {signals.ai_confidence}"
        )

    if not (0.0 <= signals.ai_value <= 1.0):
        issues.append(
            f"ai_value must be a normalized score in [0.0, 1.0], got {signals.ai_value}"
        )

    if not signals.topology_readings:
        issues.append("No topology readings provided; consensus cannot be computed")

    if (
        signals.control_domain == TopologyControlDomain.TRUST_RELATIONSHIPS
        and signals.domain_pair is None
    ):
        issues.append(
            "domain_pair must be set for TRUST_RELATIONSHIPS signals"
        )

    if (
        signals.proposed_action in (TopologyControlAction.REROUTE, TopologyControlAction.ISOLATE)
        and signals.source_node is None
    ):
        issues.append(
            f"source_node must be set for {signals.proposed_action.value} actions"
        )

    return issues


# Module-level wrappers
def get_health(governor: TopologyGovernor) -> TopologyHealthStatus:
    return governor.get_health()


def get_subsystem_health(governor: TopologyGovernor) -> Dict[str, TopologyHealthStatus]:
    return governor.get_subsystem_health()


def get_metrics(governor: TopologyGovernor) -> Dict[str, Any]:
    return governor.get_metrics()


def get_events(governor: TopologyGovernor) -> List[TopologyEvent]:
    return governor.get_events()


def get_decision_history(governor: TopologyGovernor) -> List[TopologyDecision]:
    return governor.get_decision_history()


# ===========================
# Convenience Exports
# ===========================

__all__ = [
    # Enumerations
    "TopologyTrustLevel",
    "TopologyHealthStatus",
    "TopologyControlDomain",
    "TopologyControlAction",

    # Policy & Signal Types
    "TopologyPolicy",
    "TopologySignals",
    "TopologyDecision",
    "TopologyEvent",

    # Governor
    "TopologyGovernor",

    # Module-level wrappers
    "get_health",
    "get_subsystem_health",
    "get_metrics",
    "get_events",
    "get_decision_history",

    # Factory Functions
    "create_topology_governor",
    "create_default_governor",
    "create_strict_governor",
    "create_permissive_governor",
    "validate_topology_signals",

    # Configurations
    "NODE_CONNECTIVITY",
    "TRUST_RELATIONSHIPS",
    "DEPLOYMENT_GRAPH",
    "STRUCTURAL_INTEGRITY",
    "ROUTE_RELIABILITY",

    # Probe Processing
    "TopologyReading",
    "TopologyProbeProcessor",

    # Controllers (lower-level API)
    "NodeConnectivityController",
    "TrustRelationshipController",
    "DeploymentGraphController",

    # Monitoring
    "TopologyMonitor",
]


# ===========================
# Demo Usage
# ===========================

if __name__ == "__main__":
    print("=== AILEE Topology Domain Demo ===\n")

    # Demo 1: Node Connectivity Rebalance
    print("1. Node Connectivity Rebalance")
    controller = NodeConnectivityController()

    readings = [
        TopologyReading(0.92, time.time(), "probe_cluster_a"),
        TopologyReading(0.89, time.time(), "probe_cluster_b"),
        TopologyReading(0.91, time.time(), "probe_cluster_c"),
        TopologyReading(0.88, time.time(), "probe_cluster_d"),
    ]

    result = controller.propose_rebalance(
        ai_score=0.90,
        ai_confidence=0.87,
        readings=readings,
        zone_id="region_west",
    )

    print(f"   Proposed score: 0.90 (confidence: 0.87)")
    print(f"   Trusted score:  {result.value:.3f}")
    print(f"   Status:         {result.safety_status}")
    print(f"   Fallback:       {result.used_fallback}")

    # Demo 2: Full Governor Evaluation
    print("\n2. TopologyGovernor — Trust Relationship Evaluation")
    governor = create_topology_governor()

    signals = TopologySignals(
        control_domain=TopologyControlDomain.TRUST_RELATIONSHIPS,
        proposed_action=TopologyControlAction.PROMOTE,
        ai_value=0.94,
        ai_confidence=0.91,
        topology_readings=[
            TopologyReading(0.93, time.time(), "validator_1"),
            TopologyReading(0.95, time.time(), "validator_2"),
            TopologyReading(0.92, time.time(), "validator_3"),
            TopologyReading(0.94, time.time(), "validator_4"),
            TopologyReading(0.96, time.time(), "validator_5"),
        ],
        domain_pair=("ailee.datacenter", "ailee.topology"),
        zone_id="global",
    )

    decision = governor.evaluate(signals)
    print(f"   Proposed integrity: 0.94 (confidence: 0.91)")
    print(f"   Trusted value:      {decision.trusted_value:.3f}")
    print(f"   Trust level:        {decision.authorized_level.name}")
    print(f"   Actionable:         {decision.actionable}")
    print(f"   Health:             {decision.health_status.value}")

    # Demo 3: Structural Integrity Check
    print("\n3. Structural Integrity Evaluation")
    signals_structural = TopologySignals(
        control_domain=TopologyControlDomain.STRUCTURAL_INTEGRITY,
        proposed_action=TopologyControlAction.REPAIR,
        ai_value=0.78,
        ai_confidence=0.82,
        topology_readings=[
            TopologyReading(0.76, time.time(), "struct_probe_1"),
            TopologyReading(0.80, time.time(), "struct_probe_2"),
            TopologyReading(0.77, time.time(), "struct_probe_3"),
            TopologyReading(0.79, time.time(), "struct_probe_4"),
        ],
        zone_id="core_mesh",
    )

    decision_structural = governor.evaluate(signals_structural)
    print(f"   Proposed integrity: 0.78 (confidence: 0.82)")
    print(f"   Trusted value:      {decision_structural.trusted_value:.3f}")
    print(f"   Trust level:        {decision_structural.authorized_level.name}")
    print(f"   Actionable:         {decision_structural.actionable}")

    print(f"\n   Governor metrics: {governor.get_metrics()}")
    print(f"   Subsystem health: {governor.get_subsystem_health()}")

    print("\n✓ Demo complete. Ready for production integration.")
