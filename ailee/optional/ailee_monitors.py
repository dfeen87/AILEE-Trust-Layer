"""
AILEE Trust Layer - Monitoring Utilities
Version: 1.0.0

Non-intrusive observability helpers for tracking AILEE pipeline health,
performance, and decision patterns over time.

Provides:
- Real-time metrics (fallback rate, confidence trends)
- Alerting on anomalies
- Performance tracking
- Decision pattern analysis
- Export to monitoring systems (Prometheus, etc.)
"""

from collections import deque, Counter
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import time
import statistics


@dataclass
class MetricSnapshot:
    """Point-in-time snapshot of AILEE metrics."""
    timestamp: float
    fallback_rate: float
    avg_confidence: float
    total_decisions: int
    safety_distribution: Dict[str, int]
    grace_distribution: Dict[str, int]
    consensus_distribution: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrustMonitor:
    """
    Core monitoring class for AILEE Trust Pipeline.
    Tracks decision outcomes, confidence scores, and system health.
    
    Example:
        >>> monitor = TrustMonitor(window=100)
        >>> result = pipeline.process(...)
        >>> monitor.record(result)
        >>> print(f"Fallback rate: {monitor.fallback_rate():.2%}")
    """
    
    def __init__(self, window: int = 100):
        """
        Initialize monitor with rolling window.
        
        Args:
            window: Size of rolling window for metrics
        """
        self.window = window
        self.confidence_scores = deque(maxlen=window)
        self.fallback_flags = deque(maxlen=window)
        self.safety_statuses = deque(maxlen=window)
        self.grace_statuses = deque(maxlen=window)
        self.consensus_statuses = deque(maxlen=window)
        self.timestamps = deque(maxlen=window)
        self.values = deque(maxlen=window)
        
        # Cumulative counters
        self.total_decisions = 0
        self.total_fallbacks = 0
        self.total_grace_passes = 0
        self.total_grace_fails = 0
        self.total_consensus_passes = 0
        self.total_consensus_fails = 0
        
        # Performance tracking
        self.decision_times = deque(maxlen=window)
        
        # Snapshot history
        self.snapshot_history: List[MetricSnapshot] = []
    
    def record(self, result, decision_time: Optional[float] = None) -> None:
        """
        Record a decision result.
        
        Args:
            result: DecisionResult from pipeline
            decision_time: Optional processing time in seconds
            
        Example:
            >>> start = time.time()
            >>> result = pipeline.process(...)
            >>> monitor.record(result, time.time() - start)
        """
        self.confidence_scores.append(result.confidence_score)
        self.fallback_flags.append(result.used_fallback)
        self.safety_statuses.append(result.safety_status)
        self.grace_statuses.append(result.grace_status)
        self.consensus_statuses.append(result.consensus_status)
        self.values.append(result.value)
        
        # Extract timestamp from metadata if available
        timestamp = result.metadata.get('timestamp', time.time()) if hasattr(result, 'metadata') else time.time()
        self.timestamps.append(timestamp)
        
        if decision_time is not None:
            self.decision_times.append(decision_time)
        
        # Update cumulative counters
        self.total_decisions += 1
        if result.used_fallback:
            self.total_fallbacks += 1
        
        # Extract status strings for comparison
        grace_status = str(result.grace_status)
        consensus_status = str(result.consensus_status)
        
        if grace_status in ("PASS", "GraceStatus.PASS"):
            self.total_grace_passes += 1
        elif grace_status in ("FAIL", "GraceStatus.FAIL"):
            self.total_grace_fails += 1
        
        if consensus_status in ("PASS", "ConsensusStatus.PASS"):
            self.total_consensus_passes += 1
        elif consensus_status in ("FAIL", "ConsensusStatus.FAIL"):
            self.total_consensus_fails += 1
    
    def fallback_rate(self) -> float:
        """
        Calculate fallback rate over the rolling window.
        
        Returns:
            Fraction of decisions that used fallback (0.0-1.0)
        """
        if not self.fallback_flags:
            return 0.0
        return sum(self.fallback_flags) / len(self.fallback_flags)
    
    def avg_confidence(self) -> float:
        """
        Calculate average confidence score over the rolling window.
        
        Returns:
            Mean confidence score
        """
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    def min_confidence(self) -> float:
        """Get minimum confidence in window."""
        return min(self.confidence_scores) if self.confidence_scores else 0.0
    
    def max_confidence(self) -> float:
        """Get maximum confidence in window."""
        return max(self.confidence_scores) if self.confidence_scores else 0.0
    
    def confidence_std(self) -> float:
        """Calculate standard deviation of confidence scores."""
        if len(self.confidence_scores) < 2:
            return 0.0
        return statistics.stdev(self.confidence_scores)
    
    def grace_success_rate(self) -> float:
        """
        Calculate Grace Layer success rate (PASS / (PASS + FAIL)).
        
        Returns:
            Success rate for Grace evaluations
        """
        grace_attempts = self.total_grace_passes + self.total_grace_fails
        if grace_attempts == 0:
            return 0.0
        return self.total_grace_passes / grace_attempts
    
    def consensus_success_rate(self) -> float:
        """
        Calculate Consensus Layer success rate (PASS / (PASS + FAIL)).
        
        Returns:
            Success rate for Consensus evaluations
        """
        consensus_attempts = self.total_consensus_passes + self.total_consensus_fails
        if consensus_attempts == 0:
            return 0.0
        return self.total_consensus_passes / consensus_attempts
    
    def avg_decision_time(self) -> float:
        """
        Calculate average decision processing time.
        
        Returns:
            Mean processing time in seconds
        """
        if not self.decision_times:
            return 0.0
        return statistics.fmean(self.decision_times)
    
    def value_stability(self) -> float:
        """
        Calculate stability of output values (inverse of coefficient of variation).
        
        Returns:
            Stability metric (higher = more stable)
        """
        if len(self.values) < 2:
            return 1.0
        
        mean = statistics.fmean(self.values)
        if mean == 0:
            return 0.0
        
        std = statistics.stdev(self.values)
        cv = std / abs(mean)  # Coefficient of variation
        
        # Return inverse (0 = unstable, 1 = perfectly stable)
        return 1.0 / (1.0 + cv)
    
    def get_status_distribution(self, status_type: str = "safety") -> Dict[str, int]:
        """
        Get distribution of status outcomes.
        
        Args:
            status_type: One of "safety", "grace", "consensus"
            
        Returns:
            Dictionary mapping status values to counts
        """
        if status_type == "safety":
            statuses = self.safety_statuses
        elif status_type == "grace":
            statuses = self.grace_statuses
        elif status_type == "consensus":
            statuses = self.consensus_statuses
        else:
            raise ValueError(f"Unknown status_type: {status_type}")
        
        # Convert enum values to strings for consistent counting
        status_strings = [str(s) for s in statuses]
        return dict(Counter(status_strings))
    
    def create_snapshot(self) -> MetricSnapshot:
        """
        Create a point-in-time snapshot of current metrics.
        
        Returns:
            MetricSnapshot with current state
        """
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            fallback_rate=self.fallback_rate(),
            avg_confidence=self.avg_confidence(),
            total_decisions=self.total_decisions,
            safety_distribution=self.get_status_distribution("safety"),
            grace_distribution=self.get_status_distribution("grace"),
            consensus_distribution=self.get_status_distribution("consensus"),
            metadata={
                'grace_success_rate': self.grace_success_rate(),
                'consensus_success_rate': self.consensus_success_rate(),
                'value_stability': self.value_stability(),
                'confidence_std': self.confidence_std(),
            }
        )
        
        self.snapshot_history.append(snapshot)
        return snapshot
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all metrics.
        
        Returns:
            Dictionary with all available metrics
        """
        return {
            'window_size': len(self.confidence_scores),
            'total_decisions': self.total_decisions,
            'fallback_rate': self.fallback_rate(),
            'total_fallbacks': self.total_fallbacks,
            'confidence': {
                'mean': self.avg_confidence(),
                'min': self.min_confidence(),
                'max': self.max_confidence(),
                'std': self.confidence_std(),
            },
            'grace': {
                'success_rate': self.grace_success_rate(),
                'total_passes': self.total_grace_passes,
                'total_fails': self.total_grace_fails,
                'distribution': self.get_status_distribution('grace'),
            },
            'consensus': {
                'success_rate': self.consensus_success_rate(),
                'total_passes': self.total_consensus_passes,
                'total_fails': self.total_consensus_fails,
                'distribution': self.get_status_distribution('consensus'),
            },
            'safety_distribution': self.get_status_distribution('safety'),
            'value_stability': self.value_stability(),
            'avg_decision_time': self.avg_decision_time(),
        }
    
    def reset(self) -> None:
        """Reset all metrics and counters."""
        self.confidence_scores.clear()
        self.fallback_flags.clear()
        self.safety_statuses.clear()
        self.grace_statuses.clear()
        self.consensus_statuses.clear()
        self.timestamps.clear()
        self.values.clear()
        self.decision_times.clear()
        
        self.total_decisions = 0
        self.total_fallbacks = 0
        self.total_grace_passes = 0
        self.total_grace_fails = 0
        self.total_consensus_passes = 0
        self.total_consensus_fails = 0
    
    def __repr__(self) -> str:
        return (
            f"TrustMonitor(decisions={self.total_decisions}, "
            f"fallback_rate={self.fallback_rate():.2%}, "
            f"avg_confidence={self.avg_confidence():.3f})"
        )


class AlertingMonitor(TrustMonitor):
    """
    Extended monitor with configurable alerting capabilities.
    Triggers callbacks when thresholds are breached.
    
    Example:
        >>> def alert_handler(alert_type, value, threshold):
        ...     logger.warning(f"ALERT: {alert_type} = {value} (threshold: {threshold})")
        >>> 
        >>> monitor = AlertingMonitor(
        ...     window=100,
        ...     fallback_rate_threshold=0.30,
        ...     alert_callback=alert_handler
        ... )
    """
    
    def __init__(
        self,
        window: int = 100,
        fallback_rate_threshold: Optional[float] = None,
        min_confidence_threshold: Optional[float] = None,
        max_decision_time_threshold: Optional[float] = None,
        alert_callback: Optional[Callable[[str, float, float], None]] = None,
        alert_cooldown: float = 60.0,
    ):
        """
        Initialize alerting monitor.
        
        Args:
            window: Size of rolling window
            fallback_rate_threshold: Alert if fallback rate exceeds this
            min_confidence_threshold: Alert if avg confidence falls below this
            max_decision_time_threshold: Alert if decision time exceeds this
            alert_callback: Function to call on alerts (type, value, threshold)
            alert_cooldown: Minimum seconds between repeated alerts
        """
        super().__init__(window)
        
        self.fallback_rate_threshold = fallback_rate_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.max_decision_time_threshold = max_decision_time_threshold
        self.alert_callback = alert_callback
        self.alert_cooldown = alert_cooldown
        
        # Track last alert times to prevent spam
        self.last_alert_times: Dict[str, float] = {}
    
    def record(self, result, decision_time: Optional[float] = None) -> None:
        """Record and check for alert conditions."""
        super().record(result, decision_time)
        self._check_alerts()
    
    def _check_alerts(self) -> None:
        """Check all alert conditions."""
        now = time.time()
        
        # Check fallback rate
        if self.fallback_rate_threshold is not None:
            rate = self.fallback_rate()
            if rate > self.fallback_rate_threshold:
                self._trigger_alert("high_fallback_rate", rate, self.fallback_rate_threshold, now)
        
        # Check confidence
        if self.min_confidence_threshold is not None:
            conf = self.avg_confidence()
            if conf < self.min_confidence_threshold:
                self._trigger_alert("low_confidence", conf, self.min_confidence_threshold, now)
        
        # Check decision time
        if self.max_decision_time_threshold is not None and self.decision_times:
            dt = self.avg_decision_time()
            if dt > self.max_decision_time_threshold:
                self._trigger_alert("slow_decisions", dt, self.max_decision_time_threshold, now)
    
    def _trigger_alert(self, alert_type: str, value: float, threshold: float, now: float) -> None:
        """Trigger an alert if cooldown has expired."""
        last_alert = self.last_alert_times.get(alert_type, 0.0)
        
        if now - last_alert >= self.alert_cooldown:
            if self.alert_callback:
                self.alert_callback(alert_type, value, threshold)
            self.last_alert_times[alert_type] = now


class PrometheusExporter:
    """
    Export AILEE metrics in Prometheus format.
    
    Example:
        >>> monitor = TrustMonitor()
        >>> exporter = PrometheusExporter(monitor, namespace="ailee")
        >>> metrics_text = exporter.export()
        >>> # Serve via HTTP endpoint
    """
    
    def __init__(self, monitor: TrustMonitor, namespace: str = "ailee"):
        """
        Initialize Prometheus exporter.
        
        Args:
            monitor: TrustMonitor instance to export from
            namespace: Metric namespace prefix
        """
        self.monitor = monitor
        self.namespace = namespace
    
    def export(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        
        # Fallback rate
        lines.append(f"# HELP {self.namespace}_fallback_rate Fraction of decisions using fallback")
        lines.append(f"# TYPE {self.namespace}_fallback_rate gauge")
        lines.append(f"{self.namespace}_fallback_rate {self.monitor.fallback_rate():.6f}")
        
        # Confidence metrics
        lines.append(f"# HELP {self.namespace}_confidence_avg Average confidence score")
        lines.append(f"# TYPE {self.namespace}_confidence_avg gauge")
        lines.append(f"{self.namespace}_confidence_avg {self.monitor.avg_confidence():.6f}")
        
        # Total decisions
        lines.append(f"# HELP {self.namespace}_decisions_total Total number of decisions")
        lines.append(f"# TYPE {self.namespace}_decisions_total counter")
        lines.append(f"{self.namespace}_decisions_total {self.monitor.total_decisions}")
        
        # Grace success rate
        lines.append(f"# HELP {self.namespace}_grace_success_rate Grace layer success rate")
        lines.append(f"# TYPE {self.namespace}_grace_success_rate gauge")
        lines.append(f"{self.namespace}_grace_success_rate {self.monitor.grace_success_rate():.6f}")
        
        # Consensus success rate
        lines.append(f"# HELP {self.namespace}_consensus_success_rate Consensus layer success rate")
        lines.append(f"# TYPE {self.namespace}_consensus_success_rate gauge")
        lines.append(f"{self.namespace}_consensus_success_rate {self.monitor.consensus_success_rate():.6f}")
        
        return "\n".join(lines) + "\n"


# Convenience exports
__all__ = [
    'TrustMonitor',
    'AlertingMonitor',
    'PrometheusExporter',
    'MetricSnapshot',
]


# Demo usage
if __name__ == "__main__":
    from dataclasses import dataclass
    from enum import Enum
    
    # Mock classes for demo
    class SafetyStatus(str, Enum):
        ACCEPTED = "ACCEPTED"
        BORDERLINE = "BORDERLINE"
    
    class GraceStatus(str, Enum):
        PASS = "PASS"
        FAIL = "FAIL"
        SKIPPED = "SKIPPED"
    
    class ConsensusStatus(str, Enum):
        PASS = "PASS"
        FAIL = "FAIL"
    
    @dataclass
    class MockResult:
        value: float
        safety_status: SafetyStatus
        grace_status: GraceStatus
        consensus_status: ConsensusStatus
        used_fallback: bool
        confidence_score: float
        metadata: dict
    
    print("=== AILEE Trust Monitor Demo ===\n")
    
    # Create monitor
    monitor = TrustMonitor(window=10)
    
    # Simulate some decisions
    mock_results = [
        MockResult(10.5, SafetyStatus.ACCEPTED, GraceStatus.SKIPPED, ConsensusStatus.PASS, False, 0.95, {}),
        MockResult(10.3, SafetyStatus.ACCEPTED, GraceStatus.SKIPPED, ConsensusStatus.PASS, False, 0.92),
        MockResult(15.0, SafetyStatus.BORDERLINE, GraceStatus.PASS, ConsensusStatus.PASS, False, 0.75, {}),
        MockResult(10.2, SafetyStatus.ACCEPTED, GraceStatus.SKIPPED, ConsensusStatus.PASS, False, 0.91, {}),
        MockResult(10.1, SafetyStatus.BORDERLINE, GraceStatus.FAIL, ConsensusStatus.PASS, True, 0.68, {}),
    ]
    
    for result in mock_results:
        monitor.record(result, decision_time=0.05)
    
    print(monitor)
    print(f"\nFallback rate: {monitor.fallback_rate():.2%}")
    print(f"Avg confidence: {monitor.avg_confidence():.3f}")
    print(f"Grace success rate: {monitor.grace_success_rate():.2%}")
    print(f"Value stability: {monitor.value_stability():.3f}")
    
    print("\n=== Full Summary ===")
    summary = monitor.get_summary()
    print(f"Total decisions: {summary['total_decisions']}")
    print(f"Confidence range: [{summary['confidence']['min']:.3f}, {summary['confidence']['max']:.3f}]")
    print(f"Safety distribution: {summary['safety_distribution']}")
    
    print("\n=== Prometheus Export ===")
    exporter = PrometheusExporter(monitor)
    print(exporter.export())
