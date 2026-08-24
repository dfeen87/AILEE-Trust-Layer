# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""
Governor for Video Temporal Provenance domain.
"""

from dataclasses import dataclass
from typing import List, Optional
from .config import VideoTemporalConfig
from .policy import VideoTemporalPolicy
from .ffi import TPEFFIWrapper, TemporalIntegrityMetricsPy

@dataclass
class FrameSignal:
    frame_index: int
    timestamp_sec: float
    raw_trust: float
    hash_delta: float
    dx: float
    dy: float
    flow_consistency: float

@dataclass
class VideoGovernorDecision:
    overall_trust_score: float
    safety_status: str
    anomaly_count: int
    reason: str
    metrics: TemporalIntegrityMetricsPy

class VideoTemporalGovernor:
    def __init__(self, config: Optional[VideoTemporalConfig] = None, policy: Optional[VideoTemporalPolicy] = None):
        self.config = config or VideoTemporalConfig()
        self.policy = policy or VideoTemporalPolicy()
        self.wrapper = TPEFFIWrapper()

    def reset(self):
        self.wrapper.reset()

    def evaluate_sequence(self, frames: List[FrameSignal]) -> VideoGovernorDecision:
        self.wrapper.reset()
        for f in frames:
            self.wrapper.ingest_frame(
                f.frame_index,
                f.timestamp_sec,
                f.raw_trust,
                f.hash_delta,
                f.dx,
                f.dy,
                f.flow_consistency
            )

        metrics = self.wrapper.evaluate()

        status = metrics.safety_status
        reason = "Sequence evaluated successfully."

        if metrics.overall_trust_score < self.config.min_trust_threshold:
            status = "OUTRIGHT_REJECTED"
            reason = f"Overall trust {metrics.overall_trust_score:.1f} below minimum threshold {self.config.min_trust_threshold:.1f}."
        elif metrics.anomaly_count > self.config.max_allowed_anomalies:
            status = "PARTIALLY_TRUSTED" if status != "OUTRIGHT_REJECTED" else status
            reason = f"Anomaly count {metrics.anomaly_count} exceeded max allowed {self.config.max_allowed_anomalies}."

        return VideoGovernorDecision(
            overall_trust_score=metrics.overall_trust_score,
            safety_status=status,
            anomaly_count=metrics.anomaly_count,
            reason=reason,
            metrics=metrics
        )
