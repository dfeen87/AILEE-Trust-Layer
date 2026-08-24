# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

import pytest
from ailee.domains.video_temporal_provenance import (
    VideoTemporalConfig,
    VideoTemporalPolicy,
    VideoTemporalGovernor,
    FrameSignal,
)

def test_video_temporal_governor_clean_sequence():
    gov = VideoTemporalGovernor()
    frames = [
        FrameSignal(
            frame_index=i,
            timestamp_sec=i * 0.033,
            raw_trust=95.0,
            hash_delta=0.02,
            dx=1.0,
            dy=0.5,
            flow_consistency=0.95
        )
        for i in range(10)
    ]

    decision = gov.evaluate_sequence(frames)
    assert decision.overall_trust_score >= 80.0
    assert decision.safety_status in ["ACCEPTED", "PARTIALLY_TRUSTED"]
    assert decision.anomaly_count == 0

def test_video_temporal_governor_synthetic_anomaly():
    gov = VideoTemporalGovernor()
    frames = []
    for i in range(10):
        if i == 5:
            # Synthetic anomaly
            frames.append(FrameSignal(
                frame_index=i,
                timestamp_sec=i * 0.033,
                raw_trust=40.0,
                hash_delta=0.85,
                dx=5.0,
                dy=-3.0,
                flow_consistency=0.10
            ))
        else:
            frames.append(FrameSignal(
                frame_index=i,
                timestamp_sec=i * 0.033,
                raw_trust=95.0,
                hash_delta=0.02,
                dx=1.0,
                dy=0.5,
                flow_consistency=0.95
            ))

    decision = gov.evaluate_sequence(frames)
    assert decision.anomaly_count > 0
    assert decision.safety_status in ["PARTIALLY_TRUSTED", "OUTRIGHT_REJECTED"]
