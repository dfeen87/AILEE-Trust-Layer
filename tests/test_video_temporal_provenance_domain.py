# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

import pytest
import ctypes
from ailee.domains.video_temporal_provenance import (
    VideoTemporalConfig,
    VideoTemporalPolicy,
    VideoTemporalGovernor,
    FrameSignal,
)
from ailee.domains.video_temporal_provenance.ffi import TemporalIntegrityMetricsCTypes, TPEFFIWrapper


def test_ctypes_metrics_layout_and_native_alignment():
    assert ctypes.sizeof(TemporalIntegrityMetricsCTypes) == 64
    assert TemporalIntegrityMetricsCTypes.total_frames.offset == 28
    assert TemporalIntegrityMetricsCTypes.safety_status.offset == 44
    metrics, storage = TPEFFIWrapper._new_aligned_metrics()
    assert storage is not None
    assert ctypes.addressof(metrics) % 64 == 0


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


@pytest.mark.parametrize("field,value", [
    ("raw_trust", float("nan")),
    ("raw_trust", float("inf")),
    ("flow_consistency", -0.1),
    ("timestamp_sec", -1.0),
])
def test_invalid_frames_fail_closed(field, value):
    gov = VideoTemporalGovernor()
    values = dict(frame_index=0, timestamp_sec=0.0, raw_trust=95.0, hash_delta=0.02,
                  dx=1.0, dy=0.5, flow_consistency=0.95)
    values[field] = value
    decision = gov.evaluate_sequence([FrameSignal(**values)])
    assert decision.safety_status == "OUTRIGHT_REJECTED"
    assert decision.overall_trust_score == 0.0
    assert "invalid" in decision.reason


def test_out_of_order_frames_fail_closed():
    gov = VideoTemporalGovernor()
    frames = [
        FrameSignal(1, 1.0, 95.0, 0.02, 1.0, 0.5, 0.95),
        FrameSignal(1, 2.0, 95.0, 0.02, 1.0, 0.5, 0.95),
    ]
    assert gov.evaluate_sequence(frames).safety_status == "OUTRIGHT_REJECTED"


def test_native_metrics_ctypes_matches_64_byte_abi():
    from ailee.domains.video_temporal_provenance.ffi import TemporalIntegrityMetricsCTypes
    assert ctypes.sizeof(TemporalIntegrityMetricsCTypes) == 64
