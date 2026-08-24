# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""
Python FFI wrapper for AILEE-Video C++ Temporal Provenance Engine C ABI.
"""

import ctypes
import os
import platform
from dataclasses import dataclass
from typing import List, Dict, Any

class TemporalIntegrityMetricsCTypes(ctypes.Structure):
    _align_ = 64
    _fields_ = [
        ("overall_trust_score", ctypes.c_float),
        ("mean_frame_trust", ctypes.c_float),
        ("transition_integrity_avg", ctypes.c_float),
        ("scene_boundary_trust_avg", ctypes.c_float),
        ("temporal_continuity_score", ctypes.c_float),
        ("optical_flow_stability", ctypes.c_float),
        ("rhythm_stability", ctypes.c_float),
        ("total_frames", ctypes.c_uint32),
        ("total_transitions", ctypes.c_uint32),
        ("total_scene_boundaries", ctypes.c_uint32),
        ("anomaly_count", ctypes.c_uint32),
        ("safety_status", ctypes.c_uint8),
    ]

@dataclass
class TemporalIntegrityMetricsPy:
    overall_trust_score: float
    mean_frame_trust: float
    transition_integrity_avg: float
    scene_boundary_trust_avg: float
    temporal_continuity_score: float
    optical_flow_stability: float
    rhythm_stability: float
    total_frames: int
    total_transitions: int
    total_scene_boundaries: int
    anomaly_count: int
    safety_status: str

class TPEFFIWrapper:
    def __init__(self):
        self._lib = self._load_lib()
        self._handle = None
        self._fallback_frames: List[Dict[str, Any]] = []
        if self._lib:
            self._setup_prototypes()
            self._handle = self._lib.ailee_tpe_create()

    def _load_lib(self):
        system = platform.system()
        lib_name = "libailee_video_temporal_provenance.so"
        if system == "Darwin":
            lib_name = "libailee_video_temporal_provenance.dylib"
        elif system == "Windows":
            lib_name = "ailee_video_temporal_provenance.dll"

        search_paths = [
            os.path.join(os.path.dirname(__file__), "../../../build", lib_name),
            os.path.join(os.path.dirname(__file__), "../../../", lib_name),
            lib_name,
        ]

        for p in search_paths:
            if os.path.exists(p):
                try:
                    return ctypes.CDLL(p)
                except Exception:
                    pass
        return None

    def _setup_prototypes(self):
        self._lib.ailee_tpe_create.restype = ctypes.c_void_p
        self._lib.ailee_tpe_create.argtypes = []

        self._lib.ailee_tpe_destroy.restype = None
        self._lib.ailee_tpe_destroy.argtypes = [ctypes.c_void_p]

        self._lib.ailee_tpe_reset.restype = None
        self._lib.ailee_tpe_reset.argtypes = [ctypes.c_void_p]

        self._lib.ailee_tpe_ingest_frame.restype = ctypes.c_int
        self._lib.ailee_tpe_ingest_frame.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_double,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]

        self._lib.ailee_tpe_evaluate.restype = ctypes.c_int
        self._lib.ailee_tpe_evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(TemporalIntegrityMetricsCTypes)]

        self._lib.ailee_tpe_embed_watermark.restype = ctypes.c_int
        self._lib.ailee_tpe_embed_watermark.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]

        self._lib.ailee_tpe_verify_watermark.restype = ctypes.c_int
        self._lib.ailee_tpe_verify_watermark.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]

    def reset(self):
        self._fallback_frames.clear()
        if self._lib and self._handle:
            self._lib.ailee_tpe_reset(self._handle)

    def ingest_frame(self, frame_idx: int, timestamp: float, raw_trust: float, hash_delta: float, dx: float, dy: float, flow_consistency: float) -> bool:
        self._fallback_frames.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "raw_trust": raw_trust,
            "hash_delta": hash_delta,
            "dx": dx,
            "dy": dy,
            "flow_consistency": flow_consistency,
        })
        if self._lib and self._handle:
            res = self._lib.ailee_tpe_ingest_frame(
                self._handle, frame_idx, timestamp, raw_trust, hash_delta, dx, dy, flow_consistency
            )
            return res == 1
        return True

    def evaluate(self) -> TemporalIntegrityMetricsPy:
        if self._lib and self._handle:
            metrics_c = TemporalIntegrityMetricsCTypes()
            res = self._lib.ailee_tpe_evaluate(self._handle, ctypes.byref(metrics_c))
            if res == 1:
                status_str = "ACCEPTED" if metrics_c.safety_status == 0 else ("PARTIALLY_TRUSTED" if metrics_c.safety_status == 1 else "OUTRIGHT_REJECTED")
                return TemporalIntegrityMetricsPy(
                    overall_trust_score=metrics_c.overall_trust_score,
                    mean_frame_trust=metrics_c.mean_frame_trust,
                    transition_integrity_avg=metrics_c.transition_integrity_avg,
                    scene_boundary_trust_avg=metrics_c.scene_boundary_trust_avg,
                    temporal_continuity_score=metrics_c.temporal_continuity_score,
                    optical_flow_stability=metrics_c.optical_flow_stability,
                    rhythm_stability=metrics_c.rhythm_stability,
                    total_frames=metrics_c.total_frames,
                    total_transitions=metrics_c.total_transitions,
                    total_scene_boundaries=metrics_c.total_scene_boundaries,
                    anomaly_count=metrics_c.anomaly_count,
                    safety_status=status_str
                )

        # Pure-Python fallback evaluation
        if not self._fallback_frames:
            return TemporalIntegrityMetricsPy(
                overall_trust_score=0.0,
                mean_frame_trust=0.0,
                transition_integrity_avg=0.0,
                scene_boundary_trust_avg=0.0,
                temporal_continuity_score=0.0,
                optical_flow_stability=0.0,
                rhythm_stability=0.0,
                total_frames=0,
                total_transitions=0,
                total_scene_boundaries=0,
                anomaly_count=0,
                safety_status="OUTRIGHT_REJECTED"
            )

        total_frames = len(self._fallback_frames)
        mean_frame_trust = sum(f["raw_trust"] for f in self._fallback_frames) / total_frames

        anomaly_count = 0
        scene_boundaries = 0
        for f in self._fallback_frames:
            if f["flow_consistency"] < 0.35 or f["raw_trust"] < 50.0 or abs(f["hash_delta"]) > 0.75:
                anomaly_count += 1
            if abs(f["hash_delta"]) > 0.35:
                scene_boundaries += 1

        transition_avg = max(0.0, 100.0 - (anomaly_count * 15.0))
        continuity_score = max(0.0, 100.0 - (anomaly_count * 10.0))
        sb_trust = 90.0 if scene_boundaries > 0 else 100.0

        overall_trust = max(0.0, min(100.0, (mean_frame_trust * 0.35) + (transition_avg * 0.30) + (continuity_score * 0.25) + (sb_trust * 0.10) - (anomaly_count * 10.0)))
        status = "ACCEPTED" if overall_trust >= 85.0 and anomaly_count == 0 else ("PARTIALLY_TRUSTED" if overall_trust >= 50.0 else "OUTRIGHT_REJECTED")

        return TemporalIntegrityMetricsPy(
            overall_trust_score=overall_trust,
            mean_frame_trust=mean_frame_trust,
            transition_integrity_avg=transition_avg,
            scene_boundary_trust_avg=sb_trust,
            temporal_continuity_score=continuity_score,
            optical_flow_stability=85.0,
            rhythm_stability=95.0,
            total_frames=total_frames,
            total_transitions=max(0, total_frames - 1),
            total_scene_boundaries=scene_boundaries,
            anomaly_count=anomaly_count,
            safety_status=status
        )

    def close(self):
        if hasattr(self, '_lib') and hasattr(self, '_handle') and self._lib and self._handle:
            try:
                self._lib.ailee_tpe_destroy(self._handle)
            except Exception:
                pass
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
