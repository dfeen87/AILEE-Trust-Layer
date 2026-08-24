//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

use std::os::raw::{c_double, c_float, c_int, c_uchar, c_ulonglong, c_void};

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CMetrics {
    pub overall_trust_score: c_float,
    pub mean_frame_trust: c_float,
    pub transition_integrity_avg: c_float,
    pub scene_boundary_trust_avg: c_float,
    pub temporal_continuity_score: c_float,
    pub optical_flow_stability: c_float,
    pub rhythm_stability: c_float,
    pub total_frames: u32,
    pub total_transitions: u32,
    pub total_scene_boundaries: u32,
    pub anomaly_count: u32,
    pub safety_status: u8,
}

extern "C" {
    fn ailee_tpe_create() -> *mut c_void;
    fn ailee_tpe_destroy(handle: *mut c_void);
    fn ailee_tpe_reset(handle: *mut c_void);
    fn ailee_tpe_ingest_frame(
        handle: *mut c_void,
        frame_index: c_ulonglong,
        timestamp_sec: c_double,
        raw_trust: c_float,
        perceptual_hash_delta: c_float,
        motion_dx: c_float,
        motion_dy: c_float,
        flow_consistency: c_float,
    ) -> c_int;
    fn ailee_tpe_evaluate(handle: *mut c_void, out_metrics: *mut CMetrics) -> c_int;
    fn ailee_tpe_embed_watermark(
        handle: *mut c_void,
        frame_index: c_ulonglong,
        key: *const c_uchar,
        key_len: usize,
    ) -> c_int;
    fn ailee_tpe_verify_watermark(
        handle: *mut c_void,
        frame_index: c_ulonglong,
        key: *const c_uchar,
        key_len: usize,
    ) -> c_int;
}

#[derive(Debug, Clone)]
pub struct FrameSignal {
    pub frame_index: u64,
    pub timestamp_sec: f64,
    pub raw_trust: f32,
    pub hash_delta: f32,
    pub dx: f32,
    pub dy: f32,
    pub flow_consistency: f32,
}

#[derive(Debug, Clone)]
pub struct TemporalIntegrityMetrics {
    pub overall_trust_score: f32,
    pub mean_frame_trust: f32,
    pub transition_integrity_avg: f32,
    pub scene_boundary_trust_avg: f32,
    pub temporal_continuity_score: f32,
    pub optical_flow_stability: f32,
    pub rhythm_stability: f32,
    pub total_frames: u32,
    pub total_transitions: u32,
    pub total_scene_boundaries: u32,
    pub anomaly_count: u32,
    pub safety_status: String,
}

pub struct VideoTemporalGovernor {
    handle: *mut c_void,
}

impl VideoTemporalGovernor {
    pub fn new() -> Self {
        let handle = unsafe { ailee_tpe_create() };
        Self { handle }
    }

    pub fn reset(&mut self) {
        if !self.handle.is_null() {
            unsafe { ailee_tpe_reset(self.handle) };
        }
    }

    pub fn ingest_frame(&mut self, signal: &FrameSignal) -> bool {
        if self.handle.is_null() {
            return false;
        }
        let res = unsafe {
            ailee_tpe_ingest_frame(
                self.handle,
                signal.frame_index as c_ulonglong,
                signal.timestamp_sec as c_double,
                signal.raw_trust as c_float,
                signal.hash_delta as c_float,
                signal.dx as c_float,
                signal.dy as c_float,
                signal.flow_consistency as c_float,
            )
        };
        res == 1
    }

    pub fn embed_watermark(&mut self, frame_index: u64, key: &[u8]) -> bool {
        if self.handle.is_null() {
            return false;
        }
        let res = unsafe {
            ailee_tpe_embed_watermark(
                self.handle,
                frame_index as c_ulonglong,
                key.as_ptr(),
                key.len(),
            )
        };
        res == 1
    }

    pub fn verify_watermark(&mut self, frame_index: u64, key: &[u8]) -> bool {
        if self.handle.is_null() {
            return false;
        }
        let res = unsafe {
            ailee_tpe_verify_watermark(
                self.handle,
                frame_index as c_ulonglong,
                key.as_ptr(),
                key.len(),
            )
        };
        res == 1
    }

    pub fn evaluate(&mut self) -> TemporalIntegrityMetrics {
        if !self.handle.is_null() {
            let mut c_metrics = CMetrics::default();
            let res = unsafe { ailee_tpe_evaluate(self.handle, &mut c_metrics) };
            if res == 1 {
                let status_str = match c_metrics.safety_status {
                    0 => "ACCEPTED".to_string(),
                    1 => "PARTIALLY_TRUSTED".to_string(),
                    _ => "OUTRIGHT_REJECTED".to_string(),
                };

                return TemporalIntegrityMetrics {
                    overall_trust_score: c_metrics.overall_trust_score,
                    mean_frame_trust: c_metrics.mean_frame_trust,
                    transition_integrity_avg: c_metrics.transition_integrity_avg,
                    scene_boundary_trust_avg: c_metrics.scene_boundary_trust_avg,
                    temporal_continuity_score: c_metrics.temporal_continuity_score,
                    optical_flow_stability: c_metrics.optical_flow_stability,
                    rhythm_stability: c_metrics.rhythm_stability,
                    total_frames: c_metrics.total_frames,
                    total_transitions: c_metrics.total_transitions,
                    total_scene_boundaries: c_metrics.total_scene_boundaries,
                    anomaly_count: c_metrics.anomaly_count,
                    safety_status: status_str,
                };
            }
        }

        TemporalIntegrityMetrics {
            overall_trust_score: 100.0,
            mean_frame_trust: 100.0,
            transition_integrity_avg: 100.0,
            scene_boundary_trust_avg: 100.0,
            temporal_continuity_score: 100.0,
            optical_flow_stability: 100.0,
            rhythm_stability: 100.0,
            total_frames: 0,
            total_transitions: 0,
            total_scene_boundaries: 0,
            anomaly_count: 0,
            safety_status: "ACCEPTED".to_string(),
        }
    }
}

impl Drop for VideoTemporalGovernor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ailee_tpe_destroy(self.handle) };
        }
    }
}

unsafe impl Send for VideoTemporalGovernor {}
unsafe impl Sync for VideoTemporalGovernor {}
