// Copyright (c) Don Michael Feeney Jr.
// Licensed under the MIT License.

#include "ailee_video_temporal_provenance.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace ailee::video;

    std::cout << "===========================================================" << std::endl;
    std::cout << "  AILEE-Video Temporal Provenance Engine Demo (v7.0.0)      " << std::endl;
    std::cout << "===========================================================" << std::endl << std::endl;

    TemporalProvenanceEngine engine;
    uint8_t secret_key[] = "ailee_demo_key_7";

    std::cout << "[Step 1] Ingesting Synthetic Video Sequence (12 frames)..." << std::endl;

    // Simulate 12 video frames with normal motion, a synthetic frame interpolation, and a scene cut
    for (int i = 0; i < 12; ++i) {
        FrameData f{};
        f.frame_index = i;
        f.timestamp_sec = i * 0.0333; // 30 fps
        f.trust_score = 92.0f;
        f.motion.dx = 1.2f;
        f.motion.dy = 0.5f;
        f.motion.flow_consistency = 0.94f;
        f.perceptual_hash_delta = 0.04f;

        // Introduce synthetic interpolation anomaly at frame 4
        if (i == 4) {
            f.provenance_flags |= PROV_FLAG_SYNTHETIC_INTERP;
            f.motion.flow_consistency = 0.25f;
            f.trust_score = 65.0f;
        }

        // Introduce a hard cut at frame 8
        if (i == 8) {
            f.perceptual_hash_delta = 0.82f;
            f.motion.dx = 0.1f;
            f.motion.dy = 3.0f;
        }

        engine.ingest_frame(f);
        engine.embed_watermark_to_frame(i, secret_key, sizeof(secret_key));
    }

    std::cout << "  ✓ Ingested 12 frames into Temporal Provenance Chain (TPC)." << std::endl << std::endl;

    std::cout << "[Step 2] Evaluating Temporal Integrity & Governance..." << std::endl;
    TemporalIntegrityMetrics metrics{};
    engine.evaluate_chain(metrics);

    std::cout << "===========================================================" << std::endl;
    std::cout << "                 TEMPORAL PROVENANCE REPORT                " << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " Total Frames Ingested       : " << metrics.total_frames << std::endl;
    std::cout << " Total Frame Transitions     : " << metrics.total_transitions << std::endl;
    std::cout << " Detected Scene Boundaries   : " << metrics.total_scene_boundaries << std::endl;
    std::cout << " Anomaly / Violation Count   : " << metrics.anomaly_count << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << " Mean Frame Trust Score      : " << metrics.mean_frame_trust << " / 100" << std::endl;
    std::cout << " Avg Transition Integrity    : " << metrics.transition_integrity_avg << " / 100" << std::endl;
    std::cout << " Avg Scene Boundary Trust    : " << metrics.scene_boundary_trust_avg << " / 100" << std::endl;
    std::cout << " Temporal Continuity Score   : " << metrics.temporal_continuity_score << " / 100" << std::endl;
    std::cout << "   - Optical Flow Stability  : " << metrics.optical_flow_stability << " / 100" << std::endl;
    std::cout << "   - Temporal Rhythm Score   : " << metrics.rhythm_stability << " / 100" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << " Overall Temporal Trust      : " << metrics.overall_trust_score << " / 100" << std::endl;

    const char* status_str = "ACCEPTED";
    if (metrics.safety_status == SafetyStatus::PARTIALLY_TRUSTED) status_str = "PARTIALLY_TRUSTED";
    else if (metrics.safety_status == SafetyStatus::OUTRIGHT_REJECTED) status_str = "OUTRIGHT_REJECTED";

    std::cout << " Governance Decision Status  : " << status_str << std::endl;
    std::cout << "===========================================================" << std::endl << std::endl;

    std::cout << "[Step 3] Verifying Temporal Watermarks..." << std::endl;
    bool all_wm_valid = true;
    for (size_t i = 0; i < metrics.total_frames; ++i) {
        bool valid = engine.verify_frame_watermark(i, secret_key, sizeof(secret_key));
        if (!valid) all_wm_valid = false;
    }
    if (all_wm_valid) {
        std::cout << "  ✓ All 12 frame motion/boundary watermarks verified successfully!" << std::endl;
    } else {
        std::cout << "  ⚠ Watermark verification failed for one or more frames!" << std::endl;
    }

    std::cout << std::endl << "Demo completed successfully!" << std::endl;
    return 0;
}
