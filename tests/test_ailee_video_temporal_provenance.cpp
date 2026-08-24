// Copyright (c) Don Michael Feeney Jr.
// Licensed under the MIT License.

#include "ailee_video_temporal_provenance.hpp"
#include <iostream>
#include <cassert>
#include <cstring>
#include <cmath>

void test_scene_boundary_operator() {
    std::cout << "[Test] Scene Boundary Operator..." << std::endl;
    using namespace ailee::video;

    FrameData f0{};
    f0.frame_index = 0;
    f0.timestamp_sec = 0.0;
    f0.trust_score = 95.0f;
    f0.perceptual_hash_delta = 0.05f;
    f0.motion.flow_consistency = 0.95f;

    FrameData f1{};
    f1.frame_index = 1;
    f1.timestamp_sec = 0.033;
    f1.trust_score = 92.0f;
    f1.perceptual_hash_delta = 0.85f; // Large jump -> Hard Cut
    f1.motion.flow_consistency = 0.90f;

    SceneBoundaryData sb = SceneBoundaryOperator::evaluate(f0, f1);
    assert(sb.boundary_type == SceneBoundaryType::HARD_CUT);
    assert(sb.is_natural == 1);

    // Test synthetic transition detection
    FrameData f2{};
    f2.frame_index = 2;
    f2.timestamp_sec = 0.066;
    f2.trust_score = 70.0f;
    f2.perceptual_hash_delta = 0.50f;
    f2.motion.flow_consistency = 0.20f; // Low flow consistency + high delta
    f2.provenance_flags = PROV_FLAG_SYNTHETIC_INTERP;

    SceneBoundaryData sb_synth = SceneBoundaryOperator::evaluate(f1, f2);
    assert(sb_synth.boundary_type == SceneBoundaryType::SYNTHETIC_TRANSITION);
    assert(sb_synth.is_synthetic == 1);

    std::cout << "  ✓ Scene boundary tests passed!" << std::endl;
}

void test_transition_integrity_operator() {
    std::cout << "[Test] Transition Integrity Operator..." << std::endl;
    using namespace ailee::video;

    FrameData f0{};
    f0.frame_index = 0;
    f0.motion.dx = 1.0f;
    f0.motion.dy = 0.0f;
    f0.motion.flow_consistency = 0.95f;

    FrameData f1{};
    f1.frame_index = 1;
    f1.motion.dx = 1.1f;
    f1.motion.dy = 0.05f;
    f1.motion.flow_consistency = 0.94f;

    TransitionData td = TransitionIntegrityOperator::evaluate(f0, f1);
    assert(td.transition_integrity_score > 85.0f);
    assert(td.is_ai_generated_transition == 0);

    // Synthetic transition
    f1.provenance_flags = PROV_FLAG_SYNTHETIC_INTERP;
    TransitionData td_synth = TransitionIntegrityOperator::evaluate(f0, f1);
    assert(td_synth.is_ai_generated_transition == 1);
    assert(td_synth.transition_integrity_score < td.transition_integrity_score);

    std::cout << "  ✓ Transition integrity tests passed!" << std::endl;
}

void test_temporal_continuity_operator() {
    std::cout << "[Test] Temporal Continuity Operator..." << std::endl;
    using namespace ailee::video;

    FrameData frames[5];
    for (int i = 0; i < 5; ++i) {
        frames[i].frame_index = i;
        frames[i].timestamp_sec = i * 0.033;
        frames[i].motion.flow_consistency = 0.95f;
    }

    float flow_stab = 0.0f;
    float rhythm_stab = 0.0f;
    float score = TemporalContinuityOperator::evaluate_sequence(frames, 5, &flow_stab, &rhythm_stab);

    assert(score > 85.0f);
    assert(flow_stab > 80.0f);
    assert(rhythm_stab > 80.0f);

    std::cout << "  ✓ Temporal continuity tests passed!" << std::endl;
}

void test_watermark_layer() {
    std::cout << "[Test] Temporal Watermarking Layer..." << std::endl;
    using namespace ailee::video;

    uint8_t key[16] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};
    MotionData motion{};
    motion.dx = 2.5f;
    motion.dy = -1.2f;
    motion.magnitude = 2.77f;
    motion.flow_consistency = 0.92f;

    uint8_t wm[WATERMARK_PAYLOAD_SIZE];
    TemporalWatermarkLayer::compute_watermark(key, sizeof(key), motion, SceneBoundaryType::NONE, wm);

    bool verified = TemporalWatermarkLayer::verify_watermark(key, sizeof(key), motion, SceneBoundaryType::NONE, wm);
    assert(verified);

    // Tamper test
    motion.dx = 2.6f;
    bool tamper_verified = TemporalWatermarkLayer::verify_watermark(key, sizeof(key), motion, SceneBoundaryType::NONE, wm);
    assert(!tamper_verified);

    std::cout << "  ✓ Watermark layer tests passed!" << std::endl;
}

void test_engine_integration() {
    std::cout << "[Test] Temporal Provenance Engine Integration..." << std::endl;
    using namespace ailee::video;

    TemporalProvenanceEngine engine;
    uint8_t key[] = "ailee_secret_key";

    for (int i = 0; i < 10; ++i) {
        FrameData f{};
        f.frame_index = i;
        f.timestamp_sec = i * 0.0333;
        f.trust_score = 90.0f + (i % 3);
        f.perceptual_hash_delta = (i == 5) ? 0.80f : 0.05f; // Frame 5 has a hard cut
        f.motion.dx = 1.0f;
        f.motion.dy = 0.2f;
        f.motion.flow_consistency = 0.92f;

        bool ingested = engine.ingest_frame(f);
        assert(ingested);

        engine.embed_watermark_to_frame(i, key, sizeof(key));
        bool v = engine.verify_frame_watermark(i, key, sizeof(key));
        assert(v);
    }

    TemporalIntegrityMetrics metrics;
    bool eval_ok = engine.evaluate_chain(metrics);
    assert(eval_ok);
    assert(metrics.total_frames == 10);
    assert(metrics.total_scene_boundaries >= 1);
    assert(metrics.overall_trust_score >= 70.0f);
    assert(metrics.safety_status == SafetyStatus::ACCEPTED || metrics.safety_status == SafetyStatus::PARTIALLY_TRUSTED);

    std::cout << "  ✓ Engine integration tests passed!" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "AILEE-Video Temporal Provenance Tests (v7.0.0)" << std::endl;
    std::cout << "==========================================" << std::endl;

    test_scene_boundary_operator();
    test_transition_integrity_operator();
    test_temporal_continuity_operator();
    test_watermark_layer();
    test_engine_integration();

    std::cout << "All C++ Unit Tests PASSED successfully!" << std::endl;
    return 0;
}
