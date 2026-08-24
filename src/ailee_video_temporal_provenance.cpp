// Copyright (c) Don Michael Feeney Jr.
// Licensed under the MIT License.

#include "ailee_video_temporal_provenance.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace ailee {
namespace video {

// Helper: Deterministic zero-allocation 256-bit hash (FNV1a-32 expanded bit-mixing / transform)
static void deterministic_digest(const uint8_t* key, size_t key_len, const uint8_t* payload, size_t payload_len, uint8_t out_hash[WATERMARK_PAYLOAD_SIZE]) {
    std::memset(out_hash, 0, WATERMARK_PAYLOAD_SIZE);

    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    auto mix_bytes = [&](const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            size_t idx = i % 8;
            state[idx] ^= data[i];
            state[idx] *= 16777619u;
            state[(idx + 1) % 8] += state[idx] >> 3;
            state[idx] = (state[idx] << 13) | (state[idx] >> 19);
        }
    };

    if (key && key_len > 0) {
        mix_bytes(key, key_len);
    }
    if (payload && payload_len > 0) {
        mix_bytes(payload, payload_len);
    }

    // Final bit permutation into 32 output bytes
    for (int i = 0; i < 8; ++i) {
        uint32_t val = state[i];
        out_hash[i * 4 + 0] = static_cast<uint8_t>((val >> 24) & 0xFF);
        out_hash[i * 4 + 1] = static_cast<uint8_t>((val >> 16) & 0xFF);
        out_hash[i * 4 + 2] = static_cast<uint8_t>((val >> 8) & 0xFF);
        out_hash[i * 4 + 3] = static_cast<uint8_t>(val & 0xFF);
    }
}

// TemporalProvenanceChain Implementation
TemporalProvenanceChain::TemporalProvenanceChain() {
    reset();
}

void TemporalProvenanceChain::reset() {
    frame_count_ = 0;
    transition_count_ = 0;
    scene_boundary_count_ = 0;
    std::memset(frames_, 0, sizeof(frames_));
    std::memset(transitions_, 0, sizeof(transitions_));
    std::memset(scene_boundaries_, 0, sizeof(scene_boundaries_));
}

bool TemporalProvenanceChain::add_frame(const FrameData& frame) {
    if (frame_count_ >= MAX_CHAIN_FRAMES) return false;
    frames_[frame_count_++] = frame;
    return true;
}

bool TemporalProvenanceChain::add_transition(const TransitionData& transition) {
    if (transition_count_ >= MAX_CHAIN_TRANSITIONS) return false;
    transitions_[transition_count_++] = transition;
    return true;
}

bool TemporalProvenanceChain::add_scene_boundary(const SceneBoundaryData& boundary) {
    if (scene_boundary_count_ >= MAX_SCENE_BOUNDARIES) return false;
    scene_boundaries_[scene_boundary_count_++] = boundary;
    return true;
}

const FrameData* TemporalProvenanceChain::get_frame(size_t index) const {
    if (index >= frame_count_) return nullptr;
    return &frames_[index];
}

const TransitionData* TemporalProvenanceChain::get_transition(size_t index) const {
    if (index >= transition_count_) return nullptr;
    return &transitions_[index];
}

const SceneBoundaryData* TemporalProvenanceChain::get_scene_boundary(size_t index) const {
    if (index >= scene_boundary_count_) return nullptr;
    return &scene_boundaries_[index];
}

// SceneBoundaryOperator Implementation
SceneBoundaryData SceneBoundaryOperator::evaluate(const FrameData& prev_frame, const FrameData& curr_frame) {
    SceneBoundaryData sb{};
    sb.frame_index = curr_frame.frame_index;
    sb.timestamp_sec = curr_frame.timestamp_sec;
    sb.boundary_type = SceneBoundaryType::NONE;
    sb.boundary_trust_score = 100.0f;
    sb.is_natural = 1;
    sb.is_synthetic = 0;
    sb.flags = PROV_FLAG_NONE;

    float hash_delta = std::fabs(curr_frame.perceptual_hash_delta);
    float flow_consistency = curr_frame.motion.flow_consistency;

    if (hash_delta > 0.75f) {
        sb.boundary_type = SceneBoundaryType::HARD_CUT;
    } else if (hash_delta > 0.40f && flow_consistency < 0.35f) {
        sb.boundary_type = SceneBoundaryType::SYNTHETIC_TRANSITION;
    } else if (hash_delta > 0.35f) {
        sb.boundary_type = SceneBoundaryType::CROSS_DISSOLVE;
    } else if (hash_delta > 0.25f) {
        sb.boundary_type = SceneBoundaryType::FADE;
    }

    if (sb.boundary_type == SceneBoundaryType::SYNTHETIC_TRANSITION ||
        (curr_frame.provenance_flags & PROV_FLAG_SYNTHETIC_INTERP) != 0) {
        sb.is_natural = 0;
        sb.is_synthetic = 1;
        sb.flags |= PROV_FLAG_SYNTHETIC_INTERP;
        sb.boundary_trust_score = std::max(0.0f, 60.0f - (hash_delta * 20.0f));
    } else if (sb.boundary_type != SceneBoundaryType::NONE) {
        sb.boundary_trust_score = 90.0f;
    }

    return sb;
}

// TransitionIntegrityOperator Implementation
TransitionData TransitionIntegrityOperator::evaluate(const FrameData& prev_frame, const FrameData& curr_frame) {
    TransitionData td{};
    td.from_frame_index = prev_frame.frame_index;
    td.to_frame_index = curr_frame.frame_index;
    td.anomaly_flags = PROV_FLAG_NONE;

    float motion_diff_x = std::fabs(curr_frame.motion.dx - prev_frame.motion.dx);
    float motion_diff_y = std::fabs(curr_frame.motion.dy - prev_frame.motion.dy);
    float motion_delta = std::sqrt(motion_diff_x * motion_diff_x + motion_diff_y * motion_diff_y);

    td.motion_smoothness = std::max(0.0f, 1.0f - (motion_delta / 10.0f));
    td.interpolation_confidence = curr_frame.motion.flow_consistency;

    SceneBoundaryData sb = SceneBoundaryOperator::evaluate(prev_frame, curr_frame);
    td.boundary_type = sb.boundary_type;

    if (td.boundary_type == SceneBoundaryType::SYNTHETIC_TRANSITION ||
        (curr_frame.provenance_flags & PROV_FLAG_SYNTHETIC_INTERP)) {
        td.is_ai_generated_transition = 1;
        td.anomaly_flags |= PROV_FLAG_SYNTHETIC_INTERP;
    }

    if (td.motion_smoothness < 0.2f && td.boundary_type == SceneBoundaryType::NONE) {
        td.anomaly_flags |= PROV_FLAG_MOTION_HALLUCINATED;
    }

    float base_score = 100.0f;
    if (td.is_ai_generated_transition) base_score -= 35.0f;
    if (td.anomaly_flags & PROV_FLAG_MOTION_HALLUCINATED) base_score -= 30.0f;
    base_score = std::max(0.0f, base_score * td.motion_smoothness);

    td.transition_integrity_score = base_score;
    return td;
}

// TemporalContinuityOperator Implementation
float TemporalContinuityOperator::evaluate_sequence(
    const FrameData* frames,
    size_t count,
    float* flow_stability_out,
    float* rhythm_stability_out
) {
    if (!frames || count < 2) {
        if (flow_stability_out) *flow_stability_out = 100.0f;
        if (rhythm_stability_out) *rhythm_stability_out = 100.0f;
        return 100.0f;
    }

    float total_flow = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        total_flow += frames[i].motion.flow_consistency;
    }
    float avg_flow = total_flow / static_cast<float>(count);

    float dt_sum = 0.0f;
    float dt_sq_sum = 0.0f;
    size_t dt_count = count - 1;

    for (size_t i = 1; i < count; ++i) {
        float dt = static_cast<float>(frames[i].timestamp_sec - frames[i - 1].timestamp_sec);
        dt_sum += dt;
        dt_sq_sum += dt * dt;
    }

    float mean_dt = dt_sum / static_cast<float>(dt_count);
    float dt_var = (dt_sq_sum / static_cast<float>(dt_count)) - (mean_dt * mean_dt);
    if (dt_var < 0.0f) dt_var = 0.0f;
    float dt_std = std::sqrt(dt_var);

    float flow_stab = std::min(100.0f, std::max(0.0f, avg_flow * 100.0f));
    float rhythm_stab = std::min(100.0f, std::max(0.0f, 100.0f - (dt_std * 500.0f)));

    if (flow_stability_out) *flow_stability_out = flow_stab;
    if (rhythm_stability_out) *rhythm_stability_out = rhythm_stab;

    return (flow_stab * 0.6f) + (rhythm_stab * 0.4f);
}

// TemporalWatermarkLayer Implementation
void TemporalWatermarkLayer::compute_watermark(
    const uint8_t* secret_key,
    size_t key_len,
    const MotionData& motion,
    SceneBoundaryType btype,
    uint8_t out_watermark[WATERMARK_PAYLOAD_SIZE]
) {
    alignas(16) uint8_t payload[64];
    std::memset(payload, 0, sizeof(payload));

    std::memcpy(payload, &motion.dx, sizeof(float));
    std::memcpy(payload + 4, &motion.dy, sizeof(float));
    std::memcpy(payload + 8, &motion.magnitude, sizeof(float));
    std::memcpy(payload + 12, &motion.flow_consistency, sizeof(float));
    uint8_t bt = static_cast<uint8_t>(btype);
    payload[16] = bt;

    deterministic_digest(secret_key, key_len, payload, 17, out_watermark);
}

bool TemporalWatermarkLayer::verify_watermark(
    const uint8_t* secret_key,
    size_t key_len,
    const MotionData& motion,
    SceneBoundaryType btype,
    const uint8_t watermark[WATERMARK_PAYLOAD_SIZE]
) {
    uint8_t expected[WATERMARK_PAYLOAD_SIZE];
    compute_watermark(secret_key, key_len, motion, btype, expected);

    int diff = 0;
    for (size_t i = 0; i < WATERMARK_PAYLOAD_SIZE; ++i) {
        diff |= (expected[i] ^ watermark[i]);
    }
    return diff == 0;
}

// TemporalProvenanceEngine Implementation
TemporalProvenanceEngine::TemporalProvenanceEngine() {
    reset();
}

void TemporalProvenanceEngine::reset() {
    chain_.reset();
}

bool TemporalProvenanceEngine::ingest_frame(const FrameData& frame) {
    size_t prev_count = chain_.frame_count();
    if (!chain_.add_frame(frame)) return false;

    if (prev_count > 0) {
        const FrameData* prev = chain_.get_frame(prev_count - 1);
        const FrameData* curr = chain_.get_frame(prev_count);
        if (prev && curr) {
            TransitionData td = TransitionIntegrityOperator::evaluate(*prev, *curr);
            chain_.add_transition(td);

            SceneBoundaryData sb = SceneBoundaryOperator::evaluate(*prev, *curr);
            if (sb.boundary_type != SceneBoundaryType::NONE) {
                chain_.add_scene_boundary(sb);
            }
        }
    }
    return true;
}

bool TemporalProvenanceEngine::embed_watermark_to_frame(size_t frame_idx, const uint8_t* secret_key, size_t key_len) {
    if (frame_idx >= chain_.frame_count()) return false;

    // Const-cast inside engine implementation to mutate chain frame watermark deterministically
    FrameData* frame_ptr = const_cast<FrameData*>(chain_.get_frame(frame_idx));
    if (!frame_ptr) return false;

    SceneBoundaryType btype = SceneBoundaryType::NONE;
    for (size_t i = 0; i < chain_.scene_boundary_count(); ++i) {
        const SceneBoundaryData* sb = chain_.get_scene_boundary(i);
        if (sb && sb->frame_index == frame_ptr->frame_index) {
            btype = sb->boundary_type;
            break;
        }
    }

    TemporalWatermarkLayer::compute_watermark(
        secret_key, key_len, frame_ptr->motion, btype, frame_ptr->watermark
    );
    frame_ptr->provenance_flags |= PROV_FLAG_WATERMARK_VALID;
    return true;
}

bool TemporalProvenanceEngine::verify_frame_watermark(size_t frame_idx, const uint8_t* secret_key, size_t key_len) {
    if (frame_idx >= chain_.frame_count()) return false;

    FrameData* frame_ptr = const_cast<FrameData*>(chain_.get_frame(frame_idx));
    if (!frame_ptr) return false;

    SceneBoundaryType btype = SceneBoundaryType::NONE;
    for (size_t i = 0; i < chain_.scene_boundary_count(); ++i) {
        const SceneBoundaryData* sb = chain_.get_scene_boundary(i);
        if (sb && sb->frame_index == frame_ptr->frame_index) {
            btype = sb->boundary_type;
            break;
        }
    }

    bool valid = TemporalWatermarkLayer::verify_watermark(
        secret_key, key_len, frame_ptr->motion, btype, frame_ptr->watermark
    );

    if (valid) {
        frame_ptr->provenance_flags |= PROV_FLAG_WATERMARK_VALID;
        frame_ptr->provenance_flags &= ~PROV_FLAG_WATERMARK_INVALID;
    } else {
        frame_ptr->provenance_flags |= PROV_FLAG_WATERMARK_INVALID;
        frame_ptr->provenance_flags &= ~PROV_FLAG_WATERMARK_VALID;
    }
    return valid;
}

bool TemporalProvenanceEngine::evaluate_chain(TemporalIntegrityMetrics& out_metrics) {
    std::memset(&out_metrics, 0, sizeof(out_metrics));

    size_t f_count = chain_.frame_count();
    if (f_count == 0) {
        out_metrics.overall_trust_score = 0.0f;
        out_metrics.safety_status = SafetyStatus::OUTRIGHT_REJECTED;
        return false;
    }

    out_metrics.total_frames = static_cast<uint32_t>(f_count);
    out_metrics.total_transitions = static_cast<uint32_t>(chain_.transition_count());
    out_metrics.total_scene_boundaries = static_cast<uint32_t>(chain_.scene_boundary_count());

    float sum_frame_trust = 0.0f;
    for (size_t i = 0; i < f_count; ++i) {
        const FrameData* f = chain_.get_frame(i);
        if (f) {
            sum_frame_trust += f->trust_score;
            if (f->provenance_flags & (PROV_FLAG_SYNTHETIC_INTERP | PROV_FLAG_MOTION_HALLUCINATED | PROV_FLAG_WATERMARK_INVALID | PROV_FLAG_TAMPER_DETECTED)) {
                out_metrics.anomaly_count++;
            }
        }
    }
    out_metrics.mean_frame_trust = sum_frame_trust / static_cast<float>(f_count);

    float sum_transition = 0.0f;
    for (size_t i = 0; i < chain_.transition_count(); ++i) {
        const TransitionData* t = chain_.get_transition(i);
        if (t) {
            sum_transition += t->transition_integrity_score;
            if (t->anomaly_flags != PROV_FLAG_NONE) {
                out_metrics.anomaly_count++;
            }
        }
    }
    out_metrics.transition_integrity_avg = (chain_.transition_count() > 0)
        ? (sum_transition / static_cast<float>(chain_.transition_count()))
        : 100.0f;

    float sum_sb = 0.0f;
    for (size_t i = 0; i < chain_.scene_boundary_count(); ++i) {
        const SceneBoundaryData* sb = chain_.get_scene_boundary(i);
        if (sb) {
            sum_sb += sb->boundary_trust_score;
            if (sb->is_synthetic) {
                out_metrics.anomaly_count++;
            }
        }
    }
    out_metrics.scene_boundary_trust_avg = (chain_.scene_boundary_count() > 0)
        ? (sum_sb / static_cast<float>(chain_.scene_boundary_count()))
        : 100.0f;

    // Continuities
    out_metrics.temporal_continuity_score = TemporalContinuityOperator::evaluate_sequence(
        chain_.get_frame(0), f_count, &out_metrics.optical_flow_stability, &out_metrics.rhythm_stability
    );

    // Fail-Closed Overall Trust Formula
    float weighted_trust = (out_metrics.mean_frame_trust * 0.35f) +
                           (out_metrics.transition_integrity_avg * 0.30f) +
                           (out_metrics.temporal_continuity_score * 0.25f) +
                           (out_metrics.scene_boundary_trust_avg * 0.10f);

    // Penalty for anomalies
    if (out_metrics.anomaly_count > 0) {
        weighted_trust -= (out_metrics.anomaly_count * 10.0f);
    }

    out_metrics.overall_trust_score = std::max(0.0f, std::min(100.0f, weighted_trust));

    // Governance decision
    if (out_metrics.overall_trust_score >= 85.0f && out_metrics.anomaly_count == 0) {
        out_metrics.safety_status = SafetyStatus::ACCEPTED;
    } else if (out_metrics.overall_trust_score >= 50.0f) {
        out_metrics.safety_status = SafetyStatus::PARTIALLY_TRUSTED;
    } else {
        out_metrics.safety_status = SafetyStatus::OUTRIGHT_REJECTED;
    }

    return true;
}

} // namespace video
} // namespace ailee

// C ABI Implementation
struct AileeTPEHandle {
    ailee::video::TemporalProvenanceEngine engine;
};

extern "C" {

AileeTPEHandle* ailee_tpe_create(void) {
    return new (std::nothrow) AileeTPEHandle();
}

void ailee_tpe_destroy(AileeTPEHandle* handle) {
    delete handle;
}

void ailee_tpe_reset(AileeTPEHandle* handle) {
    if (handle) {
        handle->engine.reset();
    }
}

int ailee_tpe_ingest_frame(
    AileeTPEHandle* handle,
    uint64_t frame_index,
    double timestamp_sec,
    float raw_trust,
    float perceptual_hash_delta,
    float motion_dx,
    float motion_dy,
    float flow_consistency
) {
    if (!handle) return 0;

    ailee::video::FrameData f{};
    f.frame_index = frame_index;
    f.timestamp_sec = timestamp_sec;
    f.trust_score = raw_trust;
    f.perceptual_hash_delta = perceptual_hash_delta;
    f.motion.dx = motion_dx;
    f.motion.dy = motion_dy;
    f.motion.magnitude = std::sqrt(motion_dx * motion_dx + motion_dy * motion_dy);
    f.motion.flow_consistency = flow_consistency;
    f.provenance_flags = ailee::video::PROV_FLAG_NATURAL_CAPTURE;
    if (flow_consistency < 0.35f || raw_trust < 50.0f) {
        f.provenance_flags |= ailee::video::PROV_FLAG_SYNTHETIC_INTERP;
    }

    return handle->engine.ingest_frame(f) ? 1 : 0;
}

int ailee_tpe_evaluate(
    AileeTPEHandle* handle,
    ailee::video::TemporalIntegrityMetrics* out_metrics
) {
    if (!handle || !out_metrics) return 0;
    return handle->engine.evaluate_chain(*out_metrics) ? 1 : 0;
}

int ailee_tpe_embed_watermark(
    AileeTPEHandle* handle,
    uint64_t frame_index,
    const uint8_t* key,
    size_t key_len
) {
    if (!handle) return 0;
    return handle->engine.embed_watermark_to_frame(static_cast<size_t>(frame_index), key, key_len) ? 1 : 0;
}

int ailee_tpe_verify_watermark(
    AileeTPEHandle* handle,
    uint64_t frame_index,
    const uint8_t* key,
    size_t key_len
) {
    if (!handle) return 0;
    return handle->engine.verify_frame_watermark(static_cast<size_t>(frame_index), key, key_len) ? 1 : 0;
}

} // extern "C"
