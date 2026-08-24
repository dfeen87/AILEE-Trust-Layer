// Copyright (c) Don Michael Feeney Jr.
// Licensed under the MIT License.

#ifndef AILEE_VIDEO_TEMPORAL_PROVENANCE_HPP
#define AILEE_VIDEO_TEMPORAL_PROVENANCE_HPP

#include <cstdint>
#include <cstddef>

namespace ailee {
namespace video {

constexpr size_t MAX_CHAIN_FRAMES = 1024;
constexpr size_t MAX_CHAIN_TRANSITIONS = MAX_CHAIN_FRAMES - 1;
constexpr size_t MAX_SCENE_BOUNDARIES = 128;
constexpr size_t WATERMARK_PAYLOAD_SIZE = 32;

enum ProvenanceFlags : uint32_t {
    PROV_FLAG_NONE                = 0,
    PROV_FLAG_NATURAL_CAPTURE     = 1 << 0,
    PROV_FLAG_AI_GENERATED_FRAME  = 1 << 1,
    PROV_FLAG_SYNTHETIC_INTERP    = 1 << 2,
    PROV_FLAG_MOTION_HALLUCINATED = 1 << 3,
    PROV_FLAG_PACING_DISCONTINUITY = 1 << 4,
    PROV_FLAG_WATERMARK_VALID     = 1 << 5,
    PROV_FLAG_WATERMARK_INVALID   = 1 << 6,
    PROV_FLAG_TAMPER_DETECTED     = 1 << 7
};

enum class SceneBoundaryType : uint8_t {
    NONE = 0,
    HARD_CUT = 1,
    FADE = 2,
    CROSS_DISSOLVE = 3,
    SYNTHETIC_TRANSITION = 4
};

enum class SafetyStatus : uint8_t {
    ACCEPTED = 0,
    PARTIALLY_TRUSTED = 1,
    OUTRIGHT_REJECTED = 2
};

struct alignas(64) MotionData {
    float dx;
    float dy;
    float magnitude;
    float direction;
    float flow_consistency;
    float acceleration;
    float camera_panning_rate;
    float camera_zoom_rate;
    uint8_t reserved[32];
};

struct alignas(64) FrameData {
    uint64_t frame_index;
    double timestamp_sec;
    float trust_score;
    uint32_t provenance_flags;
    float perceptual_hash_delta;
    float motion_magnitude;
    MotionData motion;
    uint8_t watermark[WATERMARK_PAYLOAD_SIZE];
};

struct alignas(64) TransitionData {
    uint64_t from_frame_index;
    uint64_t to_frame_index;
    float transition_integrity_score;
    float interpolation_confidence;
    float motion_smoothness;
    uint32_t anomaly_flags;
    SceneBoundaryType boundary_type;
    uint8_t is_ai_generated_transition;
    uint8_t watermark[WATERMARK_PAYLOAD_SIZE];
};

struct alignas(64) SceneBoundaryData {
    uint64_t frame_index;
    double timestamp_sec;
    SceneBoundaryType boundary_type;
    float boundary_trust_score;
    uint8_t is_natural;
    uint8_t is_synthetic;
    uint32_t flags;
};

struct alignas(64) TemporalIntegrityMetrics {
    float overall_trust_score;
    float mean_frame_trust;
    float transition_integrity_avg;
    float scene_boundary_trust_avg;
    float temporal_continuity_score;
    float optical_flow_stability;
    float rhythm_stability;
    uint32_t total_frames;
    uint32_t total_transitions;
    uint32_t total_scene_boundaries;
    uint32_t anomaly_count;
    SafetyStatus safety_status;
};

class TemporalProvenanceChain {
public:
    TemporalProvenanceChain();

    void reset();
    bool add_frame(const FrameData& frame);
    bool add_transition(const TransitionData& transition);
    bool add_scene_boundary(const SceneBoundaryData& boundary);

    size_t frame_count() const { return frame_count_; }
    size_t transition_count() const { return transition_count_; }
    size_t scene_boundary_count() const { return scene_boundary_count_; }

    const FrameData* get_frame(size_t index) const;
    const TransitionData* get_transition(size_t index) const;
    const SceneBoundaryData* get_scene_boundary(size_t index) const;

private:
    FrameData frames_[MAX_CHAIN_FRAMES];
    TransitionData transitions_[MAX_CHAIN_TRANSITIONS];
    SceneBoundaryData scene_boundaries_[MAX_SCENE_BOUNDARIES];
    size_t frame_count_{0};
    size_t transition_count_{0};
    size_t scene_boundary_count_{0};
};

class SceneBoundaryOperator {
public:
    static SceneBoundaryData evaluate(const FrameData& prev_frame, const FrameData& curr_frame);
};

class TransitionIntegrityOperator {
public:
    static TransitionData evaluate(const FrameData& prev_frame, const FrameData& curr_frame);
};

class TemporalContinuityOperator {
public:
    static float evaluate_sequence(const FrameData* frames, size_t count, float* flow_stability_out, float* rhythm_stability_out);
};

class TemporalWatermarkLayer {
public:
    static void compute_watermark(const uint8_t* secret_key, size_t key_len, const MotionData& motion, SceneBoundaryType btype, uint8_t out_watermark[WATERMARK_PAYLOAD_SIZE]);
    static bool verify_watermark(const uint8_t* secret_key, size_t key_len, const MotionData& motion, SceneBoundaryType btype, const uint8_t watermark[WATERMARK_PAYLOAD_SIZE]);
};

class TemporalProvenanceEngine {
public:
    TemporalProvenanceEngine();

    void reset();
    bool ingest_frame(const FrameData& frame);
    bool evaluate_chain(TemporalIntegrityMetrics& out_metrics);
    bool embed_watermark_to_frame(size_t frame_idx, const uint8_t* secret_key, size_t key_len);
    bool verify_frame_watermark(size_t frame_idx, const uint8_t* secret_key, size_t key_len);

    const TemporalProvenanceChain& chain() const { return chain_; }
    TemporalProvenanceChain& chain() { return chain_; }

private:
    TemporalProvenanceChain chain_;
};

} // namespace video
} // namespace ailee

extern "C" {

typedef struct AileeTPEHandle AileeTPEHandle;

AileeTPEHandle* ailee_tpe_create(void);
void ailee_tpe_destroy(AileeTPEHandle* handle);
void ailee_tpe_reset(AileeTPEHandle* handle);

int ailee_tpe_ingest_frame(
    AileeTPEHandle* handle,
    uint64_t frame_index,
    double timestamp_sec,
    float raw_trust,
    float perceptual_hash_delta,
    float motion_dx,
    float motion_dy,
    float flow_consistency
);

int ailee_tpe_evaluate(
    AileeTPEHandle* handle,
    ailee::video::TemporalIntegrityMetrics* out_metrics
);

int ailee_tpe_embed_watermark(
    AileeTPEHandle* handle,
    uint64_t frame_index,
    const uint8_t* key,
    size_t key_len
);

int ailee_tpe_verify_watermark(
    AileeTPEHandle* handle,
    uint64_t frame_index,
    const uint8_t* key,
    size_t key_len
);

} // extern "C"

#endif // AILEE_VIDEO_TEMPORAL_PROVENANCE_HPP