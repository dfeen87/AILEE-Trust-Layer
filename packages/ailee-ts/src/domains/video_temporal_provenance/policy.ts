//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export interface VideoTemporalPolicy {
  minOverallTrust: number;
  rejectSyntheticTransitions: boolean;
  requiredWatermarkKey?: Uint8Array;
}

export const DEFAULT_VIDEO_TEMPORAL_POLICY: VideoTemporalPolicy = {
  minOverallTrust: 70.0,
  rejectSyntheticTransitions: false,
};
