//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export interface VideoTemporalConfig {
  minTrustThreshold: number;
  strictMode: boolean;
  enableWatermarkVerification: boolean;
  maxAllowedAnomalies: number;
  flowStabilityThreshold: number;
}

export const DEFAULT_VIDEO_TEMPORAL_CONFIG: VideoTemporalConfig = {
  minTrustThreshold: 75.0,
  strictMode: false,
  enableWatermarkVerification: true,
  maxAllowedAnomalies: 1,
  flowStabilityThreshold: 0.5,
};
