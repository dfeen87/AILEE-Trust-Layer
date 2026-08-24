//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { VideoTemporalConfig, DEFAULT_VIDEO_TEMPORAL_CONFIG } from "./config.js";
import { VideoTemporalPolicy, DEFAULT_VIDEO_TEMPORAL_POLICY } from "./policy.js";
import { FrameSignal, TemporalIntegrityMetrics, TPEFFIBridge } from "./ffi.js";

export interface VideoGovernorDecision {
  overallTrustScore: number;
  safetyStatus: "ACCEPTED" | "PARTIALLY_TRUSTED" | "OUTRIGHT_REJECTED";
  anomalyCount: number;
  reason: string;
  metrics: TemporalIntegrityMetrics;
}

export class VideoTemporalGovernor {
  private config: VideoTemporalConfig;
  private policy: VideoTemporalPolicy;
  private bridge: TPEFFIBridge;

  constructor(
    config: VideoTemporalConfig = DEFAULT_VIDEO_TEMPORAL_CONFIG,
    policy: VideoTemporalPolicy = DEFAULT_VIDEO_TEMPORAL_POLICY
  ) {
    this.config = config;
    this.policy = policy;
    this.bridge = new TPEFFIBridge();
  }

  public reset(): void {
    this.bridge.reset();
  }

  public evaluateSequence(frames: FrameSignal[]): VideoGovernorDecision {
    this.bridge.reset();
    for (const f of frames) {
      this.bridge.ingestFrame(f);
    }

    const metrics = this.bridge.evaluate();
    let status = metrics.safetyStatus;
    let reason = "Sequence evaluated successfully.";

    if (metrics.overallTrustScore < this.config.minTrustThreshold) {
      status = "OUTRIGHT_REJECTED";
      reason = `Overall trust ${metrics.overallTrustScore.toFixed(1)} below threshold ${this.config.minTrustThreshold.toFixed(1)}.`;
    } else if (metrics.anomalyCount > this.config.maxAllowedAnomalies) {
      if (status !== "OUTRIGHT_REJECTED") {
        status = "PARTIALLY_TRUSTED";
      }
      reason = `Anomaly count ${metrics.anomalyCount} exceeded max allowed ${this.config.maxAllowedAnomalies}.`;
    }

    return {
      overallTrustScore: metrics.overallTrustScore,
      safetyStatus: status,
      anomalyCount: metrics.anomalyCount,
      reason,
      metrics,
    };
  }
}
