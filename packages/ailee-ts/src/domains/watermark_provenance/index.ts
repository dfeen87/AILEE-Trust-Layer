//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const WATERMARK_PROVENANCE_PRESETS = {
  BALANCED_PROVENANCE: {
    borderlineLow: 0.6,
    borderlineHigh: 0.85,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 0.0,
  },
  STRICT_HIGH_STAKES: {
    borderlineLow: 0.8,
    borderlineHigh: 0.95,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 0.0,
  },
};

export class WatermarkProvenanceHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "watermark_provenance";
  private pipeline: AileeTrustPipeline;

  constructor(preset = WATERMARK_PROVENANCE_PRESETS.BALANCED_PROVENANCE) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "watermark_detector_sensor_01",
      quality: 0.95,
      readings: {
        rawWatermarkScore: 0.82,
        detectorConfidence: 0.9,
        custodyNodeCount: 3,
        disruptionDetected: 0,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const rawScore = Number(snapshot.readings.rawWatermarkScore || 0.0);
    const confidence = Number(snapshot.readings.detectorConfidence || 0.0);
    const rawConf = Math.max(0, Math.min(1.0, rawScore * confidence));

    return this.pipeline.process(rawScore * 100, rawConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "provenance_governance_gate",
      command: { action: "FLAG_FOR_HUMAN_REVIEW" },
      priority: 1,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
