//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const IMAGING_PRESETS = {
  DOSE_SAFETY: {
    borderlineLow: 0.78,
    borderlineHigh: 0.94,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 50.0,
  },
};

export class ImagingHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "imaging";
  private pipeline: AileeTrustPipeline;

  constructor(preset = IMAGING_PRESETS.DOSE_SAFETY) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "ct_detector_array",
      quality: 0.96,
      readings: {
        photonCount: 15400,
        reconstructionConfidence: 0.89,
        radiationDoseMsv: 2.1,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const rawVal = Number(snapshot.readings.photonCount || 10000);
    const conf = Number(snapshot.readings.reconstructionConfidence || 0.8);

    return this.pipeline.process(rawVal, conf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "xray_tube_modulator",
      command: { action: "SAFE_LOW_DOSE_PRESET" },
      priority: 1,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
