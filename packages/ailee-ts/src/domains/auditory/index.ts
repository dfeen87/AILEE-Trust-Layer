//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const AUDITORY_PRESETS = {
  STRICT: {
    borderlineLow: 0.8,
    borderlineHigh: 0.95,
    hardMin: 0.0,
    hardMax: 85.0,
    defaultFallbackValue: 60.0,
  },
  BALANCED: {
    borderlineLow: 0.7,
    borderlineHigh: 0.9,
    hardMin: 0.0,
    hardMax: 90.0,
    defaultFallbackValue: 65.0,
  },
};

export class AuditoryHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "auditory";
  private pipeline: AileeTrustPipeline;
  private currentGain = 65.0;

  constructor(preset = AUDITORY_PRESETS.BALANCED) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "audio_dsp_01",
      quality: 0.95,
      readings: {
        soundPressureLevelDb: 72.5,
        signalToNoiseRatio: 18.2,
        clippingDetected: false,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const rawSpl = Number(snapshot.readings.soundPressureLevelDb || 70.0);
    const snr = Number(snapshot.readings.signalToNoiseRatio || 15.0);
    const rawConfidence = Math.min(1.0, snr / 25.0);

    const decision = this.pipeline.process(rawSpl, rawConfidence, [], trustContext);
    if (decision.safetyStatus === "ACCEPTED") {
      this.currentGain = decision.value;
    }
    return decision;
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    this.currentGain = decision.value;
    await this.writeActuators({
      actuatorId: "dsp_volume_limit",
      command: this.currentGain,
      priority: 1,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
