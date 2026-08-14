//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const NEURO_ASSISTIVE_PRESETS = {
  COGNITIVE_AUTONOMY: {
    borderlineLow: 0.8,
    borderlineHigh: 0.95,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 20.0,
  },
};

export class NeuroAssistiveHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "neuro_assistive";
  private pipeline: AileeTrustPipeline;

  constructor(preset = NEURO_ASSISTIVE_PRESETS.COGNITIVE_AUTONOMY) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "eeg_bci_headband",
      quality: 0.92,
      readings: {
        cognitiveLoadIndex: 42.0,
        userConsentVerified: true,
        fatigueScore: 18.5,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const cogLoad = Number(snapshot.readings.cognitiveLoadIndex || 30.0);
    const consent = Boolean(snapshot.readings.userConsentVerified);
    const rawConf = consent ? snapshot.quality : 0.0;

    return this.pipeline.process(cogLoad, rawConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "haptic_interface",
      command: { action: "SIMPLIFY_INTERFACE_GENTLE" },
      priority: 1,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
