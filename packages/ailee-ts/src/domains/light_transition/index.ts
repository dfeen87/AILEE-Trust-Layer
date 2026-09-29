//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, finiteSensorReading, SensorSnapshot } from "../../hardware/adapter.js";

export const LIGHT_TRANSITION_PRESETS = {
  PHOTONIC_INTEGRITY: {
    borderlineLow: 0.8,
    borderlineHigh: 0.95,
    hardMin: -30.0,
    hardMax: 10.0,
    defaultFallbackValue: -5.0,
  },
};

export class LightTransitionHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "light_transition";
  private pipeline: AileeTrustPipeline;

  constructor(preset = LIGHT_TRANSITION_PRESETS.PHOTONIC_INTEGRITY) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "fso_optical_transceiver",
      quality: 0.98,
      readings: {
        opticalPowerDbm: -2.4,
        bitErrorRate: 1e-9,
        scintillationIndex: 0.05,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const power = finiteSensorReading(snapshot, "opticalPowerDbm");
    const ber = finiteSensorReading(snapshot, "bitErrorRate");
    const rawConf = Math.max(0, Math.min(1, -Math.log10(ber + 1e-12) / 12.0));

    return this.pipeline.process(power, rawConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "laser_diode_driver",
      command: { targetPowerDbm: decision.value },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
