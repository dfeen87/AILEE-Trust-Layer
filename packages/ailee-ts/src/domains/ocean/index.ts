//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const OCEAN_PRESETS = {
  PRECAUTIONARY_RESTRAINT: {
    borderlineLow: 0.85,
    borderlineHigh: 0.98,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 0.0,
  },
};

export class OceanHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "ocean";
  private pipeline: AileeTrustPipeline;

  constructor(preset = OCEAN_PRESETS.PRECAUTIONARY_RESTRAINT) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "auv_buoy_array",
      quality: 0.95,
      readings: {
        dissolvedOxygenMgL: 6.8,
        phLevel: 8.1,
        turbidityNtu: 1.2,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const do2 = Number(snapshot.readings.dissolvedOxygenMgL || 7.0);
    return this.pipeline.process(do2, snapshot.quality, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "ocean_dosing_valve",
      command: { action: "HOLD_OBSERVE_ONLY" },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
