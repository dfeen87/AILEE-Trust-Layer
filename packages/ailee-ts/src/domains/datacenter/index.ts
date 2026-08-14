//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const DATACENTER_PRESETS = {
  PUE_OPTIMIZED: {
    borderlineLow: 0.72,
    borderlineHigh: 0.9,
    hardMin: 18.0,
    hardMax: 35.0,
    defaultFallbackValue: 22.0,
  },
};

export class DatacenterHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "datacenter";
  private pipeline: AileeTrustPipeline;

  constructor(preset = DATACENTER_PRESETS.PUE_OPTIMIZED) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "bms_rack_chiller",
      quality: 0.97,
      readings: {
        rackInletTempC: 23.5,
        pue: 1.28,
        coolantFlowRateLpm: 45.0,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const inletTemp = Number(snapshot.readings.rackInletTempC || 24.0);
    return this.pipeline.process(inletTemp, snapshot.quality, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "chiller_valve",
      command: { targetTemp: decision.value },
      priority: 1,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
