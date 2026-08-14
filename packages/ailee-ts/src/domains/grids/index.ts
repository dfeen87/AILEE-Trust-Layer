//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const GRIDS_PRESETS = {
  FREQUENCY_STABILITY: {
    borderlineLow: 0.82,
    borderlineHigh: 0.96,
    hardMin: 59.0,
    hardMax: 61.0,
    defaultFallbackValue: 60.0,
  },
};

export class GridsHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "grids";
  private pipeline: AileeTrustPipeline;

  constructor(preset = GRIDS_PRESETS.FREQUENCY_STABILITY) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "pmu_grid_node_01",
      quality: 0.99,
      readings: {
        gridFrequencyHz: 59.98,
        voltageKv: 115.2,
        phaseAngleDeg: 0.45,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const freq = Number(snapshot.readings.gridFrequencyHz || 60.0);
    return this.pipeline.process(freq, snapshot.quality, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "load_tap_changer",
      command: { targetFrequency: decision.value },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
