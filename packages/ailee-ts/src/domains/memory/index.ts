//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const MEMORY_PRESETS = {
  RAM_OOM_SAFEGUARD: {
    borderlineLow: 0.75,
    borderlineHigh: 0.92,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 70.0,
  },
};

export class MemoryHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "memory";
  private pipeline: AileeTrustPipeline;

  constructor(preset = MEMORY_PRESETS.RAM_OOM_SAFEGUARD) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "host_memory_controller",
      quality: 0.99,
      readings: {
        ramUsagePercent: 68.4,
        swapUsagePercent: 12.1,
        oomKillEvents: 0,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const ramPercent = Number(snapshot.readings.ramUsagePercent || 50.0);
    return this.pipeline.process(ramPercent, snapshot.quality, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "cgroup_memory_controller",
      command: { action: "TRIGGER_SAFE_GC_AND_BALLOON" },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
