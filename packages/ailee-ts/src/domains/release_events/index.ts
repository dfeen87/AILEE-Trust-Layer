//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const RELEASE_EVENTS_PRESETS = {
  CANARY_ROLLOUT: {
    borderlineLow: 0.8,
    borderlineHigh: 0.95,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 0.0,
  },
};

export class ReleaseEventsHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "release_events";
  private pipeline: AileeTrustPipeline;

  constructor(preset = RELEASE_EVENTS_PRESETS.CANARY_ROLLOUT) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "firmware_deployment_gate",
      quality: 0.98,
      readings: {
        errorRatePercent: 0.01,
        latencyMs: 12.5,
        targetRolloutPercent: 10.0,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const rolloutTarget = Number(snapshot.readings.targetRolloutPercent || 0.0);
    const errorRate = Number(snapshot.readings.errorRatePercent || 0.0);
    const rawConf = Math.max(0, 1.0 - errorRate * 10.0);

    return this.pipeline.process(rolloutTarget, rawConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "ota_firmware_flasher",
      command: { action: "ROLLBACK_FIRMWARE" },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
