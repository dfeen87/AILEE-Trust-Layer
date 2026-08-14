//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const TELECOMMUNICATIONS_PRESETS = {
  QOS_FRESHNESS: {
    borderlineLow: 0.75,
    borderlineHigh: 0.92,
    hardMin: 0.0,
    hardMax: 1000.0,
    defaultFallbackValue: 100.0,
  },
};

export class TelecommunicationsHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "telecommunications";
  private pipeline: AileeTrustPipeline;

  constructor(preset = TELECOMMUNICATIONS_PRESETS.QOS_FRESHNESS) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "5g_gnb_mac_scheduler",
      quality: 0.98,
      readings: {
        latencyMs: 4.2,
        jitterMs: 0.8,
        packetLossPercent: 0.001,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const latency = Number(snapshot.readings.latencyMs || 10.0);
    const loss = Number(snapshot.readings.packetLossPercent || 0.0);
    const rawConf = Math.max(0, 1.0 - loss * 100.0);

    return this.pipeline.process(latency, rawConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "qos_traffic_shaper",
      command: { action: "REROUTE_TO_SAFE_SLICE" },
      priority: 1,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
