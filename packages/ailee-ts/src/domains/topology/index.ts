//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const TOPOLOGY_PRESETS = {
  GRAPH_STABILITY: {
    borderlineLow: 0.78,
    borderlineHigh: 0.94,
    hardMin: 0.0,
    hardMax: 1.0,
    defaultFallbackValue: 0.5,
  },
};

export class TopologyHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "topology";
  private pipeline: AileeTrustPipeline;

  constructor(preset = TOPOLOGY_PRESETS.GRAPH_STABILITY) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "mesh_topology_controller",
      quality: 0.97,
      readings: {
        connectivityIndex: 0.92,
        nodeCount: 256,
        orphanNodes: 0,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const connIndex = Number(snapshot.readings.connectivityIndex || 0.5);
    return this.pipeline.process(connIndex, snapshot.quality, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "mesh_route_table",
      command: { action: "FREEZE_TOPOLOGY_MUTATIONS" },
      priority: 1,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
