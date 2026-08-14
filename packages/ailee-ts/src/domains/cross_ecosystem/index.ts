//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const CROSS_ECOSYSTEM_PRESETS = {
  SEMANTIC_INVARIANCE: {
    borderlineLow: 0.75,
    borderlineHigh: 0.92,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 0.0,
  },
};

export class CrossEcosystemHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "cross_ecosystem";
  private pipeline: AileeTrustPipeline;

  constructor(preset = CROSS_ECOSYSTEM_PRESETS.SEMANTIC_INVARIANCE) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "ecosystem_bridge_node",
      quality: 0.94,
      readings: {
        sourceProtocol: "AppleHomeKit",
        targetProtocol: "Matter",
        translationConfidence: 0.91,
        semanticEquivalenceScore: 92.5,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const semanticScore = Number(snapshot.readings.semanticEquivalenceScore || 0.0);
    const translationConf = Number(snapshot.readings.translationConfidence || 0.0);

    return this.pipeline.process(semanticScore, translationConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "protocol_translator",
      command: { action: "REVERT_TO_BASE_SEMANTICS" },
      priority: 1,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
