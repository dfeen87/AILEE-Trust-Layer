//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const GOVERNANCE_PRESETS = {
  INSTITUTIONAL_STRICT: {
    borderlineLow: 0.8,
    borderlineHigh: 0.95,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 0.0,
  },
};

export class GovernanceHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "governance";
  private pipeline: AileeTrustPipeline;

  constructor(preset = GOVERNANCE_PRESETS.INSTITUTIONAL_STRICT) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "civic_authorization_node",
      quality: 0.99,
      readings: {
        mandateValidityScore: 95.0,
        consensusQuorumPercent: 88.0,
        signoffVerified: true,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const mandate = Number(snapshot.readings.mandateValidityScore || 0.0);
    const quorum = Number(snapshot.readings.consensusQuorumPercent || 0.0);
    const rawConf = (quorum / 100.0) * snapshot.quality;

    return this.pipeline.process(mandate, rawConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "policy_enforcer",
      command: { action: "REVERT_TO_HUMAN_OVERSIGHT" },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
