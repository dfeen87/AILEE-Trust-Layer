//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const CRISPR_PRESETS = {
  THERMODYNAMIC_STRICT: {
    borderlineLow: 0.9,
    borderlineHigh: 0.99,
    hardMin: 0.0,
    hardMax: 100.0,
    defaultFallbackValue: 0.0,
  },
};

export class CrisprHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "crispr";
  private pipeline: AileeTrustPipeline;

  constructor(preset = CRISPR_PRESETS.THERMODYNAMIC_STRICT) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "gene_sequencer_node",
      quality: 0.98,
      readings: {
        pamVerified: true,
        seedMatchPercent: 100.0,
        distalMismatchScore: 0.02,
        thermodynamicTolerance: 0.96,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const seedMatch = Number(snapshot.readings.seedMatchPercent || 0.0);
    const pamVerified = Boolean(snapshot.readings.pamVerified);
    const rawConfidence = pamVerified ? seedMatch / 100.0 : 0.0;

    return this.pipeline.process(seedMatch, rawConfidence, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "fluidic_dispenser",
      command: { action: "HALT_DISPENSE", reason: "GENETIC_SAFETY_REJECTION" },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
