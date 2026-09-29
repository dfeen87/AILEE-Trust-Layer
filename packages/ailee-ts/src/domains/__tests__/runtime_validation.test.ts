//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { describe, expect, it } from "vitest";
import {
  AuditoryHardwareAdapter,
  AutomotiveHardwareAdapter,
  CrisprHardwareAdapter,
  CrossEcosystemHardwareAdapter,
  CryptoMiningHardwareAdapter,
  DatacenterHardwareAdapter,
  GovernanceHardwareAdapter,
  GridsHardwareAdapter,
  ImagingHardwareAdapter,
  LightTransitionHardwareAdapter,
  MemoryHardwareAdapter,
  NeuroAssistiveHardwareAdapter,
  OceanHardwareAdapter,
  ReleaseEventsHardwareAdapter,
  RoboticsHardwareAdapter,
  TelecommunicationsHardwareAdapter,
  TopologyHardwareAdapter,
  WatermarkProvenanceHardwareAdapter,
} from "../index.js";
import { DomainHardwareAdapter } from "../../hardware/adapter.js";

const adapters: Array<[string, DomainHardwareAdapter, string]> = [
  ["auditory", new AuditoryHardwareAdapter(), "soundPressureLevelDb"],
  ["automotive", new AutomotiveHardwareAdapter(), "wheelSpeedKmh"],
  ["crispr", new CrisprHardwareAdapter(), "seedMatchPercent"],
  ["cross ecosystem", new CrossEcosystemHardwareAdapter(), "semanticEquivalenceScore"],
  ["crypto mining", new CryptoMiningHardwareAdapter(), "chipTemperatureC"],
  ["datacenter", new DatacenterHardwareAdapter(), "rackInletTempC"],
  ["governance", new GovernanceHardwareAdapter(), "mandateValidityScore"],
  ["grids", new GridsHardwareAdapter(), "gridFrequencyHz"],
  ["imaging", new ImagingHardwareAdapter(), "photonCount"],
  ["light transition", new LightTransitionHardwareAdapter(), "opticalPowerDbm"],
  ["memory", new MemoryHardwareAdapter(), "ramUsagePercent"],
  ["neuro assistive", new NeuroAssistiveHardwareAdapter(), "cognitiveLoadIndex"],
  ["ocean", new OceanHardwareAdapter(), "dissolvedOxygenMgL"],
  ["release events", new ReleaseEventsHardwareAdapter(), "targetRolloutPercent"],
  ["robotics", new RoboticsHardwareAdapter(), "endEffectorSpeedMs"],
  ["telecommunications", new TelecommunicationsHardwareAdapter(), "latencyMs"],
  ["topology", new TopologyHardwareAdapter(), "connectivityIndex"],
  ["watermark provenance", new WatermarkProvenanceHardwareAdapter(), "rawWatermarkScore"],
];

describe.each(adapters)("%s runtime sensor validation", (_name, adapter, primaryField) => {
  it.each([undefined, null, "1", true, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "rejects malformed primary evidence %p",
    async (badValue) => {
      const snapshot = await adapter.readSensors();
      snapshot.readings[primaryField] = badValue;
      const decision = await adapter.evaluateState(snapshot);
      expect(decision.safetyStatus).toBe("OUTRIGHT_REJECTED");
      expect(decision.usedFallback).toBe(true);
      expect(decision.trustScore.aggregateScore).toBe(0);
    }
  );
});
