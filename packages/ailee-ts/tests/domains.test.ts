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
} from "../src/domains/index.js";

describe("17 Hardware Domain Adapters", () => {
  it("executes all 17 domain adapters successfully", async () => {
    const adapters = [
      new AuditoryHardwareAdapter(),
      new AutomotiveHardwareAdapter(),
      new CrisprHardwareAdapter(),
      new CrossEcosystemHardwareAdapter(),
      new CryptoMiningHardwareAdapter(),
      new DatacenterHardwareAdapter(),
      new GovernanceHardwareAdapter(),
      new GridsHardwareAdapter(),
      new ImagingHardwareAdapter(),
      new LightTransitionHardwareAdapter(),
      new MemoryHardwareAdapter(),
      new NeuroAssistiveHardwareAdapter(),
      new OceanHardwareAdapter(),
      new ReleaseEventsHardwareAdapter(),
      new RoboticsHardwareAdapter(),
      new TelecommunicationsHardwareAdapter(),
      new TopologyHardwareAdapter(),
    ];

    expect(adapters.length).toBe(17);

    for (const adapter of adapters) {
      expect(adapter.domainName).toBeDefined();
      const snapshot = await adapter.readSensors();
      expect(snapshot.timestamp).toBeGreaterThan(0);

      const decision = await adapter.evaluateState(snapshot);
      expect(decision.safetyStatus).toBeDefined();

      if (decision.usedFallback) {
        await adapter.triggerFallback(decision);
      }
    }
  });
});
