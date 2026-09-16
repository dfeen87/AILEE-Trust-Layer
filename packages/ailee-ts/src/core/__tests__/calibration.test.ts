//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { describe, expect, it } from "vitest";
import { CalibrationLayer } from "../calibration.js";

describe("CalibrationLayer", () => {
  const enabled = {
    enabled: true,
    acceptanceThreshold: 0.95,
    uncertaintyBand: 0.05,
    maxGraceMargin: 0.02,
    consensusThreshold: 0.8,
    minimumPeerCount: 2,
  };

  it("applies bounded grace only in the uncertainty zone with peer consensus", () => {
    const result = new CalibrationLayer(enabled).calibrate(0.94, {
      graceMargin: 0.02,
      peerConsensus: { agreement: 0.9, peerCount: 2 },
    });

    expect(result.confidence).toBe(0.95);
    expect(result.event).toBe("GRACE_APPLIED");
    expect(result.consensusChecked).toBe(true);
  });

  it("retains confidence at both uncertainty-zone boundaries without qualifying consensus", () => {
    const calibration = new CalibrationLayer(enabled);
    expect(calibration.calibrate(0.9).thresholdDecision).toBe("UNCERTAINTY_ZONE");
    expect(calibration.calibrate(0.9).event).toBe("CONSENSUS_NOT_MET");
    expect(calibration.calibrate(0.95).thresholdDecision).toBe("ABOVE_THRESHOLD");
    expect(calibration.calibrate(0.899).thresholdDecision).toBe("BELOW_UNCERTAINTY_ZONE");
  });

  it("fails safe to the V8 baseline for malformed metadata and invalid configuration", () => {
    const malformed = new CalibrationLayer(enabled).calibrate(0.94, { peerConsensus: { agreement: 1.2, peerCount: 2 } } as any);
    const invalidConfig = new CalibrationLayer({ ...enabled, uncertaintyBand: -1 }).calibrate(0.94);

    expect(malformed).toMatchObject({ confidence: 0.94, fallbackUsed: true, event: "INVALID_INPUT" });
    expect(invalidConfig).toMatchObject({ confidence: 0.94, fallbackUsed: true, event: "INVALID_INPUT" });
  });

  it("is deterministic and leaves V8 behavior unchanged when disabled", () => {
    const layer = new CalibrationLayer();
    const input = { graceMargin: 0.02, peerConsensus: { agreement: 1, peerCount: 3 } };

    expect(layer.calibrate(0.94, input)).toEqual(layer.calibrate(0.94, input));
    expect(layer.calibrate(0.94, input)).toMatchObject({ confidence: 0.94, event: "DISABLED", applied: false });
  });
});
