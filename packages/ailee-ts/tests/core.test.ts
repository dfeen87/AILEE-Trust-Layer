//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { describe, expect, it } from "vitest";
import { AileeTrustPipeline, ConsensusStrategy } from "../src/core/index.js";

describe("Core AILEE Trust Pipeline", () => {
  it("accepts high confidence signals", () => {
    const pipeline = new AileeTrustPipeline({
      borderlineLow: 0.7,
      borderlineHigh: 0.9,
    });

    const res = pipeline.process(10.5, 0.95);
    expect(res.safetyStatus).toBe("ACCEPTED");
    expect(res.usedFallback).toBe(false);
    expect(res.value).toBe(10.5);
  });

  it("triggers fallback on outright rejected signals", () => {
    const pipeline = new AileeTrustPipeline({
      borderlineLow: 0.7,
      borderlineHigh: 0.9,
      defaultFallbackValue: 5.0,
    });

    const res = pipeline.process(100.0, 0.1);
    expect(res.safetyStatus).toBe("OUTRIGHT_REJECTED");
    expect(res.usedFallback).toBe(true);
    expect(res.value).toBe(5.0);
  });

  it("rejects non-finite values and confidence", () => {
    const pipeline = new AileeTrustPipeline({ defaultFallbackValue: 5.0 });

    const invalidValue = pipeline.process(Number.NaN, 1.0);
    expect(invalidValue.safetyStatus).toBe("OUTRIGHT_REJECTED");
    expect(invalidValue.value).toBe(5.0);

    const invalidConfidence = pipeline.process(10.0, Number.NaN);
    expect(invalidConfidence.trustScore.confidence).toBe(0);
    expect(invalidConfidence.safetyStatus).toBe("OUTRIGHT_REJECTED");
  });

  it("evaluates grace layer for borderline signals", () => {
    const pipeline = new AileeTrustPipeline({
      borderlineLow: 0.5,
      borderlineHigh: 0.8,
    });

    pipeline.process(10.0, 0.9);
    pipeline.process(11.0, 0.9);

    const res = pipeline.process(12.0, 0.65);
    expect(res.safetyStatus).toBe("ACCEPTED");
  });

  it("executes consensus strategy", () => {
    const pipeline = new AileeTrustPipeline(
      {
        borderlineLow: 0.5,
        borderlineHigh: 0.8,
      },
      ConsensusStrategy.WeightedCombination
    );

    const res = pipeline.process(10.0, 0.9, [12.0, 11.0]);
    expect(res.consensusAchieved).toBe(true);
    expect(res.value).toBeGreaterThan(10.0);
    expect(res.value).toBeLessThan(12.0);
  });
});
