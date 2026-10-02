//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { RollingWindowStats } from "./telemetry_window.js";

export type PredictiveState = "STABLE" | "DEGRADED_SOON" | "HAZARDOUS_SOON";

export interface PredictiveResult {
  state: PredictiveState;
  score: number; // 0.0 (perfectly stable) to 1.0 (imminent hazard)
  scoreByte: number; // 0-255 UINT8 representation for fieldbus serialization
  tighteningFactor: number; // Multiplier for physical guard limits (e.g. 1.0 = normal, 0.7 = tightened, 0.4 = strictly tightened)
  reasons: string[];
  isMonotonicViolation: boolean;
  isValid: boolean;
}

export class PredictiveStabilityLayer {
  private previousScore: number = 0.0;

  constructor() {}

  public reset(): void {
    this.previousScore = 0.0;
  }

  public evaluatePredictiveState(
    stats: RollingWindowStats,
    recentTrend: number[] = []
  ): PredictiveResult {
    const reasons: string[] = [];

    if (!stats.isValid || stats.count < 5) {
      return {
        state: "STABLE",
        score: 0.0,
        scoreByte: 0,
        tighteningFactor: 1.0,
        reasons: ["Insufficient telemetry samples for predictive stability scoring; defaulting to static STABLE"],
        isMonotonicViolation: false,
        isValid: false,
      };
    }

    // Trend calculation: rate of change across recent flow/pressure samples
    let trendDelta = 0.0;
    if (recentTrend.length >= 2) {
      const first = recentTrend[0];
      const last = recentTrend[recentTrend.length - 1];
      if (Number.isFinite(first) && Number.isFinite(last)) {
        trendDelta = Math.abs(last - first) / Math.max(1, recentTrend.length - 1);
      }
    }

    // Compute risk score components
    // 1. Variance contribution
    const flowVarRisk = Math.min(0.4, stats.stdDevFlow / 10.0);
    const pressureVarRisk = Math.min(0.4, stats.stdDevPressure / 5.0);

    // 2. Degradation frequency contribution
    const degRisk = Math.min(0.5, stats.degradationFrequency * 1.5);

    // 3. Zero drift contribution
    const zeroRisk = Math.min(0.3, Math.abs(stats.meanZeroOffset) / 1.0);

    // 4. Trend acceleration contribution
    const trendRisk = Math.min(0.3, trendDelta / 20.0);

    let rawScore = flowVarRisk + pressureVarRisk + degRisk + zeroRisk + trendRisk;
    if (!Number.isFinite(rawScore) || rawScore < 0) {
      return {
        state: "STABLE",
        score: 0.0,
        scoreByte: 0,
        tighteningFactor: 1.0,
        reasons: ["Non-finite or invalid raw predictive score computed; resetting to STABLE"],
        isMonotonicViolation: true,
        isValid: false,
      };
    }

    rawScore = Math.min(1.0, Math.max(0.0, rawScore));

    // Monotonicity / range sanity validation
    // If raw score drops erratically by > 0.8 in a single evaluation without gradual recovery, flag monotonic violation
    let isMonotonicViolation = false;
    if (this.previousScore > 0.8 && rawScore < 0.1) {
      isMonotonicViolation = true;
      reasons.push(`Monotonicity violation detected: score dropped abruptly from ${this.previousScore.toFixed(2)} to ${rawScore.toFixed(2)}`);
    }

    this.previousScore = rawScore;

    // Classify predictive state and set tightening factor
    let state: PredictiveState = "STABLE";
    let tighteningFactor = 1.0;

    if (rawScore >= 0.70) {
      state = "HAZARDOUS_SOON";
      tighteningFactor = 0.4;
      reasons.push(`HAZARDOUS_SOON state predicted (score: ${rawScore.toFixed(2)}); pre-emptively tightening physical guard thresholds by 60%`);
    } else if (rawScore >= 0.35) {
      state = "DEGRADED_SOON";
      tighteningFactor = 0.7;
      reasons.push(`DEGRADED_SOON state predicted (score: ${rawScore.toFixed(2)}); pre-emptively tightening physical guard thresholds by 30%`);
    } else {
      state = "STABLE";
      tighteningFactor = 1.0;
    }

    const scoreByte = Math.min(255, Math.max(0, Math.round(rawScore * 255)));

    return {
      state,
      score: rawScore,
      scoreByte,
      tighteningFactor,
      reasons,
      isMonotonicViolation,
      isValid: !isMonotonicViolation,
    };
  }
}
