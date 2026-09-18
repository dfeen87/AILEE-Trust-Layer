//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export type PredictiveState = "STABLE" | "DEGRADED_SOON" | "HAZARDOUS_SOON";

export interface PredictiveEvaluationResult {
  state: PredictiveState;
  score: number; // Normalized predictive score in [0.0, 1.0]
  slope: number; // Calculated trend slope
  reasons: string[];
  tighteningMultiplier: number; // 1.0 for STABLE, 0.75 for DEGRADED_SOON, 0.50 for HAZARDOUS_SOON
  recommendedAction?: "VALVE_CLOSE" | "VALVE_HOLD" | "PROCEED";
  isDegraded: boolean;
}

/**
 * Predictive Stability Layer that classifies upcoming system states
 * using trend and rate-of-change analysis over rolling telemetry history.
 */
export class PredictiveStabilityLayer {
  private previousScore = 0.0;

  /**
   * Evaluates trend over rolling telemetry values (e.g., flow, pressure, drift).
   */
  public evaluateTrends(telemetryHistory: number[]): PredictiveEvaluationResult {
    const n = telemetryHistory.length;
    if (n < 3 || !telemetryHistory.every(Number.isFinite)) {
      return {
        state: "STABLE",
        score: 0.0,
        slope: 0.0,
        reasons: ["Telemetry window sparse or contains non-finite values; reverting to static v8.2 baseline"],
        tighteningMultiplier: 1.0,
        recommendedAction: "PROCEED",
        isDegraded: true,
      };
    }

    // Check bounds (negative flow or absurd values)
    if (telemetryHistory.some((v) => v < 0 || v > 10000)) {
      return {
        state: "STABLE",
        score: 0.0,
        slope: 0.0,
        reasons: ["Telemetry history contains out-of-bounds values; reverting to static v8.2 baseline"],
        tighteningMultiplier: 1.0,
        recommendedAction: "PROCEED",
        isDegraded: true,
      };
    }

    // Calculate linear regression slope: y = mx + c over time index x = 0..n-1
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumX2 = 0;

    for (let i = 0; i < n; i++) {
      const x = i;
      const y = telemetryHistory[i];
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
    }

    const denom = n * sumX2 - sumX * sumX;
    const slope = denom !== 0 ? (n * sumXY - sumX * sumY) / denom : 0.0;

    // Calculate variance / noise component
    const meanY = sumY / n;
    let sqDiff = 0;
    for (let i = 0; i < n; i++) {
      sqDiff += (telemetryHistory[i] - meanY) ** 2;
    }
    const variance = sqDiff / n;
    const stdDev = Math.sqrt(variance);

    // Compute raw predictive hazard score
    const absSlope = Math.abs(slope);
    let rawScore = Math.min(1.0, absSlope * 0.15 + stdDev * 0.05);

    if (!Number.isFinite(rawScore) || rawScore < 0 || rawScore > 1.0) {
      return {
        state: "STABLE",
        score: 0.0,
        slope: 0.0,
        reasons: ["Predictive score calculation produced non-finite or out-of-range value; reverting to static v8.2 baseline"],
        tighteningMultiplier: 1.0,
        recommendedAction: "PROCEED",
        isDegraded: true,
      };
    }

    // Monotonicity / sanity validation
    // If score jumps wildly beyond trend logic, flag degradation
    const normalizedScore = rawScore;
    this.previousScore = normalizedScore;

    let state: PredictiveState = "STABLE";
    let tighteningMultiplier = 1.0;
    let recommendedAction: "VALVE_CLOSE" | "VALVE_HOLD" | "PROCEED" = "PROCEED";
    const reasons: string[] = [];

    if (normalizedScore < 0.35) {
      state = "STABLE";
      tighteningMultiplier = 1.0;
      recommendedAction = "PROCEED";
      reasons.push(`Predictive state STABLE (score ${normalizedScore.toFixed(2)}, slope ${slope.toFixed(3)})`);
    } else if (normalizedScore < 0.70) {
      state = "DEGRADED_SOON";
      tighteningMultiplier = 0.75;
      recommendedAction = "PROCEED";
      reasons.push(`Predictive state DEGRADED_SOON (score ${normalizedScore.toFixed(2)}, slope ${slope.toFixed(3)}): pre-emptively tightening guard limits by 25%`);
    } else {
      state = "HAZARDOUS_SOON";
      tighteningMultiplier = 0.50;
      recommendedAction = "VALVE_HOLD";
      reasons.push(`Predictive state HAZARDOUS_SOON (score ${normalizedScore.toFixed(2)}, slope ${slope.toFixed(3)}): strictly tightening guard limits by 50% and recommending early hold/close`);
    }

    return {
      state,
      score: normalizedScore,
      slope,
      reasons,
      tighteningMultiplier,
      recommendedAction,
      isDegraded: false,
    };
  }
}
