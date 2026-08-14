//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { GraceEvaluationResult, TrustScore } from "./types.js";

export class GraceLayer {
  public evaluate(
    rawValue: number,
    trustScore: TrustScore,
    history: number[],
    context?: Record<string, unknown>
  ): GraceEvaluationResult {
    let trendPass = true;
    if (history.length >= 2) {
      const last = history[history.length - 1];
      const prev = history[history.length - 2];
      const expectedDelta = last - prev;
      const actualDelta = rawValue - last;

      if (expectedDelta * actualDelta < 0 && Math.abs(actualDelta) > Math.abs(expectedDelta) * 3 + 1.0) {
        trendPass = false;
      }
    }

    let contextPass = true;
    if (context && typeof context.maxAllowedChange === "number" && history.length > 0) {
      const last = history[history.length - 1];
      if (Math.abs(rawValue - last) > (context.maxAllowedChange as number)) {
        contextPass = false;
      }
    }

    if (trendPass && contextPass && trustScore.safety > 0.5) {
      return {
        passed: true,
        scoreAdjustment: 0.15,
        reason: "GRACE_PASS: Trend continuity and contextual checks validated borderline signal.",
      };
    }

    return {
      passed: false,
      scoreAdjustment: -0.1,
      reason: `GRACE_FAIL: Trend or context validation failed (trendPass: ${trendPass}, contextPass: ${contextPass}).`,
    };
  }
}
