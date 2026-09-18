//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { BrooksSafetyPolicy, DEFAULT_BROOKS_POLICY } from "../types/policy.js";

export interface RuleCheckResult {
  passed: boolean;
  status: "ACCEPTED" | "BORDERLINE" | "OUTRIGHT_REJECTED";
  confidencePenalty: number;
  reason?: string;
  recommendedAction?: "VALVE_CLOSE" | "VALVE_HOLD" | "PROCEED";
}

export class RampRateGuard {
  private policy: BrooksSafetyPolicy;

  constructor(policy: BrooksSafetyPolicy = DEFAULT_BROOKS_POLICY) {
    this.policy = policy;
  }

  /**
   * Evaluates setpoint change jump percentage relative to Full Scale flow over duration.
   * e.g., max 20% SLPM jump per 100ms.
   */
  public evaluate(currentSetpoint: number, newSetpoint: number, fullScaleFlow: number, durationMs = 100): RuleCheckResult {
    if (![currentSetpoint, newSetpoint, fullScaleFlow, durationMs, this.policy.rampRate.maxPercentJumpPer100ms, this.policy.rampRate.timeWindowMs].every(Number.isFinite)
      || fullScaleFlow <= 0 || durationMs <= 0 || this.policy.rampRate.maxPercentJumpPer100ms < 0 || this.policy.rampRate.timeWindowMs <= 0) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 1.0,
        reason: "Invalid ramp-rate input or policy configuration",
        recommendedAction: "VALVE_HOLD",
      };
    }

    const deltaPercentFS = (Math.abs(newSetpoint - currentSetpoint) / fullScaleFlow) * 100.0;
    const timeScale = durationMs > 0 ? durationMs / this.policy.rampRate.timeWindowMs : 1.0;
    const normalizedMaxJump = this.policy.rampRate.maxPercentJumpPer100ms * timeScale;

    if (deltaPercentFS > normalizedMaxJump) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 0.8,
        reason: `Ramp rate breach: flow change of ${deltaPercentFS.toFixed(2)}% FS exceeds limit of ${normalizedMaxJump.toFixed(2)}% FS per ${durationMs}ms`,
        recommendedAction: "VALVE_HOLD",
      };
    }

    return {
      passed: true,
      status: "ACCEPTED",
      confidencePenalty: 0.0,
    };
  }
}

export class ZeroDriftGuard {
  private policy: BrooksSafetyPolicy;

  constructor(policy: BrooksSafetyPolicy = DEFAULT_BROOKS_POLICY) {
    this.policy = policy;
  }

  /**
   * Analyzes zero-flow baseline telemetry.
   * If zero-drift exceeds ±0.5% of Full Scale when setpoint is 0, issue POLICY_DEGRADED warning.
   */
  public evaluate(setpoint: number, zeroOffsetPercentFS: number): RuleCheckResult {
    if (![setpoint, zeroOffsetPercentFS, this.policy.zeroDrift.maxDriftPercentFS].every(Number.isFinite) || this.policy.zeroDrift.maxDriftPercentFS < 0) {
      return { passed: false, status: "OUTRIGHT_REJECTED", confidencePenalty: 1.0, reason: "Invalid zero-drift input or policy configuration", recommendedAction: "VALVE_HOLD" };
    }
    if (this.policy.zeroDrift.requireWarningOnExceed && setpoint === 0 && Math.abs(zeroOffsetPercentFS) > this.policy.zeroDrift.maxDriftPercentFS) {
      return {
        passed: false,
        status: "BORDERLINE",
        confidencePenalty: 0.35,
        reason: `Zero-drift warning: zero offset ${zeroOffsetPercentFS.toFixed(2)}% FS exceeds safe limit of ±${this.policy.zeroDrift.maxDriftPercentFS}% FS`,
        recommendedAction: "PROCEED",
      };
    }

    return {
      passed: true,
      status: "ACCEPTED",
      confidencePenalty: 0.0,
    };
  }
}
