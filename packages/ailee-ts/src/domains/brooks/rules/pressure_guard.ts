//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { BrooksSafetyPolicy, DEFAULT_BROOKS_POLICY } from "../types/policy.js";
import { RuleCheckResult } from "./flow_bounds.js";

export class PressureGuard {
  private policy: BrooksSafetyPolicy;

  constructor(policy: BrooksSafetyPolicy = DEFAULT_BROOKS_POLICY) {
    this.policy = policy;
  }

  /**
   * Checks current line pressure against max allowable containment pressure.
   */
  public evaluateContainment(currentPressurePsi: number, overpressureTrip = false, tighteningFactor = 1.0): RuleCheckResult {
    const factor = Number.isFinite(tighteningFactor) && tighteningFactor > 0 && tighteningFactor <= 1.0 ? tighteningFactor : 1.0;
    if (!Number.isFinite(currentPressurePsi) || !Number.isFinite(this.policy.pressure.maxOperatingPressurePsi) || this.policy.pressure.maxOperatingPressurePsi < 0) {
      return { passed: false, status: "OUTRIGHT_REJECTED", confidencePenalty: 1.0, reason: "Invalid pressure input or policy configuration", recommendedAction: "VALVE_CLOSE" };
    }
    const safeMax = this.policy.pressure.maxOperatingPressurePsi * factor;
    if (overpressureTrip || currentPressurePsi > safeMax) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 1.0,
        reason: `Overpressure hazard: measured pressure ${currentPressurePsi.toFixed(2)} PSI exceeds predictive operating pressure limit of ${safeMax.toFixed(2)} PSI (tightening factor: ${factor.toFixed(2)})`,
        recommendedAction: "VALVE_CLOSE",
      };
    }

    return {
      passed: true,
      status: "ACCEPTED",
      confidencePenalty: 0.0,
    };
  }
}

export class PressureDeltaGuard {
  private policy: BrooksSafetyPolicy;

  constructor(policy: BrooksSafetyPolicy = DEFAULT_BROOKS_POLICY) {
    this.policy = policy;
  }

  /**
   * Blocks valve opening commands if upstream/downstream pressure differential exceeds device operating limits.
   */
  public evaluateDeltaP(upstreamPressurePsi: number, downstreamPressurePsi: number, requestingValveOpen: boolean, tighteningFactor = 1.0): RuleCheckResult {
    const factor = Number.isFinite(tighteningFactor) && tighteningFactor > 0 && tighteningFactor <= 1.0 ? tighteningFactor : 1.0;
    if (![upstreamPressurePsi, downstreamPressurePsi, this.policy.pressure.maxDifferentialPressurePsi].every(Number.isFinite)
      || this.policy.pressure.maxDifferentialPressurePsi < 0) {
      return { passed: false, status: "OUTRIGHT_REJECTED", confidencePenalty: 1.0, reason: "Invalid differential-pressure input or policy configuration", recommendedAction: "VALVE_CLOSE" };
    }
    const deltaP = Math.abs(upstreamPressurePsi - downstreamPressurePsi);
    const safeMaxDelta = this.policy.pressure.maxDifferentialPressurePsi * factor;

    if (requestingValveOpen && deltaP > safeMaxDelta) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 0.9,
        reason: `Pressure differential hazard: delta-P ${deltaP.toFixed(2)} PSI exceeds predictive valve opening limit of ${safeMaxDelta.toFixed(2)} PSI (tightening factor: ${factor.toFixed(2)})`,
        recommendedAction: "VALVE_CLOSE",
      };
    }

    return {
      passed: true,
      status: "ACCEPTED",
      confidencePenalty: 0.0,
    };
  }
}
