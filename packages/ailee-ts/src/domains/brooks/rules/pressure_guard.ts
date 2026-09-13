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
  public evaluateContainment(currentPressurePsi: number, overpressureTrip = false): RuleCheckResult {
    if (overpressureTrip || currentPressurePsi > this.policy.pressure.maxOperatingPressurePsi) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 1.0,
        reason: `Overpressure hazard: measured pressure ${currentPressurePsi.toFixed(2)} PSI exceeds maximum safe operating pressure limit of ${this.policy.pressure.maxOperatingPressurePsi} PSI`,
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
  public evaluateDeltaP(upstreamPressurePsi: number, downstreamPressurePsi: number, requestingValveOpen: boolean): RuleCheckResult {
    const deltaP = Math.abs(upstreamPressurePsi - downstreamPressurePsi);

    if (requestingValveOpen && deltaP > this.policy.pressure.maxDifferentialPressurePsi) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 0.9,
        reason: `Pressure differential hazard: delta-P ${deltaP.toFixed(2)} PSI exceeds valve opening limit of ${this.policy.pressure.maxDifferentialPressurePsi} PSI`,
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
