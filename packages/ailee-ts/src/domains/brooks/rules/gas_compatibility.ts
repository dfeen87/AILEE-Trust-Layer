//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { isHazardousGas, lookupGas } from "../models/gas_database.js";
import { BrooksSafetyPolicy, DEFAULT_BROOKS_POLICY } from "../types/policy.js";
import { RuleCheckResult } from "./flow_bounds.js";

export class GasSafetyGuard {
  private policy: BrooksSafetyPolicy;

  constructor(policy: BrooksSafetyPolicy = DEFAULT_BROOKS_POLICY) {
    this.policy = policy;
  }

  /**
   * Verifies setpoint for reactive or toxic gases does not exceed maximum allowable flow limits for line manifold.
   */
  public evaluateFlowLimit(gasId: string | number, targetSetpointSlpm: number, tighteningFactor = 1.0): RuleCheckResult {
    const factor = Number.isFinite(tighteningFactor) && tighteningFactor > 0 && tighteningFactor <= 1.0 ? tighteningFactor : 1.0;
    const gas = lookupGas(gasId);
    if (!gas) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 1.0,
        reason: `Unknown or unregistered gas ID: ${gasId}`,
        recommendedAction: "VALVE_CLOSE",
      };
    }

    if (!Number.isFinite(targetSetpointSlpm) || targetSetpointSlpm < 0 || !Number.isFinite(gas.defaultMaxFlowSlpm) || gas.defaultMaxFlowSlpm < 0) {
      return { passed: false, status: "OUTRIGHT_REJECTED", confidencePenalty: 1.0, reason: "Invalid gas flow setpoint or gas flow limit", recommendedAction: "VALVE_CLOSE" };
    }

    const safeMaxFlow = gas.defaultMaxFlowSlpm * factor;

    if (targetSetpointSlpm > safeMaxFlow) {
      const isHaz = isHazardousGas(gas);
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: isHaz ? 1.0 : 0.7,
        reason: `Gas flow limit breach: setpoint ${targetSetpointSlpm} SLPM exceeds predictive manifold limit of ${safeMaxFlow.toFixed(2)} SLPM for ${gas.name} (${gas.formula}) (tightening factor: ${factor.toFixed(2)})`,
        recommendedAction: isHaz ? "VALVE_CLOSE" : "VALVE_HOLD",
      };
    }

    return {
      passed: true,
      status: "ACCEPTED",
      confidencePenalty: 0.0,
    };
  }

  /**
   * Validates gas changes on active gas lines to prevent cross-contamination or chemical reaction hazards.
   */
  public evaluateGasChange(
    currentGasId: string | number,
    requestedGasId: string | number,
    currentFlowRate: number,
    isPurgeActive: boolean
  ): RuleCheckResult {
    const currentGas = lookupGas(currentGasId);
    const requestedGas = lookupGas(requestedGasId);
    const currentCanonicalGasId = currentGas?.gasId;
    const requestedCanonicalGasId = requestedGas?.gasId;

    if (currentCanonicalGasId !== undefined && requestedCanonicalGasId !== undefined && currentCanonicalGasId === requestedCanonicalGasId) {
      return { passed: true, status: "ACCEPTED", confidencePenalty: 0.0 };
    }

    if (!requestedGas) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 1.0,
        reason: `Unknown requested gas ID: ${requestedGasId}`,
        recommendedAction: "VALVE_CLOSE",
      };
    }

    if (!currentGas) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 1.0,
        reason: `Unknown current gas ID: ${currentGasId}`,
        recommendedAction: "VALVE_CLOSE",
      };
    }

    if (!Number.isFinite(currentFlowRate) || currentFlowRate < 0) {
      return { passed: false, status: "OUTRIGHT_REJECTED", confidencePenalty: 1.0, reason: "Invalid current flow rate during gas transition", recommendedAction: "VALVE_CLOSE" };
    }

    // Changing gas while active flow is present is unsafe
    if (currentFlowRate > 0.1) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 1.0,
        reason: `Unsafe gas change: active flow detected (${currentFlowRate.toFixed(2)} SLPM) during gas line transition request`,
        recommendedAction: "VALVE_CLOSE",
      };
    }

    // Hazardous gas change without line purge check
    const involvesHazard = (currentGas && isHazardousGas(currentGas)) || isHazardousGas(requestedGas);
    if (involvesHazard && this.policy.enforceGasLinePurge && !isPurgeActive) {
      return {
        passed: false,
        status: "OUTRIGHT_REJECTED",
        confidencePenalty: 1.0,
        reason: `Cross-contamination / reaction hazard: transitioning to/from hazardous gas ${requestedGas.name} requires prior line purge (PURGE_LINE)`,
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
