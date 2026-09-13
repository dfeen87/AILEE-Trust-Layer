//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { AileeConfig, DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";
import { EtherCATAdapter } from "./adapters/ethercat.js";
import { EtherNetIPAdapter } from "./adapters/ethernet_ip.js";
import { DeviceStateMachine } from "./models/device_state.js";
import { GAS_DATABASE, isHazardousGas, lookupGas } from "./models/gas_database.js";
import { RampRateGuard, ZeroDriftGuard } from "./rules/flow_bounds.js";
import { GasSafetyGuard } from "./rules/gas_compatibility.js";
import { PressureDeltaGuard, PressureGuard } from "./rules/pressure_guard.js";
import { ActuationCommand, validateActuationCommand } from "./types/commands.js";
import { BrooksSafetyPolicy, DEFAULT_BROOKS_POLICY } from "./types/policy.js";
import {
  MFCDeviceTelemetry,
  PressureControllerTelemetry,
  UltrasonicTelemetry,
  validateMFCTelemetry,
  validatePressureControllerTelemetry,
  validateUltrasonicTelemetry,
} from "./types/telemetry.js";

export * from "./types/telemetry.js";
export * from "./types/commands.js";
export * from "./types/policy.js";
export * from "./models/gas_database.js";
export * from "./models/device_state.js";
export * from "./rules/flow_bounds.js";
export * from "./rules/pressure_guard.js";
export * from "./rules/gas_compatibility.js";
export * from "./adapters/ethernet_ip.js";
export * from "./adapters/ethercat.js";

export const BROOKS_PRESETS: Record<string, AileeConfig> = {
  STRICT_PHYSICAL: {
    borderlineLow: 0.8,
    borderlineHigh: 0.95,
    hardMin: 0.0,
    hardMax: 1000.0,
    decayRate: 0.05,
    historyCapacity: 100,
    consensusThreshold: 0.8,
    defaultFallbackValue: 0.0,
  },
};

export class BrooksHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "brooks";
  private pipeline: AileeTrustPipeline;
  private policy: BrooksSafetyPolicy;
  public stateMachine: DeviceStateMachine;

  private rampGuard: RampRateGuard;
  private zeroDriftGuard: ZeroDriftGuard;
  private pressureGuard: PressureGuard;
  private pressureDeltaGuard: PressureDeltaGuard;
  private gasSafetyGuard: GasSafetyGuard;

  constructor(
    deviceId = "mfc_brooks_sla5800",
    policy: BrooksSafetyPolicy = DEFAULT_BROOKS_POLICY,
    aileeConfig: AileeConfig = BROOKS_PRESETS.STRICT_PHYSICAL
  ) {
    this.pipeline = new AileeTrustPipeline(aileeConfig);
    this.policy = policy;
    this.stateMachine = new DeviceStateMachine(deviceId, 100.0, 1); // Default N2

    this.rampGuard = new RampRateGuard(this.policy);
    this.zeroDriftGuard = new ZeroDriftGuard(this.policy);
    this.pressureGuard = new PressureGuard(this.policy);
    this.pressureDeltaGuard = new PressureDeltaGuard(this.policy);
    this.gasSafetyGuard = new GasSafetyGuard(this.policy);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: this.stateMachine.deviceId,
      quality: 0.99,
      readings: {
        flowRate: this.stateMachine.currentFlowRate,
        setpoint: this.stateMachine.currentSetpoint,
        valvePosition: 25.0,
        temperature: 23.5,
        gasId: this.stateMachine.activeGasId,
        zeroOffset: this.stateMachine.zeroOffsetPercentFS,
        pressure: this.stateMachine.currentPressurePsi,
        overpressureTrip: this.stateMachine.mode === "FAULT",
      },
    };
  }

  /**
   * Deterministic synchronous evaluation (< 2ms execution budget).
   */
  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const flowRate = Number(snapshot.readings.flowRate ?? 0.0);
    const targetSetpoint = Number(snapshot.readings.setpoint ?? flowRate);
    const gasId = (snapshot.readings.gasId as string | number) ?? this.stateMachine.activeGasId;
    const zeroOffset = Number(snapshot.readings.zeroOffset ?? 0.0);
    const pressure = Number(snapshot.readings.pressure ?? 0.0);
    const overpressureTrip = Boolean(snapshot.readings.overpressureTrip ?? false);
    const upstreamP = Number(snapshot.readings.upstreamPressure ?? pressure);
    const downstreamP = Number(snapshot.readings.downstreamPressure ?? 0.0);
    const isPurging = Boolean(snapshot.readings.isPurging ?? false);

    let confidencePenalty = 0.0;
    const reasons: string[] = [];

    // 1. Pressure Containment Check
    const pressureRes = this.pressureGuard.evaluateContainment(pressure, overpressureTrip);
    if (!pressureRes.passed) {
      confidencePenalty += pressureRes.confidencePenalty;
      reasons.push(pressureRes.reason || "Pressure containment check failed");
    }

    // 2. Pressure Delta Check (if valve open requested)
    const deltaRes = this.pressureDeltaGuard.evaluateDeltaP(upstreamP, downstreamP, targetSetpoint > 0);
    if (!deltaRes.passed) {
      confidencePenalty += deltaRes.confidencePenalty;
      reasons.push(deltaRes.reason || "Delta pressure limit exceeded");
    }

    // 3. Ramp Rate Ramp Check
    const rampRes = this.rampGuard.evaluate(this.stateMachine.currentSetpoint, targetSetpoint, this.stateMachine.fullScaleFlowSlpm);
    if (!rampRes.passed) {
      confidencePenalty += rampRes.confidencePenalty;
      reasons.push(rampRes.reason || "Ramp rate limit breached");
    }

    // 4. Gas Safety Check
    const gasRes = this.gasSafetyGuard.evaluateFlowLimit(gasId, targetSetpoint);
    if (!gasRes.passed) {
      confidencePenalty += gasRes.confidencePenalty;
      reasons.push(gasRes.reason || "Gas flow safety threshold breached");
    }

    // 5. Gas Change Check
    if (gasId !== this.stateMachine.activeGasId) {
      const gasChangeRes = this.gasSafetyGuard.evaluateGasChange(this.stateMachine.activeGasId, gasId, flowRate, isPurging);
      if (!gasChangeRes.passed) {
        confidencePenalty += gasChangeRes.confidencePenalty;
        reasons.push(gasChangeRes.reason || "Gas change rule violation");
      }
    }

    // 6. Zero Drift Warning
    const zeroRes = this.zeroDriftGuard.evaluate(targetSetpoint, zeroOffset);
    if (!zeroRes.passed) {
      confidencePenalty += zeroRes.confidencePenalty;
      reasons.push(zeroRes.reason || "Zero drift threshold warning");
    }

    const calculatedConfidence = Math.max(0.0, 1.0 - confidencePenalty);
    const decision = await this.pipeline.process(targetSetpoint, calculatedConfidence, [], trustContext);

    // Append custom rule reasons if any
    if (reasons.length > 0) {
      decision.reasons = [...decision.reasons, ...reasons];
    }

    // Handle fallback decision logic
    if (decision.safetyStatus === "OUTRIGHT_REJECTED" || decision.usedFallback) {
      await this.triggerFallback(decision);
    }

    return decision;
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    const activeGas = this.stateMachine.getActiveGas();
    const isHazardous = activeGas ? isHazardousGas(activeGas) : true;

    // Hazardous lines -> Immediate VALVE_CLOSE; Inert lines -> VALVE_HOLD
    const fallbackAction = isHazardous ? "VALVE_CLOSE" : "VALVE_HOLD";

    await this.writeActuators({
      actuatorId: `${this.stateMachine.deviceId}_actuator`,
      command: { action: fallbackAction, safeTargetFlow: 0.0, decisionId: decision.decisionId },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {
    if (typeof command.command === "object" && command.command !== null) {
      const cmdObj = command.command as Record<string, unknown>;
      if (cmdObj.action === "VALVE_CLOSE") {
        this.stateMachine.setMode("FAULT");
        this.stateMachine.currentSetpoint = 0.0;
      } else if (cmdObj.action === "VALVE_HOLD") {
        this.stateMachine.setMode("HOLD");
      }
    }
  }
}

export { BrooksHardwareAdapter as BrooksDomain };
