//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { CalibrationLayer, SelfTuningCalibrationModule } from "../../core/calibration.js";
import { AileeConfig, CalibrationConfig, CalibrationMetadata, DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";
import { EtherCATAdapter } from "./adapters/ethercat.js";
import { EtherNetIPAdapter } from "./adapters/ethernet_ip.js";
import { DeviceStateMachine } from "./models/device_state.js";
import { GAS_DATABASE, isHazardousGas, lookupGas } from "./models/gas_database.js";
import { RampRateGuard, ZeroDriftGuard } from "./rules/flow_bounds.js";
import { GasSafetyGuard } from "./rules/gas_compatibility.js";
import { PressureDeltaGuard, PressureGuard } from "./rules/pressure_guard.js";
import { PredictiveStabilityLayer } from "./rules/predictive_stability.js";
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
export * from "./rules/predictive_stability.js";
export * from "./adapters/ethernet_ip.js";
export * from "./adapters/ethercat.js";

export const BROOKS_VERSION = "8.3.0";

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
  private calibrationLayer: CalibrationLayer;
  private selfTuningModule: SelfTuningCalibrationModule;
  private predictiveLayer: PredictiveStabilityLayer;
  private telemetryHistory: number[] = [];
  private policy: BrooksSafetyPolicy;
  public stateMachine: DeviceStateMachine;
  public heartbeatTimeoutMs: number = 1000; // 1000ms maximum telemetry staleness window

  private rampGuard: RampRateGuard;
  private zeroDriftGuard: ZeroDriftGuard;
  private pressureGuard: PressureGuard;
  private pressureDeltaGuard: PressureDeltaGuard;
  private gasSafetyGuard: GasSafetyGuard;

  constructor(
    deviceId = "mfc_brooks_sla5800",
    policy: BrooksSafetyPolicy = DEFAULT_BROOKS_POLICY,
    aileeConfig: AileeConfig = BROOKS_PRESETS.STRICT_PHYSICAL,
    heartbeatTimeoutMs = 1000,
    calibrationConfig: Partial<CalibrationConfig> = {}
  ) {
    this.pipeline = new AileeTrustPipeline(aileeConfig);
    this.calibrationLayer = new CalibrationLayer({ acceptanceThreshold: aileeConfig.borderlineHigh, ...calibrationConfig });
    this.selfTuningModule = new SelfTuningCalibrationModule({
      enabled: true,
      acceptanceThreshold: aileeConfig.borderlineHigh,
      ...calibrationConfig,
    });
    this.predictiveLayer = new PredictiveStabilityLayer();
    this.policy = policy;
    this.heartbeatTimeoutMs = heartbeatTimeoutMs;
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
        telemetryTimestamp: this.stateMachine.lastTelemetryTimestamp,
        previousTelemetryTimestamp: this.stateMachine.previousTelemetryTimestamp,
        previousSetpoint: this.stateMachine.previousSetpoint,
      },
    };
  }

  /**
   * Deterministic synchronous evaluation (< 2ms execution budget).
   */
  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const now = Number.isFinite(snapshot.timestamp) ? snapshot.timestamp : Date.now();
    const telemetryTs = Number(snapshot.readings.telemetryTimestamp ?? snapshot.timestamp);
    const flowRate = Number(snapshot.readings.flowRate);
    const targetSetpoint = Number(snapshot.readings.setpoint ?? flowRate);
    const gasId = (snapshot.readings.gasId as string | number) ?? this.stateMachine.activeGasId;
    const currentSetpoint = Number(snapshot.readings.previousSetpoint ?? this.stateMachine.previousSetpoint);
    const zeroOffset = Number(snapshot.readings.zeroOffset);
    const pressure = Number(snapshot.readings.pressure);
    const overpressureTrip = Boolean(snapshot.readings.overpressureTrip ?? false);
    const upstreamP = Number(snapshot.readings.upstreamPressure ?? pressure);
    const downstreamP = Number(snapshot.readings.downstreamPressure ?? 0.0);
    const isPurging = snapshot.readings.isPurging === true;
    const previousTelemetryTs = Number(snapshot.readings.previousTelemetryTimestamp ?? this.stateMachine.previousTelemetryTimestamp);
    const sampleIntervalMs = Number.isFinite(previousTelemetryTs) ? Math.max(1, now - previousTelemetryTs) : Number.NaN;

    let confidencePenalty = 0.0;
    const reasons: string[] = [];

    // Push flow rate to telemetry history window for predictive & self-tuning analysis
    if (Number.isFinite(flowRate) && flowRate >= 0) {
      this.telemetryHistory.push(flowRate);
      if (this.telemetryHistory.length > 20) {
        this.telemetryHistory.shift();
      }
      this.selfTuningModule.recordTelemetry({ value: flowRate, isDegraded: overpressureTrip, timestamp: now });
    }

    // Evaluate Predictive Stability Layer
    const predEval = this.predictiveLayer.evaluateTrends(this.telemetryHistory);

    // Hardening Pass: Validate adaptive inputs and predictive outputs
    let isAdaptiveDegraded = predEval.isDegraded;
    let tighteningMultiplier = 1.0;
    let serializedPredictiveScore = 0;

    if (
      isAdaptiveDegraded ||
      this.telemetryHistory.length < 5 ||
      !this.telemetryHistory.every(Number.isFinite) ||
      !Number.isFinite(predEval.score) ||
      predEval.score < 0 ||
      predEval.score > 1.0
    ) {
      isAdaptiveDegraded = true;
      tighteningMultiplier = 1.0; // Revert to static v8.2 baseline configuration
      serializedPredictiveScore = 0; // Block adaptive changes from fieldbus serialization
      reasons.push("SELF_TUNING_DEGRADED: Telemetry sparse, non-finite, or out-of-bounds; reverting to static v8.2 baseline behavior.");
    } else {
      tighteningMultiplier = predEval.tighteningMultiplier;
      serializedPredictiveScore = Math.round(predEval.score * 255);
      reasons.push(...predEval.reasons);

      if (predEval.state === "HAZARDOUS_SOON") {
        confidencePenalty += 0.8;
      } else if (predEval.state === "DEGRADED_SOON") {
        confidencePenalty += 0.15;
      }
    }

    // 0. Heartbeat Timeout / Telemetry Drop Check
    if (!Number.isFinite(telemetryTs) || telemetryTs > now || !Number.isFinite(this.heartbeatTimeoutMs) || this.heartbeatTimeoutMs < 0) {
      confidencePenalty = 1.0;
      reasons.push("Invalid telemetry timestamp or heartbeat timeout configuration");
    } else if (now - telemetryTs > this.heartbeatTimeoutMs) {
      confidencePenalty = 1.0;
      reasons.push(`Heartbeat timeout: telemetry stale by ${(now - telemetryTs).toFixed(0)}ms (limit ${this.heartbeatTimeoutMs}ms)`);
    }

    if (![flowRate, targetSetpoint, currentSetpoint, zeroOffset, sampleIntervalMs].every(Number.isFinite) || flowRate < 0 || targetSetpoint < 0 || currentSetpoint < 0) {
      confidencePenalty = 1.0;
      reasons.push("Invalid flow, setpoint, zero-offset, or telemetry interval");
    }

    // 1. Pressure Containment Check (with predictive tightening)
    if (!Number.isFinite(pressure)) {
      confidencePenalty = 1.0;
      reasons.push("Invalid pressure telemetry: non-finite pressure value");
    } else {
      const pressureRes = this.pressureGuard.evaluateContainment(pressure, overpressureTrip, tighteningMultiplier);
      if (!pressureRes.passed) {
        confidencePenalty += pressureRes.confidencePenalty;
        reasons.push(pressureRes.reason || "Pressure containment check failed");
      }
    }

    // 2. Pressure Delta Check (if valve open requested)
    if (!Number.isFinite(upstreamP) || !Number.isFinite(downstreamP)) {
      confidencePenalty = 1.0;
      reasons.push("Invalid pressure telemetry: non-finite upstream/downstream pressure value");
    } else {
      const deltaRes = this.pressureDeltaGuard.evaluateDeltaP(upstreamP, downstreamP, targetSetpoint > 0, tighteningMultiplier);
      if (!deltaRes.passed) {
        confidencePenalty += deltaRes.confidencePenalty;
        reasons.push(deltaRes.reason || "Delta pressure limit exceeded");
      }
    }

    // 3. Ramp Rate Check (with predictive tightening)
    const rampRes = this.rampGuard.evaluate(currentSetpoint, targetSetpoint, this.stateMachine.fullScaleFlowSlpm, sampleIntervalMs, tighteningMultiplier);
    if (!rampRes.passed) {
      confidencePenalty += rampRes.confidencePenalty;
      reasons.push(rampRes.reason || "Ramp rate limit breached");
    }

    // 4. Gas Safety Check (with predictive tightening)
    const gasRes = this.gasSafetyGuard.evaluateFlowLimit(gasId, targetSetpoint, tighteningMultiplier);
    if (!gasRes.passed) {
      confidencePenalty += gasRes.confidencePenalty;
      reasons.push(gasRes.reason || "Gas flow safety threshold breached");
    }

    // 5. Gas Change Check
    const currentGasCanonical = lookupGas(this.stateMachine.activeGasId)?.gasId ?? this.stateMachine.activeGasId;
    const requestedGasCanonical = lookupGas(gasId)?.gasId ?? gasId;
    if (currentGasCanonical !== requestedGasCanonical) {
      const gasChangeRes = this.gasSafetyGuard.evaluateGasChange(currentGasCanonical, requestedGasCanonical, flowRate, isPurging);
      if (!gasChangeRes.passed) {
        confidencePenalty += gasChangeRes.confidencePenalty;
        reasons.push(gasChangeRes.reason || "Gas change rule violation");
      }
    }

    // 6. Zero Drift Warning (with predictive tightening)
    const zeroRes = this.zeroDriftGuard.evaluate(targetSetpoint, zeroOffset, tighteningMultiplier);
    if (!zeroRes.passed) {
      confidencePenalty += zeroRes.confidencePenalty;
      reasons.push(zeroRes.reason || "Zero drift threshold warning");
    }

    const calculatedConfidence = Number.isFinite(snapshot.quality) ? Math.max(0.0, Math.min(1.0, snapshot.quality, 1.0 - confidencePenalty)) : 0.0;
    const calibrationMetadata = trustContext?.calibration as CalibrationMetadata | undefined;
    const calibration = this.calibrationLayer.calibrate(calculatedConfidence, calibrationMetadata);
    // Never pass an invalid process target into the generic pipeline: use its
    // configured safe fallback path after the fail-closed confidence penalty.
    const safeTargetSetpoint = Number.isFinite(targetSetpoint) && targetSetpoint >= 0 ? targetSetpoint : 0.0;
    const decision = await this.pipeline.process(safeTargetSetpoint, calibration.confidence, [], trustContext);
    if (calibration.event !== "DISABLED") {
      decision.reasons = [...decision.reasons, ...calibration.reasons];
      decision.context = {
        ...trustContext,
        calibration: { ...calibration },
        predictive: {
          state: predEval.state,
          score: predEval.score,
          serializedPredictiveScore,
          tighteningMultiplier,
          selfTuningDegraded: isAdaptiveDegraded,
        },
      };
    } else {
      decision.context = {
        ...trustContext,
        predictive: {
          state: predEval.state,
          score: predEval.score,
          serializedPredictiveScore,
          tighteningMultiplier,
          selfTuningDegraded: isAdaptiveDegraded,
        },
      };
    }

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
const fallbackAction = isHazardous && this.policy.strictHazardousMode ? "VALVE_CLOSE" : "VALVE_HOLD";

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
