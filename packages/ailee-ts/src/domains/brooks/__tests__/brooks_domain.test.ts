//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { describe, expect, it } from "vitest";
import { EtherCATAdapter } from "../adapters/ethercat.js";
import { EtherNetIPAdapter } from "../adapters/ethernet_ip.js";
import manifestJson from "../configs/sla5800_manifest.json" assert { type: "json" };
import { BrooksDomain, BrooksHardwareAdapter } from "../index.js";
import { DeviceStateMachine } from "../models/device_state.js";
import { GAS_DATABASE, isHazardousGas, lookupGas } from "../models/gas_database.js";
import { RampRateGuard, ZeroDriftGuard } from "../rules/flow_bounds.js";
import { GasSafetyGuard } from "../rules/gas_compatibility.js";
import { PressureDeltaGuard, PressureGuard } from "../rules/pressure_guard.js";
import { ActuationCommand, validateActuationCommand } from "../types/commands.js";
import { DEFAULT_BROOKS_POLICY } from "../types/policy.js";
import {
  MFCDeviceTelemetry,
  PressureControllerTelemetry,
  UltrasonicTelemetry,
  validateMFCTelemetry,
  validatePressureControllerTelemetry,
  validateUltrasonicTelemetry,
} from "../types/telemetry.js";

describe("Brooks Instrument Domain Unit Tests", () => {
  describe("Telemetry & Command Validation Schemas", () => {
    it("validates MFCDeviceTelemetry correctly", () => {
      const validMFC: MFCDeviceTelemetry = {
        flowRate: 50.0,
        setpoint: 50.0,
        valvePosition: 45.0,
        temperature: 22.5,
        gasId: 1,
        zeroOffset: 0.1,
        deviceStatus: "OK",
      };
      expect(validateMFCTelemetry(validMFC).valid).toBe(true);

      const invalidMFC = {
        flowRate: "invalid",
        setpoint: 50.0,
        valvePosition: 150.0, // > 100
        temperature: 22.5,
        gasId: 1,
        zeroOffset: 0.1,
        deviceStatus: "INVALID_STATUS",
      };
      const res = validateMFCTelemetry(invalidMFC);
      expect(res.valid).toBe(false);
      expect(res.errors.length).toBeGreaterThan(0);

      expect(validateMFCTelemetry({ ...validMFC, flowRate: Infinity }).valid).toBe(false);
    });

    it("validates PressureControllerTelemetry correctly", () => {
      const validPressure: PressureControllerTelemetry = {
        pressure: 30.5,
        setpoint: 30.0,
        controlValveOpenPercent: 20.0,
        overpressureTrip: false,
        deviceStatus: "OK",
      };
      expect(validatePressureControllerTelemetry(validPressure).valid).toBe(true);

      const invalidPressure = {
        pressure: 30.5,
        setpoint: 30.0,
        controlValveOpenPercent: -5.0,
        overpressureTrip: "not_a_boolean",
        deviceStatus: "OK",
      };
      expect(validatePressureControllerTelemetry(invalidPressure).valid).toBe(false);
    });

    it("validates UltrasonicTelemetry correctly", () => {
      const validUltra: UltrasonicTelemetry = {
        flowRate: 2.5,
        signalStrength: 45.0,
        bubbleDetect: false,
        deviceStatus: "OK",
      };
      expect(validateUltrasonicTelemetry(validUltra).valid).toBe(true);

      const invalidUltra = {
        flowRate: 2.5,
        signalStrength: "weak",
        bubbleDetect: false,
        deviceStatus: "OK",
      };
      expect(validateUltrasonicTelemetry(invalidUltra).valid).toBe(false);
    });

    it("validates ActuationCommand correctly", () => {
      const validCmd: ActuationCommand = {
        targetDeviceId: "mfc_01",
        commandType: "SET_FLOW",
        payloadValue: 10.0,
        timestamp: Date.now(),
        requestingAgentId: "agent_alpha",
      };
      expect(validateActuationCommand(validCmd).valid).toBe(true);

      const invalidCmd = {
        targetDeviceId: "",
        commandType: "UNKNOWN_CMD",
        payloadValue: NaN,
        timestamp: -1,
        requestingAgentId: "",
      };
      expect(validateActuationCommand(invalidCmd).valid).toBe(false);
    });
  });

  describe("Gas Database & Manifest Models", () => {
    it("looks up gas definitions by numeric or string ID from JSON manifest", () => {
      const n2 = lookupGas(1);
      expect(n2).toBeDefined();
      expect(n2?.formula).toBe("N2");
      expect(n2?.gcf).toBe(1.0);

      const silane = lookupGas("SiH4");
      expect(silane).toBeDefined();
      expect(silane?.classifications).toContain("PYROPHORIC");

      const unknown = lookupGas(999);
      expect(unknown).toBeUndefined();
    });

    it("identifies hazardous gases correctly across all catalog gases", () => {
      const n2 = lookupGas(1)!;
      expect(isHazardousGas(n2)).toBe(false);

      const sih4 = lookupGas(28)!;
      expect(isHazardousGas(sih4)).toBe(true);

      const nh3 = lookupGas(11)!;
      expect(isHazardousGas(nh3)).toBe(true);

      const cl2 = lookupGas(42)!;
      expect(isHazardousGas(cl2)).toBe(true);
    });

    it("updates DeviceStateMachine correctly", () => {
      const sm = new DeviceStateMachine("mfc_test", 100.0, 1);
      expect(sm.mode).toBe("NORMAL");

      sm.updateMFCTelemetry({
        flowRate: 20.0,
        setpoint: 20.0,
        valvePosition: 15.0,
        temperature: 25.0,
        gasId: 1,
        zeroOffset: 0.1,
        deviceStatus: "WARN",
      });
      expect(sm.mode).toBe("DEGRADED");

      sm.updatePressureTelemetry({
        pressure: 200.0,
        setpoint: 50.0,
        controlValveOpenPercent: 100.0,
        overpressureTrip: true,
        deviceStatus: "FAULT",
      });
      expect(sm.mode).toBe("FAULT");
    });
  });

  describe("Safety Rules & Guards", () => {
    it("RampRateGuard allows normal setpoint jump and rejects excessive jump", () => {
      const guard = new RampRateGuard(DEFAULT_BROOKS_POLICY);
      const normalRes = guard.evaluate(0.0, 10.0, 100.0, 100);
      expect(normalRes.passed).toBe(true);

      const excessiveRes = guard.evaluate(0.0, 50.0, 100.0, 100);
      expect(excessiveRes.passed).toBe(false);
      expect(excessiveRes.status).toBe("OUTRIGHT_REJECTED");
      expect(excessiveRes.reason).toContain("Ramp rate breach");

      expect(guard.evaluate(0.0, Infinity, 100.0, 100).passed).toBe(false);
    });

    it("ZeroDriftGuard issues borderline warning when zero offset exceeds ±0.5%", () => {
      const guard = new ZeroDriftGuard(DEFAULT_BROOKS_POLICY);
      const normalRes = guard.evaluate(0.0, 0.2);
      expect(normalRes.passed).toBe(true);

      const driftRes = guard.evaluate(0.0, 0.8);
      expect(driftRes.passed).toBe(false);
      expect(driftRes.status).toBe("BORDERLINE");
      expect(driftRes.confidencePenalty).toBeGreaterThan(0);
    });

    it("PressureGuard and PressureDeltaGuard reject overpressure & delta-P breaches", () => {
      const pGuard = new PressureGuard(DEFAULT_BROOKS_POLICY);
      expect(pGuard.evaluateContainment(100.0).passed).toBe(true);

      const overP = pGuard.evaluateContainment(180.0);
      expect(overP.passed).toBe(false);
      expect(overP.recommendedAction).toBe("VALVE_CLOSE");

      const deltaGuard = new PressureDeltaGuard(DEFAULT_BROOKS_POLICY);
      const deltaRes = deltaGuard.evaluateDeltaP(100.0, 10.0, true);
      expect(deltaRes.passed).toBe(false);
      expect(deltaRes.recommendedAction).toBe("VALVE_CLOSE");
      expect(pGuard.evaluateContainment(NaN).passed).toBe(false);
      expect(deltaGuard.evaluateDeltaP(Infinity, 10.0, true).passed).toBe(false);
    });

    it("GasSafetyGuard enforces gas flow limits and purge requirements for hazardous lines", () => {
      const gGuard = new GasSafetyGuard(DEFAULT_BROOKS_POLICY);

      expect(gGuard.evaluateFlowLimit("SiH4", 15.0).passed).toBe(true);
      const exceedSilane = gGuard.evaluateFlowLimit("SiH4", 30.0);
      expect(exceedSilane.passed).toBe(false);
      expect(exceedSilane.recommendedAction).toBe("VALVE_CLOSE");

      const flowGasChange = gGuard.evaluateGasChange(1, 28, 5.0, false);
      expect(flowGasChange.passed).toBe(false);
      expect(flowGasChange.reason).toContain("active flow detected");

      const canonicalNoOpGasChange = gGuard.evaluateGasChange(28, "SiH4", 5.0, false);
      expect(canonicalNoOpGasChange.passed).toBe(true);

      const noPurgeGasChange = gGuard.evaluateGasChange(1, 28, 0.0, false);
      expect(noPurgeGasChange.passed).toBe(false);
      expect(noPurgeGasChange.reason).toContain("requires prior line purge");

      const purgedGasChange = gGuard.evaluateGasChange(1, 28, 0.0, true);
      expect(purgedGasChange.passed).toBe(true);
      expect(gGuard.evaluateFlowLimit(1, -1.0).passed).toBe(false);
      expect(gGuard.evaluateGasChange(1, 28, NaN, true).passed).toBe(false);
      expect(gGuard.evaluateGasChange("UNKNOWN_GAS", 1, 0.0, true).passed).toBe(false);
    });
  });

  describe("Fieldbus Protocol Adapters & Manifest Offsets", () => {
    it("EtherNet/IP CIP adapter uses SLA5800 manifest for Little-Endian frame conversion", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: 25.5,
        setpoint: 25.0,
        valvePosition: 30.0,
        temperature: 24.0,
        zeroOffset: 0.05,
        gasId: 1,
        deviceStatus: "OK",
        statusFlags: 0,
      };

      const buffer = EtherNetIPAdapter.serializeCIPFrame(telemetry, manifestJson as any);
      expect(buffer.byteLength).toBe(25);

      const parsed = EtherNetIPAdapter.parseCIPFrame(buffer, manifestJson as any);
      expect(parsed.flowRate).toBeCloseTo(25.5, 3);
      expect(parsed.setpoint).toBeCloseTo(25.0, 3);
      expect(parsed.gasId).toBe(1);
      expect(parsed.deviceStatus).toBe("OK");
    });

    it("decodes real-style little-endian EtherNet/IP status flags and multibyte values", () => {
      const buffer = new ArrayBuffer(25);
      const view = new DataView(buffer);
      view.setFloat32(0, 12.5, true);
      view.setFloat32(4, 10.0, true);
      view.setFloat32(8, 50.0, true);
      view.setFloat32(12, 22.0, true);
      view.setFloat32(16, 0.2, true);
      view.setUint16(20, 28, true);
      view.setUint16(22, 0x8000, true);

      const parsed = EtherNetIPAdapter.parseCIPFrame(buffer, manifestJson as any);
      expect(parsed.statusFlags).toBe(0x8000);
      expect(parsed.deviceStatus).toBe("FAULT");
      expect(parsed.gasId).toBe(28);
      expect(parsed.flowRate).toBeCloseTo(12.5, 3);
    });

    it("serializes named gas aliases to canonical numeric gas IDs", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: 1.0,
        setpoint: 1.0,
        valvePosition: 10.0,
        temperature: 21.0,
        zeroOffset: 0.0,
        gasId: "SiH4",
        deviceStatus: "OK",
        statusFlags: 0,
      };

      const cip = EtherNetIPAdapter.serializeCIPFrame(telemetry, manifestJson as any);
      const ethercat = EtherCATAdapter.serializePDOFrame(telemetry, manifestJson as any);
      expect(new DataView(cip).getUint16(20, true)).toBe(28);
      expect(new DataView(ethercat).getUint16(20, true)).toBe(28);
    });

    it("rejects unknown gas IDs instead of defaulting to N2 in serializers", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: 1.0,
        setpoint: 1.0,
        valvePosition: 10.0,
        temperature: 21.0,
        zeroOffset: 0.0,
        gasId: "UNKNOWN_GAS",
        deviceStatus: "OK",
        statusFlags: 0,
      };

      expect(() => EtherNetIPAdapter.serializeCIPFrame(telemetry, manifestJson as any)).toThrow("Unknown or unregistered gas ID");
      expect(() => EtherCATAdapter.serializePDOFrame(telemetry, manifestJson as any)).toThrow("Unknown or unregistered gas ID");
    });

    it("rejects non-finite process values and invalid status flags in serializers", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: Infinity, setpoint: 1.0, valvePosition: 10.0, temperature: 21.0,
        zeroOffset: 0.0, gasId: 1, deviceStatus: "OK", statusFlags: 0,
      };
      expect(() => EtherNetIPAdapter.serializeCIPFrame(telemetry)).toThrow("Invalid EtherNet/IP telemetry");
      expect(() => EtherCATAdapter.serializePDOFrame({ ...telemetry, flowRate: 1.0, statusFlags: -1 })).toThrow("Invalid EtherCAT telemetry");
    });

    it("EtherCAT PDO adapter uses SLA5800 manifest for Little-Endian frame conversion", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: 80.0,
        setpoint: 80.0,
        valvePosition: 75.0,
        temperature: 28.0,
        zeroOffset: -0.1,
        gasId: 28,
        deviceStatus: "WARN",
        statusFlags: 0x4000,
      };

      const buffer = EtherCATAdapter.serializePDOFrame(telemetry, manifestJson as any);
      expect(buffer.byteLength).toBe(25);

      const parsed = EtherCATAdapter.parsePDOFrame(buffer, manifestJson as any);
      expect(parsed.flowRate).toBeCloseTo(80.0, 3);
      expect(parsed.gasId).toBe(28);
      expect(parsed.deviceStatus).toBe("WARN");
    });
  });

  describe("Integration Tests: BrooksDomain Hardware Fallbacks", () => {
    it("triggers immediate VALVE_CLOSE on overpressure anomaly for hazardous gas lines", async () => {
      const domain = new BrooksDomain("mfc_sla5800_line_1");

      const overpressureTelemetry: MFCDeviceTelemetry = {
        flowRate: 50.0,
        setpoint: 50.0,
        valvePosition: 100.0,
        temperature: 30.0,
        zeroOffset: 0.0,
        gasId: 28, // Silane (Hazardous)
        deviceStatus: "FAULT",
        statusFlags: 0x8000,
      };

      const cipFrame = EtherNetIPAdapter.serializeCIPFrame(overpressureTelemetry);
      const parsed = EtherNetIPAdapter.parseCIPFrame(cipFrame);

      domain.stateMachine.updateMFCTelemetry(parsed);
      domain.stateMachine.currentPressurePsi = 180.0;

      const snapshot = await domain.readSensors();
      const decision = await domain.evaluateState(snapshot);

      expect(decision.safetyStatus).toBe("OUTRIGHT_REJECTED");
      expect(domain.stateMachine.mode).toBe("FAULT");
    });

    it("rejects ramp jumps using actual short sample intervals and previous setpoint", async () => {
      const domain = new BrooksDomain("mfc_ramp_guard_test");
      const now = Date.now();
      const snapshot = {
        timestamp: now,
        deviceId: "mfc_ramp_guard_test",
        quality: 0.99,
        readings: {
          flowRate: 0.0,
          setpoint: 10.0,
          valvePosition: 0.0,
          temperature: 23.0,
          gasId: 1,
          zeroOffset: 0.0,
          pressure: 10.0,
          overpressureTrip: false,
          upstreamPressure: 10.0,
          downstreamPressure: 10.0,
          telemetryTimestamp: now,
          previousTelemetryTimestamp: now - 1,
          previousSetpoint: 0.0,
        },
      };

      const decision = await domain.evaluateState(snapshot);
      expect(decision.safetyStatus).toBe("OUTRIGHT_REJECTED");
      expect(decision.reasons.some((r) => r.includes("Ramp rate breach"))).toBe(true);
    });

    it("fails closed when pressure telemetry is malformed", async () => {
      const domain = new BrooksDomain("mfc_pressure_nan_test");
      const now = Date.now();
      const snapshot = {
        timestamp: now,
        deviceId: "mfc_pressure_nan_test",
        quality: 0.99,
        readings: {
          flowRate: 0.0,
          setpoint: 0.0,
          valvePosition: 0.0,
          temperature: 23.0,
          gasId: 1,
          zeroOffset: 0.0,
          pressure: "not-a-number",
          overpressureTrip: false,
          telemetryTimestamp: now,
          previousTelemetryTimestamp: now - 100,
          previousSetpoint: 0.0,
        },
      };

      const decision = await domain.evaluateState(snapshot);
      expect(decision.safetyStatus).toBe("OUTRIGHT_REJECTED");
      expect(decision.reasons.some((r) => r.includes("Invalid pressure telemetry"))).toBe(true);
    });

    it("fails closed for malformed process values and future telemetry timestamps", async () => {
      const domain = new BrooksDomain("mfc_invalid_process_test");
      const now = Date.now();
      const snapshot = {
        timestamp: now,
        deviceId: "mfc_invalid_process_test",
        quality: 0.99,
        readings: {
          flowRate: 0.0,
          setpoint: Number.NaN,
          gasId: 1,
          zeroOffset: 0.0,
          pressure: 10.0,
          upstreamPressure: 10.0,
          downstreamPressure: 10.0,
          telemetryTimestamp: now + 1,
          previousTelemetryTimestamp: now - 100,
          previousSetpoint: 0.0,
        },
      };

      const decision = await domain.evaluateState(snapshot);
      expect(decision.safetyStatus).toBe("OUTRIGHT_REJECTED");
      expect(decision.value).toBe(0.0);
      expect(decision.reasons.some((r) => r.includes("Invalid telemetry timestamp"))).toBe(true);
      expect(decision.reasons.some((r) => r.includes("Invalid flow, setpoint"))).toBe(true);
    });

    it("captures zero-drift warning for inert gas line without triggering hazardous close", async () => {
      const domain = new BrooksDomain("mfc_inert_n2_line");
      domain.stateMachine.activeGasId = 1; // Nitrogen (Inert)

      const snapshot = await domain.readSensors();
      snapshot.readings.zeroOffset = 0.8; // > 0.5% FS zero drift breach

      const decision = await domain.evaluateState(snapshot);
      expect(decision.reasons.some((r) => r.includes("Zero-drift warning"))).toBe(true);
      expect(domain.stateMachine.mode).not.toBe("FAULT");
    });

    it("triggers heartbeat timeout fallback when telemetry staleness exceeds threshold", async () => {
      const domain = new BrooksDomain("mfc_heartbeat_test", DEFAULT_BROOKS_POLICY, undefined, 500); // 500ms timeout
      domain.stateMachine.activeGasId = 28; // Silane (Hazardous)

      const snapshot = await domain.readSensors();
      snapshot.timestamp = Date.now();
      snapshot.readings.telemetryTimestamp = Date.now() - 1500; // 1500ms stale (> 500ms)

      const decision = await domain.evaluateState(snapshot);
      expect(decision.safetyStatus).toBe("OUTRIGHT_REJECTED");
      expect(decision.reasons.some((r) => r.includes("Heartbeat timeout"))).toBe(true);
      expect(domain.stateMachine.mode).toBe("FAULT"); // Hazardous line -> VALVE_CLOSE
    });
  });
});
