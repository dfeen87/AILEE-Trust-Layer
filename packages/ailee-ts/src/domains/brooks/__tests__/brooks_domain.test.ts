//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { describe, expect, it } from "vitest";
import { EtherCATAdapter } from "../adapters/ethercat.js";
import { EtherNetIPAdapter } from "../adapters/ethernet_ip.js";
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

  describe("Gas Database & Models", () => {
    it("looks up gas definitions by numeric or string ID", () => {
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

    it("identifies hazardous gases correctly", () => {
      const n2 = lookupGas(1)!;
      expect(isHazardousGas(n2)).toBe(false);

      const sih4 = lookupGas(28)!;
      expect(isHazardousGas(sih4)).toBe(true);

      const nh3 = lookupGas(11)!;
      expect(isHazardousGas(nh3)).toBe(true);
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
      // 10 SLPM jump on 100 SLPM scale = 10% FS change (limit is 20%) -> OK
      const normalRes = guard.evaluate(0.0, 10.0, 100.0, 100);
      expect(normalRes.passed).toBe(true);

      // 50 SLPM jump on 100 SLPM scale = 50% FS change (limit is 20%) -> Breach
      const excessiveRes = guard.evaluate(0.0, 50.0, 100.0, 100);
      expect(excessiveRes.passed).toBe(false);
      expect(excessiveRes.status).toBe("OUTRIGHT_REJECTED");
      expect(excessiveRes.reason).toContain("Ramp rate breach");
    });

    it("ZeroDriftGuard issues borderline warning when zero offset exceeds ±0.5%", () => {
      const guard = new ZeroDriftGuard(DEFAULT_BROOKS_POLICY);
      const normalRes = guard.evaluate(0.0, 0.2);
      expect(normalRes.passed).toBe(true);

      const driftRes = guard.evaluate(0.0, 0.8); // 0.8% > 0.5% limit
      expect(driftRes.passed).toBe(false);
      expect(driftRes.status).toBe("BORDERLINE");
      expect(driftRes.confidencePenalty).toBeGreaterThan(0);
    });

    it("PressureGuard and PressureDeltaGuard reject overpressure & delta-P breaches", () => {
      const pGuard = new PressureGuard(DEFAULT_BROOKS_POLICY);
      expect(pGuard.evaluateContainment(100.0).passed).toBe(true);

      const overP = pGuard.evaluateContainment(180.0); // 180 > 150 max
      expect(overP.passed).toBe(false);
      expect(overP.recommendedAction).toBe("VALVE_CLOSE");

      const deltaGuard = new PressureDeltaGuard(DEFAULT_BROOKS_POLICY);
      // Upstream 100 PSI, Downstream 10 PSI -> Delta-P = 90 PSI > 50 PSI limit
      const deltaRes = deltaGuard.evaluateDeltaP(100.0, 10.0, true);
      expect(deltaRes.passed).toBe(false);
      expect(deltaRes.recommendedAction).toBe("VALVE_CLOSE");
    });

    it("GasSafetyGuard enforces gas flow limits and purge requirements for hazardous lines", () => {
      const gGuard = new GasSafetyGuard(DEFAULT_BROOKS_POLICY);

      // Silane (28) default max flow is 20 SLPM
      expect(gGuard.evaluateFlowLimit("SiH4", 15.0).passed).toBe(true);
      const exceedSilane = gGuard.evaluateFlowLimit("SiH4", 30.0);
      expect(exceedSilane.passed).toBe(false);
      expect(exceedSilane.recommendedAction).toBe("VALVE_CLOSE");

      // Changing from N2 to Silane while active flow exists
      const flowGasChange = gGuard.evaluateGasChange(1, 28, 5.0, false);
      expect(flowGasChange.passed).toBe(false);
      expect(flowGasChange.reason).toContain("active flow detected");

      // Changing from N2 to Silane without purge
      const noPurgeGasChange = gGuard.evaluateGasChange(1, 28, 0.0, false);
      expect(noPurgeGasChange.passed).toBe(false);
      expect(noPurgeGasChange.reason).toContain("requires prior line purge");

      // Changing from N2 to Silane with purge active
      const purgedGasChange = gGuard.evaluateGasChange(1, 28, 0.0, true);
      expect(purgedGasChange.passed).toBe(true);
    });
  });

  describe("Fieldbus Protocol Adapters", () => {
    it("EtherNet/IP CIP adapter correctly parses and serializes Big-Endian frames", () => {
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

      const buffer = EtherNetIPAdapter.serializeCIPFrame(telemetry);
      expect(buffer.byteLength).toBe(24);

      const parsed = EtherNetIPAdapter.parseCIPFrame(buffer);
      expect(parsed.flowRate).toBeCloseTo(25.5, 3);
      expect(parsed.setpoint).toBeCloseTo(25.0, 3);
      expect(parsed.gasId).toBe(1);
      expect(parsed.deviceStatus).toBe("OK");
    });

    it("EtherCAT PDO adapter correctly parses and serializes Little-Endian frames", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: 80.0,
        setpoint: 80.0,
        valvePosition: 75.0,
        temperature: 28.0,
        zeroOffset: -0.1,
        gasId: 28, // Silane
        deviceStatus: "WARN",
        statusFlags: 0x4000,
      };

      const buffer = EtherCATAdapter.serializePDOFrame(telemetry);
      expect(buffer.byteLength).toBe(24);

      const parsed = EtherCATAdapter.parsePDOFrame(buffer);
      expect(parsed.flowRate).toBeCloseTo(80.0, 3);
      expect(parsed.gasId).toBe(28);
      expect(parsed.deviceStatus).toBe("WARN");
    });
  });

  describe("Integration Tests: BrooksDomain AILEE Hardware Adapter", () => {
    it("simulates stream of CIP frames during over-pressurization anomaly", async () => {
      const domain = new BrooksDomain("mfc_sla5800_line_1");

      // Normal reading
      const normalSnapshot = await domain.readSensors();
      const normalDecision = await domain.evaluateState(normalSnapshot);
      expect(normalDecision.safetyStatus).toBe("ACCEPTED");

      // Over-pressurization event stream via CIP payload
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
      domain.stateMachine.currentPressurePsi = 180.0; // Exceeds 150 PSI limit

      const anomalySnapshot = await domain.readSensors();
      const anomalyDecision = await domain.evaluateState(anomalySnapshot);

      expect(anomalyDecision.safetyStatus).toBe("OUTRIGHT_REJECTED");
      expect(domain.stateMachine.mode).toBe("FAULT");
    });
  });
});
