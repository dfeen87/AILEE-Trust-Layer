//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { describe, expect, it } from "vitest";
import { EtherCATAdapter } from "../adapters/ethercat.js";
import { EtherNetIPAdapter } from "../adapters/ethernet_ip.js";
import manifestJson from "../configs/sla5800_manifest.json" assert { type: "json" };
import { BrooksDomain } from "../index.js";
import { PredictiveStabilityLayer } from "../models/predictive_stability.js";
import { RollingTelemetryWindow, SelfTuningCalibrationModule } from "../models/telemetry_window.js";
import { RampRateGuard, ZeroDriftGuard } from "../rules/flow_bounds.js";
import { GasSafetyGuard } from "../rules/gas_compatibility.js";
import { PressureDeltaGuard, PressureGuard } from "../rules/pressure_guard.js";
import { DEFAULT_BROOKS_POLICY } from "../types/policy.js";
import { MFCDeviceTelemetry } from "../types/telemetry.js";

describe("Brooks Adaptive Calibration & Predictive Stability Suite", () => {
  describe("RollingTelemetryWindow", () => {
    it("returns invalid stats for sparse window (< 5 samples)", () => {
      const window = new RollingTelemetryWindow(20);
      window.addSample({ timestamp: 1000, flowRate: 10, setpoint: 10, pressure: 20, zeroOffset: 0.1 });
      const stats = window.getStats();
      expect(stats.isValid).toBe(false);
      expect(stats.count).toBe(1);
    });

    it("computes accurate moving average, variance, stdDev, and degradation frequency", () => {
      const window = new RollingTelemetryWindow(20);
      for (let i = 0; i < 10; i++) {
        window.addSample({
          timestamp: 1000 + i * 100,
          flowRate: 10 + (i % 2 === 0 ? 1 : -1),
          setpoint: 10,
          pressure: 25 + (i % 2 === 0 ? 2 : -2),
          zeroOffset: 0.1,
          isDegraded: i >= 8, // 2 degraded out of 10
        });
      }

      const stats = window.getStats();
      expect(stats.isValid).toBe(true);
      expect(stats.count).toBe(10);
      expect(stats.meanFlow).toBeCloseTo(10.0, 2);
      expect(stats.meanPressure).toBeCloseTo(25.0, 2);
      expect(stats.degradationFrequency).toBeCloseTo(0.2, 2);
      expect(stats.stdDevFlow).toBeGreaterThan(0);
      expect(stats.stdDevPressure).toBeGreaterThan(0);
    });
  });

  describe("SelfTuningCalibrationModule", () => {
    it("returns hardened fallback for sparse window stats", () => {
      const module = new SelfTuningCalibrationModule(0.95, 0.05, 0.8);
      const window = new RollingTelemetryWindow(20);
      const tuning = module.computeTuning(window.getStats());

      expect(tuning.isHardenedFallback).toBe(true);
      expect(tuning.borderlineHigh).toBe(0.95);
      expect(tuning.uncertaintyBand).toBe(0.05);
      expect(tuning.reasons[0]).toContain("Telemetry window sparse");
    });

    it("dynamically adjusts thresholds based on process variance and degradation frequency", () => {
      const module = new SelfTuningCalibrationModule(0.95, 0.05, 0.8);
      const window = new RollingTelemetryWindow(20);

      // Add high variance & high degradation samples
      for (let i = 0; i < 10; i++) {
        window.addSample({
          timestamp: 1000 + i * 100,
          flowRate: 10 + (i % 2 === 0 ? 8 : -8),
          setpoint: 10,
          pressure: 25 + (i % 2 === 0 ? 6 : -6),
          zeroOffset: 0.4,
          isDegraded: i >= 5, // 50% degradation frequency
        });
      }

      const tuning = module.computeTuning(window.getStats());
      expect(tuning.isHardenedFallback).toBe(false);
      expect(tuning.borderlineHigh).toBeGreaterThan(0.95); // tightened
      expect(tuning.consensusWeight).toBeGreaterThan(0.8); // tightened
      expect(tuning.uncertaintyBand).toBeLessThan(0.05); // narrowed
      expect(tuning.reasons.length).toBeGreaterThan(0);
    });
  });

  describe("PredictiveStabilityLayer", () => {
    it("classifies STABLE, DEGRADED_SOON, and HAZARDOUS_SOON states", () => {
      const layer = new PredictiveStabilityLayer();
      const window = new RollingTelemetryWindow(20);

      // Stable samples
      for (let i = 0; i < 10; i++) {
        window.addSample({ timestamp: 1000 + i * 100, flowRate: 10, setpoint: 10, pressure: 20, zeroOffset: 0.0 });
      }
      const stableRes = layer.evaluatePredictiveState(window.getStats(), [10, 10]);
      expect(stableRes.state).toBe("STABLE");
      expect(stableRes.tighteningFactor).toBe(1.0);
      expect(stableRes.scoreByte).toBe(0);

      // Highly volatile samples -> HAZARDOUS_SOON
      const volatileWindow = new RollingTelemetryWindow(20);
      for (let i = 0; i < 10; i++) {
        volatileWindow.addSample({
          timestamp: 1000 + i * 100,
          flowRate: i * 15,
          setpoint: 10,
          pressure: i * 8,
          zeroOffset: 1.5,
          isDegraded: true,
        });
      }
      const hazRes = layer.evaluatePredictiveState(volatileWindow.getStats(), [0, 150]);
      expect(hazRes.state).toBe("HAZARDOUS_SOON");
      expect(hazRes.tighteningFactor).toBe(0.4);
      expect(hazRes.scoreByte).toBeGreaterThan(150);
    });

    it("detects monotonicity violations on erratic score drops", () => {
      const layer = new PredictiveStabilityLayer();
      const volatileWindow = new RollingTelemetryWindow(20);
      for (let i = 0; i < 10; i++) {
        volatileWindow.addSample({ timestamp: 1000 + i * 100, flowRate: i * 20, setpoint: 10, pressure: i * 10, zeroOffset: 2.0, isDegraded: true });
      }

      // First run establishes high score > 0.8
      const highRes = layer.evaluatePredictiveState(volatileWindow.getStats(), [0, 200]);
      expect(highRes.score).toBeGreaterThan(0.8);

      // Stable window immediately following
      const stableWindow = new RollingTelemetryWindow(20);
      for (let i = 0; i < 10; i++) {
        stableWindow.addSample({ timestamp: 2000 + i * 100, flowRate: 10, setpoint: 10, pressure: 20, zeroOffset: 0.0 });
      }

      const dropRes = layer.evaluatePredictiveState(stableWindow.getStats(), [10, 10]);
      expect(dropRes.isMonotonicViolation).toBe(true);
      expect(dropRes.isValid).toBe(false);
      expect(dropRes.reasons.some((r) => r.includes("Monotonicity violation"))).toBe(true);
    });
  });

  describe("Predictive Physical Guard Tightening", () => {
    it("tightens RampRateGuard and ZeroDriftGuard limits with tightening factor", () => {
      const rampGuard = new RampRateGuard(DEFAULT_BROOKS_POLICY);
      const zeroGuard = new ZeroDriftGuard(DEFAULT_BROOKS_POLICY);

      // Normal factor 1.0 allows 15% jump
      expect(rampGuard.evaluate(0, 15, 100, 100, 1.0).passed).toBe(true);

      // Tightened factor 0.4 caps jump to 8% -> 15% jump rejected
      const tightenedRamp = rampGuard.evaluate(0, 15, 100, 100, 0.4);
      expect(tightenedRamp.passed).toBe(false);
      expect(tightenedRamp.reason).toContain("tightening factor: 0.40");

      // Normal factor allows 0.4% zero offset
      expect(zeroGuard.evaluate(0, 0.4, 1.0).passed).toBe(true);

      // Tightened factor 0.4 caps zero drift to ±0.2% -> 0.4% offset triggers warning
      const tightenedZero = zeroGuard.evaluate(0, 0.4, 0.4);
      expect(tightenedZero.passed).toBe(false);
      expect(tightenedZero.status).toBe("BORDERLINE");
    });

    it("tightens PressureGuard, PressureDeltaGuard, and GasSafetyGuard limits", () => {
      const pGuard = new PressureGuard(DEFAULT_BROOKS_POLICY);
      const deltaGuard = new PressureDeltaGuard(DEFAULT_BROOKS_POLICY);
      const gasGuard = new GasSafetyGuard(DEFAULT_BROOKS_POLICY);

      // 100 PSI safe under 150 PSI limit
      expect(pGuard.evaluateContainment(100, false, 1.0).passed).toBe(true);
      // Tightened factor 0.4 reduces limit to 60 PSI -> 100 PSI rejected
      expect(pGuard.evaluateContainment(100, false, 0.4).passed).toBe(false);

      // 40 PSI delta safe under 50 PSI limit
      expect(deltaGuard.evaluateDeltaP(60, 20, true, 1.0).passed).toBe(true);
      // Tightened factor 0.4 reduces limit to 20 PSI -> 40 PSI delta rejected
      expect(deltaGuard.evaluateDeltaP(60, 20, true, 0.4).passed).toBe(false);

      // Silane (SiH4) limit 20 SLPM
      expect(gasGuard.evaluateFlowLimit("SiH4", 15, 1.0).passed).toBe(true);
      // Tightened factor 0.4 reduces silane limit to 8 SLPM -> 15 SLPM rejected
      expect(gasGuard.evaluateFlowLimit("SiH4", 15, 0.4).passed).toBe(false);
    });
  });

  describe("25-byte Fieldbus Predictive Score Serialization", () => {
    it("roundtrips 25-byte EtherNet/IP CIP frame with predictive score", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: 50.0,
        setpoint: 50.0,
        valvePosition: 40.0,
        temperature: 25.0,
        zeroOffset: 0.1,
        gasId: 1,
        deviceStatus: "OK",
        statusFlags: 0,
        predictiveScore: 180, // UINT8 byte
      };

      const buffer = EtherNetIPAdapter.serializeCIPFrame(telemetry, manifestJson as any);
      expect(buffer.byteLength).toBe(25);

      const parsed = EtherNetIPAdapter.parseCIPFrame(buffer, manifestJson as any);
      expect(parsed.flowRate).toBeCloseTo(50.0, 3);
      expect(parsed.predictiveScore).toBe(180);
    });

    it("roundtrips 25-byte EtherCAT PDO frame with predictive score", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: 10.0,
        setpoint: 10.0,
        valvePosition: 15.0,
        temperature: 22.0,
        zeroOffset: -0.05,
        gasId: 28,
        deviceStatus: "OK",
        statusFlags: 0,
        predictiveScore: 220,
      };

      const buffer = EtherCATAdapter.serializePDOFrame(telemetry, manifestJson as any);
      expect(buffer.byteLength).toBe(25);

      const parsed = EtherCATAdapter.parsePDOFrame(buffer, manifestJson as any);
      expect(parsed.gasId).toBe(28);
      expect(parsed.predictiveScore).toBe(220);
    });
  });

  describe("BrooksHardwareAdapter Hardening & Pre-emptive Triggers", () => {
    it("reverts to static v8.2 behavior and emits SELF_TUNING_DEGRADED on sparse telemetry", async () => {
      const domain = new BrooksDomain("mfc_hardening_test");
      const snapshot = await domain.readSensors();

      // First evaluation with empty telemetry window (<5 samples)
      const decision = await domain.evaluateState(snapshot);

      expect(decision.context?.selfTuningEvent).toBe("SELF_TUNING_DEGRADED");
      expect(decision.reasons.some((r) => r.includes("SELF_TUNING_DEGRADED"))).toBe(true);
      expect(domain.lastPredictiveScoreByte).toBe(0);
    });

    it("pre-emptively triggers valve protection on HAZARDOUS_SOON state with high risk score", async () => {
      const domain = new BrooksDomain("mfc_predictive_hazard_line");
      domain.stateMachine.activeGasId = 28; // Silane (Hazardous)
      const now = Date.now();

      // Feed 10 volatile samples to fill telemetry window and trigger HAZARDOUS_SOON state
      for (let i = 0; i < 10; i++) {
        const snapshot = {
          timestamp: now + i * 100,
          deviceId: "mfc_predictive_hazard_line",
          quality: 0.99,
          readings: {
            flowRate: i * 15,
            setpoint: 10.0,
            valvePosition: 50.0,
            temperature: 25.0,
            gasId: 28, // Silane (Hazardous)
            zeroOffset: 1.2,
            pressure: i * 10,
            upstreamPressure: 50.0,
            downstreamPressure: 40.0,
            telemetryTimestamp: now + i * 100,
            previousTelemetryTimestamp: now + (i - 1) * 100,
            previousSetpoint: 10.0,
          },
        };
        await domain.evaluateState(snapshot);
      }

      // Next snapshot evaluation should trigger pre-emptive valve protection
      const finalSnapshot = {
        timestamp: now + 1100,
        deviceId: "mfc_predictive_hazard_line",
        quality: 0.99,
        readings: {
          flowRate: 150.0,
          setpoint: 10.0,
          valvePosition: 50.0,
          temperature: 25.0,
          gasId: 28,
          zeroOffset: 1.2,
          pressure: 100.0,
          upstreamPressure: 50.0,
          downstreamPressure: 40.0,
          telemetryTimestamp: now + 1100,
          previousTelemetryTimestamp: now + 1000,
          previousSetpoint: 10.0,
        },
      };

      const decision = await domain.evaluateState(finalSnapshot);
      expect(decision.safetyStatus).toBe("OUTRIGHT_REJECTED");
      expect(decision.reasons.some((r) => r.includes("Pre-emptive override") || r.includes("HAZARDOUS_SOON"))).toBe(true);
      expect(domain.stateMachine.mode).toBe("FAULT"); // Hazardous line closed
    });
  });
});
