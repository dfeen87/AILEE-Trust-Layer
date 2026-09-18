//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { describe, expect, it } from "vitest";
import {
  RollingTelemetryWindow,
  SelfTuningCalibrationModule,
} from "../../../core/calibration.js";
import { EtherCATAdapter } from "../adapters/ethercat.js";
import { EtherNetIPAdapter } from "../adapters/ethernet_ip.js";
import { BrooksDomain } from "../index.js";
import { RampRateGuard, ZeroDriftGuard } from "../rules/flow_bounds.js";
import { GasSafetyGuard } from "../rules/gas_compatibility.js";
import { PressureDeltaGuard, PressureGuard } from "../rules/pressure_guard.js";
import { PredictiveStabilityLayer } from "../rules/predictive_stability.js";
import { MFCDeviceTelemetry } from "../types/telemetry.js";

describe("AILEE v8.3 Adaptive Calibration & Predictive Stability Tests", () => {
  describe("RollingTelemetryWindow & SelfTuningCalibrationModule", () => {
    it("calculates moving average, sample variance, stdDev, and degradation frequency correctly", () => {
      const window = new RollingTelemetryWindow(10);
      expect(window.getStats().sampleCount).toBe(0);

      window.push({ value: 10.0 });
      window.push({ value: 20.0 });
      window.push({ value: 30.0, isDegraded: true });

      const stats = window.getStats();
      expect(stats.sampleCount).toBe(3);
      expect(stats.mean).toBeCloseTo(20.0, 4);
      expect(stats.degradationFrequency).toBeCloseTo(1 / 3, 4);
      expect(stats.variance).toBeGreaterThan(0);
      expect(stats.stdDev).toBeCloseTo(Math.sqrt(stats.variance), 4);
    });

    it("dynamically adjusts thresholds based on telemetry noise and degradation frequency", () => {
      const selfTuning = new SelfTuningCalibrationModule({
        enabled: true,
        windowSize: 10,
        minSamples: 3,
        acceptanceThreshold: 0.95,
        uncertaintyBand: 0.05,
        consensusThreshold: 0.8,
      });

      // Sparse window -> degraded fallback
      selfTuning.recordTelemetry({ value: 10.0 });
      let computed = selfTuning.computeTunedConfig();
      expect(computed.isDegraded).toBe(true);
      expect(computed.config.acceptanceThreshold).toBe(0.95);

      // Add samples with high variance & degradation
      for (let i = 0; i < 5; i++) {
        selfTuning.recordTelemetry({ value: i * 20.0, isDegraded: i % 2 === 0 });
      }

      computed = selfTuning.computeTunedConfig();
      expect(computed.isDegraded).toBe(false);
      expect(computed.config.acceptanceThreshold).toBeGreaterThan(0.95);
      expect(computed.config.uncertaintyBand).toBeGreaterThan(0.05);
      expect(computed.config.consensusThreshold).toBeGreaterThan(0.8);
    });
  });

  describe("PredictiveStabilityLayer", () => {
    it("classifies telemetry trends into STABLE, DEGRADED_SOON, and HAZARDOUS_SOON", () => {
      const pred = new PredictiveStabilityLayer();

      // Stable / constant telemetry
      const stableRes = pred.evaluateTrends([10.0, 10.1, 10.0, 10.2, 10.1]);
      expect(stableRes.state).toBe("STABLE");
      expect(stableRes.tighteningMultiplier).toBe(1.0);
      expect(stableRes.isDegraded).toBe(false);

      // Rising trend -> DEGRADED_SOON or HAZARDOUS_SOON
      const risingRes = pred.evaluateTrends([10.0, 15.0, 25.0, 40.0, 60.0]);
      expect(["DEGRADED_SOON", "HAZARDOUS_SOON"]).toContain(risingRes.state);
      expect(risingRes.tighteningMultiplier).toBeLessThan(1.0);

      // Extreme steep trend -> HAZARDOUS_SOON
      const steepRes = pred.evaluateTrends([0.0, 20.0, 50.0, 90.0, 150.0]);
      expect(steepRes.state).toBe("HAZARDOUS_SOON");
      expect(steepRes.tighteningMultiplier).toBe(0.5);
    });

    it("handles sparse or malformed telemetry gracefully", () => {
      const pred = new PredictiveStabilityLayer();

      // Sparse history (<3 samples)
      const sparse = pred.evaluateTrends([10.0, 12.0]);
      expect(sparse.isDegraded).toBe(true);
      expect(sparse.tighteningMultiplier).toBe(1.0);

      // Non-finite history
      const nonFinite = pred.evaluateTrends([10.0, NaN, 12.0]);
      expect(nonFinite.isDegraded).toBe(true);
      expect(nonFinite.tighteningMultiplier).toBe(1.0);

      // Out-of-bounds history
      const oob = pred.evaluateTrends([-5.0, 10.0, 20.0]);
      expect(oob.isDegraded).toBe(true);
      expect(oob.tighteningMultiplier).toBe(1.0);
    });
  });

  describe("Physical Guard Routing with Predictive Tightening Multiplier", () => {
    it("tightens RampRateGuard thresholds under predictive multiplier", () => {
      const guard = new RampRateGuard();
      // Baseline jump of 15% FS per 100ms is allowed under standard max 20%
      expect(guard.evaluate(0.0, 15.0, 100.0, 100, 1.0).passed).toBe(true);

      // Tightened multiplier 0.50 (max jump becomes 10% FS) -> 15% jump is rejected
      const tightened = guard.evaluate(0.0, 15.0, 100.0, 100, 0.5);
      expect(tightened.passed).toBe(false);
      expect(tightened.status).toBe("OUTRIGHT_REJECTED");
    });

    it("tightens PressureDeltaGuard thresholds under predictive multiplier", () => {
      const guard = new PressureDeltaGuard();
      // Baseline delta P of 40 PSI allowed under max 50 PSI limit
      expect(guard.evaluateDeltaP(60.0, 20.0, true, 1.0).passed).toBe(true);

      // Tightened multiplier 0.75 (max delta P becomes 37.5 PSI) -> 40 PSI delta P rejected
      const tightened = guard.evaluateDeltaP(60.0, 20.0, true, 0.75);
      expect(tightened.passed).toBe(false);
      expect(tightened.recommendedAction).toBe("VALVE_CLOSE");
    });

    it("tightens ZeroDriftGuard thresholds under predictive multiplier", () => {
      const guard = new ZeroDriftGuard();
      // Baseline drift 0.4% FS passes under 0.5% limit
      expect(guard.evaluate(0.0, 0.4, 1.0).passed).toBe(true);

      // Tightened multiplier 0.50 (max drift becomes 0.25% FS) -> 0.4% drift issues warning
      const tightened = guard.evaluate(0.0, 0.4, 0.5);
      expect(tightened.passed).toBe(false);
      expect(tightened.status).toBe("BORDERLINE");
    });

    it("tightens GasSafetyGuard thresholds under predictive multiplier", () => {
      const guard = new GasSafetyGuard();
      // Silane (SiH4) default max flow = 20 SLPM. 15 SLPM passes under multiplier 1.0
      expect(guard.evaluateFlowLimit("SiH4", 15.0, 1.0).passed).toBe(true);

      // Tightened multiplier 0.50 (max flow becomes 10 SLPM) -> 15 SLPM rejected
      const tightened = guard.evaluateFlowLimit("SiH4", 15.0, 0.5);
      expect(tightened.passed).toBe(false);
      expect(tightened.recommendedAction).toBe("VALVE_CLOSE");
    });
  });

  describe("Fieldbus 25-Byte Assembly Serialization & Predictive Score", () => {
    it("serializes and parses 1-byte predictive score field in CIP and PDO frames", () => {
      const telemetry: MFCDeviceTelemetry = {
        flowRate: 35.0,
        setpoint: 35.0,
        valvePosition: 40.0,
        temperature: 25.0,
        zeroOffset: 0.05,
        gasId: 1,
        deviceStatus: "OK",
        statusFlags: 0,
        predictiveScore: 0.85, // 85% hazard score
      };

      const cipBuffer = EtherNetIPAdapter.serializeCIPFrame(telemetry);
      expect(cipBuffer.byteLength).toBe(25);
      const parsedCIP = EtherNetIPAdapter.parseCIPFrame(cipBuffer);
      expect(parsedCIP.predictiveScore).toBe(Math.round(0.85 * 255));

      const pdoBuffer = EtherCATAdapter.serializePDOFrame(telemetry);
      expect(pdoBuffer.byteLength).toBe(25);
      const parsedPDO = EtherCATAdapter.parsePDOFrame(pdoBuffer);
      expect(parsedPDO.predictiveScore).toBe(Math.round(0.85 * 255));
    });
  });

  describe("Hardening Step: Fail-Safe Reversion to Static v8.2 Behavior", () => {
    it("reverts to static v8.2 behavior and emits SELF_TUNING_DEGRADED when telemetry is sparse or malformed", async () => {
      const domain = new BrooksDomain("mfc_hardening_test");

      // Single telemetry sample (<5 required min window samples)
      const snapshot = await domain.readSensors();
      snapshot.readings.flowRate = 10.0;
      snapshot.readings.setpoint = 10.0;

      const decision = await domain.evaluateState(snapshot);

      expect(decision.reasons.some((r) => r.includes("SELF_TUNING_DEGRADED"))).toBe(true);
      const predContext = decision.context?.predictive as Record<string, unknown>;
      expect(predContext?.selfTuningDegraded).toBe(true);
      expect(predContext?.tighteningMultiplier).toBe(1.0); // Static v8.2 multiplier
      expect(predContext?.serializedPredictiveScore).toBe(0); // Predictive score blocked
    });

    it("applies predictive tightening and serializes predictive score when telemetry window is healthy", async () => {
      const domain = new BrooksDomain("mfc_healthy_predictive_test");

      // Feed 5 healthy telemetry samples
      for (let i = 0; i < 5; i++) {
        const snap = await domain.readSensors();
        snap.readings.flowRate = 10.0 + i * 5.0; // Accelerating flow
        snap.readings.setpoint = 10.0 + i * 5.0;
        await domain.evaluateState(snap);
      }

      const snap = await domain.readSensors();
      snap.readings.flowRate = 35.0;
      snap.readings.setpoint = 35.0;

      const decision = await domain.evaluateState(snap);

      const predContext = decision.context?.predictive as Record<string, unknown>;
      expect(predContext?.selfTuningDegraded).toBe(false);
      expect(predContext?.tighteningMultiplier as number).toBeLessThanOrEqual(0.75);
      expect(predContext?.serializedPredictiveScore as number).toBeGreaterThan(0);
    });
  });
});
