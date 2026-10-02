//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { MFCDeviceTelemetry, PressureControllerTelemetry, UltrasonicTelemetry } from "../types/telemetry.js";
import { GasDefinition, lookupGas } from "./gas_database.js";

export type OperationalMode = "NORMAL" | "PURGING" | "DEGRADED" | "FAULT" | "HOLD";

export class DeviceStateMachine {
  public deviceId: string;
  public mode: OperationalMode = "NORMAL";
  public activeGasId: string | number = 1; // Default Nitrogen (N2)
  public fullScaleFlowSlpm: number = 100.0;
  public currentFlowRate: number = 0.0;
  public currentSetpoint: number = 0.0;
  public previousSetpoint: number = 0.0;
  public currentPressurePsi: number = 0.0;
  public zeroOffsetPercentFS: number = 0.0;
  public lastTelemetryTimestamp: number = Date.now();
  public previousTelemetryTimestamp: number = Date.now();

  constructor(deviceId: string, fullScaleFlowSlpm = 100.0, initialGasId: string | number = 1) {
    this.deviceId = deviceId;
    this.fullScaleFlowSlpm = fullScaleFlowSlpm;
    this.activeGasId = initialGasId;
  }

  public getActiveGas(): GasDefinition | undefined {
    return lookupGas(this.activeGasId);
  }

  public updateMFCTelemetry(telemetry: MFCDeviceTelemetry, timestamp = Date.now()): void {
    this.previousSetpoint = this.currentSetpoint;
    this.previousTelemetryTimestamp = this.lastTelemetryTimestamp;
    this.currentFlowRate = telemetry.flowRate;
    this.currentSetpoint = telemetry.setpoint;
    this.zeroOffsetPercentFS = telemetry.zeroOffset;
    this.activeGasId = telemetry.gasId;
    this.lastTelemetryTimestamp = timestamp;

    if (telemetry.deviceStatus === "FAULT") {
      this.mode = "FAULT";
    } else if (telemetry.deviceStatus === "WARN" && this.mode !== "FAULT") {
      this.mode = "DEGRADED";
    }
  }

  public updatePressureTelemetry(telemetry: PressureControllerTelemetry, timestamp = Date.now()): void {
    this.currentPressurePsi = telemetry.pressure;
    this.lastTelemetryTimestamp = timestamp;

    if (telemetry.overpressureTrip || telemetry.deviceStatus === "FAULT") {
      this.mode = "FAULT";
    }
  }

  public updateUltrasonicTelemetry(telemetry: UltrasonicTelemetry, timestamp = Date.now()): void {
    this.lastTelemetryTimestamp = timestamp;
    if (telemetry.deviceStatus === "FAULT") {
      this.mode = "FAULT";
    }
  }

  public setMode(newMode: OperationalMode): void {
    this.mode = newMode;
  }
}
