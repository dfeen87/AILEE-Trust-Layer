//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export type DeviceStatus = "OK" | "WARN" | "FAULT";

export interface MFCDeviceTelemetry {
  flowRate: number; // slpm or sccm
  setpoint: number; // target flow rate
  valvePosition: number; // 0.0 to 100.0%
  temperature: number; // °C
  gasId: string | number; // Gas identifier (e.g., 1 for N2, "SiH4", etc.)
  zeroOffset: number; // Zero calibration offset (% of Full Scale)
  deviceStatus: DeviceStatus;
  statusFlags?: number; // Bitfield flags from fieldbus
}

export interface PressureControllerTelemetry {
  pressure: number; // PSI or bar
  setpoint: number; // Target pressure
  controlValveOpenPercent: number; // 0.0 to 100.0%
  overpressureTrip: boolean; // Overpressure condition triggered
  deviceStatus: DeviceStatus;
}

export interface UltrasonicTelemetry {
  flowRate: number; // L/min
  signalStrength: number; // dB
  bubbleDetect: boolean; // Liquid bubble flag
  deviceStatus: DeviceStatus;
}

export function validateMFCTelemetry(data: unknown): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  if (typeof data !== "object" || data === null) {
    return { valid: false, errors: ["Telemetry payload must be a non-null object"] };
  }

  const t = data as Record<string, unknown>;

  if (typeof t.flowRate !== "number" || Number.isNaN(t.flowRate)) errors.push("Invalid or missing flowRate");
  if (typeof t.setpoint !== "number" || Number.isNaN(t.setpoint)) errors.push("Invalid or missing setpoint");
  if (typeof t.valvePosition !== "number" || Number.isNaN(t.valvePosition) || t.valvePosition < 0 || t.valvePosition > 100) {
    errors.push("valvePosition must be a number between 0.0 and 100.0%");
  }
  if (typeof t.temperature !== "number" || Number.isNaN(t.temperature)) errors.push("Invalid or missing temperature");
  if (t.gasId === undefined || t.gasId === null || (typeof t.gasId !== "string" && typeof t.gasId !== "number")) {
    errors.push("gasId must be a non-empty string or number");
  }
  if (typeof t.zeroOffset !== "number" || Number.isNaN(t.zeroOffset)) errors.push("Invalid or missing zeroOffset");
  if (t.deviceStatus !== "OK" && t.deviceStatus !== "WARN" && t.deviceStatus !== "FAULT") {
    errors.push("deviceStatus must be 'OK', 'WARN', or 'FAULT'");
  }

  return { valid: errors.length === 0, errors };
}

export function validatePressureControllerTelemetry(data: unknown): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  if (typeof data !== "object" || data === null) {
    return { valid: false, errors: ["Pressure telemetry payload must be a non-null object"] };
  }

  const t = data as Record<string, unknown>;

  if (typeof t.pressure !== "number" || Number.isNaN(t.pressure)) errors.push("Invalid or missing pressure");
  if (typeof t.setpoint !== "number" || Number.isNaN(t.setpoint)) errors.push("Invalid or missing setpoint");
  if (
    typeof t.controlValveOpenPercent !== "number" ||
    Number.isNaN(t.controlValveOpenPercent) ||
    t.controlValveOpenPercent < 0 ||
    t.controlValveOpenPercent > 100
  ) {
    errors.push("controlValveOpenPercent must be a number between 0.0 and 100.0%");
  }
  if (typeof t.overpressureTrip !== "boolean") errors.push("overpressureTrip must be a boolean");
  if (t.deviceStatus !== "OK" && t.deviceStatus !== "WARN" && t.deviceStatus !== "FAULT") {
    errors.push("deviceStatus must be 'OK', 'WARN', or 'FAULT'");
  }

  return { valid: errors.length === 0, errors };
}

export function validateUltrasonicTelemetry(data: unknown): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  if (typeof data !== "object" || data === null) {
    return { valid: false, errors: ["Ultrasonic telemetry payload must be a non-null object"] };
  }

  const t = data as Record<string, unknown>;

  if (typeof t.flowRate !== "number" || Number.isNaN(t.flowRate)) errors.push("Invalid or missing flowRate");
  if (typeof t.signalStrength !== "number" || Number.isNaN(t.signalStrength)) errors.push("Invalid or missing signalStrength");
  if (typeof t.bubbleDetect !== "boolean") errors.push("bubbleDetect must be a boolean");
  if (t.deviceStatus !== "OK" && t.deviceStatus !== "WARN" && t.deviceStatus !== "FAULT") {
    errors.push("deviceStatus must be 'OK', 'WARN', or 'FAULT'");
  }

  return { valid: errors.length === 0, errors };
}
