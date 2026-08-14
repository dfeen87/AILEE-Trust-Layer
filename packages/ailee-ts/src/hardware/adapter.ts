//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { DecisionResult } from "../core/types.js";

export interface SensorSnapshot {
  timestamp: number;
  readings: Record<string, number | boolean | string | unknown>;
  quality: number;
  deviceId: string;
}

export interface ActuatorCommand {
  actuatorId: string;
  command: string | number | Record<string, unknown>;
  priority: number;
  safeState: boolean;
}

export interface DomainHardwareAdapter {
  domainName: string;
  readSensors(): Promise<SensorSnapshot>;
  evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult>;
  triggerFallback(decision: DecisionResult): Promise<void>;
  writeActuators(command: ActuatorCommand): Promise<void>;
}

export interface ProtocolBridge {
  connect(): Promise<boolean>;
  disconnect(): Promise<void>;
  sendTelemetry(topicOrAddress: string, payload: unknown): Promise<void>;
  onTelemetry(callback: (topicOrAddress: string, payload: unknown) => void): void;
}
