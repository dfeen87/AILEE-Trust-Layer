//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { ProtocolBridge } from "./adapter.js";

export class MockHardwareProtocolBridge implements ProtocolBridge {
  private connected = false;
  private listeners: Array<(topic: string, payload: unknown) => void> = [];

  public async connect(): Promise<boolean> {
    this.connected = true;
    return true;
  }

  public async disconnect(): Promise<void> {
    this.connected = false;
  }

  public async sendTelemetry(topicOrAddress: string, payload: unknown): Promise<void> {
    if (!this.connected) {
      throw new Error("MockHardwareProtocolBridge is not connected.");
    }
    for (const listener of this.listeners) {
      listener(topicOrAddress, payload);
    }
  }

  public onTelemetry(callback: (topicOrAddress: string, payload: unknown) => void): void {
    this.listeners.push(callback);
  }

  public isConnected(): boolean {
    return this.connected;
  }
}

export class SerialBridge implements ProtocolBridge {
  private connected = false;
  private portName: string;

  constructor(portName = "/dev/ttyUSB0") {
    this.portName = portName;
  }

  public async connect(): Promise<boolean> {
    this.connected = true;
    return true;
  }

  public async disconnect(): Promise<void> {
    this.connected = false;
  }

  public async sendTelemetry(address: string, payload: unknown): Promise<void> {
    if (!this.connected) throw new Error(`Serial port ${this.portName} not connected.`);
  }

  public onTelemetry(callback: (address: string, payload: unknown) => void): void {}
}

export class MqttBridge implements ProtocolBridge {
  private connected = false;
  private brokerUrl: string;

  constructor(brokerUrl = "mqtt://localhost:1883") {
    this.brokerUrl = brokerUrl;
  }

  public async connect(): Promise<boolean> {
    this.connected = true;
    return true;
  }

  public async disconnect(): Promise<void> {
    this.connected = false;
  }

  public async sendTelemetry(topic: string, payload: unknown): Promise<void> {
    if (!this.connected) throw new Error(`MQTT broker ${this.brokerUrl} not connected.`);
  }

  public onTelemetry(callback: (topic: string, payload: unknown) => void): void {}
}

export class HttpWebSocketBridge implements ProtocolBridge {
  private connected = false;
  private endpoint: string;

  constructor(endpoint = "ws://localhost:8080") {
    this.endpoint = endpoint;
  }

  public async connect(): Promise<boolean> {
    this.connected = true;
    return true;
  }

  public async disconnect(): Promise<void> {
    this.connected = false;
  }

  public async sendTelemetry(endpointPath: string, payload: unknown): Promise<void> {
    if (!this.connected) throw new Error(`WebSocket endpoint ${this.endpoint} not connected.`);
  }

  public onTelemetry(callback: (endpointPath: string, payload: unknown) => void): void {}
}
