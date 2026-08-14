//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const AUTOMOTIVE_PRESETS = {
  STRICT_ODD: {
    borderlineLow: 0.85,
    borderlineHigh: 0.98,
    hardMin: 0.0,
    hardMax: 130.0,
    defaultFallbackValue: 0.0,
  },
};

export class AutomotiveHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "automotive";
  private pipeline: AileeTrustPipeline;

  constructor(preset = AUTOMOTIVE_PRESETS.STRICT_ODD) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "can_bus_gateway",
      quality: 0.99,
      readings: {
        wheelSpeedKmh: 65.0,
        radarDistanceMeters: 45.2,
        cameraConfidence: 0.92,
        steeringAngleDeg: 1.2,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const speed = Number(snapshot.readings.wheelSpeedKmh || 0.0);
    const cameraConf = Number(snapshot.readings.cameraConfidence || 0.5);

    return this.pipeline.process(speed, cameraConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "brake_and_steer_mcu",
      command: { action: "EMERGENCY_DEGRADATION_STOP", targetSpeed: decision.value },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
