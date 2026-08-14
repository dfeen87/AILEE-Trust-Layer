//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const ROBOTICS_PRESETS = {
  SAFETY_RATED_STOP: {
    borderlineLow: 0.85,
    borderlineHigh: 0.98,
    hardMin: 0.0,
    hardMax: 2.0,
    defaultFallbackValue: 0.0,
  },
};

export class RoboticsHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "robotics";
  private pipeline: AileeTrustPipeline;

  constructor(preset = ROBOTICS_PRESETS.SAFETY_RATED_STOP) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "cobot_joint_controller",
      quality: 0.99,
      readings: {
        jointTorqueNm: 12.4,
        endEffectorSpeedMs: 0.85,
        humanDistanceMeters: 1.5,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const speed = Number(snapshot.readings.endEffectorSpeedMs || 0.0);
    const humanDist = Number(snapshot.readings.humanDistanceMeters || 1.0);
    const rawConf = Math.min(1.0, humanDist / 2.0);

    return this.pipeline.process(speed, rawConf, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "joint_brake",
      command: { action: "ENGAGE_SAFETY_BRAKE", velocity: decision.value },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
