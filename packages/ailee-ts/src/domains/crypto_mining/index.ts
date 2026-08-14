//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeTrustPipeline } from "../../core/pipeline.js";
import { DecisionResult } from "../../core/types.js";
import { ActuatorCommand, DomainHardwareAdapter, SensorSnapshot } from "../../hardware/adapter.js";

export const CRYPTO_MINING_PRESETS = {
  THERMAL_SAFEGUARD: {
    borderlineLow: 0.75,
    borderlineHigh: 0.9,
    hardMin: 0.0,
    hardMax: 85.0,
    defaultFallbackValue: 65.0,
  },
};

export class CryptoMiningHardwareAdapter implements DomainHardwareAdapter {
  public domainName = "crypto_mining";
  private pipeline: AileeTrustPipeline;

  constructor(preset = CRYPTO_MINING_PRESETS.THERMAL_SAFEGUARD) {
    this.pipeline = new AileeTrustPipeline(preset);
  }

  public async readSensors(): Promise<SensorSnapshot> {
    return {
      timestamp: Date.now(),
      deviceId: "asic_control_board_01",
      quality: 0.98,
      readings: {
        chipTemperatureC: 74.2,
        hashRateThs: 110.5,
        powerUsageWatts: 3200,
        fanSpeedRpm: 5500,
      },
    };
  }

  public async evaluateState(snapshot: SensorSnapshot, trustContext?: Record<string, unknown>): Promise<DecisionResult> {
    const tempC = Number(snapshot.readings.chipTemperatureC || 70.0);
    const sensorHealth = snapshot.quality;

    return this.pipeline.process(tempC, sensorHealth, [], trustContext);
  }

  public async triggerFallback(decision: DecisionResult): Promise<void> {
    await this.writeActuators({
      actuatorId: "power_and_frequency_regulator",
      command: { action: "THROTTLE_FREQUENCY", safeTemp: decision.value },
      priority: 0,
      safeState: true,
    });
  }

  public async writeActuators(command: ActuatorCommand): Promise<void> {}
}
