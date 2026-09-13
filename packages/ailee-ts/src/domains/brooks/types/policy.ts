//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export interface RampRateConfig {
  maxPercentJumpPer100ms: number; // e.g. 20.0% of Full Scale per 100ms
  timeWindowMs: number; // Baseline step time window, default 100ms
}

export interface ZeroDriftConfig {
  maxDriftPercentFS: number; // e.g. 0.5% of Full Scale
  requireWarningOnExceed: boolean;
}

export interface PressureConfig {
  maxOperatingPressurePsi: number; // Absolute safe container pressure e.g., 150 PSI
  maxDifferentialPressurePsi: number; // Max allowed pressure differential e.g., 50 PSI across valve
}

export interface BrooksSafetyPolicy {
  rampRate: RampRateConfig;
  zeroDrift: ZeroDriftConfig;
  pressure: PressureConfig;
  enforceGasLinePurge: boolean; // Require PURGE_LINE command when changing gases on hazardous lines
  strictHazardousMode: boolean; // Immediately trigger VALVE_CLOSE on any hazardous breach
}

export const DEFAULT_BROOKS_POLICY: BrooksSafetyPolicy = {
  rampRate: {
    maxPercentJumpPer100ms: 20.0,
    timeWindowMs: 100,
  },
  zeroDrift: {
    maxDriftPercentFS: 0.5,
    requireWarningOnExceed: true,
  },
  pressure: {
    maxOperatingPressurePsi: 150.0,
    maxDifferentialPressurePsi: 50.0,
  },
  enforceGasLinePurge: true,
  strictHazardousMode: true,
};
