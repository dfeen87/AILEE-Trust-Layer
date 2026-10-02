//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export interface TelemetrySample {
  timestamp: number;
  flowRate: number;
  setpoint: number;
  pressure: number;
  zeroOffset: number;
  isDegraded?: boolean;
}

export interface RollingWindowStats {
  count: number;
  meanFlow: number;
  varianceFlow: number;
  stdDevFlow: number;
  meanPressure: number;
  variancePressure: number;
  stdDevPressure: number;
  meanZeroOffset: number;
  degradationFrequency: number;
  isValid: boolean;
}

export class RollingTelemetryWindow {
  private windowSize: number;
  private samples: TelemetrySample[] = [];

  constructor(windowSize: number = 20) {
    this.windowSize = Math.max(5, windowSize);
  }

  public addSample(sample: TelemetrySample): void {
    if (!Number.isFinite(sample.timestamp) || !Number.isFinite(sample.flowRate) || !Number.isFinite(sample.pressure) || !Number.isFinite(sample.zeroOffset)) {
      return;
    }
    this.samples.push(sample);
    if (this.samples.length > this.windowSize) {
      this.samples.shift();
    }
  }

  public getSamples(): readonly TelemetrySample[] {
    return this.samples;
  }

  public clear(): void {
    this.samples = [];
  }

  public getStats(): RollingWindowStats {
    const n = this.samples.length;
    if (n < 5) {
      return {
        count: n,
        meanFlow: 0,
        varianceFlow: 0,
        stdDevFlow: 0,
        meanPressure: 0,
        variancePressure: 0,
        stdDevPressure: 0,
        meanZeroOffset: 0,
        degradationFrequency: 0,
        isValid: false,
      };
    }

    let sumFlow = 0;
    let sumPressure = 0;
    let sumZero = 0;
    let degradedCount = 0;

    for (const s of this.samples) {
      if (![s.flowRate, s.pressure, s.zeroOffset].every(Number.isFinite)) {
        return {
          count: n,
          meanFlow: 0,
          varianceFlow: 0,
          stdDevFlow: 0,
          meanPressure: 0,
          variancePressure: 0,
          stdDevPressure: 0,
          meanZeroOffset: 0,
          degradationFrequency: 0,
          isValid: false,
        };
      }
      sumFlow += s.flowRate;
      sumPressure += s.pressure;
      sumZero += s.zeroOffset;
      if (s.isDegraded) degradedCount++;
    }

    const meanFlow = sumFlow / n;
    const meanPressure = sumPressure / n;
    const meanZeroOffset = sumZero / n;

    let varFlowSum = 0;
    let varPressureSum = 0;
    for (const s of this.samples) {
      varFlowSum += Math.pow(s.flowRate - meanFlow, 2);
      varPressureSum += Math.pow(s.pressure - meanPressure, 2);
    }

    const varianceFlow = varFlowSum / n;
    const variancePressure = varPressureSum / n;

    return {
      count: n,
      meanFlow,
      varianceFlow,
      stdDevFlow: Math.sqrt(varianceFlow),
      meanPressure,
      variancePressure,
      stdDevPressure: Math.sqrt(variancePressure),
      meanZeroOffset,
      degradationFrequency: degradedCount / n,
      isValid: true,
    };
  }
}

export interface CalibrationTuning {
  borderlineHigh: number;
  uncertaintyBand: number;
  consensusWeight: number;
  reasons: string[];
  isHardenedFallback: boolean;
}

export class SelfTuningCalibrationModule {
  private baseBorderlineHigh: number;
  private baseUncertaintyBand: number;
  private baseConsensusWeight: number;

  constructor(
    baseBorderlineHigh = 0.95,
    baseUncertaintyBand = 0.05,
    baseConsensusWeight = 0.8
  ) {
    this.baseBorderlineHigh = baseBorderlineHigh;
    this.baseUncertaintyBand = baseUncertaintyBand;
    this.baseConsensusWeight = baseConsensusWeight;
  }

  public computeTuning(stats: RollingWindowStats): CalibrationTuning {
    if (!stats.isValid || stats.count < 5) {
      return {
        borderlineHigh: this.baseBorderlineHigh,
        uncertaintyBand: this.baseUncertaintyBand,
        consensusWeight: this.baseConsensusWeight,
        reasons: ["Telemetry window sparse or invalid; reverting to static baseline calibration"],
        isHardenedFallback: true,
      };
    }

    const reasons: string[] = [];
    let adjustedBorderline = this.baseBorderlineHigh;
    let adjustedUncertainty = this.baseUncertaintyBand;
    let adjustedWeight = this.baseConsensusWeight;

    // High flow variance or pressure variance indicates instability -> tighten thresholds
    if (stats.stdDevFlow > 5.0 || stats.stdDevPressure > 3.0) {
      adjustedBorderline = Math.min(0.99, adjustedBorderline + 0.02);
      adjustedUncertainty = Math.max(0.01, adjustedUncertainty - 0.01);
      reasons.push(`High process variance (flow stdDev: ${stats.stdDevFlow.toFixed(2)}, pressure stdDev: ${stats.stdDevPressure.toFixed(2)}); tightened acceptance threshold`);
    }

    // High degradation frequency -> raise consensus weight requirement
    if (stats.degradationFrequency > 0.2) {
      adjustedWeight = Math.min(0.95, adjustedWeight + 0.1);
      reasons.push(`Elevated degradation frequency (${(stats.degradationFrequency * 100).toFixed(1)}%); increased consensus weight requirement`);
    }

    // High baseline zero offset -> tighten uncertainty band
    if (Math.abs(stats.meanZeroOffset) > 0.3) {
      adjustedUncertainty = Math.max(0.01, adjustedUncertainty - 0.01);
      reasons.push(`Zero-offset drift detected (${stats.meanZeroOffset.toFixed(2)}% FS); narrowed uncertainty band`);
    }

    return {
      borderlineHigh: adjustedBorderline,
      uncertaintyBand: adjustedUncertainty,
      consensusWeight: adjustedWeight,
      reasons,
      isHardenedFallback: false,
    };
  }
}
