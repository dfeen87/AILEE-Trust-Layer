//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import {
  CalibrationConfig,
  CalibrationMetadata,
  CalibrationResult,
  DEFAULT_CALIBRATION_CONFIG,
  DEFAULT_SELF_TUNING_CONFIG,
  RollingWindowStats,
  SelfTuningCalibrationConfig,
  TelemetrySample,
} from "./types.js";

/**
 * Pure, opt-in V8.1 confidence refinement. It never scores telemetry or
 * changes a V8 decision directly: callers pass its output to the existing
 * trust pipeline and retain the original confidence on any invalid input.
 */
export class CalibrationLayer {
  private readonly config: CalibrationConfig;

  constructor(config: Partial<CalibrationConfig> = {}) {
    this.config = { ...DEFAULT_CALIBRATION_CONFIG, ...config };
  }

  public calibrate(baselineConfidence: number, metadata?: CalibrationMetadata): CalibrationResult {
    if (!this.config.enabled) {
      return this.baseline(baselineConfidence, "DISABLED", "BASELINE", "Calibration disabled; V8 baseline retained.");
    }

    if (!this.isValidConfig() || !Number.isFinite(baselineConfidence) || baselineConfidence < 0 || baselineConfidence > 1 || !this.isValidMetadata(metadata)) {
      return this.baseline(baselineConfidence, "INVALID_INPUT", "BASELINE", "Calibration input invalid; V8 baseline retained.", true);
    }

    const lowerBound = this.config.acceptanceThreshold - this.config.uncertaintyBand;
    if (baselineConfidence >= this.config.acceptanceThreshold) {
      return this.baseline(baselineConfidence, "OUTSIDE_UNCERTAINTY_BAND", "ABOVE_THRESHOLD", "Confidence already meets calibration threshold.");
    }
    if (baselineConfidence < lowerBound) {
      return this.baseline(baselineConfidence, "OUTSIDE_UNCERTAINTY_BAND", "BELOW_UNCERTAINTY_ZONE", "Confidence is below the calibration uncertainty zone.");
    }

    const consensus = metadata?.peerConsensus;
    const consensusChecked = consensus !== undefined;
    if (!consensus || consensus.peerCount < this.config.minimumPeerCount || consensus.agreement < this.config.consensusThreshold) {
      return {
        ...this.baseline(baselineConfidence, "CONSENSUS_NOT_MET", "UNCERTAINTY_ZONE", "Uncertainty-zone confidence retained because peer consensus was not met."),
        consensusChecked,
      };
    }

    const requestedGrace = metadata?.graceMargin ?? this.config.maxGraceMargin;
    const grace = Math.min(requestedGrace, this.config.maxGraceMargin, this.config.acceptanceThreshold - baselineConfidence);
    return {
      confidence: baselineConfidence + grace,
      applied: grace > 0,
      fallbackUsed: false,
      event: "GRACE_APPLIED",
      thresholdDecision: "UNCERTAINTY_ZONE",
      consensusChecked: true,
      reasons: [`Calibration applied ${grace.toFixed(3)} grace margin after peer consensus.`],
    };
  }

  private baseline(confidence: number, event: CalibrationResult["event"], thresholdDecision: CalibrationResult["thresholdDecision"], reason: string, fallbackUsed = false): CalibrationResult {
    return { confidence, applied: false, fallbackUsed, event, thresholdDecision, consensusChecked: false, reasons: [reason] };
  }

  private isValidConfig(): boolean {
    const c = this.config;
    return [c.acceptanceThreshold, c.uncertaintyBand, c.maxGraceMargin, c.consensusThreshold].every(Number.isFinite)
      && c.acceptanceThreshold >= 0 && c.acceptanceThreshold <= 1
      && c.uncertaintyBand >= 0 && c.uncertaintyBand <= c.acceptanceThreshold
      && c.maxGraceMargin >= 0 && c.consensusThreshold >= 0 && c.consensusThreshold <= 1
      && Number.isInteger(c.minimumPeerCount) && c.minimumPeerCount >= 1;
  }

  private isValidMetadata(metadata?: CalibrationMetadata): boolean {
    if (!metadata) return true;
    if (typeof metadata !== "object" || Array.isArray(metadata)) return false;
    if (metadata.graceMargin !== undefined && (!Number.isFinite(metadata.graceMargin) || metadata.graceMargin < 0)) return false;
    const consensus = metadata.peerConsensus;
    return !consensus || (typeof consensus === "object" && !Array.isArray(consensus)
      && Number.isFinite(consensus.agreement) && consensus.agreement >= 0 && consensus.agreement <= 1
      && Number.isInteger(consensus.peerCount) && consensus.peerCount >= 0);
  }
}

/**
 * Rolling window data structure storing recent telemetry samples for
 * dynamic statistical analysis (moving averages, sample variance, degradation frequency).
 */
export class RollingTelemetryWindow {
  private samples: TelemetrySample[] = [];
  private readonly capacity: number;

  constructor(capacity = 20) {
    this.capacity = Math.max(1, capacity);
  }

  public push(sample: TelemetrySample): void {
    if (!Number.isFinite(sample.value)) return;
    if (this.samples.length >= this.capacity) {
      this.samples.shift();
    }
    this.samples.push(sample);
  }

  public clear(): void {
    this.samples = [];
  }

  public size(): number {
    return this.samples.length;
  }

  public getSamples(): readonly TelemetrySample[] {
    return this.samples;
  }

  public getStats(): RollingWindowStats {
    const n = this.samples.length;
    if (n === 0) {
      return { mean: 0, variance: 0, stdDev: 0, degradationFrequency: 0, sampleCount: 0 };
    }

    let sum = 0;
    let degradedCount = 0;
    for (const s of this.samples) {
      sum += s.value;
      if (s.isDegraded) degradedCount++;
    }
    const mean = sum / n;

    let sqDiffSum = 0;
    for (const s of this.samples) {
      sqDiffSum += (s.value - mean) ** 2;
    }
    const variance = n > 1 ? sqDiffSum / (n - 1) : 0;
    const stdDev = Math.sqrt(variance);
    const degradationFrequency = degradedCount / n;

    return {
      mean,
      variance,
      stdDev,
      degradationFrequency,
      sampleCount: n,
    };
  }
}

/**
 * Self-tuning calibration module that uses rolling telemetry statistics
 * to dynamically adjust thresholds, uncertainty bands, and consensus requirements.
 */
export class SelfTuningCalibrationModule {
  private window: RollingTelemetryWindow;
  private selfTuningConfig: SelfTuningCalibrationConfig;

  constructor(config: Partial<SelfTuningCalibrationConfig> = {}) {
    this.selfTuningConfig = { ...DEFAULT_SELF_TUNING_CONFIG, ...config };
    this.window = new RollingTelemetryWindow(this.selfTuningConfig.windowSize);
  }

  public recordTelemetry(sample: TelemetrySample): void {
    this.window.push(sample);
  }

  public getWindow(): RollingTelemetryWindow {
    return this.window;
  }

  public computeTunedConfig(): {
    config: CalibrationConfig;
    isDegraded: boolean;
    reasons: string[];
    stats: RollingWindowStats;
  } {
    const stats = this.window.getStats();
    const cfg = this.selfTuningConfig;
    const reasons: string[] = [];

    // Check sparse window
    if (stats.sampleCount < cfg.minSamples) {
      return {
        config: {
          enabled: cfg.enabled,
          acceptanceThreshold: cfg.acceptanceThreshold,
          uncertaintyBand: cfg.uncertaintyBand,
          maxGraceMargin: cfg.maxGraceMargin,
          consensusThreshold: cfg.consensusThreshold,
          minimumPeerCount: cfg.minimumPeerCount,
        },
        isDegraded: true,
        reasons: [`Sparse telemetry window: ${stats.sampleCount} samples < minimum required ${cfg.minSamples}`],
        stats,
      };
    }

    // Dynamic adjustment based on variance & degradation frequency
    const varAdjustment = stats.stdDev * cfg.varianceSensitivity;
    const degAdjustment = stats.degradationFrequency * cfg.degradationSensitivity;
    const totalShift = varAdjustment + degAdjustment;

    // Tighten acceptance threshold (cap at 0.99)
    const tunedAcceptanceThreshold = Math.min(0.99, Math.max(cfg.acceptanceThreshold, cfg.acceptanceThreshold + totalShift * 0.1));

    // Widen uncertainty band (cap at tunedAcceptanceThreshold)
    const tunedUncertaintyBand = Math.min(tunedAcceptanceThreshold, Math.max(cfg.uncertaintyBand, cfg.uncertaintyBand + totalShift * 0.1));

    // Tighten consensus threshold (cap at 0.99)
    const tunedConsensusThreshold = Math.min(0.99, Math.max(cfg.consensusThreshold, cfg.consensusThreshold + totalShift * 0.15));

    reasons.push(
      `Self-tuning calibration calculated dynamic parameters (stdDev=${stats.stdDev.toFixed(4)}, degFreq=${(stats.degradationFrequency * 100).toFixed(1)}%).`
    );

    return {
      config: {
        enabled: cfg.enabled,
        acceptanceThreshold: tunedAcceptanceThreshold,
        uncertaintyBand: tunedUncertaintyBand,
        maxGraceMargin: cfg.maxGraceMargin,
        consensusThreshold: tunedConsensusThreshold,
        minimumPeerCount: cfg.minimumPeerCount,
      },
      isDegraded: false,
      reasons,
      stats,
    };
  }

  public calibrate(baselineConfidence: number, metadata?: CalibrationMetadata): CalibrationResult & { stats?: RollingWindowStats } {
    const tuned = this.computeTunedConfig();
    const calibrationLayer = new CalibrationLayer(tuned.config);
    const result = calibrationLayer.calibrate(baselineConfidence, metadata);
    if (tuned.reasons.length > 0 && tuned.config.enabled) {
      result.reasons = [...result.reasons, ...tuned.reasons];
    }
    return { ...result, stats: tuned.stats };
  }
}
