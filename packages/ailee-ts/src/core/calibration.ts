//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import {
  CalibrationConfig,
  CalibrationMetadata,
  CalibrationResult,
  DEFAULT_CALIBRATION_CONFIG,
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
