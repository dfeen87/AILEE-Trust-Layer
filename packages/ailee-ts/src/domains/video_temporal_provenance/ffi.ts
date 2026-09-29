//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export interface FrameSignal {
  frameIndex: number;
  timestampSec: number;
  rawTrust: number;
  hashDelta: number;
  dx: number;
  dy: number;
  flowConsistency: number;
}

export interface TemporalIntegrityMetrics {
  overallTrustScore: number;
  meanFrameTrust: number;
  transitionIntegrityAvg: number;
  sceneBoundaryTrustAvg: number;
  temporalContinuityScore: number;
  opticalFlowStability: number;
  rhythmStability: number;
  totalFrames: number;
  totalTransitions: number;
  totalSceneBoundaries: number;
  anomalyCount: number;
  safetyStatus: "ACCEPTED" | "PARTIALLY_TRUSTED" | "OUTRIGHT_REJECTED";
}

export class TPEFFIBridge {
  private frames: FrameSignal[] = [];

  public reset(): void {
    this.frames = [];
  }

  public ingestFrame(signal: FrameSignal): boolean {
    if (
      signal === null || typeof signal !== "object" ||
      !Number.isSafeInteger(signal.frameIndex) || signal.frameIndex < 0 ||
      ![signal.timestampSec, signal.rawTrust, signal.hashDelta, signal.dx, signal.dy, signal.flowConsistency].every(Number.isFinite) ||
      signal.timestampSec < 0 || signal.rawTrust < 0 || signal.rawTrust > 100 ||
      signal.flowConsistency < 0 || signal.flowConsistency > 1 ||
      (this.frames.length > 0 && signal.frameIndex <= this.frames[this.frames.length - 1].frameIndex) ||
      (this.frames.length > 0 && signal.timestampSec <= this.frames[this.frames.length - 1].timestampSec)
    ) {
      return false;
    }
    this.frames.push(signal);
    return true;
  }

  public evaluate(): TemporalIntegrityMetrics {
    if (this.frames.length === 0) {
      return {
        overallTrustScore: 0.0,
        meanFrameTrust: 0.0,
        transitionIntegrityAvg: 0.0,
        sceneBoundaryTrustAvg: 0.0,
        temporalContinuityScore: 0.0,
        opticalFlowStability: 0.0,
        rhythmStability: 0.0,
        totalFrames: 0,
        totalTransitions: 0,
        totalSceneBoundaries: 0,
        anomalyCount: 0,
        safetyStatus: "OUTRIGHT_REJECTED",
      };
    }

    let sumTrust = 0;
    let anomalyCount = 0;
    let sceneBoundaryCount = 0;

    for (const f of this.frames) {
      sumTrust += f.rawTrust;
      if (f.flowConsistency < 0.35 || f.rawTrust < 50 || Math.abs(f.hashDelta) > 0.75) {
        anomalyCount++;
      }
      if (Math.abs(f.hashDelta) > 0.35) sceneBoundaryCount++;
    }

    const meanFrameTrust = sumTrust / this.frames.length;
    const transitionIntegrityAvg = Math.max(0, 100.0 - anomalyCount * 15.0);
    const sceneBoundaryTrustAvg = sceneBoundaryCount > 0 ? 90.0 : 100.0;
    const temporalContinuityScore = Math.max(0, 100.0 - anomalyCount * 10.0);
    const opticalFlowStability = 85.0;
    const rhythmStability = 95.0;

    let overallTrust =
      meanFrameTrust * 0.35 +
      transitionIntegrityAvg * 0.3 +
      temporalContinuityScore * 0.25 +
      sceneBoundaryTrustAvg * 0.1 -
      anomalyCount * 10.0;

    overallTrust = Math.max(0.0, Math.min(100.0, overallTrust));

    let safetyStatus: "ACCEPTED" | "PARTIALLY_TRUSTED" | "OUTRIGHT_REJECTED" = "ACCEPTED";
    if (overallTrust < 50.0) {
      safetyStatus = "OUTRIGHT_REJECTED";
    } else if (overallTrust < 85.0 || anomalyCount > 0) {
      safetyStatus = "PARTIALLY_TRUSTED";
    }

    return {
      overallTrustScore: overallTrust,
      meanFrameTrust,
      transitionIntegrityAvg,
      sceneBoundaryTrustAvg,
      temporalContinuityScore,
      opticalFlowStability,
      rhythmStability,
      totalFrames: this.frames.length,
      totalTransitions: Math.max(0, this.frames.length - 1),
      totalSceneBoundaries: sceneBoundaryCount,
      anomalyCount,
      safetyStatus,
    };
  }
}
