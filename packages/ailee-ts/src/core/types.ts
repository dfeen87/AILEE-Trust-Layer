//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export type SafetyStatus = "ACCEPTED" | "BORDERLINE" | "OUTRIGHT_REJECTED";

export enum ConsensusStrategy {
  HighestTrust = "HighestTrust",
  MajorityVote = "MajorityVote",
  Synthesize = "Synthesize",
  WeightedCombination = "WeightedCombination",
}

export interface TrustScore {
  confidence: number;
  safety: number;
  consistency: number;
  determinism: number;
  aggregateScore: number;
}

export interface ModelOutput {
  id: string;
  value: number;
  confidence: number;
  metadata?: Record<string, unknown>;
}

export interface AileeConfig {
  borderlineLow: number;
  borderlineHigh: number;
  hardMin: number;
  hardMax: number;
  decayRate: number;
  historyCapacity: number;
  consensusThreshold: number;
  defaultFallbackValue?: number;
}

/**
 * Optional V8.1 confidence refinement settings. The layer is disabled by
 * default so existing V8 pipeline callers retain their exact confidence path.
 */
export interface CalibrationConfig {
  enabled: boolean;
  acceptanceThreshold: number;
  uncertaintyBand: number;
  maxGraceMargin: number;
  consensusThreshold: number;
  minimumPeerCount: number;
}

export interface CalibrationMetadata {
  graceMargin?: number;
  peerConsensus?: {
    agreement: number;
    peerCount: number;
  };
}

export interface CalibrationResult {
  confidence: number;
  applied: boolean;
  fallbackUsed: boolean;
  event: "DISABLED" | "OUTSIDE_UNCERTAINTY_BAND" | "GRACE_APPLIED" | "CONSENSUS_NOT_MET" | "INVALID_INPUT";
  thresholdDecision: "ABOVE_THRESHOLD" | "UNCERTAINTY_ZONE" | "BELOW_UNCERTAINTY_ZONE" | "BASELINE";
  consensusChecked: boolean;
  reasons: string[];
}

export const DEFAULT_CALIBRATION_CONFIG: CalibrationConfig = {
  enabled: false,
  acceptanceThreshold: 0.95,
  uncertaintyBand: 0.05,
  maxGraceMargin: 0.02,
  consensusThreshold: 0.8,
  minimumPeerCount: 2,
};

export const DEFAULT_CONFIG: AileeConfig = {
  borderlineLow: 0.7,
  borderlineHigh: 0.9,
  hardMin: -Infinity,
  hardMax: Infinity,
  decayRate: 0.05,
  historyCapacity: 100,
  consensusThreshold: 0.75,
};

export interface DecisionResult {
  decisionId: string;
  value: number;
  safetyStatus: SafetyStatus;
  usedFallback: boolean;
  reasons: string[];
  trustScore: TrustScore;
  consensusAchieved: boolean;
  context?: Record<string, unknown>;
  timestamp: number;
  lineageHash?: string;
}

export interface GraceEvaluationResult {
  passed: boolean;
  scoreAdjustment: number;
  reason: string;
}
