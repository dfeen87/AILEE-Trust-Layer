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
