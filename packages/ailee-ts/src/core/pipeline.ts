//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import {
  AileeConfig,
  ConsensusStrategy,
  DecisionResult,
  DEFAULT_CONFIG,
  ModelOutput,
  SafetyStatus,
  TrustScore,
} from "./types.js";
import { TrustScorer } from "./trustScorer.js";
import { GraceLayer } from "./graceLayer.js";
import { ConsensusEngine } from "./consensusEngine.js";
import { FallbackEngine } from "./fallbackEngine.js";

export class AileeTrustPipeline {
  private config: AileeConfig;
  private trustScorer: TrustScorer;
  private graceLayer: GraceLayer;
  private consensusEngine: ConsensusEngine;
  private fallbackEngine: FallbackEngine;

  constructor(config: Partial<AileeConfig> = {}, consensusStrategy: ConsensusStrategy = ConsensusStrategy.HighestTrust) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    const finiteFields = [
      this.config.borderlineLow,
      this.config.borderlineHigh,
      this.config.decayRate,
      this.config.consensusThreshold,
    ];
    if (
      !finiteFields.every(Number.isFinite) ||
      this.config.borderlineLow < 0 ||
      this.config.borderlineHigh > 1 ||
      this.config.borderlineLow > this.config.borderlineHigh ||
      this.config.consensusThreshold < 0 ||
      this.config.consensusThreshold > 1 ||
      !Number.isInteger(this.config.historyCapacity) ||
      this.config.historyCapacity < 1 ||
      Number.isNaN(this.config.hardMin) ||
      Number.isNaN(this.config.hardMax) ||
      this.config.hardMin > this.config.hardMax ||
      (this.config.defaultFallbackValue !== undefined && !Number.isFinite(this.config.defaultFallbackValue))
    ) {
      throw new RangeError("Invalid AILEE pipeline configuration");
    }
    this.trustScorer = new TrustScorer();
    this.graceLayer = new GraceLayer();
    this.consensusEngine = new ConsensusEngine(consensusStrategy, this.config.consensusThreshold);
    this.fallbackEngine = new FallbackEngine(this.config.historyCapacity);
  }

  public process(
    rawValue: number,
    rawConfidence: number,
    peerValues: number[] = [],
    context?: Record<string, unknown>
  ): DecisionResult {
    const reasons: string[] = [];
    const timestamp = Date.now();
    const decisionId = `dec_${timestamp}_${Math.random().toString(36).substring(2, 8)}`;

    const inputReasons: string[] = [];
    if (!Number.isFinite(rawValue)) inputReasons.push("Raw value must be a finite number.");
    if (!Number.isFinite(rawConfidence) || rawConfidence < 0 || rawConfidence > 1) {
      inputReasons.push("Raw confidence must be a finite number between 0 and 1.");
    }
    if (!Array.isArray(peerValues) || peerValues.some((value) => !Number.isFinite(value))) {
      inputReasons.push("Peer values must contain only finite numbers.");
    }
    if (inputReasons.length > 0) {
      const fallback = this.fallbackEngine.enforceBounds(
        this.fallbackEngine.getFallbackValue(this.config),
        this.config
      ).boundedValue;
      return {
        decisionId,
        value: fallback,
        safetyStatus: "OUTRIGHT_REJECTED",
        usedFallback: true,
        reasons: [...inputReasons, `Fallback triggered. Value assigned: ${fallback}`],
        trustScore: { confidence: 0, safety: 0, consistency: 0, determinism: 0, aggregateScore: 0 },
        consensusAchieved: false,
        context,
        timestamp,
      };
    }

    const primaryOutput: ModelOutput = {
      id: "primary",
      value: rawValue,
      confidence: rawConfidence,
      metadata: context,
    };

    const peerOutputs: ModelOutput[] = peerValues.map((val, idx) => ({
      id: `peer_${idx}`,
      value: val,
      confidence: rawConfidence,
    }));

    const history = this.fallbackEngine.getHistory();

    let trustScore: TrustScore = this.trustScorer.scoreOutput(primaryOutput, history, peerValues);
    const peerScores = peerOutputs.map((p) => this.trustScorer.scoreOutput(p, history, peerValues));

    let safetyStatus: SafetyStatus = "ACCEPTED";
    let finalValue = rawValue;
    let usedFallback = false;

    if (trustScore.aggregateScore < this.config.borderlineLow) {
      safetyStatus = "OUTRIGHT_REJECTED";
      reasons.push(`Score ${trustScore.aggregateScore.toFixed(3)} below borderline low threshold (${this.config.borderlineLow}).`);
    } else if (trustScore.aggregateScore < this.config.borderlineHigh) {
      safetyStatus = "BORDERLINE";
      reasons.push(`Score ${trustScore.aggregateScore.toFixed(3)} in borderline range.`);

      const graceRes = this.graceLayer.evaluate(rawValue, trustScore, history, context);
      reasons.push(graceRes.reason);

      if (graceRes.passed) {
        trustScore.aggregateScore = Math.min(1.0, trustScore.aggregateScore + graceRes.scoreAdjustment);
        safetyStatus = "ACCEPTED";
      } else {
        safetyStatus = "OUTRIGHT_REJECTED";
      }
    } else {
      reasons.push(`Score ${trustScore.aggregateScore.toFixed(3)} meets acceptance threshold.`);
    }

    const consensusRes = this.consensusEngine.reachConsensus(primaryOutput, trustScore, peerOutputs, peerScores);
    reasons.push(...consensusRes.reasons);

    if (safetyStatus === "ACCEPTED") {
      finalValue = consensusRes.value;
    }

    if (safetyStatus === "OUTRIGHT_REJECTED") {
      usedFallback = true;
      finalValue = this.fallbackEngine.getFallbackValue(this.config);
      reasons.push(`Fallback triggered. Value assigned: ${finalValue}`);
    }

    const { boundedValue, constrained } = this.fallbackEngine.enforceBounds(finalValue, this.config);
    if (constrained) {
      reasons.push(`Value ${finalValue} constrained by hard bounds [${this.config.hardMin}, ${this.config.hardMax}] to ${boundedValue}.`);
      finalValue = boundedValue;
    }

    if (!usedFallback) {
      this.fallbackEngine.recordValue(finalValue);
    }

    return {
      decisionId,
      value: finalValue,
      safetyStatus,
      usedFallback,
      reasons,
      trustScore,
      consensusAchieved: consensusRes.achieved,
      context,
      timestamp,
    };
  }
}
