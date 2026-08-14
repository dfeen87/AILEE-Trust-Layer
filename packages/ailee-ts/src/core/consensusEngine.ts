//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { ConsensusStrategy, ModelOutput, TrustScore } from "./types.js";

export interface ConsensusResult {
  achieved: boolean;
  value: number;
  confidence: number;
  strategyUsed: ConsensusStrategy;
  reasons: string[];
}

export class ConsensusEngine {
  private strategy: ConsensusStrategy;
  private threshold: number;

  constructor(strategy: ConsensusStrategy = ConsensusStrategy.HighestTrust, threshold = 0.75) {
    this.strategy = strategy;
    this.threshold = threshold;
  }

  public reachConsensus(
    primaryOutput: ModelOutput,
    primaryScore: TrustScore,
    peerOutputs: ModelOutput[] = [],
    peerScores: TrustScore[] = []
  ): ConsensusResult {
    const allOutputs = [primaryOutput, ...peerOutputs];
    const allScores = [primaryScore, ...peerScores];

    if (allOutputs.length === 1) {
      const achieved = primaryScore.aggregateScore >= this.threshold;
      return {
        achieved,
        value: primaryOutput.value,
        confidence: primaryOutput.confidence,
        strategyUsed: this.strategy,
        reasons: [achieved ? "Single output passed threshold." : "Single output below threshold."],
      };
    }

    switch (this.strategy) {
      case ConsensusStrategy.HighestTrust: {
        let bestIdx = 0;
        let maxScore = -1;
        for (let i = 0; i < allScores.length; i++) {
          if (allScores[i].aggregateScore > maxScore) {
            maxScore = allScores[i].aggregateScore;
            bestIdx = i;
          }
        }
        const achieved = maxScore >= this.threshold;
        return {
          achieved,
          value: allOutputs[bestIdx].value,
          confidence: allOutputs[bestIdx].confidence,
          strategyUsed: ConsensusStrategy.HighestTrust,
          reasons: [`Selected candidate ${allOutputs[bestIdx].id} with score ${maxScore.toFixed(3)}`],
        };
      }

      case ConsensusStrategy.MajorityVote: {
        const counts = new Map<number, number>();
        for (const out of allOutputs) {
          counts.set(out.value, (counts.get(out.value) || 0) + 1);
        }
        let topVal = primaryOutput.value;
        let maxCount = 0;
        for (const [val, count] of counts.entries()) {
          if (count > maxCount) {
            maxCount = count;
            topVal = val;
          }
        }
        const achieved = maxCount / allOutputs.length >= 0.5;
        return {
          achieved,
          value: topVal,
          confidence: primaryOutput.confidence,
          strategyUsed: ConsensusStrategy.MajorityVote,
          reasons: [`Majority agreement (${maxCount}/${allOutputs.length}) for value ${topVal}`],
        };
      }

      case ConsensusStrategy.WeightedCombination: {
        let weightSum = 0;
        let weightedValueSum = 0;
        for (let i = 0; i < allOutputs.length; i++) {
          const w = allScores[i].aggregateScore;
          weightSum += w;
          weightedValueSum += allOutputs[i].value * w;
        }
        const finalVal = weightSum > 0 ? weightedValueSum / weightSum : primaryOutput.value;
        const avgScore = weightSum / allOutputs.length;
        const achieved = avgScore >= this.threshold;
        return {
          achieved,
          value: finalVal,
          confidence: avgScore,
          strategyUsed: ConsensusStrategy.WeightedCombination,
          reasons: [`Weighted combination calculated over ${allOutputs.length} inputs`],
        };
      }

      case ConsensusStrategy.Synthesize:
      default: {
        const validValues = allOutputs.map((o) => o.value).sort((a, b) => a - b);
        const median = validValues[Math.floor(validValues.length / 2)];
        const achieved = primaryScore.aggregateScore >= this.threshold;
        return {
          achieved,
          value: median,
          confidence: primaryScore.confidence,
          strategyUsed: ConsensusStrategy.Synthesize,
          reasons: [`Synthesized median value ${median}`],
        };
      }
    }
  }
}
