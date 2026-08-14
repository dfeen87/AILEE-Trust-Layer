//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { ModelOutput, TrustScore } from "./types.js";

export class TrustScorer {
  private weights: { confidence: number; safety: number; consistency: number; determinism: number };

  constructor(weights = { confidence: 0.4, safety: 0.3, consistency: 0.15, determinism: 0.15 }) {
    this.weights = weights;
  }

  public scoreOutput(output: ModelOutput, history: number[] = [], peerValues: number[] = []): TrustScore {
    const confidence = Math.max(0, Math.min(1, output.confidence));

    let safety = 1.0;
    if (isNaN(output.value) || !isFinite(output.value)) {
      safety = 0.0;
    }

    let consistency = 1.0;
    if (history.length > 0) {
      const mean = history.reduce((a, b) => a + b, 0) / history.length;
      const stdDev = Math.sqrt(history.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / history.length) || 1.0;
      const zScore = Math.abs((output.value - mean) / stdDev);
      consistency = Math.max(0, 1 - zScore / 4.0);
    }

    let determinism = 1.0;
    if (peerValues.length > 0) {
      const peerMean = peerValues.reduce((a, b) => a + b, 0) / peerValues.length;
      const diff = Math.abs(output.value - peerMean);
      determinism = Math.max(0, 1 - diff / (Math.abs(peerMean) || 1.0));
    }

    const aggregateScore =
      confidence * this.weights.confidence +
      safety * this.weights.safety +
      consistency * this.weights.consistency +
      determinism * this.weights.determinism;

    return {
      confidence,
      safety,
      consistency,
      determinism,
      aggregateScore: Math.max(0, Math.min(1, aggregateScore)),
    };
  }
}
