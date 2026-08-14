//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { AileeConfig } from "./types.js";

export class FallbackEngine {
  private history: number[] = [];
  private capacity: number;

  constructor(capacity = 100) {
    this.capacity = capacity;
  }

  public recordValue(val: number): void {
    if (isNaN(val) || !isFinite(val)) return;
    this.history.push(val);
    if (this.history.length > this.capacity) {
      this.history.shift();
    }
  }

  public getFallbackValue(config: AileeConfig): number {
    if (config.defaultFallbackValue !== undefined) {
      return config.defaultFallbackValue;
    }

    if (this.history.length === 0) {
      return 0.0;
    }

    const sorted = [...this.history].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
    return sorted[mid];
  }

  public getHistory(): number[] {
    return [...this.history];
  }

  public enforceBounds(value: number, config: AileeConfig): { boundedValue: number; constrained: boolean } {
    let constrained = false;
    let boundedValue = value;

    if (boundedValue < config.hardMin) {
      boundedValue = config.hardMin;
      constrained = true;
    } else if (boundedValue > config.hardMax) {
      boundedValue = config.hardMax;
      constrained = true;
    }

    return { boundedValue, constrained };
  }
}
