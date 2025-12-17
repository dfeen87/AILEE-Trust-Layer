"""
AILEE Trust Layer — Monitoring Utilities
Non-intrusive observability helpers.
"""

from collections import deque


class TrustMonitor:
    def __init__(self, window=100):
        self.window = window
        self.confidence_scores = deque(maxlen=window)
        self.fallback_flags = deque(maxlen=window)

    def record(self, result):
        self.confidence_scores.append(result.confidence_score)
        self.fallback_flags.append(result.used_fallback)

    def fallback_rate(self):
        if not self.fallback_flags:
            return 0.0
        return sum(self.fallback_flags) / len(self.fallback_flags)

    def avg_confidence(self):
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
