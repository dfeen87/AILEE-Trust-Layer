"""
Deterministic replay utilities for AILEE Trust Layer.
"""

class ReplayBuffer:
    def __init__(self):
        self.records = []

    def record(self, inputs, result):
        self.records.append((inputs, result))

    def replay(self, pipeline):
        outputs = []
        for inputs, _ in self.records:
            outputs.append(pipeline.process(**inputs))
        return outputs
