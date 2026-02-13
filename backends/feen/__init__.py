"""
FEEN Backend for AILEE Trust Layer
===================================

Optional hardware acceleration using phononic computing.

FEEN provides physics-native confidence computation with:
- Ultra-low latency (< 1 μs)
- Ultra-low power (< 100 μW)
- Deterministic execution

AILEE retains full control over policy and trust decisions.

Usage:
    from ailee.backends.feen import FEENConfidenceScorer
    
    scorer = FEENConfidenceScorer()
    
    if scorer.is_available():
        result = scorer.compute(value, peers, history)
    else:
        # Automatic fallback to software
        pass

For integration details, see: backends/feen/INTEGRATION.md
For benchmarks, run: python backends/feen/benchmarks.py
"""

from .confidence_scorer import (
    FEENConfidenceScorer,
    ConfidenceResult,
    create_feen_scorer,
)

__all__ = [
    'FEENConfidenceScorer',
    'ConfidenceResult',
    'create_feen_scorer',
]

__version__ = '2.0.0'
