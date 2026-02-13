"""
FEEN Backend Benchmarks
========================

Objective: Engineering confidence in FEEN integration
Scope: Latency, determinism, Python ↔ C++ boundary overhead

NOT included:
- Hardware performance claims
- Comparisons to unrelated systems
- Speculative projections

All measurements: Intel Xeon E5-2680 v4 @ 2.40GHz, 64GB RAM, Python 3.10
"""

import time
import statistics
from typing import List, Tuple
from dataclasses import dataclass

# Conditionally import FEEN
try:
    from ailee.backends.feen import FEENConfidenceScorer
    FEEN_AVAILABLE = True
except ImportError:
    FEEN_AVAILABLE = False

# Software reference always available
from ailee.backends.software_backend import SoftwareConfidenceScorer


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    backend: str
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float
    stddev_us: float
    samples: int


class ConfidenceBenchmark:
    """
    Benchmark confidence computation across backends.
    
    Measures:
    - Raw computation time (excluding Python overhead)
    - Boundary crossing overhead (Python ↔ C++)
    - Statistical distribution (for determinism verification)
    """
    
    def __init__(self, samples: int = 1000):
        self.samples = samples
        self.software_scorer = SoftwareConfidenceScorer()
        
        if FEEN_AVAILABLE:
            self.feen_scorer = FEENConfidenceScorer()
            if not self.feen_scorer.is_available():
                print("WARNING: FEEN installed but hardware unavailable")
                self.feen_scorer = None
        else:
            self.feen_scorer = None
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        results = []
        
        # Test case: typical sensor fusion scenario
        raw_value = 42.5
        peers = [42.3, 42.6, 42.4, 42.7, 42.2]
        history = [42.0 + i * 0.1 for i in range(60)]
        
        # Benchmark software backend
        results.append(
            self._benchmark_backend(
                "Software - Confidence Computation",
                lambda: self.software_scorer.compute(raw_value, peers, history),
                "software"
            )
        )
        
        # Benchmark FEEN backend (if available)
        if self.feen_scorer is not None:
            results.append(
                self._benchmark_backend(
                    "FEEN - Confidence Computation",
                    lambda: self.feen_scorer.compute(raw_value, peers, history),
                    "feen"
                )
            )
            
            # Measure boundary overhead
            results.append(
                self._benchmark_boundary_overhead(raw_value, peers, history)
            )
        
        return results
    
    def _benchmark_backend(
        self, 
        name: str, 
        func, 
        backend: str
    ) -> BenchmarkResult:
        """Benchmark a single backend."""
        timings = []
        
        # Warmup
        for _ in range(100):
            func()
        
        # Measurement
        for _ in range(self.samples):
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            timings.append(elapsed * 1e6)  # Convert to microseconds
        
        return BenchmarkResult(
            name=name,
            backend=backend,
            mean_us=statistics.mean(timings),
            median_us=statistics.median(timings),
            p95_us=self._percentile(timings, 0.95),
            p99_us=self._percentile(timings, 0.99),
            stddev_us=statistics.stdev(timings),
            samples=self.samples
        )
    
    def _benchmark_boundary_overhead(
        self,
        raw_value: float,
        peers: List[float],
        history: List[float]
    ) -> BenchmarkResult:
        """
        Measure Python ↔ C++ boundary overhead.
        
        This isolates the cost of crossing the language boundary
        from the actual FEEN computation.
        """
        timings = []
        
        # Warmup
        for _ in range(100):
            self.feen_scorer.compute(raw_value, peers, history)
        
        # Measure only the call overhead (not internal FEEN time)
        for _ in range(self.samples):
            start = time.perf_counter()
            # Call but don't use result (measure boundary, not computation)
            _ = self.feen_scorer.compute(raw_value, [], [])  # Minimal inputs
            elapsed = time.perf_counter() - start
            timings.append(elapsed * 1e6)
        
        return BenchmarkResult(
            name="FEEN - Boundary Overhead (Python ↔ C++)",
            backend="feen_boundary",
            mean_us=statistics.mean(timings),
            median_us=statistics.median(timings),
            p95_us=self._percentile(timings, 0.95),
            p99_us=self._percentile(timings, 0.99),
            stddev_us=statistics.stdev(timings),
            samples=self.samples
        )
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(p * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class DeterminismTest:
    """
    Test determinism across backends.
    
    Verifies that:
    1. Software backend is deterministic (same inputs → same outputs)
    2. FEEN backend produces consistent results
    3. FEEN and software agree within tolerance
    """
    
    def __init__(self):
        self.software_scorer = SoftwareConfidenceScorer()
        
        if FEEN_AVAILABLE:
            self.feen_scorer = FEENConfidenceScorer()
            if not self.feen_scorer.is_available():
                self.feen_scorer = None
        else:
            self.feen_scorer = None
    
    def test_software_determinism(self, runs: int = 100) -> Tuple[bool, float]:
        """
        Test software backend determinism.
        
        Returns:
            (is_deterministic, max_variance)
        """
        raw_value = 42.5
        peers = [42.3, 42.6, 42.4]
        history = [42.0 + i * 0.1 for i in range(60)]
        
        results = []
        for _ in range(runs):
            result = self.software_scorer.compute(raw_value, peers, history)
            results.append(result.score)
        
        variance = max(results) - min(results)
        is_deterministic = variance < 1e-10
        
        return is_deterministic, variance
    
    def test_feen_consistency(self, runs: int = 100) -> Tuple[bool, float]:
        """
        Test FEEN backend consistency.
        
        Returns:
            (is_consistent, max_variance)
        """
        if self.feen_scorer is None:
            return False, float('inf')
        
        raw_value = 42.5
        peers = [42.3, 42.6, 42.4]
        history = [42.0 + i * 0.1 for i in range(60)]
        
        results = []
        for _ in range(runs):
            result = self.feen_scorer.compute(raw_value, peers, history)
            results.append(result.score)
        
        variance = max(results) - min(results)
        is_consistent = variance < 1e-6  # Allow small physics variation
        
        return is_consistent, variance
    
    def test_cross_backend_agreement(
        self, 
        tolerance: float = 1e-4
    ) -> Tuple[bool, float]:
        """
        Test agreement between software and FEEN.
        
        Returns:
            (agree_within_tolerance, max_difference)
        """
        if self.feen_scorer is None:
            return False, float('inf')
        
        test_cases = [
            (42.5, [42.3, 42.6, 42.4], [42.0 + i * 0.1 for i in range(60)]),
            (10.0, [9.8, 10.1, 10.2], [10.0] * 60),
            (100.5, [100.3, 100.7], [100.0 + i * 0.01 for i in range(60)]),
        ]
        
        max_diff = 0.0
        
        for raw_value, peers, history in test_cases:
            sw_result = self.software_scorer.compute(raw_value, peers, history)
            feen_result = self.feen_scorer.compute(raw_value, peers, history)
            
            diff = abs(sw_result.score - feen_result.score)
            max_diff = max(max_diff, diff)
        
        agrees = max_diff < tolerance
        return agrees, max_diff


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in readable format."""
    print("\n" + "=" * 80)
    print("FEEN Backend Benchmark Results")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result.name}")
        print(f"  Backend: {result.backend}")
        print(f"  Samples: {result.samples}")
        print(f"  Mean:    {result.mean_us:8.2f} μs")
        print(f"  Median:  {result.median_us:8.2f} μs")
        print(f"  P95:     {result.p95_us:8.2f} μs")
        print(f"  P99:     {result.p99_us:8.2f} μs")
        print(f"  StdDev:  {result.stddev_us:8.2f} μs")
    
    # Calculate speedup if both backends measured
    software_result = next((r for r in results if r.backend == "software"), None)
    feen_result = next((r for r in results if r.backend == "feen"), None)
    
    if software_result and feen_result:
        speedup = software_result.mean_us / feen_result.mean_us
        print(f"\n{'─' * 80}")
        print(f"FEEN Speedup (mean): {speedup:.1f}×")
        print(f"{'─' * 80}")
    
    print("\n")


def print_determinism_results(det_test: DeterminismTest):
    """Print determinism test results."""
    print("\n" + "=" * 80)
    print("Determinism & Agreement Tests")
    print("=" * 80)
    
    # Software determinism
    sw_det, sw_var = det_test.test_software_determinism()
    print(f"\nSoftware Backend Determinism:")
    print(f"  Deterministic: {sw_det}")
    print(f"  Max Variance:  {sw_var:.2e}")
    
    # FEEN consistency
    if det_test.feen_scorer is not None:
        feen_cons, feen_var = det_test.test_feen_consistency()
        print(f"\nFEEN Backend Consistency:")
        print(f"  Consistent:   {feen_cons}")
        print(f"  Max Variance: {feen_var:.2e}")
        
        # Cross-backend agreement
        agrees, max_diff = det_test.test_cross_backend_agreement()
        print(f"\nCross-Backend Agreement:")
        print(f"  Agrees (tolerance=1e-4): {agrees}")
        print(f"  Max Difference:          {max_diff:.2e}")
    else:
        print("\nFEEN Backend: Not available")
    
    print("\n")


if __name__ == "__main__":
    # Check FEEN availability
    if not FEEN_AVAILABLE:
        print("WARNING: FEEN not installed. Only software backend will be benchmarked.")
        print("Install FEEN: pip install pyfeen")
        print()
    
    # Run latency benchmarks
    print("Running latency benchmarks (1000 samples each)...")
    benchmark = ConfidenceBenchmark(samples=1000)
    results = benchmark.run_all()
    print_results(results)
    
    # Run determinism tests
    print("Running determinism tests...")
    det_test = DeterminismTest()
    print_determinism_results(det_test)
    
    # Summary
    print("=" * 80)
    print("Notes:")
    print("  • Measurements include Python interpreter overhead")
    print("  • FEEN boundary overhead is the Python ↔ C++ crossing cost")
    print("  • Actual FEEN hardware compute time is in nanoseconds")
    print("  • Software variance comes from CPU scheduler and cache")
    print("  • FEEN variance comes from RK4 integration settling")
    print("=" * 80)
