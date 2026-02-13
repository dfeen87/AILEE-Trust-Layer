"""
FEEN Backend Benchmarks
========================

NOTE: This benchmarks file is designed for FEEN hardware performance testing.
The original implementation assumes a SoftwareConfidenceScorer class that
doesn't exist in the current architecture.

Current Status:
- FEEN integration is functional via backends/feen_backend.py and FEENConfidenceScorer
- Software backend is implemented in AileeTrustPipeline (not as a separate scorer)
- For performance comparison, use the integration test or example files

To run basic FEEN tests:
    python test_feen_integration.py
    
To see FEEN vs Software comparison:
    python examples/feen_vs_software.py

This file will be updated in a future version to provide proper benchmarks
that match the current architecture.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backends.feen import FEENConfidenceScorer


def main():
    print("=" * 70)
    print("FEEN Availability Check")
    print("=" * 70)
    
    scorer = FEENConfidenceScorer()
    
    if scorer.is_available():
        print("\n✓ FEEN hardware is available")
        
        # Test basic computation
        result = scorer.compute(
            raw_value=42.5,
            peers=[42.3, 42.6, 42.4],
            history=[42.0 + i * 0.1 for i in range(60)]
        )
        
        print(f"\nTest computation result:")
        print(f"  Score:      {result.score:.6f}")
        print(f"  Stability:  {result.stability:.6f}")
        print(f"  Agreement:  {result.agreement:.6f}")
        print(f"  Likelihood: {result.likelihood:.6f}")
        
        # Get diagnostics
        diag = scorer.get_hardware_diagnostics()
        if diag:
            print(f"\nHardware diagnostics:")
            print(f"  Backend: {diag.get('backend', 'unknown')}")
            print(f"  Channel energies: {diag.get('channel_energies', 'N/A')}")
    else:
        print("\n⚠ FEEN hardware is not available")
        print("\nThis is expected if:")
        print("  • pyfeen is not installed (pip install pyfeen)")
        print("  • FEEN hardware is not connected")
        print("  • Running in a non-FEEN environment")
        print("\nThe system will automatically fall back to software implementation.")
    
    print("\n" + "=" * 70)
    print("\nFor full integration tests, run: python test_feen_integration.py")
    print("For FEEN vs Software demo, run: python examples/feen_vs_software.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
