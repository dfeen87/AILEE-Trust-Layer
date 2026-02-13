#!/usr/bin/env python
"""
FEEN Integration Test
=====================

This test verifies that the FEEN integration works correctly:
1. FEEN backend can be imported and instantiated
2. FEEN confidence scorer can be imported and used
3. Both backends produce valid results
4. The example file runs successfully
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ailee_trust_pipeline_v1 import AileeConfig
from backends import SoftwareBackend, FeenBackend
from backends.feen import FEENConfidenceScorer

# Constants
EXAMPLE_TIMEOUT = 10  # seconds


def test_backend_imports():
    """Test that backends can be imported."""
    print("✓ Backend imports successful")
    return True


def test_backend_instantiation():
    """Test that backends can be instantiated."""
    cfg = AileeConfig(hard_min=0.0, hard_max=100.0, consensus_quorum=3)
    
    software = SoftwareBackend(cfg)
    assert software is not None, "SoftwareBackend instantiation failed"
    
    feen = FeenBackend(cfg)
    assert feen is not None, "FeenBackend instantiation failed"
    
    print("✓ Backend instantiation successful")
    return True


def test_feen_confidence_scorer():
    """Test FEEN confidence scorer."""
    scorer = FEENConfidenceScorer()
    
    # Check availability
    is_available = scorer.is_available()
    print(f"  FEEN hardware available: {is_available}")
    
    if is_available:
        # Test computation
        result = scorer.compute(
            raw_value=10.2,
            peers=[10.0, 10.1, 10.3],
            history=[10.0] * 60
        )
        assert result is not None, "FEEN compute returned None"
        assert hasattr(result, 'score'), "FEEN result missing score"
        assert 0.0 <= result.score <= 1.0, f"FEEN score out of range: {result.score}"
        print(f"  FEEN compute result: score={result.score:.3f}")
    else:
        print("  FEEN hardware not available - using software fallback (expected)")
    
    print("✓ FEEN confidence scorer test passed")
    return True


def test_backend_processing():
    """Test that both backends can process inputs."""
    cfg = AileeConfig(hard_min=0.0, hard_max=100.0, consensus_quorum=3)
    
    software = SoftwareBackend(cfg)
    feen = FeenBackend(cfg)
    
    inputs = dict(
        raw_value=10.2,
        raw_confidence=0.92,
        peer_values=[10.0, 10.1, 10.3],
        context={"feature": "test"},
    )
    
    # Test software backend
    result_sw = software.process(**inputs)
    assert result_sw is not None, "Software backend returned None"
    assert hasattr(result_sw, 'value'), "Software result missing value"
    assert hasattr(result_sw, 'safety_status'), "Software result missing safety_status"
    print(f"  Software: value={result_sw.value}, status={result_sw.safety_status}")
    
    # Test FEEN backend
    result_feen = feen.process(**inputs)
    assert result_feen is not None, "FEEN backend returned None"
    assert hasattr(result_feen, 'value'), "FEEN result missing value"
    assert hasattr(result_feen, 'safety_status'), "FEEN result missing safety_status"
    print(f"  FEEN: value={result_feen.value}, status={result_feen.safety_status}")
    
    # Check metadata
    assert 'backend' in result_feen.metadata, "FEEN result missing backend in metadata"
    assert result_feen.metadata['backend'] == 'feen', "FEEN backend metadata incorrect"
    
    print("✓ Backend processing test passed")
    return True


def test_example_file():
    """Test that the example file can be executed."""
    import subprocess
    
    # Test from root directory
    result = subprocess.run(
        [sys.executable, 'examples/feen_vs_software.py'],
        cwd=os.path.dirname(__file__),
        capture_output=True,
        text=True,
        timeout=EXAMPLE_TIMEOUT
    )
    
    assert result.returncode == 0, f"Example failed from root: {result.stderr}"
    assert 'Software:' in result.stdout, "Example missing software output"
    assert 'FEEN:' in result.stdout, "Example missing FEEN output"
    print("  ✓ Example runs from root directory")
    
    # Test from examples directory
    result = subprocess.run(
        [sys.executable, 'feen_vs_software.py'],
        cwd=os.path.join(os.path.dirname(__file__), 'examples'),
        capture_output=True,
        text=True,
        timeout=EXAMPLE_TIMEOUT
    )
    
    assert result.returncode == 0, f"Example failed from examples dir: {result.stderr}"
    assert 'Software:' in result.stdout, "Example missing software output"
    assert 'FEEN:' in result.stdout, "Example missing FEEN output"
    print("  ✓ Example runs from examples directory")
    
    print("✓ Example file test passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FEEN Integration Test Suite")
    print("=" * 70 + "\n")
    
    tests = [
        ("Backend Imports", test_backend_imports),
        ("Backend Instantiation", test_backend_instantiation),
        ("FEEN Confidence Scorer", test_feen_confidence_scorer),
        ("Backend Processing", test_backend_processing),
        ("Example File Execution", test_example_file),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        print("-" * 70)
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    
    if failed == 0:
        print("✓ ALL TESTS PASSED - FEEN integration is working correctly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - FEEN integration has issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
