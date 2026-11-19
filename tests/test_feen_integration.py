#!/usr/bin/env python
# Licensed under the PolyForm Noncommercial License 1.0.0
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

import pytest

from ailee import AileeConfig, SafetyStatus
from ailee.backends import SoftwareBackend, FeenBackend
from ailee.backends.feen import FEENConfidenceScorer

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


@pytest.mark.parametrize(
    "field, value, error_type",
    [
        ("raw_value", True, TypeError),
        ("raw_value", "invalid", TypeError),
        ("raw_value", float("nan"), ValueError),
        ("raw_confidence", float("inf"), ValueError),
        ("raw_confidence", False, TypeError),
        ("peer_values", [0.5, float("-inf")], ValueError),
        ("peer_values", ["invalid"], TypeError),
        ("timestamp", float("nan"), ValueError),
        ("timestamp", "invalid", TypeError),
    ],
)
@pytest.mark.parametrize("fallback_to_software", [True, False])
def test_feen_and_software_reject_invalid_numeric_inputs(
    field, value, error_type, fallback_to_software
):
    config = AileeConfig(hard_min=0.0, hard_max=1.0)
    inputs = dict(raw_value=0.5, raw_confidence=0.99, timestamp=1.0)
    inputs[field] = value
    failures = []
    for backend in (
        SoftwareBackend(config),
        FeenBackend(config, fallback_to_software=fallback_to_software),
    ):
        with pytest.raises(error_type) as error:
            backend.process(**inputs)
        failures.append(str(error.value))
    assert failures[0] == failures[1]


@pytest.mark.parametrize("raw_value, limit", [(-0.25, 0.0), (1.25, 1.0)])
def test_feen_enforces_hard_envelope_before_scoring(raw_value, limit):
    config = AileeConfig(hard_min=0.0, hard_max=1.0)
    inputs = dict(raw_value=raw_value, raw_confidence=1.0, timestamp=1.0)
    expected = SoftwareBackend(config).process(**inputs)
    mediated = FeenBackend(config).process(**inputs)
    unmediated = FeenBackend(config, fallback_to_software=False).process(**inputs)

    for result in (mediated, unmediated):
        assert result.safety_status is SafetyStatus.OUTRIGHT_REJECTED
        assert result.used_fallback
    assert mediated.value == expected.value == 0.5
    assert unmediated.value == limit


def test_example_file():
    """Test that the example file can be executed."""
    import subprocess
    
    # Get project root (parent of tests directory)
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Test from root directory
    result = subprocess.run(
        [sys.executable, 'examples/feen_vs_software.py'],
        cwd=project_root,
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
        cwd=os.path.join(project_root, 'examples'),
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
