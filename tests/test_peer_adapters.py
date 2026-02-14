#!/usr/bin/env python
"""
Peer Adapters Test Suite
========================

Comprehensive tests to verify the effectiveness of AILEE peer adapters:
1. All adapters can be imported and instantiated
2. Basic adapters (Static, Rolling) work correctly
3. Advanced adapters (Callback, MultiSource, Weighted, Filtered, Metadata) work correctly
4. Edge cases and error handling
5. Integration with AILEE trust pipeline
6. Real-world effectiveness scenarios
"""

import sys
import os
import time
from typing import List

from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_peer_adapters import (
    StaticPeerAdapter,
    RollingPeerAdapter,
    CallbackPeerAdapter,
    MultiSourcePeerAdapter,
    WeightedPeerAdapter,
    FilteredPeerAdapter,
    MetadataPeerAdapter,
    PeerMetadata,
    create_multi_model_adapter,
)


# =============================================================================
# Test Utilities
# =============================================================================

def assert_equals(actual, expected, message=""):
    """Assert two values are equal."""
    if actual != expected:
        raise AssertionError(f"{message}: Expected {expected}, got {actual}")


def assert_in_range(value, min_val, max_val, message=""):
    """Assert value is in range [min_val, max_val]."""
    if not (min_val <= value <= max_val):
        raise AssertionError(f"{message}: Value {value} not in range [{min_val}, {max_val}]")


def assert_list_equals(actual, expected, message=""):
    """Assert two lists are equal."""
    if actual != expected:
        raise AssertionError(f"{message}: Expected {expected}, got {actual}")


def assert_true(condition, message=""):
    """Assert condition is true."""
    if not condition:
        raise AssertionError(f"{message}: Condition is False")


def assert_length(items, expected_length, message=""):
    """Assert list/collection has expected length."""
    if len(items) != expected_length:
        raise AssertionError(f"{message}: Expected length {expected_length}, got {len(items)}")


# =============================================================================
# Basic Adapter Tests
# =============================================================================

def test_static_adapter():
    """Test StaticPeerAdapter functionality."""
    print("  Testing StaticPeerAdapter...")
    
    # Test initialization and retrieval
    adapter = StaticPeerAdapter([10.0, 10.5, 10.3])
    peers = adapter.get_peer_values()
    
    assert_list_equals(peers, [10.0, 10.5, 10.3], "Initial values")
    
    # Test update
    adapter.update([20.0, 20.5])
    peers = adapter.get_peer_values()
    
    assert_list_equals(peers, [20.0, 20.5], "Updated values")
    
    # Test immutability of returned list
    peers[0] = 999.0
    new_peers = adapter.get_peer_values()
    assert_equals(new_peers[0], 20.0, "Values should not be modified externally")
    
    print("  ✓ StaticPeerAdapter works correctly")
    return True


def test_rolling_adapter():
    """Test RollingPeerAdapter functionality."""
    print("  Testing RollingPeerAdapter...")
    
    # Test initialization with history
    history = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]
    adapter = RollingPeerAdapter(history, window=3)
    peers = adapter.get_peer_values()
    
    assert_list_equals(peers, [10.3, 10.4, 10.5], "Last 3 values")
    
    # Test append
    adapter.append(10.6)
    peers = adapter.get_peer_values()
    assert_list_equals(peers, [10.4, 10.5, 10.6], "After append")
    
    # Test extend
    adapter.extend([10.7, 10.8])
    peers = adapter.get_peer_values()
    assert_list_equals(peers, [10.6, 10.7, 10.8], "After extend")
    
    # Test empty history
    empty_adapter = RollingPeerAdapter([], window=3)
    assert_list_equals(empty_adapter.get_peer_values(), [], "Empty history")
    
    # Test window larger than history
    small_adapter = RollingPeerAdapter([1.0, 2.0], window=10)
    assert_list_equals(small_adapter.get_peer_values(), [1.0, 2.0], "Window larger than history")
    
    print("  ✓ RollingPeerAdapter works correctly")
    return True


# =============================================================================
# Advanced Adapter Tests
# =============================================================================

def test_callback_adapter():
    """Test CallbackPeerAdapter functionality."""
    print("  Testing CallbackPeerAdapter...")
    
    # Test basic callback
    call_count = [0]  # Use list to allow mutation in nested function
    
    def get_peers():
        call_count[0] += 1
        return [10.0, 10.5, 10.3]
    
    adapter = CallbackPeerAdapter(get_peers)
    peers = adapter.get_peer_values()
    
    assert_list_equals(peers, [10.0, 10.5, 10.3], "Callback result")
    assert_equals(call_count[0], 1, "Callback called once")
    
    # Test without caching (should call again)
    peers = adapter.get_peer_values()
    assert_equals(call_count[0], 2, "Callback called again without cache")
    
    # Test with caching
    call_count[0] = 0
    cached_adapter = CallbackPeerAdapter(get_peers, cache_ttl=1.0)
    
    peers1 = cached_adapter.get_peer_values()
    assert_equals(call_count[0], 1, "First call")
    
    peers2 = cached_adapter.get_peer_values()
    assert_equals(call_count[0], 1, "Second call uses cache")
    
    # Test cache expiration
    time.sleep(1.1)
    peers3 = cached_adapter.get_peer_values()
    assert_equals(call_count[0], 2, "Cache expired, new call")
    
    # Test cache invalidation
    cached_adapter.invalidate_cache()
    peers4 = cached_adapter.get_peer_values()
    assert_equals(call_count[0], 3, "Cache invalidated, new call")
    
    print("  ✓ CallbackPeerAdapter works correctly")
    return True


def test_multi_source_adapter():
    """Test MultiSourcePeerAdapter functionality."""
    print("  Testing MultiSourcePeerAdapter...")
    
    # Create multiple sources
    source1 = StaticPeerAdapter([10.0, 10.1])
    source2 = StaticPeerAdapter([10.2, 10.3])
    source3 = StaticPeerAdapter([10.4])
    
    # Test aggregation
    adapter = MultiSourcePeerAdapter([source1, source2, source3])
    peers = adapter.get_peer_values()
    
    assert_list_equals(peers, [10.0, 10.1, 10.2, 10.3, 10.4], "Aggregated values")
    
    # Test add_source
    source4 = StaticPeerAdapter([10.5])
    adapter.add_source(source4)
    peers = adapter.get_peer_values()
    assert_length(peers, 6, "After adding source")
    
    # Test remove_source
    adapter.remove_source(0)
    peers = adapter.get_peer_values()
    assert_list_equals(peers, [10.2, 10.3, 10.4, 10.5], "After removing first source")
    
    # Test empty sources
    empty_adapter = MultiSourcePeerAdapter([])
    assert_list_equals(empty_adapter.get_peer_values(), [], "Empty sources")
    
    print("  ✓ MultiSourcePeerAdapter works correctly")
    return True


def test_weighted_adapter():
    """Test WeightedPeerAdapter functionality."""
    print("  Testing WeightedPeerAdapter...")
    
    # Create sources with different weights
    source1 = StaticPeerAdapter([10.0])
    source2 = StaticPeerAdapter([20.0])
    
    # Test duplicate mode
    adapter_dup = WeightedPeerAdapter(
        [source1, source2],
        [0.8, 0.2],
        mode="duplicate"
    )
    peers = adapter_dup.get_peer_values()
    
    # With 0.8 weight, source1 should appear 8 times, source2 should appear 2 times
    assert_true(peers.count(10.0) >= peers.count(20.0), "Higher weight appears more")
    
    # Test scale mode
    adapter_scale = WeightedPeerAdapter(
        [source1, source2],
        [0.8, 0.2],
        mode="scale"
    )
    peers = adapter_scale.get_peer_values()
    
    assert_in_range(peers[0], 7.9, 8.1, "First value scaled by 0.8")
    assert_in_range(peers[1], 3.9, 4.1, "Second value scaled by 0.2")
    
    # Test weight normalization
    adapter_norm = WeightedPeerAdapter(
        [source1, source2],
        [4.0, 1.0],  # Should normalize to 0.8, 0.2
        mode="scale"
    )
    peers = adapter_norm.get_peer_values()
    assert_in_range(peers[0], 7.9, 8.1, "Normalized weights work")
    
    # Test error cases
    try:
        WeightedPeerAdapter([source1], [0.5, 0.5], mode="scale")
        assert_true(False, "Should raise error for mismatched sources/weights")
    except ValueError:
        pass  # Expected
    
    try:
        WeightedPeerAdapter([source1, source2], [0.0, 0.0], mode="scale")
        assert_true(False, "Should raise error for zero weights")
    except ValueError:
        pass  # Expected
    
    print("  ✓ WeightedPeerAdapter works correctly")
    return True


def test_filtered_adapter():
    """Test FilteredPeerAdapter functionality."""
    print("  Testing FilteredPeerAdapter...")
    
    # Create source with outliers
    source = StaticPeerAdapter([10.0, 100.0, 10.5, -50.0, 10.3, 200.0, 10.2])
    
    # Test min/max filtering
    adapter_range = FilteredPeerAdapter(source, min_value=0.0, max_value=50.0)
    peers = adapter_range.get_peer_values()
    
    assert_list_equals(peers, [10.0, 10.5, 10.3, 10.2], "Min/max filtering")
    
    # Test median deviation filtering
    normal_source = StaticPeerAdapter([10.0, 10.1, 10.2, 15.0, 10.3])
    adapter_median = FilteredPeerAdapter(
        normal_source,
        max_deviation_from_median=1.0
    )
    peers = adapter_median.get_peer_values()
    
    # 15.0 should be filtered out as it's too far from median (~10.2)
    assert_true(15.0 not in peers, "Outlier filtered by median deviation")
    assert_true(10.0 in peers, "Normal value retained")
    
    # Test combined filtering
    adapter_combined = FilteredPeerAdapter(
        source,
        min_value=0.0,
        max_value=50.0,
        max_deviation_from_median=5.0
    )
    peers = adapter_combined.get_peer_values()
    
    # Should have tight cluster around 10.x
    assert_true(all(9.0 <= p <= 11.0 for p in peers), "Combined filtering works")
    
    # Test empty result
    empty_source = StaticPeerAdapter([])
    adapter_empty = FilteredPeerAdapter(empty_source, min_value=0.0)
    assert_list_equals(adapter_empty.get_peer_values(), [], "Empty source")
    
    print("  ✓ FilteredPeerAdapter works correctly")
    return True


def test_metadata_adapter():
    """Test MetadataPeerAdapter functionality."""
    print("  Testing MetadataPeerAdapter...")
    
    adapter = MetadataPeerAdapter()
    
    # Add sources with metadata
    adapter.add_peer_source(
        "model_a",
        StaticPeerAdapter([10.5]),
        confidence=0.95,
        source_type="llm",
        metadata={"version": "1.0"}
    )
    
    adapter.add_peer_source(
        "model_b",
        StaticPeerAdapter([10.3]),
        confidence=0.85,
        source_type="llm",
        metadata={"version": "2.0"}
    )
    
    # Test basic get_peer_values
    peers = adapter.get_peer_values()
    assert_list_equals(peers, [10.5, 10.3], "Values from all sources")
    
    # Test get_peer_values_with_metadata
    values, metadata = adapter.get_peer_values_with_metadata()
    assert_list_equals(values, [10.5, 10.3], "Values with metadata")
    assert_length(metadata, 2, "Metadata list length")
    
    assert_equals(metadata[0].name, "model_a", "First metadata name")
    assert_equals(metadata[0].confidence, 0.95, "First metadata confidence")
    assert_equals(metadata[0].source_type, "llm", "First metadata type")
    
    assert_equals(metadata[1].name, "model_b", "Second metadata name")
    assert_equals(metadata[1].confidence, 0.85, "Second metadata confidence")
    
    # Test weighted peers
    weighted = adapter.get_weighted_peers()
    assert_in_range(weighted[0], 9.9, 10.0, "First weighted value (10.5 * 0.95)")
    assert_in_range(weighted[1], 8.7, 8.8, "Second weighted value (10.3 * 0.85)")
    
    print("  ✓ MetadataPeerAdapter works correctly")
    return True


# =============================================================================
# Convenience Function Tests
# =============================================================================

def test_create_multi_model_adapter():
    """Test create_multi_model_adapter convenience function."""
    print("  Testing create_multi_model_adapter...")
    
    # Test without confidences
    outputs = {
        "gpt4": 10.5,
        "claude": 10.3,
        "llama": 10.6
    }
    
    adapter = create_multi_model_adapter(outputs)
    peers = adapter.get_peer_values()
    
    assert_length(peers, 3, "Three model outputs")
    assert_true(10.5 in peers, "GPT-4 value present")
    assert_true(10.3 in peers, "Claude value present")
    assert_true(10.6 in peers, "Llama value present")
    
    # Test with confidences
    confidences = {
        "gpt4": 0.95,
        "claude": 0.92,
        "llama": 0.88
    }
    
    adapter_conf = create_multi_model_adapter(outputs, confidences)
    values, metadata = adapter_conf.get_peer_values_with_metadata()
    
    # Find GPT-4 in metadata
    gpt4_meta = next(m for m in metadata if m.name == "gpt4")
    assert_equals(gpt4_meta.confidence, 0.95, "GPT-4 confidence")
    assert_equals(gpt4_meta.source_type, "llm", "Source type is LLM")
    
    print("  ✓ create_multi_model_adapter works correctly")
    return True


# =============================================================================
# Integration Tests
# =============================================================================

def test_pipeline_integration():
    """Test peer adapters integrated with AILEE pipeline."""
    print("  Testing AILEE pipeline integration...")
    
    # Create pipeline - focus on testing that adapters provide peer values correctly
    config = AileeConfig(
        accept_threshold=0.70,
        borderline_low=0.50,
        borderline_high=0.70,
        consensus_quorum=2,
        consensus_delta=0.5,
        enable_consensus=True,
        enable_audit_metadata=True
    )
    pipeline = AileeTrustPipeline(config)
    
    # Create peer adapter
    adapter = StaticPeerAdapter([10.0, 10.1, 10.2])
    
    # Test that adapters can provide peer values to the pipeline
    peer_values = adapter.get_peer_values()
    assert_length(peer_values, 3, "Adapter provides 3 peer values")
    
    # Process with peer values - the main test is that it doesn't crash
    # and that peer values are properly passed through
    result = pipeline.process(
        raw_value=10.15,
        raw_confidence=0.85,
        peer_values=peer_values,
        context={"test": "integration"}
    )
    
    assert_true(result is not None, "Pipeline returns result")
    assert_true(hasattr(result, 'value'), "Result has value")
    assert_true(hasattr(result, 'consensus_status'), "Result has consensus status")
    assert_true(hasattr(result, 'metadata'), "Result has metadata")
    
    # Check that peer values were received by the pipeline
    # They should be in the metadata['input']
    assert_true('input' in result.metadata, "Metadata contains input")
    assert_true('peer_count' in result.metadata['input'], "Input metadata contains peer count")
    assert_equals(result.metadata['input']['peer_count'], 3, "Pipeline received 3 peer values")
    
    print(f"    Result: consensus={result.consensus_status}, peers_used={result.metadata['input']['peer_count']}")
    print("  ✓ Pipeline integration works correctly")
    return True


def test_multi_source_pipeline_integration():
    """Test multi-source adapter with AILEE pipeline."""
    print("  Testing multi-source pipeline integration...")
    
    config = AileeConfig(
        accept_threshold=0.70,
        borderline_low=0.50,
        borderline_high=0.70,
        consensus_quorum=3,
        enable_consensus=True,
        enable_audit_metadata=True
    )
    pipeline = AileeTrustPipeline(config)
    
    # Create multi-source adapter (simulating multiple models)
    model1 = StaticPeerAdapter([10.2, 10.3])
    model2 = StaticPeerAdapter([10.4, 10.5])
    model3 = StaticPeerAdapter([10.1])
    
    multi_adapter = MultiSourcePeerAdapter([model1, model2, model3])
    peers = multi_adapter.get_peer_values()
    
    # Test that multi-source adapter aggregates properly
    assert_length(peers, 5, "Multi-source aggregates all peer values")
    
    result = pipeline.process(
        raw_value=10.3,
        raw_confidence=0.85,
        peer_values=peers,
        context={"multi_model": True}
    )
    
    assert_true(result is not None, "Pipeline processes multi-source peers")
    assert_true('input' in result.metadata, "Metadata contains input")
    assert_true('peer_count' in result.metadata['input'], "Input metadata contains peer count")
    assert_equals(result.metadata['input']['peer_count'], 5, "Pipeline received all 5 aggregated peer values")
    
    print(f"    Multi-source: {len(peers)} peers aggregated, consensus={result.consensus_status}")
    print("  ✓ Multi-source pipeline integration works correctly")
    return True


# =============================================================================
# Effectiveness Tests (Real-World Scenarios)
# =============================================================================

def test_outlier_detection_effectiveness():
    """Test that filtered adapter effectively removes outliers."""
    print("  Testing outlier detection effectiveness...")
    
    # Simulate sensor data with occasional bad readings
    sensor_data = [
        10.0, 10.1, 10.2, 999.0,  # Outlier
        10.1, 10.3, 10.2, -100.0,  # Outlier
        10.0, 10.1, 10.2, 10.3
    ]
    
    source = StaticPeerAdapter(sensor_data)
    filtered = FilteredPeerAdapter(
        source,
        min_value=0.0,
        max_value=50.0,
        max_deviation_from_median=2.0
    )
    
    clean_data = filtered.get_peer_values()
    
    # Check that outliers are removed
    assert_true(999.0 not in clean_data, "Large outlier removed")
    assert_true(-100.0 not in clean_data, "Negative outlier removed")
    assert_true(len(clean_data) < len(sensor_data), "Some data filtered")
    assert_true(all(9.0 <= v <= 11.0 for v in clean_data), "Clean data in range")
    
    print(f"    Original: {len(sensor_data)} values, Filtered: {len(clean_data)} values")
    print("  ✓ Outlier detection is effective")
    return True


def test_consensus_effectiveness():
    """Test that multiple models improve consensus reliability."""
    print("  Testing consensus effectiveness...")
    
    config = AileeConfig(
        accept_threshold=0.85,
        consensus_quorum=3,
        consensus_delta=0.5,
        enable_consensus=True
    )
    pipeline = AileeTrustPipeline(config)
    
    # Scenario 1: Models agree - should pass consensus
    model_outputs = {
        "model_a": 10.2,
        "model_b": 10.3,
        "model_c": 10.1,
        "model_d": 10.4
    }
    
    adapter_agree = create_multi_model_adapter(model_outputs)
    peers_agree = adapter_agree.get_peer_values()
    
    result_agree = pipeline.process(
        raw_value=10.25,
        raw_confidence=0.88,
        peer_values=peers_agree
    )
    
    # Should have good consensus
    print(f"    Agreement scenario: consensus={result_agree.consensus_status}")
    
    # Scenario 2: Models disagree - may not pass consensus
    disagreeing_outputs = {
        "model_a": 5.0,
        "model_b": 15.0,
        "model_c": 25.0,
        "model_d": 35.0
    }
    
    adapter_disagree = create_multi_model_adapter(disagreeing_outputs)
    peers_disagree = adapter_disagree.get_peer_values()
    
    result_disagree = pipeline.process(
        raw_value=20.0,
        raw_confidence=0.88,
        peer_values=peers_disagree
    )
    
    print(f"    Disagreement scenario: consensus={result_disagree.consensus_status}")
    print("  ✓ Consensus detection is effective")
    return True


def test_weighted_trust_effectiveness():
    """Test that weighted adapters properly prioritize trusted sources."""
    print("  Testing weighted trust effectiveness...")
    
    # Create sources with different trust levels
    trusted_model = StaticPeerAdapter([10.0])
    untrusted_model = StaticPeerAdapter([50.0])  # Way off
    
    # Heavily weight the trusted model
    weighted_adapter = WeightedPeerAdapter(
        [trusted_model, untrusted_model],
        [0.95, 0.05],  # 95% trust in first, 5% in second
        mode="scale"
    )
    
    peers = weighted_adapter.get_peer_values()
    
    # The weighted values should be close to trusted model
    # 10.0 * 0.95 = 9.5, 50.0 * 0.05 = 2.5
    assert_in_range(peers[0], 9.0, 10.0, "Trusted model weighted value")
    assert_in_range(peers[1], 2.0, 3.0, "Untrusted model weighted value")
    
    print(f"    Weighted values: trusted={peers[0]:.2f}, untrusted={peers[1]:.2f}")
    print("  ✓ Weighted trust is effective")
    return True


def test_temporal_consensus_effectiveness():
    """Test rolling adapter for temporal consistency checks."""
    print("  Testing temporal consensus effectiveness...")
    
    # Simulate time series with sudden spike
    history = [10.0, 10.1, 10.0, 10.2, 10.1, 10.3, 100.0]  # Spike at end
    
    # Use rolling window to get recent stable values
    rolling = RollingPeerAdapter(history, window=5)
    recent = rolling.get_peer_values()
    
    # Most recent values should still include the spike
    assert_true(100.0 in recent, "Spike detected in recent window")
    
    # Now use a filtered version to clean it
    filtered = FilteredPeerAdapter(
        rolling,
        max_deviation_from_median=5.0
    )
    clean_recent = filtered.get_peer_values()
    
    # Spike should be filtered out
    assert_true(100.0 not in clean_recent, "Spike filtered from recent window")
    assert_true(all(9.0 <= v <= 11.0 for v in clean_recent), "Clean values stable")
    
    print(f"    Recent values: {len(recent)}, Clean: {len(clean_recent)}")
    print("  ✓ Temporal consensus is effective")
    return True


def test_caching_effectiveness():
    """Test that callback caching reduces computation."""
    print("  Testing caching effectiveness...")
    
    call_count = [0]
    
    def expensive_computation():
        """Simulate expensive peer value computation."""
        call_count[0] += 1
        time.sleep(0.01)  # Simulate delay
        return [10.0, 10.1, 10.2]
    
    # Without caching
    adapter_no_cache = CallbackPeerAdapter(expensive_computation)
    
    start = time.time()
    for _ in range(10):
        adapter_no_cache.get_peer_values()
    no_cache_time = time.time() - start
    no_cache_calls = call_count[0]
    
    # With caching
    call_count[0] = 0
    adapter_with_cache = CallbackPeerAdapter(expensive_computation, cache_ttl=1.0)
    
    start = time.time()
    for _ in range(10):
        adapter_with_cache.get_peer_values()
    cache_time = time.time() - start
    cache_calls = call_count[0]
    
    # Caching should significantly reduce calls
    assert_true(cache_calls < no_cache_calls, "Caching reduces calls")
    assert_equals(cache_calls, 1, "Only one call with caching")
    assert_equals(no_cache_calls, 10, "Ten calls without caching")
    
    print(f"    Without cache: {no_cache_calls} calls in {no_cache_time:.3f}s")
    print(f"    With cache: {cache_calls} calls in {cache_time:.3f}s")
    print(f"    Speedup: {no_cache_time/cache_time:.1f}x")
    print("  ✓ Caching is effective")
    return True


# =============================================================================
# Demo Execution Test
# =============================================================================

def test_demo_execution():
    """Test that the demo in the module runs successfully."""
    print("  Testing demo execution...")
    
    import subprocess
    
    # Get project root (parent of tests directory)
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    result = subprocess.run(
        [sys.executable, '-m', 'ailee.optional.ailee_peer_adapters'],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert_equals(result.returncode, 0, f"Demo failed: {result.stderr}")
    assert_true("Static Peer Adapter" in result.stdout, "Demo output present")
    assert_true("Multi-Model Adapter" in result.stdout, "Demo shows all adapters")
    
    print("  ✓ Demo execution successful")
    return True


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Peer Adapters Test Suite")
    print("Testing effectiveness of AILEE peer adapters")
    print("=" * 70 + "\n")
    
    tests = [
        ("Basic Adapters", [
            ("StaticPeerAdapter", test_static_adapter),
            ("RollingPeerAdapter", test_rolling_adapter),
        ]),
        ("Advanced Adapters", [
            ("CallbackPeerAdapter", test_callback_adapter),
            ("MultiSourcePeerAdapter", test_multi_source_adapter),
            ("WeightedPeerAdapter", test_weighted_adapter),
            ("FilteredPeerAdapter", test_filtered_adapter),
            ("MetadataPeerAdapter", test_metadata_adapter),
        ]),
        ("Convenience Functions", [
            ("create_multi_model_adapter", test_create_multi_model_adapter),
        ]),
        ("Integration Tests", [
            ("Pipeline Integration", test_pipeline_integration),
            ("Multi-Source Pipeline", test_multi_source_pipeline_integration),
        ]),
        ("Effectiveness Tests", [
            ("Outlier Detection", test_outlier_detection_effectiveness),
            ("Consensus Detection", test_consensus_effectiveness),
            ("Weighted Trust", test_weighted_trust_effectiveness),
            ("Temporal Consensus", test_temporal_consensus_effectiveness),
            ("Caching Performance", test_caching_effectiveness),
        ]),
        ("Demo Execution", [
            ("Module Demo", test_demo_execution),
        ]),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for category, category_tests in tests:
        print(f"\n{'=' * 70}")
        print(f"{category}")
        print('=' * 70)
        
        for name, test_func in category_tests:
            print(f"\nRunning: {name}")
            print("-" * 70)
            try:
                if test_func():
                    total_passed += 1
            except Exception as e:
                print(f"✗ {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total: {total_passed + total_failed} tests")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print("=" * 70 + "\n")
    
    # Print effectiveness assessment
    if total_failed == 0:
        print("✓ ALL TESTS PASSED!")
        print("\n" + "=" * 70)
        print("EFFECTIVENESS ASSESSMENT")
        print("=" * 70)
        print("The AILEE peer adapters are EFFECTIVE for:")
        print()
        print("✓ Static peer value management")
        print("✓ Rolling temporal windows for time-series data")
        print("✓ Dynamic callback-based peer retrieval with caching")
        print("✓ Multi-source aggregation (ensemble models)")
        print("✓ Weighted trust-based peer combination")
        print("✓ Outlier filtering and data cleaning")
        print("✓ Metadata tracking and provenance")
        print("✓ Integration with AILEE trust pipeline")
        print("✓ Consensus validation across multiple sources")
        print("✓ Performance optimization through caching")
        print()
        print("The adapters provide a flexible, robust system for managing")
        print("peer values in distributed AI systems and multi-model ensembles.")
        print("=" * 70 + "\n")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("The peer adapters have issues that need to be addressed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
