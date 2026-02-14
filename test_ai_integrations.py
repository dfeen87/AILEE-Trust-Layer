#!/usr/bin/env python
"""
AI Integrations Test
====================

This test verifies that the AI integration adapters work correctly:
1. All adapters can be imported and instantiated
2. Adapters can extract values from mock AI responses
3. Multi-model ensemble works correctly
4. Integration with AILEE pipeline works
5. Example files run successfully
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ailee_trust_pipeline_v1 import AileeTrustPipeline, AileeConfig
from optional.ailee_ai_integrations import (
    AIResponse,
    OpenAIAdapter,
    AnthropicAdapter,
    HuggingFaceAdapter,
    LangChainAdapter,
    MultiModelEnsemble,
    create_openai_adapter,
    create_anthropic_adapter,
    create_huggingface_adapter,
    create_langchain_adapter,
    create_multi_model_ensemble,
)


def test_adapter_imports():
    """Test that all adapters can be imported."""
    print("✓ AI adapter imports successful")
    return True


def test_openai_adapter():
    """Test OpenAI adapter with mock responses."""
    print("  Testing OpenAI adapter...")
    
    # Create mock OpenAI response
    class MockChoice:
        def __init__(self):
            self.message = type('obj', (object,), {'content': 'Quality score: 85.5'})()
    
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.model = "gpt-4"
            self.usage = type('obj', (object,), {
                'prompt_tokens': 50,
                'completion_tokens': 10,
                'total_tokens': 60
            })()
    
    # Test adapter
    adapter = create_openai_adapter(use_logprobs=False)
    mock_response = MockResponse()
    
    ai_response = adapter.extract_response(mock_response)
    
    assert isinstance(ai_response, AIResponse), "Response not an AIResponse"
    assert ai_response.value == 85.5, f"Unexpected value: {ai_response.value}"
    assert 0.0 <= ai_response.confidence <= 1.0, f"Confidence out of range: {ai_response.confidence}"
    assert ai_response.metadata['framework'] == 'openai', "Wrong framework"
    assert ai_response.metadata['model'] == 'gpt-4', "Wrong model"
    
    print(f"    Value: {ai_response.value}, Confidence: {ai_response.confidence:.2f}")
    print("  ✓ OpenAI adapter works correctly")
    return True


def test_anthropic_adapter():
    """Test Anthropic adapter with mock responses."""
    print("  Testing Anthropic adapter...")
    
    # Create mock Anthropic response
    class MockContent:
        def __init__(self):
            self.text = "Score: 92.3"
    
    class MockResponse:
        def __init__(self):
            self.content = [MockContent()]
            self.model = "claude-3-5-sonnet-20241022"
            self.stop_reason = "end_turn"
            self.usage = type('obj', (object,), {
                'input_tokens': 100,
                'output_tokens': 20
            })()
    
    # Test adapter
    adapter = create_anthropic_adapter()
    mock_response = MockResponse()
    
    ai_response = adapter.extract_response(mock_response)
    
    assert isinstance(ai_response, AIResponse), "Response not an AIResponse"
    assert ai_response.value == 92.3, f"Unexpected value: {ai_response.value}"
    assert 0.0 <= ai_response.confidence <= 1.0, f"Confidence out of range: {ai_response.confidence}"
    assert ai_response.metadata['framework'] == 'anthropic', "Wrong framework"
    assert ai_response.metadata['stop_reason'] == 'end_turn', "Wrong stop_reason"
    
    print(f"    Value: {ai_response.value}, Confidence: {ai_response.confidence:.2f}")
    print("  ✓ Anthropic adapter works correctly")
    return True


def test_huggingface_adapter():
    """Test HuggingFace adapter with mock responses."""
    print("  Testing HuggingFace adapter...")
    
    # Test with classification task
    adapter_cls = create_huggingface_adapter(task="classification")
    mock_response_cls = [{'label': 'POSITIVE', 'score': 0.95}]
    
    ai_response = adapter_cls.extract_response(mock_response_cls)
    
    assert isinstance(ai_response, AIResponse), "Response not an AIResponse"
    assert ai_response.confidence == 0.95, f"Unexpected confidence: {ai_response.confidence}"
    assert ai_response.metadata['framework'] == 'huggingface', "Wrong framework"
    
    # Test with generation task
    adapter_gen = create_huggingface_adapter(task="generation")
    mock_response_gen = [{'generated_text': 'The quality score is 88.0', 'score': 0.82}]
    
    ai_response = adapter_gen.extract_response(mock_response_gen)
    
    assert isinstance(ai_response, AIResponse), "Response not an AIResponse"
    assert ai_response.value == 88.0, f"Unexpected value: {ai_response.value}"
    assert ai_response.confidence == 0.82, f"Unexpected confidence: {ai_response.confidence}"
    
    print(f"    Classification confidence: 0.95, Generation value: 88.0")
    print("  ✓ HuggingFace adapter works correctly")
    return True


def test_langchain_adapter():
    """Test LangChain adapter with mock responses."""
    print("  Testing LangChain adapter...")
    
    adapter = create_langchain_adapter()
    
    # Test with string response
    mock_response_str = "The rating is 75.5"
    ai_response = adapter.extract_response(mock_response_str)
    
    assert isinstance(ai_response, AIResponse), "Response not an AIResponse"
    assert ai_response.value == 75.5, f"Unexpected value: {ai_response.value}"
    assert ai_response.metadata['framework'] == 'langchain', "Wrong framework"
    
    # Test with dict response
    mock_response_dict = {'output': 'Score: 82.0', 'metadata': 'test'}
    ai_response = adapter.extract_response(mock_response_dict)
    
    assert isinstance(ai_response, AIResponse), "Response not an AIResponse"
    assert ai_response.value == 82.0, f"Unexpected value: {ai_response.value}"
    
    print(f"    String extraction: 75.5, Dict extraction: 82.0")
    print("  ✓ LangChain adapter works correctly")
    return True


def test_custom_extractors():
    """Test adapters with custom extractors."""
    print("  Testing custom extractors...")
    
    import json
    import re
    
    # Custom value extractor for JSON responses
    def extract_json_value(response, content, context):
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return float(data.get('score', 0))
            except:
                pass
        return 0.0
    
    # Custom confidence extractor
    def extract_custom_confidence(response, content, context):
        if 'high' in content.lower():
            return 0.95
        elif 'medium' in content.lower():
            return 0.75
        else:
            return 0.50
    
    # Create adapter with custom extractors
    adapter = OpenAIAdapter(
        value_extractor=extract_json_value,
        confidence_extractor=extract_custom_confidence
    )
    
    # Mock response with JSON
    class MockChoice:
        def __init__(self):
            self.message = type('obj', (object,), {
                'content': 'Result: {"score": 96.5} - confidence: high'
            })()
    
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.model = "gpt-4"
    
    ai_response = adapter.extract_response(MockResponse())
    
    assert ai_response.value == 96.5, f"Custom value extractor failed: {ai_response.value}"
    assert ai_response.confidence == 0.95, f"Custom confidence extractor failed: {ai_response.confidence}"
    
    print(f"    Custom extractors: value=96.5, confidence=0.95")
    print("  ✓ Custom extractors work correctly")
    return True


def test_multi_model_ensemble():
    """Test multi-model ensemble functionality."""
    print("  Testing multi-model ensemble...")
    
    ensemble = create_multi_model_ensemble()
    
    # Add responses from different "models"
    openai_adapter = create_openai_adapter(use_logprobs=False)
    anthropic_adapter = create_anthropic_adapter()
    
    # Mock OpenAI response
    class MockOpenAIChoice:
        def __init__(self):
            self.message = type('obj', (object,), {'content': 'Score: 85.0'})()
    
    class MockOpenAIResponse:
        def __init__(self):
            self.choices = [MockOpenAIChoice()]
            self.model = "gpt-4"
    
    # Mock Anthropic response
    class MockAnthropicContent:
        def __init__(self):
            self.text = "Score: 87.0"
    
    class MockAnthropicResponse:
        def __init__(self):
            self.content = [MockAnthropicContent()]
            self.model = "claude-3-5-sonnet-20241022"
            self.stop_reason = "end_turn"
    
    # Add to ensemble
    ensemble.add_response("gpt4", MockOpenAIResponse(), openai_adapter, 
                         context={'default_confidence': 0.90})
    ensemble.add_response("claude", MockAnthropicResponse(), anthropic_adapter)
    
    # Test ensemble methods
    assert len(ensemble.responses) == 2, "Ensemble should have 2 responses"
    
    primary, peers, weights = ensemble.get_consensus_inputs()
    assert len(peers) == 2, "Should have 2 peer values"
    assert 84 <= primary <= 88, f"Primary value out of expected range: {primary}"
    
    weighted_avg, avg_conf = ensemble.get_weighted_average()
    assert 85 <= weighted_avg <= 87, f"Weighted average out of range: {weighted_avg}"
    assert 0.0 <= avg_conf <= 1.0, f"Average confidence out of range: {avg_conf}"
    
    print(f"    Ensemble size: 2, Primary: {primary:.1f}, Weighted avg: {weighted_avg:.1f}")
    print("  ✓ Multi-model ensemble works correctly")
    return True


def test_ailee_pipeline_integration():
    """Test integration of adapters with AILEE pipeline."""
    print("  Testing AILEE pipeline integration...")
    
    # Create pipeline
    config = AileeConfig(
        accept_threshold=0.85,
        borderline_low=0.70,
        borderline_high=0.85,
    )
    pipeline = AileeTrustPipeline(config)
    
    # Create adapter
    adapter = create_openai_adapter(use_logprobs=False)
    
    # Mock high-confidence response
    class MockChoice:
        def __init__(self):
            self.message = type('obj', (object,), {'content': 'Score: 90.0'})()
    
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.model = "gpt-4"
    
    # Extract and validate
    ai_response = adapter.extract_response(MockResponse(), 
                                          context={'default_confidence': 0.92})
    
    result = pipeline.process(
        raw_value=ai_response.value,
        raw_confidence=0.92,  # Override for high confidence
        context={"model": "gpt-4"}
    )
    
    assert result is not None, "Pipeline result is None"
    assert hasattr(result, 'value'), "Result missing value"
    assert hasattr(result, 'safety_status'), "Result missing safety_status"
    
    print(f"    Pipeline result: value={result.value:.1f}, status={result.safety_status}")
    print("  ✓ AILEE pipeline integration works correctly")
    return True


def test_example_files():
    """Test that example files can be executed."""
    print("  Testing example files...")
    import subprocess
    
    examples = [
        'examples/ai_integration_openai.py',
        'examples/ai_integration_multi_model.py',
    ]
    
    for example in examples:
        result = subprocess.run(
            [sys.executable, example],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=15
        )
        
        assert result.returncode == 0, f"Example {example} failed: {result.stderr}"
        print(f"    ✓ {os.path.basename(example)} runs successfully")
    
    print("  ✓ All example files work correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AI Integrations Test Suite")
    print("=" * 70 + "\n")
    
    tests = [
        ("Adapter Imports", test_adapter_imports),
        ("OpenAI Adapter", test_openai_adapter),
        ("Anthropic Adapter", test_anthropic_adapter),
        ("HuggingFace Adapter", test_huggingface_adapter),
        ("LangChain Adapter", test_langchain_adapter),
        ("Custom Extractors", test_custom_extractors),
        ("Multi-Model Ensemble", test_multi_model_ensemble),
        ("AILEE Pipeline Integration", test_ailee_pipeline_integration),
        ("Example Files", test_example_files),
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
        print("✓ ALL TESTS PASSED - AI integrations are working correctly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - AI integrations have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
