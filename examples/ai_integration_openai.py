"""
AILEE Trust Layer - OpenAI Integration Example

This example demonstrates how to integrate AILEE with OpenAI's API
to add trust validation to GPT-4 and other model outputs.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ailee_trust_pipeline_v1 import AileeTrustPipeline, AileeConfig
from optional.ailee_ai_integrations import OpenAIAdapter, create_openai_adapter

# Note: This example shows the integration pattern.
# To run it, you'll need: pip install openai
# and set OPENAI_API_KEY environment variable


def example_basic_openai_integration():
    """
    Basic example: Single OpenAI call with AILEE trust validation.
    """
    print("=" * 60)
    print("Example 1: Basic OpenAI Integration")
    print("=" * 60)
    
    # Mock OpenAI response for demonstration
    # In practice, this would be from: client.chat.completions.create(...)
    class MockChoice:
        def __init__(self):
            self.message = type('obj', (object,), {'content': 'Quality score: 87.5'})()
    
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.model = "gpt-4"
            self.usage = type('obj', (object,), {
                'prompt_tokens': 50,
                'completion_tokens': 10,
                'total_tokens': 60
            })()
    
    # Create AILEE pipeline with LLM preset
    config = AileeConfig(
        accept_threshold=0.85,
        borderline_low=0.70,
        borderline_high=0.85,
    )
    pipeline = AileeTrustPipeline(config)
    
    # Create OpenAI adapter
    adapter = create_openai_adapter(use_logprobs=False)
    
    # Simulate OpenAI response
    openai_response = MockResponse()
    
    # Extract value and confidence
    ai_response = adapter.extract_response(openai_response)
    
    print(f"Extracted value: {ai_response.value}")
    print(f"Extracted confidence: {ai_response.confidence}")
    
    # Validate through AILEE
    result = pipeline.process(
        raw_value=ai_response.value,
        raw_confidence=ai_response.confidence,
        context={
            "feature": "quality_score",
            "model": ai_response.metadata.get("model"),
            "framework": "openai"
        }
    )
    
    print(f"\nAILEE Decision:")
    print(f"  Trusted Value: {result.value}")
    print(f"  Safety Status: {result.safety_status}")
    print(f"  Used Fallback: {result.used_fallback}")
    print(f"  Reasons: {', '.join(result.reasons)}")
    print()


def example_openai_with_custom_extraction():
    """
    Advanced example: Custom value extraction from OpenAI responses.
    """
    print("=" * 60)
    print("Example 2: Custom Value Extraction")
    print("=" * 60)
    
    # Custom extractor that looks for JSON-formatted responses
    def extract_json_value(response, content, context):
        import json
        import re
        # Look for JSON in the response
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return float(data.get('score', data.get('value', 0)))
            except:
                pass
        # Fallback to numeric extraction
        num_match = re.search(r'-?\d+\.?\d*', content)
        return float(num_match.group(0)) if num_match else 0.0
    
    # Create adapter with custom extractor
    adapter = OpenAIAdapter(value_extractor=extract_json_value)
    
    # Mock response with JSON
    class MockChoice:
        def __init__(self):
            self.message = type('obj', (object,), {
                'content': 'Analysis result: {"score": 92.3, "confidence": "high"}'
            })()
    
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.model = "gpt-4"
    
    response = MockResponse()
    ai_response = adapter.extract_response(response)
    
    print(f"Extracted value from JSON: {ai_response.value}")
    print(f"Content: {ai_response.metadata['content']}")
    print()


def example_openai_multi_turn_validation():
    """
    Example: Multi-turn conversation with progressive trust building.
    """
    print("=" * 60)
    print("Example 3: Multi-Turn Trust Building")
    print("=" * 60)
    
    config = AileeConfig()
    pipeline = AileeTrustPipeline(config)
    adapter = create_openai_adapter()
    
    # Simulate multiple turns
    turns = [
        "First assessment: 85",
        "Refined assessment: 87",
        "Final assessment: 86.5"
    ]
    
    print("Processing conversation turns:")
    for i, content in enumerate(turns, 1):
        # Mock response
        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (object,), {'content': content})()
        
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
                self.model = "gpt-4"
        
        response = MockResponse(content)
        ai_response = adapter.extract_response(response)
        
        # Each turn adds to history for consensus
        result = pipeline.process(
            raw_value=ai_response.value,
            raw_confidence=ai_response.confidence,
            context={"turn": i}
        )
        
        print(f"  Turn {i}: Value={result.value:.1f}, Status={result.safety_status}")
    
    print()


# Uncomment to run examples
if __name__ == "__main__":
    example_basic_openai_integration()
    example_openai_with_custom_extraction()
    example_openai_multi_turn_validation()
    
    print("=" * 60)
    print("Integration Examples Complete!")
    print("=" * 60)
    print("\nTo use with real OpenAI API:")
    print("1. pip install openai")
    print("2. Set OPENAI_API_KEY environment variable")
    print("3. Replace mock responses with actual API calls:")
    print("   from openai import OpenAI")
    print("   client = OpenAI()")
    print("   response = client.chat.completions.create(...)")
