"""
AILEE Trust Layer - Google Gemini Integration Example

This example demonstrates how to integrate AILEE with Google's Gemini API
to add trust validation to Gemini Pro and other model outputs.
"""

import sys
import os

# Add parent directory to path

from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_ai_integrations import GeminiAdapter, create_gemini_adapter

# Note: This example shows the integration pattern.
# To run it, you'll need: pip install google-generativeai
# and set GOOGLE_API_KEY environment variable


def example_basic_gemini_integration():
    """
    Basic example: Single Gemini call with AILEE trust validation.
    """
    print("=" * 60)
    print("Example 1: Basic Gemini Integration")
    print("=" * 60)
    
    # Mock Gemini response for demonstration
    # In practice, this would be from: model.generate_content(...)
    
    class MockPart:
        def __init__(self, text):
            self.text = text
    
    class MockContent:
        def __init__(self):
            self.parts = [MockPart('Quality score: 89.5')]
    
    class MockCandidate:
        def __init__(self):
            self.content = MockContent()
            self.finish_reason = 'STOP'
            
            # Mock safety ratings
            class MockRating:
                def __init__(self, category, probability):
                    self.category = category
                    self.probability = probability
            
            self.safety_ratings = [
                MockRating('HARM_CATEGORY_HARASSMENT', 'NEGLIGIBLE'),
                MockRating('HARM_CATEGORY_HATE_SPEECH', 'NEGLIGIBLE'),
                MockRating('HARM_CATEGORY_SEXUALLY_EXPLICIT', 'NEGLIGIBLE'),
                MockRating('HARM_CATEGORY_DANGEROUS_CONTENT', 'LOW'),
            ]
    
    class MockUsageMetadata:
        def __init__(self):
            self.prompt_token_count = 45
            self.candidates_token_count = 15
            self.total_token_count = 60
    
    class MockResponse:
        def __init__(self):
            self.text = 'Quality score: 89.5'
            self.candidates = [MockCandidate()]
            self.usage_metadata = MockUsageMetadata()
    
    # Create AILEE pipeline
    config = AileeConfig(
        accept_threshold=0.85,
        borderline_low=0.70,
        borderline_high=0.85,
    )
    pipeline = AileeTrustPipeline(config)
    
    # Create Gemini adapter
    adapter = create_gemini_adapter()
    
    # Simulate Gemini response
    gemini_response = MockResponse()
    
    # Extract value and confidence
    ai_response = adapter.extract_response(gemini_response)
    
    print(f"Extracted value: {ai_response.value}")
    print(f"Extracted confidence: {ai_response.confidence}")
    print(f"Finish reason: {ai_response.metadata.get('finish_reason')}")
    
    # Validate through AILEE
    result = pipeline.process(
        raw_value=ai_response.value,
        raw_confidence=ai_response.confidence,
        context={
            "feature": "quality_score",
            "framework": "gemini"
        }
    )
    
    print(f"\nAILEE Decision:")
    print(f"  Trusted Value: {result.value}")
    print(f"  Safety Status: {result.safety_status}")
    print(f"  Used Fallback: {result.used_fallback}")
    print(f"  Reasons: {', '.join(result.reasons)}")
    print()


def example_gemini_safety_ratings():
    """
    Example: Using Gemini's built-in safety ratings for confidence.
    """
    print("=" * 60)
    print("Example 2: Safety Ratings Integration")
    print("=" * 60)
    
    # Test different safety scenarios
    safety_scenarios = [
        {
            "name": "Safe content",
            "finish": "STOP",
            "ratings": [
                ("HARM_CATEGORY_HARASSMENT", "NEGLIGIBLE"),
                ("HARM_CATEGORY_HATE_SPEECH", "NEGLIGIBLE"),
            ],
        },
        {
            "name": "Moderate risk content",
            "finish": "STOP",
            "ratings": [
                ("HARM_CATEGORY_HARASSMENT", "LOW"),
                ("HARM_CATEGORY_HATE_SPEECH", "MEDIUM"),
            ],
        },
        {
            "name": "Blocked content",
            "finish": "SAFETY",
            "ratings": [
                ("HARM_CATEGORY_HARASSMENT", "HIGH"),
            ],
        },
    ]
    
    adapter = create_gemini_adapter()
    
    print("Testing confidence estimation with safety ratings:")
    print("-" * 60)
    
    for scenario in safety_scenarios:
        # Create mock response
        class MockRating:
            def __init__(self, category, probability):
                self.category = category
                self.probability = probability
        
        class MockCandidate:
            def __init__(self, finish_reason, ratings):
                class MockContent:
                    def __init__(self):
                        class MockPart:
                            def __init__(self):
                                self.text = "Score: 85.0"
                        self.parts = [MockPart()]
                
                self.content = MockContent()
                self.finish_reason = finish_reason
                self.safety_ratings = [
                    MockRating(cat, prob) for cat, prob in ratings
                ]
        
        class MockResponse:
            def __init__(self, finish_reason, ratings):
                self.text = "Score: 85.0"
                self.candidates = [MockCandidate(finish_reason, ratings)]
        
        response = MockResponse(scenario["finish"], scenario["ratings"])
        ai_response = adapter.extract_response(response)
        
        print(f"\n{scenario['name']}:")
        print(f"  Finish reason: {scenario['finish']}")
        print(f"  Extracted confidence: {ai_response.confidence:.2f}")
        print(f"  Safety ratings: {len(scenario['ratings'])} checks")
    
    print()


def example_gemini_multimodal():
    """
    Example: Gemini Pro Vision with image analysis.
    """
    print("=" * 60)
    print("Example 3: Multimodal (Vision) Integration")
    print("=" * 60)
    print("\nScenario: Analyzing image quality with Gemini Pro Vision")
    
    # Mock vision response
    class MockPart:
        def __init__(self):
            self.text = "Image quality analysis: The image is clear and well-composed. Quality score: 92/100"
    
    class MockContent:
        def __init__(self):
            self.parts = [MockPart()]
    
    class MockCandidate:
        def __init__(self):
            self.content = MockContent()
            self.finish_reason = 'STOP'
            
            class MockRating:
                def __init__(self, cat, prob):
                    self.category = cat
                    self.probability = prob
            
            self.safety_ratings = [
                MockRating('HARM_CATEGORY_DANGEROUS_CONTENT', 'NEGLIGIBLE'),
            ]
    
    class MockResponse:
        def __init__(self):
            self.text = "Image quality analysis: The image is clear and well-composed. Quality score: 92/100"
            self.candidates = [MockCandidate()]
    
    adapter = create_gemini_adapter()
    pipeline = AileeTrustPipeline(AileeConfig())
    
    vision_response = MockResponse()
    ai_response = adapter.extract_response(vision_response)
    
    print(f"Vision analysis extracted:")
    print(f"  Value: {ai_response.value}")
    print(f"  Confidence: {ai_response.confidence:.2f}")
    
    result = pipeline.process(
        raw_value=ai_response.value,
        raw_confidence=ai_response.confidence,
        context={"task": "image_quality", "model": "gemini-pro-vision"}
    )
    
    print(f"\nAILEE validation:")
    print(f"  Trusted score: {result.value:.1f}")
    print(f"  Status: {result.safety_status}")
    print()


def example_gemini_vs_others():
    """
    Example: Comparing Gemini with other models in ensemble.
    """
    print("=" * 60)
    print("Example 4: Gemini in Multi-Model Ensemble")
    print("=" * 60)
    
    from ailee.optional.ailee_ai_integrations import (
        create_multi_model_ensemble,
        OpenAIAdapter,
        AnthropicAdapter,
    )
    
    ensemble = create_multi_model_ensemble()
    
    # Mock responses from different models
    print("\nCombining Gemini with OpenAI and Anthropic:")
    
    # Gemini response
    class MockGeminiResponse:
        def __init__(self):
            self.text = "Score: 88.0"
            class MockCandidate:
                def __init__(self):
                    class MockContent:
                        def __init__(self):
                            class MockPart:
                                text = "Score: 88.0"
                            self.parts = [MockPart()]
                    self.content = MockContent()
                    self.finish_reason = 'STOP'
                    self.safety_ratings = []
            self.candidates = [MockCandidate()]
    
    # OpenAI response
    class MockOpenAIResponse:
        def __init__(self):
            class MockChoice:
                def __init__(self):
                    self.message = type('obj', (object,), {'content': 'Score: 86.5'})()
            self.choices = [MockChoice()]
            self.model = "gpt-4"
    
    # Anthropic response
    class MockAnthropicResponse:
        def __init__(self):
            class MockContent:
                text = "Score: 87.0"
            self.content = [MockContent()]
            self.model = "claude-3-5-sonnet-20241022"
            self.stop_reason = "end_turn"
    
    # Add to ensemble
    ensemble.add_response("gemini", MockGeminiResponse(), create_gemini_adapter())
    ensemble.add_response("gpt4", MockOpenAIResponse(), OpenAIAdapter(use_logprobs=False))
    ensemble.add_response("claude", MockAnthropicResponse(), AnthropicAdapter())
    
    # Get consensus
    primary, peers, confidences = ensemble.get_consensus_inputs()
    weighted_avg, avg_conf = ensemble.get_weighted_average()
    
    print(f"  Gemini: 88.0")
    print(f"  GPT-4: 86.5")
    print(f"  Claude: 87.0")
    print(f"\nEnsemble results:")
    print(f"  Primary value: {primary:.1f}")
    print(f"  Weighted average: {weighted_avg:.1f}")
    print(f"  Average confidence: {avg_conf:.2f}")
    
    # Validate through AILEE
    pipeline = AileeTrustPipeline(AileeConfig(consensus_quorum=3))
    result = pipeline.process(
        raw_value=weighted_avg,
        raw_confidence=avg_conf,
        peer_values=peers
    )
    
    print(f"\nAILEE consensus decision:")
    print(f"  Final value: {result.value:.1f}")
    print(f"  Consensus status: {result.consensus_status}")
    print()


# Uncomment to run examples
if __name__ == "__main__":
    example_basic_gemini_integration()
    example_gemini_safety_ratings()
    example_gemini_multimodal()
    example_gemini_vs_others()
    
    print("=" * 60)
    print("Gemini Integration Examples Complete!")
    print("=" * 60)
    print("\nTo use with real Gemini API:")
    print("1. pip install google-generativeai")
    print("2. Set GOOGLE_API_KEY environment variable")
    print("3. Replace mock responses with actual API calls:")
    print("   import google.generativeai as genai")
    print("   genai.configure(api_key=os.environ['GOOGLE_API_KEY'])")
    print("   model = genai.GenerativeModel('gemini-pro')")
    print("   response = model.generate_content(...)")
    print("\nGemini Features:")
    print("  ✓ Automatic confidence from safety ratings")
    print("  ✓ Finish reason analysis (STOP, SAFETY, MAX_TOKENS)")
    print("  ✓ Support for Gemini Pro and Gemini Pro Vision")
    print("  ✓ Safety rating integration")
