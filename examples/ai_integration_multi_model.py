"""
AILEE Trust Layer - Multi-Model Ensemble Example

This example demonstrates how to use AILEE with multiple AI models
(OpenAI, Anthropic, HuggingFace) for consensus-based decision making.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ailee_trust_pipeline_v1 import AileeTrustPipeline, AileeConfig
from optional.ailee_ai_integrations import (
    OpenAIAdapter,
    AnthropicAdapter,
    HuggingFaceAdapter,
    MultiModelEnsemble,
    create_multi_model_ensemble
)


def example_multi_model_consensus():
    """
    Example: Combine outputs from multiple AI models with AILEE consensus.
    """
    print("=" * 60)
    print("Multi-Model Ensemble with AILEE Trust Validation")
    print("=" * 60)
    
    # Create adapters for different frameworks
    openai_adapter = OpenAIAdapter(use_logprobs=False)
    anthropic_adapter = AnthropicAdapter()
    huggingface_adapter = HuggingFaceAdapter(task="generation")
    
    # Mock responses from different models
    # In practice, these would be actual API calls
    
    # Mock OpenAI response
    class MockOpenAIChoice:
        def __init__(self):
            self.message = type('obj', (object,), {'content': 'Score: 85.2'})()
    
    class MockOpenAIResponse:
        def __init__(self):
            self.choices = [MockOpenAIChoice()]
            self.model = "gpt-4"
    
    # Mock Anthropic response
    class MockAnthropicContent:
        def __init__(self):
            self.text = "Score: 87.5"
    
    class MockAnthropicResponse:
        def __init__(self):
            self.content = [MockAnthropicContent()]
            self.model = "claude-3-5-sonnet-20241022"
            self.stop_reason = "end_turn"
    
    # Mock HuggingFace response
    mock_hf_response = {
        'generated_text': 'The quality score is 86.0',
        'score': 0.88
    }
    
    # Create ensemble
    ensemble = create_multi_model_ensemble()
    
    # Add responses from each model
    print("\nAdding model responses to ensemble:")
    
    ensemble.add_response(
        "gpt4", 
        MockOpenAIResponse(), 
        openai_adapter,
        context={'default_confidence': 0.90}
    )
    print("  ✓ Added GPT-4 response")
    
    ensemble.add_response(
        "claude", 
        MockAnthropicResponse(), 
        anthropic_adapter
    )
    print("  ✓ Added Claude response")
    
    ensemble.add_response(
        "llama", 
        mock_hf_response, 
        huggingface_adapter
    )
    print("  ✓ Added Llama response")
    
    # Get consensus inputs
    primary_value, peer_values, confidence_weights = ensemble.get_consensus_inputs()
    
    print(f"\nEnsemble Analysis:")
    print(f"  Primary value: {primary_value:.1f}")
    print(f"  Peer values: {[f'{v:.1f}' for v in peer_values]}")
    print(f"  Confidence weights: {', '.join(f'{k}: {v:.2f}' for k, v in confidence_weights.items())}")
    
    # Get weighted average
    weighted_avg, avg_confidence = ensemble.get_weighted_average()
    print(f"  Weighted average: {weighted_avg:.1f}")
    print(f"  Average confidence: {avg_confidence:.2f}")
    
    # Validate through AILEE with consensus
    config = AileeConfig(
        consensus_quorum=3,  # Require all 3 models to agree
        consensus_delta=5.0,  # Within 5 points
    )
    pipeline = AileeTrustPipeline(config)
    
    result = pipeline.process(
        raw_value=primary_value,
        raw_confidence=avg_confidence,
        peer_values=peer_values,
        context={
            "feature": "quality_score",
            "ensemble": "gpt4+claude+llama"
        }
    )
    
    print(f"\nAILEE Trust Decision:")
    print(f"  Trusted Value: {result.value:.1f}")
    print(f"  Safety Status: {result.safety_status}")
    print(f"  Consensus Status: {result.consensus_status}")
    print(f"  Grace Status: {result.grace_status}")
    print(f"  Used Fallback: {result.used_fallback}")
    print(f"  Decision Reasons:")
    for reason in result.reasons:
        print(f"    - {reason}")
    print()


def example_weighted_ensemble():
    """
    Example: Weighted ensemble based on model confidence.
    """
    print("=" * 60)
    print("Weighted Multi-Model Ensemble")
    print("=" * 60)
    
    ensemble = create_multi_model_ensemble()
    
    # Simulate different confidence levels
    models_data = [
        ("gpt4", 88.0, 0.95),      # High confidence
        ("claude", 85.0, 0.92),    # High confidence
        ("mistral", 82.0, 0.78),   # Lower confidence
        ("llama", 90.0, 0.65),     # Lower confidence (outlier)
    ]
    
    print("\nModel outputs:")
    for name, value, conf in models_data:
        print(f"  {name:10s}: value={value:5.1f}, confidence={conf:.2f}")
        
        # Create mock response
        mock_response = {
            'generated_text': f'Value: {value}',
            'score': conf
        }
        adapter = HuggingFaceAdapter()
        ensemble.add_response(name, mock_response, adapter)
    
    # Get weighted average (high confidence models have more influence)
    weighted_value, avg_conf = ensemble.get_weighted_average()
    primary_value, peer_values, _ = ensemble.get_consensus_inputs()
    
    print(f"\nWeighted average: {weighted_value:.1f}")
    print(f"Simple average: {sum(v for _, v, _ in models_data) / len(models_data):.1f}")
    print(f"Note: Weighted average gives more weight to high-confidence models")
    
    # Use AILEE to validate
    pipeline = AileeTrustPipeline(AileeConfig(consensus_quorum=3))
    
    result = pipeline.process(
        raw_value=weighted_value,
        raw_confidence=avg_conf,
        peer_values=peer_values,
        context={"ensemble_type": "weighted"}
    )
    
    print(f"\nAILEE validates weighted ensemble:")
    print(f"  Final value: {result.value:.1f}")
    print(f"  Consensus: {result.consensus_status}")
    print()


def example_progressive_ensemble():
    """
    Example: Progressive ensemble where models are added incrementally.
    """
    print("=" * 60)
    print("Progressive Model Addition")
    print("=" * 60)
    
    pipeline = AileeTrustPipeline(AileeConfig())
    ensemble = create_multi_model_ensemble()
    adapter = HuggingFaceAdapter()
    
    # Add models progressively
    progressive_models = [
        ("model_1", 85.0, 0.80),
        ("model_2", 87.0, 0.85),
        ("model_3", 86.5, 0.90),
    ]
    
    print("\nAdding models progressively:")
    for i, (name, value, conf) in enumerate(progressive_models, 1):
        mock_response = {'generated_text': f'{value}', 'score': conf}
        ensemble.add_response(name, mock_response, adapter)
        
        if len(ensemble.responses) >= 2:  # Need at least 2 for consensus
            primary, peers, weights = ensemble.get_consensus_inputs()
            result = pipeline.process(
                raw_value=primary,
                raw_confidence=list(weights.values())[0],
                peer_values=peers if len(peers) > 1 else None
            )
            print(f"  After {i} models: value={result.value:.1f}, "
                  f"consensus={result.consensus_status}, "
                  f"peers={len(peers)}")
    print()


if __name__ == "__main__":
    example_multi_model_consensus()
    example_weighted_ensemble()
    example_progressive_ensemble()
    
    print("=" * 60)
    print("Multi-Model Integration Examples Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. AILEE can validate outputs from multiple AI models")
    print("2. Consensus checking ensures model agreement")
    print("3. Weighted ensembles give more influence to confident models")
    print("4. Models can be added progressively for incremental validation")
