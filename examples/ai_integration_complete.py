"""
AILEE Trust Layer - Complete AI Integration Walkthrough

This example demonstrates a complete end-to-end workflow using AILEE
with AI systems for a real-world use case: Content Quality Scoring.
"""

import sys
import os

# Add parent directory to path

from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_config_presets import LLM_SCORING, CONSERVATIVE, TRADING_SIGNAL, BALANCED
from ailee.optional.ailee_ai_integrations import (
    create_openai_adapter,
    create_anthropic_adapter,
    create_multi_model_ensemble,
    HuggingFaceAdapter,
)


def create_pipeline(preset_name):
    """Helper to create pipeline from preset name."""
    from ailee.optional.ailee_config_presets import get_preset
    config = get_preset(preset_name)
    return AileeTrustPipeline(config)


def example_single_model_integration():
    """
    Example 1: Simple single-model integration with trust validation.
    
    Use case: Content quality scoring with safety checks.
    """
    print("=" * 70)
    print("Example 1: Single Model Integration")
    print("=" * 70)
    print("\nScenario: Using AI to score content quality (0-100)")
    print("Goal: Only act on high-confidence AI outputs\n")
    
    # Step 1: Configure AILEE for LLM scoring
    pipeline = create_pipeline("llm_scoring")
    adapter = create_openai_adapter()
    
    # Step 2: Simulate AI responses with varying quality
    test_cases = [
        ("High quality article", 92.0, 0.95),
        ("Medium quality article", 75.0, 0.78),
        ("Low quality article", 45.0, 0.60),
        ("Uncertain assessment", 85.0, 0.65),
    ]
    
    print("Processing content through AI + AILEE:")
    print("-" * 70)
    
    for content_desc, ai_score, ai_confidence in test_cases:
        # Simulate AI extraction (in real use, this comes from actual API)
        result = pipeline.process(
            raw_value=ai_score,
            raw_confidence=ai_confidence,
            context={"content_type": "article"}
        )
        
        # Decision logic based on AILEE validation
        if result.safety_status == "ACCEPTED" and not result.used_fallback:
            action = "✓ PUBLISH"
        elif result.safety_status == "BORDERLINE":
            action = "⚠ REVIEW"
        else:
            action = "✗ REJECT"
        
        print(f"{content_desc:25s} | Score: {ai_score:5.1f} | "
              f"Conf: {ai_confidence:.2f} | {action}")
        print(f"  └─ AILEE: {result.safety_status}, "
              f"Fallback: {result.used_fallback}")
    
    print()


def example_multi_model_consensus():
    """
    Example 2: Multi-model consensus for critical decisions.
    
    Use case: Medical content validation requiring agreement.
    """
    print("=" * 70)
    print("Example 2: Multi-Model Consensus")
    print("=" * 70)
    print("\nScenario: Validating medical information")
    print("Goal: Require multiple AI models to agree before publishing\n")
    
    # Step 1: Configure for high-stakes decisions
    config = AileeConfig(
        accept_threshold=0.90,      # High confidence required
        consensus_quorum=3,          # All 3 models must agree
        consensus_delta=5.0,         # Within 5 points
    )
    pipeline = AileeTrustPipeline(config)
    
    # Step 2: Simulate multiple AI model responses
    medical_articles = [
        {
            "title": "COVID-19 Vaccine Efficacy",
            "gpt4": (95.0, 0.96),
            "claude": (93.0, 0.94),
            "medical_llm": (94.0, 0.95),
        },
        {
            "title": "Alternative Medicine Claims",
            "gpt4": (45.0, 0.85),
            "claude": (72.0, 0.80),
            "medical_llm": (50.0, 0.88),
        },
    ]
    
    print("Medical content validation with multi-model consensus:")
    print("-" * 70)
    
    for article in medical_articles:
        ensemble = create_multi_model_ensemble()
        
        # Add each model's assessment
        for model_name in ["gpt4", "claude", "medical_llm"]:
            score, conf = article[model_name]
            # In real use, these would be actual AI responses
            mock_response = {"generated_text": f"Score: {score}", "score": conf}
            
            adapter = HuggingFaceAdapter()
            ensemble.add_response(model_name, mock_response, adapter)
        
        # Get consensus
        primary, peers, confidences = ensemble.get_consensus_inputs()
        avg_conf = sum(confidences.values()) / len(confidences)
        
        # Validate with AILEE
        result = pipeline.process(
            raw_value=primary,
            raw_confidence=avg_conf,
            peer_values=peers,
            context={"domain": "medical", "critical": True}
        )
        
        # Decision
        if (result.consensus_status == "PASS" and 
            result.safety_status == "ACCEPTED" and 
            not result.used_fallback):
            action = "✓ APPROVED"
        else:
            action = "⚠ HUMAN REVIEW REQUIRED"
        
        print(f"\n{article['title']}")
        print(f"  Model scores: GPT-4={article['gpt4'][0]:.0f}, "
              f"Claude={article['claude'][0]:.0f}, "
              f"Medical={article['medical_llm'][0]:.0f}")
        print(f"  Consensus: {result.consensus_status}, "
              f"Safety: {result.safety_status}")
        print(f"  Decision: {action}")
    
    print()


def example_confidence_building():
    """
    Example 3: Building trust over time with historical data.
    
    Use case: AI-powered trading signals with trend validation.
    """
    print("=" * 70)
    print("Example 3: Progressive Trust Building")
    print("=" * 70)
    print("\nScenario: AI trading signal validation")
    print("Goal: Build confidence through consistent predictions\n")
    
    pipeline = AileeTrustPipeline(TRADING_SIGNAL)
    
    # Simulate a sequence of AI predictions
    predictions = [
        ("Hour 1", 105.2, 0.75),
        ("Hour 2", 106.1, 0.78),
        ("Hour 3", 107.0, 0.82),
        ("Hour 4", 106.8, 0.85),
        ("Hour 5", 108.5, 0.88),
    ]
    
    print("AI trading signal validation over time:")
    print("-" * 70)
    
    for time, prediction, ai_conf in predictions:
        result = pipeline.process(
            raw_value=prediction,
            raw_confidence=ai_conf,
            context={"symbol": "STOCK", "timeframe": "1h"}
        )
        
        # Trading decision
        if result.safety_status == "ACCEPTED" and not result.used_fallback:
            action = "🟢 EXECUTE TRADE"
        elif result.safety_status == "BORDERLINE":
            action = "🟡 MONITOR"
        else:
            action = "🔴 SKIP"
        
        confidence_trend = "↑" if ai_conf > 0.80 else "→"
        
        print(f"{time} | Prediction: ${prediction:6.2f} | "
              f"AI Conf: {ai_conf:.2f} {confidence_trend} | {action}")
    
    print("\nNote: Trust builds as AI shows consistent, confident predictions")
    print()


def example_adaptive_thresholds():
    """
    Example 4: Different thresholds for different risk levels.
    
    Use case: Content moderation with varying safety requirements.
    """
    print("=" * 70)
    print("Example 4: Adaptive Safety Thresholds")
    print("=" * 70)
    print("\nScenario: Content moderation with risk-based validation")
    print("Goal: Apply stricter checks for sensitive content\n")
    
    # Different configs for different content types
    configs = {
        "general": create_pipeline("balanced"),
        "children": create_pipeline("conservative"),
        "adult": create_pipeline("permissive"),
    }
    
    # Test content with various risk levels
    content_items = [
        ("General news article", 85.0, 0.88, "general"),
        ("Children's educational video", 82.0, 0.85, "children"),
        ("Adult comedy special", 75.0, 0.80, "adult"),
        ("Borderline political content", 70.0, 0.75, "general"),
    ]
    
    print("Content moderation with adaptive thresholds:")
    print("-" * 70)
    
    for content, safety_score, ai_conf, category in content_items:
        pipeline = configs[category]
        
        result = pipeline.process(
            raw_value=safety_score,
            raw_confidence=ai_conf,
            context={"category": category}
        )
        
        if result.safety_status == "ACCEPTED":
            action = "✓ ALLOW"
        elif result.safety_status == "BORDERLINE":
            action = "⚠ FLAG"
        else:
            action = "✗ BLOCK"
        
        print(f"{content:35s} | {category:8s} | "
              f"Score: {safety_score:4.0f} | {action}")
    
    print("\nNote: Children's content requires higher confidence (conservative)")
    print("      Adult content allows more flexibility (permissive)")
    print()


def main():
    """Run all integration examples."""
    print("\n" + "=" * 70)
    print("AILEE + AI Integration: Complete Walkthrough")
    print("=" * 70)
    print("\nThis demo shows how AILEE adds trust validation to AI systems")
    print("across various real-world scenarios.\n")
    
    example_single_model_integration()
    example_multi_model_consensus()
    example_confidence_building()
    example_adaptive_thresholds()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✓ AILEE provides trust validation for any AI output")
    print("✓ Supports single models or multi-model consensus")
    print("✓ Adapts to different risk levels and use cases")
    print("✓ Builds confidence through consistent performance")
    print("✓ Prevents acting on uncertain AI predictions\n")
    
    print("Next Steps:")
    print("  1. Choose your AI framework (OpenAI, Anthropic, HuggingFace)")
    print("  2. Select appropriate AILEE config preset for your use case")
    print("  3. Integrate adapter to extract values and confidence")
    print("  4. Process through AILEE pipeline before acting on AI output")
    print("  5. Monitor and tune thresholds based on your requirements\n")
    
    print("Documentation: docs/AI_INTEGRATION_GUIDE.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
