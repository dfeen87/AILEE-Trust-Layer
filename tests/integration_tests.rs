//! Integration tests for the AILEE Trust Layer
//!
//! Tests demonstrate complete workflows from generation request to verified result

use ailee_trust_core::prelude::*;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};

// Mock local model for testing
struct MockLocalModel {
    id: String,
    response: String,
    confidence: f64,
}

#[async_trait]
impl ModelAdapter for MockLocalModel {
    async fn generate(
        &self,
        _prompt: &str,
    ) -> Result<ModelOutput, ailee_trust_core::model::ModelError> {
        Ok(ModelOutput::new(&self.response)
            .with_confidence(self.confidence)
            .with_token_count(self.response.split_whitespace().count())
            .with_latency(50))
    }

    fn model_id(&self) -> &str {
        &self.id
    }

    fn model_capabilities(&self) -> &HashSet<ModelCapability> {
        static CAPS: std::sync::OnceLock<HashSet<ModelCapability>> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| {
            let mut caps = HashSet::new();
            caps.insert(ModelCapability::Chat);
            caps.insert(ModelCapability::Code);
            caps
        })
    }

    fn model_locality(&self) -> ModelLocality {
        ModelLocality::Local
    }
}

#[tokio::test]
async fn test_end_to_end_generation_workflow() {
    // 1. Create a generation request
    let request = GenerationRequest::new("Write a hello world function in Python", TaskType::Code)
        .with_trust_threshold(0.7)
        .with_execution_mode(ExecutionMode::Local)
        .with_determinism_level(DeterminismLevel::Full);

    // 2. Create mock models
    let model1 = MockLocalModel {
        id: "model1".to_string(),
        response: "def hello_world():\n    print('Hello, World!')".to_string(),
        confidence: 0.95,
    };

    let model2 = MockLocalModel {
        id: "model2".to_string(),
        response: "def hello_world():\n    print('Hello, World!')".to_string(),
        confidence: 0.90,
    };

    let model3 = MockLocalModel {
        id: "model3".to_string(),
        response: "def hello():\n    return 'Hello, World!'".to_string(),
        confidence: 0.85,
    };

    // 3. Generate outputs from all models
    let mut outputs = HashMap::new();
    outputs.insert(
        model1.model_id().to_string(),
        model1.generate(&request.prompt).await.unwrap(),
    );
    outputs.insert(
        model2.model_id().to_string(),
        model2.generate(&request.prompt).await.unwrap(),
    );
    outputs.insert(
        model3.model_id().to_string(),
        model3.generate(&request.prompt).await.unwrap(),
    );

    // 4. Compute trust scores
    let mut trust_scorer = TrustScorer::new();
    let trust_scores = trust_scorer.score_outputs(&outputs);

    // Verify trust scores exist for all models
    assert_eq!(trust_scores.len(), 3);
    for (model_id, score) in &trust_scores {
        println!(
            "Model {}: confidence={:.3}, safety={:.3}, consistency={:.3}, aggregate={:.3}",
            model_id,
            score.confidence_score,
            score.safety_score,
            score.consistency_score,
            score.aggregate_score
        );
        assert!(score.aggregate_score >= 0.0 && score.aggregate_score <= 1.0);
    }

    // 5. Reach consensus
    let consensus_engine = ConsensusEngine::new(ConsensusStrategy::MajorityVote)
        .with_trust_threshold(0.5)
        .with_min_models(2);

    let consensus_result = consensus_engine.reach_consensus(&outputs, &trust_scores);

    // Verify consensus was achieved
    assert!(consensus_result.metadata.consensus_achieved);
    assert!(!consensus_result.output.is_empty());
    println!("Consensus output: {}", consensus_result.output);
    println!("Consensus reason: {}", consensus_result.metadata.reason);

    // 6. Build lineage
    let lineage = Lineage::build(&request, &outputs, &consensus_result.output);

    // Verify lineage
    assert_eq!(lineage.contributing_models.len(), 3);
    assert!(!lineage.verification_hash.is_empty());
    assert_eq!(lineage.verification_hash.len(), 64); // SHA-256 hex
    println!("Verification hash: {}", lineage.verification_hash);

    // 7. Verify the lineage
    assert!(lineage.verify(&request, &outputs, &consensus_result.output));

    // 8. Build final result
    let mut execution_metadata = HashMap::new();
    execution_metadata.insert("consensus_strategy".to_string(), "MajorityVote".to_string());

    let final_result = GenerationResult {
        final_output: consensus_result.output,
        aggregate_trust_score: consensus_result.trust_score,
        model_trust_scores: trust_scores,
        lineage,
        execution_metadata,
    };

    // Verify final result meets threshold
    assert!(final_result.meets_threshold(request.trust_threshold));
}

#[tokio::test]
async fn test_deterministic_replay() {
    // Test that the same inputs produce the same hash
    let request = GenerationRequest::new("test prompt", TaskType::Chat).with_trust_threshold(0.8);

    let mut outputs = HashMap::new();
    outputs.insert(
        "m1".to_string(),
        ModelOutput::new("response 1").with_confidence(0.9),
    );
    outputs.insert(
        "m2".to_string(),
        ModelOutput::new("response 2").with_confidence(0.85),
    );

    let final_output = "response 1";

    // Build lineage twice
    let lineage1 = Lineage::build(&request, &outputs, final_output);
    let lineage2 = Lineage::build(&request, &outputs, final_output);

    // Hashes should be identical (deterministic)
    assert_eq!(lineage1.verification_hash, lineage2.verification_hash);

    // Both should verify
    assert!(lineage1.verify(&request, &outputs, final_output));
    assert!(lineage2.verify(&request, &outputs, final_output));
}

#[tokio::test]
async fn test_trust_threshold_enforcement() {
    // Test that low-trust outputs are filtered
    let _request = GenerationRequest::new("test", TaskType::Analysis).with_trust_threshold(0.9); // High threshold

    let mut outputs = HashMap::new();
    outputs.insert(
        "low_trust".to_string(),
        ModelOutput::new("error: failed to process").with_confidence(0.3),
    );
    outputs.insert(
        "high_trust".to_string(),
        ModelOutput::new("Successfully analyzed the data").with_confidence(0.95),
    );

    let mut trust_scorer = TrustScorer::new();
    let trust_scores = trust_scorer.score_outputs(&outputs);

    let consensus_engine =
        ConsensusEngine::new(ConsensusStrategy::HighestTrust).with_trust_threshold(0.9);

    let result = consensus_engine.reach_consensus(&outputs, &trust_scores);

    // Should select high-trust output
    assert!(result.output.contains("Successfully"));
}

#[tokio::test]
async fn test_offline_execution() {
    // Test that the system works without network (all local models)
    let request = GenerationRequest::new("Offline test", TaskType::Code)
        .with_execution_mode(ExecutionMode::Local);

    let local_model = MockLocalModel {
        id: "offline-model".to_string(),
        response: "fn main() { println!(\"Offline works!\"); }".to_string(),
        confidence: 0.85,
    };

    let mut outputs = HashMap::new();
    outputs.insert(
        local_model.model_id().to_string(),
        local_model.generate(&request.prompt).await.unwrap(),
    );

    let mut trust_scorer = TrustScorer::new();
    let trust_scores = trust_scorer.score_outputs(&outputs);

    let consensus_engine = ConsensusEngine::new(ConsensusStrategy::HighestTrust);
    let result = consensus_engine.reach_consensus(&outputs, &trust_scores);

    // Should work offline
    assert!(!result.output.is_empty());
    assert!(result.output.contains("Offline works!"));
}

#[tokio::test]
async fn test_model_disagreement_handling() {
    // Test consensus when models disagree
    let _request = GenerationRequest::new("What is 2+2?", TaskType::Chat);

    let mut outputs = HashMap::new();
    outputs.insert(
        "correct1".to_string(),
        ModelOutput::new("The answer is 4").with_confidence(0.95),
    );
    outputs.insert(
        "correct2".to_string(),
        ModelOutput::new("2 + 2 equals 4").with_confidence(0.90),
    );
    outputs.insert(
        "incorrect".to_string(),
        ModelOutput::new("The answer is 5").with_confidence(0.60),
    );

    let mut trust_scorer = TrustScorer::new();
    let trust_scores = trust_scorer.score_outputs(&outputs);

    // Use majority vote
    let consensus_engine =
        ConsensusEngine::new(ConsensusStrategy::MajorityVote).with_trust_threshold(0.5);

    let result = consensus_engine.reach_consensus(&outputs, &trust_scores);

    // Majority should win (similar answers about "4")
    assert!(result.output.contains("4"));
    assert_eq!(result.metadata.strategy, ConsensusStrategy::MajorityVote);
}

#[tokio::test]
async fn test_consistency_scoring_over_time() {
    // Test that consistency scores improve with similar outputs
    let mut trust_scorer = TrustScorer::new();

    // First output - baseline
    let output1 = ModelOutput::new("The sky is blue");
    let score1 = trust_scorer.score_output(&output1);
    assert_eq!(score1.consistency_score, 0.5); // Neutral with no history

    // Similar output - should have high consistency
    let output2 = ModelOutput::new("The sky is blue today");
    let score2 = trust_scorer.score_output(&output2);
    assert!(score2.consistency_score > 0.5);

    // Another similar output - should maintain high consistency
    let output3 = ModelOutput::new("The sky is blue and clear");
    let score3 = trust_scorer.score_output(&output3);
    assert!(score3.consistency_score > 0.5);

    // Completely different - lower consistency
    let output4 = ModelOutput::new("Cats are mammals");
    let score4 = trust_scorer.score_output(&output4);
    assert!(score4.consistency_score < score2.consistency_score);
}

#[tokio::test]
async fn test_degraded_mode_with_no_trusted_models() {
    // Test that system degrades gracefully when no models meet threshold
    let _request = GenerationRequest::new("test", TaskType::Code).with_trust_threshold(0.99); // Unrealistically high

    let mut outputs = HashMap::new();
    outputs.insert(
        "model1".to_string(),
        ModelOutput::new("output").with_confidence(0.5),
    );

    let mut trust_scorer = TrustScorer::new();
    let trust_scores = trust_scorer.score_outputs(&outputs);

    let consensus_engine =
        ConsensusEngine::new(ConsensusStrategy::HighestTrust).with_trust_threshold(0.99);

    let result = consensus_engine.reach_consensus(&outputs, &trust_scores);

    // Should degrade gracefully
    assert!(!result.output.is_empty());
    assert!(result.metadata.reason.contains("Degraded"));
}
