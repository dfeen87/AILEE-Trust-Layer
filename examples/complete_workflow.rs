//! Example: Complete AILEE Trust Layer workflow
//!
//! This example demonstrates the full end-to-end workflow of the AILEE Trust Layer,
//! from generation request to verified, consensus-based results.

use ailee_trust_core::prelude::*;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};

// Example implementation of a model adapter for a local code generation model
struct LocalCodeModel {
    name: String,
}

#[async_trait]
impl ModelAdapter for LocalCodeModel {
    async fn generate(
        &self,
        prompt: &str,
    ) -> Result<ModelOutput, ailee_trust_core::model::ModelError> {
        // Simulate code generation
        let code = if prompt.contains("hello world") || prompt.contains("Hello World") {
            "def hello_world():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    hello_world()"
        } else if prompt.contains("fibonacci") {
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        } else {
            "# Code generation based on prompt\npass"
        };

        // Simulate latency
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        Ok(ModelOutput::new(code)
            .with_confidence(0.92)
            .with_token_count(code.split_whitespace().count())
            .with_latency(50))
    }

    fn model_id(&self) -> &str {
        &self.name
    }

    fn model_capabilities(&self) -> &HashSet<ModelCapability> {
        static CAPS: std::sync::OnceLock<HashSet<ModelCapability>> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| {
            let mut caps = HashSet::new();
            caps.insert(ModelCapability::Code);
            caps
        })
    }

    fn model_locality(&self) -> ModelLocality {
        ModelLocality::Local
    }
}

// Example implementation of a remote analysis model
struct RemoteAnalysisModel {
    name: String,
}

#[async_trait]
impl ModelAdapter for RemoteAnalysisModel {
    async fn generate(
        &self,
        prompt: &str,
    ) -> Result<ModelOutput, ailee_trust_core::model::ModelError> {
        // Simulate remote analysis
        let analysis = format!(
            "Analysis of the prompt '{}': This appears to be a request for code generation. \
             Recommendation: Use a structured approach with proper error handling.",
            prompt.chars().take(50).collect::<String>()
        );

        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

        Ok(ModelOutput::new(analysis)
            .with_confidence(0.88)
            .with_token_count(30)
            .with_latency(150))
    }

    fn model_id(&self) -> &str {
        &self.name
    }

    fn model_capabilities(&self) -> &HashSet<ModelCapability> {
        static CAPS: std::sync::OnceLock<HashSet<ModelCapability>> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| {
            let mut caps = HashSet::new();
            caps.insert(ModelCapability::Analysis);
            caps
        })
    }

    fn model_locality(&self) -> ModelLocality {
        ModelLocality::Remote
    }
}

async fn generate_with_trust_layer() -> GenerationResult {
    println!("=== AILEE Trust Layer Example ===\n");

    // Step 1: Create a generation request
    println!("Step 1: Creating generation request...");
    let request = GenerationRequest::new("Write a Python hello world function", TaskType::Code)
        .with_trust_threshold(0.75)
        .with_execution_mode(ExecutionMode::Hybrid)
        .with_determinism_level(DeterminismLevel::BestEffort)
        .with_context("language", "Python")
        .with_context("user_id", "example_user");

    println!("  - Task: {:?}", request.task_type);
    println!("  - Trust threshold: {:.2}", request.trust_threshold);
    println!("  - Execution mode: {:?}", request.execution_mode);
    println!();

    // Step 2: Initialize models
    println!("Step 2: Initializing models...");
    let models: Vec<Box<dyn ModelAdapter>> = vec![
        Box::new(LocalCodeModel {
            name: "local-codegen-v1".to_string(),
        }),
        Box::new(LocalCodeModel {
            name: "local-codegen-v2".to_string(),
        }),
        Box::new(RemoteAnalysisModel {
            name: "remote-analyzer-v1".to_string(),
        }),
    ];

    println!("  - {} models initialized", models.len());
    for model in &models {
        println!("    - {} ({:?})", model.model_id(), model.model_locality());
    }
    println!();

    // Step 3: Generate outputs from all models
    println!("Step 3: Generating outputs from models...");
    let mut outputs = HashMap::new();

    for model in models {
        match model.generate(&request.prompt).await {
            Ok(output) => {
                println!(
                    "  - {} generated {} tokens (confidence: {:.2})",
                    model.model_id(),
                    output.token_count.unwrap_or(0),
                    output.model_confidence.unwrap_or(0.0)
                );
                outputs.insert(model.model_id().to_string(), output);
            }
            Err(e) => {
                println!("  - {} failed: {}", model.model_id(), e);
            }
        }
    }
    println!();

    // Step 4: Compute trust scores
    println!("Step 4: Computing trust scores...");
    let mut trust_scorer = TrustScorer::new();
    let trust_scores = trust_scorer.score_outputs(&outputs);

    println!("  Model Trust Scores:");
    println!(
        "  {:<25} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Model", "Confidence", "Safety", "Consistency", "Determinism", "Aggregate"
    );
    println!("  {}", "-".repeat(85));

    for (model_id, score) in &trust_scores {
        println!(
            "  {:<25} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            model_id,
            score.confidence_score,
            score.safety_score,
            score.consistency_score,
            score.determinism_score,
            score.aggregate_score
        );
    }
    println!();

    // Step 5: Reach consensus
    println!("Step 5: Reaching consensus...");
    let consensus_engine = ConsensusEngine::new(ConsensusStrategy::HighestTrust)
        .with_trust_threshold(request.trust_threshold)
        .with_min_models(1);

    let consensus_result = consensus_engine.reach_consensus(&outputs, &trust_scores);

    println!("  - Strategy: {:?}", consensus_result.metadata.strategy);
    println!(
        "  - Consensus achieved: {}",
        consensus_result.metadata.consensus_achieved
    );
    println!(
        "  - Agreement ratio: {:.2}",
        consensus_result.metadata.agreement_ratio
    );
    println!("  - Reason: {}", consensus_result.metadata.reason);
    println!();

    // Step 6: Build lineage and verification
    println!("Step 6: Building lineage and verification...");
    let lineage = Lineage::build(&request, &outputs, &consensus_result.output);

    println!(
        "  - Contributing models: {}",
        lineage.contributing_models.len()
    );
    println!(
        "  - Verification hash: {}",
        &lineage.verification_hash[..16]
    );
    println!("  - Timestamp: {}", lineage.timestamp);
    println!(
        "  - Lineage verified: {}",
        lineage.verify(&request, &outputs, &consensus_result.output)
    );
    println!();

    // Step 7: Create final result
    println!("Step 7: Creating final generation result...");
    let mut execution_metadata = HashMap::new();
    execution_metadata.insert("version".to_string(), "1.0".to_string());
    execution_metadata.insert(
        "strategy".to_string(),
        format!("{:?}", consensus_result.metadata.strategy),
    );

    let result = GenerationResult {
        final_output: consensus_result.output.clone(),
        aggregate_trust_score: consensus_result.trust_score,
        model_trust_scores: trust_scores,
        lineage,
        execution_metadata,
    };

    println!(
        "  - Aggregate trust score: {:.3}",
        result.aggregate_trust_score
    );
    println!(
        "  - Meets threshold: {}",
        result.meets_threshold(request.trust_threshold)
    );
    println!();

    // Step 8: Display final output
    println!("=== Final Output ===");
    println!("{}", result.final_output);
    println!("\n=== Generation Complete ===");

    result
}

#[tokio::main]
async fn main() {
    let result = generate_with_trust_layer().await;

    // The result is now ready to be used by the application
    assert!(result.aggregate_trust_score > 0.0);
    assert!(!result.final_output.is_empty());
}
