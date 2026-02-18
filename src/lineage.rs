//! Lineage and verification module for cryptographic proof of generation.
//!
//! Provides deterministic verification using SHA-256 hashing over canonical
//! serialization of inputs and outputs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::generation::GenerationRequest;
use crate::model::ModelOutput;

/// Lineage information tracking the provenance of a generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    /// Models that contributed to the result
    pub contributing_models: Vec<String>,

    /// Timestamp of generation (Unix timestamp)
    pub timestamp: u64,

    /// Cryptographic hash of inputs and outputs (SHA-256)
    pub verification_hash: String,

    /// Request that initiated this generation
    pub request_summary: RequestSummary,

    /// Model outputs summary
    pub outputs_summary: Vec<OutputSummary>,

    /// Replay metadata for deterministic verification
    pub replay_metadata: HashMap<String, String>,
}

impl Lineage {
    /// Create a new empty lineage
    pub fn new() -> Self {
        Self {
            contributing_models: Vec::new(),
            timestamp: 0,
            verification_hash: String::new(),
            request_summary: RequestSummary::default(),
            outputs_summary: Vec::new(),
            replay_metadata: HashMap::new(),
        }
    }

    /// Build lineage from request and outputs
    pub fn build(
        request: &GenerationRequest,
        outputs: &HashMap<String, ModelOutput>,
        final_output: &str,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let contributing_models: Vec<String> = outputs.keys().cloned().collect();

        let request_summary = RequestSummary::from_request(request);

        let outputs_summary: Vec<OutputSummary> = outputs
            .iter()
            .map(|(model_id, output)| OutputSummary::from_output(model_id, output))
            .collect();

        let verification_hash = Self::compute_hash(request, outputs, final_output);

        let mut replay_metadata = HashMap::new();
        replay_metadata.insert("timestamp".to_string(), timestamp.to_string());
        replay_metadata.insert("model_count".to_string(), outputs.len().to_string());

        Self {
            contributing_models,
            timestamp,
            verification_hash,
            request_summary,
            outputs_summary,
            replay_metadata,
        }
    }

    /// Compute SHA-256 hash of inputs and outputs
    fn compute_hash(
        request: &GenerationRequest,
        outputs: &HashMap<String, ModelOutput>,
        final_output: &str,
    ) -> String {
        let mut hasher = Sha256::new();

        // Hash the request in canonical form
        let request_json = serde_json::to_string(request).unwrap_or_default();
        hasher.update(request_json.as_bytes());

        // Hash outputs in sorted order for determinism
        let mut sorted_outputs: Vec<_> = outputs.iter().collect();
        sorted_outputs.sort_by_key(|(model_id, _)| *model_id);

        for (model_id, output) in sorted_outputs {
            hasher.update(model_id.as_bytes());
            hasher.update(output.text.as_bytes());
            if let Some(conf) = output.model_confidence {
                hasher.update(conf.to_string().as_bytes());
            }
        }

        // Hash the final output
        hasher.update(final_output.as_bytes());

        // Return hex-encoded hash
        format!("{:x}", hasher.finalize())
    }

    /// Verify that a result matches the recorded hash
    pub fn verify(
        &self,
        request: &GenerationRequest,
        outputs: &HashMap<String, ModelOutput>,
        final_output: &str,
    ) -> bool {
        let computed_hash = Self::compute_hash(request, outputs, final_output);
        computed_hash == self.verification_hash
    }

    /// Add replay metadata
    pub fn add_replay_metadata(&mut self, key: String, value: String) {
        self.replay_metadata.insert(key, value);
    }
}

impl Default for Lineage {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of the generation request
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestSummary {
    /// Task type
    pub task_type: String,

    /// Trust threshold
    pub trust_threshold: f64,

    /// Execution mode
    pub execution_mode: String,

    /// Determinism level
    pub determinism_level: String,

    /// Prompt hash (for privacy)
    pub prompt_hash: String,
}

impl RequestSummary {
    /// Create summary from request
    fn from_request(request: &GenerationRequest) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(request.prompt.as_bytes());
        let prompt_hash = format!("{:x}", hasher.finalize());

        Self {
            task_type: format!("{:?}", request.task_type),
            trust_threshold: request.trust_threshold,
            execution_mode: format!("{:?}", request.execution_mode),
            determinism_level: format!("{:?}", request.determinism_level),
            prompt_hash,
        }
    }
}

/// Summary of a model output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSummary {
    /// Model ID
    pub model_id: String,

    /// Output hash (for privacy)
    pub output_hash: String,

    /// Model confidence
    pub model_confidence: Option<f64>,

    /// Token count
    pub token_count: Option<usize>,

    /// Latency
    pub latency_ms: Option<u64>,
}

impl OutputSummary {
    /// Create summary from output
    fn from_output(model_id: &str, output: &ModelOutput) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(output.text.as_bytes());
        let output_hash = format!("{:x}", hasher.finalize());

        Self {
            model_id: model_id.to_string(),
            output_hash,
            model_confidence: output.model_confidence,
            token_count: output.token_count,
            latency_ms: output.latency_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{DeterminismLevel, ExecutionMode, TaskType};

    #[test]
    fn test_lineage_creation() {
        let lineage = Lineage::new();
        assert!(lineage.contributing_models.is_empty());
        assert_eq!(lineage.timestamp, 0);
        assert!(lineage.verification_hash.is_empty());
    }

    #[test]
    fn test_lineage_build() {
        let request = GenerationRequest::new("test prompt", TaskType::Chat)
            .with_trust_threshold(0.8)
            .with_execution_mode(ExecutionMode::Local)
            .with_determinism_level(DeterminismLevel::Full);

        let mut outputs = HashMap::new();
        outputs.insert(
            "model1".to_string(),
            ModelOutput::new("output1").with_confidence(0.9),
        );
        outputs.insert(
            "model2".to_string(),
            ModelOutput::new("output2").with_confidence(0.85),
        );

        let lineage = Lineage::build(&request, &outputs, "final output");

        assert_eq!(lineage.contributing_models.len(), 2);
        assert!(!lineage.verification_hash.is_empty());
        assert_eq!(lineage.verification_hash.len(), 64); // SHA-256 hex length
        assert_eq!(lineage.outputs_summary.len(), 2);
    }

    #[test]
    fn test_hash_determinism() {
        let request = GenerationRequest::new("test", TaskType::Code);

        let mut outputs = HashMap::new();
        outputs.insert("m1".to_string(), ModelOutput::new("out1"));
        outputs.insert("m2".to_string(), ModelOutput::new("out2"));

        let hash1 = Lineage::compute_hash(&request, &outputs, "final");
        let hash2 = Lineage::compute_hash(&request, &outputs, "final");

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_sensitivity() {
        let request1 = GenerationRequest::new("prompt1", TaskType::Chat);
        let request2 = GenerationRequest::new("prompt2", TaskType::Chat);

        let outputs = HashMap::new();

        let hash1 = Lineage::compute_hash(&request1, &outputs, "final");
        let hash2 = Lineage::compute_hash(&request2, &outputs, "final");

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_verification() {
        let request = GenerationRequest::new("test", TaskType::Analysis);

        let mut outputs = HashMap::new();
        outputs.insert("m1".to_string(), ModelOutput::new("output1"));

        let lineage = Lineage::build(&request, &outputs, "final output");

        // Verify with same inputs
        assert!(lineage.verify(&request, &outputs, "final output"));

        // Verify with different final output
        assert!(!lineage.verify(&request, &outputs, "different output"));
    }

    #[test]
    fn test_request_summary() {
        let request =
            GenerationRequest::new("test prompt", TaskType::Chat).with_trust_threshold(0.75);

        let summary = RequestSummary::from_request(&request);

        assert_eq!(summary.task_type, "Chat");
        assert_eq!(summary.trust_threshold, 0.75);
        assert!(!summary.prompt_hash.is_empty());
    }

    #[test]
    fn test_output_summary() {
        let output = ModelOutput::new("test output")
            .with_confidence(0.8)
            .with_token_count(50)
            .with_latency(100);

        let summary = OutputSummary::from_output("test-model", &output);

        assert_eq!(summary.model_id, "test-model");
        assert!(!summary.output_hash.is_empty());
        assert_eq!(summary.model_confidence, Some(0.8));
        assert_eq!(summary.token_count, Some(50));
        assert_eq!(summary.latency_ms, Some(100));
    }

    #[test]
    fn test_replay_metadata() {
        let mut lineage = Lineage::new();
        lineage.add_replay_metadata("key1".to_string(), "value1".to_string());
        lineage.add_replay_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(
            lineage.replay_metadata.get("key1"),
            Some(&"value1".to_string())
        );
        assert_eq!(
            lineage.replay_metadata.get("key2"),
            Some(&"value2".to_string())
        );
    }
}
