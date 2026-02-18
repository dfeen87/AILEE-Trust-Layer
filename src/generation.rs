//! Generation request and result types for the AILEE Trust Layer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::lineage::Lineage;
use crate::trust::TrustScore;

/// Type of generative task being requested
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Conversational chat response
    Chat,
    /// Code generation or completion
    Code,
    /// Data analysis and interpretation
    Analysis,
}

/// Execution mode for the generation request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Execute only on local models
    Local,
    /// Execute only on remote models
    Remote,
    /// Execute on both local and remote models
    Hybrid,
}

/// Determinism level for generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeterminismLevel {
    /// Fully deterministic (same input → same output)
    Full,
    /// Best-effort deterministic
    BestEffort,
    /// Non-deterministic (allows sampling)
    None,
}

/// A request for generative AI output with trust constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// The prompt or input for generation
    pub prompt: String,

    /// Type of task being requested
    pub task_type: TaskType,

    /// Minimum acceptable trust score (0.0 to 1.0)
    pub trust_threshold: f64,

    /// Execution mode (local, remote, or hybrid)
    pub execution_mode: ExecutionMode,

    /// Determinism level
    pub determinism_level: DeterminismLevel,

    /// Additional context or metadata
    #[serde(default)]
    pub context: HashMap<String, String>,
}

impl GenerationRequest {
    /// Create a new generation request
    pub fn new(prompt: impl Into<String>, task_type: TaskType) -> Self {
        Self {
            prompt: prompt.into(),
            task_type,
            trust_threshold: 0.7,
            execution_mode: ExecutionMode::Hybrid,
            determinism_level: DeterminismLevel::BestEffort,
            context: HashMap::new(),
        }
    }

    /// Set the trust threshold
    pub fn with_trust_threshold(mut self, threshold: f64) -> Self {
        self.trust_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the execution mode
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Set the determinism level
    pub fn with_determinism_level(mut self, level: DeterminismLevel) -> Self {
        self.determinism_level = level;
        self
    }

    /// Add context metadata
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// The final result of a generation request with trust scoring and lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// The final selected or synthesized output
    pub final_output: String,

    /// Aggregate trust score across all dimensions
    pub aggregate_trust_score: f64,

    /// Per-model trust scores
    pub model_trust_scores: HashMap<String, TrustScore>,

    /// Model lineage information
    pub lineage: Lineage,

    /// Execution metadata
    pub execution_metadata: HashMap<String, String>,
}

impl GenerationResult {
    /// Check if the result meets the trust threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.aggregate_trust_score >= threshold
    }

    /// Get the trust score for a specific model
    pub fn model_score(&self, model_id: &str) -> Option<&TrustScore> {
        self.model_trust_scores.get(model_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_request_builder() {
        let request = GenerationRequest::new("test prompt", TaskType::Chat)
            .with_trust_threshold(0.8)
            .with_execution_mode(ExecutionMode::Local)
            .with_determinism_level(DeterminismLevel::Full)
            .with_context("user_id", "123");

        assert_eq!(request.prompt, "test prompt");
        assert_eq!(request.task_type, TaskType::Chat);
        assert_eq!(request.trust_threshold, 0.8);
        assert_eq!(request.execution_mode, ExecutionMode::Local);
        assert_eq!(request.determinism_level, DeterminismLevel::Full);
        assert_eq!(request.context.get("user_id"), Some(&"123".to_string()));
    }

    #[test]
    fn test_trust_threshold_clamping() {
        let request = GenerationRequest::new("test", TaskType::Code).with_trust_threshold(1.5);
        assert_eq!(request.trust_threshold, 1.0);

        let request = GenerationRequest::new("test", TaskType::Code).with_trust_threshold(-0.5);
        assert_eq!(request.trust_threshold, 0.0);
    }

    #[test]
    fn test_generation_result_threshold_check() {
        let result = GenerationResult {
            final_output: "test".to_string(),
            aggregate_trust_score: 0.85,
            model_trust_scores: HashMap::new(),
            lineage: Lineage::new(),
            execution_metadata: HashMap::new(),
        };

        assert!(result.meets_threshold(0.8));
        assert!(!result.meets_threshold(0.9));
    }
}
