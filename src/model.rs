//! Model adapter trait and related types for integrating AI models with AILEE.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

/// Model capability flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCapability {
    /// Can generate chat responses
    Chat,
    /// Can generate code
    Code,
    /// Can perform analysis
    Analysis,
    /// Supports streaming output
    Streaming,
    /// Supports function calling
    FunctionCalling,
}

/// Model locality information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelLocality {
    /// Model runs locally
    Local,
    /// Model runs remotely
    Remote,
}

/// Output from a model generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    /// The generated text
    pub text: String,

    /// Model's self-reported confidence (0.0 to 1.0)
    pub model_confidence: Option<f64>,

    /// Token count or length metric
    pub token_count: Option<usize>,

    /// Generation latency in milliseconds
    pub latency_ms: Option<u64>,

    /// Additional metadata from the model
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl ModelOutput {
    /// Create a new model output with just text
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model_confidence: None,
            token_count: None,
            latency_ms: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the model confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.model_confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set the token count
    pub fn with_token_count(mut self, count: usize) -> Self {
        self.token_count = Some(count);
        self
    }

    /// Set the latency
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }
}

/// Errors that can occur during model generation
#[derive(Error, Debug)]
pub enum ModelError {
    /// Model is not available
    #[error("Model not available: {0}")]
    NotAvailable(String),

    /// Generation failed
    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Timeout
    #[error("Generation timeout")]
    Timeout,

    /// Other error
    #[error("Other error: {0}")]
    Other(String),
}

/// Trait for model adapters that integrate AI models with AILEE
#[async_trait]
pub trait ModelAdapter: Send + Sync {
    /// Generate output from a prompt
    async fn generate(&self, prompt: &str) -> Result<ModelOutput, ModelError>;

    /// Get the unique model identifier
    fn model_id(&self) -> &str;

    /// Get the model's capabilities
    fn model_capabilities(&self) -> &HashSet<ModelCapability>;

    /// Get the model's locality (local or remote)
    fn model_locality(&self) -> ModelLocality;

    /// Check if the model supports a specific capability
    fn supports_capability(&self, capability: ModelCapability) -> bool {
        self.model_capabilities().contains(&capability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock model adapter for testing
    struct MockModel {
        id: String,
        capabilities: HashSet<ModelCapability>,
        locality: ModelLocality,
    }

    #[async_trait]
    impl ModelAdapter for MockModel {
        async fn generate(&self, prompt: &str) -> Result<ModelOutput, ModelError> {
            Ok(ModelOutput::new(format!("Response to: {}", prompt))
                .with_confidence(0.9)
                .with_token_count(10))
        }

        fn model_id(&self) -> &str {
            &self.id
        }

        fn model_capabilities(&self) -> &HashSet<ModelCapability> {
            &self.capabilities
        }

        fn model_locality(&self) -> ModelLocality {
            self.locality
        }
    }

    #[tokio::test]
    async fn test_mock_model_generation() {
        let mut caps = HashSet::new();
        caps.insert(ModelCapability::Chat);

        let model = MockModel {
            id: "test-model".to_string(),
            capabilities: caps,
            locality: ModelLocality::Local,
        };

        let output = model.generate("test prompt").await.unwrap();
        assert_eq!(output.text, "Response to: test prompt");
        assert_eq!(output.model_confidence, Some(0.9));
        assert_eq!(output.token_count, Some(10));
    }

    #[test]
    fn test_model_output_builder() {
        let output = ModelOutput::new("test output")
            .with_confidence(0.95)
            .with_token_count(20)
            .with_latency(150);

        assert_eq!(output.text, "test output");
        assert_eq!(output.model_confidence, Some(0.95));
        assert_eq!(output.token_count, Some(20));
        assert_eq!(output.latency_ms, Some(150));
    }

    #[test]
    fn test_confidence_clamping() {
        let output = ModelOutput::new("test").with_confidence(1.5);
        assert_eq!(output.model_confidence, Some(1.0));

        let output = ModelOutput::new("test").with_confidence(-0.5);
        assert_eq!(output.model_confidence, Some(0.0));
    }
}
