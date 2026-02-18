//! Trust scoring module for evaluating model outputs.
//!
//! Provides multi-dimensional trust evaluation using deterministic,
//! dependency-light similarity methods.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::ModelOutput;

/// Multi-dimensional trust score for a model output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScore {
    /// Confidence score (0.0 to 1.0)
    pub confidence_score: f64,

    /// Safety score (0.0 to 1.0)
    pub safety_score: f64,

    /// Consistency score (0.0 to 1.0)
    pub consistency_score: f64,

    /// Determinism score (0.0 to 1.0)
    pub determinism_score: f64,

    /// Aggregate score (weighted combination)
    pub aggregate_score: f64,
}

impl TrustScore {
    /// Create a new trust score with all dimensions
    pub fn new(confidence: f64, safety: f64, consistency: f64, determinism: f64) -> Self {
        let aggregate = Self::compute_aggregate(confidence, safety, consistency, determinism);
        Self {
            confidence_score: confidence,
            safety_score: safety,
            consistency_score: consistency,
            determinism_score: determinism,
            aggregate_score: aggregate,
        }
    }

    /// Compute aggregate score using weighted combination
    fn compute_aggregate(confidence: f64, safety: f64, consistency: f64, determinism: f64) -> f64 {
        // Weighted average: prioritize safety, then consistency, then confidence
        const SAFETY_WEIGHT: f64 = 0.35;
        const CONSISTENCY_WEIGHT: f64 = 0.30;
        const CONFIDENCE_WEIGHT: f64 = 0.20;
        const DETERMINISM_WEIGHT: f64 = 0.15;

        (safety * SAFETY_WEIGHT
            + consistency * CONSISTENCY_WEIGHT
            + confidence * CONFIDENCE_WEIGHT
            + determinism * DETERMINISM_WEIGHT)
            .clamp(0.0, 1.0)
    }
}

/// Trust scorer for evaluating model outputs
pub struct TrustScorer {
    /// Historical outputs for consistency comparison
    history: Vec<String>,

    /// Maximum history size
    max_history: usize,
}

impl TrustScorer {
    /// Create a new trust scorer
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Create a scorer with custom history size
    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    /// Score a single model output
    pub fn score_output(&mut self, output: &ModelOutput) -> TrustScore {
        let confidence = self.compute_confidence(output);
        let safety = self.compute_safety(output);
        let consistency = self.compute_consistency(output);
        let determinism = self.compute_determinism(output);

        // Add to history for future consistency checks
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(output.text.clone());

        TrustScore::new(confidence, safety, consistency, determinism)
    }

    /// Score multiple outputs and compute aggregate
    pub fn score_outputs(
        &mut self,
        outputs: &HashMap<String, ModelOutput>,
    ) -> HashMap<String, TrustScore> {
        outputs
            .iter()
            .map(|(model_id, output)| (model_id.clone(), self.score_output(output)))
            .collect()
    }

    /// Compute confidence score from model output
    fn compute_confidence(&self, output: &ModelOutput) -> f64 {
        // Use model's self-reported confidence if available
        if let Some(conf) = output.model_confidence {
            return conf;
        }

        // Fall back to heuristic based on output characteristics
        let text_length = output.text.len();
        if text_length == 0 {
            return 0.0;
        }

        // Longer, more detailed responses generally indicate higher confidence
        // But cap at reasonable lengths to avoid bias
        let length_score = (text_length as f64 / 500.0).min(1.0);

        // Presence of hedging words reduces confidence
        let hedging_penalty = self.count_hedging_words(&output.text) as f64 * 0.1;

        (length_score - hedging_penalty).clamp(0.0, 1.0)
    }

    /// Compute safety score based on content analysis
    fn compute_safety(&self, output: &ModelOutput) -> f64 {
        let text = output.text.to_lowercase();

        // Check for potentially unsafe patterns
        let unsafe_patterns = [
            "error",
            "exception",
            "failed",
            "invalid",
            "corrupt",
            "unsafe",
            "dangerous",
            "harmful",
            "malicious",
        ];

        let unsafe_count = unsafe_patterns
            .iter()
            .filter(|&&pattern| text.contains(pattern))
            .count();

        // Higher unsafe count = lower safety score
        (1.0 - (unsafe_count as f64 * 0.1)).clamp(0.0, 1.0)
    }

    /// Compute consistency score by comparing with historical outputs
    fn compute_consistency(&self, output: &ModelOutput) -> f64 {
        if self.history.is_empty() {
            return 0.5; // Neutral score with no history
        }

        // Compare with recent history using token overlap
        let recent_history: Vec<_> = self.history.iter().rev().take(5).collect();

        if recent_history.is_empty() {
            return 0.5;
        }

        let similarities: Vec<f64> = recent_history
            .iter()
            .map(|hist| token_similarity(&output.text, hist))
            .collect();

        // Average similarity to recent outputs
        let avg_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;
        avg_similarity
    }

    /// Compute determinism score based on output characteristics
    fn compute_determinism(&self, output: &ModelOutput) -> f64 {
        // If latency is highly variable, determinism is lower
        let latency_score = if let Some(_latency) = output.latency_ms {
            // Consistent latency indicates deterministic execution
            0.8
        } else {
            0.5
        };

        // Check for random/nondeterministic markers
        let text_lower = output.text.to_lowercase();
        let nondeterministic_markers = ["random", "varies", "depends on timing"];

        let has_nondeterministic = nondeterministic_markers
            .iter()
            .any(|&marker| text_lower.contains(marker));

        if has_nondeterministic {
            latency_score * 0.5
        } else {
            latency_score
        }
    }

    /// Count hedging words in text
    fn count_hedging_words(&self, text: &str) -> usize {
        let hedging_words = [
            "maybe",
            "perhaps",
            "possibly",
            "might",
            "could",
            "uncertain",
            "unsure",
            "probably",
            "likely",
        ];

        let text_lower = text.to_lowercase();
        hedging_words
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count()
    }
}

impl Default for TrustScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute token-based similarity between two texts using Jaccard similarity
fn token_similarity(text1: &str, text2: &str) -> f64 {
    let text1_lower = text1.to_lowercase();
    let text2_lower = text2.to_lowercase();

    let tokens1: std::collections::HashSet<_> = text1_lower.split_whitespace().collect();

    let tokens2: std::collections::HashSet<_> = text2_lower.split_whitespace().collect();

    if tokens1.is_empty() && tokens2.is_empty() {
        return 1.0;
    }

    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.0;
    }

    let intersection = tokens1.intersection(&tokens2).count();
    let union = tokens1.union(&tokens2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute normalized edit distance between two texts
#[allow(dead_code)]
fn normalized_edit_distance(text1: &str, text2: &str) -> f64 {
    let len1 = text1.len();
    let len2 = text2.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }

    let max_len = len1.max(len2);
    if max_len == 0 {
        return 1.0;
    }

    let distance = levenshtein_distance(text1, text2);
    1.0 - (distance as f64 / max_len as f64)
}

/// Compute Levenshtein distance between two strings
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
        row[0] = i;
    }
    for (j, item) in matrix[0].iter_mut().enumerate().take(len2 + 1) {
        *item = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                .min(matrix[i + 1][j] + 1)
                .min(matrix[i][j] + cost);
        }
    }

    matrix[len1][len2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_score_creation() {
        let score = TrustScore::new(0.9, 0.8, 0.7, 0.85);
        assert_eq!(score.confidence_score, 0.9);
        assert_eq!(score.safety_score, 0.8);
        assert_eq!(score.consistency_score, 0.7);
        assert_eq!(score.determinism_score, 0.85);
        assert!(score.aggregate_score > 0.0 && score.aggregate_score <= 1.0);
    }

    #[test]
    fn test_token_similarity() {
        let text1 = "hello world";
        let text2 = "hello world";
        assert_eq!(token_similarity(text1, text2), 1.0);

        let text1 = "hello world";
        let text2 = "goodbye world";
        let sim = token_similarity(text1, text2);
        assert!(sim > 0.0 && sim < 1.0);

        let text1 = "completely different";
        let text2 = "nothing alike";
        let sim = token_similarity(text1, text2);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("", "test"), 4);
    }

    #[test]
    fn test_trust_scorer_confidence() {
        let mut scorer = TrustScorer::new();

        // High confidence with model-reported score
        let output = ModelOutput::new("test output").with_confidence(0.95);
        let score = scorer.score_output(&output);
        assert_eq!(score.confidence_score, 0.95);

        // Confidence based on heuristics
        let output = ModelOutput::new("A detailed response without hedging");
        let score = scorer.score_output(&output);
        assert!(score.confidence_score > 0.0);
    }

    #[test]
    fn test_trust_scorer_safety() {
        let mut scorer = TrustScorer::new();

        // Safe output
        let output = ModelOutput::new("This is a safe and valid response");
        let score = scorer.score_output(&output);
        assert!(score.safety_score > 0.8);

        // Unsafe output
        let output = ModelOutput::new("Error: failed to process dangerous input");
        let score = scorer.score_output(&output);
        assert!(score.safety_score < 0.8);
    }

    #[test]
    fn test_trust_scorer_consistency() {
        let mut scorer = TrustScorer::new();

        // First output - no history
        let output1 = ModelOutput::new("The sky is blue");
        let score1 = scorer.score_output(&output1);
        assert_eq!(score1.consistency_score, 0.5); // Neutral with no history

        // Similar output - high consistency
        let output2 = ModelOutput::new("The sky is blue today");
        let score2 = scorer.score_output(&output2);
        assert!(score2.consistency_score > 0.5);

        // Different output - lower consistency
        let output3 = ModelOutput::new("Completely different topic about cars");
        let score3 = scorer.score_output(&output3);
        assert!(score3.consistency_score < score2.consistency_score);
    }
}
