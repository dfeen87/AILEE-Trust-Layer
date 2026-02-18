//! Consensus engine for selecting or synthesizing final outputs.
//!
//! Provides deterministic consensus mechanisms with explainable metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::ModelOutput;
use crate::trust::TrustScore;

/// Strategy for consensus selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusStrategy {
    /// Select the output with highest trust score
    HighestTrust,

    /// Select the most common output (majority vote)
    MajorityVote,

    /// Synthesize from multiple outputs
    Synthesize,

    /// Weighted combination based on trust scores
    WeightedCombination,
}

/// Metadata explaining consensus decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMetadata {
    /// Strategy used
    pub strategy: ConsensusStrategy,

    /// Reason for selection
    pub reason: String,

    /// Models that participated
    pub participating_models: Vec<String>,

    /// Models that agreed with final output
    pub agreeing_models: Vec<String>,

    /// Agreement ratio (0.0 to 1.0)
    pub agreement_ratio: f64,

    /// Whether consensus was achieved
    pub consensus_achieved: bool,

    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl ConsensusMetadata {
    /// Create new consensus metadata
    pub fn new(strategy: ConsensusStrategy, reason: String) -> Self {
        Self {
            strategy,
            reason,
            participating_models: Vec::new(),
            agreeing_models: Vec::new(),
            agreement_ratio: 0.0,
            consensus_achieved: false,
            metadata: HashMap::new(),
        }
    }

    /// Set participating models
    pub fn with_participants(mut self, models: Vec<String>) -> Self {
        self.participating_models = models;
        self
    }

    /// Set agreeing models
    pub fn with_agreeing(mut self, models: Vec<String>) -> Self {
        self.agreeing_models = models;
        self
    }

    /// Compute and set agreement ratio
    pub fn with_agreement_ratio(mut self) -> Self {
        if !self.participating_models.is_empty() {
            self.agreement_ratio =
                self.agreeing_models.len() as f64 / self.participating_models.len() as f64;
        }
        self
    }

    /// Mark consensus as achieved
    pub fn achieved(mut self) -> Self {
        self.consensus_achieved = true;
        self
    }
}

/// Result of consensus process
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Selected or synthesized output
    pub output: String,

    /// Metadata explaining the decision
    pub metadata: ConsensusMetadata,

    /// Aggregate trust score for selected output
    pub trust_score: f64,
}

/// Consensus engine for output selection and synthesis
pub struct ConsensusEngine {
    strategy: ConsensusStrategy,
    trust_threshold: f64,
    min_models: usize,
}

impl ConsensusEngine {
    /// Create a new consensus engine with strategy
    pub fn new(strategy: ConsensusStrategy) -> Self {
        Self {
            strategy,
            trust_threshold: 0.7,
            min_models: 1,
        }
    }

    /// Set trust threshold
    pub fn with_trust_threshold(mut self, threshold: f64) -> Self {
        self.trust_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set minimum number of models required
    pub fn with_min_models(mut self, min_models: usize) -> Self {
        self.min_models = min_models.max(1);
        self
    }

    /// Reach consensus on outputs from multiple models
    pub fn reach_consensus(
        &self,
        outputs: &HashMap<String, ModelOutput>,
        trust_scores: &HashMap<String, TrustScore>,
    ) -> ConsensusResult {
        // Filter outputs that meet trust threshold
        let trusted_outputs: HashMap<_, _> = outputs
            .iter()
            .filter(|(model_id, _)| {
                trust_scores
                    .get(*model_id)
                    .map(|score| score.aggregate_score >= self.trust_threshold)
                    .unwrap_or(false)
            })
            .collect();

        if trusted_outputs.is_empty() {
            return self.degraded_consensus(outputs, trust_scores);
        }

        match self.strategy {
            ConsensusStrategy::HighestTrust => {
                self.highest_trust_consensus(&trusted_outputs, trust_scores)
            }
            ConsensusStrategy::MajorityVote => {
                self.majority_vote_consensus(&trusted_outputs, trust_scores)
            }
            ConsensusStrategy::Synthesize => {
                self.synthesize_consensus(&trusted_outputs, trust_scores)
            }
            ConsensusStrategy::WeightedCombination => {
                self.weighted_consensus(&trusted_outputs, trust_scores)
            }
        }
    }

    /// Select output with highest trust score
    fn highest_trust_consensus(
        &self,
        outputs: &HashMap<&String, &ModelOutput>,
        trust_scores: &HashMap<String, TrustScore>,
    ) -> ConsensusResult {
        let (best_model, best_output, best_score) = outputs
            .iter()
            .filter_map(|(model_id, output)| {
                trust_scores
                    .get(*model_id)
                    .map(|score| (*model_id, *output, score.aggregate_score))
            })
            .max_by(|(_, _, score1), (_, _, score2)| {
                score1
                    .partial_cmp(score2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let participating: Vec<String> = outputs.keys().map(|s| (*s).clone()).collect();

        let metadata = ConsensusMetadata::new(
            ConsensusStrategy::HighestTrust,
            format!(
                "Selected output from {} with trust score {:.3}",
                best_model, best_score
            ),
        )
        .with_participants(participating.clone())
        .with_agreeing(vec![best_model.clone()])
        .with_agreement_ratio()
        .achieved();

        ConsensusResult {
            output: best_output.text.clone(),
            metadata,
            trust_score: best_score,
        }
    }

    /// Use majority voting on outputs
    fn majority_vote_consensus(
        &self,
        outputs: &HashMap<&String, &ModelOutput>,
        trust_scores: &HashMap<String, TrustScore>,
    ) -> ConsensusResult {
        // Count occurrences of each output
        let mut output_counts: HashMap<String, Vec<String>> = HashMap::new();
        for (model_id, output) in outputs {
            output_counts
                .entry(output.text.clone())
                .or_default()
                .push((*model_id).clone());
        }

        // Find most common output
        let (majority_output, agreeing_models) = output_counts
            .into_iter()
            .max_by_key(|(_, models)| models.len())
            .unwrap();

        // Compute average trust score of agreeing models
        let avg_trust: f64 = agreeing_models
            .iter()
            .filter_map(|model_id| trust_scores.get(model_id))
            .map(|score| score.aggregate_score)
            .sum::<f64>()
            / agreeing_models.len() as f64;

        let participating: Vec<String> = outputs.keys().map(|s| (*s).clone()).collect();

        let metadata = ConsensusMetadata::new(
            ConsensusStrategy::MajorityVote,
            format!(
                "Majority vote: {} of {} models agreed",
                agreeing_models.len(),
                outputs.len()
            ),
        )
        .with_participants(participating)
        .with_agreeing(agreeing_models)
        .with_agreement_ratio()
        .achieved();

        ConsensusResult {
            output: majority_output,
            metadata,
            trust_score: avg_trust,
        }
    }

    /// Synthesize output from multiple sources
    fn synthesize_consensus(
        &self,
        outputs: &HashMap<&String, &ModelOutput>,
        trust_scores: &HashMap<String, TrustScore>,
    ) -> ConsensusResult {
        // For now, use weighted combination as synthesis
        // In production, this could use more sophisticated NLP
        self.weighted_consensus(outputs, trust_scores)
    }

    /// Weighted combination based on trust scores
    fn weighted_consensus(
        &self,
        outputs: &HashMap<&String, &ModelOutput>,
        trust_scores: &HashMap<String, TrustScore>,
    ) -> ConsensusResult {
        // Select output with highest weighted score
        // In a more sophisticated implementation, this could blend outputs

        let (best_model, best_output, best_score) = outputs
            .iter()
            .filter_map(|(model_id, output)| {
                trust_scores.get(*model_id).map(|score| {
                    let weighted =
                        score.aggregate_score * (1.0 + output.model_confidence.unwrap_or(0.5));
                    (*model_id, *output, weighted)
                })
            })
            .max_by(|(_, _, score1), (_, _, score2)| {
                score1
                    .partial_cmp(score2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let participating: Vec<String> = outputs.keys().map(|s| (*s).clone()).collect();

        let metadata = ConsensusMetadata::new(
            ConsensusStrategy::WeightedCombination,
            format!(
                "Weighted selection from {} with combined score {:.3}",
                best_model, best_score
            ),
        )
        .with_participants(participating.clone())
        .with_agreeing(vec![best_model.clone()])
        .with_agreement_ratio()
        .achieved();

        let actual_score = trust_scores
            .get(best_model)
            .map(|s| s.aggregate_score)
            .unwrap_or(0.0);

        ConsensusResult {
            output: best_output.text.clone(),
            metadata,
            trust_score: actual_score,
        }
    }

    /// Degraded consensus when no outputs meet threshold
    fn degraded_consensus(
        &self,
        outputs: &HashMap<String, ModelOutput>,
        trust_scores: &HashMap<String, TrustScore>,
    ) -> ConsensusResult {
        if outputs.is_empty() {
            let metadata =
                ConsensusMetadata::new(self.strategy, "No outputs available".to_string());

            return ConsensusResult {
                output: String::new(),
                metadata,
                trust_score: 0.0,
            };
        }

        // Select best available output even if below threshold
        let (best_model, best_output, best_score) = outputs
            .iter()
            .filter_map(|(model_id, output)| {
                trust_scores
                    .get(model_id)
                    .map(|score| (model_id, output, score.aggregate_score))
            })
            .max_by(|(_, _, score1), (_, _, score2)| {
                score1
                    .partial_cmp(score2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| {
                let (id, out) = outputs.iter().next().unwrap();
                (id, out, 0.0)
            });

        let participating: Vec<String> = outputs.keys().cloned().collect();

        let metadata = ConsensusMetadata::new(
            self.strategy,
            format!(
                "Degraded mode: Selected best available from {} (score {:.3}, below threshold {:.3})",
                best_model, best_score, self.trust_threshold
            ),
        )
        .with_participants(participating)
        .with_agreeing(vec![best_model.clone()]);

        ConsensusResult {
            output: best_output.text.clone(),
            metadata,
            trust_score: best_score,
        }
    }
}

impl Default for ConsensusEngine {
    fn default() -> Self {
        Self::new(ConsensusStrategy::HighestTrust)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_output(text: &str, confidence: f64) -> ModelOutput {
        ModelOutput::new(text).with_confidence(confidence)
    }

    fn create_test_score(aggregate: f64) -> TrustScore {
        TrustScore::new(aggregate, aggregate, aggregate, aggregate)
    }

    #[test]
    fn test_highest_trust_consensus() {
        let engine =
            ConsensusEngine::new(ConsensusStrategy::HighestTrust).with_trust_threshold(0.5);

        let mut outputs = HashMap::new();
        outputs.insert("model1".to_string(), create_test_output("output1", 0.8));
        outputs.insert("model2".to_string(), create_test_output("output2", 0.9));
        outputs.insert("model3".to_string(), create_test_output("output3", 0.7));

        let mut trust_scores = HashMap::new();
        trust_scores.insert("model1".to_string(), create_test_score(0.8));
        trust_scores.insert("model2".to_string(), create_test_score(0.9));
        trust_scores.insert("model3".to_string(), create_test_score(0.7));

        let result = engine.reach_consensus(&outputs, &trust_scores);
        assert_eq!(result.output, "output2");
        assert!(result.metadata.consensus_achieved);
        assert_eq!(result.trust_score, 0.9);
    }

    #[test]
    fn test_majority_vote_consensus() {
        let engine =
            ConsensusEngine::new(ConsensusStrategy::MajorityVote).with_trust_threshold(0.5);

        let mut outputs = HashMap::new();
        outputs.insert("model1".to_string(), create_test_output("same output", 0.8));
        outputs.insert("model2".to_string(), create_test_output("same output", 0.7));
        outputs.insert("model3".to_string(), create_test_output("different", 0.9));

        let mut trust_scores = HashMap::new();
        trust_scores.insert("model1".to_string(), create_test_score(0.8));
        trust_scores.insert("model2".to_string(), create_test_score(0.7));
        trust_scores.insert("model3".to_string(), create_test_score(0.9));

        let result = engine.reach_consensus(&outputs, &trust_scores);
        assert_eq!(result.output, "same output");
        assert!(result.metadata.consensus_achieved);
        assert_eq!(result.metadata.agreeing_models.len(), 2);
    }

    #[test]
    fn test_degraded_consensus() {
        let engine =
            ConsensusEngine::new(ConsensusStrategy::HighestTrust).with_trust_threshold(0.95); // Very high threshold

        let mut outputs = HashMap::new();
        outputs.insert("model1".to_string(), create_test_output("output1", 0.5));

        let mut trust_scores = HashMap::new();
        trust_scores.insert("model1".to_string(), create_test_score(0.5));

        let result = engine.reach_consensus(&outputs, &trust_scores);
        assert_eq!(result.output, "output1");
        assert!(result.metadata.reason.contains("Degraded mode"));
    }

    #[test]
    fn test_empty_outputs() {
        let engine = ConsensusEngine::new(ConsensusStrategy::HighestTrust);
        let outputs = HashMap::new();
        let trust_scores = HashMap::new();

        let result = engine.reach_consensus(&outputs, &trust_scores);
        assert_eq!(result.output, "");
        assert_eq!(result.trust_score, 0.0);
    }

    #[test]
    fn test_consensus_metadata() {
        let metadata =
            ConsensusMetadata::new(ConsensusStrategy::HighestTrust, "Test reason".to_string())
                .with_participants(vec!["m1".to_string(), "m2".to_string(), "m3".to_string()])
                .with_agreeing(vec!["m1".to_string(), "m2".to_string()])
                .with_agreement_ratio()
                .achieved();

        assert_eq!(metadata.participating_models.len(), 3);
        assert_eq!(metadata.agreeing_models.len(), 2);
        assert!((metadata.agreement_ratio - 0.666).abs() < 0.01);
        assert!(metadata.consensus_achieved);
    }
}
