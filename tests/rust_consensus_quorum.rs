//! Regression coverage for the configured Rust consensus evidence quorum.

use ailee_trust_core::prelude::*;
use std::collections::HashMap;

fn strategies() -> [ConsensusStrategy; 4] {
    [
        ConsensusStrategy::HighestTrust,
        ConsensusStrategy::MajorityVote,
        ConsensusStrategy::Synthesize,
        ConsensusStrategy::WeightedCombination,
    ]
}

fn score(aggregate: f64) -> TrustScore {
    TrustScore::new(aggregate, aggregate, aggregate, aggregate)
}

#[test]
fn insufficient_trusted_models_cannot_claim_consensus() {
    let outputs = HashMap::from([(
        "only-model".to_string(),
        ModelOutput::new("available output"),
    )]);
    let scores = HashMap::from([("only-model".to_string(), score(0.9))]);

    for strategy in strategies() {
        let result = ConsensusEngine::new(strategy)
            .with_min_models(3)
            .reach_consensus(&outputs, &scores);

        assert_eq!(result.output, "available output");
        assert!(
            !result.metadata.consensus_achieved,
            "{strategy:?} reported consensus with one model despite a three-model quorum"
        );
        assert_eq!(result.trust_score, 0.9);
        assert_eq!(result.metadata.strategy, strategy);
        assert_eq!(result.metadata.participating_models, ["only-model"]);
        assert_eq!(result.metadata.agreeing_models, ["only-model"]);
        assert!(result
            .metadata
            .reason
            .contains("Insufficient trusted models"));
        assert!(result.metadata.reason.contains("1 available, 3 required"));
        assert!(!result.metadata.reason.contains("below threshold"));
    }
}

#[test]
fn only_threshold_eligible_outputs_count_toward_quorum() {
    let outputs = HashMap::from([
        ("trusted".to_string(), ModelOutput::new("best available")),
        ("low".to_string(), ModelOutput::new("low trust")),
        ("missing".to_string(), ModelOutput::new("no evidence")),
        ("nan".to_string(), ModelOutput::new("malformed aggregate")),
    ]);
    let mut malformed = score(0.9);
    malformed.aggregate_score = f64::NAN;
    let scores = HashMap::from([
        ("trusted".to_string(), score(0.9)),
        ("low".to_string(), score(0.69)),
        ("nan".to_string(), malformed),
        ("not-an-output".to_string(), score(0.99)),
    ]);

    for strategy in strategies() {
        let result = ConsensusEngine::new(strategy)
            .with_trust_threshold(0.7)
            .with_min_models(2)
            .reach_consensus(&outputs, &scores);

        assert!(!result.metadata.consensus_achieved);
        assert!(result.metadata.reason.contains("1 available, 2 required"));
        // Degraded selection retains the established best-available contract;
        // eligibility does not turn that output into achieved consensus.
        assert!(outputs.values().any(|output| output.text == result.output));
    }
}

#[test]
fn exact_and_excess_quorum_preserve_successful_consensus() {
    let outputs = HashMap::from([
        ("first".to_string(), ModelOutput::new("shared output")),
        ("second".to_string(), ModelOutput::new("shared output")),
        ("third".to_string(), ModelOutput::new("shared output")),
    ]);
    let scores = HashMap::from([
        ("first".to_string(), score(0.9)),
        ("second".to_string(), score(0.8)),
        ("third".to_string(), score(0.7)),
    ]);

    for strategy in strategies() {
        for required in [2, 3] {
            let result = ConsensusEngine::new(strategy)
                .with_trust_threshold(0.7)
                .with_min_models(required)
                .reach_consensus(&outputs, &scores);

            assert!(result.metadata.consensus_achieved);
            assert_eq!(result.output, "shared output");
            assert_eq!(result.metadata.participating_models.len(), 3);
        }
    }
}

#[test]
fn default_and_zero_minimum_retain_single_model_success() {
    let outputs = HashMap::from([("only".to_string(), ModelOutput::new("output"))]);
    let scores = HashMap::from([("only".to_string(), score(0.9))]);

    for strategy in strategies() {
        for engine in [
            ConsensusEngine::new(strategy),
            ConsensusEngine::new(strategy).with_min_models(0),
        ] {
            let result = engine.reach_consensus(&outputs, &scores);
            assert!(result.metadata.consensus_achieved);
            assert_eq!(result.output, "output");
        }
    }
}

#[test]
fn absent_trusted_evidence_keeps_existing_degraded_results() {
    for strategy in strategies() {
        let engine = ConsensusEngine::new(strategy).with_min_models(3);
        let empty = engine.reach_consensus(&HashMap::new(), &HashMap::new());
        assert!(!empty.metadata.consensus_achieved);
        assert_eq!(empty.output, "");
        assert_eq!(empty.metadata.reason, "No outputs available");

        let outputs = HashMap::from([("low".to_string(), ModelOutput::new("available"))]);
        let scores = HashMap::from([("low".to_string(), score(0.2))]);
        let low = engine.reach_consensus(&outputs, &scores);
        assert!(!low.metadata.consensus_achieved);
        assert_eq!(low.output, "available");
        assert_eq!(low.trust_score, 0.2);
        assert!(low.metadata.reason.contains("below threshold"));
    }
}
