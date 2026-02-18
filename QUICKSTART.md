# Quick Start Guide - AILEE Trust Layer (Rust)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ailee_trust_core = { path = "path/to/ailee_trust_core" }
tokio = { version = "1.40", features = ["full"] }
async-trait = "0.1"
```

## 5-Minute Example

```rust
use ailee_trust_core::prelude::*;
use async_trait::async_trait;
use std::collections::HashSet;

// 1. Implement ModelAdapter for your model
struct MyModel;

#[async_trait]
impl ModelAdapter for MyModel {
    async fn generate(&self, prompt: &str) -> Result<ModelOutput, ModelError> {
        // Your model logic here
        Ok(ModelOutput::new(format!("Response to: {}", prompt))
            .with_confidence(0.9))
    }
    
    fn model_id(&self) -> &str { "my-model-v1" }
    
    fn model_capabilities(&self) -> &HashSet<ModelCapability> {
        static CAPS: std::sync::OnceLock<HashSet<ModelCapability>> = 
            std::sync::OnceLock::new();
        CAPS.get_or_init(|| {
            let mut caps = HashSet::new();
            caps.insert(ModelCapability::Chat);
            caps
        })
    }
    
    fn model_locality(&self) -> ModelLocality { 
        ModelLocality::Local 
    }
}

// 2. Use the trust layer
#[tokio::main]
async fn main() {
    // Create request
    let request = GenerationRequest::new(
        "What is Rust?",
        TaskType::Chat
    ).with_trust_threshold(0.75);
    
    // Generate outputs
    let model = MyModel;
    let mut outputs = std::collections::HashMap::new();
    outputs.insert(
        model.model_id().to_string(),
        model.generate(&request.prompt).await.unwrap()
    );
    
    // Score outputs
    let mut scorer = TrustScorer::new();
    let scores = scorer.score_outputs(&outputs);
    
    // Reach consensus
    let consensus = ConsensusEngine::new(ConsensusStrategy::HighestTrust)
        .reach_consensus(&outputs, &scores);
    
    // Build result with lineage
    let lineage = Lineage::build(&request, &outputs, &consensus.output);
    
    let result = GenerationResult {
        final_output: consensus.output,
        aggregate_trust_score: consensus.trust_score,
        model_trust_scores: scores,
        lineage,
        execution_metadata: std::collections::HashMap::new(),
    };
    
    println!("Output: {}", result.final_output);
    println!("Trust: {:.3}", result.aggregate_trust_score);
}
```

## Key Concepts

### Trust Scoring
Every output gets scored on 4 dimensions:
- **Confidence**: Model's certainty
- **Safety**: Content safety check
- **Consistency**: Match with history
- **Determinism**: Repeatability

### Consensus Strategies
- **HighestTrust**: Pick best scored output
- **MajorityVote**: Pick most common output
- **WeightedCombination**: Weight by scores
- **Synthesize**: Combine outputs

### Verification
Every generation includes a SHA-256 hash for verification:

```rust
let lineage = Lineage::build(&request, &outputs, &final_output);
assert!(lineage.verify(&request, &outputs, &final_output));
```

## Running Tests

```bash
# All tests
cargo test

# Integration tests only
cargo test --test integration_tests

# With output
cargo test -- --nocapture
```

## Running Examples

```bash
# Complete workflow
cargo run --example complete_workflow

# See the full output
cargo run --example complete_workflow 2>&1 | less
```

## Common Patterns

### Multiple Models
```rust
let models: Vec<Box<dyn ModelAdapter>> = vec![
    Box::new(Model1),
    Box::new(Model2),
    Box::new(Model3),
];

let mut outputs = HashMap::new();
for model in models {
    let output = model.generate(&prompt).await?;
    outputs.insert(model.model_id().to_string(), output);
}
```

### Custom Trust Threshold
```rust
let request = GenerationRequest::new(prompt, TaskType::Code)
    .with_trust_threshold(0.9);  // High threshold for critical tasks

let consensus = ConsensusEngine::new(strategy)
    .with_trust_threshold(request.trust_threshold);
```

### Degraded Mode
```rust
// System automatically degrades if no models meet threshold
let result = consensus_engine.reach_consensus(&outputs, &scores);

if result.metadata.reason.contains("Degraded") {
    println!("Warning: Using degraded consensus");
}
```

## Best Practices

1. **Always verify lineage** for critical outputs
2. **Set appropriate trust thresholds** based on use case
3. **Use multiple models** for better consensus
4. **Monitor trust scores** over time
5. **Handle degraded mode** gracefully

## Troubleshooting

**No models meet threshold?**
- Lower trust threshold OR
- Improve model outputs OR
- Let system use degraded mode

**Low consistency scores?**
- Expected for first few outputs
- Scores improve with history
- Adjust `max_history` if needed

**Want deterministic results?**
- Use `DeterminismLevel::Full`
- Ensure models are deterministic
- Same inputs = same hash

## Documentation

Generate full API docs:
```bash
cargo doc --open
```

## Support

- See `RUST_README.md` for full documentation
- Check `examples/` for more patterns
- Run `cargo test` to see test examples
