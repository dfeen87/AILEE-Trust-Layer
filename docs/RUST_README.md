# AILEE Trust Layer - Core Rust Implementation

Production-grade Rust implementation of the AILEE Trust Layer, providing deterministic, trust-scored generative AI with consensus-based output selection and cryptographic verification.

## Overview

AILEE Trust Layer is a substrate-agnostic trust and intelligence layer that provides:

- **Generative Engine**: Model abstraction with async execution
- **Trust Scoring**: Multi-dimensional evaluation (confidence, safety, consistency, determinism)
- **Consensus Engine**: Intelligent output selection and synthesis
- **Lineage & Verification**: Cryptographic proof with SHA-256 hashing
- **Deterministic Replay**: Reproducible generation verification

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AILEE Trust Layer                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GenerationRequest                                           │
│       │                                                      │
│       ├─► ModelAdapter(s) ──► ModelOutput(s)                │
│       │                              │                       │
│       │                              ▼                       │
│       │                        TrustScorer                   │
│       │                              │                       │
│       │                              ▼                       │
│       │                        TrustScore(s)                 │
│       │                              │                       │
│       │                              ▼                       │
│       ├──────────────────────► ConsensusEngine               │
│       │                              │                       │
│       │                              ▼                       │
│       │                        ConsensusResult               │
│       │                              │                       │
│       └──────────────────────────────┼─────► Lineage         │
│                                      │                       │
│                                      ▼                       │
│                              GenerationResult                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Model Abstraction

The `ModelAdapter` trait provides a clean interface for integrating any AI model:

```rust
use ailee_trust_core::prelude::*;
use async_trait::async_trait;

struct MyModel;

#[async_trait]
impl ModelAdapter for MyModel {
    async fn generate(&self, prompt: &str) -> Result<ModelOutput, ModelError> {
        // Your model implementation
        Ok(ModelOutput::new("response"))
    }
    
    fn model_id(&self) -> &str { "my-model" }
    fn model_capabilities(&self) -> &HashSet<ModelCapability> { /* ... */ }
    fn model_locality(&self) -> ModelLocality { ModelLocality::Local }
}
```

### 2. Multi-Dimensional Trust Scoring

Trust scores evaluate outputs across four dimensions:

- **Confidence**: Model's certainty (from model metadata or heuristics)
- **Safety**: Content safety analysis
- **Consistency**: Similarity to historical outputs
- **Determinism**: Repeatability indicators

```rust
let mut scorer = TrustScorer::new();
let score = scorer.score_output(&output);

println!("Confidence: {:.3}", score.confidence_score);
println!("Safety: {:.3}", score.safety_score);
println!("Consistency: {:.3}", score.consistency_score);
println!("Determinism: {:.3}", score.determinism_score);
println!("Aggregate: {:.3}", score.aggregate_score);
```

### 3. Consensus Strategies

Multiple strategies for intelligent output selection:

- **HighestTrust**: Select output with best trust score
- **MajorityVote**: Choose most common output
- **Synthesize**: Combine multiple outputs
- **WeightedCombination**: Weight by trust scores

```rust
let engine = ConsensusEngine::new(ConsensusStrategy::HighestTrust)
    .with_trust_threshold(0.75)
    .with_min_models(2);

let result = engine.reach_consensus(&outputs, &trust_scores);
```

### 4. Cryptographic Verification

Every generation produces a SHA-256 hash for verification:

```rust
let lineage = Lineage::build(&request, &outputs, &final_output);

// Verify later
assert!(lineage.verify(&request, &outputs, &final_output));
```

## Usage

### Basic Example

```rust
use ailee_trust_core::prelude::*;

#[tokio::main]
async fn main() {
    // 1. Create request
    let request = GenerationRequest::new("Hello world", TaskType::Code)
        .with_trust_threshold(0.75);
    
    // 2. Generate outputs (from your models)
    let mut outputs = HashMap::new();
    // ... populate outputs ...
    
    // 3. Score outputs
    let mut scorer = TrustScorer::new();
    let scores = scorer.score_outputs(&outputs);
    
    // 4. Reach consensus
    let consensus = ConsensusEngine::new(ConsensusStrategy::HighestTrust)
        .reach_consensus(&outputs, &scores);
    
    // 5. Build lineage
    let lineage = Lineage::build(&request, &outputs, &consensus.output);
    
    // 6. Create result
    let result = GenerationResult {
        final_output: consensus.output,
        aggregate_trust_score: consensus.trust_score,
        model_trust_scores: scores,
        lineage,
        execution_metadata: HashMap::new(),
    };
}
```

### Complete Example

See `examples/complete_workflow.rs` for a full end-to-end demonstration.

## Design Principles

1. **Substrate Agnostic**: No dependencies on specific execution environments
2. **Fully Async**: All operations use async/await for scalability
3. **No Global State**: Thread-safe, no hidden state
4. **Deterministic**: Same inputs produce same outputs (where possible)
5. **Offline Capable**: Works without network connectivity
6. **Explicit Logic**: All trust decisions are inspectable and reproducible

## Testing

The library includes comprehensive tests:

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration_tests

# Run example
cargo run --example complete_workflow
```

## Integration with Execution Substrates

AILEE is designed to integrate with execution substrates like Ambient AI VCP:

1. **Clean Separation**: AILEE handles trust/intelligence, substrate handles execution
2. **Standard Interfaces**: Use `ModelAdapter` trait for model integration
3. **Metadata Support**: Pass substrate-specific context via request metadata
4. **Async Compatible**: Works with any async runtime

## Performance Characteristics

- **Trust Scoring**: O(n) per output, where n is output length
- **Consensus**: O(m) where m is number of models
- **Hashing**: O(n) for verification hash computation
- **Memory**: Bounded history for consistency checks (configurable)

## Security

- Uses SHA-256 for cryptographic hashing
- No external dependencies for trust scoring (deterministic algorithms only)
- All inputs and outputs are included in verification hash
- Sorted serialization ensures hash determinism

## Requirements

- Rust 2021 edition
- Tokio runtime for async execution
- Minimal dependencies (see Cargo.toml)

## License

MIT License

## Contributing

Contributions welcome! Please ensure:

- All tests pass: `cargo test`
- Code is formatted: `cargo fmt`
- No clippy warnings: `cargo clippy --all-targets --all-features -- -D warnings`
