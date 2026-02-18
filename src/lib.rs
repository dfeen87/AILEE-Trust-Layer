//! # AILEE Trust Layer
//!
//! A deterministic, trust-scored generative AI platform that provides:
//! - Generative engine with model abstraction
//! - Multi-dimensional trust scoring
//! - Consensus-based output selection
//! - Cryptographic lineage and verification
//! - Fully async, deterministic, and offline-capable design
//!
//! ## Architecture
//!
//! AILEE is a substrate-agnostic trust and intelligence layer that can run
//! on top of any compatible execution substrate (e.g., Ambient AI VCP).
//!
//! ## Core Components
//!
//! - **GenerationRequest**: Input specification with prompt, task type, and constraints
//! - **ModelAdapter**: Trait for model integration
//! - **TrustScorer**: Multi-dimensional trust evaluation
//! - **ConsensusEngine**: Output selection and synthesis
//! - **GenerationResult**: Verified output with lineage and metadata

pub mod consensus;
pub mod generation;
pub mod lineage;
pub mod model;
pub mod trust;

pub use consensus::{ConsensusEngine, ConsensusMetadata, ConsensusStrategy};
pub use generation::{GenerationRequest, GenerationResult};
pub use lineage::Lineage;
pub use model::{ModelAdapter, ModelCapability, ModelLocality, ModelOutput};
pub use trust::{TrustScore, TrustScorer};

/// Re-export common types
pub mod prelude {
    pub use crate::consensus::{ConsensusEngine, ConsensusMetadata, ConsensusStrategy};
    pub use crate::generation::{
        DeterminismLevel, ExecutionMode, GenerationRequest, GenerationResult, TaskType,
    };
    pub use crate::lineage::Lineage;
    pub use crate::model::{ModelAdapter, ModelCapability, ModelLocality, ModelOutput};
    pub use crate::trust::{TrustScore, TrustScorer};
}
