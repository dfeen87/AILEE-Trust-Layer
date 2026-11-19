"""Opt-in experimental OpenNAM metadata contracts for synthetic research."""

from .contracts import (
    SCHEMA_VERSION,
    CatalogEntry,
    CatalogRef,
    ContractError,
    EvidenceCatalog,
    EvidenceRecord,
    EvidenceRequest,
    Observation,
    Outcome,
    Reason,
    ReviewPolicy,
    ReviewStatus,
    Scope,
)

__all__ = [
    "SCHEMA_VERSION", "CatalogEntry", "CatalogRef", "ContractError",
    "EvidenceCatalog", "EvidenceRecord", "EvidenceRequest", "Observation",
    "Outcome", "Reason", "ReviewPolicy", "ReviewStatus", "Scope",
]
