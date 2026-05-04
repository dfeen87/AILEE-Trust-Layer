"""
AILEE Trust Layer — Memory Domain
Version: 4.6.0

Governance and trust-scoring framework for AI-driven memory control
systems, including RAM allocation, heap monitoring, swap management, and process memory.
"""

from .ailee_memory_domain import (
    # === Primary API ===
    MemoryGovernor,
    create_memory_governor,
    create_default_governor,
    create_strict_governor,
    create_permissive_governor,
    validate_memory_signals,

    # === Enumerations ===
    MemoryTrustLevel,
    MemoryHealthStatus,
    MemoryDomain,
    MemoryAction,

    # === Core Data Structures ===
    MemoryPolicy,
    MemoryReading,
    MemorySignals,
    MemoryDecision,
    MemoryEvent,

    # === Domain Configurations ===
    RAM_ALLOCATION,
    HEAP_MONITORING,
    SWAP_MANAGEMENT,
    PROCESS_MEMORY,
)

__version__ = "4.6.0"
__author__ = "AILEE Trust Layer Development Team"
__license__ = "MIT"

__all__ = [
    # Primary API
    "MemoryGovernor",
    "create_memory_governor",
    "create_default_governor",
    "create_strict_governor",
    "create_permissive_governor",
    "validate_memory_signals",

    # Enumerations
    "MemoryTrustLevel",
    "MemoryHealthStatus",
    "MemoryDomain",
    "MemoryAction",

    # Core Data Structures
    "MemoryPolicy",
    "MemoryReading",
    "MemorySignals",
    "MemoryDecision",
    "MemoryEvent",

    # Domain Configurations
    "RAM_ALLOCATION",
    "HEAP_MONITORING",
    "SWAP_MANAGEMENT",
    "PROCESS_MEMORY",

    # Version
    "__version__",
]
