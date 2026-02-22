"""
AILEE Trust Layer — Crypto Mining Domain
Version: 2.0.0

Governance and trust-scoring framework for AI-driven crypto mining control
systems, including hash rate optimization, thermal management, power
consumption, and mining pool selection.

Core principle: AI proposes; AILEE decides whether it is safe to act.

Primary entry point: ``MiningGovernor``
Quick start with: ``create_mining_governor()``

Quick Start::

    from ailee.domains.crypto_mining import (
        MiningGovernor,
        MiningSignals,
        CryptoMiningTrustLevel,
        MiningDomain,
        MiningAction,
        HardwareReading,
        create_mining_governor,
    )
    import time

    governor = create_mining_governor()

    signals = MiningSignals(
        mining_domain=MiningDomain.HASH_RATE,
        proposed_action=MiningAction.TUNE_HASH_RATE,
        ai_value=95.0,
        ai_confidence=0.88,
        hardware_readings=[
            HardwareReading(93.5, time.time(), "rig_01"),
            HardwareReading(94.2, time.time(), "rig_02"),
            HardwareReading(95.1, time.time(), "rig_03"),
        ],
        rig_id="rig_01",
    )

    decision = governor.evaluate(signals)
    if decision.actionable:
        miner.set_hash_rate("rig_01", decision.trusted_value)
    else:
        logger.warning("Action withheld — trust level: %s", decision.authorized_level)

For detailed documentation, see: https://github.com/dfeen87/AILEE-Trust-Layer
"""

from .ailee_crypto_mining_domain import (
    # === Primary API ===
    MiningGovernor,
    create_mining_governor,
    create_default_governor,
    create_strict_governor,
    create_permissive_governor,
    validate_mining_signals,

    # === Enumerations ===
    CryptoMiningTrustLevel,
    MiningOperationStatus,
    MiningDomain,
    MiningAction,

    # === Core Data Structures ===
    MiningPolicy,
    HardwareReading,
    MiningSignals,
    MiningDecision,
    MiningEvent,

    # === Domain Configurations ===
    HASH_RATE_OPTIMIZATION,
    THERMAL_PROTECTION,
    POOL_SWITCHING,
    POWER_MANAGEMENT,
)

__version__ = "2.0.0"
__author__ = "AILEE Trust Layer Development Team"
__license__ = "MIT"
__doc_url__ = "https://github.com/dfeen87/AILEE-Trust-Layer"
__source_url__ = "https://github.com/dfeen87/AILEE-Trust-Layer"
__bug_tracker_url__ = "https://github.com/dfeen87/AILEE-Trust-Layer/issues"

__all__ = [
    # Primary API
    "MiningGovernor",
    "create_mining_governor",
    "create_default_governor",
    "create_strict_governor",
    "create_permissive_governor",
    "validate_mining_signals",

    # Enumerations
    "CryptoMiningTrustLevel",
    "MiningOperationStatus",
    "MiningDomain",
    "MiningAction",

    # Core Data Structures
    "MiningPolicy",
    "HardwareReading",
    "MiningSignals",
    "MiningDecision",
    "MiningEvent",

    # Domain Configurations
    "HASH_RATE_OPTIMIZATION",
    "THERMAL_PROTECTION",
    "POOL_SWITCHING",
    "POWER_MANAGEMENT",

    # Version
    "__version__",
]
