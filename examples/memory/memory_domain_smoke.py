# Licensed under the PolyForm Noncommercial License 1.0.0
"""Runnable smoke example for the AILEE memory-domain governor."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time

from ailee.domains.memory.ailee_memory_domain import (
    MemoryAction,
    MemoryDomain,
    MemoryReading,
    MemorySignals,
    create_memory_governor,
)


def main() -> None:
    """Evaluate a sample RAM allocation governance decision."""
    gov = create_memory_governor()
    signals = MemorySignals(
        memory_domain=MemoryDomain.RAM_ALLOCATION,
        proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
        ai_value=0.82,
        ai_confidence=0.91,
        memory_readings=[
            MemoryReading(0.81, time.time(), "node_01_ram"),
            MemoryReading(0.83, time.time(), "node_02_ram"),
            MemoryReading(0.80, time.time(), "node_03_ram"),
        ],
        node_id="cluster_a",
    )
    decision = gov.evaluate(signals)
    print(decision.actionable, decision.trusted_value, decision.health_status)


if __name__ == "__main__":
    main()
