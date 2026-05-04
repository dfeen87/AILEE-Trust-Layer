from ailee.domains.memory.ailee_memory_domain import (
    create_memory_governor, MemorySignals, MemoryDomain, MemoryAction, MemoryReading
)
import time
gov = create_memory_governor()
signals = MemorySignals(
    memory_domain=MemoryDomain.RAM_ALLOCATION,
    proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
    ai_value=0.82,            # AI proposes 82% utilization cap
    ai_confidence=0.91,
    memory_readings=[
        MemoryReading(0.81, time.time(), "node_01_ram"),
        MemoryReading(0.83, time.time(), "node_02_ram"),
        MemoryReading(0.80, time.time(), "node_03_ram"),
    ],
    node_id="cluster_a",
)
dec = gov.evaluate(signals)
print(dec.actionable, dec.trusted_value, dec.health_status)
