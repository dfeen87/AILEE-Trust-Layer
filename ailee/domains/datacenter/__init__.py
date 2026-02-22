"""
AILEE Trust Layer — Data Center Domain
Version: 2.0.0

Governance and trust-scoring framework for AI-driven data center control
systems, including cooling, power, and workload management.

Core principle: AI proposes; AILEE decides whether it is safe to act.

Primary entry point: ``DatacenterGovernor``
Quick start with: ``create_datacenter_governor()``

Quick Start::

    from ailee.domains.datacenter import (
        DatacenterGovernor,
        DatacenterSignals,
        DatacenterTrustLevel,
        ControlDomain,
        ControlAction,
        SensorReading,
        create_datacenter_governor,
    )
    import time

    governor = create_datacenter_governor()

    signals = DatacenterSignals(
        control_domain=ControlDomain.COOLING,
        proposed_action=ControlAction.SETPOINT_CHANGE,
        ai_value=22.0,
        ai_confidence=0.88,
        sensor_readings=[
            SensorReading(22.3, time.time(), "rack_01_inlet"),
            SensorReading(22.5, time.time(), "rack_02_inlet"),
            SensorReading(22.1, time.time(), "rack_03_inlet"),
            SensorReading(22.4, time.time(), "rack_04_inlet"),
        ],
        zone_id="zone_a",
    )

    decision = governor.evaluate(signals)
    if decision.actionable:
        bms.set_temperature("zone_a", decision.trusted_value)
    else:
        logger.warning("Action withheld — trust level: %s", decision.authorized_level)

For detailed documentation, see: https://github.com/dfeen87/AILEE-Trust-Layer
"""

from .ailee_datacenter_domain import (
    # === Primary API ===
    DatacenterGovernor,
    create_datacenter_governor,
    create_default_governor,
    create_strict_governor,
    create_permissive_governor,
    validate_datacenter_signals,

    # === Enumerations ===
    DatacenterTrustLevel,
    FacilityHealthStatus,
    ControlDomain,
    ControlAction,

    # === Core Data Structures ===
    DatacenterPolicy,
    DatacenterSignals,
    DatacenterDecision,
    DatacenterEvent,

    # === Domain Configurations ===
    COOLING_CONTROL,
    POWER_CAPPING,
    WORKLOAD_PLACEMENT,

    # === Telemetry ===
    SensorReading,
    TelemetryProcessor,

    # === Controllers (lower-level API) ===
    CoolingController,
    PowerCapController,
    WorkloadPlacementGovernor,

    # === Monitoring ===
    DataCenterMonitor,
)

__version__ = "2.0.0"
__author__ = "AILEE Trust Layer Development Team"
__license__ = "MIT"
__doc_url__ = "https://github.com/dfeen87/AILEE-Trust-Layer"
__source_url__ = "https://github.com/dfeen87/AILEE-Trust-Layer"
__bug_tracker_url__ = "https://github.com/dfeen87/AILEE-Trust-Layer/issues"

__all__ = [
    # Primary API
    "DatacenterGovernor",
    "create_datacenter_governor",
    "create_default_governor",
    "create_strict_governor",
    "create_permissive_governor",
    "validate_datacenter_signals",

    # Enumerations
    "DatacenterTrustLevel",
    "FacilityHealthStatus",
    "ControlDomain",
    "ControlAction",

    # Core Data Structures
    "DatacenterPolicy",
    "DatacenterSignals",
    "DatacenterDecision",
    "DatacenterEvent",

    # Domain Configurations
    "COOLING_CONTROL",
    "POWER_CAPPING",
    "WORKLOAD_PLACEMENT",

    # Telemetry
    "SensorReading",
    "TelemetryProcessor",

    # Controllers (lower-level API)
    "CoolingController",
    "PowerCapController",
    "WorkloadPlacementGovernor",

    # Monitoring
    "DataCenterMonitor",

    # Version
    "__version__",
]
