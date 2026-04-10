
import pytest
from ailee.domains.automotive import (
    get_health,
    get_subsystem_health,
    get_metrics,
    get_events,
    get_decision_history,
    create_strict_governor,
    create_permissive_governor,
    create_default_governor,
    AutonomyTrustLevel,
    AutonomyHealthStatus,
    AutonomyControlDomain,
    AutonomyControlAction,
)

def test_automotive_enums():
    assert AutonomyTrustLevel.NO_ACTION == 0
    assert AutonomyHealthStatus.OPTIMAL == "OPTIMAL"

def test_automotive_factory_and_monitor():
    gov = create_strict_governor()
    assert gov is not None
    assert get_health(gov) is not None
    assert isinstance(get_subsystem_health(gov), dict)
    assert isinstance(get_metrics(gov), dict)
    assert isinstance(get_events(gov), list)
    assert isinstance(get_decision_history(gov), list)

def test_automotive_backward_compatibility():
    pass
