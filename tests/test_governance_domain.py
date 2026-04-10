
import pytest
from ailee.domains.governance import (
    get_health,
    get_subsystem_health,
    get_metrics,
    get_events,
    get_decision_history,
    create_strict_governor,
    create_permissive_governor,
    create_default_governor,
    GovernanceTrustLevel,
    GovernanceHealthStatus,
    GovernanceControlDomain,
    GovernanceControlAction,
)

def _test_governance_enums():
    assert GovernanceTrustLevel.NO_ACTION == 0
    assert GovernanceHealthStatus.OPTIMAL == "OPTIMAL"

def test_governance_factory_and_monitor():
    gov = create_strict_governor()
    assert gov is not None
    assert get_health(gov) is not None
    assert isinstance(get_subsystem_health(gov), dict)
    assert isinstance(get_metrics(gov), dict)
    assert isinstance(get_events(gov), list)
    assert isinstance(get_decision_history(gov), list)

def test_governance_backward_compatibility():
    pass
