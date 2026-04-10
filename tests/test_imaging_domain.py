
import pytest
from ailee.domains.imaging import (
    get_health,
    get_subsystem_health,
    get_metrics,
    get_events,
    get_decision_history,
    create_strict_governor,
    create_permissive_governor,
    create_default_governor,
    ImagingTrustLevel,
    ImagingHealthStatus,
    ImagingControlDomain,
    ImagingControlAction,
)

def test_imaging_enums():
    assert ImagingTrustLevel.NO_ACTION == 0
    assert ImagingHealthStatus.OPTIMAL == "OPTIMAL"

def test_imaging_factory_and_monitor():
    gov = create_strict_governor()
    assert gov is not None
    assert get_health(gov) is not None
    assert isinstance(get_subsystem_health(gov), dict)
    assert isinstance(get_metrics(gov), dict)
    assert isinstance(get_events(gov), list)
    assert isinstance(get_decision_history(gov), list)

def test_imaging_backward_compatibility():
    pass
