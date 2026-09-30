# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Composition boundary that keeps commercial and process trust independent."""

from dataclasses import dataclass
from typing import Optional

from .industrial import ProcessInterval, ThroughputGovernor, ThroughputResult
from .licensing import (
    AuthorizationDecision, AuthorizationRequest, LicenseContract, LicenseGovernor,
)


@dataclass(frozen=True)
class GovernedAnalyticsDecision:
    """Independent licensing and analytical decisions, without an aggregate score."""

    analytics_permitted: bool
    authorization: AuthorizationDecision
    throughput: ThroughputResult
    observed_process_state: str
    machine_control_issued: bool = False


class GovernedAnalyticsService:
    """Read-only composition; licensing never changes machine process state."""

    def __init__(
        self,
        license_governor: Optional[LicenseGovernor] = None,
        throughput_governor: Optional[ThroughputGovernor] = None,
    ):
        self.license_governor = license_governor or LicenseGovernor()
        self.throughput_governor = throughput_governor or ThroughputGovernor()

    def evaluate(
        self,
        contract: LicenseContract,
        request: AuthorizationRequest,
        interval: ProcessInterval,
    ) -> GovernedAnalyticsDecision:
        authorization = self.license_governor.authorize(contract, request)
        throughput = self.throughput_governor.evaluate(interval, request.evaluated_at)
        return GovernedAnalyticsDecision(
            analytics_permitted=authorization.authorized and throughput.available,
            authorization=authorization,
            throughput=throughput,
            observed_process_state=interval.start_state.value,
        )
