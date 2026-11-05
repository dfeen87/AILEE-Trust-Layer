# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""
AILEE Action Authorization Wrapper — Cybersecurity Domain.

Provides deterministic action interception and trust enforcement for external
link display, software recommendations, web visits, file downloads, and installations.
"""

from typing import Any, Callable, Dict, Optional

from .cybersecurity import (
    ActionTier,
    CybersecurityAuthorizationRequest,
    CybersecurityDecision,
    CybersecurityGovernor,
    DecisionOutcome,
    create_cybersecurity_governor,
)


class CybersecurityActionWrapper:
    """
    Interception wrapper for agent tools and execution pipelines.

    Guarantees deterministic trust governance before opening URLs, recommending
    software, downloading binaries, or executing installers.
    """

    def __init__(
        self,
        action_fn: Callable[..., Any],
        governor: Optional[CybersecurityGovernor] = None,
        shadow_mode: bool = False,
        require_confirmation_for_review: bool = True,
        policy_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.action_fn = action_fn
        self.shadow_mode = shadow_mode
        self.require_confirmation_for_review = require_confirmation_for_review
        if governor is not None:
            self.governor = governor
        else:
            overrides = policy_overrides or {}
            self.governor = create_cybersecurity_governor(**overrides)

    def execute(
        self,
        request: CybersecurityAuthorizationRequest,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute the intercepted action only if authorized by the cybersecurity governor.
        """
        decision = self.governor.authorize(request)

        if self.shadow_mode:
            result = self.action_fn(*args, **kwargs)
            return {
                "blocked": False,
                "shadow_mode": True,
                "result": result,
                "shadow_decision": decision,
                "outcome": decision.outcome.value,
                "reason": decision.reason.value,
            }

        if decision.outcome == DecisionOutcome.DENY:
            return {
                "blocked": True,
                "requires_review": False,
                "outcome": DecisionOutcome.DENY.value,
                "reason": decision.reason.value,
                "decision": decision,
                "result": None,
                "canonical_url": decision.canonical_url,
            }

        if (
            decision.outcome == DecisionOutcome.REVIEW
            and self.require_confirmation_for_review
        ):
            return {
                "blocked": True,
                "requires_review": True,
                "outcome": DecisionOutcome.REVIEW.value,
                "reason": decision.reason.value,
                "decision": decision,
                "result": None,
                "canonical_url": decision.canonical_url,
            }

        # Authorized
        result = self.action_fn(*args, **kwargs)
        return {
            "blocked": False,
            "requires_review": False,
            "outcome": DecisionOutcome.ALLOW.value,
            "reason": decision.reason.value,
            "decision": decision,
            "result": result,
            "canonical_url": decision.canonical_url,
        }


def create_cybersecurity_action_wrapper(
    action_fn: Callable[..., Any],
    governor: Optional[CybersecurityGovernor] = None,
    shadow_mode: bool = False,
    **policy_overrides: Any,
) -> CybersecurityActionWrapper:
    """Create a new CybersecurityActionWrapper instance."""
    return CybersecurityActionWrapper(
        action_fn=action_fn,
        governor=governor,
        shadow_mode=shadow_mode,
        policy_overrides=policy_overrides if policy_overrides else None,
    )
