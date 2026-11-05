# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""
AILEE Action Authorization Wrapper — Cybersecurity Domain.

Provides deterministic action interception, strict parameter binding,
and trust enforcement for external link display, software recommendations,
web visits, file downloads, and installations.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from .cybersecurity import (
    ActionTier,
    CybersecurityAuthorizationRequest,
    CybersecurityDecision,
    CybersecurityGovernor,
    CybersecurityReason,
    DecisionOutcome,
    HIGH_RISK_ACTION_TIERS,
    create_cybersecurity_governor,
)

_URL_PARAM_NAMES = frozenset(
    [
        "url",
        "target",
        "target_url",
        "destination",
        "download_url",
        "link",
        "href",
        "package_url",
        "installer_url",
    ]
)


def _looks_like_url(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    v = val.strip()
    if v.startswith(("http://", "https://", "ftp://", "file://")):
        return True
    if "." in v and "/" in v and " " not in v:
        return True
    return False


class CybersecurityActionWrapper:
    """
    Interception wrapper for agent tools and execution pipelines.

    Guarantees deterministic trust governance and binds the exact execution target
    to the authorized request before opening URLs, recommending software,
    downloading binaries, or executing installers.
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
        Execute the intercepted action only if authorized by the cybersecurity governor
        and verify that the executed target matches the authorized URL.
        """
        decision = self.governor.authorize(request)
        is_high_risk = request.action_tier in HIGH_RISK_ACTION_TIERS

        # -----------------------------------------------------------------
        # 1. Parameter Inspection & Target Binding
        # -----------------------------------------------------------------
        # Detect any candidate URL arguments passed to the underlying tool
        bound_args = list(args)
        bound_kwargs = dict(kwargs)
        url_substituted = False

        # Positional arguments check
        for idx, arg in enumerate(args):
            if _looks_like_url(arg):
                arg_str = str(arg).strip()
                # Must match request URL or canonical URL
                canonical = decision.canonical_url or request.url
                if (
                    arg_str != request.url.strip()
                    and arg_str != canonical.strip()
                ):
                    return {
                        "blocked": True,
                        "requires_review": False,
                        "shadow_mode": self.shadow_mode,
                        "outcome": DecisionOutcome.DENY.value,
                        "reason": CybersecurityReason.PARAMETER_MISMATCH.value,
                        "decision": decision,
                        "result": None,
                        "canonical_url": decision.canonical_url,
                        "details": {
                            "error": (
                                f"Execution target substitution rejected: positional arg[{idx}] '{arg_str}' "
                                f"does not match authorized URL '{canonical}'"
                            )
                        },
                    }
                # Bind exact canonical authorized URL
                if decision.canonical_url:
                    bound_args[idx] = decision.canonical_url
                    url_substituted = True

        # Keyword arguments check
        for key, val in kwargs.items():
            if key in _URL_PARAM_NAMES or _looks_like_url(val):
                val_str = str(val).strip()
                canonical = decision.canonical_url or request.url
                if (
                    val_str != request.url.strip()
                    and val_str != canonical.strip()
                ):
                    return {
                        "blocked": True,
                        "requires_review": False,
                        "shadow_mode": self.shadow_mode,
                        "outcome": DecisionOutcome.DENY.value,
                        "reason": CybersecurityReason.PARAMETER_MISMATCH.value,
                        "decision": decision,
                        "result": None,
                        "canonical_url": decision.canonical_url,
                        "details": {
                            "error": (
                                f"Execution target substitution rejected: kwarg '{key}'='{val_str}' "
                                f"does not match authorized URL '{canonical}'"
                            )
                        },
                    }
                if decision.canonical_url:
                    bound_kwargs[key] = decision.canonical_url
                    url_substituted = True

        # If no URL argument was passed, bind canonical URL if positional arguments are empty
        if (
            not url_substituted
            and not bound_args
            and "url" not in bound_kwargs
            and decision.canonical_url
        ):
            bound_args.append(decision.canonical_url)

        # -----------------------------------------------------------------
        # 2. Enforcement for DENY Decisions
        # Invariant: A DENY decision MUST NEVER execute, even in shadow_mode!
        # -----------------------------------------------------------------
        if decision.outcome == DecisionOutcome.DENY:
            return {
                "blocked": True,
                "requires_review": False,
                "shadow_mode": self.shadow_mode,
                "outcome": DecisionOutcome.DENY.value,
                "reason": decision.reason.value,
                "decision": decision,
                "result": None,
                "canonical_url": decision.canonical_url,
            }

        # -----------------------------------------------------------------
        # 3. Enforcement for REVIEW Decisions
        # Invariant: High-risk actions ALWAYS require confirmation.
        # Disabling confirmation can NEVER bypass review for high-risk actions.
        # -----------------------------------------------------------------
        if decision.outcome == DecisionOutcome.REVIEW:
            if is_high_risk or self.require_confirmation_for_review:
                return {
                    "blocked": True,
                    "requires_review": True,
                    "shadow_mode": self.shadow_mode,
                    "outcome": DecisionOutcome.REVIEW.value,
                    "reason": decision.reason.value,
                    "decision": decision,
                    "result": None,
                    "canonical_url": decision.canonical_url,
                }
            # For low-risk informational display with confirmation explicitly disabled in non-shadow mode
            if not self.shadow_mode:
                result = self.action_fn(*bound_args, **bound_kwargs)
                return {
                    "blocked": False,
                    "requires_review": False,
                    "shadow_mode": False,
                    "outcome": DecisionOutcome.REVIEW.value,
                    "reason": decision.reason.value,
                    "decision": decision,
                    "result": result,
                    "canonical_url": decision.canonical_url,
                }

        # -----------------------------------------------------------------
        # 4. Authorized Execution (ALLOW)
        # -----------------------------------------------------------------
        result = self.action_fn(*bound_args, **bound_kwargs)
        return {
            "blocked": False,
            "requires_review": False,
            "shadow_mode": self.shadow_mode,
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
    require_confirmation_for_review: bool = True,
    **policy_overrides: Any,
) -> CybersecurityActionWrapper:
    """Create a new CybersecurityActionWrapper instance."""
    return CybersecurityActionWrapper(
        action_fn=action_fn,
        governor=governor,
        shadow_mode=shadow_mode,
        require_confirmation_for_review=require_confirmation_for_review,
        policy_overrides=policy_overrides if policy_overrides else None,
    )
