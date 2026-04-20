import logging
from ailee import AileeClient, AileeConfig

logger = logging.getLogger("Governance_AILEE_Governance")

class GovernanceGovernanceWrapper:
    def __init__(self, legacy_ai_controller, start_in_shadow_mode=False):
        self.legacy_ai = legacy_ai_controller
        self.ailee_enforced = not start_in_shadow_mode
        self.trust_layer = AileeClient(AileeConfig(borderline_low=0.75, borderline_high=0.90))
        logger.info(f"System Initialized. AILEE Enforced: {self.ailee_enforced}")

    def revert_to_legacy_logic(self, reason: str):
        self.ailee_enforced = False
        logger.critical(f"🛑 AILEE GOVERNANCE SEVERED. Reverting to legacy AI logic. Reason: {reason}")

    def enforce_ailee_governance(self):
        self.ailee_enforced = True
        logger.info("✅ AILEE GOVERNANCE RE-ENGAGED. Trust thresholds active.")

    def execute_policy_automation(self, policy_data: dict, peer_policies: list = None):
        """
        Executes policy_automation with AILEE governance.
        """
        raw_policy_value, raw_confidence = self.legacy_ai.predict_policy(policy_data)

        if not self.ailee_enforced:
            logger.warning(f"LEGACY BYPASS: Executing unverified AI command: {raw_policy_value}")
            self._actuate_policy(raw_policy_value)
            return raw_policy_value

        peer_values = [p.get_prediction() for p in peer_policies] if peer_policies else []
        result = self.trust_layer.process(
            raw_value=raw_policy_value, raw_confidence=raw_confidence, peer_values=peer_values,
            context={"operation": "policy_automation", "policy_data": policy_data}
        )

        if result.safety_status == "OUTRIGHT_REJECTED":
            logger.error(f"AILEE INTERVENTION: Unsafe policy rejected. Using safe fallback: {result.value}")
            self._actuate_policy(result.value)
            return result.value

        logger.info(f"AILEE APPROVED: Executing verified command: {result.value}")
        self._actuate_policy(result.value)
        return result.value

    def _actuate_policy(self, final_policy_value: float):
        pass # Placeholder for physical actuation
