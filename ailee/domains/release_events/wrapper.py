import logging
from ailee import AileeClient, AileeConfig

logger = logging.getLogger("ReleaseEvents_AILEE_Governance")

class ReleaseEventsGovernanceWrapper:
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

    def execute_rollout_gate(self, release_data: dict, peer_gates: list = None):
        """
        Executes rollout_gate with AILEE governance.
        """
        raw_rollout_rate, raw_confidence = self.legacy_ai.predict_rollout(release_data)

        if not self.ailee_enforced:
            logger.warning(f"LEGACY BYPASS: Executing unverified AI command: {raw_rollout_rate}")
            self._actuate_pipeline(raw_rollout_rate)
            return raw_rollout_rate

        peer_values = [p.get_prediction() for p in peer_gates] if peer_gates else []
        result = self.trust_layer.process(
            raw_value=raw_rollout_rate, raw_confidence=raw_confidence, peer_values=peer_values,
            context={"operation": "rollout_gate", "release_data": release_data}
        )

        if result.safety_status == "OUTRIGHT_REJECTED":
            logger.error(f"AILEE INTERVENTION: Unsafe rollout rejected. Using safe fallback: {result.value}")
            self._actuate_pipeline(result.value)
            return result.value

        logger.info(f"AILEE APPROVED: Executing verified command: {result.value}")
        self._actuate_pipeline(result.value)
        return result.value

    def _actuate_pipeline(self, final_rollout_rate: float):
        pass # Placeholder for physical actuation
