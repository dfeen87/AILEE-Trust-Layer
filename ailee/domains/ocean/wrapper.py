import logging
from ailee import AileeClient, AileeConfig

logger = logging.getLogger("Ocean_AILEE_Governance")

class OceanGovernanceWrapper:
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

    def execute_ecological_intervention(self, ocean_data: dict, peer_buoys: list = None):
        """
        Executes ecological_intervention with AILEE governance.
        """
        raw_intervention_level, raw_confidence = self.legacy_ai.predict_intervention(ocean_data)

        if not self.ailee_enforced:
            logger.warning(f"LEGACY BYPASS: Executing unverified AI command: {raw_intervention_level}")
            self._actuate_system(raw_intervention_level)
            return raw_intervention_level

        peer_values = [p.get_prediction() for p in peer_buoys] if peer_buoys else []
        result = self.trust_layer.process(
            raw_value=raw_intervention_level, raw_confidence=raw_confidence, peer_values=peer_values,
            context={"operation": "ecological_intervention", "ocean_data": ocean_data}
        )

        if result.safety_status == "OUTRIGHT_REJECTED":
            logger.error(f"AILEE INTERVENTION: Unsafe ecological intervention rejected. Using safe fallback: {result.value}")
            self._actuate_system(result.value)
            return result.value

        logger.info(f"AILEE APPROVED: Executing verified command: {result.value}")
        self._actuate_system(result.value)
        return result.value

    def _actuate_system(self, final_intervention_level: float):
        pass # Placeholder for physical actuation
