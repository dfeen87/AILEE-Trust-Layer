import logging
from ailee import AileeClient, AileeConfig

logger = logging.getLogger("Auditory_AILEE_Governance")

class AuditoryGovernanceWrapper:
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

    def execute_enhancement_level(self, audio_data: dict, peer_devices: list = None):
        """
        Executes enhancement_level with AILEE governance.
        """
        raw_level, raw_confidence = self.legacy_ai.predict_enhancement(audio_data)

        if not self.ailee_enforced:
            logger.warning(f"LEGACY BYPASS: Executing unverified AI command: {raw_level}")
            self._actuate_device(raw_level)
            return raw_level

        peer_values = [p.get_prediction() for p in peer_devices] if peer_devices else []
        result = self.trust_layer.process(
            raw_value=raw_level, raw_confidence=raw_confidence, peer_values=peer_values,
            context={"operation": "enhancement_level", "audio_data": audio_data}
        )

        if result.safety_status == "OUTRIGHT_REJECTED":
            logger.error(f"AILEE INTERVENTION: Unsafe enhancement level rejected. Using safe fallback: {result.value}")
            self._actuate_device(result.value)
            return result.value

        logger.info(f"AILEE APPROVED: Executing verified command: {result.value}")
        self._actuate_device(result.value)
        return result.value

    def _actuate_device(self, final_level: float):
        pass # Placeholder for physical actuation
