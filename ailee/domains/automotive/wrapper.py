import logging
from ailee import AileeClient, AileeConfig

logger = logging.getLogger("Automotive_AILEE_Governance")

class AutomotiveGovernanceWrapper:
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

    def execute_autonomous_maneuver(self, telemetry_data: dict, peer_vehicles: list = None):
        """
        Executes autonomous_maneuver with AILEE governance.
        """
        raw_steering_angle, raw_confidence = self.legacy_ai.predict_maneuver(telemetry_data)

        if not self.ailee_enforced:
            logger.warning(f"LEGACY BYPASS: Executing unverified AI command: {raw_steering_angle}")
            self._actuate_vehicle(raw_steering_angle)
            return raw_steering_angle

        peer_values = [p.get_prediction() for p in peer_vehicles] if peer_vehicles else []
        result = self.trust_layer.process(
            raw_value=raw_steering_angle, raw_confidence=raw_confidence, peer_values=peer_values,
            context={"operation": "autonomous_maneuver", "telemetry_data": telemetry_data}
        )

        if result.safety_status == "OUTRIGHT_REJECTED":
            logger.error(f"AILEE INTERVENTION: Unsafe maneuver rejected. Using safe fallback: {result.value}")
            self._actuate_vehicle(result.value)
            return result.value

        logger.info(f"AILEE APPROVED: Executing verified command: {result.value}")
        self._actuate_vehicle(result.value)
        return result.value

    def _actuate_vehicle(self, final_steering_angle: float):
        pass # Placeholder for physical actuation
