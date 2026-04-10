import re

with open('ailee/domains/crypto_mining/ailee_crypto_mining_domain.py', 'r') as f:
    content = f.read()

INJECTION = """
# Compatibility
def get_health(self):
    return getattr(self, '_health_status', self.policy.min_trust_for_action)

MiningGovernor.get_health = get_health

def get_subsystem_health(self):
    return {"default": self.get_health()}

MiningGovernor.get_subsystem_health = get_subsystem_health

def get_events(self):
    return getattr(self, '_events', [])

MiningGovernor.get_events = get_events

def get_metrics(self):
    return {"decisions": len(self.get_events())}

MiningGovernor.get_metrics = get_metrics

def get_decision_history(self):
    return getattr(self, '_events', [])

MiningGovernor.get_decision_history = get_decision_history
"""
with open('ailee/domains/crypto_mining/ailee_crypto_mining_domain.py', 'a') as f:
    f.write(INJECTION)
