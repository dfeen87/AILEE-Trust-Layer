import sys
import os

# Add parent directory to path to allow imports when running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ailee_trust_pipeline_v1 import AileeConfig
from backends import SoftwareBackend, FeenBackend

cfg = AileeConfig(hard_min=0.0, hard_max=100.0, consensus_quorum=3)

software = SoftwareBackend(cfg)
feen = FeenBackend(cfg)

inputs = dict(
    raw_value=10.2,
    raw_confidence=0.92,
    peer_values=[10.0, 10.1, 10.3],
    context={"feature": "latency_ms"},
)

print("Software:", software.process(**inputs))
print("FEEN:", feen.process(**inputs))
