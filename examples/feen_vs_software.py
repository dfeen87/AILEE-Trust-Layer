import sys
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add parent directory to path to allow imports when running from examples directory

from ailee import AileeConfig
from ailee.backends import SoftwareBackend, FeenBackend

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
