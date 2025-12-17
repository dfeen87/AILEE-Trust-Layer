"""
Serialization helpers for AILEE DecisionResult.
"""

import json
from dataclasses import asdict


def decision_to_dict(result):
    return asdict(result)


def decision_to_json(result, indent=2):
    return json.dumps(asdict(result), indent=indent)
