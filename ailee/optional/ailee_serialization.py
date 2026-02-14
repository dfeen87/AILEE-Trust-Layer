"""
AILEE Trust Pipeline - Serialization Helpers
Version: 1.0.0

Provides utilities for serializing/deserializing DecisionResult objects
for logging, storage, transmission, and audit trails.
"""

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional
from enum import Enum


def _enum_serializer(obj: Any) -> Any:
    """Convert Enum values to their string representation."""
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _serialize_status(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def decision_to_dict(result) -> Dict[str, Any]:
    """
    Convert a DecisionResult to a dictionary.
    
    Args:
        result: DecisionResult instance
        
    Returns:
        Dictionary representation with all fields
        
    Example:
        >>> result = pipeline.process(...)
        >>> data = decision_to_dict(result)
        >>> print(data['value'])
    """
    if not is_dataclass(result):
        raise TypeError(f"Expected dataclass instance, got {type(result).__name__}")
    
    data = asdict(result)
    
    # Ensure enums are converted to their string values
    if 'safety_status' in data:
        data['safety_status'] = _serialize_status(data['safety_status'])
    if 'grace_status' in data:
        data['grace_status'] = _serialize_status(data['grace_status'])
    if 'consensus_status' in data:
        data['consensus_status'] = _serialize_status(data['consensus_status'])
    
    return data


def decision_to_json(result, indent: Optional[int] = 2, compact: bool = False) -> str:
    """
    Convert a DecisionResult to a JSON string.
    
    Args:
        result: DecisionResult instance
        indent: Number of spaces for indentation (None for compact)
        compact: If True, produces minimal JSON (overrides indent)
        
    Returns:
        JSON string representation
        
    Example:
        >>> result = pipeline.process(...)
        >>> json_str = decision_to_json(result)
        >>> print(json_str)
    """
    data = decision_to_dict(result)
    
    if compact:
        return json.dumps(data, separators=(',', ':'), default=_enum_serializer)
    
    return json.dumps(data, indent=indent, default=_enum_serializer)


def decision_from_dict(data: Dict[str, Any]):
    """
    Reconstruct a DecisionResult from a dictionary.
    
    Note: This requires the DecisionResult, SafetyStatus, GraceStatus, 
    and ConsensusStatus classes to be imported in your scope.
    
    Args:
        data: Dictionary representation of DecisionResult
        
    Returns:
        DecisionResult instance
        
    Example:
        >>> from ailee_trust_pipeline_v1 import DecisionResult, SafetyStatus
        >>> data = {'value': 10.5, 'safety_status': 'ACCEPTED', ...}
        >>> result = decision_from_dict(data)
    """
    # This is a placeholder - requires importing the actual classes
    # Users should implement this based on their import structure
    raise NotImplementedError(
        "decision_from_dict requires DecisionResult and Enum classes. "
        "Import them in your scope and implement deserialization."
    )


def decision_to_audit_log(result, include_metadata: bool = True) -> str:
    """
    Convert a DecisionResult to a human-readable audit log entry.
    
    Args:
        result: DecisionResult instance
        include_metadata: Whether to include full metadata
        
    Returns:
        Formatted audit log string
        
    Example:
        >>> result = pipeline.process(...)
        >>> log_entry = decision_to_audit_log(result)
        >>> logger.info(log_entry)
    """
    data = decision_to_dict(result)
    
    lines = [
        "=" * 60,
        "AILEE DECISION AUDIT LOG",
        "=" * 60,
        f"Final Value: {data['value']:.6f}",
        f"Safety Status: {data['safety_status']}",
        f"Grace Status: {data['grace_status']}",
        f"Consensus Status: {data['consensus_status']}",
        f"Used Fallback: {data['used_fallback']}",
        f"Confidence Score: {data['confidence_score']:.4f}",
        "",
        "Decision Trace:",
    ]
    
    for i, reason in enumerate(data.get('reasons', []), 1):
        lines.append(f"  {i}. {reason}")
    
    if include_metadata and data.get('metadata'):
        lines.append("")
        lines.append("Metadata:")
        lines.append(json.dumps(data['metadata'], indent=2))
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def decision_to_csv_row(result, include_header: bool = False) -> str:
    """
    Convert a DecisionResult to a CSV row (useful for batch logging).
    
    Args:
        result: DecisionResult instance
        include_header: If True, prepend CSV header row
        
    Returns:
        CSV formatted string
        
    Example:
        >>> results = [pipeline.process(...) for _ in range(10)]
        >>> with open('audit.csv', 'w') as f:
        >>>     f.write(decision_to_csv_row(results[0], include_header=True))
        >>>     for r in results[1:]:
        >>>         f.write(decision_to_csv_row(r))
    """
    data = decision_to_dict(result)
    
    header = "value,safety_status,grace_status,consensus_status,used_fallback,confidence_score,timestamp\n"
    
    timestamp = data.get('metadata', {}).get('timestamp', 'N/A')
    
    row = (
        f"{data['value']:.6f},"
        f"{data['safety_status']},"
        f"{data['grace_status']},"
        f"{data['consensus_status']},"
        f"{data['used_fallback']},"
        f"{data['confidence_score']:.4f},"
        f"{timestamp}\n"
    )
    
    if include_header:
        return header + row
    return row


def decision_to_compact_string(result) -> str:
    """
    Convert a DecisionResult to a compact, single-line string.
    Useful for inline logging or console output.
    
    Args:
        result: DecisionResult instance
        
    Returns:
        Compact string representation
        
    Example:
        >>> result = pipeline.process(...)
        >>> print(decision_to_compact_string(result))
        "value=10.5 safety=ACCEPTED grace=SKIPPED consensus=PASS fallback=False conf=0.95"
    """
    data = decision_to_dict(result)
    
    return (
        f"value={data['value']:.3f} "
        f"safety={data['safety_status']} "
        f"grace={data['grace_status']} "
        f"consensus={data['consensus_status']} "
        f"fallback={data['used_fallback']} "
        f"conf={data['confidence_score']:.2f}"
    )


# Convenience exports
__all__ = [
    'decision_to_dict',
    'decision_to_json',
    'decision_from_dict',
    'decision_to_audit_log',
    'decision_to_csv_row',
    'decision_to_compact_string',
]


# Demo usage
if __name__ == "__main__":
    # Mock DecisionResult for demonstration
    from dataclasses import dataclass, field
    from typing import List, Dict, Any
    from enum import Enum
    
    class SafetyStatus(str, Enum):
        ACCEPTED = "ACCEPTED"
        BORDERLINE = "BORDERLINE"
        OUTRIGHT_REJECTED = "OUTRIGHT_REJECTED"
    
    class GraceStatus(str, Enum):
        PASS = "PASS"
        FAIL = "FAIL"
        SKIPPED = "SKIPPED"
    
    class ConsensusStatus(str, Enum):
        PASS = "PASS"
        FAIL = "FAIL"
        SKIPPED = "SKIPPED"
    
    @dataclass(frozen=True)
    class DecisionResult:
        value: float
        safety_status: SafetyStatus
        grace_status: GraceStatus
        consensus_status: ConsensusStatus
        used_fallback: bool
        confidence_score: float
        reasons: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Create sample result
    result = DecisionResult(
        value=10.5,
        safety_status=SafetyStatus.ACCEPTED,
        grace_status=GraceStatus.SKIPPED,
        consensus_status=ConsensusStatus.PASS,
        used_fallback=False,
        confidence_score=0.95,
        reasons=["Safety: ACCEPTED (confidence_score=0.950 >= 0.90).", "Consensus: PASS (ratio=0.75, raw_ok=True)."],
        metadata={"timestamp": 1234567890.0, "context": {"feature": "temperature"}}
    )
    
    print("=== Dictionary ===")
    print(decision_to_dict(result))
    print()
    
    print("=== JSON (Pretty) ===")
    print(decision_to_json(result))
    print()
    
    print("=== JSON (Compact) ===")
    print(decision_to_json(result, compact=True))
    print()
    
    print("=== Audit Log ===")
    print(decision_to_audit_log(result))
    print()
    
    print("=== CSV Row ===")
    print(decision_to_csv_row(result, include_header=True))
    print()
    
    print("=== Compact String ===")
    print(decision_to_compact_string(result))
