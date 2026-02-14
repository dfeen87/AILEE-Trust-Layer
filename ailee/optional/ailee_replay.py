"""
AILEE Trust Pipeline - Deterministic Replay Utilities
Version: 1.0.0

Provides utilities for recording, replaying, and analyzing AILEE pipeline
execution for debugging, testing, regression analysis, and audit verification.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict, is_dataclass
import json
import time
from pathlib import Path
from enum import Enum


@dataclass
class ReplayRecord:
    """Single recorded execution of the AILEE pipeline."""
    inputs: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    record_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """
    Buffer for recording and replaying AILEE Trust Pipeline executions.
    
    Use cases:
    - Regression testing (verify behavior hasn't changed)
    - Debugging (replay problematic scenarios)
    - Audit trails (deterministic verification)
    - Performance analysis (compare before/after optimization)
    - Configuration tuning (test different configs on same data)
    """
    
    def __init__(self):
        self.records: List[ReplayRecord] = []
        self._record_count = 0
    
    def record(
        self, 
        inputs: Dict[str, Any], 
        result: Any,
        record_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a single pipeline execution.
        
        Args:
            inputs: Dictionary of inputs passed to pipeline.process()
            result: DecisionResult returned from pipeline.process()
            record_id: Optional unique identifier for this record
            metadata: Optional additional metadata (e.g., environment info)
            
        Example:
            >>> buffer = ReplayBuffer()
            >>> inputs = {'raw_value': 10.5, 'raw_confidence': 0.75}
            >>> result = pipeline.process(**inputs)
            >>> buffer.record(inputs, result)
        """
        result_dict = self._normalize_result(result)
        
        record = ReplayRecord(
            inputs=dict(inputs),
            result=result_dict,
            timestamp=time.time(),
            record_id=record_id or f"record_{self._record_count:06d}",
            metadata=dict(metadata or {})
        )
        
        self.records.append(record)
        self._record_count += 1
    
    def replay(self, pipeline, verbose: bool = False) -> List[Any]:
        """
        Replay all recorded executions through the pipeline.
        
        Args:
            pipeline: AileeTrustPipeline instance (fresh or with state)
            verbose: If True, print progress during replay
            
        Returns:
            List of DecisionResult objects from replay
            
        Example:
            >>> buffer = ReplayBuffer()
            >>> # ... record some executions ...
            >>> new_pipeline = AileeTrustPipeline(new_config)
            >>> replayed_results = buffer.replay(new_pipeline)
        """
        outputs = []
        
        for i, record in enumerate(self.records):
            if verbose:
                print(f"Replaying {i+1}/{len(self.records)}: {record.record_id}")
            
            result = pipeline.process(**record.inputs)
            outputs.append(result)
        
        return outputs
    
    def compare_replay(
        self, 
        pipeline, 
        tolerance: float = 1e-6,
        compare_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Replay and compare results with original recordings.
        
        Args:
            pipeline: AileeTrustPipeline instance
            tolerance: Absolute tolerance for value comparison
            compare_metadata: If True, also compare metadata fields
            
        Returns:
            Dictionary containing comparison statistics and mismatches
            
        Example:
            >>> comparison = buffer.compare_replay(pipeline, tolerance=0.001)
            >>> print(f"Match rate: {comparison['match_rate']:.2%}")
            >>> if comparison['mismatches']:
            ...     print(f"Found {len(comparison['mismatches'])} mismatches")
        """
        replayed = self.replay(pipeline)
        
        matches = 0
        mismatches = []
        value_diffs = []
        
        for i, (record, replayed_result) in enumerate(zip(self.records, replayed)):
            original = record.result
            replayed_dict = self._normalize_result(replayed_result)
            
            # Compare key fields
            value_match = abs(original['value'] - replayed_dict['value']) <= tolerance
            safety_match = original['safety_status'] == replayed_dict['safety_status']
            grace_match = original['grace_status'] == replayed_dict['grace_status']
            consensus_match = original['consensus_status'] == replayed_dict['consensus_status']
            fallback_match = original['used_fallback'] == replayed_dict['used_fallback']
            
            is_match = all([value_match, safety_match, grace_match, consensus_match, fallback_match])
            
            if is_match:
                matches += 1
            else:
                mismatch = {
                    'record_id': record.record_id,
                    'index': i,
                    'original_value': original['value'],
                    'replayed_value': replayed_dict['value'],
                    'value_diff': abs(original['value'] - replayed_dict['value']),
                    'original_safety': original['safety_status'],
                    'replayed_safety': replayed_dict['safety_status'],
                    'original_grace': original['grace_status'],
                    'replayed_grace': replayed_dict['grace_status'],
                    'original_consensus': original['consensus_status'],
                    'replayed_consensus': replayed_dict['consensus_status'],
                    'original_fallback': original['used_fallback'],
                    'replayed_fallback': replayed_dict['used_fallback'],
                }
                mismatches.append(mismatch)
            
            value_diffs.append(abs(original['value'] - replayed_dict['value']))
        
        total = len(self.records)
        match_rate = matches / total if total > 0 else 0.0
        
        return {
            'total_records': total,
            'matches': matches,
            'mismatches_count': len(mismatches),
            'match_rate': match_rate,
            'mismatches': mismatches,
            'max_value_diff': max(value_diffs) if value_diffs else 0.0,
            'mean_value_diff': sum(value_diffs) / len(value_diffs) if value_diffs else 0.0,
        }
    
    def save(self, filepath: str) -> None:
        """
        Save replay buffer to JSON file.
        
        Args:
            filepath: Path to save the replay buffer
            
        Example:
            >>> buffer.save('replay_log_20250101.json')
        """
        data = {
            'version': '1.0.0',
            'record_count': len(self.records),
            'records': [asdict(record) for record in self.records]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ReplayBuffer':
        """
        Load replay buffer from JSON file.
        
        Args:
            filepath: Path to the saved replay buffer
            
        Returns:
            ReplayBuffer instance with loaded records
            
        Example:
            >>> buffer = ReplayBuffer.load('replay_log_20250101.json')
            >>> results = buffer.replay(pipeline)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        buffer = cls()
        
        for record_data in data['records']:
            record = ReplayRecord(**record_data)
            buffer.records.append(record)
        
        buffer._record_count = len(buffer.records)
        
        return buffer

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        if is_dataclass(result):
            data = asdict(result)
        elif isinstance(result, dict):
            data = dict(result)
        else:
            raise TypeError("ReplayBuffer requires DecisionResult or dict-like results.")

        for key in ("safety_status", "grace_status", "consensus_status"):
            if key in data and isinstance(data[key], Enum):
                data[key] = data[key].value
        return data
    
    def filter_by_status(
        self, 
        safety_status: Optional[str] = None,
        grace_status: Optional[str] = None,
        consensus_status: Optional[str] = None,
        used_fallback: Optional[bool] = None
    ) -> 'ReplayBuffer':
        """
        Create a filtered replay buffer based on result status.
        
        Args:
            safety_status: Filter by safety status (e.g., "BORDERLINE")
            grace_status: Filter by grace status (e.g., "PASS")
            consensus_status: Filter by consensus status (e.g., "FAIL")
            used_fallback: Filter by fallback usage
            
        Returns:
            New ReplayBuffer with filtered records
            
        Example:
            >>> # Get all borderline cases that passed grace
            >>> borderline = buffer.filter_by_status(
            ...     safety_status="BORDERLINE",
            ...     grace_status="PASS"
            ... )
        """
        filtered = ReplayBuffer()
        
        for record in self.records:
            result = record.result
            
            if safety_status and result.get('safety_status') != safety_status:
                continue
            if grace_status and result.get('grace_status') != grace_status:
                continue
            if consensus_status and result.get('consensus_status') != consensus_status:
                continue
            if used_fallback is not None and result.get('used_fallback') != used_fallback:
                continue
            
            filtered.records.append(record)
        
        filtered._record_count = len(filtered.records)
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over all recorded executions.
        
        Returns:
            Dictionary with statistics about the replay buffer
            
        Example:
            >>> stats = buffer.get_statistics()
            >>> print(f"Fallback rate: {stats['fallback_rate']:.2%}")
        """
        if not self.records:
            return {
                'total_records': 0,
                'fallback_rate': 0.0,
                'safety_distribution': {},
                'grace_distribution': {},
                'consensus_distribution': {},
            }
        
        total = len(self.records)
        fallback_count = sum(1 for r in self.records if r.result.get('used_fallback'))
        
        safety_dist = {}
        grace_dist = {}
        consensus_dist = {}
        
        for record in self.records:
            result = record.result
            
            safety = result.get('safety_status', 'UNKNOWN')
            grace = result.get('grace_status', 'UNKNOWN')
            consensus = result.get('consensus_status', 'UNKNOWN')
            
            safety_dist[safety] = safety_dist.get(safety, 0) + 1
            grace_dist[grace] = grace_dist.get(grace, 0) + 1
            consensus_dist[consensus] = consensus_dist.get(consensus, 0) + 1
        
        return {
            'total_records': total,
            'fallback_count': fallback_count,
            'fallback_rate': fallback_count / total,
            'safety_distribution': safety_dist,
            'grace_distribution': grace_dist,
            'consensus_distribution': consensus_dist,
        }
    
    def clear(self) -> None:
        """Clear all recorded executions."""
        self.records.clear()
        self._record_count = 0
    
    def __len__(self) -> int:
        """Return number of recorded executions."""
        return len(self.records)
    
    def __repr__(self) -> str:
        return f"ReplayBuffer(records={len(self.records)})"


# Convenience exports
__all__ = [
    'ReplayBuffer',
    'ReplayRecord',
]


# Demo usage
if __name__ == "__main__":
    print("=== AILEE Replay Buffer Demo ===\n")
    
    # Simulate some recordings
    buffer = ReplayBuffer()
    
    # Mock some executions
    mock_inputs = [
        {'raw_value': 10.5, 'raw_confidence': 0.95},
        {'raw_value': 10.2, 'raw_confidence': 0.92},
        {'raw_value': 15.0, 'raw_confidence': 0.75},
        {'raw_value': 10.3, 'raw_confidence': 0.91},
        {'raw_value': 200.0, 'raw_confidence': 0.10},
    ]
    
    mock_results = [
        {'value': 10.5, 'safety_status': 'ACCEPTED', 'grace_status': 'SKIPPED', 
         'consensus_status': 'PASS', 'used_fallback': False, 'confidence_score': 0.95},
        {'value': 10.2, 'safety_status': 'ACCEPTED', 'grace_status': 'SKIPPED',
         'consensus_status': 'PASS', 'used_fallback': False, 'confidence_score': 0.92},
        {'value': 15.0, 'safety_status': 'BORDERLINE', 'grace_status': 'PASS',
         'consensus_status': 'PASS', 'used_fallback': False, 'confidence_score': 0.75},
        {'value': 10.3, 'safety_status': 'ACCEPTED', 'grace_status': 'SKIPPED',
         'consensus_status': 'PASS', 'used_fallback': False, 'confidence_score': 0.91},
        {'value': 10.2, 'safety_status': 'OUTRIGHT_REJECTED', 'grace_status': 'SKIPPED',
         'consensus_status': 'SKIPPED', 'used_fallback': True, 'confidence_score': 0.10},
    ]
    
    for inputs, result in zip(mock_inputs, mock_results):
        buffer.record(inputs, result)
    
    print(f"Buffer: {buffer}")
    print(f"Total records: {len(buffer)}\n")
    
    # Statistics
    stats = buffer.get_statistics()
    print("=== Statistics ===")
    print(f"Fallback rate: {stats['fallback_rate']:.2%}")
    print(f"Safety distribution: {stats['safety_distribution']}")
    print(f"Grace distribution: {stats['grace_distribution']}\n")
    
    # Filter examples
    borderline = buffer.filter_by_status(safety_status="BORDERLINE")
    print(f"Borderline cases: {len(borderline)}")
    
    fallback_cases = buffer.filter_by_status(used_fallback=True)
    print(f"Fallback cases: {len(fallback_cases)}\n")
    
    # Save/load demo
    print("=== Save/Load Demo ===")
    buffer.save('/tmp/ailee_replay_demo.json')
    print("Saved to /tmp/ailee_replay_demo.json")
    
    loaded = ReplayBuffer.load('/tmp/ailee_replay_demo.json')
    print(f"Loaded buffer: {loaded}")
    print(f"Records match: {len(buffer) == len(loaded)}")
