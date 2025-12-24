"""
AILEE Trust Pipeline - Peer Adapters
Version: 1.0.0

Adapters for integrating peer values from distributed or multi-model systems
into the AILEE consensus layer.

Peer adapters enable:
- Multi-model consensus (LLM ensembles, model fusion)
- Distributed sensor networks
- Multi-agent systems
- Federated AI deployments
"""

from typing import List, Protocol, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import time
import statistics


class PeerAdapter(Protocol):
    """Protocol defining the interface for peer value providers."""
    
    def get_peer_values(self) -> List[float]:
        """
        Retrieve current peer values for consensus checking.
        
        Returns:
            List of float values from peer systems
        """
        ...


# ===========================
# Basic Adapters
# ===========================

class StaticPeerAdapter:
    """
    Adapter for static/fixed peer values.
    Useful for testing or when peer values are pre-computed.
    
    Example:
        >>> adapter = StaticPeerAdapter([10.2, 10.5, 10.3])
        >>> peers = adapter.get_peer_values()
    """
    
    def __init__(self, peers: List[float]):
        """
        Initialize with static peer values.
        
        Args:
            peers: List of static peer values
        """
        self.peers = list(peers)
    
    def get_peer_values(self) -> List[float]:
        """Return the static peer values."""
        return self.peers[:]
    
    def update(self, peers: List[float]) -> None:
        """
        Update the static peer values.
        
        Args:
            peers: New list of peer values
        """
        self.peers = list(peers)


class RollingPeerAdapter:
    """
    Adapter that returns recent values from a rolling history.
    Useful for temporal consensus (comparing with recent past values).
    
    Example:
        >>> history = [10.1, 10.2, 10.3, 10.4, 10.5]
        >>> adapter = RollingPeerAdapter(history, window=3)
        >>> peers = adapter.get_peer_values()  # [10.3, 10.4, 10.5]
    """
    
    def __init__(self, history: List[float], window: int = 10):
        """
        Initialize with a history buffer.
        
        Args:
            history: List of historical values
            window: Number of recent values to return
        """
        self.history = list(history)
        self.window = window
    
    def get_peer_values(self) -> List[float]:
        """Return the most recent window of values."""
        return self.history[-self.window:] if self.history else []
    
    def append(self, value: float) -> None:
        """
        Add a new value to the history.
        
        Args:
            value: New value to append
        """
        self.history.append(value)
    
    def extend(self, values: List[float]) -> None:
        """
        Add multiple values to the history.
        
        Args:
            values: List of values to append
        """
        self.history.extend(values)


# ===========================
# Advanced Adapters
# ===========================

class CallbackPeerAdapter:
    """
    Adapter that calls a function to retrieve peer values dynamically.
    Useful for integrating with external APIs, databases, or live systems.
    
    Example:
        >>> def get_remote_peers():
        ...     return fetch_from_api('/peer-values')
        >>> adapter = CallbackPeerAdapter(get_remote_peers)
        >>> peers = adapter.get_peer_values()
    """
    
    def __init__(
        self, 
        callback: Callable[[], List[float]],
        cache_ttl: Optional[float] = None
    ):
        """
        Initialize with a callback function.
        
        Args:
            callback: Function that returns List[float] of peer values
            cache_ttl: Optional time-to-live for caching (seconds)
        """
        self.callback = callback
        self.cache_ttl = cache_ttl
        self._cached_values: Optional[List[float]] = None
        self._cache_timestamp: Optional[float] = None
    
    def get_peer_values(self) -> List[float]:
        """
        Call the callback to get peer values, with optional caching.
        
        Returns:
            List of peer values from callback
        """
        now = time.time()
        
        # Check cache validity
        if self.cache_ttl is not None and self._cached_values is not None:
            if self._cache_timestamp is not None:
                age = now - self._cache_timestamp
                if age < self.cache_ttl:
                    return self._cached_values[:]
        
        # Fetch fresh values
        values = self.callback()
        
        # Update cache
        if self.cache_ttl is not None:
            self._cached_values = list(values)
            self._cache_timestamp = now
        
        return list(values)
    
    def invalidate_cache(self) -> None:
        """Force cache invalidation on next call."""
        self._cached_values = None
        self._cache_timestamp = None


class MultiSourcePeerAdapter:
    """
    Adapter that aggregates peer values from multiple sources.
    Useful for combining different model outputs or sensor readings.
    
    Example:
        >>> model1 = StaticPeerAdapter([10.2, 10.3])
        >>> model2 = StaticPeerAdapter([10.4, 10.5])
        >>> adapter = MultiSourcePeerAdapter([model1, model2])
        >>> peers = adapter.get_peer_values()  # [10.2, 10.3, 10.4, 10.5]
    """
    
    def __init__(self, sources: List[PeerAdapter]):
        """
        Initialize with multiple peer adapters.
        
        Args:
            sources: List of PeerAdapter instances
        """
        self.sources = list(sources)
    
    def get_peer_values(self) -> List[float]:
        """Aggregate values from all sources."""
        aggregated = []
        for source in self.sources:
            aggregated.extend(source.get_peer_values())
        return aggregated
    
    def add_source(self, source: PeerAdapter) -> None:
        """
        Add a new peer source.
        
        Args:
            source: PeerAdapter to add
        """
        self.sources.append(source)
    
    def remove_source(self, index: int) -> None:
        """
        Remove a peer source by index.
        
        Args:
            index: Index of source to remove
        """
        if 0 <= index < len(self.sources):
            self.sources.pop(index)


class WeightedPeerAdapter:
    """
    Adapter that applies weights to peer values from different sources.
    Useful when some peers are more trusted than others.
    
    Example:
        >>> sources = [
        ...     StaticPeerAdapter([10.0]),
        ...     StaticPeerAdapter([12.0]),
        ... ]
        >>> weights = [0.8, 0.2]  # Trust first source more
        >>> adapter = WeightedPeerAdapter(sources, weights)
        >>> peers = adapter.get_peer_values()  # Weighted combination
    """
    
    def __init__(
        self, 
        sources: List[PeerAdapter], 
        weights: List[float],
        mode: str = "duplicate"
    ):
        """
        Initialize with sources and their weights.
        
        Args:
            sources: List of PeerAdapter instances
            weights: Corresponding weights (will be normalized)
            mode: "duplicate" (repeat values) or "scale" (scale values)
        """
        if len(sources) != len(weights):
            raise ValueError("Number of sources must match number of weights")
        
        self.sources = list(sources)
        
        # Normalize weights
        total = sum(weights)
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        self.weights = [w / total for w in weights]
        self.mode = mode
    
    def get_peer_values(self) -> List[float]:
        """
        Get weighted peer values.
        
        Returns:
            List of peer values adjusted by weights
        """
        if self.mode == "duplicate":
            # Repeat values based on weights (round to integers)
            weighted = []
            for source, weight in zip(self.sources, self.weights):
                values = source.get_peer_values()
                repeat_count = max(1, int(weight * 10))  # Scale to reasonable count
                weighted.extend(values * repeat_count)
            return weighted
        
        elif self.mode == "scale":
            # Scale values by weights
            weighted = []
            for source, weight in zip(self.sources, self.weights):
                values = source.get_peer_values()
                weighted.extend([v * weight for v in values])
            return weighted
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class FilteredPeerAdapter:
    """
    Adapter that filters peer values based on criteria.
    Useful for removing outliers or invalid readings.
    
    Example:
        >>> source = StaticPeerAdapter([10.0, 100.0, 10.5, -50.0, 10.3])
        >>> adapter = FilteredPeerAdapter(source, min_value=0.0, max_value=50.0)
        >>> peers = adapter.get_peer_values()  # [10.0, 10.5, 10.3]
    """
    
    def __init__(
        self,
        source: PeerAdapter,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        max_deviation_from_median: Optional[float] = None
    ):
        """
        Initialize with source and filter criteria.
        
        Args:
            source: Source peer adapter
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            max_deviation_from_median: Maximum absolute deviation from median
        """
        self.source = source
        self.min_value = min_value
        self.max_value = max_value
        self.max_deviation_from_median = max_deviation_from_median
    
    def get_peer_values(self) -> List[float]:
        """Get filtered peer values."""
        values = self.source.get_peer_values()
        
        if not values:
            return []
        
        filtered = values[:]
        
        # Apply min/max filters
        if self.min_value is not None:
            filtered = [v for v in filtered if v >= self.min_value]
        
        if self.max_value is not None:
            filtered = [v for v in filtered if v <= self.max_value]
        
        # Apply median deviation filter
        if self.max_deviation_from_median is not None and filtered:
            median = statistics.median(filtered)
            filtered = [
                v for v in filtered 
                if abs(v - median) <= self.max_deviation_from_median
            ]
        
        return filtered


@dataclass
class PeerMetadata:
    """Metadata about a peer source."""
    name: str
    source_type: str
    confidence: float = 1.0
    last_update: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetadataPeerAdapter:
    """
    Adapter that includes metadata about peer sources.
    Useful for tracking provenance and trust signals.
    
    Example:
        >>> adapter = MetadataPeerAdapter()
        >>> adapter.add_peer_source(
        ...     "model_a", 
        ...     StaticPeerAdapter([10.5]), 
        ...     confidence=0.95
        ... )
        >>> peers, metadata = adapter.get_peer_values_with_metadata()
    """
    
    def __init__(self):
        self.sources: Dict[str, PeerAdapter] = {}
        self.metadata: Dict[str, PeerMetadata] = {}
    
    def add_peer_source(
        self,
        name: str,
        source: PeerAdapter,
        confidence: float = 1.0,
        source_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a peer source with metadata.
        
        Args:
            name: Unique identifier for this source
            source: PeerAdapter instance
            confidence: Confidence/trust level (0-1)
            source_type: Type descriptor (e.g., "llm", "sensor", "model")
            metadata: Additional metadata
        """
        self.sources[name] = source
        self.metadata[name] = PeerMetadata(
            name=name,
            source_type=source_type,
            confidence=confidence,
            last_update=time.time(),
            metadata=dict(metadata or {})
        )
    
    def get_peer_values(self) -> List[float]:
        """Get all peer values (without metadata)."""
        values = []
        for source in self.sources.values():
            values.extend(source.get_peer_values())
        return values
    
    def get_peer_values_with_metadata(self) -> tuple[List[float], List[PeerMetadata]]:
        """
        Get peer values along with their metadata.
        
        Returns:
            Tuple of (values, metadata_list)
        """
        values = []
        metadata_list = []
        
        for name, source in self.sources.items():
            source_values = source.get_peer_values()
            values.extend(source_values)
            
            # Duplicate metadata for each value from this source
            meta = self.metadata[name]
            metadata_list.extend([meta] * len(source_values))
        
        return values, metadata_list
    
    def get_weighted_peers(self) -> List[float]:
        """
        Get peer values weighted by their confidence scores.
        
        Returns:
            List of confidence-weighted peer values
        """
        weighted = []
        for name, source in self.sources.items():
            values = source.get_peer_values()
            confidence = self.metadata[name].confidence
            weighted.extend([v * confidence for v in values])
        return weighted


# ===========================
# Convenience Functions
# ===========================

def create_multi_model_adapter(
    model_outputs: Dict[str, float],
    confidences: Optional[Dict[str, float]] = None
) -> MetadataPeerAdapter:
    """
    Convenience function to create an adapter from multiple model outputs.
    
    Args:
        model_outputs: Dictionary mapping model names to output values
        confidences: Optional dictionary mapping model names to confidence scores
        
    Returns:
        MetadataPeerAdapter configured for multi-model consensus
        
    Example:
        >>> outputs = {"gpt4": 10.5, "claude": 10.3, "llama": 10.6}
        >>> confidences = {"gpt4": 0.95, "claude": 0.92, "llama": 0.88}
        >>> adapter = create_multi_model_adapter(outputs, confidences)
    """
    adapter = MetadataPeerAdapter()
    
    for name, value in model_outputs.items():
        confidence = confidences.get(name, 1.0) if confidences else 1.0
        source = StaticPeerAdapter([value])
        adapter.add_peer_source(
            name=name,
            source=source,
            confidence=confidence,
            source_type="llm"
        )
    
    return adapter


# Convenience exports
__all__ = [
    'PeerAdapter',
    'StaticPeerAdapter',
    'RollingPeerAdapter',
    'CallbackPeerAdapter',
    'MultiSourcePeerAdapter',
    'WeightedPeerAdapter',
    'FilteredPeerAdapter',
    'MetadataPeerAdapter',
    'PeerMetadata',
    'create_multi_model_adapter',
]


# Demo usage
if __name__ == "__main__":
    print("=== AILEE Peer Adapters Demo ===\n")
    
    # 1. Static peers
    print("1. Static Peer Adapter")
    static = StaticPeerAdapter([10.2, 10.5, 10.3])
    print(f"   Peers: {static.get_peer_values()}\n")
    
    # 2. Rolling history
    print("2. Rolling Peer Adapter")
    history = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]
    rolling = RollingPeerAdapter(history, window=3)
    print(f"   Last 3 values: {rolling.get_peer_values()}\n")
    
    # 3. Multi-source aggregation
    print("3. Multi-Source Adapter")
    source1 = StaticPeerAdapter([10.2, 10.3])
    source2 = StaticPeerAdapter([10.4, 10.5])
    multi = MultiSourcePeerAdapter([source1, source2])
    print(f"   Aggregated: {multi.get_peer_values()}\n")
    
    # 4. Filtered peers
    print("4. Filtered Adapter (remove outliers)")
    noisy_source = StaticPeerAdapter([10.0, 100.0, 10.5, -50.0, 10.3])
    filtered = FilteredPeerAdapter(noisy_source, min_value=0.0, max_value=50.0)
    print(f"   Filtered: {filtered.get_peer_values()}\n")
    
    # 5. Multi-model with metadata
    print("5. Multi-Model Adapter")
    model_outputs = {
        "gpt4": 10.5,
        "claude": 10.3,
        "llama": 10.6
    }
    confidences = {
        "gpt4": 0.95,
        "claude": 0.92,
        "llama": 0.88
    }
    mm_adapter = create_multi_model_adapter(model_outputs, confidences)
    values, metadata = mm_adapter.get_peer_values_with_metadata()
    print(f"   Values: {values}")
    print(f"   Sources: {[m.name for m in metadata]}")
    print(f"   Confidences: {[m.confidence for m in metadata]}")
