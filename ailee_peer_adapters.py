"""
Peer adapters for distributed or multi-model systems.
"""

from typing import List, Protocol


class PeerAdapter(Protocol):
    def get_peer_values(self) -> List[float]:
        ...


class StaticPeerAdapter:
    def __init__(self, peers: List[float]):
        self.peers = peers

    def get_peer_values(self) -> List[float]:
        return self.peers


class RollingPeerAdapter:
    def __init__(self, history, window=10):
        self.history = history
        self.window = window

    def get_peer_values(self) -> List[float]:
        return self.history[-self.window:]
