# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Micro-Training Dispatcher (Python)
Applies training signals atomically to specific model targets, logs updates, and triggers callbacks.
"""

import json
import logging
import os
import threading
from typing import Callable, List
from ..schemas import TrainingSignal

logger = logging.getLogger("ailee.governance_v1.training")


class DispatcherRegistry:
    """Registry maintaining training signal logs and callback hooks."""

    def __init__(self, log_file_path: str = "training_signals.jsonl"):
        self.log_file_path = log_file_path
        self.callbacks: List[Callable[[TrainingSignal], None]] = []
        self._lock = threading.Lock()
        self._history: List[TrainingSignal] = []
        self._seen_ids: set = set()

    def register_callback(self, callback: Callable[[TrainingSignal], None]):
        with self._lock:
            self.callbacks.append(callback)

    def dispatch(self, signal: TrainingSignal):
        """
        Atomically applies, logs, and broadcasts training signals.
        Requirements:
          - Every training signal must reference ledger IDs
          - Duplicate signals are rejected/ignored
          - Updates must be atomic and logged
          - Callbacks executed outside lock to prevent deadlock
        """
        if not signal.source_ledger_ids:
            raise ValueError("TrainingSignal must reference at least one source ledger ID.")

        with self._lock:
            if signal.id in self._seen_ids:
                logger.warning(f"Duplicate TrainingSignal '{signal.id}' detected. Skipping.")
                return

            # 1. Record in memory
            self._seen_ids.add(signal.id)
            self._history.append(signal)

            # 2. Append to log file atomically
            record = {
                "id": signal.id,
                "target_model": signal.target_model,
                "features": signal.features,
                "label": signal.label,
                "source_ledger_ids": signal.source_ledger_ids,
            }

            os.makedirs(os.path.dirname(os.path.abspath(self.log_file_path)) or ".", exist_ok=True)
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

            callbacks_to_invoke = list(self.callbacks)

        # 3. Notify callbacks outside lock to prevent deadlocks
        for cb in callbacks_to_invoke:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Error in training signal callback: {e}")

    def get_history(self) -> List[TrainingSignal]:
        with self._lock:
            return list(self._history)


_GLOBAL_DISPATCHER = DispatcherRegistry()


def apply_training_signal(signal: TrainingSignal, dispatcher: DispatcherRegistry = None):
    target_dispatcher = dispatcher or _GLOBAL_DISPATCHER
    target_dispatcher.dispatch(signal)


def register_training_callback(callback: Callable[[TrainingSignal], None], dispatcher: DispatcherRegistry = None):
    target_dispatcher = dispatcher or _GLOBAL_DISPATCHER
    target_dispatcher.register_callback(callback)
