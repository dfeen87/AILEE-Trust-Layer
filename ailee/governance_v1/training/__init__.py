# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Training Package
"""

from .dispatcher import (
    DispatcherRegistry,
    apply_training_signal,
    register_training_callback,
)

__all__ = [
    "DispatcherRegistry",
    "apply_training_signal",
    "register_training_callback",
]
