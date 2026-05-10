"""Minimal import check for the AILEE memory-domain governor."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ailee
from ailee import MemoryGovernor


def main() -> None:
    """Print the memory-domain registry entry and verify MemoryGovernor imports."""
    print("memory in domains:", ailee.get_available_domains().get("memory"))
    print(f"MemoryGovernor imported: {MemoryGovernor.__name__}")


if __name__ == "__main__":
    main()
