"""Release gate proving that active cross-runtime metadata stays aligned."""

import json
import re
from pathlib import Path

import ailee


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "10.0.2"


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_active_release_metadata_is_consistent():
    assert ailee.__version__ == EXPECTED_VERSION
    assert f'version="{EXPECTED_VERSION}"' in _text("setup.py")
    assert f'version = "{EXPECTED_VERSION}"' in _text("Cargo.toml")
    assert re.search(
        rf'\[\[package\]\]\nname = "ailee_trust_core"\nversion = "{re.escape(EXPECTED_VERSION)}"',
        _text("Cargo.lock"),
    )
    assert f"VERSION {EXPECTED_VERSION}" in _text("CMakeLists.txt")
    assert json.loads(_text("packages/ailee-ts/package.json"))["version"] == EXPECTED_VERSION
    npm_lock = json.loads(_text("packages/ailee-ts/package-lock.json"))
    assert npm_lock["version"] == EXPECTED_VERSION
    assert npm_lock["packages"][""]["version"] == EXPECTED_VERSION
    assert f'VERSION = "{EXPECTED_VERSION}"' in _text("packages/ailee-ts/src/index.ts")
    assert _text("CITATION.cff").count(f'version: "{EXPECTED_VERSION}"') == 2
    assert f"Current: v{EXPECTED_VERSION}" in _text("README.md")
    assert re.search(
        rf"assert ailee\.__version__ == ['\"]{re.escape(EXPECTED_VERSION)}['\"]",
        _text(".github/workflows/ci.yml"),
    )
