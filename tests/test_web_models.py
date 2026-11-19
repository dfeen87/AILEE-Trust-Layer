import builtins
import importlib.util
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _load_models(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckduckgo_search", SimpleNamespace(DDGS=Mock()))
    spec = importlib.util.find_spec("ailee.web.models")
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    return models


@pytest.mark.parametrize("sdk_error", [False, True], ids=["success", "sdk-error"])
def test_provider_logs_omit_credentials(monkeypatch, caplog, sdk_error):
    models = _load_models(monkeypatch)
    dummy = "TEST_SYNTHETIC_VALUE_ONLY"
    create = Mock(return_value=object())
    if sdk_error:
        create.side_effect = RuntimeError(f"synthetic SDK error: {dummy}")
    endpoint = SimpleNamespace(create=create, generate_content=create)
    provider = Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=endpoint), messages=endpoint, models=endpoint,
    ))

    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(variable, dummy)
    for flag in ("OPENAI_AVAILABLE", "ANTHROPIC_AVAILABLE", "GEMINI_AVAILABLE"):
        monkeypatch.setattr(models, flag, True)
    monkeypatch.setattr(models, "OpenAI", provider, raising=False)
    monkeypatch.setattr(models, "Anthropic", provider, raising=False)
    monkeypatch.setattr(models, "genai", SimpleNamespace(Client=provider), raising=False)

    with caplog.at_level(logging.INFO, logger=models.logger.name):
        responses = models.generate_with_models("Synthetic query", "")

    assert provider.call_count == 3
    assert create.call_count == 3
    assert [response[0] for response in responses] == (
        ["mock_fallback"] if sdk_error else ["openai", "anthropic", "gemini"]
    )
    for name in ("OpenAI", "Anthropic", "Gemini"):
        assert f"Calling {name}" in caplog.text
        assert (f"{name} generation failed (RuntimeError)" in caplog.text) is sdk_error
    assert dummy not in caplog.text
    assert dummy[:4] not in caplog.text
    assert dummy[-4:] not in caplog.text


def test_provider_import_logs_omit_raw_error_text(monkeypatch, caplog):
    synthetic_detail = "TEST_SYNTHETIC_IMPORT_DETAIL"
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name in {"openai", "anthropic", "google.genai"}:
            raise ImportError(synthetic_detail)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    _load_models(monkeypatch)

    assert synthetic_detail not in caplog.text
    for name in ("OpenAI", "Anthropic", "Google GenAI"):
        assert f"{name} library unavailable (ImportError)" in caplog.text
