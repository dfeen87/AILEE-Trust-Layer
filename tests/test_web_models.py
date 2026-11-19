import importlib.util
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock


def test_provider_logs_omit_credentials_even_on_sdk_errors(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "duckduckgo_search", SimpleNamespace(DDGS=Mock()))
    spec = importlib.util.find_spec("ailee.web.models")
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    dummy = "TEST_SYNTHETIC_VALUE_ONLY"
    provider = Mock(side_effect=RuntimeError(f"synthetic SDK error: {dummy}"))

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
    assert responses[0][0] == "mock_fallback"
    for name in ("OpenAI", "Anthropic", "Gemini"):
        assert f"Calling {name}" in caplog.text
        assert f"{name} generation failed (RuntimeError)" in caplog.text
    assert dummy not in caplog.text
    assert dummy[:4] not in caplog.text
    assert dummy[-4:] not in caplog.text
