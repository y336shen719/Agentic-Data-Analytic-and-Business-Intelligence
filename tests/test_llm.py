"""Unit tests for LLM config and factory. No real API key or network calls."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import llm as llm_module
from agent import config as config_module
from agent.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    ConfigError,
    LLMConfig,
    load_llm_config,
)


@pytest.mark.unit
def test_defaults_when_env_empty() -> None:
    cfg = load_llm_config({})
    assert cfg.model == DEFAULT_OPENAI_MODEL
    assert cfg.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert cfg.max_retries == DEFAULT_MAX_RETRIES
    assert cfg.api_key is None


@pytest.mark.unit
def test_env_overrides() -> None:
    cfg = load_llm_config({
        "OPENAI_API_KEY": " sk-test ",
        "OPENAI_MODEL": "custom-model",
        "OPENAI_TIMEOUT_SECONDS": "12.5",
        "OPENAI_MAX_RETRIES": "0",
    })
    assert cfg.model == "custom-model"
    assert cfg.timeout_seconds == 12.5
    assert cfg.max_retries == 0
    assert cfg.api_key == "sk-test"


@pytest.mark.unit
def test_blank_model_falls_back_to_default() -> None:
    assert load_llm_config({"OPENAI_MODEL": "  "}).model == DEFAULT_OPENAI_MODEL


@pytest.mark.unit
@pytest.mark.parametrize("with_env_key", [False, True])
def test_config_reads_api_key_file(tmp_path, with_env_key):
    secret_file = tmp_path / "openai-key"
    secret_file.write_text(" file-test-key\n", encoding="utf-8")
    env = {"OPENAI_API_KEY_FILE": str(secret_file)}
    if with_env_key:
        env["OPENAI_API_KEY"] = "environment-test-key"
    assert load_llm_config(env).api_key == "file-test-key"


@pytest.mark.unit
def test_config_bad_key_file_does_not_fall_back(tmp_path):
    with pytest.raises(ConfigError, match="OPENAI_API_KEY_FILE") as caught:
        load_llm_config({
            "OPENAI_API_KEY_FILE": str(tmp_path / "missing"),
            "OPENAI_API_KEY": "environment-test-key",
        })
    assert str(tmp_path) not in str(caught.value)
    assert "environment-test-key" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.unit
def test_dotenv_loading_remains_enabled(tmp_path, monkeypatch):
    # This test explicitly loads only the synthetic temporary file, even in CI.
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("OPENAI_API_KEY=dotenv-test-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    real_loader = config_module.load_dotenv
    monkeypatch.setattr(config_module, "load_dotenv", lambda **kwargs: real_loader(dotenv_file, **kwargs))
    assert load_llm_config().api_key == "dotenv-test-key"
    monkeypatch.setenv("OPENAI_API_KEY", "existing-test-key")
    assert load_llm_config().api_key == "existing-test-key"


@pytest.mark.unit
@pytest.mark.parametrize(
    "env",
    [
        {"OPENAI_TIMEOUT_SECONDS": "abc"},
        {"OPENAI_TIMEOUT_SECONDS": "0"},
        {"OPENAI_TIMEOUT_SECONDS": "-5"},
        {"OPENAI_TIMEOUT_SECONDS": "nan"},
        {"OPENAI_TIMEOUT_SECONDS": "9999"},
        {"OPENAI_MAX_RETRIES": "x"},
        {"OPENAI_MAX_RETRIES": "-1"},
        {"OPENAI_MAX_RETRIES": "99"},
    ],
)
def test_invalid_limits_rejected(env: dict) -> None:
    with pytest.raises(ConfigError):
        load_llm_config(env)


@pytest.mark.unit
def test_repr_does_not_leak_api_key() -> None:
    cfg = LLMConfig(api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(cfg)


@pytest.mark.unit
def test_create_llm_missing_key_raises() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        llm_module.create_llm(LLMConfig(api_key=None))


@pytest.mark.unit
def test_create_llm_passes_config_to_chat_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls = MagicMock(name="ChatOpenAI")
    monkeypatch.setattr(llm_module, "ChatOpenAI", fake_cls)

    cfg = LLMConfig(model="m", timeout_seconds=7.0, max_retries=1, api_key="sk-test")
    result = llm_module.create_llm(cfg)

    assert result is fake_cls.return_value
    kwargs = fake_cls.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["timeout"] == 7.0
    assert kwargs["max_retries"] == 1
    assert kwargs["api_key"].get_secret_value() == "sk-test"
    assert "temperature" not in kwargs  # not set: may be unsupported by the model


@pytest.mark.unit
def test_create_llm_builds_real_client_offline() -> None:
    """Constructing ChatOpenAI makes no network call; verifies kwargs are valid."""
    client = llm_module.create_llm(LLMConfig(model="m", api_key="sk-test", max_retries=1))
    assert client.model_name == "m"
    assert client.max_retries == 1
