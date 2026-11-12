"""Credential boundary tests using synthetic files and mocked network clients."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent import tracing


@pytest.fixture
def isolated_secrets(monkeypatch):
    import streamlit as st

    monkeypatch.setattr(tracing, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(tracing, "_has_streamlit_context", lambda: True)
    monkeypatch.setattr(tracing, "_CACHED_CLIENT", None)
    monkeypatch.setattr(st, "secrets", {})
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "DIAGNOSTIC_PASSWORD", "ADMIN_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"{name}_FILE", raising=False)
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"):
        monkeypatch.setenv(name, "")
    return st


def install_real_streamlit_secrets(tmp_path, monkeypatch, st, content):
    """Exercise Streamlit's TOML parsing and environment promotion with a fake file."""
    from streamlit.runtime import secrets as secrets_module

    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(content, encoding="utf-8")
    store = secrets_module.Secrets()
    original_get_option = secrets_module.config.get_option
    monkeypatch.setattr(
        secrets_module.config, "get_option",
        lambda key: [str(secrets_file)] if key == "secrets.files" else original_get_option(key),
    )
    monkeypatch.setattr(secrets_module.Secrets, "_maybe_install_file_watchers", lambda self: None)
    monkeypatch.setattr(st, "secrets", store)
    return store


@pytest.mark.parametrize("source", ["file", "streamlit_nested"])
def test_real_tracer_uses_explicit_client_without_exporting_key(tmp_path, monkeypatch, isolated_secrets, source):
    from langchain_core.runnables import RunnableLambda

    key = "synthetic-tracing-secret"
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    if source == "file":
        secret_file = tmp_path / "key"
        secret_file.write_text(key, encoding="utf-8")
        monkeypatch.setenv("LANGSMITH_API_KEY_FILE", str(secret_file))
    else:
        install_real_streamlit_secrets(
            tmp_path, monkeypatch, isolated_secrets,
            f'[credentials]\nLANGSMITH_API_KEY = "{key}"\n',
        )
    client = MagicMock()
    client.api_key = key
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("langsmith.Client", factory)

    tracing.sync_secrets_to_environment()
    assert tracing.setup_langsmith_environment() is True
    config = tracing.create_runnable_tracing_config()
    tracer = config["callbacks"][0]
    assert tracer.client is client
    assert RunnableLambda(lambda value: value + " done").invoke("input", config=config) == "input done"
    tracer.wait_for_futures()
    assert client.create_run.called
    assert factory.call_args.kwargs["api_key"] == key
    assert "LANGSMITH_API_KEY" not in os.environ
    assert "LANGCHAIN_API_KEY" not in os.environ


def test_legacy_top_level_streamlit_key_is_still_supported(tmp_path, monkeypatch, isolated_secrets):
    install_real_streamlit_secrets(
        tmp_path, monkeypatch, isolated_secrets,
        'LANGSMITH_API_KEY = "synthetic-legacy-secret"\n',
    )
    assert tracing.load_langsmith_config().api_key == "synthetic-legacy-secret"
    # This export is performed by Streamlit itself, not by our reader.
    assert os.environ["LANGSMITH_API_KEY"] == "synthetic-legacy-secret"


@pytest.mark.parametrize("name", ["DIAGNOSTIC_PASSWORD", "ADMIN_PASSWORD"])
def test_nested_streamlit_diagnostic_password_is_not_exported(tmp_path, monkeypatch, isolated_secrets, name):
    install_real_streamlit_secrets(
        tmp_path, monkeypatch, isolated_secrets,
        f'[credentials]\n{name} = "synthetic-nested-password"\n',
    )
    tracing.sync_secrets_to_environment()
    assert tracing.has_diagnostic_password_configured()
    assert tracing.verify_diagnostic_access("synthetic-nested-password")
    assert "DIAGNOSTIC_PASSWORD" not in os.environ
    assert "ADMIN_PASSWORD" not in os.environ


def test_setup_preserves_existing_environment_keys(isolated_secrets, monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "existing-canonical")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "existing-legacy")
    tracing.setup_langsmith_environment(tracing.LangSmithConfig(enabled=True, api_key="file-key"))
    assert os.environ["LANGSMITH_API_KEY"] == "existing-canonical"
    assert os.environ["LANGCHAIN_API_KEY"] == "existing-legacy"


@pytest.mark.parametrize("name", ["DIAGNOSTIC_PASSWORD", "ADMIN_PASSWORD"])
def test_diagnostic_file_precedes_environment_and_streamlit(tmp_path, monkeypatch, isolated_secrets, name):
    secret_file = tmp_path / "password"
    secret_file.write_text(" synthetic-diagnostic-password\n", encoding="utf-8")
    monkeypatch.setenv(f"{name}_FILE", str(secret_file))
    monkeypatch.setenv("DIAGNOSTIC_PASSWORD", "old-password")
    isolated_secrets.secrets = {"DIAGNOSTIC_PASSWORD": "streamlit-password"}
    assert tracing.has_diagnostic_password_configured()
    assert tracing.verify_diagnostic_access("synthetic-diagnostic-password")
    assert not tracing.verify_diagnostic_access("old-password")
    assert os.environ["DIAGNOSTIC_PASSWORD"] == "old-password"


@pytest.mark.parametrize("kind", ["missing", "empty", "permission"])
def test_bad_diagnostic_file_disables_without_fallback(tmp_path, monkeypatch, isolated_secrets, caplog, kind):
    secret_file = tmp_path / "private-password-path"
    if kind == "empty":
        secret_file.write_text(" \n", encoding="utf-8")
    elif kind == "permission":
        def deny_read(self, *args, **kwargs):
            raise PermissionError("synthetic-private-content")

        monkeypatch.setattr(Path, "read_text", deny_read)
    monkeypatch.setenv("DIAGNOSTIC_PASSWORD_FILE", str(secret_file))
    monkeypatch.setenv("DIAGNOSTIC_PASSWORD", "old-password")
    monkeypatch.setenv("ADMIN_PASSWORD", "legacy-password")
    isolated_secrets.secrets = {"DIAGNOSTIC_PASSWORD": "streamlit-password"}
    assert not tracing.has_diagnostic_password_configured()
    assert not tracing.verify_diagnostic_access("old-password")
    assert not tracing.verify_diagnostic_access("legacy-password")
    assert str(secret_file) not in caplog.text
    assert "synthetic-private-content" not in caplog.text


def test_streamlit_diagnostic_password_remains_supported(isolated_secrets):
    isolated_secrets.secrets = {"DIAGNOSTIC_PASSWORD": "streamlit-password"}
    assert tracing.has_diagnostic_password_configured()
    assert tracing.verify_diagnostic_access("streamlit-password")
