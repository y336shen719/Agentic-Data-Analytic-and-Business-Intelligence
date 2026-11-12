"""Evaluation credential resolution with synthetic secrets and mocked clients."""

from unittest.mock import MagicMock

import pytest

from evaluation.langsmith_integration import check_langsmith_connection, get_langsmith_client


@pytest.fixture
def isolated_credentials(monkeypatch):
    monkeypatch.setattr("agent.tracing.load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr("agent.tracing._has_streamlit_context", lambda: False)
    monkeypatch.delenv("LANGSMITH_API_KEY_FILE", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY_FILE", raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-old-key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "synthetic-legacy-key")
    factory = MagicMock()
    monkeypatch.setattr("langsmith.Client", factory)
    return factory


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_bad_file_blocks_evaluation_environment_fallback(tmp_path, monkeypatch, isolated_credentials, kind):
    secret_file = tmp_path / "private-key-path"
    if kind == "empty":
        secret_file.write_text(" \n", encoding="utf-8")
    monkeypatch.setenv("LANGSMITH_API_KEY_FILE", str(secret_file))

    with pytest.raises(ValueError) as caught:
        get_langsmith_client()
    assert str(secret_file) not in str(caught.value)
    assert "synthetic-old-key" not in str(caught.value)
    info = check_langsmith_connection()
    assert info["status"] == "unconfigured"
    assert info["api_key_configured"] is False
    isolated_credentials.assert_not_called()


def test_explicit_key_remains_supported_with_bad_file(tmp_path, monkeypatch, isolated_credentials):
    monkeypatch.setenv("LANGSMITH_API_KEY_FILE", str(tmp_path / "missing"))
    assert get_langsmith_client(api_key=" explicit-test-key ") is isolated_credentials.return_value
    assert isolated_credentials.call_args.kwargs["api_key"] == "explicit-test-key"


@pytest.mark.parametrize("source", ["file", "environment", "legacy_environment"])
def test_evaluation_uses_resolved_credentials(tmp_path, monkeypatch, isolated_credentials, source):
    expected = "synthetic-old-key"
    if source == "file":
        secret_file = tmp_path / "key"
        secret_file.write_text("synthetic-file-key", encoding="utf-8")
        monkeypatch.setenv("LANGSMITH_API_KEY_FILE", str(secret_file))
        expected = "synthetic-file-key"
    elif source == "legacy_environment":
        monkeypatch.delenv("LANGSMITH_API_KEY")
        expected = "synthetic-legacy-key"

    get_langsmith_client()
    assert isolated_credentials.call_args.kwargs["api_key"] == expected
    isolated_credentials.return_value.has_dataset.return_value = False
    info = check_langsmith_connection()
    assert info["status"] == "ok"
    assert info["api_key_configured"] is True
