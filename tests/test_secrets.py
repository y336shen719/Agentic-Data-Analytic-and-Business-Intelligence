"""Offline secret-loading tests using temporary files and synthetic values."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.secrets import SecretReadError, read_secret

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("env", [{}, {"TEST_SECRET": ""}, {"TEST_SECRET": "  "}])
def test_absent_value_returns_none(env):
    assert read_secret("TEST_SECRET", env) is None


def test_environment_value_is_stripped():
    assert read_secret("TEST_SECRET", {"TEST_SECRET": " synthetic-value \n"}) == "synthetic-value"


def test_file_only_configuration_returns_secret(tmp_path):
    secret_file = tmp_path / "secret"
    secret_file.write_text(" file-only-value\n", encoding="utf-8")
    env = {"TEST_SECRET_FILE": str(secret_file)}

    assert read_secret("TEST_SECRET", env) == "file-only-value"
    assert env == {"TEST_SECRET_FILE": str(secret_file)}


def test_file_wins_and_does_not_mutate_environment(tmp_path, monkeypatch):
    secret_file = tmp_path / "secret"
    secret_file.write_text(" file-value\n", encoding="utf-8")
    monkeypatch.setenv("TEST_SECRET_FILE", str(secret_file))
    monkeypatch.setenv("TEST_SECRET", "environment-value")
    import os

    before = dict(os.environ)
    assert read_secret("TEST_SECRET") == "file-value"
    assert dict(os.environ) == before


def test_injected_mapping_is_isolated_from_process_environment(monkeypatch):
    monkeypatch.setenv("TEST_SECRET", "process-value")
    monkeypatch.setenv("TEST_SECRET_FILE", "/not-used")
    assert read_secret("TEST_SECRET", {}) is None


def test_blank_file_setting_falls_back_to_value():
    env = {"TEST_SECRET_FILE": " \n", "TEST_SECRET": "fallback"}
    assert read_secret("TEST_SECRET", env) == "fallback"


@pytest.mark.parametrize("kind", ["missing", "directory", "empty", "invalid_utf8", "permission"])
def test_bad_file_fails_without_fallback_or_sensitive_error(tmp_path, monkeypatch, kind):
    secret_file = tmp_path / "private-filename"
    if kind == "directory":
        secret_file.mkdir()
    elif kind == "empty":
        secret_file.write_text(" \n", encoding="utf-8")
    elif kind == "invalid_utf8":
        secret_file.write_bytes(b"\xffsynthetic-private-content")
    elif kind == "permission":
        def deny_read(self, *args, **kwargs):
            raise PermissionError("synthetic-private-content")

        monkeypatch.setattr(Path, "read_text", deny_read)

    env = {"TEST_SECRET_FILE": str(secret_file), "TEST_SECRET": "synthetic-fallback"}
    before = dict(env)
    with pytest.raises(SecretReadError) as caught:
        read_secret("TEST_SECRET", env)

    message = str(caught.value)
    assert "TEST_SECRET_FILE" in message
    assert str(secret_file) not in message
    assert "synthetic-private-content" not in message
    assert "synthetic-fallback" not in message
    assert caught.value.__context__ is None or caught.value.__suppress_context__
    assert env == before
