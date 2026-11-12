"""Opt-in offline execution for CI; ordinary local test behavior is unchanged."""

import socket

import pytest


def pytest_addoption(parser):
    parser.addoption("--offline", action="store_true", help="Block test TCP connections and isolate credentials.")


def pytest_configure(config):
    if not config.getoption("--offline"):
        return

    guard = pytest.MonkeyPatch()
    config._offline_guard = guard
    # Apply before collection: some application modules initialize tracing at import.
    guard.setenv("PYTHON_DOTENV_DISABLED", "1")
    for name in (
        "OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
        "APP_ACCESS_PASSWORD", "DIAGNOSTIC_PASSWORD", "ADMIN_PASSWORD",
    ):
        guard.setenv(name, "")
        guard.delenv(f"{name}_FILE", raising=False)
    guard.setenv("LANGSMITH_TRACING", "false")
    guard.setenv("LANGCHAIN_TRACING_V2", "false")

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError("Network connections are disabled by pytest --offline.")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError("Network connections are disabled by pytest --offline.")
        return original_connect_ex(sock, address)

    guard.setattr(socket.socket, "connect", connect)
    guard.setattr(socket.socket, "connect_ex", connect_ex)

    import streamlit

    original_get_option = streamlit.config.get_option

    def get_option(key):
        if key == "secrets.files":
            return []
        return original_get_option(key)

    # Tests can still supply synthetic Secrets stores or temporary TOML paths.
    guard.setattr(streamlit.config, "get_option", get_option)


def pytest_unconfigure(config):
    guard = getattr(config, "_offline_guard", None)
    if guard is not None:
        guard.undo()
