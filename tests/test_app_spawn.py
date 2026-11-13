"""Regression coverage for Streamlit's main module being replayed by spawn."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"


def _run_isolated(code, *args):
    # Never use developer credentials or the repository .env in these processes.
    env = os.environ.copy()
    for name in (
        "OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
        "APP_ACCESS_PASSWORD", "DIAGNOSTIC_PASSWORD", "ADMIN_PASSWORD",
    ):
        env.pop(name, None)
        env.pop(f"{name}_FILE", None)
    env.update(
        PYTHON_DOTENV_DISABLED="1",
        PYTHONDONTWRITEBYTECODE="1",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        PYTHONPATH=str(PROJECT_ROOT),
    )
    return subprocess.run(
        [sys.executable, "-B", "-c", code, *map(str, args)],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
        timeout=45, check=True,
    )


@pytest.mark.parametrize("run_name", ["app", "__mp_main__"])
def test_app_loading_does_not_import_ui_or_read_configuration(run_name):
    result = _run_isolated('''
import importlib.abc
import runpy
import sys

class BlockRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "streamlit", "agent", "views", "tools", "data",
            "openai", "langsmith", "langchain", "langgraph",
        }:
            raise AssertionError("Unexpected runtime import: " + fullname)

sys.meta_path.insert(0, BlockRuntimeImports())
namespace = runpy.run_path(sys.argv[1], run_name=sys.argv[2])
assert callable(namespace["main"])
assert callable(namespace["get_app_access_password"])
assert callable(namespace["check_app_access"])
print("PASS")
''', APP_PATH, run_name)
    assert result.stdout.strip() == "PASS"


def test_spawn_replays_app_without_loading_ui_and_sql_still_executes(tmp_path):
    # An importable target is required: spawn must not depend on the launcher's
    # original __main__, because Streamlit replaces it with app.py at runtime.
    (tmp_path / "spawn_probe.py").write_text('''
def probe(connection):
    import sys
    from tools.sql_tool import get_db_connection
    connection.send(sorted(
        name for name in ("streamlit", "openai", "langsmith", "agent.ui_session")
        if name in sys.modules
    ))
    connection.close()
''', encoding="utf-8")
    result = _run_isolated('''
import json
import multiprocessing
import sys
from pathlib import Path
import duckdb

sys.path.insert(0, sys.argv[2])
from spawn_probe import probe
from tools.sql_tool import execute_sql

db = Path(sys.argv[2]) / "test.duckdb"
with duckdb.connect(str(db)) as connection:
    connection.execute("CREATE TABLE orders AS SELECT 1 AS order_id")

# Match Streamlit's __main__ metadata rather than a python -c invocation.
sys.modules["__main__"].__file__ = sys.argv[1]
sys.modules["__main__"].__spec__ = None
context = multiprocessing.get_context("spawn")
parent, child = context.Pipe(duplex=False)
worker = context.Process(target=probe, args=(child,))
try:
    worker.start()
    child.close()
    assert parent.poll(20), "Spawn replay did not reach the worker target"
    assert parent.recv() == [], "SQL worker loaded UI/LLM dependencies"
    worker.join(5)
    assert worker.exitcode == 0
finally:
    if worker.is_alive():
        worker.terminate()
        worker.join(5)
    parent.close()
    child.close()

result = execute_sql("SELECT count(*) FROM orders", db_path=db)
assert result["status"] == "success", result
assert result["rows"] == [[1]], result
assert not multiprocessing.active_children()
print(json.dumps({"status": result["status"]}))
''', APP_PATH, tmp_path)
    assert json.loads(result.stdout.strip()) == {"status": "success"}


def test_legacy_view_exports_are_resolved_lazily(monkeypatch):
    import importlib
    import runpy
    from types import SimpleNamespace

    namespace = runpy.run_path(str(APP_PATH), run_name="app")
    calls = []
    sentinel = object()

    def import_view(name):
        calls.append(name)
        return SimpleNamespace(**{key: sentinel for key in namespace["__all__"]})

    monkeypatch.setattr(importlib, "import_module", import_view)
    for name in namespace["__all__"]:
        assert namespace["__getattr__"](name) is sentinel
        assert calls[-1] == (
            "views.database_explorer"
            if name == "render_database_explorer_page" else "views.analytics"
        )
    with pytest.raises(AttributeError):
        namespace["__getattr__"]("unknown_export")
