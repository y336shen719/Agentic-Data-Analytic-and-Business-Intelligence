"""Deterministic classification of ``tools.sql_tool.execute_sql`` results.

Classification keys on the structured ``error_type`` field emitted by the SQL tool.
The only message check is an anchored prefix for ``SQLValidationError``, because that
single exception type covers both syntax problems ("Malformed SQL statement: ...")
and security/policy violations. Anything unrecognised is NOT retryable.
"""

from __future__ import annotations

from typing import Any, Dict

SUCCESS = "success"
RECOVERABLE_SQL_ERROR = "recoverable_sql_error"
NON_RETRYABLE_SECURITY_ERROR = "non_retryable_security_error"
NON_RETRYABLE_RESOURCE_ERROR = "non_retryable_resource_error"
NON_RETRYABLE_UNKNOWN_ERROR = "non_retryable_unknown_error"
MALFORMED_RESULT = "malformed_result"

# DuckDB exception class names (sent by the worker as type(exc).__name__) that
# indicate a fixable SQL construction problem.
RECOVERABLE_ERROR_TYPES = frozenset(
    {
        "ParserException",  # syntax error
        "SyntaxException",
        "BinderException",  # unknown column, bad function args, GROUP BY misuse
        "CatalogException",  # unknown table/function name
        "ConversionException",  # failed cast
        "TypeMismatchException",
        "InvalidTypeException",
    }
)

# Security / policy: never retry, even though the LLM could rewrite the SQL.
SECURITY_ERROR_TYPES = frozenset({"SQLValidationError", "PermissionException"})

# Exact prefix produced by tools.sql_tool.validate_sql when sqlglot cannot parse the query.
_MALFORMED_SQL_PREFIX = "Malformed SQL statement:"

# Operational failures: timeouts, memory, worker/IPC failures, bad config/paths.
RESOURCE_ERROR_TYPES = frozenset(
    {
        "TimeoutError",
        "OutOfMemoryException",
        "InterruptException",
        "ProcessError",
        "PayloadSizeError",
        "IPCError",
        "OSError",
        "IOException",
        "FileNotFoundError",
        "ValueError",
    }
)


def _malformed(reason: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "category": MALFORMED_RESULT,
        "retryable": False,
        "error_type": "MalformedResult",
        "error_message": reason,
    }


def classify_sql_result(result: Any) -> Dict[str, Any]:
    """Classify an ``execute_sql`` result.

    Success (including zero rows) returns a shallow copy of the result plus
    ``category="success"`` and ``retryable=False``. Errors return
    ``{status, category, retryable, error_type, error_message}``. The input is never mutated.
    """
    if not isinstance(result, dict):
        return _malformed(f"SQL tool result must be a dict, got {type(result).__name__}.")

    status = result.get("status")

    if status == "success":
        if not isinstance(result.get("columns"), list) or not isinstance(result.get("rows"), list):
            return _malformed("Successful SQL result is missing 'columns' or 'rows' lists.")
        return {**result, "status": "success", "category": SUCCESS, "retryable": False}

    if status != "error":
        return _malformed(f"Unexpected or missing result status: {status!r}.")

    error_type = result.get("error_type")
    error_message = result.get("error_message")
    if not isinstance(error_type, str) or not error_type.strip():
        return _malformed("Error result is missing a valid 'error_type'.")
    if not isinstance(error_message, str):
        return _malformed("Error result is missing a valid 'error_message'.")

    if error_type == "SQLValidationError" and error_message.startswith(_MALFORMED_SQL_PREFIX):
        category = RECOVERABLE_SQL_ERROR
    elif error_type in SECURITY_ERROR_TYPES:
        category = NON_RETRYABLE_SECURITY_ERROR
    elif error_type in RECOVERABLE_ERROR_TYPES:
        category = RECOVERABLE_SQL_ERROR
    elif error_type in RESOURCE_ERROR_TYPES:
        category = NON_RETRYABLE_RESOURCE_ERROR
    else:
        category = NON_RETRYABLE_UNKNOWN_ERROR

    return {
        "status": "error",
        "category": category,
        "retryable": category == RECOVERABLE_SQL_ERROR,
        "error_type": error_type,
        "error_message": error_message,
    }
