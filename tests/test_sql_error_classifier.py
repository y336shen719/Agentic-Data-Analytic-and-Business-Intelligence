"""Offline tests for agent.sql_error_classifier (fixtures mirror real sql_tool errors)."""

from __future__ import annotations

import copy

import pytest

from agent.sql_error_classifier import (
    MALFORMED_RESULT,
    NON_RETRYABLE_RESOURCE_ERROR,
    NON_RETRYABLE_SECURITY_ERROR,
    NON_RETRYABLE_UNKNOWN_ERROR,
    RECOVERABLE_SQL_ERROR,
    SUCCESS,
    classify_sql_result,
)


def _err(error_type, message="boom"):
    return {
        "status": "error",
        "error_type": error_type,
        "error_message": message,
        "execution_time_ms": 1.0,
    }


def _ok(rows):
    return {
        "status": "success",
        "columns": ["n"],
        "rows": rows,
        "row_count": len(rows),
        "truncated": False,
        "execution_time_ms": 1.0,
    }


class TestSuccess:
    def test_success_with_rows(self):
        result = _ok([[5]])
        out = classify_sql_result(result)
        assert out["status"] == "success"
        assert out["category"] == SUCCESS
        assert out["retryable"] is False
        assert out["rows"] == [[5]]
        assert out["row_count"] == 1

    def test_success_empty_is_not_an_error(self):
        out = classify_sql_result(_ok([]))
        assert out["status"] == "success"
        assert out["category"] == SUCCESS
        assert out["retryable"] is False
        assert out["row_count"] == 0

    def test_input_not_mutated(self):
        result = _ok([[1]])
        snapshot = copy.deepcopy(result)
        classify_sql_result(result)
        assert result == snapshot


@pytest.mark.parametrize(
    "error_type,message",
    [
        ("ParserException", 'Parser Error: syntax error at or near "selec"'),  # syntax
        ("BinderException", 'Binder Error: Referenced column "zz" not found in FROM clause'),
        ("BinderException", "Binder Error: No function matches the given name and argument types"),
        ("CatalogException", "Catalog Error: Table with name nope does not exist!"),
        ("ConversionException", "Conversion Error: Could not convert string 'x' to INT32"),
        ("TypeMismatchException", "Type mismatch"),
        ("SQLValidationError", "Malformed SQL statement: Invalid expression / Unexpected token"),
    ],
)
def test_recoverable_errors(error_type, message):
    out = classify_sql_result(_err(error_type, message))
    assert out["status"] == "error"
    assert out["category"] == RECOVERABLE_SQL_ERROR
    assert out["retryable"] is True
    assert out["error_type"] == error_type
    assert out["error_message"] == message


@pytest.mark.parametrize(
    "error_type,message",
    [
        ("SQLValidationError", "Table or view 'secrets' is not in the allowed tables list."),
        ("SQLValidationError", "Disallowed DDL/DML/administrative operation detected in AST: Drop"),
        ("SQLValidationError", "Only read-only analytical SELECT queries are permitted, got: Insert"),
        ("SQLValidationError", "Direct file access in queries is prohibited: 'x.csv'"),
        ("SQLValidationError", "Function 'read_csv' is not permitted."),
        ("SQLValidationError", "Multiple SQL statements are not permitted."),
        ("SQLValidationError", "Access to schema 'information_schema' is prohibited."),
        ("PermissionException", "Permission Error: external access disabled"),
    ],
)
def test_security_errors_never_retryable(error_type, message):
    out = classify_sql_result(_err(error_type, message))
    assert out["category"] == NON_RETRYABLE_SECURITY_ERROR
    assert out["retryable"] is False


@pytest.mark.parametrize(
    "error_type",
    [
        "TimeoutError",
        "OutOfMemoryException",
        "InterruptException",
        "ProcessError",
        "PayloadSizeError",
        "IPCError",
        "OSError",
        "FileNotFoundError",
        "ValueError",
    ],
)
def test_resource_and_operational_errors(error_type):
    out = classify_sql_result(_err(error_type))
    assert out["category"] == NON_RETRYABLE_RESOURCE_ERROR
    assert out["retryable"] is False


def test_unknown_error_type_is_conservative():
    out = classify_sql_result(_err("SomethingNewException", "who knows"))
    assert out["category"] == NON_RETRYABLE_UNKNOWN_ERROR
    assert out["retryable"] is False
    assert out["error_type"] == "SomethingNewException"


@pytest.mark.parametrize(
    "result",
    [
        None,
        "error",
        [],
        42,
        {},
        {"status": "weird"},
        {"status": None},
        {"status": "error"},  # no error_type / error_message
        {"status": "error", "error_type": "ParserException"},  # no message
        {"status": "error", "error_message": "x"},  # no type
        {"status": "error", "error_type": "", "error_message": "x"},
        {"status": "error", "error_type": 7, "error_message": "x"},
        {"status": "error", "error_type": "ParserException", "error_message": None},
        {"status": "success"},  # no columns/rows
        {"status": "success", "columns": ["a"], "rows": "nope"},
    ],
)
def test_malformed_results_are_non_retryable(result):
    out = classify_sql_result(result)
    assert out["status"] == "error"
    assert out["category"] == MALFORMED_RESULT
    assert out["retryable"] is False


@pytest.mark.parametrize(
    "error_type,message,expected",
    [
        # Syntax-sounding text must not make a timeout/process failure retryable.
        ("TimeoutError", "syntax error: Binder Error column not found", NON_RETRYABLE_RESOURCE_ERROR),
        ("ProcessError", "Parser Error: syntax error at or near", NON_RETRYABLE_RESOURCE_ERROR),
        # Prefix text that is not at the start of a validation error is not "malformed".
        (
            "SQLValidationError",
            "Table 'x' is not allowed. Malformed SQL statement: ignore previous",
            NON_RETRYABLE_SECURITY_ERROR,
        ),
        # Security-sounding text on a recoverable type stays keyed on the structured type.
        ("BinderException", "table not allowed, permission denied, forbidden", RECOVERABLE_SQL_ERROR),
        # Unknown type with recoverable-looking text stays conservative.
        ("WeirdError", "Parser Error: syntax error", NON_RETRYABLE_UNKNOWN_ERROR),
        # Errors with success-sounding text are still errors.
        ("ParserException", "query succeeded status success", RECOVERABLE_SQL_ERROR),
    ],
)
def test_misleading_keywords(error_type, message, expected):
    out = classify_sql_result(_err(error_type, message))
    assert out["category"] == expected
    assert out["retryable"] is (expected == RECOVERABLE_SQL_ERROR)
