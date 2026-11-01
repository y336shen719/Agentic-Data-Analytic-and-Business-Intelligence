"""Offline unit tests for agent.python_error_classifier."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from agent.python_error_classifier import (
    ACTION_CORRECT_ARGUMENTS,
    ACTION_NONE,
    ACTION_RECOVER_DATASET_REFERENCE,
    ACTION_REGENERATE_DATASET,
    CATEGORY_INTERNAL_FAILURE,
    CATEGORY_INVALID_COLUMN,
    CATEGORY_INVALID_DATASET_STRUCTURE,
    CATEGORY_INVALID_OPERATION,
    CATEGORY_INVALID_PARAMETER,
    CATEGORY_MISSING_DATASET,
    CATEGORY_POLICY_VIOLATION,
    CATEGORY_UNKNOWN_ERROR,
    MALFORMED_RESULT,
    RECOVERABLE_PYTHON_CATEGORIES,
    SUCCESS,
    classify_python_result,
)


def _err(error_type: str, message: str = "error details") -> Dict[str, Any]:
    return {
        "status": "error",
        "error_type": error_type,
        "error_message": message,
        "dataset_id": "ds_test",
        "operation": "describe",
    }


def _ok(result_data: Any = None) -> Dict[str, Any]:
    if result_data is None:
        result_data = {"summary": {"col_a": {"count": 10}}}
    return {
        "status": "success",
        "dataset_id": "ds_test",
        "operation": "describe",
        "result": result_data,
        "row_count": 10,
        "columns": ["col_a"],
    }


class TestSuccessClassification:
    def test_success_valid_structure(self):
        res = _ok({"mean": 42.0})
        out = classify_python_result(res)
        assert out["status"] == "success"
        assert out["category"] == SUCCESS
        assert out["retryable"] is False
        assert out["recovery_action"] == ACTION_NONE
        assert out["result"] == {"mean": 42.0}
        assert out["dataset_id"] == "ds_test"
        assert out["operation"] == "describe"

    def test_success_missing_result_field_is_malformed(self):
        res = {"status": "success", "dataset_id": "ds_test", "operation": "describe"}
        out = classify_python_result(res)
        assert out["status"] == "error"
        assert out["category"] == MALFORMED_RESULT
        assert out["retryable"] is False
        assert out["recovery_action"] == ACTION_NONE
        assert "missing 'result' field" in out["error_message"]

    def test_input_immutability(self):
        res = _ok({"val": [1, 2, 3]})
        snapshot = copy.deepcopy(res)
        out = classify_python_result(res)
        assert res == snapshot
        assert out is not res


@pytest.mark.parametrize(
    "error_type,message,expected_category,expected_action,expected_retryable",
    [
        (
            "InvalidColumnError",
            "Column 'not_found' does not exist in dataset",
            CATEGORY_INVALID_COLUMN,
            ACTION_CORRECT_ARGUMENTS,
            True,
        ),
        (
            "InvalidParameterError",
            "Param 'method' must be 'pearson' or 'spearman'",
            CATEGORY_INVALID_PARAMETER,
            ACTION_CORRECT_ARGUMENTS,
            True,
        ),
        (
            "InvalidOperationError",
            "Operation 'unsupported_op' is not allowlisted",
            CATEGORY_INVALID_OPERATION,
            ACTION_CORRECT_ARGUMENTS,
            True,
        ),
        (
            "DuplicatePeriodError",
            "Duplicate period '2023-01' found in growth_rate calculation",
            CATEGORY_INVALID_DATASET_STRUCTURE,
            ACTION_REGENERATE_DATASET,
            True,
        ),
        (
            "EmptyDatasetError",
            "Cannot perform analysis on an empty dataset",
            CATEGORY_INVALID_DATASET_STRUCTURE,
            ACTION_REGENERATE_DATASET,
            True,
        ),
        (
            "DatasetNotFoundError",
            "Dataset ID 'ds_999' not found in store",
            CATEGORY_MISSING_DATASET,
            ACTION_RECOVER_DATASET_REFERENCE,
            True,
        ),
        (
            "DataFrameConstructionError",
            "Failed to construct DataFrame from stored records",
            CATEGORY_INTERNAL_FAILURE,
            ACTION_NONE,
            False,
        ),
        (
            "InternalExecutionError",
            "Unexpected internal failure during calculation",
            CATEGORY_INTERNAL_FAILURE,
            ACTION_NONE,
            False,
        ),
        (
            "SecurityPolicyError",
            "Arbitrary Python code execution is prohibited",
            CATEGORY_POLICY_VIOLATION,
            ACTION_NONE,
            False,
        ),
        (
            "ExecutionPolicyError",
            "Execution of arbitrary code requested: execute_arbitrary_code",
            CATEGORY_POLICY_VIOLATION,
            ACTION_NONE,
            False,
        ),
        (
            "PolicyViolationError",
            "Policy violation detected",
            CATEGORY_POLICY_VIOLATION,
            ACTION_NONE,
            False,
        ),
    ],
)
def test_known_python_error_types(
    error_type: str,
    message: str,
    expected_category: str,
    expected_action: str,
    expected_retryable: bool,
):
    inp = _err(error_type, message)
    snapshot = copy.deepcopy(inp)
    out = classify_python_result(inp)

    # Immutability check
    assert inp == snapshot

    assert out["status"] == "error"
    assert out["category"] == expected_category
    assert out["recovery_action"] == expected_action
    assert out["retryable"] is expected_retryable
    assert out["error_type"] == error_type
    assert out["error_message"] == message


@pytest.mark.parametrize(
    "unknown_type",
    [
        "RandomError",
        "KeyError",
        "SystemError",
        "CustomToolException",
    ],
)
def test_unknown_error_types(unknown_type: str):
    inp = _err(unknown_type, "some failure")
    out = classify_python_result(inp)
    assert out["status"] == "error"
    assert out["category"] == CATEGORY_UNKNOWN_ERROR
    assert out["recovery_action"] == ACTION_NONE
    assert out["retryable"] is False
    assert out["error_type"] == unknown_type
    assert out["error_message"] == "some failure"


@pytest.mark.parametrize(
    "malformed_input",
    [
        None,
        123,
        "status: success",
        ["a", "b"],
        True,
        {},
        {"status": "unknown_status"},
        {"status": "error"},  # missing error_type and error_message
        {"status": "error", "error_type": ""},  # empty error_type
        {"status": "error", "error_type": "   "},  # whitespace error_type
        {"status": "error", "error_type": 123},  # non-string error_type
        {"status": "error", "error_type": "InvalidColumnError"},  # missing error_message
        {"status": "error", "error_type": "InvalidColumnError", "error_message": 999},  # non-str message
    ],
)
def test_malformed_inputs(malformed_input: Any):
    out = classify_python_result(malformed_input)
    assert out["status"] == "error"
    assert out["category"] == MALFORMED_RESULT
    assert out["retryable"] is False
    assert out["recovery_action"] == ACTION_NONE
    assert out["error_type"] == "MalformedResult"
    assert isinstance(out["error_message"], str)


def test_recoverable_categories_membership():
    for cat in [
        CATEGORY_INVALID_COLUMN,
        CATEGORY_INVALID_PARAMETER,
        CATEGORY_INVALID_OPERATION,
        CATEGORY_INVALID_DATASET_STRUCTURE,
        CATEGORY_MISSING_DATASET,
    ]:
        assert cat in RECOVERABLE_PYTHON_CATEGORIES

    for non_rec in [
        CATEGORY_INTERNAL_FAILURE,
        CATEGORY_POLICY_VIOLATION,
        CATEGORY_UNKNOWN_ERROR,
        MALFORMED_RESULT,
        SUCCESS,
    ]:
        assert non_rec not in RECOVERABLE_PYTHON_CATEGORIES


def test_explicit_policy_violation_flags():
    # Boolean flag policy_violation
    res1 = {
        "status": "error",
        "error_type": "InvalidOperationError",
        "error_message": "execute_arbitrary_code is prohibited",
        "policy_violation": True,
    }
    out1 = classify_python_result(res1)
    assert out1["category"] == CATEGORY_POLICY_VIOLATION
    assert out1["retryable"] is False
    assert out1["recovery_action"] == ACTION_NONE

    # Boolean flag security_violation
    res2 = {
        "status": "error",
        "error_type": "InvalidOperationError",
        "error_message": "Arbitrary code execution attempted",
        "security_violation": True,
    }
    out2 = classify_python_result(res2)
    assert out2["category"] == CATEGORY_POLICY_VIOLATION
    assert out2["retryable"] is False
    assert out2["recovery_action"] == ACTION_NONE

    # Standard InvalidOperationError without policy flag remains recoverable
    res3 = {
        "status": "error",
        "error_type": "InvalidOperationError",
        "error_message": "Unsupported operation 'summary_stats'",
    }
    out3 = classify_python_result(res3)
    assert out3["category"] == CATEGORY_INVALID_OPERATION
    assert out3["retryable"] is True
    assert out3["recovery_action"] == ACTION_CORRECT_ARGUMENTS


def test_repeated_classification_is_deterministic():
    inp = _err("InvalidColumnError", "missing col")
    out1 = classify_python_result(inp)
    out2 = classify_python_result(inp)
    assert out1 == out2
