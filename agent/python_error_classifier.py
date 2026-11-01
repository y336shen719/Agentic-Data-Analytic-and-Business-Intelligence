"""Deterministic classification of ``tools.python_analysis.PythonAnalysisEngine`` results.

Follows the pure, deterministic classification patterns established in
``agent.sql_error_classifier``. Does not execute code, call an LLM, mutate input,
or perform retries directly.
"""

from __future__ import annotations

from typing import Any, Dict

# Category constants
SUCCESS: str = "success"
CATEGORY_INVALID_COLUMN: str = "invalid_column"
CATEGORY_INVALID_PARAMETER: str = "invalid_parameter"
CATEGORY_INVALID_OPERATION: str = "invalid_operation"
CATEGORY_INVALID_DATASET_STRUCTURE: str = "invalid_dataset_structure"
CATEGORY_MISSING_DATASET: str = "missing_dataset"
CATEGORY_INTERNAL_FAILURE: str = "internal_failure"
CATEGORY_POLICY_VIOLATION: str = "policy_violation"
CATEGORY_UNKNOWN_ERROR: str = "unknown_error"
MALFORMED_RESULT: str = "malformed_result"

# Recovery action constants
ACTION_CORRECT_ARGUMENTS: str = "correct_python_arguments"
ACTION_REGENERATE_DATASET: str = "regenerate_dataset"
ACTION_RECOVER_DATASET_REFERENCE: str = "recover_dataset_reference"
ACTION_NONE: str = "none"

# Set of categories considered retryable
RECOVERABLE_PYTHON_CATEGORIES: frozenset[str] = frozenset(
    {
        CATEGORY_INVALID_COLUMN,
        CATEGORY_INVALID_PARAMETER,
        CATEGORY_INVALID_OPERATION,
        CATEGORY_INVALID_DATASET_STRUCTURE,
        CATEGORY_MISSING_DATASET,
    }
)

# Known PythonAnalysisEngine error types mapped to category and recovery action
_PYTHON_ERROR_CONFIG: Dict[str, tuple[str, str]] = {
    "InvalidColumnError": (CATEGORY_INVALID_COLUMN, ACTION_CORRECT_ARGUMENTS),
    "InvalidParameterError": (CATEGORY_INVALID_PARAMETER, ACTION_CORRECT_ARGUMENTS),
    "InvalidOperationError": (CATEGORY_INVALID_OPERATION, ACTION_CORRECT_ARGUMENTS),
    "DuplicatePeriodError": (CATEGORY_INVALID_DATASET_STRUCTURE, ACTION_REGENERATE_DATASET),
    "EmptyDatasetError": (CATEGORY_INVALID_DATASET_STRUCTURE, ACTION_REGENERATE_DATASET),
    "DatasetNotFoundError": (CATEGORY_MISSING_DATASET, ACTION_RECOVER_DATASET_REFERENCE),
    "DataFrameConstructionError": (CATEGORY_INTERNAL_FAILURE, ACTION_NONE),
    "InternalExecutionError": (CATEGORY_INTERNAL_FAILURE, ACTION_NONE),
    "SecurityPolicyError": (CATEGORY_POLICY_VIOLATION, ACTION_NONE),
    "ExecutionPolicyError": (CATEGORY_POLICY_VIOLATION, ACTION_NONE),
    "PolicyViolationError": (CATEGORY_POLICY_VIOLATION, ACTION_NONE),
}


def _malformed(reason: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "category": MALFORMED_RESULT,
        "retryable": False,
        "recovery_action": ACTION_NONE,
        "error_type": "MalformedResult",
        "error_message": reason,
    }


def classify_python_result(result: Any) -> Dict[str, Any]:
    """Classify an ``execute_python_analysis`` tool result.

    Pure, deterministic classifier:
    - Success returns a shallow copy of result plus category='success', retryable=False,
      recovery_action='none'.
    - Errors return {status, category, retryable, recovery_action, error_type, error_message}.
    - The input result dictionary is never mutated.

    Args:
        result: The structured output dictionary from execute_python_analysis.

    Returns:
        Standardized classification dictionary.
    """
    if not isinstance(result, dict):
        return _malformed(f"Python tool result must be a dict, got {type(result).__name__}.")

    status = result.get("status")

    if status == "success":
        if "result" not in result:
            return _malformed("Successful Python result is missing 'result' field.")
        return {
            **result,
            "status": "success",
            "category": SUCCESS,
            "retryable": False,
            "recovery_action": ACTION_NONE,
        }

    if status != "error":
        return _malformed(f"Unexpected or missing result status: {status!r}.")

    error_type = result.get("error_type")
    error_message = result.get("error_message")
    if not isinstance(error_type, str) or not error_type.strip():
        return _malformed("Error result is missing a valid 'error_type'.")
    if not isinstance(error_message, str):
        return _malformed("Error result is missing a valid 'error_message'.")

    if result.get("policy_violation") is True or result.get("security_violation") is True:
        category = CATEGORY_POLICY_VIOLATION
        action = ACTION_NONE
        retryable = False
    elif error_type in _PYTHON_ERROR_CONFIG:
        category, action = _PYTHON_ERROR_CONFIG[error_type]
        retryable = category in RECOVERABLE_PYTHON_CATEGORIES
    else:
        category = CATEGORY_UNKNOWN_ERROR
        action = ACTION_NONE
        retryable = False

    return {
        "status": "error",
        "category": category,
        "retryable": retryable,
        "recovery_action": action,
        "error_type": error_type,
        "error_message": error_message,
    }
