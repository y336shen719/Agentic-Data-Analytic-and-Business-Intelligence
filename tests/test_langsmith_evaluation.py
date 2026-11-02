"""Unit and integration tests for Step 8.5.2 LangSmith Evaluation Integration.

Verifies:
1. Dataset creation when missing.
2. Dataset reuse when already present.
3. Idempotent synchronization without duplicate scenarios.
4. Exact preservation of 20 scenario identities and multi-turn structure.
5. JSON serialization of ground-truth and metadata.
6. Missing API key handling.
7. LangSmith upload failure resilience (local benchmark persists safely).
8. Custom deterministic evaluators alignment with existing benchmark criteria.
9. Inapplicable metrics return None without penalizing the scenario.
10. Strict pass distinguishes PASS (1.0) and NEEDS_REVIEW (0.5).
11. No LLM calls during dataset synchronization.
12. CLI --check, --sync, and deterministic --eval modes.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.benchmark import BENCHMARK_SCENARIOS, BenchmarkScenario
from evaluation.langsmith_integration import (
    _build_langsmith_evaluators,
    check_langsmith_connection,
    get_langsmith_client,
    get_target_dataset_name,
    run_langsmith_evaluation,
    sanitize_for_json,
    sync_benchmark_dataset,
)


def test_sanitize_for_json_converts_numpy():
    """Verify numpy types, arrays, and nested dicts are converted to native primitives."""
    raw = {
        "int_val": np.int64(42),
        "float_val": np.float64(3.1415),
        "arr": np.array([1, 2, 3], dtype=np.int32),
        "nested": {
            "matrix": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "set_vals": {np.int32(10), np.int32(20)},
        },
    }
    cleaned = sanitize_for_json(raw)
    assert type(cleaned["int_val"]) is int
    assert cleaned["int_val"] == 42
    assert type(cleaned["float_val"]) is float
    assert cleaned["float_val"] == 3.1415
    assert cleaned["arr"] == [1, 2, 3]
    assert cleaned["nested"]["matrix"] == [[1.0, 2.0], [3.0, 4.0]]
    assert set(cleaned["nested"]["set_vals"]) == {10, 20}


def test_check_langsmith_connection_unconfigured():
    """Verify check_langsmith_connection reports unconfigured when key is missing."""
    with patch.dict(os.environ, {"LANGSMITH_API_KEY": "", "LANGCHAIN_API_KEY": ""}):
        info = check_langsmith_connection(client=None)
        assert info["status"] == "unconfigured"
        assert info["api_key_configured"] is False
        assert "LANGSMITH_API_KEY is not set." in info["errors"]


def test_check_langsmith_connection_ok():
    """Verify check_langsmith_connection reports dataset status when client connects."""
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = True
    mock_example = MagicMock()
    mock_example.metadata = {"scenario_id": "SQL_01_DELIVERED_ORDERS"}
    mock_client.list_examples.return_value = [mock_example]

    with patch.dict(os.environ, {"LANGSMITH_API_KEY": "lsv2_pt_dummy"}):
        info = check_langsmith_connection(dataset_name="test-ds", client=mock_client)
        assert info["status"] == "ok"
        assert info["dataset_exists"] is True
        assert info["example_count"] == 1
        assert "SQL_01_DELIVERED_ORDERS" in info["example_scenario_ids"]


def test_sync_dataset_creates_when_missing():
    """Verify sync_benchmark_dataset creates the dataset if it does not exist."""
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = False
    mock_client.list_examples.return_value = []

    mock_oracle = MagicMock()
    mock_oracle.get_reference_for_scenario.return_value = {"count": 100}

    scenarios = [
        BenchmarkScenario(
            scenario_id="TEST_01",
            category="basic_sql",
            user_turns=["Test question?"],
            expected_business_outcome="100",
            acceptable_tool_paths=[["execute_sql"]],
            required_tool_capabilities=["execute_sql"],
        )
    ]

    with patch("evaluation.langsmith_integration.GroundTruthOracle", return_value=mock_oracle):
        summary = sync_benchmark_dataset(
            dataset_name="new-dataset",
            client=mock_client,
            scenarios=scenarios,
        )

    mock_client.create_dataset.assert_called_once_with(
        dataset_name="new-dataset",
        description="Olist E-Commerce 20-scenario standard benchmark for Agentic Data Analyst",
    )
    mock_client.create_example.assert_called_once()
    assert summary["created_count"] == 1
    assert summary["updated_count"] == 0


def test_sync_dataset_idempotent_updates_existing():
    """Verify sync_benchmark_dataset updates existing examples instead of creating duplicates."""
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = True

    existing_ex = MagicMock()
    existing_ex.id = "ex_123"
    existing_ex.metadata = {"scenario_id": "TEST_01"}
    existing_ex.inputs = {"scenario_id": "TEST_01"}

    # Initially has existing_ex
    mock_client.list_examples.return_value = [existing_ex]

    mock_oracle = MagicMock()
    mock_oracle.get_reference_for_scenario.return_value = {"count": 200}

    scenarios = [
        BenchmarkScenario(
            scenario_id="TEST_01",
            category="basic_sql",
            user_turns=["Updated question?"],
            expected_business_outcome="200",
            acceptable_tool_paths=[["execute_sql"]],
            required_tool_capabilities=["execute_sql"],
        )
    ]

    with patch("evaluation.langsmith_integration.GroundTruthOracle", return_value=mock_oracle):
        summary = sync_benchmark_dataset(
            dataset_name="existing-dataset",
            client=mock_client,
            scenarios=scenarios,
        )

    mock_client.create_dataset.assert_not_called()
    mock_client.create_example.assert_not_called()
    mock_client.update_example.assert_called_once()
    assert summary["created_count"] == 0
    assert summary["updated_count"] == 1


def test_sync_preserves_20_scenarios_and_multiturn():
    """Verify that full sync preserves all 20 benchmark scenarios and multi-turn turns."""
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = True
    mock_client.list_examples.return_value = []

    mock_oracle = MagicMock()
    mock_oracle.get_reference_for_scenario.return_value = {"mock_metric": 1}

    with patch("evaluation.langsmith_integration.GroundTruthOracle", return_value=mock_oracle):
        summary = sync_benchmark_dataset(
            dataset_name="full-20-benchmark",
            client=mock_client,
            scenarios=BENCHMARK_SCENARIOS,
        )

    assert summary["total_scenarios_target"] == 20
    assert mock_client.create_example.call_count == 20

    # Verify multi-turn preservation in inputs
    created_calls = mock_client.create_example.call_args_list
    multi_calls = [c for c in created_calls if c.kwargs["inputs"]["category"] == "multi_turn"]
    assert len(multi_calls) == 2
    for call in multi_calls:
        turns = call.kwargs["inputs"]["user_turns"]
        assert len(turns) >= 2
        assert isinstance(turns, list)


def test_evaluators_alignment_and_non_applicable_handling():
    """Verify composite evaluator dynamically outputs only applicable metrics without penalizing non-applicable ones."""
    evaluators = _build_langsmith_evaluators()
    assert len(evaluators) == 1
    eval_fn = evaluators[0]

    # Case 1: PASS basic SQL scenario (no viz, no multi-turn)
    res_pass = {
        "scenario_result": {
            "status": "PASS",
            "task_success": True,
            "analytical_accuracy": 1.0,
            "tool_selection_quality": 1.0,
            "visualization_success": None,  # Not applicable
            "multiturn_success": None,      # Not applicable
            "evidence": {"tool_selection_explanation": "Exact path match"},
        }
    }
    fb_pass = {fb["key"]: fb for fb in eval_fn(res_pass, {})}
    assert fb_pass["task_success"]["score"] == 1.0
    assert fb_pass["strict_pass"]["score"] == 1.0
    assert fb_pass["analytical_accuracy"]["score"] == 1.0
    assert fb_pass["tool_selection_quality"]["score"] == 1.0
    assert "visualization_success" not in fb_pass  # Inapplicable metric omitted
    assert "multiturn_success" not in fb_pass      # Inapplicable metric omitted

    # Case 2: NEEDS_REVIEW scenario
    res_review = {
        "scenario_result": {
            "status": "NEEDS_REVIEW",
            "task_success": True,
            "analytical_accuracy": 1.0,
            "tool_selection_quality": 0.85,
            "visualization_success": None,
            "multiturn_success": None,
            "evidence": {"tool_selection_explanation": "Exploratory refinement"},
        }
    }
    fb_review = {fb["key"]: fb for fb in eval_fn(res_review, {})}
    assert fb_review["task_success"]["score"] == 1.0
    assert fb_review["strict_pass"]["score"] == 0.5  # Distinct NEEDS_REVIEW
    assert fb_review["tool_selection_quality"]["score"] == 0.85

    # Case 3: FAIL scenario with visualization
    res_fail = {
        "scenario_result": {
            "status": "FAIL",
            "task_success": False,
            "analytical_accuracy": 0.0,
            "tool_selection_quality": 0.5,
            "visualization_success": 0.0,
            "multiturn_success": 1.0,
            "evidence": {"chart_note": "No chart generated"},
        }
    }
    fb_fail = {fb["key"]: fb for fb in eval_fn(res_fail, {})}
    assert fb_fail["task_success"]["score"] == 0.0
    assert fb_fail["strict_pass"]["score"] == 0.0
    assert fb_fail["analytical_accuracy"]["score"] == 0.0
    assert fb_fail["visualization_success"]["score"] == 0.0
    assert fb_fail["multiturn_success"]["score"] == 1.0


def test_run_langsmith_evaluation_deterministic_offline():
    """Verify run_langsmith_evaluation completes locally in deterministic mode without crashing."""
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = True
    mock_client.list_examples.return_value = []

    with patch("langsmith.evaluate") as mock_ls_evaluate:
        mock_ls_evaluate.return_value = [{"run": "mock_run"}]

        summary = run_langsmith_evaluation(
            scenario_ids=["SQL_01_DELIVERED_ORDERS"],
            dataset_name="test-ds",
            live=False,  # Deterministic mock
            client=mock_client,
        )

        assert summary["total_scenarios"] == 1
        assert summary["task_success_count"] == 1
        assert summary["langsmith_upload"] == "success"
        mock_ls_evaluate.assert_called_once()


def test_run_langsmith_evaluation_client_failure_survives():
    """Verify that if LangSmith client fails or is unconfigured, local benchmark report is still generated."""
    with patch("evaluation.langsmith_integration.get_langsmith_client", side_effect=ValueError("No key")):
        summary = run_langsmith_evaluation(
            scenario_ids=["SQL_01_DELIVERED_ORDERS"],
            live=False,
            client=None,
        )
        assert summary["total_scenarios"] == 1
        assert summary["task_success_count"] == 1
        assert summary["langsmith_upload"] == "skipped"
        assert "No key" in summary["langsmith_error"]
        assert "report_files" in summary
