"""LangSmith Evaluation Integration for the Agentic Data Analyst benchmark.

Provides:
1. Idempotent synchronization of the canonical 20 benchmark scenarios to a LangSmith Dataset.
2. Connection and configuration verification without executing LLM calls.
3. LangSmith evaluation execution wrapping the existing benchmark runner and publishing
   authoritative custom metrics (task_success, strict_pass, analytical_accuracy,
   tool_selection_quality, visualization_success, multiturn_success).
4. Full backward-compatibility with local deterministic evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from agent.tracing import load_langsmith_config, setup_langsmith_environment
from data.prepare_data import DEFAULT_DB_PATH
from evaluation.benchmark import BENCHMARK_SCENARIOS, BenchmarkScenario, get_scenario_by_id
from evaluation.evaluate import EvaluationRunner
from evaluation.ground_truth import GroundTruthOracle
from evaluation.metrics import ScenarioEvaluationResult

logger = logging.getLogger("langsmith_eval")

DEFAULT_BENCHMARK_DATASET: str = "agentic-data-analyst-benchmark-v1"


def get_target_dataset_name(custom_name: Optional[str] = None) -> str:
    """Resolve the benchmark dataset name from args, env, or default."""
    if custom_name and str(custom_name).strip():
        return str(custom_name).strip()
    env_ds = os.environ.get("LANGSMITH_DATASET")
    if env_ds and str(env_ds).strip():
        return str(env_ds).strip()
    return DEFAULT_BENCHMARK_DATASET


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy and non-standard types into JSON-serializable primitives."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    return obj


def get_langsmith_client(api_key: Optional[str] = None) -> Any:
    """Initialize and return a LangSmith Client instance.

    Raises:
        ImportError: If langsmith is not installed.
        ValueError: If no valid API key is configured or discoverable.
    """
    try:
        import langsmith
    except ImportError as exc:
        raise ImportError(
            "The 'langsmith' package is required for LangSmith integration. "
            "Please install it with: pip install langsmith"
        ) from exc

    cfg = load_langsmith_config()
    effective_key = (
        (api_key or "").strip()
        or cfg.api_key
        or os.environ.get("LANGSMITH_API_KEY")
        or os.environ.get("LANGCHAIN_API_KEY")
    )

    if not effective_key:
        raise ValueError(
            "LANGSMITH_API_KEY is not configured. Please set LANGSMITH_API_KEY in .env "
            "or export it in your environment."
        )

    return langsmith.Client(api_key=effective_key, api_url=cfg.endpoint)


def check_langsmith_connection(
    dataset_name: Optional[str] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Verify LangSmith connectivity, credentials, and dataset status without LLM calls."""
    target_ds = get_target_dataset_name(dataset_name)
    cfg = load_langsmith_config()

    info: Dict[str, Any] = {
        "status": "ok",
        "tracing_enabled": cfg.enabled,
        "project": cfg.project,
        "endpoint": cfg.endpoint,
        "dataset_name": target_ds,
        "api_key_configured": bool(cfg.api_key or os.environ.get("LANGSMITH_API_KEY")),
        "dataset_exists": False,
        "example_count": 0,
        "errors": [],
    }

    if not info["api_key_configured"]:
        info["status"] = "unconfigured"
        info["errors"].append("LANGSMITH_API_KEY is not set.")
        return info

    try:
        ls_client = client or get_langsmith_client()
        ds_exists = ls_client.has_dataset(dataset_name=target_ds)
        info["dataset_exists"] = ds_exists

        if ds_exists:
            examples = list(ls_client.list_examples(dataset_name=target_ds))
            info["example_count"] = len(examples)
            info["example_scenario_ids"] = [
                (e.metadata.get("scenario_id") or e.inputs.get("scenario_id", "unknown"))
                for e in examples
            ]
    except Exception as exc:
        info["status"] = "error"
        info["errors"].append(f"LangSmith connection failed: {exc}")

    return info


def sync_benchmark_dataset(
    dataset_name: Optional[str] = None,
    client: Optional[Any] = None,
    db_path: Optional[Union[str, Path]] = None,
    scenarios: Optional[Sequence[BenchmarkScenario]] = None,
) -> Dict[str, Any]:
    """Idempotently synchronize the 20 benchmark scenarios to a LangSmith Dataset.

    Reads benchmark definitions directly from existing code, computes verified reference
    outputs from GroundTruthOracle, and creates or updates Dataset Examples in LangSmith.
    Does NOT invoke the OpenAI API or execute the Agent.

    Returns:
        Summary dict containing counts of created, updated, and existing examples.
    """
    target_ds = get_target_dataset_name(dataset_name)
    target_scenarios = list(scenarios or BENCHMARK_SCENARIOS)

    ls_client = client or get_langsmith_client()
    oracle = GroundTruthOracle(db_path=db_path)

    # 1. Ensure dataset exists
    if not ls_client.has_dataset(dataset_name=target_ds):
        logger.info("Creating new LangSmith dataset: '%s'", target_ds)
        ls_client.create_dataset(
            dataset_name=target_ds,
            description="Olist E-Commerce 20-scenario standard benchmark for Agentic Data Analyst",
        )
    else:
        logger.info("Found existing LangSmith dataset: '%s'", target_ds)

    # 2. Fetch existing examples to ensure idempotence
    existing_examples = list(ls_client.list_examples(dataset_name=target_ds))
    examples_by_scenario_id: Dict[str, Any] = {}
    for ex in existing_examples:
        sid = (ex.metadata or {}).get("scenario_id") or (ex.inputs or {}).get("scenario_id")
        if sid:
            examples_by_scenario_id[sid] = ex

    created_count = 0
    updated_count = 0

    # 3. Synchronize all 20 canonical scenarios
    for sc in target_scenarios:
        # Precompute verified reference metrics directly from GroundTruthOracle
        try:
            raw_gt = oracle.get_reference_for_scenario(sc.scenario_id)
        except Exception as exc:
            logger.warning("Could not precompute ground truth for %s: %s", sc.scenario_id, exc)
            raw_gt = {}

        gt_clean = sanitize_for_json(raw_gt)

        inputs = {
            "scenario_id": sc.scenario_id,
            "category": sc.category,
            "user_turns": sc.user_turns,
            "initial_query": sc.user_turns[0] if sc.user_turns else "",
        }

        outputs = {
            "ground_truth": gt_clean,
            "expected_business_outcome": sc.expected_business_outcome,
            "numeric_keys": sc.numeric_keys,
            "chart_required": sc.chart_required,
            "expected_chart_type": sc.expected_chart_type,
        }

        metadata = {
            "scenario_id": sc.scenario_id,
            "category": sc.category,
            "acceptable_tool_paths": sc.acceptable_tool_paths,
            "required_tool_capabilities": sc.required_tool_capabilities,
            "numeric_rel_tol": sc.numeric_rel_tol,
            "numeric_abs_tol": sc.numeric_abs_tol,
            "evaluation_method": sc.evaluation_method,
            "notes": sc.notes,
        }

        if sc.scenario_id in examples_by_scenario_id:
            ex = examples_by_scenario_id[sc.scenario_id]
            ls_client.update_example(
                example_id=ex.id,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            updated_count += 1
        else:
            ls_client.create_example(
                dataset_name=target_ds,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            created_count += 1

    # 4. Verify total scenario count in dataset
    final_examples = list(ls_client.list_examples(dataset_name=target_ds))
    total_count = len(final_examples)

    summary = {
        "status": "success",
        "dataset_name": target_ds,
        "total_scenarios_target": len(target_scenarios),
        "dataset_example_count": total_count,
        "created_count": created_count,
        "updated_count": updated_count,
        "synced_scenario_ids": [s.scenario_id for s in target_scenarios],
    }

    logger.info(
        "Dataset synchronization complete for '%s': %d created, %d updated, total=%d.",
        target_ds,
        created_count,
        updated_count,
        total_count,
    )
    return summary


def _extract_res_dict(run: Any) -> Dict[str, Any]:
    """Extract scenario_result dictionary safely from dict, Run, or RunTree."""
    if isinstance(run, dict):
        outputs = run.get("outputs") if "outputs" in run else run
    else:
        outputs = getattr(run, "outputs", None) or {}
    if isinstance(outputs, dict):
        return outputs.get("scenario_result", {})
    return {}


def _build_langsmith_evaluators() -> List[Callable[[Any, Any], List[Dict[str, Any]]]]:
    """Construct deterministic LangSmith evaluators matching the custom benchmark metrics.

    Returns a list containing a composite feedback evaluator that outputs a dynamic list
    of evaluations only for applicable metrics, preventing spurious failures on non-applicable checks.
    """

    def benchmark_composite_evaluator(run: Any, example: Any) -> List[Dict[str, Any]]:
        res_dict = _extract_res_dict(run)
        feedbacks: List[Dict[str, Any]] = []

        # 1. task_success
        feedbacks.append({
            "key": "task_success",
            "score": 1.0 if res_dict.get("task_success") else 0.0,
            "comment": f"Status: {res_dict.get('status', 'UNKNOWN')}",
        })

        # 2. strict_pass
        status = res_dict.get("status")
        strict_score = 1.0 if status == "PASS" else (0.5 if status == "NEEDS_REVIEW" else 0.0)
        feedbacks.append({
            "key": "strict_pass",
            "score": strict_score,
            "comment": f"Outcome: {status}",
        })

        # 3. analytical_accuracy (only if applicable)
        acc = res_dict.get("analytical_accuracy")
        if acc is not None:
            feedbacks.append({
                "key": "analytical_accuracy",
                "score": float(acc),
                "comment": f"Accuracy: {acc}",
            })

        # 4. tool_selection_quality
        tool_q = float(res_dict.get("tool_selection_quality", 1.0))
        evidence = res_dict.get("evidence", {})
        expl = evidence.get("tool_selection_explanation", "")
        feedbacks.append({
            "key": "tool_selection_quality",
            "score": tool_q,
            "comment": expl[:200] if expl else f"Score: {tool_q}",
        })

        # 5. visualization_success (only if applicable)
        viz = res_dict.get("visualization_success")
        if viz is not None:
            note = evidence.get("chart_note", "")
            feedbacks.append({
                "key": "visualization_success",
                "score": float(viz),
                "comment": note[:200] if note else f"Viz success: {viz}",
            })

        # 6. multiturn_success (only if applicable)
        mt = res_dict.get("multiturn_success")
        if mt is not None:
            feedbacks.append({
                "key": "multiturn_success",
                "score": float(mt),
                "comment": f"Multi-turn success: {mt}",
            })

        return feedbacks

    return [benchmark_composite_evaluator]


def run_langsmith_evaluation(
    scenario_ids: Optional[Sequence[str]] = None,
    dataset_name: Optional[str] = None,
    live: bool = False,
    client: Optional[Any] = None,
    db_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    experiment_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute evaluation and publish deterministic results to LangSmith Dataset & Experiment.

    Requires live=True to execute against the real OpenAI model.
    In deterministic mode (live=False), runs locally with mock responses for testing.
    """
    mode = "live" if live else "deterministic"
    target_ds = get_target_dataset_name(dataset_name)

    # 1. Run authoritative local benchmark runner
    runner = EvaluationRunner(mode=mode, db_path=db_path, output_dir=output_dir)
    benchmark_summary = runner.run_benchmark(scenario_ids=scenario_ids)

    # 2. Check if LangSmith publishing is possible
    cfg = load_langsmith_config()
    ls_client = client
    if ls_client is None:
        try:
            ls_client = get_langsmith_client()
        except Exception as exc:
            logger.warning("LangSmith client unavailable (%s); evaluation recorded locally only.", exc)
            benchmark_summary["langsmith_upload"] = "skipped"
            benchmark_summary["langsmith_error"] = str(exc)
            return benchmark_summary

    # Ensure dataset exists before publishing evaluation
    try:
        if not ls_client.has_dataset(dataset_name=target_ds):
            logger.info("Dataset '%s' does not exist yet; auto-synchronizing first.", target_ds)
            sync_benchmark_dataset(dataset_name=target_ds, client=ls_client, db_path=db_path)
    except Exception as exc:
        logger.warning("Could not verify or synchronize dataset '%s': %s", target_ds, exc)
        benchmark_summary["langsmith_upload"] = "failed"
        benchmark_summary["langsmith_error"] = str(exc)
        return benchmark_summary

    # 3. Publish scenario results as an Evaluation Experiment in LangSmith
    prefix = experiment_prefix or f"benchmark-{mode}"
    experiment_name = f"{prefix}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    published_count = 0
    errors: List[str] = []

    try:
        import langsmith

        # Target runner function taking a dataset example input
        def evaluate_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
            sid = inputs.get("scenario_id")
            sc = get_scenario_by_id(sid)
            if not sc:
                raise ValueError(f"Unknown scenario ID: {sid}")
            res = runner.run_scenario(sc)
            final_ans = ""
            for turn in reversed(res.evidence.get("turns_output", [])):
                if turn.get("role") == "assistant" and turn.get("content"):
                    final_ans = turn["content"]
                    break
            return {
                "final_answer": final_ans,
                "status": res.status,
                "task_success": res.task_success,
                "actual_tool_sequence": res.actual_tool_sequence,
                "scenario_result": res.to_dict(),
            }

        evaluators = _build_langsmith_evaluators()

        logger.info("Publishing evaluation experiment '%s' to dataset '%s'...", experiment_name, target_ds)
        # Filter examples if scenario_ids specified
        example_filter = None
        if scenario_ids:
            target_ids = set(scenario_ids)
            all_examples = list(ls_client.list_examples(dataset_name=target_ds))
            filtered_examples = [
                e for e in all_examples
                if ((e.metadata or {}).get("scenario_id") in target_ids
                    or (e.inputs or {}).get("scenario_id") in target_ids)
            ]
            data_source = filtered_examples
        else:
            data_source = target_ds

        eval_results = langsmith.evaluate(
            evaluate_target,
            data=data_source,
            evaluators=evaluators,
            experiment_prefix=prefix,
            client=ls_client,
            metadata={
                "benchmark_mode": mode,
                "model": "gpt-6-luna",
                "application": "agentic-data-analyst",
            },
        )
        published_count = len(list(eval_results)) if eval_results else 0
        benchmark_summary["langsmith_upload"] = "success"
        benchmark_summary["langsmith_experiment_name"] = experiment_name
        benchmark_summary["langsmith_published_count"] = published_count

    except Exception as exc:
        logger.exception("Failed to run LangSmith evaluate SDK: %s", exc)
        benchmark_summary["langsmith_upload"] = "partial_or_failed"
        benchmark_summary["langsmith_error"] = str(exc)

    return benchmark_summary


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def main() -> None:
    """CLI entry point for LangSmith evaluation integration."""
    parser = argparse.ArgumentParser(
        description="Agentic Data Analyst — LangSmith Evaluation & Dataset Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify LangSmith configuration, credentials, and dataset status without executing LLM calls.",
    )
    group.add_argument(
        "--sync",
        action="store_true",
        help="Idempotently synchronize the 20 benchmark scenarios to LangSmith Dataset.",
    )
    group.add_argument(
        "--eval",
        action="store_true",
        help="Execute the benchmark and publish evaluated metrics to LangSmith.",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute real LLM calls against the live OpenAI API (incurs API cost).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Target LangSmith dataset name (default: agentic-data-analyst-benchmark-v1).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Target a specific scenario ID (e.g. BI_01_TOP_5_CATEGORIES_REVENUE).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to DuckDB database file.",
    )

    args = parser.parse_args()

    # 1. Check mode
    if args.check:
        print("\n=== LangSmith Configuration & Connection Check ===")
        info = check_langsmith_connection(dataset_name=args.dataset)
        print(f"Status             : {info.get('status')}")
        print(f"Tracing Enabled    : {info.get('tracing_enabled')}")
        print(f"Project            : {info.get('project')}")
        print(f"Target Dataset     : {info.get('dataset_name')}")
        print(f"API Key Configured : {info.get('api_key_configured')}")
        print(f"Dataset Exists     : {info.get('dataset_exists')}")
        print(f"Dataset Examples   : {info.get('example_count')}")
        if info.get("errors"):
            print("Errors / Warnings  :")
            for err in info["errors"]:
                print(f"  - {err}")
        print("==================================================\n")
        sys.exit(0 if info.get("status") == "ok" else 1)

    # 2. Sync mode
    if args.sync:
        print("\n=== Synchronizing Benchmark Scenarios to LangSmith ===")
        try:
            res = sync_benchmark_dataset(
                dataset_name=args.dataset,
                db_path=args.db_path,
            )
            print(f"Dataset Name       : {res.get('dataset_name')}")
            print(f"Target Scenarios   : {res.get('total_scenarios_target')}")
            print(f"Total in Dataset   : {res.get('dataset_example_count')}")
            print(f"Created (New)      : {res.get('created_count')}")
            print(f"Updated (Existing) : {res.get('updated_count')}")
            print("Status             : SUCCESS (0 OpenAI API calls consumed)")
            print("=====================================================\n")
        except Exception as exc:
            print(f"Sync Failed: {exc}")
            sys.exit(1)
        sys.exit(0)

    # 3. Eval mode
    if args.eval:
        if not args.live:
            print(
                "\n[NOTE] Running in DETERMINISTIC mode (mocked responses, 0 OpenAI cost).\n"
                "To run live evaluation against real LLM, pass the '--live' flag.\n"
            )

        scenario_ids = [args.scenario] if args.scenario else None
        res = run_langsmith_evaluation(
            scenario_ids=scenario_ids,
            dataset_name=args.dataset,
            live=args.live,
            db_path=args.db_path,
        )

        print("\n=== Evaluation & LangSmith Publishing Summary ===")
        print(f"Execution Mode     : {res.get('mode')}")
        print(f"Total Scenarios    : {res.get('total_scenarios')}")
        print(f"Pass Rate          : {res.get('pass_rate_pct')}%")
        print(f"Task Success Count : {res.get('task_success_count')}/{res.get('total_scenarios')}")
        print(f"LangSmith Upload   : {res.get('langsmith_upload')}")
        if res.get("langsmith_experiment_name"):
            print(f"Experiment Name    : {res.get('langsmith_experiment_name')}")
        if res.get("langsmith_error"):
            print(f"LangSmith Note     : {res.get('langsmith_error')}")
        print("==================================================\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
