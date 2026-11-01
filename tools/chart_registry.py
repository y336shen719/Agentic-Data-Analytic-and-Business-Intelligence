"""In-memory, session-scoped Chart Artifact Registry for Plotly figure specifications.

This module provides transient storage for complete Plotly Figure specifications generated
by ChartGenerationEngine. Lightweight references (chart_id, metadata, title) are returned
to the LLM conversation context, while the full, bulky Figure JSON payloads remain in application
memory and can be safely retrieved by future UI layers (e.g. Streamlit).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import time
from typing import Any, Dict, List, Optional
import uuid

from tools.chart_generation import SUPPORTED_CHART_TYPES

DEFAULT_MAX_CHARTS: int = 100
DEFAULT_MAX_REGISTRY_BYTES: int = 20 * 1024 * 1024  # 20MB


class ChartRegistryError(Exception):
    """Base exception for chart registry operations."""


class ChartNotFoundError(ChartRegistryError, KeyError):
    """Raised when a requested chart_id does not exist in the registry."""


class ChartRegistryCapacityError(ChartRegistryError, ValueError):
    """Raised when storing a chart exceeds maximum chart count or memory capacity."""


class InvalidChartRecordError(ChartRegistryError, ValueError):
    """Raised when chart data provided for storage is malformed or invalid."""


@dataclass(frozen=True)
class ChartRecord:
    """Immutable representation of a stored chart artifact."""

    chart_id: str
    dataset_id: str
    chart_type: str
    title: str
    figure: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: float
    size_bytes: int


def _estimate_figure_bytes(figure: Dict[str, Any]) -> int:
    """Estimate memory footprint of a Plotly figure dict in bytes."""
    try:
        return len(json.dumps(figure).encode("utf-8", errors="replace")) + 512
    except Exception:
        return 1024


class ChartRegistry:
    """Bounded, session-scoped in-memory registry for Plotly chart specifications.

    Ensures full Plotly JSON figures are preserved outside LLM conversation contexts,
    prevents cross-session leakage, and enforces strict memory and chart count limits.
    """

    def __init__(
        self,
        max_charts: int = DEFAULT_MAX_CHARTS,
        max_memory_bytes: int = DEFAULT_MAX_REGISTRY_BYTES,
    ) -> None:
        """Initialize a new ChartRegistry instance.

        Args:
            max_charts: Maximum number of charts stored concurrently (> 0).
            max_memory_bytes: Maximum approximate memory capacity in bytes (> 0).
        """
        if isinstance(max_charts, bool) or not isinstance(max_charts, int) or max_charts <= 0:
            raise ValueError(f"max_charts must be a positive integer, got {max_charts!r}")
        if isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0:
            raise ValueError(f"max_memory_bytes must be a positive integer, got {max_memory_bytes!r}")

        self._max_charts = max_charts
        self._max_memory_bytes = max_memory_bytes
        self._charts: Dict[str, ChartRecord] = {}
        self._total_bytes: int = 0

    @property
    def max_charts(self) -> int:
        """Maximum number of charts allowed."""
        return self._max_charts

    @property
    def max_memory_bytes(self) -> int:
        """Maximum memory capacity in bytes."""
        return self._max_memory_bytes

    @property
    def total_memory_bytes(self) -> int:
        """Current estimated total stored bytes across all charts."""
        return self._total_bytes

    def __len__(self) -> int:
        """Return the number of currently stored charts."""
        return len(self._charts)

    def __contains__(self, chart_id: str) -> bool:
        """Check if a chart_id exists in the registry."""
        return chart_id in self._charts

    def save(
        self,
        figure: Dict[str, Any],
        dataset_id: str,
        chart_type: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a Plotly Figure specification and return an opaque chart ID.

        Args:
            figure: Full Plotly Figure specification dictionary.
            dataset_id: Associated originating dataset ID.
            chart_type: Chart type string ('bar', 'line', 'scatter', etc.).
            title: Chart title string.
            metadata: Optional execution/dimension metadata dictionary.

        Returns:
            Opaque chart ID string (e.g. 'chart_a1b2c3d4e5f6').

        Raises:
            InvalidChartRecordError: If input fields are invalid or malformed.
            ChartRegistryCapacityError: If storage bounds would be exceeded.
        """
        if not isinstance(figure, dict) or not figure:
            raise InvalidChartRecordError("figure must be a non-empty dictionary.")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise InvalidChartRecordError("dataset_id must be a non-empty string.")
        if not isinstance(chart_type, str) or chart_type.lower().strip() not in SUPPORTED_CHART_TYPES:
            raise InvalidChartRecordError(
                f"chart_type must be one of {sorted(SUPPORTED_CHART_TYPES)}, got {chart_type!r}"
            )
        if not isinstance(title, str):
            raise InvalidChartRecordError("title must be a string.")

        clean_metadata = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
        clean_figure = copy.deepcopy(figure)
        est_bytes = _estimate_figure_bytes(clean_figure)

        if len(self._charts) >= self._max_charts:
            raise ChartRegistryCapacityError(
                f"ChartRegistry has reached maximum capacity of {self._max_charts} charts."
            )

        if self._total_bytes + est_bytes > self._max_memory_bytes:
            raise ChartRegistryCapacityError(
                f"Storing chart would exceed memory limit ({self._max_memory_bytes} bytes). "
                f"Current: {self._total_bytes}, required: {est_bytes}."
            )

        chart_id = f"chart_{uuid.uuid4().hex[:12]}"
        record = ChartRecord(
            chart_id=chart_id,
            dataset_id=dataset_id,
            chart_type=chart_type.lower().strip(),
            title=title,
            figure=clean_figure,
            metadata=clean_metadata,
            created_at=time.time(),
            size_bytes=est_bytes,
        )

        self._charts[chart_id] = record
        self._total_bytes += est_bytes
        return chart_id

    def get(self, chart_id: str) -> ChartRecord:
        """Retrieve a stored ChartRecord by its opaque chart_id.

        Args:
            chart_id: The opaque chart identifier.

        Returns:
            The stored ChartRecord (deep copy to prevent mutation leakage).

        Raises:
            ChartNotFoundError: If chart_id is missing or invalid.
        """
        if not isinstance(chart_id, str) or not chart_id.strip():
            raise ChartNotFoundError(f"Chart ID must be a non-empty string, got {chart_id!r}")
        if chart_id not in self._charts:
            raise ChartNotFoundError(f"Chart ID '{chart_id}' not found in registry.")
        return copy.deepcopy(self._charts[chart_id])

    def list_charts(self) -> List[Dict[str, Any]]:
        """Return a lightweight list of metadata summaries for all stored charts."""
        return [
            {
                "chart_id": rec.chart_id,
                "dataset_id": rec.dataset_id,
                "chart_type": rec.chart_type,
                "title": rec.title,
                "created_at": rec.created_at,
                "metadata": copy.deepcopy(rec.metadata),
            }
            for rec in self._charts.values()
        ]

    def clear(self) -> None:
        """Evict all stored charts and reset total memory counter."""
        self._charts.clear()
        self._total_bytes = 0
