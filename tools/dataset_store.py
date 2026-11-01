"""In-memory, bounded dataset storage for SQL query results.

This module provides the data handoff mechanism from SQL query execution to downstream
analytical tools (e.g., Python Analysis Tool). Structured query results are held in an
isolated DatasetStore instance rather than being duplicated in full inside LLM conversation
message histories.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import time
from typing import Any, Dict, List, Optional
import uuid


class DatasetStoreError(Exception):
    """Base exception for dataset store errors."""


class DatasetNotFoundError(DatasetStoreError, KeyError):
    """Raised when a requested dataset_id does not exist or has been deleted."""


class DatasetCapacityError(DatasetStoreError):
    """Raised when storing a dataset would exceed count or memory limits."""


class InvalidDatasetError(DatasetStoreError, ValueError):
    """Raised when an invalid or malformed SQL result is provided for storage."""


@dataclass(frozen=True)
class DatasetRecord:
    """Immutable representation of a stored SQL query result dataset."""

    dataset_id: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    created_at: float
    source_sql: Optional[str] = None
    truncated: bool = False
    size_bytes: int = 0


def _estimate_payload_bytes(columns: List[str], rows: List[List[Any]]) -> int:
    """Estimate serialized memory footprint in bytes for a dataset payload."""
    col_bytes = sum(len(c.encode("utf-8", errors="replace")) for c in columns)
    try:
        data_json = json.dumps(rows, default=str)
        row_bytes = len(data_json.encode("utf-8", errors="replace"))
    except Exception:
        # Fallback estimation if custom non-serializable objects exist
        row_bytes = sum(
            sum(len(str(val).encode("utf-8", errors="replace")) for val in r)
            for r in rows
        )
    # Include modest base metadata overhead (256 bytes)
    return col_bytes + row_bytes + 256


class DatasetStore:
    """Bounded, session-scoped in-memory store for analytical SQL query results.

    Attributes:
        max_datasets: Maximum number of datasets allowed in the store at once.
        max_memory_bytes: Maximum approximate memory capacity in bytes.
    """

    def __init__(
        self,
        max_datasets: int = 50,
        max_memory_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        """Initialize a new DatasetStore instance.

        Args:
            max_datasets: Maximum number of datasets stored concurrently (> 0).
            max_memory_bytes: Maximum memory capacity in bytes (> 0). Defaults to 50MB.
        """
        if isinstance(max_datasets, bool) or not isinstance(max_datasets, int) or max_datasets <= 0:
            raise ValueError(f"max_datasets must be a positive integer, got {max_datasets!r}")
        if isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int) or max_memory_bytes <= 0:
            raise ValueError(f"max_memory_bytes must be a positive integer, got {max_memory_bytes!r}")

        self._max_datasets = max_datasets
        self._max_memory_bytes = max_memory_bytes
        self._datasets: Dict[str, DatasetRecord] = {}
        self._total_bytes: int = 0

    @property
    def max_datasets(self) -> int:
        """Maximum number of datasets allowed."""
        return self._max_datasets

    @property
    def max_memory_bytes(self) -> int:
        """Maximum memory capacity in bytes."""
        return self._max_memory_bytes

    @property
    def total_memory_bytes(self) -> int:
        """Current estimated total stored bytes across all datasets."""
        return self._total_bytes

    def __len__(self) -> int:
        """Return the number of currently stored datasets."""
        return len(self._datasets)

    def save(self, sql_result: Dict[str, Any], query: Optional[str] = None) -> str:
        """Store a successful SQL query result and return an opaque dataset reference ID.

        Args:
            sql_result: Structured result dictionary from `tools.sql_tool.execute_sql`.
            query: Optional SQL query string associated with this result for provenance.

        Returns:
            Opaque unique dataset ID string (e.g. 'ds_...').

        Raises:
            InvalidDatasetError: If the SQL result is not a successful result dict or is malformed.
            DatasetCapacityError: If storing the dataset exceeds count or memory capacity.
        """
        if not isinstance(sql_result, dict):
            raise InvalidDatasetError(f"sql_result must be a dict, got {type(sql_result).__name__}.")

        status = sql_result.get("status")
        if status != "success":
            raise InvalidDatasetError(
                f"Cannot save non-successful SQL result with status {status!r}."
            )

        columns = sql_result.get("columns")
        rows = sql_result.get("rows")
        if not isinstance(columns, list):
            raise InvalidDatasetError("SQL result is missing a valid 'columns' list.")
        if not isinstance(rows, list):
            raise InvalidDatasetError("SQL result is missing a valid 'rows' list.")

        for i, col in enumerate(columns):
            if not isinstance(col, str):
                raise InvalidDatasetError(f"Column name at index {i} must be a str, got {type(col).__name__}.")

        # Deep-copy rows and columns to isolate the store from external mutation
        copied_columns = [str(c) for c in columns]
        copied_rows = copy.deepcopy(rows)
        row_count = int(sql_result.get("row_count", len(copied_rows)))
        truncated = bool(sql_result.get("truncated", False))

        estimated_bytes = _estimate_payload_bytes(copied_columns, copied_rows)

        # Capacity checks
        if len(self._datasets) >= self._max_datasets:
            raise DatasetCapacityError(
                f"Dataset store reached maximum dataset count limit ({self._max_datasets})."
            )

        if self._total_bytes + estimated_bytes > self._max_memory_bytes:
            raise DatasetCapacityError(
                f"Dataset store memory limit exceeded ({self._max_memory_bytes} bytes). "
                f"Required: {estimated_bytes} bytes, Available: {self._max_memory_bytes - self._total_bytes} bytes."
            )

        dataset_id = f"ds_{uuid.uuid4().hex}"
        record = DatasetRecord(
            dataset_id=dataset_id,
            columns=copied_columns,
            rows=copied_rows,
            row_count=row_count,
            created_at=time.time(),
            source_sql=query if isinstance(query, str) else None,
            truncated=truncated,
            size_bytes=estimated_bytes,
        )

        self._datasets[dataset_id] = record
        self._total_bytes += estimated_bytes
        return dataset_id

    def get(self, dataset_id: str) -> DatasetRecord:
        """Retrieve a stored dataset by its opaque ID.

        Args:
            dataset_id: The unique dataset identifier.

        Returns:
            A deep copy of the DatasetRecord to ensure stored data cannot be mutated.

        Raises:
            DatasetNotFoundError: If dataset_id does not exist.
        """
        if not isinstance(dataset_id, str) or dataset_id not in self._datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")

        rec = self._datasets[dataset_id]
        # Return deep copy of rows and columns to prevent external mutation
        return DatasetRecord(
            dataset_id=rec.dataset_id,
            columns=list(rec.columns),
            rows=copy.deepcopy(rec.rows),
            row_count=rec.row_count,
            created_at=rec.created_at,
            source_sql=rec.source_sql,
            truncated=rec.truncated,
            size_bytes=rec.size_bytes,
        )

    def describe(self, dataset_id: str) -> Dict[str, Any]:
        """Return lightweight metadata describing a stored dataset.

        Args:
            dataset_id: The unique dataset identifier.

        Returns:
            Dict containing metadata (id, columns, row_count, source_sql, created_at, size_bytes).

        Raises:
            DatasetNotFoundError: If dataset_id does not exist.
        """
        if not isinstance(dataset_id, str) or dataset_id not in self._datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")

        rec = self._datasets[dataset_id]
        return {
            "dataset_id": rec.dataset_id,
            "columns": list(rec.columns),
            "row_count": rec.row_count,
            "truncated": rec.truncated,
            "source_sql": rec.source_sql,
            "created_at": rec.created_at,
            "size_bytes": rec.size_bytes,
        }

    def delete(self, dataset_id: str) -> bool:
        """Delete a single dataset by ID and free its memory allocation.

        Args:
            dataset_id: The unique dataset identifier.

        Returns:
            True if the dataset was found and deleted, False otherwise.
        """
        if not isinstance(dataset_id, str) or dataset_id not in self._datasets:
            return False

        rec = self._datasets.pop(dataset_id)
        self._total_bytes = max(0, self._total_bytes - rec.size_bytes)
        return True

    def clear(self) -> None:
        """Clear all stored datasets and reset memory usage to zero."""
        self._datasets.clear()
        self._total_bytes = 0

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return metadata summaries for all datasets currently held in this store."""
        return [self.describe(d_id) for d_id in self._datasets]
