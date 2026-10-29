"""Lightweight, reusable DuckDB connection, schema discovery, and safe SQL execution module.

Provides read-only DuckDB database connections, dynamic schema discovery,
centralized business metadata for analysis-ready views, AST-based query validation,
and secure execution with bounded timeouts and resource caps.
Designed for integration into LangGraph-based agent tools without LLM dependencies.
"""

from __future__ import annotations

from contextlib import contextmanager
import datetime
import decimal
import logging
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import selectors
import struct
import time
from typing import Any, Collection, Dict, Generator, List, Optional, Set, Tuple, Union
import uuid

import duckdb
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, build_scope

from data.prepare_data import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

# Schema that owns the Olist application views; used to scope business metadata.
_APP_SCHEMA = "main"

# Application-controlled upper bounds for execution limits
MAX_PERMITTED_ROWS: int = 5000
MAX_PERMITTED_TIMEOUT_SECONDS: float = 30.0
MAX_SERIALIZED_PAYLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MB serialized IPC payload limit

# Default allowlist of known Olist base tables and views.
_DEFAULT_ALLOWED_TABLES: frozenset[str] = frozenset({
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "customers",
    "sellers",
    "products",
    "geolocation",
    "category_translation",
    "order_summary",
    "category_sales_summary",
})

# Allowlist of safe analytical expression classes in SQLGlot (built-in Func representations)
_ALLOWED_FUNC_NAMES: frozenset[str] = frozenset({
    # Math & Numeric
    "Abs", "Ceil", "Floor", "Round", "Sqrt", "Cbrt", "Exp", "Ln", "Log",
    "Pow", "Sign", "Greatest", "Least", "Sin", "Cos", "Tan", "Asin",
    "Acos", "Atan", "Atan2", "Degrees", "Radians",
    # String
    "Lower", "Upper", "Trim", "Concat", "ConcatWs", "Substring",
    "Length", "Replace", "Split", "Left", "Right", "Repeat",
    "Pad", "StrPosition", "RegexpLike", "RegexpReplace",
    # Date & Time
    "CurrentDate", "CurrentTime", "CurrentTimestamp", "Date",
    "DateAdd", "DateDiff", "DateSub", "DateTrunc", "TimestampTrunc",
    "TimeToStr", "StrToTime", "TimeToUnix", "UnixToTime", "Extract",
    "Year", "Month", "Day",
    # Aggregates & Stats
    "Count", "Sum", "Avg", "Min", "Max", "GroupConcat", "ArrayAgg",
    "Stddev", "StddevPop", "StddevSamp", "Variance", "VariancePop",
    # Window
    "RowNumber", "Rank", "DenseRank", "Lag", "Lead", "FirstValue",
    "LastValue", "Ntile", "CumeDist", "PercentRank",
    # Conditional, Case & Type
    "Case", "If", "Coalesce", "Nullif", "Cast", "TryCast", "Collate",
    # Anonymous functions (validated by name against _ALLOWED_ANONYMOUS_FUNCTIONS)
    "Anonymous",
})

_ALLOWED_FUNC_CLASSES: frozenset[type] = frozenset(
    getattr(exp, n) for n in _ALLOWED_FUNC_NAMES if hasattr(exp, n)
)

# Allowlist of safe analytical functions that may appear as Anonymous functions in SQLGlot.
_ALLOWED_ANONYMOUS_FUNCTIONS: frozenset[str] = frozenset({
    # Math & Numeric
    "abs", "round", "ceil", "ceiling", "floor", "power", "pow", "sqrt",
    "cbrt", "exp", "ln", "log", "log10", "log2", "sign", "pi", "degrees",
    "radians", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "random", "gen_random_uuid", "greatest", "least",
    # String
    "lower", "upper", "trim", "ltrim", "rtrim", "concat", "concat_ws",
    "substr", "substring", "length", "len", "replace", "split_part",
    "left", "right", "reverse", "repeat", "lpad", "rpad", "position",
    "strpos", "instr", "chr", "ascii", "format", "regexp_replace",
    "regexp_matches", "string_agg", "starts_with", "ends_with", "contains",
    # Date & Time
    "date_trunc", "datetrunc", "date_diff", "datediff", "date_add", "date_sub",
    "date_part", "epoch", "epoch_ms", "to_timestamp", "strptime", "strftime",
    "current_date", "current_time", "current_timestamp", "now", "today",
    "extract", "age", "year", "month", "day", "hour", "minute", "second",
    # Aggregates & Stats
    "count", "sum", "avg", "min", "max", "median", "mode",
    "approx_count_distinct", "approx_quantile", "quantile_cont", "quantile_disc",
    "stddev", "stddev_pop", "stddev_samp", "variance", "var_pop", "var_samp",
    "corr", "covar_pop", "covar_samp", "array_agg", "list",
    # Window
    "row_number", "rank", "dense_rank", "ntile", "lag", "lead",
    "first_value", "last_value", "cume_dist", "percent_rank",
    # Conditional & Null handling
    "coalesce", "nullif", "ifnull", "nvl",
})


class SQLValidationError(ValueError):
    """Raised when a SQL query violates security or analytical policies."""


# ==============================================================================
# Centralized Business Metadata for Analysis-Ready Views
# ==============================================================================

# Keyed by (schema, view_name) so metadata is never attached to an unrelated view
# in a different schema that happens to share the same name.
VIEW_BUSINESS_METADATA: Dict[Tuple[str, str], Dict[str, Any]] = {
    (_APP_SCHEMA, "order_summary"): {
        "grain": "One row per order.",
        "purpose": "Order-level analysis, payments, revenue, and delivery metrics.",
        "notes": "Pre-aggregated at order grain to prevent join fanout across items, payments, and reviews.",
        "column_descriptions": {
            "order_id": "Unique identifier of the order (primary key, one row per order).",
            "customer_id": "Unique customer identifier tied to this specific order.",
            "customer_unique_id": "Persistent unique customer identifier across repeated purchases.",
            "customer_city": "City where the customer is located.",
            "customer_state": "Two-letter state abbreviation of the customer.",
            "customer_zip_code_prefix": "First 5 digits of the customer ZIP code.",
            "order_status": "Current lifecycle status of the order (e.g., delivered, shipped, canceled).",
            "order_purchase_timestamp": "Timestamp when the order was placed by the customer.",
            "order_approved_at": "Timestamp when payment for the order was approved.",
            "order_delivered_carrier_date": "Timestamp when the order was handed over to the logistics carrier.",
            "order_delivered_customer_date": "Timestamp when the order was delivered to the customer.",
            "order_estimated_delivery_date": "Estimated delivery date promised to the customer at purchase.",
            "item_count": "Total count of items in the order (0 for unfulfilled orders without item records).",
            "unique_products_count": "Count of distinct product IDs in the order.",
            "unique_sellers_count": "Count of distinct seller IDs fulfilling items in the order.",
            "item_subtotal": "Sum of item merchandise prices excluding freight (DECIMAL(18,2)).",
            "freight_total": "Sum of freight/shipping charges across order items (DECIMAL(18,2)).",
            "order_gross_value": "Total gross order value: item_subtotal + freight_total (DECIMAL(18,2)).",
            "payment_count": "Total count of payment installment/sequence records for the order.",
            "payment_value_total": "Total monetary value paid across all payment methods (DECIMAL(18,2)).",
            "payment_types": "Comma-separated list of distinct payment types used (e.g., credit_card, voucher).",
            "max_installments": "Maximum number of installments selected across payment methods for the order.",
            "review_count": "Total count of customer reviews submitted for the order.",
            "avg_review_score": "Average review rating score for the order (1.0 to 5.0).",
            "is_delivered": "Boolean flag indicating whether order_status is 'delivered'.",
            "delivery_duration_days": "Number of days from purchase to delivery (NULL if not delivered).",
        },
    },
    (_APP_SCHEMA, "category_sales_summary"): {
        "grain": "One row per month and product category.",
        "purpose": "Category-level sales and revenue analysis across time.",
        "warning": (
            "Distinct order counts (total_orders, delivered_orders) are non-additive across categories. "
            "Averages (avg_item_price) should be reconstructed using appropriate weights "
            "(sum(total_item_revenue) / sum(total_items_sold))."
        ),
        "column_descriptions": {
            "year_month": "Month of order purchase in 'YYYY-MM' format.",
            "order_year": "Calendar year of order purchase (INTEGER).",
            "order_month": "Calendar month of order purchase (1-12, INTEGER).",
            "product_category_name": "Original Portuguese product category name ('uncategorized' if missing).",
            "category_name_english": "English translation of product category name (fallback to Portuguese or 'uncategorized').",
            "total_orders": "Count of distinct orders in this month and category. WARNING: Non-additive across categories.",
            "total_items_sold": "Total count of items sold in this month and category.",
            "total_item_revenue": "Total revenue from item prices excluding freight (DECIMAL(18,2)).",
            "total_freight_value": "Total freight charges collected for items in this category (DECIMAL(18,2)).",
            "total_gross_sales": "Total gross sales revenue: total_item_revenue + total_freight_value (DECIMAL(18,2)).",
            "delivered_orders": "Count of distinct delivered orders in this month and category. WARNING: Non-additive across categories.",
            "delivered_items_sold": "Count of items sold in delivered orders in this month and category.",
            "delivered_item_revenue": "Total revenue from item prices in delivered orders (DECIMAL(18,2)).",
            "avg_item_price": "Average item price: total_item_revenue / total_items_sold. Non-additive; reconstruct with weighted average.",
            "active_sellers": "Count of distinct sellers with at least one item sold in this month and category.",
        },
    },
}


# ==============================================================================
# Database Connection Management
# ==============================================================================

def get_db_connection(db_path: Optional[Union[str, Path]] = None) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection to the target database.

    Args:
        db_path: Path to the DuckDB database file. Defaults to DEFAULT_DB_PATH
            defined in data/prepare_data.py.

    Returns:
        duckdb.DuckDBPyConnection: An active read-only connection.

    Raises:
        FileNotFoundError: If the database file does not exist.
        ValueError: If db_path points to a directory rather than a file.
        duckdb.Error: If DuckDB fails to open the database file.
    """
    target_path = Path(db_path) if db_path is not None else Path(DEFAULT_DB_PATH)
    resolved_path = target_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"DuckDB database file not found at: {resolved_path}. "
            "Please ensure the database has been initialized."
        )

    if resolved_path.is_dir():
        raise ValueError(
            f"Provided database path is a directory, not a DuckDB file: {resolved_path}"
        )

    try:
        con = duckdb.connect(database=str(resolved_path), read_only=True)
        return con
    except Exception as exc:
        logger.error("Failed to connect to DuckDB at %s in read-only mode: %s", resolved_path, exc)
        raise


@contextmanager
def db_connection(db_path: Optional[Union[str, Path]] = None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager for acquiring and safely closing a read-only DuckDB connection.

    The connection is closed in the finally block so cleanup is guaranteed
    even if an exception is raised inside the with-block.

    Args:
        db_path: Path to the DuckDB database file. Defaults to DEFAULT_DB_PATH.

    Yields:
        duckdb.DuckDBPyConnection: An active read-only connection.
    """
    con = get_db_connection(db_path)
    try:
        yield con
    finally:
        con.close()


# ==============================================================================
# Dynamic Schema Discovery
# ==============================================================================

def get_database_schema(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Dynamically discover tables, views, columns, and business metadata from DuckDB.

    Queries DuckDB's information_schema to discover all user-defined base tables
    and views within the primary database, their columns, data types, and
    nullability, while excluding internal system catalogs.

    Nullability note:
        The ``nullable`` field reflects what DuckDB's information_schema reports
        (the ``is_nullable`` column). It represents the catalog-level NOT NULL
        constraint, not an empirically validated data-quality guarantee. View
        columns are almost always reported as nullable by DuckDB regardless of
        the underlying base-table constraints. If the value cannot be interpreted
        reliably, ``None`` is returned rather than inventing a value.

    Business metadata (grain, purpose, warnings, column descriptions) is attached
    only to views in the application schema (``main``) that are registered in
    ``VIEW_BUSINESS_METADATA``. Views with the same name in other schemas are
    not enriched.

    Args:
        db_path: Path to the DuckDB database file. Defaults to DEFAULT_DB_PATH.

    Returns:
        Dict with keys ``"tables"`` and ``"views"``, each a list sorted
        deterministically by ``(schema, name)``:
    """
    with db_connection(db_path) as con:
        # Scope to the primary database only (excludes ATTACHed databases).
        table_rows: List[Tuple[str, str, str]] = con.execute("""
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables
            WHERE table_catalog = current_database()
              AND table_schema NOT IN ('information_schema', 'pg_catalog')
              AND table_schema NOT LIKE 'pg_%'
              AND table_name NOT LIKE 'sqlite_%'
              AND table_name NOT LIKE 'duckdb_%'
            ORDER BY table_schema ASC, table_name ASC;
        """).fetchall()

        column_rows: List[Tuple[str, str, str, str, Optional[str], int]] = con.execute("""
            SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
            FROM information_schema.columns
            WHERE table_catalog = current_database()
              AND table_schema NOT IN ('information_schema', 'pg_catalog')
              AND table_schema NOT LIKE 'pg_%'
              AND table_name NOT LIKE 'sqlite_%'
              AND table_name NOT LIKE 'duckdb_%'
            ORDER BY table_schema ASC, table_name ASC, ordinal_position ASC;
        """).fetchall()

    # Group columns by (schema, table_name) to prevent cross-schema merging.
    columns_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for tbl_schema, tbl_name, col_name, data_type, is_nullable, _ord_pos in column_rows:
        nullable: Optional[bool] = None
        if is_nullable is not None:
            norm = str(is_nullable).strip().upper()
            if norm == "YES":
                nullable = True
            elif norm == "NO":
                nullable = False

        col_def: Dict[str, Any] = {
            "name": str(col_name),
            "data_type": str(data_type),
            "nullable": nullable,
        }

        # Attach per-column business description only for the exact (schema, view) pair.
        meta_key = (tbl_schema, tbl_name)
        if meta_key in VIEW_BUSINESS_METADATA:
            desc_map = VIEW_BUSINESS_METADATA[meta_key].get("column_descriptions", {})
            if col_name in desc_map:
                col_def["description"] = desc_map[col_name]

        columns_by_key.setdefault(meta_key, []).append(col_def)

    tables_list: List[Dict[str, Any]] = []
    views_list: List[Dict[str, Any]] = []

    for tbl_schema, tbl_name, tbl_type in table_rows:
        key = (tbl_schema, tbl_name)
        cols = columns_by_key.get(key, [])
        norm_type = str(tbl_type).upper()

        if "VIEW" in norm_type:
            view_item: Dict[str, Any] = {
                "name": tbl_name,
                "schema": tbl_schema,
                "type": "VIEW",
                "columns": cols,
            }
            if key in VIEW_BUSINESS_METADATA:
                meta = VIEW_BUSINESS_METADATA[key]
                for field in ("grain", "purpose", "warning", "notes"):
                    if field in meta:
                        view_item[field] = meta[field]
            views_list.append(view_item)
        else:
            tables_list.append({
                "name": tbl_name,
                "schema": tbl_schema,
                "type": "BASE TABLE",
                "columns": cols,
            })

    # Already ordered by (schema, name) from the SQL query; re-sort defensively.
    tables_list.sort(key=lambda t: (t["schema"], t["name"]))
    views_list.sort(key=lambda v: (v["schema"], v["name"]))

    return {"tables": tables_list, "views": views_list}


# ==============================================================================
# SQL Validation (AST-based)
# ==============================================================================

def validate_sql(query: str, allowed_tables: Optional[Collection[str]] = None) -> None:
    """Validate that query is a single, safe, read-only analytical SQL statement.

    Args:
        query: SQL query string to validate.
        allowed_tables: Optional collection of permitted table/view names.
            If None, defaults to the standard application tables and views.

    Raises:
        SQLValidationError: If query violates security, read-only, or AST policies.
    """
    if not isinstance(query, str) or not query.strip():
        raise SQLValidationError("SQL query must be a non-empty string.")

    allowed = (
        _DEFAULT_ALLOWED_TABLES
        if allowed_tables is None
        else frozenset(t.lower() for t in allowed_tables)
    )

    try:
        raw_trees = sqlglot.parse(query, read="duckdb")
    except Exception as exc:
        raise SQLValidationError(f"Malformed SQL statement: {exc}") from exc

    trees = [t for t in raw_trees if t is not None]
    if not trees:
        raise SQLValidationError("SQL query contains no executable statements.")
    if len(trees) > 1:
        raise SQLValidationError("Multiple SQL statements are not permitted.")

    root = trees[0]
    if not isinstance(root, exp.Query):
        raise SQLValidationError(
            f"Only read-only analytical SELECT queries are permitted, got: {type(root).__name__}"
        )

    # Check for forbidden modifying / administrative constructs anywhere in the AST
    forbidden_types = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Into,
        exp.Command,
        exp.Pragma,
        exp.Attach,
        exp.Detach,
        exp.Copy,
        exp.Set,
        exp.Transaction,
        exp.Commit,
        exp.Rollback,
    )
    forbidden_matches = list(root.find_all(forbidden_types))
    if forbidden_matches:
        node_name = type(forbidden_matches[0]).__name__
        raise SQLValidationError(
            f"Disallowed DDL/DML/administrative operation detected in AST: {node_name}"
        )

    # Validate function calls against conservative allowlist
    for func in root.find_all(exp.Func):
        if type(func) not in _ALLOWED_FUNC_CLASSES:
            raise SQLValidationError(
                f"Function expression of type '{type(func).__name__}' is not permitted."
            )

        fname = (getattr(func, "name", "") or "").lower()
        if fname.startswith(("read_", "scan_", "write_", "duckdb_")) or fname in {
            "cfree", "glob", "httpfs", "sniff_csv", "parquet_scan", "query",
            "query_table", "postgres_scan", "sqlite_scan", "attach", "getenv"
        }:
            raise SQLValidationError(f"Function '{fname}' is not permitted.")

        if isinstance(func, exp.Anonymous):
            if fname not in _ALLOWED_ANONYMOUS_FUNCTIONS:
                raise SQLValidationError(
                    f"Function '{func.name}' is not in the allowed analytical functions list."
                )

    # Scope-aware table validation using SQLGlot's build_scope (FAIL-CLOSED)
    try:
        root_scope = build_scope(root)
    except Exception as scope_exc:
        raise SQLValidationError(f"SQL scope analysis failed: {scope_exc}") from scope_exc

    if root_scope is None:
        raise SQLValidationError("SQL scope analysis returned no valid query scope.")

    for scope in root_scope.traverse():
        in_scope_cte_names: Set[str] = {
            name.lower()
            for name, src in scope.sources.items()
            if isinstance(src, Scope)
        }
        for tbl in scope.tables:
            if not isinstance(tbl.this, (exp.Identifier, exp.Var)):
                raise SQLValidationError(
                    "Table-valued functions or non-standard table references are not permitted."
                )

            tbl_name = tbl.name.strip().lower()
            if not tbl_name:
                raise SQLValidationError("Invalid or empty table reference.")

            if any(tbl_name.endswith(ext) for ext in (".csv", ".parquet", ".json", ".tsv", ".txt", ".db", ".duckdb")):
                raise SQLValidationError(f"Direct file access in queries is prohibited: '{tbl.name}'")
            if "/" in tbl_name or "\\" in tbl_name or tbl_name.startswith(("http:", "https:", "s3:", "gcs:", "gs:", "azure:", "hf:")):
                raise SQLValidationError(f"External URL or file path reference is prohibited: '{tbl.name}'")

            if tbl.catalog:
                raise SQLValidationError(f"External database catalog references are prohibited: '{tbl.catalog}'")
            if tbl.db and tbl.db.lower() not in ("main", ""):
                raise SQLValidationError(
                    f"Access to schema '{tbl.db}' is prohibited. Only 'main' schema tables and views are allowed."
                )

            # If schema is qualified (e.g. main.t), it cannot resolve to a CTE; must be in allowed
            if tbl.db:
                if tbl_name not in allowed:
                    raise SQLValidationError(
                        f"Table or view '{tbl.name}' is not in the allowed tables list."
                    )
            else:
                # Unqualified: check if it resolves to an in-scope CTE / subquery source
                is_cte_in_scope = tbl_name in in_scope_cte_names
                if not is_cte_in_scope and tbl_name not in allowed:
                    raise SQLValidationError(
                        f"Table or view '{tbl.name}' is not in the allowed tables list."
                    )

    # Defense-in-depth: scan all exp.Table nodes across the AST for syntax anomalies & disallowed schemas
    for tbl in root.find_all(exp.Table):
        if not isinstance(tbl.this, (exp.Identifier, exp.Var)):
            raise SQLValidationError(
                "Table-valued functions or non-standard table references are not permitted."
            )
        tbl_name = tbl.name.strip().lower()
        if not tbl_name:
            raise SQLValidationError("Invalid or empty table reference.")
        if any(tbl_name.endswith(ext) for ext in (".csv", ".parquet", ".json", ".tsv", ".txt", ".db", ".duckdb")):
            raise SQLValidationError(f"Direct file access in queries is prohibited: '{tbl.name}'")
        if "/" in tbl_name or "\\" in tbl_name or tbl_name.startswith(("http:", "https:", "s3:", "gcs:", "gs:", "azure:", "hf:")):
            raise SQLValidationError(f"External URL or file path reference is prohibited: '{tbl.name}'")
        if tbl.catalog:
            raise SQLValidationError(f"External database catalog references are prohibited: '{tbl.catalog}'")
        if tbl.db and tbl.db.lower() not in ("main", ""):
            raise SQLValidationError(
                f"Access to schema '{tbl.db}' is prohibited. Only 'main' schema tables and views are allowed."
            )
        if tbl.db and tbl_name not in allowed:
            raise SQLValidationError(
                f"Table or view '{tbl.name}' is not in the allowed tables list."
            )


# ==============================================================================
# Result Serialization and Sanitization Helpers
# ==============================================================================

def _serialize_cell_value(val: Any) -> Any:
    """Convert DuckDB cell value to a strictly JSON-compliant representation."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return val
    if isinstance(val, decimal.Decimal):
        return str(val)
    if isinstance(val, (datetime.datetime, datetime.date, datetime.time)):
        return val.isoformat()
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, (bytes, bytearray)):
        return val.hex()
    if isinstance(val, (list, tuple)):
        return [_serialize_cell_value(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _serialize_cell_value(v) for k, v in val.items()}
    return str(val)


def _sanitize_error_message(msg: str, target_path: Optional[Path] = None) -> str:
    """Sanitize error messages to avoid exposing infrastructure, paths, or secrets."""
    if not msg:
        return ""
    if target_path is not None:
        msg = msg.replace(str(target_path), "<database_file>")
        try:
            msg = msg.replace(str(target_path.resolve()), "<database_file>")
        except Exception:
            pass
    # Strip local path prefixes
    msg = re.sub(r"/(?:Users|home|private|tmp|var)/[^\s\"':;]+", "<path_redacted>", msg)
    # Strip DuckDB internal source code line mentions
    msg = re.sub(r"at line \d+ of [^\s]+", "", msg)
    return msg.strip()


# ==============================================================================
# Isolated Worker Process for Hard Timeout Enforcement
# ==============================================================================

def _read_exact(fd: int, n: int, deadline: float) -> bytes:
    """Read exact n bytes from fd bounded by monotonic deadline using selectors."""
    buf = bytearray()
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    try:
        while len(buf) < n:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError("Timed out reading data from worker pipe.")
            events = sel.select(timeout=max(0.0, remaining))
            if not events:
                raise TimeoutError("Timed out waiting for data from worker pipe.")
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                raise EOFError("Worker pipe closed before expected bytes were received.")
            buf.extend(chunk)
        return bytes(buf)
    finally:
        sel.close()


def _recv_framed_payload(
    fd: int,
    deadline: float,
    max_bytes: int = MAX_SERIALIZED_PAYLOAD_BYTES,
) -> Any:
    """Receive a length-prefixed, pickle-serialized payload with deadline and size bounds."""
    header_bytes = _read_exact(fd, 4, deadline)
    (length,) = struct.unpack("!I", header_bytes)
    if length > max_bytes:
        raise ValueError(
            f"Serialized query result size ({length} bytes) exceeds the maximum allowed limit of {max_bytes} bytes."
        )
    payload_bytes = _read_exact(fd, length, deadline)
    return pickle.loads(payload_bytes)


def _send_framed_payload(
    conn: Any,
    payload: Any,
    max_bytes: int = MAX_SERIALIZED_PAYLOAD_BYTES,
) -> None:
    """Serialize and send length-prefixed payload over connection fd."""
    try:
        raw_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ser_err:
        err_payload = ("error", type(ser_err).__name__, f"Failed to serialize result: {ser_err}")
        raw_bytes = pickle.dumps(err_payload, protocol=pickle.HIGHEST_PROTOCOL)

    if len(raw_bytes) > max_bytes:
        err_payload = (
            "error",
            "PayloadSizeError",
            f"Serialized query result size ({len(raw_bytes)} bytes) exceeds the maximum allowed limit of {max_bytes} bytes.",
        )
        raw_bytes = pickle.dumps(err_payload, protocol=pickle.HIGHEST_PROTOCOL)

    fd = conn.fileno()
    header = struct.pack("!I", len(raw_bytes))
    full_data = header + raw_bytes
    total_sent = 0
    while total_sent < len(full_data):
        sent = os.write(fd, full_data[total_sent:])
        if sent == 0:
            raise BrokenPipeError("Worker pipe closed while writing.")
        total_sent += sent


def _worker_query_runner(
    db_path_str: str,
    query: str,
    max_rows: int,
    conn: Any,
) -> None:
    """Worker process target: runs query with sandbox settings, sends framed result via Pipe.

    Note on memory limits: The configured max_memory (512MB) instructs DuckDB's
    internal query buffer manager, but does not represent a strict OS-level cgroup
    limit on the total RSS of the worker process.
    """
    config = {
        "enable_external_access": "false",
        "max_memory": "512MB",
        "threads": "2",
        "lock_configuration": "true",
    }
    db_con = None
    try:
        db_con = duckdb.connect(database=db_path_str, read_only=True, config=config)
        cursor = db_con.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        raw_rows = cursor.fetchmany(max_rows + 1)
        truncated = len(raw_rows) > max_rows
        rows_to_return = raw_rows[:max_rows]
        converted_rows = [
            [_serialize_cell_value(val) for val in row]
            for row in rows_to_return
        ]
        _send_framed_payload(
            conn,
            ("success", columns, converted_rows, len(converted_rows), truncated),
        )
    except Exception as exc:
        try:
            _send_framed_payload(conn, ("error", type(exc).__name__, str(exc)))
        except Exception:
            pass
    finally:
        if db_con is not None:
            try:
                db_con.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


def _terminate_worker(worker: Optional[multiprocessing.Process]) -> None:
    """Terminate and join a worker process to prevent zombie processes."""
    if worker is None:
        return
    try:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=0.2)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=0.2)
    except Exception as exc:
        logger.warning("Error during worker process cleanup: %s", exc)


# Module-level factory and runner target to facilitate deterministic test injection in spawn mode
_PROCESS_FACTORY = multiprocessing.get_context("spawn").Process
_WORKER_TARGET = _worker_query_runner


# ==============================================================================
# Safe SQL Execution Interface
# ==============================================================================

def execute_sql(
    query: str,
    db_path: Optional[Union[str, Path]] = None,
    max_rows: int = 1000,
    timeout_seconds: float = 10.0,
    allowed_tables: Optional[Collection[str]] = None,
) -> Dict[str, Any]:
    """Safely validate and execute a read-only SQL query against DuckDB.

    Enforces AST validation, DuckDB sandboxing (external access disabled, locked config),
    memory/thread caps, application-level row limits (with truncation flag), and hard
    process-level timeout enforcement via a framed deadlock-free Pipe IPC.

    Args:
        query: Analytical SQL query string.
        db_path: Path to DuckDB database file. Defaults to DEFAULT_DB_PATH.
        max_rows: Maximum rows to return (default 1000, capped at 5000).
        timeout_seconds: Maximum execution time in seconds (default 10.0, capped at 30.0).
        allowed_tables: Optional override for permitted table names. If None,
            dynamically determined from target database.

    Returns:
        Dict adhering to structured format:
            Success: {"status": "success", "columns": [...], "rows": [...],
                      "row_count": int, "truncated": bool, "execution_time_ms": float}
            Error:   {"status": "error", "error_type": str,
                      "error_message": str, "execution_time_ms": float}
    """
    start_time = time.perf_counter()

    # Validate configurable limits
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": "ValueError",
            "error_message": f"max_rows must be a positive integer, got: {max_rows}",
            "execution_time_ms": elapsed,
        }

    if max_rows > MAX_PERMITTED_ROWS:
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": "ValueError",
            "error_message": f"max_rows cannot exceed {MAX_PERMITTED_ROWS}, got: {max_rows}",
            "execution_time_ms": elapsed,
        }

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or math.isnan(timeout_seconds)
        or math.isinf(timeout_seconds)
        or timeout_seconds <= 0
    ):
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": "ValueError",
            "error_message": f"timeout_seconds must be a positive finite number, got: {timeout_seconds}",
            "execution_time_ms": elapsed,
        }

    if timeout_seconds > MAX_PERMITTED_TIMEOUT_SECONDS:
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": "ValueError",
            "error_message": f"timeout_seconds cannot exceed {MAX_PERMITTED_TIMEOUT_SECONDS}, got: {timeout_seconds}",
            "execution_time_ms": elapsed,
        }

    # Bounded end-to-end deadline initialized immediately after argument validation
    deadline = time.monotonic() + timeout_seconds

    target_path = Path(db_path) if db_path is not None else Path(DEFAULT_DB_PATH)
    resolved_path = target_path.resolve()

    if not resolved_path.exists():
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": "FileNotFoundError",
            "error_message": "DuckDB database file not found at: <database_file>",
            "execution_time_ms": elapsed,
        }

    if resolved_path.is_dir():
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": "ValueError",
            "error_message": "Provided database path is a directory, not a DuckDB file.",
            "execution_time_ms": elapsed,
        }

    # Determine allowed tables (fail-closed: never fall back if custom discovery fails)
    if allowed_tables is None:
        if resolved_path == Path(DEFAULT_DB_PATH).resolve():
            allowed = _DEFAULT_ALLOWED_TABLES
        else:
            try:
                schema_info = get_database_schema(resolved_path)
                allowed = frozenset(
                    [t["name"].lower() for t in schema_info["tables"] if t.get("schema") == _APP_SCHEMA]
                    + [v["name"].lower() for v in schema_info["views"] if v.get("schema") == _APP_SCHEMA]
                )
            except Exception as disc_err:
                elapsed = round((time.perf_counter() - start_time) * 1000, 2)
                sanitized_msg = _sanitize_error_message(f"Schema discovery failed: {disc_err}", resolved_path)
                return {
                    "status": "error",
                    "error_type": type(disc_err).__name__,
                    "error_message": sanitized_msg,
                    "execution_time_ms": elapsed,
                }
    else:
        allowed = frozenset(t.lower() for t in allowed_tables)

    # 1. AST Validation
    try:
        validate_sql(query, allowed_tables=allowed)
    except SQLValidationError as val_err:
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": "SQLValidationError",
            "error_message": str(val_err),
            "execution_time_ms": elapsed,
        }

    # Check whether timeout already expired during preparation and validation
    if time.monotonic() >= deadline:
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": "TimeoutError",
            "error_message": f"Query execution exceeded timeout limit of {timeout_seconds} seconds before worker start.",
            "execution_time_ms": elapsed,
        }

    # 2. Execution via isolated worker process with framed Pipe IPC
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    worker = None
    try:
        worker = _PROCESS_FACTORY(
            target=_WORKER_TARGET,
            args=(str(resolved_path), query, max_rows, child_conn),
        )
        worker.start()
    except Exception as start_err:
        try:
            parent_conn.close()
        except Exception:
            pass
        try:
            child_conn.close()
        except Exception:
            pass
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "error",
            "error_type": type(start_err).__name__,
            "error_message": f"Failed to start worker process: {_sanitize_error_message(str(start_err))}",
            "execution_time_ms": elapsed,
        }
    finally:
        # Parent closes its reference to child_conn immediately so EOF can trigger on worker exit
        try:
            child_conn.close()
        except Exception:
            pass

    payload = None
    ipc_error: Optional[Tuple[str, str]] = None
    timeout_occurred = False

    try:
        payload = _recv_framed_payload(parent_conn.fileno(), deadline=deadline)
    except TimeoutError:
        timeout_occurred = True
    except (EOFError, BrokenPipeError) as eof_err:
        ipc_error = ("ProcessError", f"Worker process terminated unexpectedly before completing communication: {eof_err}")
    except ValueError as val_err:
        ipc_error = ("PayloadSizeError", str(val_err))
    except Exception as exc:
        ipc_error = ("IPCError", f"IPC communication error during query execution: {exc}")

    # Bounded worker process join and cleanup
    worker_timed_out = False
    worker_exitcode: Optional[int] = None
    try:
        join_remaining = max(0.01, min(1.0, deadline - time.monotonic())) if not timeout_occurred else 0.01
        worker.join(timeout=join_remaining)
        if worker.is_alive():
            _terminate_worker(worker)
            # If payload was received but worker hung during cleanup, classify as timeout failure
            if not timeout_occurred and payload is not None:
                worker_timed_out = True
        worker_exitcode = worker.exitcode
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        if worker is not None:
            try:
                worker.close()
            except Exception:
                pass

    elapsed = round((time.perf_counter() - start_time) * 1000, 2)

    # 1. Timeout takes precedence
    if timeout_occurred or worker_timed_out:
        return {
            "status": "error",
            "error_type": "TimeoutError",
            "error_message": f"Query execution exceeded timeout limit of {timeout_seconds} seconds.",
            "execution_time_ms": elapsed,
        }

    # 2. Worker abnormal exit takes precedence over payload!
    # Even if payload was received, non-zero exitcode indicates abnormal process failure.
    if worker_exitcode != 0:
        return {
            "status": "error",
            "error_type": "ProcessError",
            "error_message": f"Worker process terminated abnormally with exit code {worker_exitcode}.",
            "execution_time_ms": elapsed,
        }

    # 3. IPC errors
    if ipc_error is not None:
        err_type, err_msg = ipc_error
        return {
            "status": "error",
            "error_type": err_type,
            "error_message": err_msg,
            "execution_time_ms": elapsed,
        }

    # 4. Missing payload
    if payload is None:
        return {
            "status": "error",
            "error_type": "ProcessError",
            "error_message": f"Worker process terminated with exit code {worker_exitcode} without sending a result.",
            "execution_time_ms": elapsed,
        }

    # 5. Normal result dispatch
    msg_type, *content = payload
    if msg_type == "success":
        cols, rows, count, trunc = content
        return {
            "status": "success",
            "columns": cols,
            "rows": rows,
            "row_count": count,
            "truncated": trunc,
            "execution_time_ms": elapsed,
        }
    else:
        err_type, raw_err_msg = content
        sanitized = _sanitize_error_message(raw_err_msg, resolved_path)
        return {
            "status": "error",
            "error_type": err_type,
            "error_message": sanitized,
            "execution_time_ms": elapsed,
        }
