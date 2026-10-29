"""Reproducible data preparation pipeline for the Olist e-commerce dataset into DuckDB.

This module:
1. Ingests all 9 Olist CSV files into a persistent DuckDB database (data/ecommerce.duckdb).
2. Casts timestamp columns to proper SQL TIMESTAMP types, monetary amounts to DECIMAL(18,2),
   and identifiers to VARCHAR.
3. Preserves original data faithfully without dropping rows or silently imputing values.
4. Builds two analysis-ready views:
   - order_summary: Granularity of 1 row per order, avoiding join fanout.
   - category_sales_summary: Monthly category sales for time-series and category BI.
5. Performs and logs data quality checks (primary keys, foreign keys, duplicates, nulls).
6. Separates critical validation failures (which raise CriticalValidationError and trigger
   transaction rollback) from informational source-data anomalies.
7. Executes base table imports, view creation, and validation within an atomic transaction.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_data")

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "ecommerce.duckdb"


class CriticalValidationError(Exception):
    """Raised when one or more critical data integrity checks fail."""

    def __init__(self, failures: List[str]) -> None:
        self.failures = failures
        bullet_list = "\n".join(f"  - {f}" for f in failures)
        super().__init__(f"Critical validation failures encountered ({len(failures)}):\n{bullet_list}")


# Schema definitions mapping logical table names to raw CSVs and explicit SQL data types.
# Monetary amounts (price, freight_value, payment_value) use DECIMAL(18,2) for exact precision.
TABLE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "customers": {
        "file": "olist_customers_dataset.csv",
        "primary_key": ["customer_id"],
        "columns": {
            "customer_id": "VARCHAR",
            "customer_unique_id": "VARCHAR",
            "customer_zip_code_prefix": "VARCHAR",
            "customer_city": "VARCHAR",
            "customer_state": "VARCHAR",
        },
    },
    "geolocation": {
        "file": "olist_geolocation_dataset.csv",
        "primary_key": None,  # Contains multiple entries and exact duplicates per zip code
        "columns": {
            "geolocation_zip_code_prefix": "VARCHAR",
            "geolocation_lat": "DOUBLE",
            "geolocation_lng": "DOUBLE",
            "geolocation_city": "VARCHAR",
            "geolocation_state": "VARCHAR",
        },
    },
    "order_items": {
        "file": "olist_order_items_dataset.csv",
        "primary_key": ["order_id", "order_item_id"],
        "columns": {
            "order_id": "VARCHAR",
            "order_item_id": "INTEGER",
            "product_id": "VARCHAR",
            "seller_id": "VARCHAR",
            "shipping_limit_date": "TIMESTAMP",
            "price": "DECIMAL(18,2)",
            "freight_value": "DECIMAL(18,2)",
        },
    },
    "order_payments": {
        "file": "olist_order_payments_dataset.csv",
        "primary_key": ["order_id", "payment_sequential"],
        "columns": {
            "order_id": "VARCHAR",
            "payment_sequential": "INTEGER",
            "payment_type": "VARCHAR",
            "payment_installments": "INTEGER",
            "payment_value": "DECIMAL(18,2)",
        },
    },
    "order_reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "primary_key": ["review_id", "order_id"],
        "columns": {
            "review_id": "VARCHAR",
            "order_id": "VARCHAR",
            "review_score": "INTEGER",
            "review_comment_title": "VARCHAR",
            "review_comment_message": "VARCHAR",
            "review_creation_date": "TIMESTAMP",
            "review_answer_timestamp": "TIMESTAMP",
        },
    },
    "orders": {
        "file": "olist_orders_dataset.csv",
        "primary_key": ["order_id"],
        "columns": {
            "order_id": "VARCHAR",
            "customer_id": "VARCHAR",
            "order_status": "VARCHAR",
            "order_purchase_timestamp": "TIMESTAMP",
            "order_approved_at": "TIMESTAMP",
            "order_delivered_carrier_date": "TIMESTAMP",
            "order_delivered_customer_date": "TIMESTAMP",
            "order_estimated_delivery_date": "TIMESTAMP",
        },
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "primary_key": ["product_id"],
        "columns": {
            "product_id": "VARCHAR",
            "product_category_name": "VARCHAR",
            "product_name_lenght": "INTEGER",
            "product_description_lenght": "INTEGER",
            "product_photos_qty": "INTEGER",
            "product_weight_g": "INTEGER",
            "product_length_cm": "INTEGER",
            "product_height_cm": "INTEGER",
            "product_width_cm": "INTEGER",
        },
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "primary_key": ["seller_id"],
        "columns": {
            "seller_id": "VARCHAR",
            "seller_zip_code_prefix": "VARCHAR",
            "seller_city": "VARCHAR",
            "seller_state": "VARCHAR",
        },
    },
    "category_translation": {
        "file": "product_category_name_translation.csv",
        "primary_key": ["product_category_name"],
        "columns": {
            "product_category_name": "VARCHAR",
            "product_category_name_english": "VARCHAR",
        },
    },
}

VIEW_DEFINITIONS: Dict[str, str] = {
    "order_summary": """
CREATE OR REPLACE VIEW order_summary AS
WITH items_agg AS (
    SELECT 
        order_id,
        count(order_item_id)::INTEGER AS item_count,
        count(DISTINCT product_id)::INTEGER AS unique_products_count,
        count(DISTINCT seller_id)::INTEGER AS unique_sellers_count,
        coalesce(sum(price), 0.00)::DECIMAL(18,2) AS item_subtotal,
        coalesce(sum(freight_value), 0.00)::DECIMAL(18,2) AS freight_total,
        coalesce(sum(price + freight_value), 0.00)::DECIMAL(18,2) AS order_gross_value
    FROM order_items
    GROUP BY order_id
),
payments_agg AS (
    SELECT 
        order_id,
        count(payment_sequential)::INTEGER AS payment_count,
        coalesce(sum(payment_value), 0.00)::DECIMAL(18,2) AS payment_value_total,
        string_agg(DISTINCT payment_type, ', ' ORDER BY payment_type) AS payment_types,
        max(payment_installments)::INTEGER AS max_installments
    FROM order_payments
    GROUP BY order_id
),
reviews_agg AS (
    SELECT 
        order_id,
        count(review_id)::INTEGER AS review_count,
        round(avg(review_score), 2) AS avg_review_score
    FROM order_reviews
    GROUP BY order_id
)
SELECT 
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    c.customer_zip_code_prefix,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    coalesce(i.item_count, 0) AS item_count,
    coalesce(i.unique_products_count, 0) AS unique_products_count,
    coalesce(i.unique_sellers_count, 0) AS unique_sellers_count,
    coalesce(i.item_subtotal, 0.00)::DECIMAL(18,2) AS item_subtotal,
    coalesce(i.freight_total, 0.00)::DECIMAL(18,2) AS freight_total,
    coalesce(i.order_gross_value, 0.00)::DECIMAL(18,2) AS order_gross_value,
    coalesce(p.payment_count, 0) AS payment_count,
    coalesce(p.payment_value_total, 0.00)::DECIMAL(18,2) AS payment_value_total,
    p.payment_types,
    p.max_installments,
    coalesce(r.review_count, 0) AS review_count,
    r.avg_review_score,
    (o.order_status = 'delivered') AS is_delivered,
    CASE 
        WHEN o.order_delivered_customer_date IS NOT NULL 
        THEN round(epoch(o.order_delivered_customer_date - o.order_purchase_timestamp) / 86400.0, 2)
        ELSE NULL 
    END AS delivery_duration_days
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN items_agg i ON o.order_id = i.order_id
LEFT JOIN payments_agg p ON o.order_id = p.order_id
LEFT JOIN reviews_agg r ON o.order_id = r.order_id;
""",
    "category_sales_summary": """
CREATE OR REPLACE VIEW category_sales_summary AS
SELECT 
    strftime(o.order_purchase_timestamp, '%Y-%m') AS year_month,
    extract(year FROM o.order_purchase_timestamp)::INTEGER AS order_year,
    extract(month FROM o.order_purchase_timestamp)::INTEGER AS order_month,
    coalesce(p.product_category_name, 'uncategorized') AS product_category_name,
    coalesce(t.product_category_name_english, coalesce(p.product_category_name, 'uncategorized')) AS category_name_english,
    count(DISTINCT o.order_id)::INTEGER AS total_orders,
    count(oi.order_item_id)::INTEGER AS total_items_sold,
    coalesce(sum(oi.price), 0.00)::DECIMAL(18,2) AS total_item_revenue,
    coalesce(sum(oi.freight_value), 0.00)::DECIMAL(18,2) AS total_freight_value,
    coalesce(sum(oi.price + oi.freight_value), 0.00)::DECIMAL(18,2) AS total_gross_sales,
    count(DISTINCT CASE WHEN o.order_status = 'delivered' THEN o.order_id END)::INTEGER AS delivered_orders,
    count(CASE WHEN o.order_status = 'delivered' THEN oi.order_item_id END)::INTEGER AS delivered_items_sold,
    coalesce(sum(CASE WHEN o.order_status = 'delivered' THEN oi.price END), 0.00)::DECIMAL(18,2) AS delivered_item_revenue,
    round(sum(oi.price) / count(oi.order_item_id), 2)::DECIMAL(18,2) AS avg_item_price,
    count(DISTINCT oi.seller_id)::INTEGER AS active_sellers
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
GROUP BY 1, 2, 3, 4, 5;
""",
}


def verify_raw_files(raw_dir: Path) -> None:
    """Ensure all 9 source CSV files exist in the specified raw directory."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    missing_files = []
    for meta in TABLE_SCHEMAS.values():
        file_path = raw_dir / meta["file"]
        if not file_path.exists():
            missing_files.append(meta["file"])

    if missing_files:
        raise FileNotFoundError(f"Missing required CSV files in {raw_dir}: {missing_files}")
    logger.info("All 9 raw CSV files verified successfully in %s", raw_dir)


def import_base_tables(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Import CSVs into DuckDB with explicit column data types."""
    schemas = table_schemas if table_schemas is not None else TABLE_SCHEMAS
    logger.info("Starting base tables ingestion...")
    table_counts: Dict[str, int] = {}

    for table_name, meta in schemas.items():
        file_path = (raw_dir / meta["file"]).resolve()
        columns_spec = ", ".join(
            [f"'{col}': '{dtype}'" for col, dtype in meta["columns"].items()]
        )

        sql = f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM read_csv(
                '{file_path.as_posix()}',
                header=True,
                columns={{{columns_spec}}}
            );
        """
        con.execute(sql)
        row_count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        table_counts[table_name] = row_count
        logger.info(
            "Created table '%s' with %d rows (source: %s)",
            table_name,
            row_count,
            meta["file"],
        )

    return table_counts


def create_analysis_views(
    con: duckdb.DuckDBPyConnection,
    view_definitions: Optional[Dict[str, str]] = None,
) -> None:
    """Create the analysis-ready views: order_summary and category_sales_summary."""
    views = view_definitions if view_definitions is not None else VIEW_DEFINITIONS
    logger.info("Creating analysis-ready views...")
    for view_name, ddl in views.items():
        con.execute(ddl)
        cnt = con.execute(f"SELECT count(*) FROM {view_name}").fetchone()[0]
        logger.info("Created view '%s' with %d rows", view_name, cnt)


DEFAULT_FK_CHECKS: List[Tuple[str, str]] = [
    ("orders.customer_id -> customers.customer_id",
     "SELECT count(*) FROM orders o LEFT JOIN customers c ON o.customer_id = c.customer_id WHERE c.customer_id IS NULL"),
    ("order_items.order_id -> orders.order_id",
     "SELECT count(*) FROM order_items oi LEFT JOIN orders o ON oi.order_id = o.order_id WHERE o.order_id IS NULL"),
    ("order_items.product_id -> products.product_id",
     "SELECT count(*) FROM order_items oi LEFT JOIN products p ON oi.product_id = p.product_id WHERE p.product_id IS NULL"),
    ("order_items.seller_id -> sellers.seller_id",
     "SELECT count(*) FROM order_items oi LEFT JOIN sellers s ON oi.seller_id = s.seller_id WHERE s.seller_id IS NULL"),
    ("order_payments.order_id -> orders.order_id",
     "SELECT count(*) FROM order_payments op LEFT JOIN orders o ON op.order_id = o.order_id WHERE o.order_id IS NULL"),
    ("order_reviews.order_id -> orders.order_id",
     "SELECT count(*) FROM order_reviews r LEFT JOIN orders o ON r.order_id = o.order_id WHERE o.order_id IS NULL"),
]


def validate_source_row_counts(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Compare CSV-aware logical record counts against imported base tables."""
    schemas = table_schemas if table_schemas is not None else TABLE_SCHEMAS
    results: Dict[str, Any] = {}
    failures: List[str] = []

    for table_name, meta in schemas.items():
        file_path = (raw_dir / meta["file"]).resolve()
        csv_count = con.execute(
            f"SELECT count(*) FROM read_csv('{file_path.as_posix()}', header=True)"
        ).fetchone()[0]
        db_count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]

        matches = csv_count == db_count
        results[table_name] = {
            "source_csv_rows": csv_count,
            "imported_db_rows": db_count,
            "matches": matches,
        }
        if not matches:
            failures.append(
                f"Row count mismatch in '{table_name}': source CSV={csv_count}, database={db_count}"
            )

    return results, failures


def validate_primary_keys(
    con: duckdb.DuckDBPyConnection,
    table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Check candidate primary key non-nullness and uniqueness using SQL GROUP BY."""
    schemas = table_schemas if table_schemas is not None else TABLE_SCHEMAS
    results: Dict[str, Any] = {}
    failures: List[str] = []

    for table_name, meta in schemas.items():
        pk = meta["primary_key"]
        if pk is None:
            continue

        cols_sql = ", ".join([f'"{c}"' for c in pk])
        null_condition = " OR ".join([f'"{c}" IS NULL' for c in pk])

        # 1. Non-nullness check
        null_count = con.execute(
            f'SELECT count(*) FROM {table_name} WHERE {null_condition}'
        ).fetchone()[0]

        # 2. Multi-column uniqueness check using GROUP BY (prevents string concatenation collisions)
        dup_row = con.execute(f"""
            SELECT 
                count(*) AS duplicate_groups,
                coalesce(sum(cnt - 1), 0)::BIGINT AS excess_duplicate_rows
            FROM (
                SELECT {cols_sql}, count(*) AS cnt
                FROM {table_name}
                GROUP BY {cols_sql}
                HAVING count(*) > 1
            )
        """).fetchone()

        dup_groups = dup_row[0]
        excess_dups = dup_row[1]
        is_unique = dup_groups == 0
        has_no_nulls = null_count == 0

        results[table_name] = {
            "primary_key": pk,
            "null_count": null_count,
            "duplicate_groups": dup_groups,
            "excess_duplicate_rows": excess_dups,
            "is_unique": is_unique,
            "has_no_nulls": has_no_nulls,
        }

        if not has_no_nulls:
            failures.append(f"Table '{table_name}' has {null_count} rows with NULL in primary key {pk}")
        if not is_unique:
            failures.append(
                f"Table '{table_name}' has {dup_groups} duplicate primary key groups ({excess_dups} duplicate rows) for {pk}"
            )

    return results, failures


def validate_foreign_keys(
    con: duckdb.DuckDBPyConnection,
    fk_checks: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Verify referential integrity across related tables."""
    checks = fk_checks if fk_checks is not None else DEFAULT_FK_CHECKS
    results: Dict[str, Any] = {}
    failures: List[str] = []

    for label, query in checks:
        orphan_count = con.execute(query).fetchone()[0]
        is_valid = orphan_count == 0
        results[label] = {
            "orphan_count": orphan_count,
            "is_valid": is_valid,
        }
        if not is_valid:
            failures.append(f"Foreign key violation: {label} has {orphan_count} orphan records")

    return results, failures


def validate_views_and_metrics(
    con: duckdb.DuckDBPyConnection,
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate view granularity, anti-fanout properties, and financial reconciliation."""
    results: Dict[str, Any] = {}
    failures: List[str] = []

    # 1. order_summary grain validation
    orders_count = con.execute("SELECT count(*) FROM orders").fetchone()[0]
    os_count = con.execute("SELECT count(*) FROM order_summary").fetchone()[0]
    os_distinct_orders = con.execute("SELECT count(DISTINCT order_id) FROM order_summary").fetchone()[0]

    if os_count != orders_count:
        failures.append(f"order_summary row count ({os_count}) does not match orders table ({orders_count})")
    if os_distinct_orders != os_count:
        failures.append(f"order_summary violates 1-row-per-order grain: {os_distinct_orders} distinct vs {os_count} rows")

    # 2. Financial totals reconciliation (Exact decimal match)
    raw_item_sum = con.execute("SELECT sum(price) FROM order_items").fetchone()[0]
    raw_freight_sum = con.execute("SELECT sum(freight_value) FROM order_items").fetchone()[0]
    raw_payment_sum = con.execute("SELECT sum(payment_value) FROM order_payments").fetchone()[0]

    os_item_sum = con.execute("SELECT sum(item_subtotal) FROM order_summary").fetchone()[0]
    os_freight_sum = con.execute("SELECT sum(freight_total) FROM order_summary").fetchone()[0]
    os_payment_sum = con.execute("SELECT sum(payment_value_total) FROM order_summary").fetchone()[0]

    cat_item_sum = con.execute("SELECT sum(total_item_revenue) FROM category_sales_summary").fetchone()[0]
    cat_freight_sum = con.execute("SELECT sum(total_freight_value) FROM category_sales_summary").fetchone()[0]

    if os_item_sum != raw_item_sum:
        failures.append(f"Financial mismatch: order_summary item subtotal ({os_item_sum}) != order_items ({raw_item_sum})")
    if os_freight_sum != raw_freight_sum:
        failures.append(f"Financial mismatch: order_summary freight total ({os_freight_sum}) != order_items ({raw_freight_sum})")
    if os_payment_sum != raw_payment_sum:
        failures.append(f"Financial mismatch: order_summary payment sum ({os_payment_sum}) != order_payments ({raw_payment_sum})")
    if cat_item_sum != raw_item_sum:
        failures.append(f"Financial mismatch: category_sales_summary revenue ({cat_item_sum}) != order_items ({raw_item_sum})")
    if cat_freight_sum != raw_freight_sum:
        failures.append(f"Financial mismatch: category_sales_summary freight ({cat_freight_sum}) != order_items ({raw_freight_sum})")

    # 3. Item count preservation (verifying zero lost or multiplied order-item rows)
    raw_item_count = con.execute("SELECT count(*) FROM order_items").fetchone()[0]
    os_item_count = con.execute("SELECT sum(item_count) FROM order_summary").fetchone()[0]
    cat_item_count = con.execute("SELECT sum(total_items_sold) FROM category_sales_summary").fetchone()[0]

    if os_item_count != raw_item_count:
        failures.append(f"Item count mismatch: order_summary sum(item_count)={os_item_count} != order_items={raw_item_count}")
    if cat_item_count != raw_item_count:
        failures.append(f"Item count mismatch: category_sales_summary sum(total_items_sold)={cat_item_count} != order_items={raw_item_count}")

    results["order_summary_row_count"] = os_count
    results["category_sales_summary_row_count"] = con.execute("SELECT count(*) FROM category_sales_summary").fetchone()[0]
    results["financial_reconciliation_exact"] = len(failures) == 0

    return results, failures


def collect_informational_anomalies(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Report known source dataset anomalies (preserved in base data without failing the pipeline)."""
    existing_tables = set(
        r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    )
    anomalies: Dict[str, Any] = {}

    if "geolocation" in existing_tables:
        anomalies["geolocation_exact_duplicate_rows"] = con.execute("""
            SELECT count(*) - count(DISTINCT geolocation_zip_code_prefix || '_' || geolocation_lat::VARCHAR || '_' || geolocation_lng::VARCHAR || '_' || geolocation_city || '_' || geolocation_state)
            FROM geolocation
        """).fetchone()[0]

    if "products" in existing_tables and "category_translation" in existing_tables:
        anomalies["product_categories_missing_english_translation"] = con.execute("""
            SELECT count(DISTINCT p.product_category_name)
            FROM products p
            LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
            WHERE p.product_category_name IS NOT NULL AND t.product_category_name IS NULL
        """).fetchone()[0]

    if "orders" in existing_tables and "order_items" in existing_tables:
        anomalies["orders_without_order_items"] = con.execute("""
            SELECT count(*)
            FROM orders o
            LEFT JOIN (SELECT DISTINCT order_id FROM order_items) oi ON o.order_id = oi.order_id
            WHERE oi.order_id IS NULL
        """).fetchone()[0]

    if "orders" in existing_tables and "order_payments" in existing_tables:
        anomalies["orders_without_payments"] = con.execute("""
            SELECT count(*)
            FROM orders o
            LEFT JOIN (SELECT DISTINCT order_id FROM order_payments) op ON o.order_id = op.order_id
            WHERE op.order_id IS NULL
        """).fetchone()[0]

    if "orders" in existing_tables:
        has_delivery_col = con.execute("""
            SELECT count(*) FROM information_schema.columns 
            WHERE table_name = 'orders' AND column_name = 'order_delivered_customer_date'
        """).fetchone()[0] > 0
        if has_delivery_col:
            anomalies["orders_with_null_delivered_date"] = con.execute("""
                SELECT count(*) FROM orders WHERE order_delivered_customer_date IS NULL
            """).fetchone()[0]

    return anomalies


def run_data_validation(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    fk_checks: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Execute comprehensive validation suite and enforce critical data-quality checks."""
    logger.info("Running data validation checks...")
    all_critical_failures: List[str] = []

    # 1. Source row count reconciliation
    row_counts, count_failures = validate_source_row_counts(con, raw_dir, table_schemas=table_schemas)
    all_critical_failures.extend(count_failures)

    # 2. Candidate primary key validation (non-nullness & group-by uniqueness)
    pk_results, pk_failures = validate_primary_keys(con, table_schemas=table_schemas)
    all_critical_failures.extend(pk_failures)

    # 3. Foreign key referential integrity
    fk_results, fk_failures = validate_foreign_keys(con, fk_checks=fk_checks)
    all_critical_failures.extend(fk_failures)

    # 4. Analysis views & financial metric reconciliation
    existing_tables = set(
        r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    )
    if "orders" in existing_tables and "order_summary" in existing_tables and "category_sales_summary" in existing_tables:
        view_results, view_failures = validate_views_and_metrics(con)
        all_critical_failures.extend(view_failures)
    else:
        view_results = {}

    # 5. Informational anomalies (non-fatal)
    anomalies = collect_informational_anomalies(con)

    report = {
        "row_counts": row_counts,
        "primary_keys": pk_results,
        "foreign_keys": fk_results,
        "view_integrity": view_results,
        "informational_anomalies": anomalies,
        "critical_failures": all_critical_failures,
    }

    if all_critical_failures:
        for f in all_critical_failures:
            logger.error("CRITICAL CHECK FAILED: %s", f)
        raise CriticalValidationError(all_critical_failures)

    return report


def log_validation_summary(report: Dict[str, Any]) -> None:
    """Format and print an executive summary of validation results."""
    logger.info("================ DATA VALIDATION REPORT ================")
    logger.info("1. SOURCE-TO-DATABASE ROW COUNTS:")
    for tbl, info in report["row_counts"].items():
        logger.info(
            "   - %-22s: CSV=%d, DB=%d -> %s",
            tbl,
            info["source_csv_rows"],
            info["imported_db_rows"],
            "PASSED" if info["matches"] else "FAILED",
        )

    logger.info("2. PRIMARY KEY INTEGRITY (NON-NULL & MULTI-COLUMN UNIQUE):")
    for tbl, info in report["primary_keys"].items():
        status = "PASSED (Unique, Non-null)" if info["is_unique"] and info["has_no_nulls"] else "FAILED"
        logger.info("   - %-22s: key=%s -> %s (nulls=%d, dup_groups=%d)", tbl, info["primary_key"], status, info["null_count"], info["duplicate_groups"])

    logger.info("3. REFERENTIAL INTEGRITY (FOREIGN KEYS):")
    for fk, info in report["foreign_keys"].items():
        status = "PASSED (0 orphans)" if info["is_valid"] else f"FAILED ({info['orphan_count']} orphans)"
        logger.info("   - %-45s: %s", fk, status)

    logger.info("4. INFORMATIONAL SOURCE-DATA ANOMALIES (PRESERVED IN BASE DATA):")
    for anomaly, count in report["informational_anomalies"].items():
        logger.info("   - %-45s: %d", anomaly, count)

    logger.info("5. VIEW AGGREGATION INTEGRITY & ANTI-FANOUT:")
    for metric, val in report["view_integrity"].items():
        logger.info("   - %-45s: %s", metric, val)
    logger.info("========================================================")


def build_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    table_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    view_definitions: Optional[Dict[str, str]] = None,
    fk_checks: Optional[List[Tuple[str, str]]] = None,
) -> duckdb.DuckDBPyConnection:
    """Run the complete end-to-end data preparation workflow inside an atomic transaction."""
    logger.info("Starting Olist data preparation pipeline...")
    logger.info("Source directory: %s", raw_dir)
    logger.info("Target DuckDB:    %s", db_path)

    schemas = table_schemas if table_schemas is not None else TABLE_SCHEMAS
    views = view_definitions if view_definitions is not None else VIEW_DEFINITIONS

    # Verify source files if using default schemas
    if table_schemas is None:
        verify_raw_files(raw_dir)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=str(db_path), read_only=False)

    try:
        con.execute("BEGIN TRANSACTION")
        import_base_tables(con, raw_dir, table_schemas=schemas)
        create_analysis_views(con, view_definitions=views)
        report = run_data_validation(
            con, raw_dir, table_schemas=schemas, fk_checks=fk_checks
        )
        con.execute("COMMIT")
        log_validation_summary(report)
        logger.info("Data preparation committed successfully. Database written to %s", db_path)
        return con
    except Exception as e:
        logger.error("Transaction rolled back due to error: %s", e)
        try:
            con.execute("ROLLBACK")
        except Exception as rb_err:
            logger.warning("Rollback error (transaction may already be aborted): %s", rb_err)
        con.close()
        raise


if __name__ == "__main__":
    try:
        con = build_pipeline()
        con.close()
    except CriticalValidationError as e:
        logger.error("Pipeline terminated due to critical integrity failure: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline terminated unexpectedly: %s", e)
        sys.exit(1)
