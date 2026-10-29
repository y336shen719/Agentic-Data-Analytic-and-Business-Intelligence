"""Unit and integration tests for DuckDB connection and schema discovery (Step 3.1).

Test categories:
  Unit   — isolated tmp_path databases, no production data, no external services.
  Integration — read-only access to data/ecommerce.duckdb; skipped when absent.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict

import duckdb
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_data import DEFAULT_DB_PATH
from tools.sql_tool import (
    SQLValidationError,
    VIEW_BUSINESS_METADATA,
    _APP_SCHEMA,
    db_connection,
    execute_sql,
    get_database_schema,
    get_db_connection,
    validate_sql,
)


# ==============================================================================
# Helpers
# ==============================================================================

def _make_db(tmp_path: Path, name: str = "test.duckdb") -> Path:
    return tmp_path / name


# ==============================================================================
# Database Connection — Unit Tests
# ==============================================================================

@pytest.mark.unit
def test_read_only_connection_success(tmp_path: Path) -> None:
    """Successful read-only connection to an existing DuckDB file."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE t (id INT);")
        con.execute("INSERT INTO t VALUES (1), (2);")

    ro = get_db_connection(db_file)
    try:
        assert ro.execute("SELECT count(*) FROM t").fetchone()[0] == 2
    finally:
        ro.close()


@pytest.mark.unit
def test_connection_failure_nonexistent_file(tmp_path: Path) -> None:
    """FileNotFoundError with helpful message when the database file is missing."""
    missing = tmp_path / "ghost.duckdb"
    with pytest.raises(FileNotFoundError) as exc_info:
        get_db_connection(missing)
    assert "DuckDB database file not found" in str(exc_info.value)
    assert str(missing) in str(exc_info.value)


@pytest.mark.unit
def test_connection_failure_directory_path(tmp_path: Path) -> None:
    """ValueError raised when a directory is supplied as the database path."""
    with pytest.raises(ValueError) as exc_info:
        get_db_connection(tmp_path)
    assert "is a directory" in str(exc_info.value)


@pytest.mark.unit
def test_read_only_prevents_insert(tmp_path: Path) -> None:
    """INSERT is rejected by a read-only connection."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE products (sku VARCHAR);")
        con.execute("INSERT INTO products VALUES ('A');")

    ro = get_db_connection(db_file)
    try:
        with pytest.raises(duckdb.Error):
            ro.execute("INSERT INTO products VALUES ('B');")
    finally:
        ro.close()


@pytest.mark.unit
def test_read_only_prevents_create_table(tmp_path: Path) -> None:
    """CREATE TABLE is rejected by a read-only connection."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE anchor (x INT);")

    ro = get_db_connection(db_file)
    try:
        with pytest.raises(duckdb.Error):
            ro.execute("CREATE TABLE rogue (y INT);")
    finally:
        ro.close()


@pytest.mark.unit
def test_read_only_prevents_drop_table(tmp_path: Path) -> None:
    """DROP TABLE is rejected by a read-only connection; original data survives."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE keep_me (x INT);")
        con.execute("INSERT INTO keep_me VALUES (42);")

    ro = get_db_connection(db_file)
    try:
        with pytest.raises(duckdb.Error):
            ro.execute("DROP TABLE keep_me;")
    finally:
        ro.close()

    with duckdb.connect(str(db_file), read_only=True) as verify:
        assert verify.execute("SELECT x FROM keep_me").fetchone()[0] == 42


@pytest.mark.unit
def test_context_manager_closes_connection(tmp_path: Path) -> None:
    """db_connection closes the connection on normal exit."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE dummy (x INT);")

    with db_connection(db_file) as ctx_con:
        assert ctx_con.execute("SELECT 1").fetchone()[0] == 1

    with pytest.raises(duckdb.Error):
        ctx_con.execute("SELECT 1")


@pytest.mark.unit
def test_context_manager_closes_on_exception(tmp_path: Path) -> None:
    """db_connection closes the connection even when an exception is raised inside."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE dummy (x INT);")

    captured_con = None
    with pytest.raises(RuntimeError):
        with db_connection(db_file) as ctx_con:
            captured_con = ctx_con
            raise RuntimeError("deliberate error")

    assert captured_con is not None
    with pytest.raises(duckdb.Error):
        captured_con.execute("SELECT 1")


# ==============================================================================
# Schema Discovery — base table and view discovery
# ==============================================================================

@pytest.mark.unit
def test_schema_discovery_base_table(tmp_path: Path) -> None:
    """Base table is discovered with correct name, schema, type, and columns."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE users (user_id VARCHAR, age INTEGER);")

    schema = get_database_schema(db_file)
    assert len(schema["tables"]) == 1
    t = schema["tables"][0]
    assert t["name"] == "users"
    assert t["schema"] == "main"
    assert t["type"] == "BASE TABLE"
    assert [c["name"] for c in t["columns"]] == ["user_id", "age"]


@pytest.mark.unit
def test_schema_discovery_view(tmp_path: Path) -> None:
    """View is discovered with correct name, schema, and type."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE orders (order_id VARCHAR, total DECIMAL(18,2));")
        con.execute("CREATE VIEW high_value AS SELECT * FROM orders WHERE total > 100;")

    schema = get_database_schema(db_file)
    assert len(schema["views"]) == 1
    v = schema["views"][0]
    assert v["name"] == "high_value"
    assert v["schema"] == "main"
    assert v["type"] == "VIEW"
    assert [c["name"] for c in v["columns"]] == ["order_id", "total"]


@pytest.mark.unit
def test_column_names_and_data_types(tmp_path: Path) -> None:
    """Column data types are faithfully reported for diverse SQL types."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("""
            CREATE TABLE all_types (
                col_str    VARCHAR,
                col_int    INTEGER,
                col_dec    DECIMAL(18,2),
                col_ts     TIMESTAMP,
                col_bool   BOOLEAN,
                col_double DOUBLE
            );
        """)

    schema = get_database_schema(db_file)
    col_map = {c["name"]: c["data_type"] for c in schema["tables"][0]["columns"]}

    assert col_map["col_str"]    == "VARCHAR"
    assert col_map["col_int"]    == "INTEGER"
    assert col_map["col_dec"]    == "DECIMAL(18,2)"
    assert col_map["col_ts"]     == "TIMESTAMP"
    assert col_map["col_bool"]   == "BOOLEAN"
    assert col_map["col_double"] == "DOUBLE"


@pytest.mark.unit
def test_nullability_metadata_not_null(tmp_path: Path) -> None:
    """NOT NULL columns are reported as nullable=False; unconstrained as nullable=True."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("""
            CREATE TABLE demo (
                req_id  INTEGER NOT NULL,
                opt_val VARCHAR,
                req_str VARCHAR NOT NULL
            );
        """)

    schema = get_database_schema(db_file)
    cols = {c["name"]: c["nullable"] for c in schema["tables"][0]["columns"]}

    assert cols["req_id"]  is False
    assert cols["opt_val"] is True
    assert cols["req_str"] is False


@pytest.mark.unit
def test_nullability_view_columns_reported_as_nullable(tmp_path: Path) -> None:
    """View columns are typically reported as nullable=True by DuckDB regardless of base constraints."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE base (id INTEGER NOT NULL, val VARCHAR NOT NULL);")
        con.execute("CREATE VIEW v AS SELECT * FROM base;")

    schema = get_database_schema(db_file)
    view = schema["views"][0]
    # DuckDB reports view columns as nullable=True even when derived from NOT NULL base columns.
    for col in view["columns"]:
        assert col["nullable"] in (True, None), (
            f"Expected view column '{col['name']}' to be nullable=True or None, got {col['nullable']}"
        )


@pytest.mark.unit
def test_deterministic_schema_ordering(tmp_path: Path) -> None:
    """Tables and views are sorted by (schema, name); columns by ordinal position."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE z_table (c3 INT, c1 VARCHAR, c2 DOUBLE);")
        con.execute("CREATE TABLE a_table (id INT);")
        con.execute("CREATE TABLE m_table (id INT);")
        con.execute("CREATE VIEW z_view AS SELECT 1 AS num;")
        con.execute("CREATE VIEW a_view AS SELECT 2 AS num;")

    schema = get_database_schema(db_file)

    assert [t["name"] for t in schema["tables"]] == ["a_table", "m_table", "z_table"]
    assert [v["name"] for v in schema["views"]]  == ["a_view", "z_view"]
    # z_table columns preserve definition (ordinal) order, not alphabetical.
    assert [c["name"] for c in schema["tables"][2]["columns"]] == ["c3", "c1", "c2"]


@pytest.mark.unit
def test_exclusion_of_system_objects(tmp_path: Path) -> None:
    """No pg_*, duckdb_*, or information_schema objects appear in results."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE app_table (id INT);")

    schema = get_database_schema(db_file)
    all_names = [t["name"] for t in schema["tables"]] + [v["name"] for v in schema["views"]]

    assert all_names == ["app_table"]
    for name in all_names:
        assert not name.startswith("pg_")
        assert not name.startswith("duckdb_")
        assert not name.startswith("sqlite_")


@pytest.mark.unit
def test_json_serializability(tmp_path: Path) -> None:
    """Returned schema is fully JSON-serializable (round-trip safe)."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE sample (id INT, val VARCHAR);")
        con.execute("CREATE VIEW v_sample AS SELECT * FROM sample;")

    schema = get_database_schema(db_file)
    assert json.loads(json.dumps(schema)) == schema


@pytest.mark.unit
def test_schema_discovery_closes_connection(tmp_path: Path) -> None:
    """get_database_schema releases its connection so the file can be opened read-write."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE t (x INT);")

    get_database_schema(db_file)

    with duckdb.connect(str(db_file)) as rw:
        rw.execute("INSERT INTO t VALUES (99);")
        assert rw.execute("SELECT count(*) FROM t").fetchone()[0] == 1


# ==============================================================================
# Multi-Schema Collision Tests
# ==============================================================================

@pytest.mark.unit
def test_multi_schema_same_table_name_no_column_merge(tmp_path: Path) -> None:
    """Two tables with the same name in different schemas must not merge columns."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE SCHEMA archive;")
        con.execute("CREATE TABLE main.orders (main_col INTEGER);")
        con.execute("CREATE TABLE archive.orders (archive_col VARCHAR);")

    schema = get_database_schema(db_file)
    tables = {(t["schema"], t["name"]): t for t in schema["tables"]}

    assert ("main", "orders") in tables
    assert ("archive", "orders") in tables

    main_cols = [c["name"] for c in tables[("main", "orders")]["columns"]]
    arch_cols = [c["name"] for c in tables[("archive", "orders")]["columns"]]

    assert main_cols == ["main_col"],    f"main.orders columns wrong: {main_cols}"
    assert arch_cols == ["archive_col"], f"archive.orders columns wrong: {arch_cols}"


@pytest.mark.unit
def test_multi_schema_table_and_view_same_name(tmp_path: Path) -> None:
    """A table in one schema and a view in another with the same name are kept separate."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE SCHEMA reporting;")
        con.execute("CREATE TABLE main.sales (amount DECIMAL(10,2));")
        con.execute("CREATE VIEW reporting.sales AS SELECT amount * 1.1 AS adjusted FROM main.sales;")

    schema = get_database_schema(db_file)
    tables = {(t["schema"], t["name"]): t for t in schema["tables"]}
    views  = {(v["schema"], v["name"]): v for v in schema["views"]}

    assert ("main", "sales")      in tables
    assert ("reporting", "sales") in views

    assert [c["name"] for c in tables[("main", "sales")]["columns"]]      == ["amount"]
    assert [c["name"] for c in views[("reporting", "sales")]["columns"]]  == ["adjusted"]


@pytest.mark.unit
def test_multi_schema_sorting_by_schema_then_name(tmp_path: Path) -> None:
    """Tables are sorted by (schema, name), not just by name."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE SCHEMA zzz;")
        con.execute("CREATE TABLE zzz.alpha (x INT);")
        con.execute("CREATE TABLE main.beta (y INT);")
        con.execute("CREATE TABLE main.alpha (z INT);")

    schema = get_database_schema(db_file)
    order = [(t["schema"], t["name"]) for t in schema["tables"]]
    assert order == [("main", "alpha"), ("main", "beta"), ("zzz", "alpha")]


@pytest.mark.unit
def test_schema_field_present_on_all_objects(tmp_path: Path) -> None:
    """Every returned table and view has a non-empty 'schema' field."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE t (id INT);")
        con.execute("CREATE VIEW v AS SELECT id FROM t;")

    schema = get_database_schema(db_file)
    for t in schema["tables"]:
        assert "schema" in t and t["schema"], f"Table missing schema: {t}"
    for v in schema["views"]:
        assert "schema" in v and v["schema"], f"View missing schema: {v}"


# ==============================================================================
# Business Metadata — Unit Tests
# ==============================================================================

@pytest.mark.unit
def test_business_metadata_order_summary(tmp_path: Path) -> None:
    """order_summary in main schema receives grain, purpose, notes, column descriptions."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("""
            CREATE VIEW order_summary AS
            SELECT
                'ord1'::VARCHAR        AS order_id,
                'cust1'::VARCHAR       AS customer_id,
                100.00::DECIMAL(18,2)  AS order_gross_value,
                5.0::DOUBLE            AS avg_review_score;
        """)

    schema = get_database_schema(db_file)
    view = {v["name"]: v for v in schema["views"]}["order_summary"]

    assert view["schema"]  == "main"
    assert view["grain"]   == "One row per order."
    assert "payments, revenue, and delivery metrics" in view["purpose"]
    assert "fanout" in view["notes"].lower()

    col_desc = {c["name"]: c.get("description", "") for c in view["columns"]}
    assert "one row per order" in col_desc["order_id"].lower()
    assert col_desc["order_gross_value"] != ""


@pytest.mark.unit
def test_business_metadata_category_sales_summary(tmp_path: Path) -> None:
    """category_sales_summary in main schema receives grain, purpose, warning, column descriptions."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("""
            CREATE VIEW category_sales_summary AS
            SELECT
                '2018-01'::VARCHAR       AS year_month,
                'beauty'::VARCHAR        AS category_name_english,
                10::INTEGER              AS total_orders,
                150.00::DECIMAL(18,2)    AS total_item_revenue,
                15.00::DECIMAL(18,2)     AS avg_item_price;
        """)

    schema = get_database_schema(db_file)
    view = {v["name"]: v for v in schema["views"]}["category_sales_summary"]

    assert view["schema"] == "main"
    assert view["grain"]  == "One row per month and product category."
    assert "Category-level sales" in view["purpose"]
    assert "non-additive" in view["warning"].lower()

    col_desc = {c["name"]: c.get("description", "") for c in view["columns"]}
    assert "non-additive"   in col_desc["total_orders"].lower()
    assert "weighted average" in col_desc["avg_item_price"].lower()


@pytest.mark.unit
def test_business_metadata_not_attached_to_other_schema(tmp_path: Path) -> None:
    """A view named order_summary in a non-main schema must NOT receive business metadata."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE SCHEMA staging;")
        con.execute("""
            CREATE VIEW staging.order_summary AS
            SELECT 'x'::VARCHAR AS some_col;
        """)

    schema = get_database_schema(db_file)
    views = {(v["schema"], v["name"]): v for v in schema["views"]}

    assert ("staging", "order_summary") in views
    other_view = views[("staging", "order_summary")]

    assert "grain"   not in other_view
    assert "purpose" not in other_view
    assert "warning" not in other_view
    assert "notes"   not in other_view
    for col in other_view["columns"]:
        assert "description" not in col


@pytest.mark.unit
def test_business_metadata_columns_all_documented(tmp_path: Path) -> None:
    """Every key in VIEW_BUSINESS_METADATA column_descriptions is a valid column name (no stale docs)."""
    # This test creates the actual views as defined in prepare_data.py VIEW_DEFINITIONS
    # using a synthetic minimal setup to check column-name alignment.
    from tools.sql_tool import VIEW_BUSINESS_METADATA, _APP_SCHEMA

    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        # Minimal tables required by the views
        con.execute("""
            CREATE TABLE orders (
                order_id VARCHAR, customer_id VARCHAR, order_status VARCHAR,
                order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP,
                order_delivered_carrier_date TIMESTAMP, order_delivered_customer_date TIMESTAMP,
                order_estimated_delivery_date TIMESTAMP
            );
        """)
        con.execute("""
            CREATE TABLE customers (
                customer_id VARCHAR, customer_unique_id VARCHAR,
                customer_zip_code_prefix VARCHAR, customer_city VARCHAR, customer_state VARCHAR
            );
        """)
        con.execute("""
            CREATE TABLE order_items (
                order_id VARCHAR, order_item_id INTEGER, product_id VARCHAR,
                seller_id VARCHAR, shipping_limit_date TIMESTAMP,
                price DECIMAL(18,2), freight_value DECIMAL(18,2)
            );
        """)
        con.execute("""
            CREATE TABLE order_payments (
                order_id VARCHAR, payment_sequential INTEGER, payment_type VARCHAR,
                payment_installments INTEGER, payment_value DECIMAL(18,2)
            );
        """)
        con.execute("""
            CREATE TABLE order_reviews (
                review_id VARCHAR, order_id VARCHAR, review_score INTEGER,
                review_comment_title VARCHAR, review_comment_message VARCHAR,
                review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP
            );
        """)
        con.execute("""
            CREATE TABLE products (
                product_id VARCHAR, product_category_name VARCHAR,
                product_name_lenght INTEGER, product_description_lenght INTEGER,
                product_photos_qty INTEGER, product_weight_g INTEGER,
                product_length_cm INTEGER, product_height_cm INTEGER, product_width_cm INTEGER
            );
        """)
        con.execute("""
            CREATE TABLE category_translation (
                product_category_name VARCHAR, product_category_name_english VARCHAR
            );
        """)

        # Build the real views from prepare_data.py definitions
        from data.prepare_data import VIEW_DEFINITIONS
        for view_sql in VIEW_DEFINITIONS.values():
            con.execute(view_sql)

    schema = get_database_schema(db_file)
    views_by_name = {v["name"]: v for v in schema["views"]}

    for (meta_schema, view_name), meta in VIEW_BUSINESS_METADATA.items():
        if meta_schema != _APP_SCHEMA:
            continue
        assert view_name in views_by_name, f"View '{view_name}' not found in schema"
        actual_col_names = {c["name"] for c in views_by_name[view_name]["columns"]}
        documented_cols  = set(meta.get("column_descriptions", {}).keys())

        stale = documented_cols - actual_col_names
        assert not stale, (
            f"VIEW_BUSINESS_METADATA['{view_name}'] documents non-existent columns: {stale}"
        )


# ==============================================================================
# Integration Tests (Production database)
# ==============================================================================

@pytest.mark.integration
def test_prod_schema_discovery_all_tables_and_views() -> None:
    """All 9 Olist tables and 2 analysis views are discoverable in the production database."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    schema = get_database_schema(DEFAULT_DB_PATH)

    expected_tables = {
        "category_translation", "customers", "geolocation",
        "order_items", "order_payments", "order_reviews",
        "orders", "products", "sellers",
    }
    expected_views = {"category_sales_summary", "order_summary"}

    discovered_tables = {t["name"] for t in schema["tables"]}
    discovered_views  = {v["name"] for v in schema["views"]}

    assert discovered_tables == expected_tables, (
        f"Table mismatch: {discovered_tables ^ expected_tables}"
    )
    assert discovered_views == expected_views, (
        f"View mismatch: {discovered_views ^ expected_views}"
    )


@pytest.mark.integration
def test_prod_schema_field_is_main() -> None:
    """All production tables and views are in the 'main' schema."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    schema = get_database_schema(DEFAULT_DB_PATH)
    for t in schema["tables"]:
        assert t["schema"] == "main", f"Unexpected schema for table {t['name']}: {t['schema']}"
    for v in schema["views"]:
        assert v["schema"] == "main", f"Unexpected schema for view {v['name']}: {v['schema']}"


@pytest.mark.integration
def test_prod_order_summary_metadata_and_columns() -> None:
    """order_summary has 26 columns, correct business metadata, all columns documented."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    schema = get_database_schema(DEFAULT_DB_PATH)
    view = next(v for v in schema["views"] if v["name"] == "order_summary")

    assert len(view["columns"]) == 26
    assert view["grain"] == "One row per order."
    assert "payments, revenue, and delivery metrics" in view["purpose"]
    assert all("description" in c for c in view["columns"]), (
        "Some order_summary columns are missing descriptions"
    )


@pytest.mark.integration
def test_prod_category_sales_summary_metadata_and_columns() -> None:
    """category_sales_summary has 15 columns, non-additive warning, all columns documented."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    schema = get_database_schema(DEFAULT_DB_PATH)
    view = next(v for v in schema["views"] if v["name"] == "category_sales_summary")

    assert len(view["columns"]) == 15
    assert "non-additive" in view["warning"].lower()
    assert all("description" in c for c in view["columns"]), (
        "Some category_sales_summary columns are missing descriptions"
    )


@pytest.mark.integration
def test_prod_json_round_trip() -> None:
    """Production schema is fully JSON-serializable."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    schema = get_database_schema(DEFAULT_DB_PATH)
    assert json.loads(json.dumps(schema)) == schema


# ==============================================================================
# Step 3.2 Helpers
# ==============================================================================

def _make_analytical_test_db(tmp_path: Path) -> Path:
    db_file = _make_db(tmp_path, "analytics.duckdb")
    with duckdb.connect(str(db_file)) as con:
        con.execute("""
            CREATE TABLE customers (
                customer_id VARCHAR PRIMARY KEY,
                customer_state VARCHAR
            );
            INSERT INTO customers VALUES
                ('c1', 'SP'),
                ('c2', 'RJ'),
                ('c3', 'SP');

            CREATE TABLE orders (
                order_id VARCHAR PRIMARY KEY,
                customer_id VARCHAR,
                order_status VARCHAR,
                order_purchase_timestamp TIMESTAMP,
                order_value DECIMAL(10,2)
            );
            INSERT INTO orders VALUES
                ('o1', 'c1', 'delivered', TIMESTAMP '2023-01-15 10:00:00', 100.50),
                ('o2', 'c2', 'shipped', TIMESTAMP '2023-02-20 14:30:00', 50.25),
                ('o3', 'c1', 'delivered', TIMESTAMP '2023-03-10 09:15:00', 200.00);
        """)
    return db_file


# ==============================================================================
# Step 3.2 — Valid Analytical SQL Execution (Unit Tests)
# ==============================================================================

@pytest.mark.unit
def test_execute_sql_simple_select(tmp_path: Path) -> None:
    """Simple SELECT returns success, correct columns, rows, and timing."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT customer_id, customer_state FROM customers ORDER BY customer_id", db_path=db_file)

    assert res["status"] == "success"
    assert res["columns"] == ["customer_id", "customer_state"]
    assert res["rows"] == [["c1", "SP"], ["c2", "RJ"], ["c3", "SP"]]
    assert res["row_count"] == 3
    assert res["truncated"] is False
    assert res["execution_time_ms"] > 0


@pytest.mark.unit
def test_execute_sql_where_filtering(tmp_path: Path) -> None:
    """WHERE clause filtering returns only matching rows."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT customer_id FROM customers WHERE customer_state = 'SP' ORDER BY customer_id", db_path=db_file)

    assert res["status"] == "success"
    assert res["rows"] == [["c1"], ["c3"]]
    assert res["row_count"] == 2


@pytest.mark.unit
def test_execute_sql_joins(tmp_path: Path) -> None:
    """JOIN queries execute properly across allowed tables."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("""
        SELECT o.order_id, c.customer_state
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        ORDER BY o.order_id
    """, db_path=db_file)

    assert res["status"] == "success"
    assert res["columns"] == ["order_id", "customer_state"]
    assert res["rows"] == [["o1", "SP"], ["o2", "RJ"], ["o3", "SP"]]


@pytest.mark.unit
def test_execute_sql_group_by_aggregation(tmp_path: Path) -> None:
    """GROUP BY with aggregations computes correct results."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("""
        SELECT customer_id, count(*) AS order_cnt, sum(order_value) AS total_val
        FROM orders
        GROUP BY customer_id
        ORDER BY customer_id
    """, db_path=db_file)

    assert res["status"] == "success"
    assert res["columns"] == ["customer_id", "order_cnt", "total_val"]
    assert res["rows"] == [["c1", 2, "300.50"], ["c2", 1, "50.25"]]


@pytest.mark.unit
def test_execute_sql_order_by_limit(tmp_path: Path) -> None:
    """ORDER BY and SQL LIMIT are respected by the execution engine."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT order_id, order_value FROM orders ORDER BY order_value DESC LIMIT 1", db_path=db_file)

    assert res["status"] == "success"
    assert res["rows"] == [["o3", "200.00"]]
    assert res["row_count"] == 1
    assert res["truncated"] is False


@pytest.mark.unit
def test_execute_sql_cte(tmp_path: Path) -> None:
    """Common Table Expressions (WITH ... SELECT) are supported and CTE alias is recognized."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("""
        WITH high_value AS (
            SELECT order_id, order_value FROM orders WHERE order_value > 75.00
        )
        SELECT order_id FROM high_value ORDER BY order_id
    """, db_path=db_file)

    assert res["status"] == "success"
    assert res["rows"] == [["o1"], ["o3"]]


@pytest.mark.unit
def test_execute_sql_window_functions(tmp_path: Path) -> None:
    """Window functions (ROW_NUMBER, RANK) execute and return correct rankings."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("""
        SELECT order_id, row_number() OVER (ORDER BY order_purchase_timestamp) AS seq
        FROM orders
        ORDER BY seq
    """, db_path=db_file)

    assert res["status"] == "success"
    assert res["rows"] == [["o1", 1], ["o2", 2], ["o3", 3]]


@pytest.mark.unit
def test_execute_sql_nested_subqueries(tmp_path: Path) -> None:
    """Nested subqueries in WHERE and FROM clauses execute cleanly."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("""
        SELECT order_id
        FROM orders
        WHERE customer_id IN (SELECT customer_id FROM customers WHERE customer_state = 'RJ')
    """, db_path=db_file)

    assert res["status"] == "success"
    assert res["rows"] == [["o2"]]


@pytest.mark.unit
def test_execute_sql_empty_result(tmp_path: Path) -> None:
    """Zero-row queries return correct column names and an empty rows list."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT customer_id, customer_state FROM customers WHERE 1 = 0", db_path=db_file)

    assert res["status"] == "success"
    assert res["columns"] == ["customer_id", "customer_state"]
    assert res["rows"] == []
    assert res["row_count"] == 0
    assert res["truncated"] is False


@pytest.mark.unit
def test_execute_sql_null_values(tmp_path: Path) -> None:
    """NULL values are faithfully converted to JSON null (None in Python)."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE demo (id INT, val VARCHAR);")
        con.execute("INSERT INTO demo VALUES (1, NULL), (2, 'non-null');")

    res = execute_sql("SELECT id, val FROM demo ORDER BY id", db_path=db_file)
    assert res["status"] == "success"
    assert res["rows"][0] == [1, None]
    assert res["rows"][1] == [2, "non-null"]


@pytest.mark.unit
def test_execute_sql_decimal_serialization(tmp_path: Path) -> None:
    """DECIMAL values are serialized as strings to preserve exact decimal precision."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE finances (amount DECIMAL(18,4));")
        con.execute("INSERT INTO finances VALUES (123456789.9876);")

    res = execute_sql("SELECT amount FROM finances", db_path=db_file)
    assert res["status"] == "success"
    assert res["rows"] == [["123456789.9876"]]
    assert isinstance(res["rows"][0][0], str)


@pytest.mark.unit
def test_execute_sql_date_timestamp_serialization(tmp_path: Path) -> None:
    """DATE and TIMESTAMP values are serialized to ISO format strings."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE schedule (d DATE, ts TIMESTAMP);")
        con.execute("INSERT INTO schedule VALUES (DATE '2023-05-12', TIMESTAMP '2023-05-12 15:45:00');")

    res = execute_sql("SELECT d, ts FROM schedule", db_path=db_file)
    assert res["status"] == "success"
    assert res["rows"] == [["2023-05-12", "2023-05-12T15:45:00"]]


@pytest.mark.unit
def test_execute_sql_json_serializability(tmp_path: Path) -> None:
    """All successful responses pass strict json.dumps(res, allow_nan=False)."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT * FROM orders", db_path=db_file)

    dumped = json.dumps(res, allow_nan=False)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"
    assert reloaded["row_count"] == 3


# ==============================================================================
# Step 3.2 — SQL Safety and Validation Tests (Unit Tests)
# ==============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("dml", [
    "INSERT INTO orders VALUES ('o9', 'c1', 'shipped', now(), 10.0)",
    "UPDATE orders SET order_status = 'canceled' WHERE order_id = 'o1'",
    "DELETE FROM orders WHERE order_id = 'o1'",
])
def test_validate_sql_rejects_dml(dml: str) -> None:
    """Modifying DML statements (INSERT, UPDATE, DELETE) are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql(dml)


@pytest.mark.unit
@pytest.mark.parametrize("ddl", [
    "CREATE TABLE rogue (x INT)",
    "ALTER TABLE orders ADD COLUMN extra VARCHAR",
    "DROP TABLE orders",
    "TRUNCATE orders",
])
def test_validate_sql_rejects_ddl(ddl: str) -> None:
    """DDL statements (CREATE, ALTER, DROP, TRUNCATE) are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql(ddl)


@pytest.mark.unit
def test_validate_sql_rejects_copy_export() -> None:
    """COPY statements are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql("COPY orders TO '/tmp/leak.csv'")


@pytest.mark.unit
def test_validate_sql_rejects_attach_detach() -> None:
    """ATTACH and DETACH database statements are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql("ATTACH '/tmp/other.duckdb'")
    with pytest.raises(SQLValidationError):
        validate_sql("DETACH other")


@pytest.mark.unit
def test_validate_sql_rejects_pragma_set() -> None:
    """PRAGMA and SET configuration statements are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql("PRAGMA version")
    with pytest.raises(SQLValidationError):
        validate_sql("SET threads = 8")


@pytest.mark.unit
def test_validate_sql_rejects_install_load() -> None:
    """INSTALL and LOAD extension statements are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql("INSTALL httpfs")
    with pytest.raises(SQLValidationError):
        validate_sql("LOAD httpfs")


@pytest.mark.unit
@pytest.mark.parametrize("multi_stmt", [
    "SELECT * FROM orders; SELECT * FROM customers;",
    "SELECT * FROM orders; DROP TABLE orders;",
    "SELECT 1; -- comment\n DROP TABLE orders;",
])
def test_validate_sql_rejects_multiple_statements(multi_stmt: str) -> None:
    """Multiple SQL statements in a single call are strictly prohibited."""
    with pytest.raises(SQLValidationError):
        validate_sql(multi_stmt)


@pytest.mark.unit
def test_validate_sql_rejects_nested_destructive_in_cte() -> None:
    """Unsafe statements nested inside CTEs or subqueries are detected and rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql("WITH evil AS (CREATE TABLE hacked AS SELECT 1) SELECT * FROM evil")


@pytest.mark.unit
@pytest.mark.parametrize("file_reader", [
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('s3://mybucket/data.parquet')",
    "SELECT * FROM read_json('/tmp/test.json')",
    "SELECT * FROM 'secret.csv'",
    "SELECT * FROM 'data.parquet'",
])
def test_validate_sql_rejects_file_readers(file_reader: str) -> None:
    """Table-valued file reader functions and raw file path references are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql(file_reader)


@pytest.mark.unit
def test_validate_sql_rejects_external_urls() -> None:
    """External URLs and remote scanning endpoints are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM 'https://attacker.com/payload.parquet'")


@pytest.mark.unit
def test_validate_sql_rejects_table_valued_functions() -> None:
    """Built-in table-valued catalog or filesystem inspection functions are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM duckdb_settings()")
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM glob('*.csv')")


@pytest.mark.unit
def test_validate_sql_rejects_unauthorized_schemas_tables() -> None:
    """Catalog schemas (information_schema, pg_catalog) and unlisted tables are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM information_schema.tables")
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM pg_catalog.pg_tables")
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM secret_payroll_table")


@pytest.mark.unit
@pytest.mark.parametrize("tricky_query", [
    "SELECT * FROM orders WHERE customer_id = '1' /* inline comment */ ; DROP TABLE orders",
    "SELECT * FROM orders -- line comment\n; DROP TABLE orders",
    "",
    "   ",
    ";;;",
])
def test_validate_sql_rejects_tricky_bypasses(tricky_query: str) -> None:
    """Tricky comment injections and empty queries are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql(tricky_query)


@pytest.mark.unit
def test_rejected_queries_do_not_modify_database(tmp_path: Path) -> None:
    """Rejected unsafe queries do not modify the database or corrupt state."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("DROP TABLE orders;", db_path=db_file)
    assert res["status"] == "error"
    assert res["error_type"] == "SQLValidationError"

    # Verify original table and row count remain intact
    verify = execute_sql("SELECT count(*) AS cnt FROM orders", db_path=db_file)
    assert verify["status"] == "success"
    assert verify["rows"] == [[3]]


# ==============================================================================
# Step 3.2 — Execution Limits and Timeouts (Unit Tests)
# ==============================================================================

@pytest.mark.unit
def test_execute_sql_default_and_custom_row_limits(tmp_path: Path) -> None:
    """Custom max_rows caps the returned rows and sets truncated flag appropriately."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE nums AS SELECT range AS n FROM range(20);")

    # Limit to 5
    res5 = execute_sql("SELECT n FROM nums ORDER BY n", db_path=db_file, max_rows=5)
    assert res5["status"] == "success"
    assert len(res5["rows"]) == 5
    assert res5["row_count"] == 5
    assert res5["truncated"] is True

    # Limit to 30 (more than table size)
    res30 = execute_sql("SELECT n FROM nums ORDER BY n", db_path=db_file, max_rows=30)
    assert res30["status"] == "success"
    assert len(res30["rows"]) == 20
    assert res30["row_count"] == 20
    assert res30["truncated"] is False


@pytest.mark.unit
def test_execute_sql_truncation_flag_accurate(tmp_path: Path) -> None:
    """Truncation flag is False when rows == max_rows, True when rows > max_rows."""
    db_file = _make_db(tmp_path)
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE exact (x INT);")
        con.execute("INSERT INTO exact VALUES (1), (2), (3);")

    # Exactly 3 rows with max_rows=3 -> truncated=False
    res_exact = execute_sql("SELECT x FROM exact", db_path=db_file, max_rows=3)
    assert res_exact["status"] == "success"
    assert res_exact["row_count"] == 3
    assert res_exact["truncated"] is False

    # 3 rows with max_rows=2 -> truncated=True
    res_trunc = execute_sql("SELECT x FROM exact", db_path=db_file, max_rows=2)
    assert res_trunc["status"] == "success"
    assert res_trunc["row_count"] == 2
    assert res_trunc["truncated"] is True


@pytest.mark.unit
@pytest.mark.parametrize("invalid_args", [
    {"max_rows": 0},
    {"max_rows": -5},
    {"timeout_seconds": 0},
    {"timeout_seconds": -1.0},
])
def test_execute_sql_invalid_limits(tmp_path: Path, invalid_args: Dict[str, Any]) -> None:
    """Invalid row or timeout limits return a structured ValueError response."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT * FROM customers", db_path=db_file, **invalid_args)

    assert res["status"] == "error"
    assert res["error_type"] == "ValueError"
    assert "positive" in res["error_message"].lower()


@pytest.mark.unit
def test_execute_sql_timeout_enforcement(tmp_path: Path) -> None:
    """Long-running queries are genuinely terminated within the specified timeout."""
    db_file = _make_analytical_test_db(tmp_path)
    # A query that takes a very long time (cross product of recursive series)
    slow_query = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT count(*) FROM r"

    res = execute_sql(slow_query, db_path=db_file, timeout_seconds=0.3)
    assert res["status"] == "error"
    assert res["error_type"] == "TimeoutError"
    assert "timeout limit" in res["error_message"].lower()


@pytest.mark.unit
def test_execute_sql_cleanup_after_timeout(tmp_path: Path) -> None:
    """After a query times out, the database connection is closed and file remains usable."""
    db_file = _make_analytical_test_db(tmp_path)
    slow_query = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT count(*) FROM r"

    execute_sql(slow_query, db_path=db_file, timeout_seconds=0.2)

    # Immediately execute a normal fast query on the same database
    fast_res = execute_sql("SELECT count(*) AS cnt FROM customers", db_path=db_file)
    assert fast_res["status"] == "success"
    assert fast_res["rows"] == [[3]]


# ==============================================================================
# Step 3.2 — Error Handling and Sanitization (Unit Tests)
# ==============================================================================

@pytest.mark.unit
def test_execute_sql_syntax_error(tmp_path: Path) -> None:
    """Malformed SQL syntax returns a structured error response."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT FROM WHERE", db_path=db_file)

    assert res["status"] == "error"
    assert res["error_type"] in ("SQLValidationError", "ParserException")
    assert res["execution_time_ms"] > 0


@pytest.mark.unit
def test_execute_sql_unknown_column(tmp_path: Path) -> None:
    """Referencing a non-existent column returns a sanitized BinderException."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT nonexistent_col FROM customers", db_path=db_file)

    assert res["status"] == "error"
    assert "BinderException" in res["error_type"] or "Binder" in res["error_type"]
    assert "nonexistent_col" in res["error_message"]


@pytest.mark.unit
def test_execute_sql_unknown_table(tmp_path: Path) -> None:
    """Referencing an unauthorized table returns SQLValidationError."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT * FROM non_existent_table", db_path=db_file)

    assert res["status"] == "error"
    assert res["error_type"] == "SQLValidationError"
    assert "not in the allowed tables list" in res["error_message"]


@pytest.mark.unit
def test_execute_sql_missing_db_file(tmp_path: Path) -> None:
    """A missing database file returns a sanitized FileNotFoundError."""
    ghost = tmp_path / "ghost.duckdb"
    res = execute_sql("SELECT 1", db_path=ghost)

    assert res["status"] == "error"
    assert res["error_type"] == "FileNotFoundError"
    assert "<database_file>" in res["error_message"]


@pytest.mark.unit
def test_execute_sql_error_response_sanitized(tmp_path: Path) -> None:
    """Error responses do not leak local filesystem paths or infrastructure paths."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT bad_column FROM customers", db_path=db_file)

    assert res["status"] == "error"
    # Ensure raw user path is not in the error message
    assert str(tmp_path) not in res["error_message"]


@pytest.mark.unit
def test_execute_sql_error_response_json_serializable(tmp_path: Path) -> None:
    """Error responses are fully JSON-serializable."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT bad_column FROM customers", db_path=db_file)

    dumped = json.dumps(res, allow_nan=False)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "error"


# ==============================================================================
# Step 3.2 — Production Database Integration Tests
# ==============================================================================

@pytest.mark.integration
def test_prod_execute_sql_order_summary_query() -> None:
    """Analytical query over order_summary executes cleanly on production DB."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    query = """
        SELECT customer_state, count(*) AS order_count, round(sum(order_gross_value), 2) AS total_gross
        FROM order_summary
        WHERE is_delivered = true
        GROUP BY customer_state
        ORDER BY total_gross DESC
        LIMIT 5;
    """
    res = execute_sql(query, db_path=DEFAULT_DB_PATH)

    assert res["status"] == "success"
    assert res["columns"] == ["customer_state", "order_count", "total_gross"]
    assert len(res["rows"]) == 5
    assert res["row_count"] == 5
    assert res["truncated"] is False
    assert res["rows"][0][0] == "SP"  # Sao Paulo has highest order volume in Olist


@pytest.mark.integration
def test_prod_execute_sql_category_sales_summary_query() -> None:
    """Grouped query over category_sales_summary executes cleanly on production DB."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    query = """
        SELECT category_name_english, sum(total_items_sold) AS items_sold, sum(total_gross_sales) AS gross_sales
        FROM category_sales_summary
        GROUP BY category_name_english
        ORDER BY gross_sales DESC
        LIMIT 3;
    """
    res = execute_sql(query, db_path=DEFAULT_DB_PATH)

    assert res["status"] == "success"
    assert res["columns"] == ["category_name_english", "items_sold", "gross_sales"]
    assert len(res["rows"]) == 3
    assert res["row_count"] == 3


@pytest.mark.integration
def test_prod_execute_sql_base_tables_join() -> None:
    """Read-only join query across base tables executes cleanly on production DB."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    query = """
        SELECT o.order_id, c.customer_city, p.payment_type, p.payment_value
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_payments p ON o.order_id = p.order_id
        WHERE o.order_status = 'delivered'
        ORDER BY o.order_purchase_timestamp DESC
        LIMIT 5;
    """
    res = execute_sql(query, db_path=DEFAULT_DB_PATH)

    assert res["status"] == "success"
    assert res["columns"] == ["order_id", "customer_city", "payment_type", "payment_value"]
    assert len(res["rows"]) == 5


@pytest.mark.integration
def test_prod_execute_sql_structured_output_format() -> None:
    """Structured output format is strictly adhered to and JSON-serializable on production."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    res = execute_sql("SELECT count(*) AS total_orders FROM orders", db_path=DEFAULT_DB_PATH)

    assert res["status"] == "success"
    assert "columns" in res and "rows" in res and "row_count" in res
    assert "truncated" in res and "execution_time_ms" in res
    assert res["rows"][0][0] == 99441
    # Strict JSON test
    assert json.loads(json.dumps(res, allow_nan=False)) == res


@pytest.mark.integration
def test_prod_execute_sql_database_preservation() -> None:
    """Executing queries leaves the production database file and row counts unchanged."""
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database not found at {DEFAULT_DB_PATH}.")

    with duckdb.connect(str(DEFAULT_DB_PATH), read_only=True) as con:
        before_count = con.execute("SELECT count(*) FROM orders").fetchone()[0]

    # Attempt a read query and an unauthorized query
    _ = execute_sql("SELECT * FROM orders LIMIT 10", db_path=DEFAULT_DB_PATH)
    _ = execute_sql("DROP TABLE orders", db_path=DEFAULT_DB_PATH)

    with duckdb.connect(str(DEFAULT_DB_PATH), read_only=True) as con:
        after_count = con.execute("SELECT count(*) FROM orders").fetchone()[0]

    assert before_count == after_count == 99441


# ==============================================================================
# Step 3.2 Hardening — Deadlock, Scope, Fail-Closed & Limits Tests
# ==============================================================================

@pytest.mark.unit
def test_execute_sql_large_result_no_ipc_deadlock(tmp_path: Path) -> None:
    """A large result (4500 rows with 100-char strings) transfers through Pipe without deadlock."""
    db_file = _make_db(tmp_path, "large_pipe.duckdb")
    with duckdb.connect(str(db_file)) as con:
        con.execute("CREATE TABLE big AS SELECT range AS id, repeat('A', 100) AS padding FROM range(4500);")

    res = execute_sql("SELECT id, padding FROM big ORDER BY id", db_path=db_file, max_rows=5000, timeout_seconds=10.0)

    assert res["status"] == "success"
    assert res["row_count"] == 4500
    assert len(res["rows"]) == 4500
    assert res["truncated"] is False


@pytest.mark.unit
def test_execute_sql_max_rows_upper_bound(tmp_path: Path) -> None:
    """max_rows exceeding MAX_PERMITTED_ROWS (5000) is rejected with ValueError."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT * FROM customers", db_path=db_file, max_rows=5001)

    assert res["status"] == "error"
    assert res["error_type"] == "ValueError"
    assert "5000" in res["error_message"]


@pytest.mark.unit
def test_execute_sql_max_timeout_upper_bound(tmp_path: Path) -> None:
    """timeout_seconds exceeding MAX_PERMITTED_TIMEOUT_SECONDS (30.0) is rejected."""
    db_file = _make_analytical_test_db(tmp_path)
    res = execute_sql("SELECT * FROM customers", db_path=db_file, timeout_seconds=35.0)

    assert res["status"] == "error"
    assert res["error_type"] == "ValueError"
    assert "30.0" in res["error_message"]


@pytest.mark.unit
def test_execute_sql_rejects_boolean_limits(tmp_path: Path) -> None:
    """Booleans (True/False) are strictly rejected as integer or float limit arguments."""
    db_file = _make_analytical_test_db(tmp_path)

    res_row = execute_sql("SELECT * FROM customers", db_path=db_file, max_rows=True)  # type: ignore
    assert res_row["status"] == "error"
    assert res_row["error_type"] == "ValueError"

    res_time = execute_sql("SELECT * FROM customers", db_path=db_file, timeout_seconds=True)  # type: ignore
    assert res_time["status"] == "error"
    assert res_time["error_type"] == "ValueError"


@pytest.mark.unit
def test_execute_sql_rejects_nan_inf_timeout(tmp_path: Path) -> None:
    """NaN and Infinite timeout values are rejected with ValueError."""
    db_file = _make_analytical_test_db(tmp_path)

    res_nan = execute_sql("SELECT * FROM customers", db_path=db_file, timeout_seconds=float("nan"))
    assert res_nan["status"] == "error"
    assert res_nan["error_type"] == "ValueError"

    res_inf = execute_sql("SELECT * FROM customers", db_path=db_file, timeout_seconds=float("inf"))
    assert res_inf["status"] == "error"
    assert res_inf["error_type"] == "ValueError"


@pytest.mark.unit
def test_validate_sql_rejects_schema_qualified_cte_bypass() -> None:
    """A CTE name cannot be accessed using schema qualification (e.g. main.cte_name)."""
    # 'my_cte' is a CTE, but qualifying with 'main.' means it must be a real table in allowed_tables
    with pytest.raises(SQLValidationError) as exc_info:
        validate_sql("WITH my_cte AS (SELECT 1 AS a) SELECT * FROM main.my_cte", allowed_tables={"orders"})
    assert "not in the allowed tables list" in str(exc_info.value)


@pytest.mark.unit
def test_validate_sql_rejects_cte_out_of_scope() -> None:
    """A CTE defined inside a subquery cannot be accessed in an outer scope."""
    query = """
        SELECT *
        FROM (
            WITH inner_cte AS (SELECT 1 AS x)
            SELECT * FROM inner_cte
        ) sub
        JOIN inner_cte ON sub.x = inner_cte.x
    """
    with pytest.raises(SQLValidationError) as exc_info:
        validate_sql(query, allowed_tables={"orders"})
    assert "inner_cte" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("tx_cmd", [
    "BEGIN TRANSACTION",
    "COMMIT",
    "ROLLBACK",
])
def test_validate_sql_rejects_transaction_commands(tx_cmd: str) -> None:
    """Transaction control statements are rejected."""
    with pytest.raises(SQLValidationError):
        validate_sql(tx_cmd)


@pytest.mark.unit
def test_execute_sql_fail_closed_custom_schema_discovery(tmp_path: Path) -> None:
    """When schema discovery fails on a corrupted database file, execute_sql does not fall back."""
    bad_db = tmp_path / "corrupted.duckdb"
    # Create invalid file content (not a DuckDB database)
    bad_db.write_bytes(b"THIS IS NOT A DUCKDB FILE")

    res = execute_sql("SELECT * FROM orders", db_path=bad_db)

    # Must fail closed with explicit error, NOT fall back to default tables
    assert res["status"] == "error"
    assert "Schema discovery failed" in res["error_message"] or "IO Error" in res["error_type"] or "IOException" in res["error_type"]


@pytest.mark.unit
def test_no_leaked_child_processes(tmp_path: Path) -> None:
    """Repeated executions do not leave zombie or orphaned child processes."""
    import multiprocessing
    db_file = _make_analytical_test_db(tmp_path)

    for _ in range(5):
        res = execute_sql("SELECT count(*) FROM customers", db_path=db_file)
        assert res["status"] == "success"

    # Verify no active children belonging to current process
    active_children = multiprocessing.active_children()
    assert len(active_children) == 0, f"Leaked worker processes: {active_children}"


# ==============================================================================
# Step 3.2 Targeted Stabilization Tests (IPC, Scope Validation, Worker Cleanup)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SQL Scope Validation Fail-Closed Tests
# ------------------------------------------------------------------------------

@pytest.mark.unit
def test_sql_scope_build_scope_exception_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When build_scope() raises an exception, validation fails closed with SQLValidationError."""
    import sqlglot.optimizer.scope

    def _broken_build_scope(tree: Any) -> Any:
        raise RuntimeError("Internal SQLGlot Scope Construction Failure")

    monkeypatch.setattr(sqlglot.optimizer.scope, "build_scope", _broken_build_scope)
    monkeypatch.setattr("tools.sql_tool.build_scope", _broken_build_scope)

    with pytest.raises(SQLValidationError) as exc_info:
        validate_sql("SELECT * FROM orders")
    assert "SQL scope analysis failed" in str(exc_info.value)


@pytest.mark.unit
def test_sql_scope_build_scope_returns_none_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When build_scope() returns None, query is rejected immediately."""
    monkeypatch.setattr("tools.sql_tool.build_scope", lambda tree: None)

    with pytest.raises(SQLValidationError) as exc_info:
        validate_sql("SELECT * FROM orders")
    assert "returned no valid query scope" in str(exc_info.value)


@pytest.mark.unit
def test_sql_scope_nested_ctes_and_valid_resolution() -> None:
    """Nested CTEs resolve appropriately in their respective valid scopes."""
    query = """
    WITH base_orders AS (
        SELECT order_id, customer_id, order_status FROM orders
    ),
    delivered_orders AS (
        SELECT order_id, customer_id FROM base_orders WHERE order_status = 'delivered'
    )
    SELECT d.customer_id, count(d.order_id) AS total_delivered
    FROM delivered_orders d
    GROUP BY d.customer_id
    """
    # Should validate without raising
    validate_sql(query)


@pytest.mark.unit
def test_sql_scope_cte_shadowing_isolated() -> None:
    """CTE shadowing a physical table name resolves safely within its local scope."""
    query = """
    WITH customers AS (
        SELECT 123 AS customer_id, 'SP' AS customer_state
    )
    SELECT customer_id, customer_state FROM customers
    """
    validate_sql(query)


@pytest.mark.unit
def test_sql_scope_rejects_out_of_scope_cte_reference() -> None:
    """A CTE defined within a subquery/inner CTE cannot be referenced in outer scopes."""
    query = """
    WITH cte_outer AS (
        WITH cte_inner AS (
            SELECT 1 AS secret_col
        )
        SELECT * FROM cte_inner
    )
    SELECT * FROM cte_outer JOIN cte_inner ON cte_outer.secret_col = cte_inner.secret_col
    """
    with pytest.raises(SQLValidationError) as exc_info:
        validate_sql(query)
    assert "Table or view 'cte_inner' is not in the allowed tables list" in str(exc_info.value)


@pytest.mark.unit
def test_sql_scope_schema_qualified_table_allowed() -> None:
    """main.orders is recognized and allowed as an explicit schema reference."""
    validate_sql("SELECT order_id FROM main.orders")


@pytest.mark.unit
def test_sql_scope_rejects_schema_qualified_cte_bypass() -> None:
    """main.my_cte cannot bypass table authorization by mimicking a CTE name."""
    query = """
    WITH my_cte AS (
        SELECT 1 AS x
    )
    SELECT * FROM main.my_cte
    """
    with pytest.raises(SQLValidationError) as exc_info:
        validate_sql(query)
    assert "Table or view 'my_cte' is not in the allowed tables list" in str(exc_info.value)


@pytest.mark.unit
def test_sql_scope_valid_subqueries_and_window_functions() -> None:
    """Valid subqueries and analytical window functions validate successfully."""
    query = """
    SELECT
        order_id,
        item_price,
        dense_rank() OVER (PARTITION BY seller_id ORDER BY item_price DESC) AS price_rank
    FROM (
        SELECT order_id, price AS item_price, seller_id FROM order_items
    ) sub
    WHERE item_price > 50.0
    """
    validate_sql(query)


# ------------------------------------------------------------------------------
# 2. Timeout and IPC Reliability Tests
# ------------------------------------------------------------------------------

@pytest.mark.unit
def test_ipc_payload_size_limit_rejection(tmp_path: Path) -> None:
    """Queries generating a serialized payload exceeding MAX_SERIALIZED_PAYLOAD_BYTES are rejected."""
    from tools.sql_tool import MAX_SERIALIZED_PAYLOAD_BYTES
    db_file = _make_db(tmp_path, "giant_payload.duckdb")
    with duckdb.connect(str(db_file)) as con:
        # Create a single row containing a string larger than 15MB
        con.execute("CREATE TABLE giant AS SELECT repeat('Z', 16000000) AS big_text;")

    res = execute_sql("SELECT big_text FROM giant", db_path=db_file, max_rows=1)
    assert res["status"] == "error"
    assert res["error_type"] == "PayloadSizeError"
    assert "exceeds the maximum allowed limit" in res["error_message"]


# Module-scoped mock workers for deterministic spawn-compatible process testing
def _mock_partial_writer_worker(db_path_str: str, query: str, max_rows: int, conn: Any) -> None:
    fd = conn.fileno()
    os.write(fd, b"\x00\x00")
    conn.close()


def _mock_crashing_worker(db_path_str: str, query: str, max_rows: int, conn: Any) -> None:
    conn.close()
    os._exit(42)


def _mock_hanging_cleanup_worker(db_path_str: str, query: str, max_rows: int, conn: Any) -> None:
    from tools.sql_tool import _send_framed_payload
    _send_framed_payload(conn, ("success", ["cnt"], [[10]], 1, False))
    time.sleep(10.0)


def _mock_crash_after_send_worker(db_path_str: str, query: str, max_rows: int, conn: Any) -> None:
    from tools.sql_tool import _send_framed_payload
    _send_framed_payload(conn, ("success", ["cnt"], [[10]], 1, False))
    conn.close()
    os._exit(99)


def _failing_process_factory(*args: Any, **kwargs: Any) -> Any:
    class _FailingProcess:
        def start(self) -> None:
            raise OSError("Process table full / fork failed")
    return _FailingProcess()


@pytest.mark.unit
def test_ipc_partial_interrupted_transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When worker pipe closes prematurely during transfer, parent detects EOFError without hanging."""
    db_file = _make_analytical_test_db(tmp_path)
    monkeypatch.setattr("tools.sql_tool._WORKER_TARGET", _mock_partial_writer_worker)

    start = time.perf_counter()
    res = execute_sql("SELECT count(*) FROM customers", db_path=db_file, timeout_seconds=3.0)
    elapsed = time.perf_counter() - start

    assert res["status"] == "error"
    assert res["error_type"] == "ProcessError"
    # Should complete quickly on EOF, not wait out the full 3 seconds
    assert elapsed < 2.0


@pytest.mark.unit
def test_ipc_worker_crashes_without_sending_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When worker terminates abruptly with non-zero exit code without sending data."""
    db_file = _make_analytical_test_db(tmp_path)
    monkeypatch.setattr("tools.sql_tool._WORKER_TARGET", _mock_crashing_worker)

    res = execute_sql("SELECT count(*) FROM customers", db_path=db_file, timeout_seconds=3.0)
    assert res["status"] == "error"
    assert res["error_type"] == "ProcessError"
    assert "42" in res["error_message"]


@pytest.mark.unit
def test_ipc_worker_startup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When multiprocessing.Process.start() raises, parent handles error cleanly and reaps handles."""
    db_file = _make_analytical_test_db(tmp_path)
    monkeypatch.setattr("tools.sql_tool._PROCESS_FACTORY", _failing_process_factory)

    res = execute_sql("SELECT count(*) FROM customers", db_path=db_file)
    assert res["status"] == "error"
    assert res["error_type"] == "OSError"
    assert "Failed to start worker process" in res["error_message"]


# ------------------------------------------------------------------------------
# 3. Worker Cleanup and Success Reporting Tests
# ------------------------------------------------------------------------------

@pytest.mark.unit
def test_worker_sends_success_then_hangs_in_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a worker sends a success payload but hangs in cleanup, execute_sql MUST NOT report success."""
    db_file = _make_analytical_test_db(tmp_path)
    monkeypatch.setattr("tools.sql_tool._WORKER_TARGET", _mock_hanging_cleanup_worker)

    start = time.perf_counter()
    res = execute_sql("SELECT count(*) FROM customers", db_path=db_file, timeout_seconds=0.8)
    elapsed = time.perf_counter() - start

    # Must be classified as TimeoutError, NOT success!
    assert res["status"] == "error"
    assert res["error_type"] == "TimeoutError"
    assert elapsed < 2.5

    # Verify hanging worker was terminated
    active = multiprocessing.active_children()
    assert len(active) == 0


@pytest.mark.unit
def test_worker_sends_success_then_crashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a worker sends success but terminates with exitcode != 0, execute_sql MUST NOT report success."""
    db_file = _make_analytical_test_db(tmp_path)
    monkeypatch.setattr("tools.sql_tool._WORKER_TARGET", _mock_crash_after_send_worker)

    res = execute_sql("SELECT count(*) FROM customers", db_path=db_file, timeout_seconds=3.0)
    # Must report ProcessError, NOT success!
    assert res["status"] == "error"
    assert res["error_type"] == "ProcessError"
    assert "99" in res["error_message"]

