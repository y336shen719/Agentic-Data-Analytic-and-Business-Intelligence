"""Tests for the Olist data preparation pipeline, DuckDB initialization, and data quality validation.

Organized into:
1. Fast, isolated unit tests using in-memory DuckDB and deterministic synthetic fixtures (@pytest.mark.unit).
2. End-to-end integration tests verifying production data and schema invariants (@pytest.mark.integration).
"""

import csv
from decimal import Decimal
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import duckdb
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_data import (
    DEFAULT_DB_PATH,
    DEFAULT_FK_CHECKS,
    DEFAULT_RAW_DIR,
    TABLE_SCHEMAS,
    VIEW_DEFINITIONS,
    CriticalValidationError,
    build_pipeline,
    create_analysis_views,
    run_data_validation,
    validate_foreign_keys,
    validate_primary_keys,
    validate_source_row_counts,
    validate_views_and_metrics,
)

EXPECTED_ROW_COUNTS = {
    "customers": 99441,
    "geolocation": 1000163,
    "order_items": 112650,
    "order_payments": 103886,
    "order_reviews": 99224,
    "orders": 99441,
    "products": 32951,
    "sellers": 3095,
    "category_translation": 71,
}


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def prod_db_con():
    """Provides a read-only DuckDB connection to the existing production database.
    
    Guarantees that production data/ecommerce.duckdb is never modified or overwritten.
    """
    if not DEFAULT_DB_PATH.exists():
        pytest.skip(f"Production database '{DEFAULT_DB_PATH}' does not exist. Skipping integration tests.")

    con = duckdb.connect(database=str(DEFAULT_DB_PATH), read_only=True)
    yield con
    con.close()


@pytest.fixture
def mem_db():
    """Provides an isolated, in-memory DuckDB connection for lightweight unit tests."""
    con = duckdb.connect(":memory:")
    yield con
    con.close()


@pytest.fixture
def synthetic_ecom_db(mem_db):
    """Provides an in-memory DuckDB database seeded with controlled, deterministic synthetic e-commerce data."""
    # 1. Base table DDL
    mem_db.execute("""
        CREATE TABLE customers (
            customer_id VARCHAR, customer_unique_id VARCHAR,
            customer_zip_code_prefix VARCHAR, customer_city VARCHAR, customer_state VARCHAR
        );
        CREATE TABLE orders (
            order_id VARCHAR, customer_id VARCHAR, order_status VARCHAR,
            order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP,
            order_delivered_carrier_date TIMESTAMP, order_delivered_customer_date TIMESTAMP,
            order_estimated_delivery_date TIMESTAMP
        );
        CREATE TABLE order_items (
            order_id VARCHAR, order_item_id INTEGER, product_id VARCHAR, seller_id VARCHAR,
            shipping_limit_date TIMESTAMP, price DECIMAL(18,2), freight_value DECIMAL(18,2)
        );
        CREATE TABLE order_payments (
            order_id VARCHAR, payment_sequential INTEGER, payment_type VARCHAR,
            payment_installments INTEGER, payment_value DECIMAL(18,2)
        );
        CREATE TABLE order_reviews (
            review_id VARCHAR, order_id VARCHAR, review_score INTEGER,
            review_comment_title VARCHAR, review_comment_message VARCHAR,
            review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP
        );
        CREATE TABLE products (
            product_id VARCHAR, product_category_name VARCHAR,
            product_name_lenght INTEGER, product_description_lenght INTEGER,
            product_photos_qty INTEGER, product_weight_g INTEGER,
            product_length_cm INTEGER, product_height_cm INTEGER, product_width_cm INTEGER
        );
        CREATE TABLE sellers (
            seller_id VARCHAR, seller_zip_code_prefix VARCHAR, seller_city VARCHAR, seller_state VARCHAR
        );
        CREATE TABLE category_translation (
            product_category_name VARCHAR, product_category_name_english VARCHAR
        );
    """)

    # 2. Seed deterministic rows
    # Order 1: delivered, 2 items, 2 payments (split), 1 review
    mem_db.execute("""
        INSERT INTO customers VALUES 
            ('c1', 'u1', '01001', 'Sao Paulo', 'SP'),
            ('c2', 'u2', '20000', 'Rio', 'RJ'),
            ('c3', 'u3', '30000', 'Belo Horizonte', 'MG');

        INSERT INTO orders VALUES 
            ('o1', 'c1', 'delivered', '2017-05-01 10:00:00', '2017-05-01 10:10:00', '2017-05-02 12:00:00', '2017-05-05 15:00:00', '2017-05-10 00:00:00'),
            ('o2', 'c2', 'canceled',  '2017-05-02 10:00:00', '2017-05-02 10:10:00', NULL, NULL, '2017-05-15 00:00:00'),
            ('o3', 'c3', 'shipped',   '2017-06-01 12:00:00', '2017-06-01 12:15:00', '2017-06-02 14:00:00', NULL, '2017-06-10 00:00:00');

        INSERT INTO order_items VALUES 
            ('o1', 1, 'p1', 's1', '2017-05-03 10:00:00', 100.00, 15.00),
            ('o1', 2, 'p2', 's2', '2017-05-03 10:00:00', 50.00, 10.00),
            ('o3', 1, 'p3', 's1', '2017-06-03 10:00:00', 80.00, 12.00);

        INSERT INTO order_payments VALUES 
            ('o1', 1, 'credit_card', 2, 100.00),
            ('o1', 2, 'voucher',     1, 75.00),
            ('o2', 1, 'credit_card', 1, 50.00);
            -- Note: 'o3' has zero payment records to test unpaid order preservation

        INSERT INTO order_reviews VALUES 
            ('r1', 'o1', 5, 'Great', 'Good product', '2017-05-06 00:00:00', '2017-05-07 10:00:00');

        INSERT INTO products VALUES 
            ('p1', 'informatica', 10, 20, 1, 500, 20, 10, 15),
            ('p2', 'beleza_saude', 10, 20, 1, 300, 10, 5, 10),
            ('p3', 'pc_gamer', 10, 20, 1, 1000, 40, 20, 30);

        INSERT INTO sellers VALUES 
            ('s1', '01001', 'Sao Paulo', 'SP'),
            ('s2', '13023', 'Campinas', 'SP');

        INSERT INTO category_translation VALUES 
            ('informatica', 'computers'),
            ('beleza_saude', 'health_beauty');
            -- Note: 'pc_gamer' has no translation to test fallback
    """)

    # 3. Create analysis views
    create_analysis_views(mem_db)
    return mem_db


# ==============================================================================
# Unit Tests: Primary Key & Foreign Key Validation Functions
# ==============================================================================

@pytest.mark.unit
def test_validate_primary_keys_valid_single_and_composite(mem_db):
    """Directly test validate_primary_keys on valid single-column and composite primary keys."""
    mem_db.execute("""
        CREATE TABLE test_single_pk (id VARCHAR, val VARCHAR);
        INSERT INTO test_single_pk VALUES ('k1', 'v1'), ('k2', 'v2');

        CREATE TABLE test_composite_pk (order_id VARCHAR, item_id INT, val VARCHAR);
        INSERT INTO test_composite_pk VALUES ('o1', 1, 'v1'), ('o1', 2, 'v2'), ('o2', 1, 'v3');
    """)

    schemas = {
        "test_single_pk": {"primary_key": ["id"]},
        "test_composite_pk": {"primary_key": ["order_id", "item_id"]},
    }

    results, failures = validate_primary_keys(mem_db, table_schemas=schemas)

    assert len(failures) == 0
    assert results["test_single_pk"]["is_unique"] is True
    assert results["test_single_pk"]["has_no_nulls"] is True
    assert results["test_composite_pk"]["is_unique"] is True
    assert results["test_composite_pk"]["has_no_nulls"] is True


@pytest.mark.unit
def test_validate_primary_keys_detects_duplicate_single_key(mem_db):
    """Directly test validate_primary_keys fails on duplicate single-column primary keys."""
    mem_db.execute("""
        CREATE TABLE test_dup_single (id VARCHAR, val VARCHAR);
        INSERT INTO test_dup_single VALUES ('k1', 'v1'), ('k1', 'v2_duplicate');
    """)

    schemas = {"test_dup_single": {"primary_key": ["id"]}}
    results, failures = validate_primary_keys(mem_db, table_schemas=schemas)

    assert len(failures) == 1
    assert "duplicate primary key groups" in failures[0]
    assert results["test_dup_single"]["duplicate_groups"] == 1
    assert results["test_dup_single"]["excess_duplicate_rows"] == 1


@pytest.mark.unit
def test_validate_primary_keys_detects_duplicate_composite_key(mem_db):
    """Directly test validate_primary_keys fails on duplicate composite keys."""
    mem_db.execute("""
        CREATE TABLE test_dup_composite (order_id VARCHAR, item_id INT);
        INSERT INTO test_dup_composite VALUES ('o1', 1), ('o1', 1);
    """)

    schemas = {"test_dup_composite": {"primary_key": ["order_id", "item_id"]}}
    results, failures = validate_primary_keys(mem_db, table_schemas=schemas)

    assert len(failures) == 1
    assert "duplicate primary key groups" in failures[0]
    assert results["test_dup_composite"]["is_unique"] is False


@pytest.mark.unit
def test_validate_primary_keys_detects_null_primary_keys(mem_db):
    """Directly test validate_primary_keys fails when primary key contains NULL."""
    mem_db.execute("""
        CREATE TABLE test_null_pk (order_id VARCHAR, item_id INT);
        INSERT INTO test_null_pk VALUES ('o1', 1), ('o1', NULL), (NULL, 2);
    """)

    schemas = {"test_null_pk": {"primary_key": ["order_id", "item_id"]}}
    results, failures = validate_primary_keys(mem_db, table_schemas=schemas)

    assert len(failures) == 1
    assert "2 rows with NULL in primary key" in failures[0]
    assert results["test_null_pk"]["null_count"] == 2
    assert results["test_null_pk"]["has_no_nulls"] is False


@pytest.mark.unit
def test_validate_primary_keys_string_collision_regression(mem_db):
    """Ensure multi-column GROUP BY validation in validate_primary_keys prevents string concatenation collisions."""
    mem_db.execute("""
        CREATE TABLE test_collision (part_a VARCHAR, part_b VARCHAR);
        -- 'foo_bar' + '_' + 'baz' == 'foo' + '_' + 'bar_baz' under naive concatenation
        INSERT INTO test_collision VALUES 
            ('foo_bar', 'baz'),
            ('foo', 'bar_baz');
    """)

    schemas = {"test_collision": {"primary_key": ["part_a", "part_b"]}}
    results, failures = validate_primary_keys(mem_db, table_schemas=schemas)

    assert len(failures) == 0, "GROUP BY uniqueness should not falsely collide string pairs"
    assert results["test_collision"]["is_unique"] is True
    assert results["test_collision"]["duplicate_groups"] == 0


@pytest.mark.unit
def test_validate_foreign_keys_valid_and_orphans(mem_db):
    """Directly test validate_foreign_keys for valid references and orphan detections."""
    mem_db.execute("""
        CREATE TABLE parent_tbl (p_id VARCHAR);
        CREATE TABLE child_tbl (c_id VARCHAR, p_id VARCHAR);

        INSERT INTO parent_tbl VALUES ('p1'), ('p2');
        INSERT INTO child_tbl VALUES ('c1', 'p1'), ('c2', 'p2');
    """)

    valid_checks = [
        ("child_tbl.p_id -> parent_tbl.p_id",
         "SELECT count(*) FROM child_tbl c LEFT JOIN parent_tbl p ON c.p_id = p.p_id WHERE p.p_id IS NULL")
    ]
    results_valid, failures_valid = validate_foreign_keys(mem_db, fk_checks=valid_checks)
    assert len(failures_valid) == 0
    assert results_valid["child_tbl.p_id -> parent_tbl.p_id"]["is_valid"] is True

    # Inject orphan
    mem_db.execute("INSERT INTO child_tbl VALUES ('c3', 'p_nonexistent')")
    results_orphan, failures_orphan = validate_foreign_keys(mem_db, fk_checks=valid_checks)
    assert len(failures_orphan) == 1
    assert "orphan records" in failures_orphan[0]
    assert results_orphan["child_tbl.p_id -> parent_tbl.p_id"]["orphan_count"] == 1


# ==============================================================================
# Unit Tests: Source Row Count & View/Metric Validation Functions
# ==============================================================================

@pytest.mark.unit
def test_validate_source_row_counts_direct(tmp_path, mem_db):
    """Directly test CSV-aware validate_source_row_counts against small synthetic CSVs."""
    raw_dir = tmp_path / "synthetic_raw"
    raw_dir.mkdir()

    csv_file = raw_dir / "sample_data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "comment"])
        writer.writerow(["1", "simple line"])
        writer.writerow(["2", "multiline\nquoted\ncomment"])  # Quoted multiline CSV record
        writer.writerow(["3", "another line"])

    schemas = {
        "sample_tbl": {
            "file": "sample_data.csv",
            "columns": {"id": "VARCHAR", "comment": "VARCHAR"}
        }
    }

    # Ingest using DuckDB CSV reader
    mem_db.execute(f"CREATE TABLE sample_tbl AS SELECT * FROM read_csv('{csv_file.as_posix()}', header=True)")

    # 1. Matching case
    results, failures = validate_source_row_counts(mem_db, raw_dir, table_schemas=schemas)
    assert len(failures) == 0
    assert results["sample_tbl"]["source_csv_rows"] == 3
    assert results["sample_tbl"]["imported_db_rows"] == 3

    # 2. Mismatch case: delete one row from database
    mem_db.execute("DELETE FROM sample_tbl WHERE id = '3'")
    results_mismatch, failures_mismatch = validate_source_row_counts(mem_db, raw_dir, table_schemas=schemas)
    assert len(failures_mismatch) == 1
    assert "Row count mismatch in 'sample_tbl'" in failures_mismatch[0]


@pytest.mark.unit
def test_validate_views_and_metrics_direct(synthetic_ecom_db):
    """Directly test validate_views_and_metrics function on matching and mismatched states."""
    # 1. Matching state: should report zero failures
    results, failures = validate_views_and_metrics(synthetic_ecom_db)
    assert len(failures) == 0
    assert results["order_summary_row_count"] == 3
    assert results["financial_reconciliation_exact"] is True

    # 2. Mismatch state: replace order_summary with a view that zeroes out financial totals
    synthetic_ecom_db.execute("""
        CREATE OR REPLACE VIEW order_summary AS 
        SELECT o.order_id, 0::INT AS item_count, 0.00::DECIMAL(18,2) AS item_subtotal, 
               0.00::DECIMAL(18,2) AS freight_total, 0.00::DECIMAL(18,2) AS payment_value_total
        FROM orders o;
    """)

    results_err, failures_err = validate_views_and_metrics(synthetic_ecom_db)
    assert len(failures_err) > 0
    assert any("Financial mismatch: order_summary item subtotal" in f for f in failures_err)
    assert any("Item count mismatch" in f for f in failures_err)


@pytest.mark.unit
def test_run_data_validation_raises_critical_validation_error(synthetic_ecom_db, tmp_path):
    """Verify run_data_validation executes complete suite and raises CriticalValidationError on critical defect."""
    # Inject an orphan foreign key reference
    synthetic_ecom_db.execute("INSERT INTO order_items VALUES ('o_orphan', 1, 'p1', 's1', '2017-05-03 10:00:00', 10.00, 5.00)")

    fk_checks = [
        ("order_items.order_id -> orders.order_id",
         "SELECT count(*) FROM order_items oi LEFT JOIN orders o ON oi.order_id = o.order_id WHERE o.order_id IS NULL")
    ]

    with pytest.raises(CriticalValidationError) as exc_info:
        run_data_validation(
            synthetic_ecom_db,
            raw_dir=tmp_path,
            table_schemas={},  # Empty schemas bypass row count and PK checks
            fk_checks=fk_checks,
        )

    assert "Critical validation failures encountered" in str(exc_info.value)
    assert "Foreign key violation: order_items.order_id -> orders.order_id has 1 orphan records" in str(exc_info.value)


# ==============================================================================
# Unit Tests: Analysis-Ready Views Logic on Synthetic Data
# ==============================================================================

@pytest.mark.unit
def test_synthetic_order_summary_view_logic(synthetic_ecom_db):
    """Verify order_summary view logic: 1 row per order, anti-fanout, metrics, and zero defaults."""
    # 1. Grain and set equality
    orders_count = synthetic_ecom_db.execute("SELECT count(*) FROM orders").fetchone()[0]
    view_count = synthetic_ecom_db.execute("SELECT count(*) FROM order_summary").fetchone()[0]
    assert view_count == orders_count == 3

    order_ids_orders = set(r[0] for r in synthetic_ecom_db.execute("SELECT order_id FROM orders").fetchall())
    order_ids_view = set(r[0] for r in synthetic_ecom_db.execute("SELECT order_id FROM order_summary").fetchall())
    assert order_ids_view == order_ids_orders == {"o1", "o2", "o3"}

    # 2. Check Order 1: delivered, 2 items, 2 payments (split payment), 1 review
    o1 = synthetic_ecom_db.execute("""
        SELECT item_count, item_subtotal, freight_total, order_gross_value, 
               payment_count, payment_value_total, is_delivered, delivery_duration_days
        FROM order_summary WHERE order_id = 'o1'
    """).fetchone()

    assert o1[0] == 2                                   # 2 items
    assert o1[1] == Decimal("150.00")                   # 100.00 + 50.00
    assert o1[2] == Decimal("25.00")                    # 15.00 + 10.00
    assert o1[3] == Decimal("175.00")                   # 150.00 + 25.00
    assert o1[4] == 2                                   # 2 payment records
    assert o1[5] == Decimal("175.00")                   # 100.00 + 75.00
    assert o1[6] is True                                # is_delivered
    assert o1[7] == pytest.approx(4.21, rel=1e-2)       # May 1 10:00 to May 5 15:00 = 4.208 days

    # 3. Check Order 2: canceled, 0 items, 1 payment
    o2 = synthetic_ecom_db.execute("""
        SELECT item_count, item_subtotal, freight_total, order_gross_value, 
               payment_count, payment_value_total, is_delivered, delivery_duration_days
        FROM order_summary WHERE order_id = 'o2'
    """).fetchone()

    assert o2[0] == 0                                   # 0 items
    assert o2[1] == Decimal("0.00")                     # modeling default
    assert o2[2] == Decimal("0.00")
    assert o2[3] == Decimal("0.00")
    assert o2[4] == 1                                   # 1 payment
    assert o2[5] == Decimal("50.00")
    assert o2[6] is False                               # not delivered
    assert o2[7] is None                                # null duration

    # 4. Check Order 3: shipped, 1 item, 0 payments (unpaid or pending)
    o3 = synthetic_ecom_db.execute("""
        SELECT item_count, item_subtotal, freight_total, payment_count, payment_value_total, is_delivered
        FROM order_summary WHERE order_id = 'o3'
    """).fetchone()

    assert o3[0] == 1                                   # 1 item
    assert o3[1] == Decimal("80.00")                   # 80.00 subtotal
    assert o3[2] == Decimal("12.00")                   # 12.00 freight
    assert o3[3] == 0                                   # 0 payment records
    assert o3[4] == Decimal("0.00")                     # 0.00 default total
    assert o3[5] is False                               # not delivered


@pytest.mark.unit
def test_order_summary_order_without_payments_preservation(mem_db):
    """Explicitly verify that an order with zero payment records yields payment_count=0,
    payment_value_total=0.00, and is fully preserved in order_summary."""
    mem_db.execute("""
        CREATE TABLE customers (
            customer_id VARCHAR, customer_unique_id VARCHAR,
            customer_zip_code_prefix VARCHAR, customer_city VARCHAR, customer_state VARCHAR
        );
        CREATE TABLE orders (
            order_id VARCHAR, customer_id VARCHAR, order_status VARCHAR,
            order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP,
            order_delivered_carrier_date TIMESTAMP, order_delivered_customer_date TIMESTAMP,
            order_estimated_delivery_date TIMESTAMP
        );
        CREATE TABLE order_items (
            order_id VARCHAR, order_item_id INTEGER, product_id VARCHAR, seller_id VARCHAR,
            shipping_limit_date TIMESTAMP, price DECIMAL(18,2), freight_value DECIMAL(18,2)
        );
        CREATE TABLE order_payments (
            order_id VARCHAR, payment_sequential INTEGER, payment_type VARCHAR,
            payment_installments INTEGER, payment_value DECIMAL(18,2)
        );
        CREATE TABLE order_reviews (
            review_id VARCHAR, order_id VARCHAR, review_score INTEGER,
            review_comment_title VARCHAR, review_comment_message VARCHAR,
            review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP
        );

        INSERT INTO customers VALUES ('c100', 'u100', '01001', 'City', 'SP');
        INSERT INTO orders VALUES (
            'o_unpaid', 'c100', 'delivered',
            '2017-05-01 10:00:00', '2017-05-01 10:10:00',
            '2017-05-02 12:00:00', '2017-05-05 15:00:00', '2017-05-10 00:00:00'
        );
        INSERT INTO order_items VALUES ('o_unpaid', 1, 'p1', 's1', '2017-05-03 10:00:00', 45.00, 10.00);
        -- Zero payments inserted into order_payments
    """)

    create_analysis_views(mem_db, view_definitions={"order_summary": VIEW_DEFINITIONS["order_summary"]})

    row = mem_db.execute("""
        SELECT order_id, item_count, item_subtotal, freight_total, order_gross_value,
               payment_count, payment_value_total, is_delivered
        FROM order_summary
        WHERE order_id = 'o_unpaid'
    """).fetchone()

    assert row is not None, "Order with zero payments must be preserved in order_summary"
    assert row[0] == "o_unpaid"
    assert row[1] == 1
    assert row[2] == Decimal("45.00")
    assert row[3] == Decimal("10.00")
    assert row[4] == Decimal("55.00")
    assert row[5] == 0, "payment_count must default to 0 for orders without payment records"
    assert row[6] == Decimal("0.00"), "payment_value_total must default to 0.00 for orders without payment records"
    assert row[7] is True


@pytest.mark.unit
def test_synthetic_category_sales_summary_logic(synthetic_ecom_db):
    """Verify category_sales_summary logic: monthly grouping, translation fallback, delivered filtering."""
    rows = synthetic_ecom_db.execute("""
        SELECT year_month, category_name_english, total_orders, total_items_sold, 
               total_item_revenue, total_freight_value, total_gross_sales, 
               delivered_orders, delivered_items_sold, delivered_item_revenue
        FROM category_sales_summary
        ORDER BY year_month, category_name_english
    """).fetchall()

    assert len(rows) == 3

    # Row 1: 2017-05, computers (Order 1, item 1, delivered)
    r1 = rows[0]
    assert r1[0] == "2017-05"
    assert r1[1] == "computers"
    assert r1[2] == 1                         # total_orders
    assert r1[3] == 1                         # total_items_sold
    assert r1[4] == Decimal("100.00")         # total_item_revenue
    assert r1[5] == Decimal("15.00")          # total_freight_value
    assert r1[6] == Decimal("115.00")         # total_gross_sales
    assert r1[7] == 1                         # delivered_orders
    assert r1[8] == 1                         # delivered_items_sold
    assert r1[9] == Decimal("100.00")         # delivered_item_revenue

    # Row 2: 2017-05, health_beauty (Order 1, item 2, delivered)
    r2 = rows[1]
    assert r2[0] == "2017-05"
    assert r2[1] == "health_beauty"
    assert r2[4] == Decimal("50.00")
    assert r2[9] == Decimal("50.00")

    # Row 3: 2017-06, pc_gamer (Order 3, item 1, shipped - NOT delivered, fallback to original category name)
    r3 = rows[2]
    assert r3[0] == "2017-06"
    assert r3[1] == "pc_gamer", "Category without English translation must fallback to original category name"
    assert r3[3] == 1                         # 1 item sold
    assert r3[4] == Decimal("80.00")          # total_item_revenue
    assert r3[7] == 0                         # delivered_orders = 0
    assert r3[8] == 0                         # delivered_items_sold = 0
    assert r3[9] == Decimal("0.00")           # delivered_item_revenue = 0.00


@pytest.mark.unit
def test_category_sales_summary_translation_fallback_isolated(mem_db):
    """Verify that products with categories missing English translation fall back to original Portuguese category name."""
    mem_db.execute("""
        CREATE TABLE orders (
            order_id VARCHAR, customer_id VARCHAR, order_status VARCHAR,
            order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP,
            order_delivered_carrier_date TIMESTAMP, order_delivered_customer_date TIMESTAMP,
            order_estimated_delivery_date TIMESTAMP
        );
        CREATE TABLE order_items (
            order_id VARCHAR, order_item_id INTEGER, product_id VARCHAR, seller_id VARCHAR,
            shipping_limit_date TIMESTAMP, price DECIMAL(18,2), freight_value DECIMAL(18,2)
        );
        CREATE TABLE products (
            product_id VARCHAR, product_category_name VARCHAR,
            product_name_lenght INTEGER, product_description_lenght INTEGER,
            product_photos_qty INTEGER, product_weight_g INTEGER,
            product_length_cm INTEGER, product_height_cm INTEGER, product_width_cm INTEGER
        );
        CREATE TABLE category_translation (
            product_category_name VARCHAR, product_category_name_english VARCHAR
        );

        INSERT INTO orders VALUES ('o1', 'c1', 'delivered', '2017-08-10 12:00:00', NULL, NULL, NULL, NULL);
        INSERT INTO products VALUES ('p_untranslated', 'artigos_de_natal', NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        INSERT INTO order_items VALUES ('o1', 1, 'p_untranslated', 's1', '2017-08-15 00:00:00', 99.90, 15.00);
        -- 'artigos_de_natal' has NO translation in category_translation
    """)

    create_analysis_views(mem_db, view_definitions={"category_sales_summary": VIEW_DEFINITIONS["category_sales_summary"]})

    row = mem_db.execute("""
        SELECT year_month, product_category_name, category_name_english, total_orders, total_items_sold, total_item_revenue
        FROM category_sales_summary
    """).fetchone()

    assert row is not None
    assert row[0] == "2017-08"
    assert row[1] == "artigos_de_natal"
    assert row[2] == "artigos_de_natal", "category_name_english must fall back to original category name when translation is missing"
    assert row[3] == 1
    assert row[4] == 1
    assert row[5] == Decimal("99.90")


# ==============================================================================
# Unit Tests: Transaction Rollback & Idempotency
# ==============================================================================

@pytest.mark.unit
def test_transaction_rollback_with_synthetic_csv(tmp_path):
    """Verify DuckDB explicit transaction rolls back cleanly on critical validation failure using synthetic CSVs."""
    temp_db_path = tmp_path / "test_rollback.duckdb"
    con = duckdb.connect(str(temp_db_path))

    # Pre-seed committed data
    con.execute("CREATE TABLE pre_existing_marker (id INT, status VARCHAR)")
    con.execute("INSERT INTO pre_existing_marker VALUES (42, 'committed_and_safe')")
    con.close()

    # Create synthetic raw directory with 2 CSVs
    raw_dir = tmp_path / "synth_raw"
    raw_dir.mkdir()

    cust_csv = raw_dir / "customers.csv"
    with open(cust_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["customer_id", "name"])
        writer.writerow(["c1", "Alice"])
        writer.writerow(["c1", "Alice_Duplicate"])  # Injected duplicate PK

    orders_csv = raw_dir / "orders.csv"
    with open(orders_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "customer_id"])
        writer.writerow(["o1", "c1"])

    synth_schemas = {
        "customers": {
            "file": "customers.csv",
            "primary_key": ["customer_id"],
            "columns": {"customer_id": "VARCHAR", "name": "VARCHAR"},
        },
        "orders": {
            "file": "orders.csv",
            "primary_key": ["order_id"],
            "columns": {"order_id": "VARCHAR", "customer_id": "VARCHAR"},
        },
    }

    # Execute build_pipeline expecting failure
    with pytest.raises(CriticalValidationError) as exc_info:
        build_pipeline(
            raw_dir=raw_dir,
            db_path=temp_db_path,
            table_schemas=synth_schemas,
            view_definitions={},
            fk_checks=[],
        )

    assert "duplicate primary key groups" in str(exc_info.value)

    # Reconnect and verify atomicity
    verify_con = duckdb.connect(str(temp_db_path), read_only=True)
    tables = [r[0] for r in verify_con.execute("SHOW TABLES").fetchall()]
    marker_content = verify_con.execute("SELECT * FROM pre_existing_marker").fetchall()
    verify_con.close()

    assert tables == ["pre_existing_marker"]
    assert marker_content == [(42, "committed_and_safe")]


@pytest.mark.unit
def test_pipeline_idempotency_on_temporary_db(tmp_path):
    """Run pipeline twice on a temporary database to verify idempotent repeat execution without row duplication."""
    temp_db = tmp_path / "test_idempotent.duckdb"
    raw_dir = tmp_path / "idempotent_raw"
    raw_dir.mkdir()

    # Create simple valid CSV
    sample_csv = raw_dir / "items.csv"
    with open(sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "val"])
        writer.writerow(["i1", "10"])
        writer.writerow(["i2", "20"])

    schemas = {
        "items": {
            "file": "items.csv",
            "primary_key": ["item_id"],
            "columns": {"item_id": "VARCHAR", "val": "INTEGER"},
        }
    }

    # Run 1
    con1 = build_pipeline(
        raw_dir=raw_dir,
        db_path=temp_db,
        table_schemas=schemas,
        view_definitions={},
        fk_checks=[],
    )
    count1 = con1.execute("SELECT count(*) FROM items").fetchone()[0]
    con1.close()

    # Run 2
    con2 = build_pipeline(
        raw_dir=raw_dir,
        db_path=temp_db,
        table_schemas=schemas,
        view_definitions={},
        fk_checks=[],
    )
    count2 = con2.execute("SELECT count(*) FROM items").fetchone()[0]
    con2.close()

    assert count1 == count2 == 2


# ==============================================================================
# Integration Tests: Full Olist Dataset Validation
# ==============================================================================

@pytest.mark.integration
def test_prod_all_base_tables_exist_and_match_raw_sources(prod_db_con):
    """Verify that all 9 base tables exist and their row counts match Olist raw source files."""
    tables = [row[0] for row in prod_db_con.execute("SHOW TABLES").fetchall()]
    for expected_table, expected_count in EXPECTED_ROW_COUNTS.items():
        assert expected_table in tables, f"Expected table '{expected_table}' not found in DuckDB"
        actual_count = prod_db_con.execute(f"SELECT count(*) FROM {expected_table}").fetchone()[0]
        assert actual_count == expected_count, (
            f"Row count mismatch for {expected_table}: expected {expected_count}, got {actual_count}"
        )


@pytest.mark.integration
def test_prod_validation_functions_pass(prod_db_con):
    """Directly test production validation functions on the loaded Olist database."""
    # 1. Source row count reconciliation
    res_src, fail_src = validate_source_row_counts(prod_db_con, DEFAULT_RAW_DIR)
    assert len(fail_src) == 0, f"Source row count failures: {fail_src}"

    # 2. Primary key validation
    res_pk, fail_pk = validate_primary_keys(prod_db_con)
    assert len(fail_pk) == 0, f"Primary key failures: {fail_pk}"

    # 3. Foreign key referential integrity
    res_fk, fail_fk = validate_foreign_keys(prod_db_con)
    assert len(fail_fk) == 0, f"Foreign key failures: {fail_fk}"

    # 4. View and financial metrics reconciliation
    res_views, fail_views = validate_views_and_metrics(prod_db_con)
    assert len(fail_views) == 0, f"View reconciliation failures: {fail_views}"


@pytest.mark.integration
def test_prod_timestamp_and_decimal_types(prod_db_con):
    """Verify production column types: TIMESTAMP for dates, DECIMAL(18,2) for money, VARCHAR for IDs."""
    timestamp_checks = [
        ("orders", "order_purchase_timestamp"),
        ("orders", "order_approved_at"),
        ("orders", "order_delivered_carrier_date"),
        ("orders", "order_delivered_customer_date"),
        ("orders", "order_estimated_delivery_date"),
        ("order_items", "shipping_limit_date"),
        ("order_reviews", "review_creation_date"),
        ("order_reviews", "review_answer_timestamp"),
    ]
    for table_name, col_name in timestamp_checks:
        col_type = prod_db_con.execute(f"""
            SELECT data_type FROM information_schema.columns 
            WHERE table_name = '{table_name}' AND column_name = '{col_name}'
        """).fetchone()[0]
        assert "TIMESTAMP" in col_type.upper()

    decimal_checks = [
        ("order_items", "price"),
        ("order_items", "freight_value"),
        ("order_payments", "payment_value"),
    ]
    for table_name, col_name in decimal_checks:
        col_type = prod_db_con.execute(f"""
            SELECT data_type FROM information_schema.columns 
            WHERE table_name = '{table_name}' AND column_name = '{col_name}'
        """).fetchone()[0]
        assert "DECIMAL(18,2)" in col_type.upper()


@pytest.mark.integration
def test_prod_order_summary_view_grain_and_set_equality(prod_db_con):
    """Verify order_summary has 99,441 rows, exact 1:1 match with orders, and preserves zero-item orders."""
    orders_count = prod_db_con.execute("SELECT count(*) FROM orders").fetchone()[0]
    view_count = prod_db_con.execute("SELECT count(*) FROM order_summary").fetchone()[0]
    distinct_orders = prod_db_con.execute("SELECT count(DISTINCT order_id) FROM order_summary").fetchone()[0]

    assert view_count == orders_count == 99441
    assert distinct_orders == view_count

    # Verify set equality between orders.order_id and order_summary.order_id
    mismatches = prod_db_con.execute("""
        SELECT count(*) 
        FROM orders o 
        FULL OUTER JOIN order_summary s ON o.order_id = s.order_id 
        WHERE o.order_id IS NULL OR s.order_id IS NULL
    """).fetchone()[0]
    assert mismatches == 0, "order_summary order_id set does not match orders.order_id set"

    # Verify zero-item orders modeling default
    zero_item_count = prod_db_con.execute("SELECT count(*) FROM order_summary WHERE item_count = 0").fetchone()[0]
    assert zero_item_count == 775


@pytest.mark.integration
def test_prod_category_sales_summary_grain_and_aggregations(prod_db_con):
    """Verify category_sales_summary has 1,283 rows and no duplicate (year_month, category) pairs."""
    view_count = prod_db_con.execute("SELECT count(*) FROM category_sales_summary").fetchone()[0]
    assert view_count == 1283

    dup_groups = prod_db_con.execute("""
        SELECT count(*) 
        FROM (
            SELECT year_month, category_name_english 
            FROM category_sales_summary 
            GROUP BY year_month, category_name_english 
            HAVING count(*) > 1
        )
    """).fetchone()[0]
    assert dup_groups == 0, "category_sales_summary has duplicate month-category pairs"

    # Verify item counts match order_items total
    items_sold = prod_db_con.execute("SELECT sum(total_items_sold) FROM category_sales_summary").fetchone()[0]
    assert items_sold == 112650


@pytest.mark.integration
def test_prod_financial_aggregations_anti_fanout_exact_decimal(prod_db_con):
    """Verify exact decimal financial reconciliation across base tables and views."""
    raw_item_sum = prod_db_con.execute("SELECT sum(price) FROM order_items").fetchone()[0]
    raw_freight_sum = prod_db_con.execute("SELECT sum(freight_value) FROM order_items").fetchone()[0]
    raw_payment_sum = prod_db_con.execute("SELECT sum(payment_value) FROM order_payments").fetchone()[0]

    os_item_sum = prod_db_con.execute("SELECT sum(item_subtotal) FROM order_summary").fetchone()[0]
    os_freight_sum = prod_db_con.execute("SELECT sum(freight_total) FROM order_summary").fetchone()[0]
    os_payment_sum = prod_db_con.execute("SELECT sum(payment_value_total) FROM order_summary").fetchone()[0]

    cat_item_sum = prod_db_con.execute("SELECT sum(total_item_revenue) FROM category_sales_summary").fetchone()[0]
    cat_freight_sum = prod_db_con.execute("SELECT sum(total_freight_value) FROM category_sales_summary").fetchone()[0]

    assert os_item_sum == raw_item_sum == Decimal("13591643.70")
    assert os_freight_sum == raw_freight_sum == Decimal("2251909.54")
    assert os_payment_sum == raw_payment_sum == Decimal("16008872.12")
    assert cat_item_sum == raw_item_sum == Decimal("13591643.70")
    assert cat_freight_sum == raw_freight_sum == Decimal("2251909.54")


@pytest.mark.integration
def test_prod_non_additive_metrics_and_weighted_average(prod_db_con):
    """Verify non-additive metrics behavior: distinct orders cannot be summed across categories."""
    total_distinct_orders = prod_db_con.execute(
        "SELECT count(DISTINCT order_id) FROM order_items"
    ).fetchone()[0]

    sum_category_orders = prod_db_con.execute(
        "SELECT sum(total_orders) FROM category_sales_summary"
    ).fetchone()[0]

    assert sum_category_orders > total_distinct_orders, "Sum of distinct counts across categories should produce double-counting"

    # Verify weighted average reconstruction
    true_weighted_avg = prod_db_con.execute("""
        SELECT round(sum(total_item_revenue) / sum(total_items_sold), 2)
        FROM category_sales_summary
    """).fetchone()[0]

    overall_item_avg = prod_db_con.execute("""
        SELECT round(sum(price) / count(*), 2)
        FROM order_items
    """).fetchone()[0]

    assert true_weighted_avg == overall_item_avg
