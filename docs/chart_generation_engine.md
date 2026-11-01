# Chart Generation Engine Specification

## 1. Overview

`ChartGenerationEngine` ([tools/chart_generation.py](file:///Users/yimingshen20000719/Desktop/Agentic AI System/tools/chart_generation.py)) is a secure, deterministic, reusable visualization component that translates stored datasets in `DatasetStore` into interactive Plotly charts.

## 2. Six Supported Chart Types & Parameters

| Chart Type | Key Use Cases | Required Columns | Optional Parameters |
|---|---|---|---|
| `bar` | Category revenue, orders by state, GroupBy results, Top-N | `x_column` (categorical/string), `y_column` (numeric) | `orientation` ('v'/'h'), `sort` ('asc'/'desc'), `top_n` (int), `color_column` (str), `x_label`, `y_label`, `title` |
| `line` | Monthly revenue trend, growth rate over time | `x_column` (temporal/string), `y_column` (numeric) | `markers` (bool), `sort_x` (bool, default True), `color_column` (str), `x_label`, `y_label`, `title` |
| `scatter` | Price vs freight, correlation exploration, outlier visualization | `x_column` (numeric), `y_column` (numeric) | `color_column` (str), `outlier_column` (str, low-cardinality indicator), `x_label`, `y_label`, `title` |
| `histogram` | Order value distribution, delivery durations | `x_column` (numeric) | `nbins` (int, 1..max_bins), `color_column` (str), `x_label`, `title` |
| `box` | Delivery duration distributions, price IQR comparisons | `y_column` (numeric) | `x_column` (categorical grouping), `points` ('outliers'/'all'/False), `color_column` (str), `x_label`, `y_label`, `title` |
| `heatmap` | Feature correlation matrix (Pearson/Spearman) | None required (uses all numeric or `columns`) | `columns` (list[str]), `correlation_method` ('pearson'/'spearman'), `annotated` (bool), `is_precomputed` (bool), `title` |

## 3. Data Integrity & Validation Policies

### A. Undefined Correlations
- Zero-variance columns (e.g. constant values) have undefined correlation coefficients.
- Preserved strictly as `None` (JSON `null`) on heatmap matrices rather than being misrepresented as `0.0`.
- Text annotations show `"N/A"` for undefined pairs.
- Metadata reports `undefined_correlations_count` and an explicit warning.

### B. Strict Numeric Conversion & Missing Values
- **Valid values**: finite integers, floats, NumPy scalars, finite Decimals, and valid numeric strings (e.g., `'100'`).
- **Invalid non-null values**: any string or object that cannot be parsed numerically (e.g., `'unknown'`, `'invalid'`) immediately raises `InvalidColumnError` (no silent NaN conversion).
- **Non-finite values**: positive/negative infinity and `Decimal('Infinity')` are strictly rejected with `InvalidColumnError`.
- **All-null columns**: rejected with `InvalidColumnError`.
- **Missing value transparency**: valid numeric observations mixed with genuine nulls (`None`, `NaN`, `pd.NA`) are filtered for plotting. Metadata accurately reports `total_rows`, `rows_used`, and `excluded_rows`.

### C. Strict Parameter Types
- Boolean parameters (`sort_x`, `markers`, `annotated`, `is_precomputed`) require strict boolean types (`True`/`False`), rejecting implicit coercion of strings like `'false'`, integers like `0`, or `None`.
- Box plot `points` strictly accepts `'outliers'`, `'all'`, or `False`.
- Optional strings (`color_column`, `outlier_column`, `x_label`, `y_label`, `orientation`, `sort`) require valid strings.

### D. Grouping Cardinality & Trace Bounds
- `max_categories` limits trace explosion across all grouped charts:
  - Bar: categorical axis cardinality (unless `top_n` is supplied) and `color_column`.
  - Line, Scatter, Histogram, Box: `color_column` cardinality.
  - Scatter: `outlier_column` cardinality.
  - Box: `x_column` grouping cardinality.

### E. Top-N Input Row Semantics vs Category Aggregation
- `top_n` selects the Top-N input rows after sorting by the numeric measure; it does **not** perform automatic category-level aggregation.
- If a dataset contains duplicate categories (e.g. two rows for 'Electronics' with values 100.0 and 200.0), Top-N with `sort='desc'` selects the individual 200.0 row, and never silently aggregates Electronics into 300.0.
- For business questions requiring Top-N aggregated categories (e.g. "Top 5 Product Categories by Total Revenue"), grouping and aggregation must occur first via SQL (`GROUP BY category`) or `PythonAnalysisEngine.analyze('groupby')`, followed by chart sorting and Top-N selection. ChartGenerationEngine never performs silent or implicit aggregation.

## 4. Python Analysis Engine Compatibility & Handoff Contract

The existing `PythonAnalysisEngine` returns structured Python dictionaries/lists:

- `growth_rate`: Returns `List[Dict[str, Any]]` with `period`, `value`, `growth_rate`. The first period has `growth_rate: None`, which is safely excluded from line charts and recorded in `excluded_rows`.
- `groupby`: Returns `List[Dict[str, Any]]` with group keys and `{agg_col}_{agg_func}`.
- `describe`: Returns `Dict[str, Dict[str, Any]]` summarizing numeric statistics.
- `correlation`: Returns `Dict[str, Dict[str, Optional[float]]]`. Converted by adapter preserving row labels in `column_name`. Can be plotted directly on Heatmap using `is_precomputed: True` without recalculating correlations.
- `detect_outliers`: Returns statistical bounds and a sample preview (`outliers_preview`), not a row-level dataset with indicator flags. Highlighting outliers on scatter plots requires an explicit indicator column.

### Adapter: `analysis_result_to_dataset_payload`
`tools.chart_generation.analysis_result_to_dataset_payload(analysis_result)` adapts real outputs from `growth_rate`, `groupby`, `describe`, and `correlation` into `{"status": "success", "columns": [...], "rows": [...], "row_count": len(rows)}` for storage in `DatasetStore`.

## 5. LangChain Tool Wrapper & ChartRegistry (Step 6.2)

### A. LangChain `generate_chart` Tool
`create_chart_generation_tool` ([agent/tool_wrappers.py](file:///Users/yimingshen20000719/Desktop/Agentic AI System/agent/tool_wrappers.py)) exposes the engine as a LangChain `BaseTool` with strict Pydantic argument validation (`ChartGenerationInput`):
- `dataset_id: str`
- `chart_type: str` (exactly six allowlisted types: `bar`, `line`, `scatter`, `histogram`, `box`, `heatmap`)
- `x_column: Optional[str]`
- `y_column: Optional[str]`
- `params: Optional[Dict[str, Any]]`

### B. ChartRegistry (Lightweight LLM References vs Full Figure)
`ChartRegistry` ([tools/chart_registry.py](file:///Users/yimingshen20000719/Desktop/Agentic AI System/tools/chart_registry.py)) keeps bulky Plotly Figure JSON out of the LLM context window:
- Returns compact tool response:
  ```json
  {
    "status": "success",
    "chart_id": "chart_7a8b9c0d1e2f",
    "dataset_id": "ds_sales",
    "chart_type": "line",
    "title": "Monthly Revenue Trend",
    "metadata": {
      "rows_used": 24,
      "total_rows": 24
    }
  }
  ```
- Retains complete Plotly figure specification in application memory.
- Enforces strict memory and item bounds (`max_charts`, `max_memory_bytes`).
- Ensures session isolation without cross-session chart leakage.

### C. Automatic Derived Dataset Handoff
When `execute_python_analysis` executes successfully, supported operations (`growth_rate`, `groupby`, `correlation`, `describe`) are automatically converted and saved to the shared `DatasetStore`, returning `derived_dataset_id` to the LLM. If conversion is unsupported (`detect_outliers`) or storage fails (e.g. capacity limits), `derived_dataset_error` conveys the exact reason safely without failing the primary calculation.

### D. Engine & Store Identity Consistency
When pre-instantiated engines are supplied alongside a `DatasetStore`, factories strictly validate object identity (`engine.dataset_store is dataset_store`), raising a `ValueError` immediately upon mismatch.

## 6. Scope Boundaries & Deferred Work

- **Completed in Step 6.1**:
  - `tools/chart_generation.py` engine and adapter implementation.
  - Strict numeric validation, missing value transparency, and undefined correlation preservation.
  - 69 comprehensive unit and integration tests (`tests/test_chart_generation.py`).
- **Completed in Step 6.2**:
  - LangChain `generate_chart` tool wrapper with strict Pydantic schema in `agent/tool_wrappers.py`.
  - Accurate parameter names: `sort`, `color_column`, `nbins` (avoiding misleading aliases).
  - In-memory session-scoped `ChartRegistry` in `tools/chart_registry.py`.
  - Automatic derived dataset registration in `execute_python_analysis` with actionable `derived_dataset_error`.
  - Four-tool LangGraph integration and SQL/Python retry state isolation.
  - 21 comprehensive integration tests in `tests/test_chart_tool_integration.py`.
- **Deferred to Future Steps**:
  - Streamlit UI rendering (`st.plotly_chart`).
  - Persistent disk/database-backed `ChartStore`.
  - Chart-specific automatic error retry classifier.
  - Chart-specific automatic error retry classifier.

