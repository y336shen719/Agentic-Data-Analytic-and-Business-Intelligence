# Python Analysis Engine

## 1. Architecture & Responsibilities

The **Python Analysis Engine** (`tools/python_analysis.py`) provides controlled, deterministic in-memory computation for analytical datasets materialized by the SQL tool.

```text
LLM (Tool Call)
   │
   ▼
[execute_sql Tool] ──► [DuckDB Engine]
   │                          │
   ▼                          ▼
[DatasetStore] ◄──── (Structured Result)
   │
   │ dataset_id
   ▼
[PythonAnalysisEngine]
   │
   ├── 1. DatasetStore.get(dataset_id)
   ├── 2. Build Pandas DataFrame
   ├── 3. Validate Operation & Parameters
   ├── 4. Execute Allowlisted Operation
   └── 5. Return Clean JSON Primitives
```

### Key Responsibilities
- **Controlled Operation Dispatch**: Dispatches only explicitly allowed analytical operations (`describe`, `correlation`, `growth_rate`, `detect_outliers`, `groupby`).
- **Data Hydration**: Converts `DatasetRecord` rows/columns into a pandas DataFrame without re-querying the database.
- **Data Cleansing & JSON Conformity**: Recursively sanitizes results (converts `None`, `NaN`, `pd.NA`, `pd.NaT`, infinities to JSON `null`, NumPy scalars to native Python primitives, and timestamps to ISO strings).
- **Security Barrier**: Strictly prohibits arbitrary code execution (`eval`, `exec`, shell calls, dynamic imports).

---

## 2. SQL vs. Python Responsibility Boundaries

| Responsibility | Handled By | Rationale |
| :--- | :--- | :--- |
| Filtering, Joins, Large Aggregations | **SQL Tool (DuckDB)** | Executes close to the database; enforces read-only access and memory limits. |
| Time-series Sorting & Period Growth | **Python Engine** | Pandas supports lag/lead percentage change across time and groups with precision. |
| Statistical Summaries & IQR Outliers | **Python Engine** | NumPy/Pandas provide standardized quartiles, medians, and IQR bounds. |
| Multi-column Pearson Correlation | **Python Engine** | Efficient matrix calculation over numerical arrays. |
| Secondary GroupBy | **Python Engine** | Light post-processing of already aggregated SQL result sets. |

---

## 3. Supported Operations & Parameter Schemas

### 1. `describe`
Computes non-null count, mean, sample standard deviation ($ddof=1$), minimum, 25th percentile, median, 75th percentile, and maximum.
- **Parameters**:
  - `columns` (`Optional[List[str]]`): List of numeric columns to evaluate.
- **Numeric Column Selection Rules**:
  - When `columns` is omitted, the engine **only automatically selects genuine numeric data types** (`is_numeric_dtype`). Numeric-looking string IDs (e.g., `"1001"`) are strictly ignored.
  - When `columns` are explicitly requested, string columns containing parseable numbers are converted; columns containing malformed non-numeric values fail loudly with `InvalidColumnError`.

### 2. `correlation`
Computes the Pearson correlation matrix across selected numerical columns.
- **Parameters**:
  - `columns` (`List[str]`, required, $\ge 2$ columns): Numerical column names.
- **Null & Constant Handling**: Constant columns ($std=0$) or insufficient observations ($N < 2$) yield `null`.

### 3. `growth_rate`
Calculates period-over-period percentage growth rate:
$$\text{growth\_rate} = \frac{\text{current\_value} - \text{previous\_value}}{\text{previous\_value}}$$
- **Parameters**:
  - `period_column` (`str`, required): Date, timestamp, or period string column.
  - `value_column` (`str`, required): Numeric metric column.
  - `group_column` (`Optional[str]`): Optional categorical column for independent group-level growth.
- **Accepted Date & Period Formats**:
  - ISO-8601 full date strings (`YYYY-MM-DD`), datetime/timestamps (`YYYY-MM-DDTHH:MM:SS`), and monthly period formats (`YYYY-MM`).
  - Periods are strictly validated. Non-date or malformed period values raise `InvalidParameterError` immediately; silent fallback to lexicographic sorting is disallowed.
  - Null/missing period values raise `InvalidParameterError`.
- **Duplicate Checking**:
  - Duplicate periods are detected using **normalized timestamps** so equivalent formats (e.g. `'2023-01-01'` and `'2023-01-01T00:00:00'`) cannot bypass validation. Ambiguous duplicates raise `DuplicatePeriodError`.
- **Formula & Baseline Limitations**:
  - Consistent formula applies to both positive and negative nonzero baselines. Note that percentage change from a negative baseline flips sign (e.g., moving from $-100$ to $-50$ is a $-50\%$ change under the standard formula).
  - Returns `null` when previous value is $0$, missing, or nonfinite.

### 4. `detect_outliers`
Detects numerical outliers using the Interquartile Range (IQR) rule:
$$\text{IQR} = Q_3 - Q_1, \quad [\text{Lower}, \text{Upper}] = [Q_1 - k \times \text{IQR}, Q_3 + k \times \text{IQR}]$$
- **Parameters**:
  - `column` (`str`, required): Numeric column name.
  - `iqr_multiplier` (`float`, default `1.5`): Positive finite multiplier $k$.
- **Output**: Returns $Q_1$, $Q_3$, $\text{IQR}$, lower and upper bounds, outlier count, and bounded preview of outlier rows.

### 5. `groupby`
Secondary post-query aggregation over already-materialized dataset rows.
- **Parameters**:
  - `by` (`str | List[str]`, required): Grouping column(s).
  - `agg_column` (`str`, required): Column to aggregate.
  - `agg_func` (`str`, required): One of `"sum"`, `"mean"`, `"count"`, `"min"`, `"max"`.

---

## 4. Standardized Response Format

### Success Response Example
```json
{
  "status": "success",
  "dataset_id": "ds_45a1c028e3b7498c85412eb37f59d57a",
  "operation": "describe",
  "result": {
    "payment_value": {
      "count": 100,
      "mean": 154.2,
      "std": 45.8,
      "min": 20.0,
      "25%": 80.0,
      "median": 130.5,
      "75%": 190.0,
      "max": 450.0
    }
  },
  "truncated": false,
  "warnings": []
}
```

### Error Response Example
```json
{
  "status": "error",
  "dataset_id": "ds_45a1c028e3b7498c85412eb37f59d57a",
  "operation": "detect_outliers",
  "error_type": "InvalidColumnError",
  "error_message": "Column 'nonexistent' does not exist in dataset."
}
```

---

## 5. Safeguards, Limits, and Missing-Value Semantics

- **Configuration Validation**:
  - `max_preview_rows`, `max_table_rows`, and `max_corr_columns` must be strictly positive integers. Floating-point values, strings, Booleans, zero, and negative values are rejected at initialization.
- **Output Size Limits**:
  - Bounded outputs (`growth_rate`, `groupby`) truncate row generation at the DataFrame stage to avoid wasteful dictionary allocations.
  - Warnings state the total number of records/groups before truncation (e.g. `Growth rate output was bounded to the first 200 periods (out of 1000 total).`).
- **Missing-Value Semantics**:
  - Python `None`, `np.nan`, `pd.NA`, `pd.NaT`, `+inf`, `-inf` serialize cleanly as JSON `null`.
  - Serializing via `json.dumps(response, allow_nan=False)` is guaranteed to succeed.
- **Truncated Datasets**:
  - If `record.truncated == True`, the response reflects `"truncated": true` and issues a warning:
    `"Dataset was truncated by SQL execution limits; analytical results reflect only the materialized sample."`

---

## 6. Security Boundaries

- **No Arbitrary Code Execution**: No `eval`, `exec`, `subprocess`, dynamic code compilation, or user-supplied script evaluation.
- **Session Isolation**: Requires an injected `DatasetStore` instance. No global stores or cross-session state.
- **Column Safety**: Calculations use isolated local Series and structures, preventing accidental column collisions with internal names like `_clean_val` or `_sort_key`.
