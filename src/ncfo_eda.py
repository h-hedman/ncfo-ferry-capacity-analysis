"""
ncfo_eda.py — DOT BTS NCFO Operator Survey: EDA, QC, and Canonicalization
==========================================================================

Performs exploratory data analysis, quality control, and schema canonicalization
on multi-year National Census of Ferry Operators (NCFO) survey data from the
Bureau of Transportation Statistics (BTS).

Outputs
-------
- Processed CSV      : data/processed/operator_master.csv
- EDA/QC log         : logs/operator/operator_eda_<timestamp>.txt
- Missingness figure : figures/operator/operator_missingness_<timestamp>.png

Design Notes
------------
- No hardcoded absolute paths; project root is resolved dynamically.
- Source year is treated as authoritative (``data_year = source_year``).
- Raw embedded year fields are retained as ``raw_data_year`` for QC only.
- Schema drift across survey years is handled via column padding to a
  fixed canonical column set.
- Additional survey types (e.g., route, terminal) can be added later by
  extending ``CANONICAL_COLS`` and adding new ``SourceSpec`` entries.

Usage
-----
    python -m src.ncfo_eda
    python src/ncfo_eda.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

def find_project_root(start: Path | None = None) -> Path:
    """
    Traverse upward from this file until a directory containing both
    ``data/`` and ``src/`` is found. Falls back to ``Path.cwd()`` if
    no such ancestor exists.

    Parameters
    ----------
    start : Path, optional
        Starting path for traversal. Defaults to the location of this file.

    Returns
    -------
    Path
        Resolved project root directory.
    """
    anchor = (start or Path(__file__).resolve())
    cur = anchor if anchor.is_dir() else anchor.parent

    for _ in range(20):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent

    return Path.cwd().resolve()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TxtLogger:
    """
    Accumulates plain-text log lines and flushes them to a UTF-8 file.

    Parameters
    ----------
    out_path : Path
        Destination file path. Parent directories are created on ``save()``.
    """

    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self._lines: List[str] = []

    def write(self, line: str = "") -> None:
        """Append a single line to the log buffer."""
        self._lines.append(line)

    def section(self, title: str) -> None:
        """Append a section header with an underline."""
        self._lines += ["", title, "=" * len(title)]

    def save(self) -> None:
        """Write the buffered log to disk."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("\n".join(self._lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Schema utilities
# ---------------------------------------------------------------------------

def to_snake(s: str) -> str:
    """
    Normalize an arbitrary column header to ``lower_snake_case``.

    Strips BOM characters, replaces whitespace and non-alphanumeric
    characters with underscores, and collapses consecutive underscores.
    """
    s = s.strip().strip("\ufeff").lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def coerce_numeric(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Parse a messy string series to ``float64``.

    Handles common formatting artifacts including currency symbols,
    thousands separators, and parenthesized negatives.

    Parameters
    ----------
    series : pd.Series
        Raw input column, possibly of mixed or string dtype.

    Returns
    -------
    parsed : pd.Series
        Float-coerced series; unparseable values become ``NaN``.
    stats : dict
        Counts of ``n_nonnull``, ``n_parsed``, and ``n_failed``.
    """
    stats = {"n_nonnull": int(series.notna().sum()), "n_parsed": 0, "n_failed": 0}

    if series.dtype.kind in {"i", "u", "f"}:
        stats["n_parsed"] = stats["n_nonnull"]
        return series.astype("float64"), stats

    normed = (
        series.astype("string")
              .str.strip()
              .str.replace(r"[\$,]", "", regex=True)
              .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
              .replace({"": pd.NA, "na": pd.NA, "n/a": pd.NA,
                        "none": pd.NA, "null": pd.NA, "-": pd.NA})
    )

    parsed = pd.to_numeric(normed, errors="coerce").astype("float64")
    stats["n_parsed"] = int(parsed.notna().sum())
    stats["n_failed"] = int((normed.notna() & parsed.isna()).sum())
    return parsed, stats


def normalize_binary(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Standardize a yes/no flag column to nullable integer ``{0, 1, NA}``.

    Recognizes common affirmative/negative tokens (case-insensitive).

    Parameters
    ----------
    series : pd.Series
        Raw flag column.

    Returns
    -------
    normed : pd.Series (Int64)
        Mapped binary series.
    stats : dict
        Counts of ``n_nonnull``, ``n_mapped``, and ``n_unmapped``.
    """
    _MAP = {
        "y": 1, "yes": 1, "true": 1, "t": 1, "1": 1,
        "n": 0, "no": 0, "false": 0, "f": 0, "0": 0,
    }
    stats = {"n_nonnull": int(series.notna().sum()), "n_mapped": 0, "n_unmapped": 0}

    if series.dtype.kind in {"i", "u", "f"}:
        normed = series.where(series.isin([0, 1]), other=pd.NA).astype("Int64")
        stats["n_mapped"] = int(normed.notna().sum())
        stats["n_unmapped"] = stats["n_nonnull"] - stats["n_mapped"]
        return normed, stats

    normed = series.astype("string").str.strip().str.lower().map(_MAP).astype("Int64")
    s = series.astype("string").str.strip().replace({"": pd.NA})
    stats["n_mapped"] = int(normed.notna().sum())
    stats["n_unmapped"] = int((s.notna() & normed.isna()).sum())
    return normed, stats


# ---------------------------------------------------------------------------
# Canonical schema — Operator survey
# ---------------------------------------------------------------------------

_BASE_COLS: List[str] = [
    "operator_id", "operator_name", "op_strcity", "op_state",
    "op_strzip", "op_country", "url", "federal_state_local",
    "ticket_revenue", "private_contract_revenue", "advertising_revenue",
    "public_contract_revenue", "federal_funding_revenue",
    "state_funding_revenue", "local_funding_revenue", "other_funding_revenue",
    "trip_purpose_commuter_transit", "trip_purpose_pleasure",
    "trip_purpose_emergency", "trip_purpose_roadway_conn",
    "trip_purpose_lifeline", "trip_purpose_nps",
    "trip_purpose_other", "trip_purpose_other_desc",
    "accepts_public_funding",
]

_PUB_FUND_COLS: List[str] = [
    col
    for i in range(1, 9)
    for col in (f"pub_fund_type_{i}", f"pub_fund_source_{i}", f"pub_fund_prog_{i}")
]

CANONICAL_COLS: List[str] = (
    _BASE_COLS
    + _PUB_FUND_COLS
    + ["census_year", "data_year", "raw_data_year", "source_year", "source_file", "ingest_ts"]
)

# Column names that encode the survey's own "data year" field; retained for QC only.
_RAW_YEAR_NAMES = {"data year", "data_year", "datayear"}

REVENUE_COLS: List[str] = [
    "ticket_revenue", "private_contract_revenue", "advertising_revenue",
    "public_contract_revenue", "federal_funding_revenue",
    "state_funding_revenue", "local_funding_revenue", "other_funding_revenue",
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceSpec:
    """Descriptor for a single survey-year raw data file."""
    year: int
    path: Path
    kind: str  # "csv" | "xlsx"


def read_source(spec: SourceSpec) -> pd.DataFrame:
    """
    Load a raw survey file, coercing all fields to string and standardizing
    common NA tokens.

    Parameters
    ----------
    spec : SourceSpec
        File descriptor including path and format kind.

    Returns
    -------
    pd.DataFrame
        Raw string-typed dataframe.

    Raises
    ------
    ValueError
        If ``spec.kind`` is not ``"csv"`` or ``"xlsx"``.
    """
    _NA_TOKENS = ["", "NA", "N/A", "NULL", "null", "None", "none"]

    if spec.kind == "csv":
        return pd.read_csv(spec.path, dtype=str,
                           na_values=_NA_TOKENS, keep_default_na=True)

    if spec.kind == "xlsx":
        df = pd.read_excel(spec.path, dtype=str)
        return df.replace({tok: pd.NA for tok in _NA_TOKENS})

    raise ValueError(f"Unsupported file kind: '{spec.kind}'. Expected 'csv' or 'xlsx'.")


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canonicalize_operator(
    df: pd.DataFrame,
    spec: SourceSpec,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply canonical schema to a raw operator survey dataframe.

    Steps
    -----
    1. Identify and extract any embedded raw year column (QC only).
    2. Normalize all remaining column headers to ``lower_snake_case``.
    3. Attach ``raw_data_year``, authoritative ``data_year``, and provenance fields.
    4. Pad any missing canonical columns with ``NA``.
    5. Reorder to ``CANONICAL_COLS``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw string-typed dataframe from ``read_source()``.
    spec : SourceSpec
        Source descriptor; ``spec.year`` is the authoritative data year.

    Returns
    -------
    canon_df : pd.DataFrame
        Canonicalized dataframe aligned to ``CANONICAL_COLS``.
    header_stats : dict
        Diagnostic counts: ``n_raw_cols``, ``n_raw_year_cols_found``.
    """
    header_stats = {
        "n_raw_cols": int(df.shape[1]),
        "n_raw_year_cols_found": 0,
    }

    # Identify embedded raw year fields
    raw_year_cols = [c for c in df.columns
                     if c is not None and c.strip().lower() in _RAW_YEAR_NAMES]
    header_stats["n_raw_year_cols_found"] = len(raw_year_cols)

    raw_data_year = (
        df[raw_year_cols[0]] if raw_year_cols
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    )

    # Drop raw year cols before renaming to prevent collision
    df = df.drop(columns=raw_year_cols, errors="ignore")
    df = df.rename(columns={c: to_snake(str(c)) for c in df.columns})

    # Attach metadata and provenance
    df["raw_data_year"] = raw_data_year
    df["data_year"]     = spec.year
    df["source_year"]   = spec.year
    df["source_file"]   = spec.path.name
    df["ingest_ts"]     = datetime.utcnow().isoformat(timespec="seconds")

    # Pad missing canonical columns and enforce column order
    for col in CANONICAL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[CANONICAL_COLS], header_stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_missingness_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    Render and save a binary missingness heatmap (rows x columns).

    Each cell is colored to indicate whether the value is missing.
    Suitable for datasets up to ~2,000 rows.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to visualize.
    out_path : Path
        Destination PNG file path.
    title : str
        Figure title.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    miss = df.isna().to_numpy(dtype=np.uint8)

    fig_width = max(10, min(24, df.shape[1] * 0.35))
    plt.figure(figsize=(fig_width, 10))
    plt.imshow(miss, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.xticks(ticks=np.arange(df.shape[1]), labels=df.columns,
               rotation=90, fontsize=7)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# QC reporting
# ---------------------------------------------------------------------------

def qc_overview(df: pd.DataFrame, logger: TxtLogger, title: str) -> None:
    """Log shape, top missingness rates, duplicate counts, and key ID field stats."""
    logger.section(title)
    logger.write(f"Rows : {df.shape[0]:,}")
    logger.write(f"Cols : {df.shape[1]:,}")

    logger.write("")
    logger.write("Missingness rate (top 20 columns):")
    for col, rate in df.isna().mean().sort_values(ascending=False).head(20).items():
        logger.write(f"  {col:40s} {rate:.3f}")

    logger.write("")
    logger.write(f"Exact-duplicate rows: {int(df.duplicated().sum()):,}")

    for key in ("operator_id", "operator_name"):
        if key not in df.columns:
            continue
        s = df[key].astype("string")
        logger.write(
            f"{key}: null={int(df[key].isna().sum()):,}  "
            f"blank={int((s.str.strip() == '').sum()):,}  "
            f"unique={int(df[key].nunique(dropna=True)):,}"
        )


def qc_year_consistency(df: pd.DataFrame, logger: TxtLogger) -> None:
    """
    Compare the embedded raw year field against the authoritative ``data_year``.

    Helps verify that source-file year labels are consistent with the
    year assigned during ingestion.
    """
    logger.section("Year Consistency QC (raw_data_year vs. data_year)")

    if "raw_data_year" not in df.columns:
        logger.write("raw_data_year not present — skipping.")
        return

    raw = (
        df["raw_data_year"].astype("string").str.strip()
          .replace({"": pd.NA, "na": pd.NA, "n/a": pd.NA,
                    "none": pd.NA, "null": pd.NA})
    )
    auth = df["data_year"].astype("Int64").astype("string")

    n_present  = int(raw.notna().sum())
    n_mismatch = int((raw.notna() & (raw != auth)).sum())

    logger.write(f"Rows with embedded raw year      : {n_present:,}")
    logger.write(f"Mismatches (raw != authoritative): {n_mismatch:,}")

    if n_present:
        logger.write("")
        logger.write("Most common raw year values (top 10):")
        for val, cnt in raw.value_counts(dropna=True).head(10).items():
            logger.write(f"  {val:10s}  {int(cnt):,}")


def qc_binary_fields(df: pd.DataFrame, logger: TxtLogger, fields: List[str]) -> None:
    """
    Report mapping success for yes/no flag fields. Diagnostic only;
    source values are not modified.

    Parameters
    ----------
    fields : list of str
        Column names to evaluate.
    """
    logger.section("Binary Flag Field QC")

    for field in fields:
        if field not in df.columns:
            continue

        normed, stats = normalize_binary(df[field])
        logger.write(
            f"{field}: nonnull={stats['n_nonnull']:,}  "
            f"mapped={stats['n_mapped']:,}  unmapped={stats['n_unmapped']:,}"
        )

        if stats["n_unmapped"] > 0:
            s = df[field].astype("string").str.strip().replace({"": pd.NA})
            unmapped = s[normed.isna() & s.notna()].value_counts().head(5)
            if not unmapped.empty:
                logger.write("  Unmapped values (top 5):")
                for val, cnt in unmapped.items():
                    logger.write(f"    {str(val):20s}  {int(cnt):,}")


def qc_revenue_fields(df: pd.DataFrame, logger: TxtLogger, fields: List[str]) -> None:
    """
    Report numeric parse success for revenue columns.

    Parameters
    ----------
    fields : list of str
        Column names expected to contain numeric revenue values.
    """
    logger.section("Revenue Field Numeric Coercion QC")

    for field in fields:
        if field not in df.columns:
            continue

        parsed, stats = coerce_numeric(df[field])
        logger.write(
            f"{field}: nonnull={stats['n_nonnull']:,}  "
            f"parsed={stats['n_parsed']:,}  failed={stats['n_failed']:,}"
        )

        if stats["n_failed"] > 0:
            s = df[field].astype("string").str.strip().replace({"": pd.NA})
            failures = s[s.notna() & parsed.isna()].value_counts().head(5)
            logger.write("  Parse-failure examples (top 5):")
            for val, cnt in failures.items():
                logger.write(f"    {str(val):25s}  {int(cnt):,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Orchestrate the NCFO operator survey EDA/QC pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, 2 if a required input file is missing.
    """
    root   = find_project_root()
    survey = "operator"

    data_raw       = root / "data" / "raw" / survey
    data_processed = root / "data" / "processed"
    logs_dir       = root / "logs" / survey
    figs_dir       = root / "figures" / survey

    sources: List[SourceSpec] = [
        SourceSpec(year=2022, path=data_raw / "operator_2022.csv",  kind="csv"),
        SourceSpec(year=2020, path=data_raw / "operator_2020.csv",  kind="csv"),
        SourceSpec(year=2018, path=data_raw / "operator_2018.xlsx", kind="xlsx"),
    ]

    ts           = _timestamp()
    log_path     = logs_dir / f"{survey}_eda_{ts}.txt"
    out_csv      = data_processed / f"{survey}_master.csv"
    heatmap_path = figs_dir / f"{survey}_missingness_{ts}.png"

    logger = TxtLogger(log_path)

    # Run header
    logger.section("DOT BTS NCFO — Operator Survey EDA/QC + Canonicalization")
    logger.write(f"Run timestamp : {datetime.now().isoformat(timespec='seconds')}")
    logger.write(f"Project root  : {root}")
    logger.write("")
    logger.write("Input files:")
    for s in sources:
        logger.write(f"  {s.year}  {s.path}  [{s.kind}]")
    logger.write("")
    logger.write(f"Processed CSV : {out_csv}")
    logger.write(f"Log file      : {log_path}")
    logger.write(f"Figures dir   : {figs_dir}")

    # Ingest + canonicalize
    logger.section("Per-Source Ingest and Schema Canonicalization")
    frames: List[pd.DataFrame] = []

    for spec in sources:
        if not spec.path.exists():
            logger.write(f"[ERROR] Input file not found: {spec.path}")
            logger.save()
            return 2

        df_raw   = read_source(spec)
        canon_df, h_stats = canonicalize_operator(df_raw, spec)

        logger.write(
            f"  {spec.year}: raw rows={df_raw.shape[0]:,}  "
            f"raw cols={df_raw.shape[1]:,}  "
            f"raw_year_cols_found={h_stats['n_raw_year_cols_found']}"
        )
        print(f"[OK] {spec.year} — {df_raw.shape[0]:,} rows ingested")
        frames.append(canon_df)

    # Union and export
    master = pd.concat(frames, ignore_index=True)
    data_processed.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_csv, index=False)

    # QC reporting
    qc_overview(master, logger, "Master Dataset — Post-Union Overview")
    qc_year_consistency(master, logger)

    flag_fields = (
        [c for c in master.columns if c.startswith("trip_purpose_")]
        + ["accepts_public_funding"]
    )
    qc_binary_fields(master, logger, flag_fields)
    qc_revenue_fields(master, logger, REVENUE_COLS)

    # Figures
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_missingness_heatmap(
        master, heatmap_path,
        title="NCFO Operator Survey — Missingness Heatmap (1 = Missing)",
    )

    # Footer
    logger.section("Outputs Written")
    logger.write(f"Processed CSV      : {out_csv}")
    logger.write(f"EDA/QC log         : {log_path}")
    logger.write(f"Missingness figure : {heatmap_path}")
    logger.save()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())