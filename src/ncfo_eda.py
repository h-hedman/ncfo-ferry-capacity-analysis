"""
ncfo_eda.py — DOT BTS NCFO Multi-Survey EDA, QC, and Descriptive Analytics
============================================================================

Performs comprehensive exploratory data analysis, quality control, schema
canonicalization, and descriptive statistics on multi-year National Census
of Ferry Operators (NCFO) survey data from the Bureau of Transportation
Statistics (BTS).

Surveys supported
-----------------
    operator         — ferry operators (governance, revenue mix, trip purpose)
    vessel           — fleet assets (capacity, fuel, age, physical specs)
    terminal         — terminal locations (geography, multimodal access)
    segment          — route network edges (type, NPS service)
    operator_segment — service activity (demand, capacity, seasonality)

Outputs per survey
------------------
    data/processed/<survey>_master.csv   — canonicalized multi-year union
    logs/<survey>/<survey>_eda_<ts>.txt  — full EDA/QC log
    figures/<survey>/                    — all 300 DPI PNG figures

Design principles
-----------------
    - No hardcoded absolute paths; root resolved dynamically via find_project_root().
    - Each survey has its own canonical column list, type specs, and EDA module.
    - A shared EDA engine (eda_core) provides all generic descriptive functions.
    - run_survey() is the single entry point per survey; run_all() runs all.
    - Adding a new survey year requires only a new SourceSpec entry in SOURCES.
    - All figures: 300 DPI, consistent style, labeled axes, informative titles.

Year-lag note
-------------
    The 2020 NCFO census collected data for year 2019. The pipeline treats
    source_year (the census wave) as authoritative for panel tracking.
    raw_data_year is retained for QC only and the known lag is documented
    explicitly in the log.

Usage
-----
    # Run all surveys
    python src/ncfo_eda.py

    # Run a single survey
    python src/ncfo_eda.py --survey operator

    # Run specific surveys
    python src/ncfo_eda.py --survey operator vessel terminal
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONSTANTS
# ============================================================================

CENSUS_YEARS = [2018, 2020, 2022]

# Known survey lag: 2020 census = 2019 data year
CENSUS_TO_DATA_YEAR: Dict[int, int] = {
    2018: 2018,
    2020: 2019,
    2022: 2022,
}

FIG_DPI = 300

# Government/federal audience palette — clean, accessible, Excel-familiar.
# Primary: BTS navy blue. Secondary: warm gray. Accent: USDOT orange.
# All colors are colorblind-safe and print well in grayscale.
GOV_NAVY    = "#003875"   # BTS/USDOT navy — primary bars, headings
GOV_BLUE    = "#005DAA"   # medium blue — secondary bars, grouped charts year 1
GOV_STEEL   = "#4A90C4"   # steel blue — tertiary / year 2
GOV_LTBLUE  = "#A8CADF"   # light blue — year 3 / backgrounds
GOV_ORANGE  = "#C85200"   # USDOT accent orange — warnings, highlights
GOV_GRAY    = "#6D6E71"   # neutral gray — grid lines, secondary text
GOV_LTGRAY  = "#D9D9D9"   # light gray — backgrounds, missing fill
GOV_GREEN   = "#4A7B3F"   # muted green — "present/complete" indicator
GOV_RED     = "#B22222"   # muted red — "missing/sparse" indicator
GOV_WHITE   = "#FFFFFF"

# Ordered palette list for multi-series charts (year 2018, 2020, 2022)
PALETTE = [GOV_NAVY, GOV_STEEL, GOV_ORANGE, GOV_GRAY, GOV_LTBLUE]
YEAR_COLORS = {2018: GOV_NAVY, 2020: GOV_STEEL, 2022: GOV_ORANGE}

STYLE = "seaborn-v0_8-whitegrid"

# Variable group color coding for missingness figure
GROUP_COLORS = {
    "identifier":  GOV_NAVY,
    "geographic":  GOV_STEEL,
    "revenue":     GOV_ORANGE,
    "trip_purpose":GOV_GREEN,
    "funding":     GOV_GRAY,
    "capacity":    GOV_BLUE,
    "physical":    GOV_LTBLUE,
    "operational": GOV_NAVY,
    "other":       GOV_LTGRAY,
}

NA_TOKENS = ["", "NA", "N/A", "NULL", "null", "None", "none", "na", "n/a"]

# ============================================================================
# PROJECT ROOT
# ============================================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk upward from this file until a directory containing both
    ``data/`` and ``src/`` subdirectories is found.

    Falls back to ``Path.cwd()`` if no such ancestor exists.
    """
    anchor = (start or Path(__file__).resolve())
    cur = anchor if anchor.is_dir() else anchor.parent
    for _ in range(20):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return Path.cwd().resolve()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# LOGGER
# ============================================================================

class TxtLogger:
    """
    Accumulates plain-text log lines and flushes to a UTF-8 file.

    Supports section headers, indented sub-items, tabular formatting,
    and inline console mirroring.

    Parameters
    ----------
    out_path : Path
        Destination file. Parent directories created on ``save()``.
    echo : bool
        If True, each written line is also printed to stdout.
    """

    def __init__(self, out_path: Path, echo: bool = True) -> None:
        self.out_path = out_path
        self.echo = echo
        self._lines: List[str] = []

    def write(self, line: str = "") -> None:
        self._lines.append(line)
        if self.echo:
            print(line)

    def section(self, title: str, level: int = 1) -> None:
        char = "=" if level == 1 else "-"
        self._lines += ["", title, char * len(title)]
        if self.echo:
            print(f"\n{title}\n{char * len(title)}")

    def kv(self, key: str, value, width: int = 35) -> None:
        self.write(f"  {key:<{width}} {value}")

    def table_row(self, cells: List[str], widths: List[int]) -> None:
        row = "  " + "  ".join(str(c)[:w].ljust(w) for c, w in zip(cells, widths))
        self.write(row)

    def save(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("\n".join(self._lines), encoding="utf-8")


# ============================================================================
# SCHEMA UTILITIES
# ============================================================================

def to_snake(s: str) -> str:
    """Normalize column header to lower_snake_case."""
    s = str(s).strip().strip("\ufeff").lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def coerce_numeric(series: pd.Series) -> Tuple[pd.Series, Dict]:
    """
    Parse a messy string series to float64.

    Handles currency symbols, thousands separators, and parenthesized
    negatives. Returns parsed series and diagnostic stats dict.
    """
    stats = {"n_nonnull": int(series.notna().sum()), "n_parsed": 0, "n_failed": 0}
    if series.dtype.kind in {"i", "u", "f"}:
        stats["n_parsed"] = stats["n_nonnull"]
        return series.astype("float64"), stats

    normed = (
        series.astype("string").str.strip()
              .str.replace(r"[\$,]", "", regex=True)
              .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
              .replace({"": pd.NA, "na": pd.NA, "n/a": pd.NA,
                        "none": pd.NA, "null": pd.NA, "-": pd.NA})
    )
    parsed = pd.to_numeric(normed, errors="coerce").astype("float64")
    stats["n_parsed"] = int(parsed.notna().sum())
    stats["n_failed"] = int((normed.notna() & parsed.isna()).sum())
    return parsed, stats


def normalize_binary(series: pd.Series) -> Tuple[pd.Series, Dict]:
    """
    Standardize yes/no flag column to nullable Int64 {0, 1, NA}.
    """
    _MAP = {"y": 1, "yes": 1, "true": 1, "t": 1, "1": 1,
            "n": 0, "no": 0, "false": 0, "f": 0, "0": 0}
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


# ============================================================================
# I/O
# ============================================================================

@dataclass(frozen=True)
class SourceSpec:
    """Descriptor for one survey-year raw data file."""
    year: int       # Census wave year (authoritative)
    path: Path
    kind: str       # "csv" | "xlsx"


def read_source(spec: SourceSpec) -> pd.DataFrame:
    """Load raw survey file; all fields coerced to string."""
    if spec.kind == "csv":
        return pd.read_csv(spec.path, dtype=str,
                           na_values=NA_TOKENS, keep_default_na=True)
    if spec.kind == "xlsx":
        df = pd.read_excel(spec.path, dtype=str)
        return df.replace({tok: pd.NA for tok in NA_TOKENS})
    raise ValueError(f"Unsupported kind: '{spec.kind}'")


# ============================================================================
# CANONICAL SCHEMAS
# ============================================================================

# --- Operator ---
_OP_BASE = [
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
_OP_FUND = [c for i in range(1, 9)
            for c in (f"pub_fund_type_{i}", f"pub_fund_source_{i}", f"pub_fund_prog_{i}")]
OP_CANONICAL = _OP_BASE + _OP_FUND + [
    "census_year", "data_year", "raw_data_year", "source_year", "source_file", "ingest_ts"
]

OP_REVENUE_COLS = [
    "ticket_revenue", "private_contract_revenue", "advertising_revenue",
    "public_contract_revenue", "federal_funding_revenue",
    "state_funding_revenue", "local_funding_revenue", "other_funding_revenue",
]
OP_TRIP_PURPOSE_COLS = [
    "trip_purpose_commuter_transit", "trip_purpose_pleasure",
    "trip_purpose_emergency", "trip_purpose_roadway_conn",
    "trip_purpose_lifeline", "trip_purpose_nps", "trip_purpose_other",
]

# --- Vessel ---
VES_CANONICAL = [
    "vessel_id", "operator_id", "vessel_name", "uscg_number",
    "in_service", "carries_passengers", "carries_vehicles", "carries_freight",
    "passenger_capacity", "vehicle_capacity",
    "fuel_type", "fuel_other", "typical_speed", "year_built",
    "main_horsepower_ahead", "main_horsepower_astern",
    "hull_material", "hull_shape", "propulsion_type", "self_prop_indicator",
    "registered_breadth", "registered_depth", "registered_length",
    "registered_net_tons", "registered_gross_tons",
    "vessel_ownership", "vessel_owned_by", "vessel_operation", "vessel_operated_by",
    "fuel_mileage", "ada_accessible", "expected_lifespan",
    "census_year_miles", "vessel_type",
    "census_year", "data_year", "raw_data_year", "source_year", "source_file", "ingest_ts"
]
VES_NUMERIC_COLS = [
    "passenger_capacity", "vehicle_capacity", "typical_speed", "year_built",
    "main_horsepower_ahead", "main_horsepower_astern",
    "registered_breadth", "registered_depth", "registered_length",
    "registered_net_tons", "registered_gross_tons",
    "fuel_mileage", "expected_lifespan", "census_year_miles",
]
VES_BINARY_COLS = [
    "in_service", "carries_passengers", "carries_vehicles", "carries_freight",
    "ada_accessible",
]
VES_CATEGORICAL_COLS = [
    "fuel_type", "hull_material", "hull_shape", "propulsion_type",
    "vessel_ownership", "vessel_operation", "vessel_type",
]

# --- Terminal ---
TERM_CANONICAL = [
    "terminal_id", "operator_id", "terminal_name",
    "term_city", "term_state", "term_country",
    "latitude", "longitude",
    "in_operation", "parking", "local_bus", "intercity_bus",
    "local_rail", "intercity_rail", "bike_share",
    "terminal_ownership", "terminal_owned_by",
    "terminal_operation", "terminal_operated_by",
    "census_year", "data_year", "raw_data_year", "source_year", "source_file", "ingest_ts"
]
TERM_BINARY_COLS = [
    "in_operation", "parking", "local_bus", "intercity_bus",
    "local_rail", "intercity_rail", "bike_share",
]
TERM_MULTIMODAL_COLS = [
    "parking", "local_bus", "intercity_bus",
    "local_rail", "intercity_rail", "bike_share",
]

# --- Segment ---
SEG_CANONICAL = [
    "segment_id", "segment_name",
    "seg_terminal1_id", "seg_terminal2_id",
    "seg_type", "serves_nps",
    "census_year", "data_year", "raw_data_year", "source_year", "source_file", "ingest_ts"
]

# --- Operator-Segment ---
_OPSEG_VESSEL_IDS = [f"vessel_id{i}" for i in range(1, 14)]
OPSEG_CANONICAL = [
    "operator_id", "segment_id",
    "seg_length", "average_trip_time",
    "segment_season_start", "segment_season_end",
    "trips_per_year",
    "route_rates_regulated", "route_rate_regulator",
    "most_used_vessel_id",
] + _OPSEG_VESSEL_IDS + [
    "passengers", "vehicles",
    "avg_daily_brd_pax", "avg_daily_brd_veh",
    "census_year", "data_year", "raw_data_year", "source_year", "source_file", "ingest_ts"
]
OPSEG_NUMERIC_COLS = [
    "seg_length", "average_trip_time", "trips_per_year",
    "passengers", "vehicles", "avg_daily_brd_pax", "avg_daily_brd_veh",
]

# Map: survey name → canonical column list
CANONICAL_MAP: Dict[str, List[str]] = {
    "operator":          OP_CANONICAL,
    "vessel":            VES_CANONICAL,
    "terminal":          TERM_CANONICAL,
    "segment":           SEG_CANONICAL,
    "operator_segment":  OPSEG_CANONICAL,
}

_RAW_YEAR_NAMES = {"data year", "data_year", "datayear"}


# ============================================================================
# CANONICALIZATION
# ============================================================================

def canonicalize(
    df: pd.DataFrame,
    spec: SourceSpec,
    canonical_cols: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply canonical schema to any raw NCFO survey dataframe.

    Steps
    -----
    1. Extract embedded raw year column (retained as raw_data_year; QC only).
    2. Normalize all column headers to lower_snake_case.
    3. Attach authoritative data_year (from CENSUS_TO_DATA_YEAR mapping),
       provenance fields, and ingest timestamp.
    4. Pad missing canonical columns with NA.
    5. Reorder columns to canonical order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw string-typed dataframe from read_source().
    spec : SourceSpec
        Source descriptor; spec.year is the census wave year.
    canonical_cols : list of str
        Target canonical column list for this survey type.

    Returns
    -------
    canon_df : pd.DataFrame
    header_stats : dict
        n_raw_cols, n_raw_year_cols_found
    """
    h_stats = {"n_raw_cols": int(df.shape[1]), "n_raw_year_cols_found": 0}

    raw_year_cols = [c for c in df.columns
                     if c is not None and str(c).strip().lower() in _RAW_YEAR_NAMES]
    h_stats["n_raw_year_cols_found"] = len(raw_year_cols)

    raw_data_year = (
        df[raw_year_cols[0]] if raw_year_cols
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    )

    df = df.drop(columns=raw_year_cols, errors="ignore")
    df = df.rename(columns={c: to_snake(str(c)) for c in df.columns})

    data_year = CENSUS_TO_DATA_YEAR.get(spec.year, spec.year)

    df["raw_data_year"] = raw_data_year
    df["data_year"]     = data_year
    df["census_year"]   = spec.year
    df["source_year"]   = spec.year
    df["source_file"]   = spec.path.name
    df["ingest_ts"]     = datetime.utcnow().isoformat(timespec="seconds")

    for col in canonical_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[canonical_cols], h_stats


# ============================================================================
# EDA CORE — shared descriptive functions
# ============================================================================

class EdaCore:
    """
    Generic EDA engine callable on any canonicalized NCFO survey table.

    All methods write to a TxtLogger and optionally save figures to a
    designated output directory.

    Parameters
    ----------
    df : pd.DataFrame
        Canonicalized multi-year master dataframe.
    logger : TxtLogger
    figs_dir : Path
        Output directory for all figures generated by this instance.
    survey : str
        Survey name (used in figure titles and filenames).
    """

    def __init__(self, df: pd.DataFrame, logger: TxtLogger,
                 figs_dir: Path, survey: str) -> None:
        self.df       = df
        self.log      = logger
        self.figs_dir = figs_dir
        self.survey   = survey
        figs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Section 1 — Dataset overview
    # ------------------------------------------------------------------

    def overview(self) -> None:
        """Log shape, dtypes summary, memory, year breakdown, duplicates."""
        self.log.section("1. DATASET OVERVIEW")
        df = self.df

        self.log.kv("Total rows",    f"{df.shape[0]:,}")
        self.log.kv("Total columns", f"{df.shape[1]:,}")
        self.log.kv("Memory usage",  f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        self.log.kv("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")

        # Year breakdown
        if "source_year" in df.columns:
            self.log.write("")
            self.log.write("  Rows by census year:")
            for yr, cnt in df["source_year"].value_counts().sort_index().items():
                data_yr = CENSUS_TO_DATA_YEAR.get(int(yr), yr)
                self.log.write(f"    census {yr} (data {data_yr}): {cnt:,} rows")

        # Column type summary
        self.log.write("")
        self.log.write("  Column dtypes:")
        for dtype, cols in df.dtypes.groupby(df.dtypes).groups.items():
            self.log.write(f"    {str(dtype):<12} {len(cols):3d} columns")

    # ------------------------------------------------------------------
    # Section 2 — Missingness
    # ------------------------------------------------------------------

    def missingness(self, var_groups: Optional[Dict[str, List[str]]] = None) -> None:
        """
        Full column-by-column missingness report plus 2-panel summary figure.

        Panel A: Horizontal bar chart — % complete per field, sorted and
                 color-coded by variable group. Readable at any column count.
        Panel B: Year-over-year valid-N comparison for key analytical fields.

        Also writes an auto-generated plain-English descriptive paragraph
        summarizing the dataset for inclusion in reports.

        Parameters
        ----------
        var_groups : dict, optional
            Mapping of group label → list of column names, used to color-code
            Panel A. If None, a best-effort grouping is applied automatically.
        """
        self.log.section("2. MISSINGNESS PROFILE")
        df = self.df

        miss_rates = df.isna().mean().sort_values(ascending=False)
        complete_rates = 1 - miss_rates

        self.log.write(f"  Columns fully complete (0% missing):  "
                       f"{int((miss_rates == 0).sum()):,}")
        self.log.write(f"  Columns with any missing:             "
                       f"{int((miss_rates > 0).sum()):,}")
        self.log.write(f"  Columns >20% missing (caution):       "
                       f"{int((miss_rates > 0.2).sum()):,}")
        self.log.write(f"  Columns >50% missing (sparse):        "
                       f"{int((miss_rates > 0.5).sum()):,}")
        self.log.write(f"  Columns >90% missing (near-empty):    "
                       f"{int((miss_rates > 0.9).sum()):,}")
        self.log.write(f"  Overall cell missingness rate:        "
                       f"{df.isna().values.mean():.3f}")

        # Full column table
        self.log.write("")
        self.log.write("  All columns — completeness:")
        self.log.write(f"  {'Column':<45} {'% Complete':>10}  {'N Valid':>10}  "
                       f"{'N Missing':>10}  {'Status'}")
        self.log.write("  " + "-" * 90)
        for col, rate in miss_rates.items():
            n_miss  = int(df[col].isna().sum())
            n_valid = df.shape[0] - n_miss
            pct_ok  = (1 - rate) * 100
            status  = ("COMPLETE" if rate == 0 else
                       "good"     if rate <= 0.2 else
                       "CAUTION"  if rate <= 0.5 else
                       "SPARSE"   if rate <= 0.9 else
                       "NEAR-EMPTY")
            self.log.write(f"  {col:<45} {pct_ok:>9.1f}%  {n_valid:>10,}  "
                           f"{n_miss:>10,}  {status}")

        # Per-year missingness for key fields
        if "source_year" in df.columns:
            self.log.write("")
            self.log.write("  Completeness by census year (columns with any missing):")
            top_miss = miss_rates[(miss_rates > 0) & (miss_rates < 1.0)].head(20).index.tolist()
            years    = sorted(df["source_year"].unique())
            hdr      = ["Column"] + [str(y) for y in years]
            widths   = [45] + [10] * len(years)
            self.log.table_row(hdr, widths)
            self.log.write("  " + "-" * (45 + 12 * len(years)))
            for col in top_miss:
                row = [col]
                for yr in years:
                    sub = df[df["source_year"] == yr][col]
                    pct = (1 - sub.isna().mean()) * 100
                    row.append(f"{pct:.0f}%")
                self.log.table_row(row, widths)

        # Auto-generated descriptive paragraph
        self._write_descriptive_paragraph(miss_rates)

        # 2-panel summary figure
        self._plot_missingness_summary(complete_rates, var_groups)

    def _write_descriptive_paragraph(self, miss_rates: pd.Series) -> None:
        """
        Auto-generate a plain-English descriptive paragraph for the survey.
        Written to the log under a clearly labeled section for easy copy-paste.
        """
        self.log.section("2b. AUTO-GENERATED DESCRIPTIVE PARAGRAPH (copy-paste ready)")

        df   = self.df
        N    = df.shape[0]
        n_cols = df.shape[1]

        # Year breakdown
        year_parts = []
        if "source_year" in df.columns:
            for yr in sorted(df["source_year"].unique()):
                n_yr = int((df["source_year"] == yr).sum())
                data_yr = CENSUS_TO_DATA_YEAR.get(int(yr), yr)
                year_parts.append(f"{yr} census [data year {data_yr}]: n={n_yr:,}")

        year_str = "; ".join(year_parts) if year_parts else f"N={N:,}"

        # Completeness summary
        n_complete = int((miss_rates == 0).sum())
        n_sparse   = int((miss_rates > 0.5).sum())
        n_near_empty = int((miss_rates > 0.9).sum())

        # Identify most and least complete analytical fields
        # (exclude provenance/metadata cols)
        meta_cols = {"census_year", "data_year", "raw_data_year",
                     "source_year", "source_file", "ingest_ts"}
        analytic_rates = miss_rates[[c for c in miss_rates.index
                                     if c not in meta_cols]]
        best_fields  = analytic_rates[analytic_rates <= 0.05].index.tolist()[:4]
        worst_fields = analytic_rates[analytic_rates >= 0.80].index.tolist()[:4]

        # Unique entity count
        id_col = next((c for c in ["operator_id", "vessel_id", "terminal_id",
                                    "segment_id"] if c in df.columns), None)
        entity_str = ""
        if id_col:
            n_unique = int(df[id_col].nunique(dropna=True))
            entity_str = f" representing {n_unique:,} unique {id_col.replace('_id','s')}"

        para = (
            f"The NCFO {self.survey.replace('_', ' ').title()} master dataset contains "
            f"{N:,} records{entity_str} across {n_cols} canonical fields, "
            f"spanning three census waves ({year_str}). "
            f"Of {n_cols} fields, {n_complete} are fully complete across all records; "
            f"{n_sparse} fields are sparse (>50% missing), of which {n_near_empty} are "
            f"near-empty (>90% missing) and retained for structural completeness only. "
        )

        if best_fields:
            para += (f"Fields with high completeness (≥95%) include: "
                     f"{', '.join(best_fields)}. ")
        if worst_fields:
            para += (f"Fields with substantial missingness (≥80%) include: "
                     f"{', '.join(worst_fields)}, "
                     f"reflecting voluntary or conditional survey responses. ")

        para += (
            f"Overall cell-level missingness is "
            f"{df.isna().values.mean():.1%}. "
            f"All missingness patterns were assessed per census wave to distinguish "
            f"structural survey non-response from schema drift across years."
        )

        # Word-wrap at ~100 chars for log readability
        import textwrap
        wrapped = textwrap.fill(para, width=100, initial_indent="  ",
                                subsequent_indent="  ")
        self.log.write(wrapped)
        self.log.write("")
        self.log.write("  [End of auto-paragraph — edit as needed before use]")

    def _get_var_group(self, col: str) -> str:
        """Assign a variable group label to a column name for color-coding."""
        c = col.lower()
        if any(x in c for x in ["_id", "name", "url"]):
            return "identifier"
        if any(x in c for x in ["city", "state", "zip", "country", "lat", "lon",
                                  "latitude", "longitude"]):
            return "geographic"
        if "revenue" in c or "funding_revenue" in c:
            return "revenue"
        if "trip_purpose" in c:
            return "trip_purpose"
        if "pub_fund" in c or "funding" in c:
            return "funding"
        if any(x in c for x in ["capacity", "passenger", "vehicle", "freight"]):
            return "capacity"
        if any(x in c for x in ["length", "breadth", "depth", "tons", "speed",
                                  "horsepower", "year_built", "lifespan"]):
            return "physical"
        if any(x in c for x in ["trips", "passengers", "vehicles", "boarding",
                                  "season", "avg_daily"]):
            return "operational"
        if any(x in c for x in ["census_year", "data_year", "raw_data", "source",
                                  "ingest"]):
            return "metadata"
        return "other"

    def _plot_missingness_summary(
        self,
        complete_rates: pd.Series,
        var_groups: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Missingness summary figure — Panel A always shown, Panel B only when
        there are analytically interesting fields with year-to-year variation.

        Panel A — Completeness bar chart (horizontal, sorted, color-coded by group).
        Panel B — Year-over-year completeness for most-variable fields (omitted when
                   all fields are 100% complete or fewer than 2 fields qualify).
        """
        # No grid style — clean white background
        plt.style.use("seaborn-v0_8-white")

        df = self.df
        N  = df.shape[0]

        # Exclude pure metadata columns from the figure
        meta_cols  = {"raw_data_year", "source_file", "ingest_ts"}
        plot_rates = complete_rates[[c for c in complete_rates.index
                                     if c not in meta_cols]]
        plot_rates = plot_rates.sort_values(ascending=True)

        # Assign group colors
        bar_colors = [GROUP_COLORS.get(self._get_var_group(c), GOV_LTGRAY)
                      for c in plot_rates.index]

        # ---- Determine whether Panel B has content ----
        yr_col = ("source_year" if "source_year" in df.columns else
                  "census_year" if "census_year" in df.columns else None)

        panel_b_cols = []
        if yr_col:
            years = sorted(df[yr_col].dropna().unique())
            analytic_candidates = [
                c for c in plot_rates.index
                if self._get_var_group(c) not in {"metadata", "identifier"}
                and 0.01 < plot_rates[c] < 0.999
            ]
            if analytic_candidates and len(years) >= 2:
                yr_compl = {}
                for col in analytic_candidates:
                    rates_by_yr = [
                        1 - df[df[yr_col] == yr][col].isna().mean()
                        for yr in years
                    ]
                    yr_compl[col] = np.std(rates_by_yr)
                # Only include fields with non-trivial year variation (std > 0.01)
                panel_b_cols = [
                    c for c in sorted(analytic_candidates,
                                      key=lambda c: yr_compl.get(c, 0),
                                      reverse=True)
                    if yr_compl.get(c, 0) > 0.01
                ][:12]

        show_panel_b = len(panel_b_cols) >= 2

        # ---- Figure layout ----
        n_fields   = len(plot_rates)
        bar_height = 0.42
        # Taller per-field allocation so labels are never squished
        fig_h = max(8, min(28, n_fields * bar_height + 3))

        if show_panel_b:
            fig = plt.figure(figsize=(14, fig_h))
        else:
            fig = plt.figure(figsize=(13, fig_h))
        ax_a = fig.add_subplot(1, 1, 1)
        ax_b = None  # Panel B saved separately

        # ---- Panel A: completeness bars ----
        y_pos = np.arange(n_fields)
        bars  = ax_a.barh(y_pos, plot_rates.values * 100,
                          height=bar_height * 0.82,
                          color=bar_colors, edgecolor="white", linewidth=0.4)

        # Subtle horizontal grid behind bars only (x-axis)
        ax_a.xaxis.grid(True, color=GOV_LTGRAY, linewidth=0.5, alpha=0.6)
        ax_a.set_axisbelow(True)

        # Threshold reference lines
        ax_a.axvline(80, color=GOV_ORANGE, ls="--", lw=1.2, alpha=0.8,
                     label="80% threshold")
        ax_a.axvline(50, color=GOV_RED,    ls=":",  lw=1.0, alpha=0.7,
                     label="50% threshold")
        ax_a.axvline(100, color=GOV_GRAY,  ls="-",  lw=0.5, alpha=0.4)

        # Value labels on bars (only for partial completeness)
        for bar, rate in zip(bars, plot_rates.values):
            pct = rate * 100
            if 0.5 < pct < 99.5:
                ax_a.text(
                    min(pct + 1.2, 97), bar.get_y() + bar.get_height() / 2,
                    f"{pct:.0f}%",
                    va="center", ha="left",
                    fontsize=6.5, color=GOV_GRAY,
                )

        # Y-axis: clean field names
        clean_labels = [c.replace("_", " ").replace("pub fund", "funding")
                        for c in plot_rates.index]
        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels(clean_labels, fontsize=7.5)
        ax_a.set_xlabel("% of Records Complete", fontsize=10, fontweight="bold",
                         labelpad=8)
        ax_a.set_xlim(0, 108)
        ax_a.set_title(
            "Panel A — Field Completeness\n(all census years combined)",
            fontsize=10, fontweight="bold", pad=10, loc="left"
        )
        ax_a.tick_params(axis="x", labelsize=9)
        ax_a.spines[["top", "right"]].set_visible(False)

        # Color legend for variable groups
        seen_groups = list(dict.fromkeys(
            self._get_var_group(c) for c in plot_rates.index
        ))
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, fc=GROUP_COLORS.get(g, GOV_LTGRAY),
                           ec="none", label=g.replace("_", " ").title())
            for g in seen_groups if g != "metadata"
        ]
        if legend_patches:
            ax_a.legend(handles=legend_patches,
                        title="Variable Group", title_fontsize=8,
                        fontsize=7.5, loc="lower right",
                        framealpha=0.92, edgecolor=GOV_LTGRAY)

        # Panel B is saved as a separate PNG below — nothing to draw on main fig here

        # ---- Save Panel A as its own PNG ----
        survey_title = self.survey.replace("_", " ").title()
        fig.suptitle(
            f"NCFO {survey_title} Survey — Field Completeness\n"
            f"Total records: {N:,}  |  Census years: 2018, 2020, 2022  |  Fields: {n_fields}",
            fontsize=12, fontweight="bold", y=1.01,
        )
        fig.text(
            0.5, -0.01,
            "Source: National Census of Ferry Operators (NCFO), Bureau of Transportation Statistics. "
            "Colors indicate variable group. Dashed line = 80% completeness threshold.",
            ha="center", fontsize=7.5, color=GOV_GRAY, style="italic"
        )
        out_a = self.figs_dir / f"{self.survey}_completeness_panel_a.png"
        plt.savefig(out_a, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        self.log.write(f"\n  [FIGURE] {out_a.name}")

        # ---- Save Panel B as its own PNG (only when meaningful) ----
        if show_panel_b and yr_col:
            fig_b, ax_b2 = plt.subplots(figsize=(12, 6))
            plt.style.use("seaborn-v0_8-white")

            x       = np.arange(len(panel_b_cols))
            w       = 0.72 / max(len(years), 1)
            yr_list = list(years)

            for i, yr in enumerate(yr_list):
                yr_int = int(yr)
                color  = YEAR_COLORS.get(yr_int, PALETTE[i % len(PALETTE)])
                vals   = [
                    (1 - df[df[yr_col] == yr][col].isna().mean()) * 100
                    for col in panel_b_cols
                ]
                ax_b2.bar(x + i * w, vals, w * 0.92,
                          label=f"Census {yr_int}",
                          color=color, edgecolor="white", linewidth=0.4,
                          alpha=0.90)

            ax_b2.yaxis.grid(True, color=GOV_LTGRAY, linewidth=0.5, alpha=0.6)
            ax_b2.set_axisbelow(True)
            clean_b = [c.replace("_", " ")[:20] for c in panel_b_cols]
            ax_b2.set_xticks(x + w * (len(yr_list) - 1) / 2)
            ax_b2.set_xticklabels(clean_b, rotation=40, ha="right", fontsize=9)
            ax_b2.set_ylabel("% Complete", fontsize=10, fontweight="bold")
            ax_b2.set_ylim(0, 115)
            ax_b2.axhline(80, color=GOV_ORANGE, ls="--", lw=1.0, alpha=0.7,
                          label="80% threshold")
            ax_b2.legend(fontsize=9, framealpha=0.9, edgecolor=GOV_LTGRAY)
            ax_b2.tick_params(axis="both", labelsize=9)
            ax_b2.spines[["top", "right"]].set_visible(False)
            fig_b.suptitle(
                f"NCFO {survey_title} Survey — Completeness by Census Year\n"
                f"(fields with greatest year-to-year variation)",
                fontsize=12, fontweight="bold"
            )
            fig_b.text(
                0.5, -0.02,
                "Source: National Census of Ferry Operators (NCFO), Bureau of Transportation Statistics.",
                ha="center", fontsize=7.5, color=GOV_GRAY, style="italic"
            )
            plt.tight_layout()
            out_b = self.figs_dir / f"{self.survey}_completeness_panel_b.png"
            plt.savefig(out_b, dpi=FIG_DPI, bbox_inches="tight")
            plt.close()
            self.log.write(f"\n  [FIGURE] {out_b.name}")

    # ------------------------------------------------------------------
    # Section 3 — Continuous variable descriptives
    # ------------------------------------------------------------------

    def continuous_descriptives(self, cols: List[str], section_num: int = 3) -> None:
        """
        Full univariate descriptive statistics for specified continuous columns.

        Reports: N valid, N missing, mean, median, SD, IQR, min, max,
        skewness, kurtosis, and percentiles (1, 5, 25, 50, 75, 95, 99).
        Saves distribution figure grid.
        """
        self.log.section(f"{section_num}. CONTINUOUS VARIABLE DESCRIPTIVES")

        numeric_data: Dict[str, pd.Series] = {}
        for col in cols:
            if col not in self.df.columns:
                continue
            parsed, stats = coerce_numeric(self.df[col])
            numeric_data[col] = parsed

            self.log.write("")
            self.log.write(f"  [{col}]")
            self.log.kv("    N valid",   f"{stats['n_parsed']:,}")
            self.log.kv("    N missing", f"{int(self.df.shape[0] - stats['n_parsed']):,}")
            self.log.kv("    N parse failures", f"{stats['n_failed']:,}")

            if stats["n_parsed"] < 2:
                self.log.write("    [insufficient data for descriptives]")
                continue

            s = parsed.dropna()
            self.log.kv("    Mean",     f"{s.mean():.4f}")
            self.log.kv("    Median",   f"{s.median():.4f}")
            self.log.kv("    Std Dev",  f"{s.std():.4f}")
            self.log.kv("    IQR",      f"{s.quantile(0.75) - s.quantile(0.25):.4f}")
            self.log.kv("    Min",      f"{s.min():.4f}")
            self.log.kv("    Max",      f"{s.max():.4f}")
            self.log.kv("    Skewness", f"{float(scipy_stats.skew(s)):.4f}")
            self.log.kv("    Kurtosis", f"{float(scipy_stats.kurtosis(s)):.4f}")

            pcts = s.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            self.log.write("    Percentiles:")
            for p, v in pcts.items():
                self.log.write(f"      P{int(p*100):2d}: {v:.4f}")

            # Normality test (Shapiro for N<=5000, else skewness-based note)
            if 3 <= len(s) <= 5000:
                stat_sw, p_sw = scipy_stats.shapiro(s.sample(min(len(s), 5000),
                                                              random_state=42))
                self.log.kv("    Shapiro-Wilk W", f"{stat_sw:.4f}")
                self.log.kv("    Shapiro-Wilk p", f"{p_sw:.4f}")
                norm_flag = "NORMAL (p>0.05)" if p_sw > 0.05 else "NON-NORMAL (p<=0.05)"
                self.log.kv("    Normality",      norm_flag)
            else:
                self.log.write("    [Normality: N too large for Shapiro; see skewness]")

            # Outliers via IQR
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            self.log.kv("    IQR outliers (1.5×)", f"{n_out:,} ({n_out/len(s):.2%})")

            # Zero inflation
            n_zero = int((s == 0).sum())
            self.log.kv("    Zero values",  f"{n_zero:,} ({n_zero/len(s):.2%})")

            # Year-stratified means
            if "source_year" in self.df.columns:
                self.log.write("    Mean by census year:")
                for yr in sorted(self.df["source_year"].unique()):
                    mask = self.df["source_year"] == yr
                    sub = parsed[mask].dropna()
                    if len(sub):
                        self.log.write(f"      {yr}: n={len(sub):,}  "
                                       f"mean={sub.mean():.3f}  "
                                       f"median={sub.median():.3f}")

        if numeric_data:
            self._plot_distributions(numeric_data, section_num)

    def _plot_distributions(self, numeric_data: Dict[str, pd.Series],
                             section_num: int) -> None:
        """Distribution grid: histogram + KDE for each continuous variable.

        For heavily right-skewed data (skewness > 2 and range > 3 orders of
        magnitude), automatically applies log10(1+x) transform so the
        distribution shape is actually visible. The axis label notes the
        transform.
        """
        n = len(numeric_data)
        if n == 0:
            return
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for ax, (col, series) in zip(axes, numeric_data.items()):
            s = series.dropna()
            if len(s) < 2:
                ax.set_visible(False)
                continue

            # Decide whether to log-transform
            skew = float(s.skew()) if len(s) > 3 else 0
            data_range = (s.max() - s.min())
            use_log = (
                skew > 2.0
                and s.min() >= 0
                and data_range > 0
                and s.max() > 100 * max(s.median(), 1)
            )

            if use_log:
                plot_s   = np.log10(s + 1)
                xlabel   = "log10(1 + Value)"   # ASCII — no encoding issues
                med_disp = f"Median={s.median():.0f}"
                med_line = np.log10(s.median() + 1)
            else:
                plot_s   = s
                xlabel   = "Value"
                med_disp = f"Median={s.median():.1f}"
                med_line = s.median()

            ax.hist(plot_s, bins=40, color=GOV_LTBLUE, edgecolor="white",
                    linewidth=0.4, alpha=0.85, density=True)
            try:
                kde_x = np.linspace(plot_s.min(), plot_s.max(), 300)
                kde   = scipy_stats.gaussian_kde(plot_s)
                # Use dark navy for KDE so it reads clearly over light blue bars
                ax.plot(kde_x, kde(kde_x), color=GOV_NAVY, lw=2.0)
            except Exception:
                pass
            ax.axvline(med_line, color=GOV_ORANGE, ls="--", lw=1.4,
                       label=med_disp)
            ax.set_title(col.replace("_", " ").title(), fontsize=8.5, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel("Density", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5)
            ax.spines[["top", "right"]].set_visible(False)
            # Faint x-grid only — no background fill
            ax.xaxis.grid(True, color=GOV_LTGRAY, linewidth=0.4, alpha=0.5)
            ax.set_axisbelow(True)

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(f"NCFO {self.survey.replace('_', ' ').title()} — Continuous Variable Distributions",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = self.figs_dir / f"{self.survey}_sec{section_num}_distributions.png"
        plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        self.log.write(f"\n  [FIGURE] {out.name}")

    # ------------------------------------------------------------------
    # Section 4 — Categorical variable descriptives
    # ------------------------------------------------------------------

    def categorical_descriptives(self, cols: List[str], section_num: int = 4,
                                  top_n: int = 20) -> None:
        """
        Frequency tables and bar charts for categorical/ordinal columns.

        Parameters
        ----------
        top_n : int
            Maximum number of categories to show in frequency table and figure.
        """
        self.log.section(f"{section_num}. CATEGORICAL VARIABLE DESCRIPTIVES")

        plot_data: Dict[str, pd.Series] = {}
        for col in cols:
            if col not in self.df.columns:
                continue
            s = self.df[col].astype("string").str.strip().replace({"": pd.NA})
            vc = s.value_counts(dropna=True)

            self.log.write(f"\n  [{col}]")
            self.log.kv("    N valid",    f"{int(s.notna().sum()):,}")
            self.log.kv("    N missing",  f"{int(s.isna().sum()):,}")
            self.log.kv("    N unique",   f"{int(vc.shape[0]):,}")
            self.log.write("    Top categories:")
            for val, cnt in vc.head(top_n).items():
                pct = cnt / s.notna().sum() * 100
                self.log.write(f"      {str(val):<35} {cnt:>6,}  ({pct:5.1f}%)")
            if len(vc) > top_n:
                self.log.write(f"      ... ({len(vc) - top_n} more categories)")
            plot_data[col] = vc.head(top_n)

        if plot_data:
            self._plot_categorical_bars(plot_data, section_num)

    def _plot_categorical_bars(self, plot_data: Dict[str, pd.Series],
                                section_num: int) -> None:
        n = len(plot_data)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 5.5, nrows * 4.5))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for ax, (col, vc) in zip(axes, plot_data.items()):
            labels = [str(v)[:25] for v in vc.index]
            bars = ax.barh(labels[::-1], vc.values[::-1],
                           color=GOV_BLUE, edgecolor="white", linewidth=0.4,
                           alpha=0.90)
            ax.set_title(col.replace("_", " ").title(), fontsize=9, fontweight="bold")
            ax.set_xlabel("Count", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(f"NCFO {self.survey.title()} — Categorical Distributions",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = self.figs_dir / f"{self.survey}_sec{section_num}_categorical.png"
        plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        self.log.write(f"\n  [FIGURE] {out.name}")

    # ------------------------------------------------------------------
    # Section 5 — Binary flag profiles
    # ------------------------------------------------------------------

    def binary_flag_profile(self, cols: List[str], section_num: int = 5) -> None:
        """
        Prevalence table for all binary (0/1) flag fields.

        Reports mapping quality, prevalence (% = 1), and year-stratified
        prevalence. Saves a grouped bar chart.
        """
        self.log.section(f"{section_num}. BINARY FLAG PROFILES")

        prevalence: Dict[str, float] = {}
        for col in cols:
            if col not in self.df.columns:
                continue
            normed, stats = normalize_binary(self.df[col])
            n_valid = stats["n_mapped"]
            n_yes   = int((normed == 1).sum()) if n_valid else 0
            pct_yes = n_yes / n_valid * 100 if n_valid else float("nan")

            self.log.write(f"\n  [{col}]")
            self.log.kv("    N valid",    f"{n_valid:,}")
            self.log.kv("    N missing",  f"{int(self.df.shape[0] - n_valid):,}")
            self.log.kv("    N unmapped", f"{stats['n_unmapped']:,}")
            self.log.kv("    N = 1 (Yes)",f"{n_yes:,} ({pct_yes:.1f}%)")
            self.log.kv("    N = 0 (No)", f"{n_valid - n_yes:,} "
                                          f"({100 - pct_yes:.1f}%)")

            prevalence[col] = pct_yes

            if stats["n_unmapped"] > 0:
                s = self.df[col].astype("string").str.strip().replace({"": pd.NA})
                unmapped = s[normed.isna() & s.notna()].value_counts().head(5)
                self.log.write("    Unmapped values:")
                for v, c in unmapped.items():
                    self.log.write(f"      {str(v):<30} {c:,}")

            # Year-stratified prevalence
            if "source_year" in self.df.columns:
                self.log.write("    Prevalence by census year:")
                for yr in sorted(self.df["source_year"].unique()):
                    mask = self.df["source_year"] == yr
                    sub  = normalize_binary(self.df.loc[mask, col])[0]
                    n_v  = int(sub.notna().sum())
                    n_y  = int((sub == 1).sum()) if n_v else 0
                    self.log.write(f"      {yr}: {n_y:,}/{n_v:,} "
                                   f"({n_y/n_v*100:.1f}%)" if n_v else f"      {yr}: N/A")

        if prevalence:
            self._plot_binary_prevalence(prevalence, section_num)

    def _plot_binary_prevalence(self, prevalence: Dict[str, float],
                                 section_num: int) -> None:
        labels = [c.replace("_", "\n").replace("trip\npurpose\n", "")
                  for c in prevalence.keys()]
        values = list(prevalence.values())
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
        bars = ax.bar(range(len(labels)), values,
                      color=[GOV_BLUE if v >= 50 else GOV_STEEL for v in values],
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Prevalence (%)", fontsize=10)
        ax.set_ylim(0, 115)
        ax.axhline(50, color=GOV_GRAY, ls="--", lw=0.9, alpha=0.7,
                   label="50% line")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"NCFO {self.survey.title()} — Binary Flag Prevalence",
                     fontsize=11, fontweight="bold")
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5, f"{val:.0f}%",
                        ha="center", va="bottom", fontsize=7.5)
        plt.tight_layout()
        out = self.figs_dir / f"{self.survey}_sec{section_num}_binary_flags.png"
        plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        self.log.write(f"\n  [FIGURE] {out.name}")

    # ------------------------------------------------------------------
    # Section 6 — Year-over-year panel tracking
    # ------------------------------------------------------------------

    def panel_tracking(self, id_col: str, section_num: int = 6) -> None:
        """
        Track entity persistence across census waves.

        Identifies entities present in 1, 2, or all 3 census years.
        Reports entry (new), exit (dropped), and persistent entities.
        """
        self.log.section(f"{section_num}. PANEL TRACKING — {id_col.upper()}")

        # Accept source_year (raw ingest) or census_year (from-processed) as year column
        yr_col = "source_year" if "source_year" in self.df.columns else (
                 "census_year" if "census_year" in self.df.columns else None)

        if id_col not in self.df.columns or yr_col is None:
            self.log.write("  [SKIP] Required columns not present.")
            return

        years = sorted(self.df[yr_col].dropna().unique())
        year_sets: Dict[int, set] = {}
        for yr in years:
            ids = self.df[self.df[yr_col] == yr][id_col].dropna()
            year_sets[yr] = set(ids.astype(str).str.strip())
            self.log.kv(f"  Census {yr} — unique IDs", f"{len(year_sets[yr]):,}")

        if len(years) >= 2:
            self.log.write("")
            all_ids = set.union(*year_sets.values())
            self.log.kv("  Total unique IDs across all years", f"{len(all_ids):,}")

            # Persistence
            waves_present = {
                eid: sum(1 for ys in year_sets.values() if eid in ys)
                for eid in all_ids
            }
            for n_waves in sorted(set(waves_present.values()), reverse=True):
                cnt = sum(1 for v in waves_present.values() if v == n_waves)
                self.log.write(f"  Present in {n_waves} wave(s): {cnt:,} IDs")

        # Pairwise entry/exit
        for i in range(len(years) - 1):
            y_a, y_b = years[i], years[i + 1]
            if y_a not in year_sets or y_b not in year_sets:
                continue
            new_ids  = year_sets[y_b] - year_sets[y_a]
            lost_ids = year_sets[y_a] - year_sets[y_b]
            self.log.write(f"\n  {y_a} → {y_b}:")
            self.log.kv("    New entries",    f"{len(new_ids):,}")
            self.log.kv("    Dropped entries",f"{len(lost_ids):,}")
            self.log.kv("    Persisted",
                        f"{len(year_sets[y_a] & year_sets[y_b]):,}")

    # ------------------------------------------------------------------
    # Section 7 — Year consistency QC
    # ------------------------------------------------------------------

    def year_consistency(self, section_num: int = 7) -> None:
        """
        Compare embedded raw_data_year against authoritative data_year.

        Documents known 2020-census / 2019-data-year lag explicitly.
        """
        self.log.section(f"{section_num}. YEAR CONSISTENCY QC")

        if "raw_data_year" not in self.df.columns:
            self.log.write("  raw_data_year not present — skipping.")
            return

        self.log.write(
            "  NOTE: The 2020 NCFO census collected data for year 2019. "
            "The pipeline treats census_year as authoritative for panel tracking. "
            "Mismatches below between raw_data_year and data_year for the 2020 wave "
            "reflect this known survey lag and are expected."
        )

        raw = (
            self.df["raw_data_year"].astype("string").str.strip()
                .replace({"": pd.NA, "na": pd.NA, "n/a": pd.NA,
                          "none": pd.NA, "null": pd.NA})
        )
        auth = self.df["data_year"].astype(str)

        self.log.kv("  Rows with embedded raw year", f"{int(raw.notna().sum()):,}")
        self.log.kv("  Mismatches (raw != auth)",    f"{int((raw.notna() & (raw != auth)).sum()):,}")

        self.log.write("\n  Raw year value counts:")
        for val, cnt in raw.value_counts(dropna=True).items():
            self.log.write(f"    {str(val):<10}  {int(cnt):,}")

    # ------------------------------------------------------------------
    # Section 8 — Correlation matrix (continuous cols)
    # ------------------------------------------------------------------

    def correlation_matrix(self, cols: List[str], section_num: int = 8) -> None:
        """
        Pearson and Spearman correlation matrices for continuous columns.

        Saves heatmap figures for both. Flags strong correlations (|r|>0.7).
        """
        self.log.section(f"{section_num}. CORRELATION MATRIX")

        numeric_df = pd.DataFrame()
        for col in cols:
            if col in self.df.columns:
                parsed, _ = coerce_numeric(self.df[col])
                numeric_df[col] = parsed

        valid_cols = [c for c in numeric_df.columns
                      if numeric_df[c].notna().sum() >= 10]
        if len(valid_cols) < 2:
            self.log.write("  [SKIP] Fewer than 2 valid numeric columns.")
            return

        sub = numeric_df[valid_cols].dropna(how="all")

        pearson = sub.corr(method="pearson")
        spearman = sub.corr(method="spearman")

        # Log strong correlations
        self.log.write("  Strong Pearson correlations (|r| > 0.70):")
        found = False
        for c1, c2 in combinations(valid_cols, 2):
            r = pearson.loc[c1, c2]
            if abs(r) > 0.70:
                self.log.write(f"    {c1} × {c2}: r = {r:.3f}")
                found = True
        if not found:
            self.log.write("    None found above threshold.")

        self.log.write("\n  Strong Spearman correlations (|rho| > 0.70):")
        found = False
        for c1, c2 in combinations(valid_cols, 2):
            r = spearman.loc[c1, c2]
            if abs(r) > 0.70:
                self.log.write(f"    {c1} × {c2}: rho = {r:.3f}")
                found = True
        if not found:
            self.log.write("    None found above threshold.")

        self._plot_corr_heatmap(pearson, "Pearson", section_num)
        self._plot_corr_heatmap(spearman, "Spearman", section_num)

    def _plot_corr_heatmap(self, corr: pd.DataFrame, method: str,
                            section_num: int) -> None:
        n = len(corr)
        fig_sz = max(6, min(16, n * 0.9))
        fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.85))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=f"{method} Correlation (r)")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([c.replace("_", "\n") for c in corr.columns],
                           fontsize=7.5, rotation=45, ha="right")
        ax.set_yticklabels([c.replace("_", "\n") for c in corr.index],
                           fontsize=7.5)
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color=GOV_WHITE if abs(val) > 0.65 else GOV_GRAY)
        ax.set_title(
            f"NCFO {self.survey.replace('_',' ').title()} — "
            f"{method} Correlation Matrix",
            fontsize=11, fontweight="bold"
        )
        ax.spines[["top", "right", "left", "bottom"]].set_linewidth(0.5)
        plt.tight_layout()
        out = self.figs_dir / (f"{self.survey}_sec{section_num}_"
                               f"corr_{method.lower()}.png")
        plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        self.log.write(f"\n  [FIGURE] {out.name}")


# ============================================================================
# SURVEY-SPECIFIC EDA MODULES
# ============================================================================

# ----------------------------------------------------------------------------
# OPERATOR
# ----------------------------------------------------------------------------

def eda_operator(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    """
    Comprehensive EDA for the NCFO Operator survey.

    Sections
    --------
    1–8  : Core EdaCore analyses (overview, missingness, distributions, etc.)
    9    : Revenue mix composition (row sums, compositional profiles)
    10   : Operator funding typology (public vs. private vs. mixed)
    11   : Trip purpose co-occurrence and operator typology
    12   : Public funding depth analysis (number of funding sources)
    13   : Geographic distribution (state, country)
    14   : Operator name consistency across years (panel QC)
    """
    core = EdaCore(df, logger, figs_dir, survey="operator")

    # --- Core sections ---
    core.overview()
    core.missingness()
    core.continuous_descriptives(OP_REVENUE_COLS, section_num=3)
    core.categorical_descriptives(["op_state", "op_country", "federal_state_local"],
                                   section_num=4)
    core.binary_flag_profile(OP_TRIP_PURPOSE_COLS + ["accepts_public_funding"],
                              section_num=5)
    core.panel_tracking("operator_id", section_num=6)
    core.year_consistency(section_num=7)
    core.correlation_matrix(OP_REVENUE_COLS, section_num=8)

    # --- Section 9: Revenue mix composition ---
    _eda_op_revenue_composition(df, logger, figs_dir)

    # --- Section 10: Funding typology ---
    _eda_op_funding_typology(df, logger, figs_dir)

    # --- Section 11: Trip purpose co-occurrence ---
    _eda_op_trip_purpose(df, logger, figs_dir)

    # --- Section 12: Public funding depth ---
    _eda_op_funding_depth(df, logger, figs_dir)

    # --- Section 13: Geographic distribution ---
    _eda_op_geography(df, logger, figs_dir)

    # --- Section 14: Name consistency QC ---
    _eda_op_name_consistency(df, logger)


def _eda_op_revenue_composition(df: pd.DataFrame, logger: TxtLogger,
                                 figs_dir: Path) -> None:
    logger.section("9. REVENUE MIX COMPOSITION")

    rev_df = pd.DataFrame()
    for col in OP_REVENUE_COLS:
        if col in df.columns:
            parsed, _ = coerce_numeric(df[col])
            rev_df[col] = parsed

    if rev_df.empty:
        logger.write("  [SKIP] No revenue columns found.")
        return

    # Row-sum check: should be ~100 for complete reporters
    row_sums = rev_df.sum(axis=1, skipna=False)
    n_complete = int(rev_df.notna().all(axis=1).sum())
    sums_complete = row_sums[rev_df.notna().all(axis=1)]

    logger.kv("  Operators with all 8 revenue fields", f"{n_complete:,}")
    if len(sums_complete):
        logger.kv("  Row sum mean (complete reporters)", f"{sums_complete.mean():.2f}")
        logger.kv("  Row sum median",                   f"{sums_complete.median():.2f}")
        logger.kv("  Sum ~100 (95–105)",
                  f"{int(((sums_complete >= 95) & (sums_complete <= 105)).sum()):,}")
        logger.kv("  Sum <90 or >110 (suspicious)",
                  f"{int(((sums_complete < 90) | (sums_complete > 110)).sum()):,}")

    # Median revenue share per source
    logger.write("\n  Median revenue share by source (all reporters):")
    medians = rev_df.median().sort_values(ascending=False)
    for col, med in medians.items():
        short = col.replace("_revenue", "").replace("_funding", "_fund")
        logger.write(f"    {short:<35} {med:6.1f}%")

    # Zero-revenue reporters per category
    logger.write("\n  Operators reporting 0% from each source:")
    for col in OP_REVENUE_COLS:
        if col in rev_df.columns:
            n_zero = int((rev_df[col] == 0).sum())
            short  = col.replace("_revenue", "").replace("_funding", "_fund")
            logger.write(f"    {short:<35} {n_zero:,}")

    # Figure: median revenue shares
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Left: median shares bar
    ax = axes[0]
    short_labels = [c.replace("_revenue", "").replace("_funding", "\nfunding")
                    .replace("_", " ").title()
                    for c in medians.index]
    bars = ax.bar(range(len(medians)), medians.values,
                  color=GOV_BLUE, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.set_xticks(range(len(medians)))
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Median Revenue Share (%)", fontsize=10, fontweight="bold")
    ax.set_title("Median Revenue Mix Across All Operators",
                 fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    # Right: box plots
    ax2 = axes[1]
    plot_data = [rev_df[c].dropna().values for c in medians.index if c in rev_df.columns]
    bp = ax2.boxplot(plot_data, tick_labels=short_labels, vert=True, patch_artist=True,
                     medianprops=dict(color=GOV_ORANGE, lw=2),
                     boxprops=dict(facecolor=GOV_LTBLUE, alpha=0.75),
                     whiskerprops=dict(color=GOV_GRAY),
                     capprops=dict(color=GOV_GRAY),
                     flierprops=dict(marker="o", markersize=3,
                                     markerfacecolor=GOV_GRAY, alpha=0.5))
    ax2.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    ax2.set_ylabel("Revenue Share (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Revenue Share Distribution (box = IQR, line = median)",
                  fontsize=10, fontweight="bold", loc="left")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("NCFO Operator Survey — Revenue Source Composition Analysis",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.02,
             "Source: NCFO Operator Survey, BTS. Percentages reflect share of annual revenue. "
             "Operators with all 8 fields complete used for row-sum validation.",
             ha="center", fontsize=8, color=GOV_GRAY, style="italic")
    plt.tight_layout()
    out = figs_dir / "operator_sec9_revenue_composition.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    logger.write(f"\n  [FIGURE] {out.name}")


def _eda_op_funding_typology(df: pd.DataFrame, logger: TxtLogger,
                              figs_dir: Path) -> None:
    logger.section("10. OPERATOR FUNDING TYPOLOGY")

    if "accepts_public_funding" not in df.columns:
        logger.write("  [SKIP]")
        return

    normed, _ = normalize_binary(df["accepts_public_funding"])
    n_pub  = int((normed == 1).sum())
    n_priv = int((normed == 0).sum())
    n_miss = int(normed.isna().sum())

    logger.kv("  Accepts public funding (Yes)",  f"{n_pub:,} ({n_pub/(n_pub+n_priv)*100:.1f}%)" if n_pub+n_priv else "N/A")
    logger.kv("  Does not accept (No)",          f"{n_priv:,}" )
    logger.kv("  Missing/Unknown",               f"{n_miss:,}")

    # Public funding types breakdown — type fields only, valid codes only
    # Valid values: FEDERAL, STATE, LOCAL (text) or 0/1/2 (legacy numeric codes)
    VALID_FUND_TYPES = {"FEDERAL", "STATE", "LOCAL", "0", "1", "2"}
    logger.write("\n  Public funding type distribution (across all type slots 1–8):")
    logger.write("  Note: Only pub_fund_type_N fields included; coded 0/1/2 = LOCAL/STATE/FEDERAL")
    type_vals = []
    for i in range(1, 9):
        col = f"pub_fund_type_{i}"
        if col in df.columns:
            raw_vals = (
                df[col].astype("string").str.strip().str.upper()
                       .replace({"": pd.NA, "NAN": pd.NA, "NA": pd.NA})
                       .dropna()
            )
            # Keep only valid type codes; flag anything else as encoding noise
            valid   = raw_vals[raw_vals.isin(VALID_FUND_TYPES)]
            invalid = raw_vals[~raw_vals.isin(VALID_FUND_TYPES)]
            type_vals.extend(valid.tolist())
            if len(invalid) > 0:
                logger.write(f"    [QC] {col}: {len(invalid)} non-type values found "
                             f"(likely field mapping error) — excluded from counts")

    # Recode all variants to unified labels
    RECODE = {
        "0": "LOCAL", "1": "STATE", "2": "FEDERAL",
        "LOCAL (CODE 0)": "LOCAL", "STATE (CODE 1)": "STATE",
        "FEDERAL (CODE 2)": "FEDERAL",
    }
    if type_vals:
        type_series = pd.Series(type_vals).str.upper().replace(RECODE)
        logger.write("")
        for val, cnt in type_series.value_counts().items():
            logger.write(f"    {str(val):<25} {cnt:,}")

    # Cross-tab: federal_state_local × accepts_public_funding
    if "federal_state_local" in df.columns:
        logger.write("\n  Cross-tab: federal_state_local × accepts_public_funding:")
        fsl = normalize_binary(df["federal_state_local"])[0]
        ct  = pd.crosstab(fsl.astype("object").fillna("Missing"),
                          normed.astype("object").fillna("Missing"),
                          margins=True)
        logger.write(ct.to_string(max_cols=10))


def _eda_op_trip_purpose(df: pd.DataFrame, logger: TxtLogger,
                          figs_dir: Path) -> None:
    logger.section("11. TRIP PURPOSE PROFILES AND CO-OCCURRENCE")

    purpose_cols = [c for c in OP_TRIP_PURPOSE_COLS if c in df.columns]
    if not purpose_cols:
        logger.write("  [SKIP]")
        return

    # Build binary purpose matrix
    purpose_df = pd.DataFrame()
    for col in purpose_cols:
        purpose_df[col], _ = normalize_binary(df[col])
    purpose_df = purpose_df.astype(float)

    # Count of purposes per operator
    n_purposes = purpose_df.sum(axis=1)
    logger.write("  Number of trip purposes per operator:")
    for n, cnt in n_purposes.value_counts().sort_index().items():
        logger.write(f"    {int(n)} purposes: {cnt:,} operators")

    # Co-occurrence matrix
    logger.write("\n  Trip purpose co-occurrence (% of operators with both):")
    short = {c: c.replace("trip_purpose_", "") for c in purpose_cols}
    n_total = len(purpose_df)
    for c1, c2 in combinations(purpose_cols, 2):
        both = int(((purpose_df[c1] == 1) & (purpose_df[c2] == 1)).sum())
        logger.write(f"    {short[c1]:<25} × {short[c2]:<25}: "
                     f"{both:,} ({both/n_total*100:.1f}%)")

    # Figure: co-occurrence heatmap + prevalence bar
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Co-occurrence heatmap — navy-to-white sequential (gov-appropriate)
    co_mat = purpose_df[purpose_cols].T.dot(purpose_df[purpose_cols]) / len(purpose_df) * 100
    ax = axes[0]
    im = ax.imshow(co_mat.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046,
                        label="Co-occurrence (% of all operators)")
    cbar.ax.tick_params(labelsize=8)
    labels_short = [c.replace("trip_purpose_", "").replace("_", " ").title()
                    for c in purpose_cols]
    ax.set_xticks(range(len(labels_short)))
    ax.set_yticks(range(len(labels_short)))
    ax.set_xticklabels(labels_short, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels_short, fontsize=8.5)
    for i in range(len(purpose_cols)):
        for j in range(len(purpose_cols)):
            val = co_mat.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7.5,
                    color=GOV_WHITE if val > 55 else GOV_GRAY)
    ax.set_title("Panel A — Trip Purpose Co-occurrence\n(% of operators with both purposes)",
                 fontsize=10, fontweight="bold", loc="left")

    # Prevalence by year — grouped bars
    ax2 = axes[1]
    if "source_year" in df.columns:
        years   = sorted(df["source_year"].unique())
        x       = np.arange(len(labels_short))
        w       = 0.72 / max(len(years), 1)
        for i, yr in enumerate(years):
            yr_int = int(yr)
            color  = YEAR_COLORS.get(yr_int, PALETTE[i % len(PALETTE)])
            mask   = df["source_year"] == yr
            prev   = [normalize_binary(df.loc[mask, c])[0].mean() * 100
                      for c in purpose_cols]
            ax2.bar(x + i * w, prev, w * 0.92,
                    label=f"Census {yr_int}", color=color,
                    edgecolor="white", linewidth=0.4, alpha=0.90)
        ax2.set_xticks(x + w * (len(years) - 1) / 2)
        ax2.set_xticklabels(labels_short, rotation=40, ha="right", fontsize=8.5)
        ax2.set_ylabel("% of Operators (Prevalence)", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, 115)
        ax2.axhline(50, color=GOV_GRAY, ls="--", lw=0.8, alpha=0.6)
        ax2.legend(fontsize=9, framealpha=0.92, edgecolor=GOV_LTGRAY)
        ax2.set_title("Panel B — Trip Purpose Prevalence by Census Year",
                      fontsize=10, fontweight="bold", loc="left")
        ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("NCFO Operator Survey — Trip Purpose Profiles",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.02,
             "Source: NCFO Operator Survey, BTS. "
             "Co-occurrence = % of all operators reporting both purposes.",
             ha="center", fontsize=8, color=GOV_GRAY, style="italic")
    plt.tight_layout()
    out = figs_dir / "operator_sec11_trip_purpose.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    logger.write(f"\n  [FIGURE] {out.name}")


def _eda_op_funding_depth(df: pd.DataFrame, logger: TxtLogger,
                           figs_dir: Path) -> None:
    logger.section("12. PUBLIC FUNDING DEPTH")

    type_cols = [f"pub_fund_type_{i}" for i in range(1, 9) if f"pub_fund_type_{i}" in df.columns]
    if not type_cols:
        logger.write("  [SKIP]")
        return

    # Count filled funding slots per operator
    n_slots = df[type_cols].notna().sum(axis=1)
    logger.write("  Number of distinct funding slots used per operator:")
    for n, cnt in n_slots.value_counts().sort_index().items():
        logger.write(f"    {int(n)} slots: {cnt:,} operators")
    logger.kv("  Mean funding slots (all operators)", f"{n_slots.mean():.2f}")
    logger.kv("  Max funding slots",                  f"{int(n_slots.max()):,}")

    # Year-stratified
    if "source_year" in df.columns:
        logger.write("\n  Mean funding slots by census year:")
        for yr in sorted(df["source_year"].unique()):
            mask = df["source_year"] == yr
            sub  = n_slots[mask]
            logger.write(f"    {yr}: mean={sub.mean():.2f}  max={int(sub.max()):,}")


def _eda_op_geography(df: pd.DataFrame, logger: TxtLogger,
                       figs_dir: Path) -> None:
    logger.section("13. GEOGRAPHIC DISTRIBUTION")

    for col in ["op_state", "op_country"]:
        if col not in df.columns:
            continue
        s  = df[col].astype("string").str.strip().str.upper().replace({"": pd.NA})
        vc = s.value_counts(dropna=True)
        logger.write(f"\n  {col} — top 30:")
        for val, cnt in vc.head(30).items():
            logger.write(f"    {str(val):<20} {cnt:,}")

    # Figure: operators per state
    if "op_state" in df.columns:
        s   = df[["op_state", "source_year"]].copy()
        s["op_state"] = s["op_state"].astype("string").str.strip().str.upper()
        if "source_year" in df.columns:
            ct = s.groupby(["op_state", "source_year"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(15, 6))
            plt.style.use("seaborn-v0_8-whitegrid")
            year_cols_present = [c for c in ct.columns]
            colors = [YEAR_COLORS.get(int(c), PALETTE[i])
                      for i, c in enumerate(year_cols_present)]
            ct.plot(kind="bar", ax=ax, color=colors,
                    edgecolor="white", linewidth=0.3, width=0.75, alpha=0.90)
            ax.set_xlabel("State / Territory", fontsize=10, fontweight="bold")
            ax.set_ylabel("Number of Operators", fontsize=10, fontweight="bold")
            ax.set_title("NCFO Operator Survey — Operators per State by Census Year",
                         fontsize=11, fontweight="bold", loc="left")
            ax.tick_params(axis="x", labelsize=8, rotation=90)
            ax.tick_params(axis="y", labelsize=9)
            ax.legend(title="Census Year", fontsize=9, title_fontsize=9,
                      framealpha=0.92, edgecolor=GOV_LTGRAY)
            ax.spines[["top", "right"]].set_visible(False)
            fig.text(0.5, -0.02,
                     "Source: NCFO Operator Survey, BTS. State = mailing address of operator.",
                     ha="center", fontsize=8, color=GOV_GRAY, style="italic")
            plt.tight_layout()
            out = figs_dir / "operator_sec13_state_distribution.png"
            plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
            plt.close()
            logger.write(f"\n  [FIGURE] {out.name}")


def _eda_op_name_consistency(df: pd.DataFrame, logger: TxtLogger) -> None:
    logger.section("14. OPERATOR NAME CONSISTENCY (PANEL QC)")

    if not {"operator_id", "operator_name", "source_year"}.issubset(df.columns):
        logger.write("  [SKIP]")
        return

    # For each operator_id, find all distinct names used across years
    id_names = (
        df.groupby("operator_id")["operator_name"]
          .apply(lambda x: x.astype("string").str.strip().dropna().unique().tolist())
    )

    multi_name = id_names[id_names.apply(len) > 1]
    logger.kv("  Operator IDs with consistent name (1 name)", f"{int((id_names.apply(len) == 1).sum()):,}")
    logger.kv("  Operator IDs with name variations",          f"{len(multi_name):,}")

    if len(multi_name) > 0:
        logger.write("\n  Operators with name variations (sample up to 15):")
        for oid, names in list(multi_name.items())[:15]:
            logger.write(f"    ID {oid}: {names}")


# ----------------------------------------------------------------------------
# VESSEL — survey-specific EDA stub (extend as data becomes available)
# ----------------------------------------------------------------------------

def eda_vessel(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    """
    Comprehensive EDA for the NCFO Vessel survey.

    Sections
    --------
    1–8  : Core EdaCore analyses
    9    : Fleet age analysis (year_built distribution, age by vessel type)
    10   : Capacity analysis (passenger, vehicle by fuel type, vessel type)
    11   : Fuel type and propulsion profiling
    12   : Physical dimensions analysis (length, gross tons, horsepower)
    13   : ADA accessibility and ownership governance
    """
    core = EdaCore(df, logger, figs_dir, survey="vessel")
    core.overview()
    core.missingness()
    core.continuous_descriptives(VES_NUMERIC_COLS, section_num=3)
    core.categorical_descriptives(VES_CATEGORICAL_COLS, section_num=4)
    core.binary_flag_profile(VES_BINARY_COLS, section_num=5)
    core.panel_tracking("vessel_id", section_num=6)
    core.year_consistency(section_num=7)
    core.correlation_matrix(VES_NUMERIC_COLS, section_num=8)

    _eda_ves_fleet_age(df, logger, figs_dir)
    _eda_ves_capacity(df, logger, figs_dir)
    _eda_ves_fuel_propulsion(df, logger, figs_dir)


def _eda_ves_fleet_age(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("9. FLEET AGE ANALYSIS")

    if "year_built" not in df.columns:
        logger.write("  [SKIP]")
        return

    yb, _ = coerce_numeric(df["year_built"])
    current_year = 2022  # Use most recent census year as reference
    age = current_year - yb
    age = age[(age >= 0) & (age <= 150)]  # Plausibility filter

    logger.kv("  Valid year_built records", f"{int(yb.notna().sum()):,}")
    logger.kv("  Fleet age (years) — mean",   f"{age.mean():.1f}")
    logger.kv("  Fleet age (years) — median", f"{age.median():.1f}")
    logger.kv("  Fleet age (years) — max",    f"{age.max():.0f}")
    logger.kv("  Vessels < 10 years old",     f"{int((age < 10).sum()):,}")
    logger.kv("  Vessels > 40 years old",     f"{int((age > 40).sum()):,}")

    if "vessel_type" in df.columns:
        logger.write("\n  Median fleet age by vessel type:")
        for vt in df["vessel_type"].dropna().unique():
            mask = df["vessel_type"].astype("string") == str(vt)
            sub  = age[mask].dropna()
            if len(sub):
                logger.write(f"    {str(vt)}: n={len(sub):,}  median age={sub.median():.1f} yr")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plt.style.use("seaborn-v0_8-whitegrid")
    ax.hist(age.dropna(), bins=30, color=GOV_BLUE, edgecolor="white",
            linewidth=0.5, alpha=0.88)
    ax.axvline(age.median(), color=GOV_ORANGE, ls="--", lw=2.0,
               label=f"Median = {age.median():.0f} years")
    ax.axvline(age.mean(), color=GOV_NAVY, ls=":", lw=1.6,
               label=f"Mean = {age.mean():.0f} years")
    ax.set_xlabel("Vessel Age (years as of 2022 census)", fontsize=10,
                  fontweight="bold")
    ax.set_ylabel("Number of Vessels", fontsize=10, fontweight="bold")
    ax.set_title("NCFO Vessel Survey — Fleet Age Distribution",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, framealpha=0.92, edgecolor=GOV_LTGRAY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.01,
             "Source: NCFO Vessel Survey, BTS. Age calculated as 2022 minus year_built. "
             "Vessels with implausible age (>150 years or negative) excluded.",
             ha="center", fontsize=8, color=GOV_GRAY, style="italic")
    plt.tight_layout()
    out = figs_dir / "vessel_sec9_fleet_age.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    logger.write(f"\n  [FIGURE] {out.name}")


def _eda_ves_capacity(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("10. CAPACITY ANALYSIS")

    for cap_col in ["passenger_capacity", "vehicle_capacity"]:
        if cap_col not in df.columns:
            continue
        parsed, stats = coerce_numeric(df[cap_col])
        # Plausibility filter: passenger ferries typically < 5000 pax
        cap_limit = 5000 if "passenger" in cap_col else 500
        parsed_clean = parsed[(parsed >= 0) & (parsed <= cap_limit)]
        logger.write(f"\n  [{cap_col}]")
        logger.kv("    N valid",    f"{stats['n_parsed']:,}")
        logger.kv("    N > limit",  f"{int((parsed > cap_limit).sum()):,} (plausibility flagged)")
        logger.kv("    Mean (clean)",   f"{parsed_clean.mean():.1f}")
        logger.kv("    Median (clean)", f"{parsed_clean.median():.1f}")
        logger.kv("    Max (clean)",    f"{parsed_clean.max():.0f}")

    if {"passenger_capacity", "fuel_type"}.issubset(df.columns):
        logger.write("\n  Median passenger capacity by fuel type:")
        pc, _ = coerce_numeric(df["passenger_capacity"])
        for ft in df["fuel_type"].dropna().unique():
            mask = df["fuel_type"].astype("string").str.strip() == str(ft)
            sub  = pc[mask].dropna()
            if len(sub):
                logger.write(f"    {str(ft):<20} n={len(sub):,}  "
                             f"median={sub.median():.0f}  mean={sub.mean():.0f}")


def _eda_ves_fuel_propulsion(df: pd.DataFrame, logger: TxtLogger,
                              figs_dir: Path) -> None:
    logger.section("11. FUEL TYPE AND PROPULSION ANALYSIS")

    for col in ["fuel_type", "propulsion_type", "hull_material"]:
        if col not in df.columns:
            continue
        s  = df[col].astype("string").str.strip().replace({"": pd.NA})
        vc = s.value_counts(dropna=True)
        logger.write(f"\n  [{col}] — {int(s.notna().sum()):,} valid:")
        for val, cnt in vc.items():
            logger.write(f"    {str(val):<35} {cnt:,} ({cnt/len(s)*100:.1f}%)")

    # Fuel type trend over years
    if {"fuel_type", "source_year"}.issubset(df.columns):
        logger.write("\n  Fuel type by census year:")
        ct = pd.crosstab(
            df["fuel_type"].astype("string").str.strip(),
            df["source_year"]
        )
        logger.write(ct.to_string())


# ----------------------------------------------------------------------------
# TERMINAL — survey-specific EDA stub
# ----------------------------------------------------------------------------

def eda_terminal(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    """
    Comprehensive EDA for the NCFO Terminal survey.

    Sections
    --------
    1–8  : Core EdaCore analyses
    9    : Multimodal connectivity scoring and profiling
    10   : Geographic analysis (state, lat/lon bounding box validation)
    11   : Governance structure (ownership × operation cross-tab)
    """
    core = EdaCore(df, logger, figs_dir, survey="terminal")
    core.overview()
    core.missingness()
    core.continuous_descriptives(["latitude", "longitude"], section_num=3)
    core.categorical_descriptives(
        ["term_state", "term_country", "terminal_ownership", "terminal_operation"],
        section_num=4)
    core.binary_flag_profile(TERM_BINARY_COLS, section_num=5)
    core.panel_tracking("terminal_id", section_num=6)
    core.year_consistency(section_num=7)

    _eda_term_multimodal(df, logger, figs_dir)
    _eda_term_geography(df, logger, figs_dir)
    _eda_term_governance(df, logger, figs_dir)


def _eda_term_multimodal(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("9. MULTIMODAL CONNECTIVITY SCORING")

    avail_cols = [c for c in TERM_MULTIMODAL_COLS if c in df.columns]
    if not avail_cols:
        logger.write("  [SKIP]")
        return

    score_df = pd.DataFrame()
    for col in avail_cols:
        score_df[col], _ = normalize_binary(df[col])
    score_df = score_df.astype(float)

    score = score_df.sum(axis=1)
    logger.kv("  Max possible score",  f"{len(avail_cols)}")
    logger.kv("  Mean connectivity score",   f"{score.mean():.2f}")
    logger.kv("  Median connectivity score", f"{score.median():.2f}")
    logger.write("\n  Score distribution:")
    for n, cnt in score.value_counts().sort_index().items():
        logger.write(f"    Score {int(n)}: {cnt:,} terminals")

    logger.write("\n  Prevalence of each connectivity type:")
    for col in avail_cols:
        n_yes = int((score_df[col] == 1).sum())
        logger.write(f"    {col:<25} {n_yes:,} ({n_yes/len(df)*100:.1f}%)")


def _eda_term_geography(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("10. GEOGRAPHIC VALIDATION")

    for coord, (lo, hi) in [("latitude", (24.0, 72.0)), ("longitude", (-180.0, -60.0))]:
        if coord not in df.columns:
            continue
        parsed, stats = coerce_numeric(df[coord])
        n_valid   = stats["n_parsed"]
        n_in_box  = int(((parsed >= lo) & (parsed <= hi)).sum())
        n_out_box = n_valid - n_in_box
        logger.kv(f"  {coord} valid",          f"{n_valid:,}")
        logger.kv(f"  {coord} in US bounding box",  f"{n_in_box:,}")
        logger.kv(f"  {coord} OUTSIDE bounding box",f"{n_out_box:,}")

    if "term_state" in df.columns:
        logger.write("\n  Terminals per state:")
        s  = df["term_state"].astype("string").str.strip().str.upper()
        vc = s.value_counts(dropna=True)
        for val, cnt in vc.head(25).items():
            logger.write(f"    {str(val):<10} {cnt:,}")


def _eda_term_governance(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("11. TERMINAL GOVERNANCE STRUCTURE")

    for col in ["terminal_ownership", "terminal_operation"]:
        if col not in df.columns:
            continue
        s  = df[col].astype("string").str.strip().replace({"": pd.NA})
        vc = s.value_counts(dropna=True)
        logger.write(f"\n  [{col}]:")
        for val, cnt in vc.items():
            logger.write(f"    {str(val):<20} {cnt:,}")

    if {"terminal_ownership", "terminal_operation"}.issubset(df.columns):
        logger.write("\n  Cross-tab: ownership × operation:")
        own = df["terminal_ownership"].astype("string").str.strip().replace({"": pd.NA})
        ops = df["terminal_operation"].astype("string").str.strip().replace({"": pd.NA})
        ct  = pd.crosstab(own.fillna("Missing"), ops.fillna("Missing"), margins=True)
        logger.write(ct.to_string())


# ----------------------------------------------------------------------------
# SEGMENT — survey-specific EDA stub
# ----------------------------------------------------------------------------

def eda_segment(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    """
    Comprehensive EDA for the NCFO Segment survey.

    Sections
    --------
    1–7  : Core EdaCore analyses
    8    : Segment type and NPS service profiling
    9    : Network topology (terminal degree, segment counts)
    """
    core = EdaCore(df, logger, figs_dir, survey="segment")
    core.overview()
    core.missingness()
    core.categorical_descriptives(["seg_type"], section_num=3)
    core.binary_flag_profile(["serves_nps"], section_num=4)
    core.panel_tracking("segment_id", section_num=5)
    core.year_consistency(section_num=6)
    _eda_seg_type_nps(df, logger, figs_dir)
    _eda_seg_network(df, logger, figs_dir)


def _eda_seg_type_nps(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("7. SEGMENT TYPE AND NPS SERVICE PROFILE")

    seg_type_map = {"1": "Intrastate", "2": "Interstate", "3": "International"}
    if "seg_type" in df.columns:
        s  = df["seg_type"].astype("string").str.strip().map(seg_type_map)
        vc = s.value_counts(dropna=True)
        logger.write("  Segment type distribution:")
        for val, cnt in vc.items():
            logger.write(f"    {str(val):<20} {cnt:,} ({cnt/len(df)*100:.1f}%)")

    if {"seg_type", "serves_nps"}.issubset(df.columns):
        logger.write("\n  NPS service by segment type:")
        nps, _ = normalize_binary(df["serves_nps"])
        for raw_type, label in seg_type_map.items():
            mask = df["seg_type"].astype("string").str.strip() == raw_type
            sub  = nps[mask]
            n_nps = int((sub == 1).sum())
            logger.write(f"    {label}: {n_nps:,}/{int(mask.sum()):,} serve NPS "
                         f"({n_nps/int(mask.sum())*100:.1f}%)"
                         if mask.sum() else f"    {label}: no records")


def _eda_seg_network(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("8. NETWORK TOPOLOGY")

    for col in ["seg_terminal1_id", "seg_terminal2_id"]:
        if col not in df.columns:
            continue
        s  = df[col].astype("string").str.strip().replace({"": pd.NA})
        logger.kv(f"  Unique terminal IDs in {col}", f"{int(s.nunique()):,}")

    if {"seg_terminal1_id", "seg_terminal2_id"}.issubset(df.columns):
        all_terminals = pd.concat([
            df["seg_terminal1_id"].astype("string").str.strip(),
            df["seg_terminal2_id"].astype("string").str.strip(),
        ]).replace({"": pd.NA}).dropna()

        degree = all_terminals.value_counts()
        logger.kv("  Total unique terminal nodes", f"{int(degree.shape[0]):,}")
        logger.kv("  Max terminal degree (segments)", f"{int(degree.max()):,}")
        logger.kv("  Mean terminal degree",           f"{degree.mean():.2f}")
        logger.write("\n  Terminal degree distribution (top 10 hubs):")
        for tid, deg in degree.head(10).items():
            logger.write(f"    Terminal {tid}: {deg:,} segments")


# ----------------------------------------------------------------------------
# OPERATOR-SEGMENT — survey-specific EDA stub
# ----------------------------------------------------------------------------

def eda_operator_segment(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    """
    Comprehensive EDA for the NCFO Operator-Segment survey.

    This is the most analytically dense table — demand meets capacity here.

    Sections
    --------
    1–8  : Core EdaCore analyses
    9    : Demand metrics (passengers, vehicles, daily boardings)
    10   : Service supply metrics (trips/year, segment length, trip time)
    11   : Seasonality analysis (season length, year-round vs. seasonal)
    12   : Fleet deployment (vessels per segment, most-used vessel linkage)
    13   : Fare regulation profiling
    14   : Derived utilization metrics (requires vessel join — documented)
    """
    core = EdaCore(df, logger, figs_dir, survey="operator_segment")
    core.overview()
    core.missingness()
    core.continuous_descriptives(OPSEG_NUMERIC_COLS, section_num=3)
    core.categorical_descriptives(["route_rates_regulated"], section_num=4)
    core.panel_tracking("segment_id", section_num=5)
    core.year_consistency(section_num=6)
    core.correlation_matrix(OPSEG_NUMERIC_COLS, section_num=7)

    _eda_opseg_demand(df, logger, figs_dir)
    _eda_opseg_service(df, logger, figs_dir)
    _eda_opseg_seasonality(df, logger, figs_dir)
    _eda_opseg_fleet_deployment(df, logger, figs_dir)
    _eda_opseg_fare_regulation(df, logger, figs_dir)
    _eda_opseg_utilization_note(df, logger)


def _eda_opseg_demand(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("8. DEMAND METRICS")

    for col in ["passengers", "vehicles", "avg_daily_brd_pax", "avg_daily_brd_veh"]:
        if col not in df.columns:
            continue
        parsed, stats = coerce_numeric(df[col])
        s = parsed[(parsed >= 0)].dropna()
        logger.write(f"\n  [{col}]")
        logger.kv("    N valid",   f"{stats['n_parsed']:,}")
        logger.kv("    N missing", f"{int(df.shape[0] - stats['n_parsed']):,}")
        if len(s) < 2:
            continue
        logger.kv("    Mean",    f"{s.mean():,.0f}")
        logger.kv("    Median",  f"{s.median():,.0f}")
        logger.kv("    Max",     f"{s.max():,.0f}")
        logger.kv("    P95",     f"{s.quantile(0.95):,.0f}")
        logger.kv("    Zeros",   f"{int((s == 0).sum()):,} ({(s==0).mean():.1%})")
        logger.kv("    Log10 mean", f"{np.log10(s[s>0]).mean():.2f}")

        # Year-stratified
        if "source_year" in df.columns:
            logger.write("    By census year:")
            for yr in sorted(df["source_year"].unique()):
                mask = df["source_year"] == yr
                sub  = parsed[mask][(parsed[mask] >= 0)].dropna()
                if len(sub):
                    logger.write(f"      {yr}: n={len(sub):,}  "
                                 f"median={sub.median():,.0f}  "
                                 f"sum={sub.sum():,.0f}")


def _eda_opseg_service(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("9. SERVICE SUPPLY METRICS")

    for col, unit in [("seg_length", "nautical miles"),
                      ("average_trip_time", "minutes"),
                      ("trips_per_year", "trips")]:
        if col not in df.columns:
            continue
        parsed, stats = coerce_numeric(df[col])
        s = parsed[(parsed > 0)].dropna()
        logger.write(f"\n  [{col}] ({unit})")
        logger.kv("    N valid",  f"{stats['n_parsed']:,}")
        logger.kv("    Mean",     f"{s.mean():.2f}")
        logger.kv("    Median",   f"{s.median():.2f}")
        logger.kv("    Min",      f"{s.min():.2f}")
        logger.kv("    Max",      f"{s.max():.2f}")
        logger.kv("    P95",      f"{s.quantile(0.95):.2f}")

    # Implied speed check (seg_length / trip_time_hours)
    if {"seg_length", "average_trip_time"}.issubset(df.columns):
        length, _ = coerce_numeric(df["seg_length"])
        time_h, _ = coerce_numeric(df["average_trip_time"])
        time_h = time_h / 60
        implied_speed = length / time_h.replace(0, np.nan)
        plausible = implied_speed[(implied_speed > 0) & (implied_speed < 50)]
        logger.write("\n  Implied vessel speed (seg_length / trip_time):")
        logger.kv("    N computable",    f"{int(implied_speed.notna().sum()):,}")
        logger.kv("    N plausible (<50 knots)", f"{len(plausible):,}")
        logger.kv("    Mean speed (knots)",   f"{plausible.mean():.2f}")
        logger.kv("    Median speed (knots)", f"{plausible.median():.2f}")
        logger.kv("    N implausible",
                  f"{int((implied_speed.notna() & ((implied_speed <= 0) | (implied_speed >= 50))).sum()):,}")


def _eda_opseg_seasonality(df: pd.DataFrame, logger: TxtLogger, figs_dir: Path) -> None:
    logger.section("10. SEASONALITY ANALYSIS")

    if not {"segment_season_start", "segment_season_end"}.issubset(df.columns):
        logger.write("  [SKIP]")
        return

    def parse_mmdd(s: pd.Series) -> pd.Series:
        return pd.to_datetime(
            "2022/" + s.astype("string").str.strip(),
            format="%Y/%m/%d", errors="coerce"
        )

    start = parse_mmdd(df["segment_season_start"])
    end   = parse_mmdd(df["segment_season_end"])

    n_start  = int(start.notna().sum())
    n_end    = int(end.notna().sum())
    n_both   = int((start.notna() & end.notna()).sum())

    logger.kv("  N with season start date", f"{n_start:,}")
    logger.kv("  N with season end date",   f"{n_end:,}")
    logger.kv("  N with both dates",        f"{n_both:,}")

    if n_both > 0:
        # Handle wrap-around seasons (e.g., Nov–Mar)
        # Use np.where to avoid SettingWithCopyWarning on timedelta series
        raw_days = (end - start).dt.days
        season_len = pd.Series(
            np.where(raw_days < 0, raw_days + 365, raw_days),
            index=raw_days.index
        )
        valid_len = season_len[(season_len > 0) & (season_len <= 365)]

        logger.kv("  Season length — mean (days)",   f"{valid_len.mean():.1f}")
        logger.kv("  Season length — median (days)", f"{valid_len.median():.1f}")
        logger.kv("  Year-round (365 days)",
                  f"{int((valid_len >= 360).sum()):,} ({(valid_len >= 360).mean():.1%})")
        logger.kv("  Seasonal (<270 days)",
                  f"{int((valid_len < 270).sum()):,} ({(valid_len < 270).mean():.1%})")

        # Season start month distribution
        logger.write("\n  Season start month distribution:")
        for m, cnt in start.dt.month.value_counts().sort_index().items():
            logger.write(f"    Month {int(m):2d}: {cnt:,}")


def _eda_opseg_fleet_deployment(df: pd.DataFrame, logger: TxtLogger,
                                 figs_dir: Path) -> None:
    logger.section("11. FLEET DEPLOYMENT PER SEGMENT")

    vessel_cols = [c for c in _OPSEG_VESSEL_IDS if c in df.columns]
    if not vessel_cols:
        logger.write("  [SKIP]")
        return

    n_vessels = df[vessel_cols].notna().sum(axis=1)
    logger.write("  Vessels deployed per operator-segment:")
    for n, cnt in n_vessels.value_counts().sort_index().items():
        logger.write(f"    {int(n)} vessels: {cnt:,} operator-segments")
    logger.kv("  Mean vessels per segment",  f"{n_vessels.mean():.2f}")
    logger.kv("  Max vessels per segment",   f"{int(n_vessels.max()):,}")

    # Most-used vessel linkage completeness
    if "most_used_vessel_id" in df.columns:
        n_linked = int(df["most_used_vessel_id"].notna().sum())
        logger.kv("  Segments with most_used_vessel_id", f"{n_linked:,} ({n_linked/len(df):.1%})")


def _eda_opseg_fare_regulation(df: pd.DataFrame, logger: TxtLogger,
                                figs_dir: Path) -> None:
    logger.section("12. FARE REGULATION PROFILING")

    if "route_rates_regulated" not in df.columns:
        logger.write("  [SKIP]")
        return

    reg_map = {"0": "Not Regulated", "1": "Regulated", "2": "Unknown"}
    s  = df["route_rates_regulated"].astype("string").str.strip().map(reg_map)
    vc = s.value_counts(dropna=True)

    logger.write("  Fare regulation status:")
    for val, cnt in vc.items():
        logger.write(f"    {str(val):<20} {cnt:,} ({cnt/len(df)*100:.1f}%)")

    if "source_year" in df.columns:
        logger.write("\n  Regulation status by census year:")
        ct = pd.crosstab(
            df["route_rates_regulated"].astype("string").str.strip().map(reg_map).fillna("Missing"),
            df["source_year"]
        )
        logger.write(ct.to_string())


def _eda_opseg_utilization_note(df: pd.DataFrame, logger: TxtLogger) -> None:
    logger.section("13. UTILIZATION RATE NOTE")
    logger.write(
        "  Utilization rate = passengers / (trips_per_year × vessel_passenger_capacity)\n"
        "  requires joining operator_segment to vessel via most_used_vessel_id.\n"
        "  This join is computed in the cross-survey analysis module (see run_cross_survey).\n"
        "  Fields available here for the numerator:\n"
        "    - passengers (annual total)\n"
        "    - avg_daily_brd_pax (daily average)\n"
        "  Fields requiring join for denominator:\n"
        "    - passenger_capacity (from vessel table)\n"
        "    - trips_per_year (available here)\n"
        "  Run run_cross_survey() after all individual surveys to compute utilization."
    )


# ============================================================================
# SOURCES REGISTRY
# ============================================================================

def build_sources(root: Path) -> Dict[str, List[SourceSpec]]:
    """
    Build the full registry of raw data file paths for all surveys and years.

    Modify paths here if your directory structure differs.
    All file existence checks are deferred to run_survey().

    Returns
    -------
    dict mapping survey name → list of SourceSpec
    """
    raw = root / "data" / "raw"
    return {
        "operator": [
            SourceSpec(2022, raw / "operator" / "operator_2022.csv",  "csv"),
            SourceSpec(2020, raw / "operator" / "operator_2020.csv",  "csv"),
            SourceSpec(2018, raw / "operator" / "operator_2018.xlsx", "xlsx"),
        ],
        "vessel": [
            SourceSpec(2022, raw / "vessel" / "vessel_2022.csv",  "csv"),
            SourceSpec(2020, raw / "vessel" / "vessel_2020.csv",  "csv"),
            SourceSpec(2018, raw / "vessel" / "vessel_2018.xlsx", "xlsx"),
        ],
        "terminal": [
            SourceSpec(2022, raw / "terminal" / "terminal_2022.csv",  "csv"),
            SourceSpec(2020, raw / "terminal" / "terminal_2020.csv",  "csv"),
            SourceSpec(2018, raw / "terminal" / "terminal_2018.xlsx", "xlsx"),
        ],
        "segment": [
            SourceSpec(2022, raw / "segment" / "segment_2022.csv",  "csv"),
            SourceSpec(2020, raw / "segment" / "segment_2020.csv",  "csv"),
            SourceSpec(2018, raw / "segment" / "segment_2018.xlsx", "xlsx"),
        ],
        "operator_segment": [
            SourceSpec(2022, raw / "operator_segment" / "operator_segment_2022.csv",  "csv"),
            SourceSpec(2020, raw / "operator_segment" / "operator_segment_2020.csv",  "csv"),
            SourceSpec(2018, raw / "operator_segment" / "operator_segment_2018.xlsx", "xlsx"),
        ],
    }


# ============================================================================
# SURVEY RUNNER
# ============================================================================

EDA_DISPATCH = {
    "operator":         eda_operator,
    "vessel":           eda_vessel,
    "terminal":         eda_terminal,
    "segment":          eda_segment,
    "operator_segment": eda_operator_segment,
}


def run_survey(survey: str, root: Path) -> int:
    """
    Full EDA pipeline for a single survey type across all census years.

    1. Ingests and canonicalizes each source year.
    2. Unions into a master dataframe.
    3. Exports processed CSV.
    4. Runs survey-specific EDA module.
    5. Saves log and all figures.

    Parameters
    ----------
    survey : str
        One of: operator, vessel, terminal, segment, operator_segment
    root : Path
        Project root directory.

    Returns
    -------
    int
        0 on success, 2 on missing file error.
    """
    sources   = build_sources(root)
    canonical = CANONICAL_MAP[survey]
    eda_fn    = EDA_DISPATCH[survey]

    ts          = _ts()
    log_path    = root / "logs" / survey / f"{survey}_eda_{ts}.txt"
    out_csv     = root / "data" / "processed" / f"{survey}_master.csv"
    figs_dir    = root / "figures" / survey

    logger = TxtLogger(log_path, echo=True)

    # Header
    logger.section(f"DOT BTS NCFO — {survey.upper()} Survey EDA/QC")
    logger.kv("Run timestamp",  datetime.now().isoformat(timespec="seconds"))
    logger.kv("Project root",   str(root))
    logger.kv("Processed CSV",  str(out_csv))
    logger.kv("Figures dir",    str(figs_dir))
    logger.write("\nInput files:")
    for s in sources[survey]:
        logger.write(f"  {s.year}  {s.path}  [{s.kind}]")

    # Ingest + canonicalize — skip missing files with warning, don't hard-stop
    logger.section("INGEST AND SCHEMA CANONICALIZATION")
    frames: List[pd.DataFrame] = []
    n_skipped = 0

    for spec in sources[survey]:
        if not spec.path.exists():
            logger.write(f"  [WARN] File not found — skipping: {spec.path}")
            logger.write(f"         Expected: {spec.path.name}")
            logger.write(f"         Check that filename matches exactly (case-sensitive on Linux).")
            n_skipped += 1
            continue

        df_raw            = read_source(spec)
        canon_df, h_stats = canonicalize(df_raw, spec, canonical)

        logger.write(
            f"  {spec.year}: raw rows={df_raw.shape[0]:,}  "
            f"raw cols={df_raw.shape[1]:,}  "
            f"raw_year_cols_found={h_stats['n_raw_year_cols_found']}  "
            f"(data_year={CENSUS_TO_DATA_YEAR.get(spec.year, spec.year)})"
        )
        frames.append(canon_df)

    if not frames:
        logger.write(f"\n  [ERROR] No input files found for survey '{survey}'. "
                     f"All {n_skipped} source(s) missing.")
        logger.write(f"  Verify filenames in build_sources() match your actual files.")
        logger.write(f"  Tip: run  python src/ncfo_eda.py --list-files  to see expected paths.")
        logger.save()
        return 2

    if n_skipped:
        logger.write(f"\n  [WARN] {n_skipped} year(s) skipped due to missing files. "
                     f"Results reflect {len(frames)} available year(s) only.")

    master = pd.concat(frames, ignore_index=True)

    # Export
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_csv, index=False)
    logger.kv("Master rows", f"{master.shape[0]:,}")
    logger.kv("Master cols", f"{master.shape[1]:,}")

    # Run EDA
    eda_fn(master, logger, figs_dir)

    # Footer
    logger.section("OUTPUTS WRITTEN")
    logger.kv("Processed CSV", str(out_csv))
    logger.kv("Log file",      str(log_path))
    logger.kv("Figures dir",   str(figs_dir))
    logger.save()

    print(f"\n[DONE] {survey} — log: {log_path.name}")
    return 0


# ============================================================================
# CROSS-SURVEY ANALYSIS
# ============================================================================

def run_survey_from_processed(survey: str, root: Path) -> int:
    """
    Run full EDA on an already-processed master CSV, skipping raw ingest.

    Use this when the multi-year union has already been produced (either by
    a prior run of run_survey() or by an external process) and only the EDA
    figures and log need to be regenerated.

    The processed CSV must exist at:
        data/processed/<survey>_master.csv

    Parameters
    ----------
    survey : str
        One of: operator, vessel, terminal, segment, operator_segment
    root : Path
        Project root directory.

    Returns
    -------
    int
        0 on success, 2 if the processed CSV is missing.
    """
    eda_fn   = EDA_DISPATCH[survey]
    ts       = _ts()
    csv_path = root / "data" / "processed" / f"{survey}_master.csv"
    log_path = root / "logs" / survey / f"{survey}_eda_{ts}.txt"
    figs_dir = root / "figures" / survey

    logger = TxtLogger(log_path, echo=True)

    logger.section(f"DOT BTS NCFO — {survey.upper()} Survey EDA/QC  [from-processed mode]")
    logger.kv("Run timestamp", datetime.now().isoformat(timespec="seconds"))
    logger.kv("Project root",  str(root))
    logger.kv("Source CSV",    str(csv_path))
    logger.kv("Figures dir",   str(figs_dir))

    if not csv_path.exists():
        logger.write(f"\n  [ERROR] Processed CSV not found: {csv_path}")
        logger.write(f"  Run  python src/ncfo_eda.py --survey {survey}  first to generate it,")
        logger.write(f"  or place your pre-built master CSV at the path above.")
        logger.save()
        return 2

    master = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Normalize column headers to lower_snake_case.
    # Processed CSVs written by external pipelines may have uppercase headers
    # (e.g., VESSEL_ID, CENSUS_YEAR). All EDA logic expects lowercase.
    master.columns = [to_snake(c) for c in master.columns]

    # Restore numeric dtypes for year columns used in year-stratified analyses
    for col in ("source_year", "census_year", "data_year"):
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")

    logger.kv("Rows loaded", f"{master.shape[0]:,}")
    logger.kv("Cols loaded", f"{master.shape[1]:,}")

    if "source_year" in master.columns:
        logger.write("\n  Census year breakdown:")
        for yr, cnt in master["source_year"].value_counts().sort_index().items():
            data_yr = CENSUS_TO_DATA_YEAR.get(int(yr), yr) if pd.notna(yr) else "?"
            logger.write(f"    census {yr} (data {data_yr}): {cnt:,} rows")

    # Run EDA
    eda_fn(master, logger, figs_dir)

    # Footer
    logger.section("OUTPUTS WRITTEN")
    logger.kv("Source CSV",  str(csv_path))
    logger.kv("Log file",    str(log_path))
    logger.kv("Figures dir", str(figs_dir))
    logger.save()

    print(f"\n[DONE] {survey} — log: {log_path.name}")
    return 0


# ============================================================================
# CROSS-SURVEY ANALYSIS
# ============================================================================

def run_cross_survey(root: Path) -> None:
    """
    Cross-survey joined analyses requiring multiple canonicalized master CSVs.

    Requires operator_master.csv, vessel_master.csv, terminal_master.csv,
    segment_master.csv, and operator_segment_master.csv to be present in
    data/processed/.

    Analyses
    --------
    - Utilization rate: passengers / (trips_per_year × passenger_capacity)
    - Vessel capacity attribution per segment
    - Terminal multimodal score by operator governance type
    - Operator typology enriched with segment demand
    """
    processed = root / "data" / "processed"
    figs_dir  = root / "figures" / "cross_survey"
    figs_dir.mkdir(parents=True, exist_ok=True)

    ts       = _ts()
    log_path = root / "logs" / "cross_survey" / f"cross_survey_eda_{ts}.txt"
    logger   = TxtLogger(log_path, echo=True)

    logger.section("DOT BTS NCFO — CROSS-SURVEY JOINED ANALYSIS")
    logger.kv("Run timestamp", datetime.now().isoformat(timespec="seconds"))

    # Load available masters
    masters: Dict[str, pd.DataFrame] = {}
    for survey in ["operator", "vessel", "terminal", "segment", "operator_segment"]:
        p = processed / f"{survey}_master.csv"
        if p.exists():
            df = pd.read_csv(p, dtype=str, low_memory=False)
            df.columns = [to_snake(c) for c in df.columns]
            masters[survey] = df
            logger.kv(f"  Loaded {survey}", f"{masters[survey].shape[0]:,} rows")
        else:
            logger.write(f"  [MISSING] {survey}_master.csv — skipping dependent analyses")

    # --- Utilization rate ---
    if {"operator_segment", "vessel"}.issubset(masters):
        logger.section("UTILIZATION RATE COMPUTATION")
        opseg  = masters["operator_segment"].copy()
        vessel = masters["vessel"].copy()

        # Coerce key fields
        for col in ["passengers", "trips_per_year"]:
            if col in opseg.columns:
                opseg[col], _ = coerce_numeric(opseg[col])
        if "passenger_capacity" in vessel.columns:
            vessel["passenger_capacity"], _ = coerce_numeric(vessel["passenger_capacity"])

        # Join on most_used_vessel_id → vessel_id
        # Use year-matched join when a year column is available in both tables;
        # fall back to vessel_id-only join if year columns differ between paths.
        yr_col_os = ("source_year" if "source_year" in opseg.columns else
                     "census_year" if "census_year" in opseg.columns else None)
        yr_col_v  = ("source_year" if "source_year" in vessel.columns else
                     "census_year" if "census_year" in vessel.columns else None)

        vessel_cols = ["vessel_id", "passenger_capacity"]
        if yr_col_v:
            vessel_cols.append(yr_col_v)

        if "most_used_vessel_id" in opseg.columns:
            if yr_col_os and yr_col_v and yr_col_os == yr_col_v:
                # Year-matched join (most precise)
                opseg_join = opseg.merge(
                    vessel[vessel_cols],
                    left_on=["most_used_vessel_id", yr_col_os],
                    right_on=["vessel_id",          yr_col_v],
                    how="left"
                )
                logger.write(f"  Join method: vessel_id × {yr_col_os} (year-matched)")
            else:
                # Vessel_id-only join — deduplicate vessel to most recent record per id
                vessel_dedup = (
                    vessel.sort_values(yr_col_v, ascending=False)
                    .drop_duplicates(subset=["vessel_id"])
                    [["vessel_id", "passenger_capacity"]]
                ) if yr_col_v else vessel[["vessel_id", "passenger_capacity"]].drop_duplicates()
                opseg_join = opseg.merge(
                    vessel_dedup,
                    left_on="most_used_vessel_id",
                    right_on="vessel_id",
                    how="left"
                )
                logger.write("  Join method: vessel_id only (year columns differ between tables)")

            pc   = pd.to_numeric(opseg_join["passenger_capacity"], errors="coerce")
            tpy  = pd.to_numeric(opseg_join["trips_per_year"],      errors="coerce")
            pax  = pd.to_numeric(opseg_join["passengers"],          errors="coerce")

            theoretical_cap = tpy * pc
            util_rate = pax / theoretical_cap.replace(0, np.nan)

            n_computable = int(util_rate.notna().sum())
            util_clean   = util_rate[(util_rate > 0) & (util_rate <= 2.0)]

            logger.kv("  Segments with computable utilization", f"{n_computable:,}")
            logger.kv("  Utilization rate — mean",   f"{util_clean.mean():.3f}")
            logger.kv("  Utilization rate — median", f"{util_clean.median():.3f}")
            logger.kv("  Utilization rate — P95",    f"{util_clean.quantile(0.95):.3f}")
            logger.kv("  Segments >100% util (data anomaly)", f"{int((util_rate > 1.0).sum()):,}")

            # Figure: utilization distribution
            fig, ax = plt.subplots(figsize=(10, 5.5))
            plt.style.use("seaborn-v0_8-whitegrid")
            ax.hist(util_clean, bins=40, color=GOV_BLUE, edgecolor="white",
                    linewidth=0.5, alpha=0.88)
            ax.axvline(util_clean.median(), color=GOV_ORANGE, ls="--", lw=2.0,
                       label=f"Median = {util_clean.median():.2f}")
            ax.axvline(util_clean.mean(), color=GOV_NAVY, ls=":", lw=1.6,
                       label=f"Mean = {util_clean.mean():.2f}")
            ax.axvline(1.0, color=GOV_RED, ls="-", lw=1.2, alpha=0.7,
                       label="100% capacity")
            ax.set_xlabel("Passenger Utilization Rate\n"
                          "(annual boardings ÷ theoretical annual capacity)",
                          fontsize=10, fontweight="bold")
            ax.set_ylabel("Number of Route Segments", fontsize=10, fontweight="bold")
            ax.set_title("NCFO — Passenger Utilization Rate Distribution\n"
                         "(operator-segment level, most-used vessel capacity)",
                         fontsize=11, fontweight="bold", loc="left")
            ax.legend(fontsize=9, framealpha=0.92, edgecolor=GOV_LTGRAY)
            ax.spines[["top", "right"]].set_visible(False)
            fig.text(0.5, -0.01,
                     "Source: NCFO Operator-Segment and Vessel surveys, BTS. "
                     "Theoretical capacity = trips_per_year × passenger_capacity. "
                     "Rates >2.0 excluded as data anomalies.",
                     ha="center", fontsize=8, color=GOV_GRAY, style="italic")
            plt.tight_layout()
            out = figs_dir / "cross_utilization_rate.png"
            plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
            plt.close()
            logger.write(f"\n  [FIGURE] {out.name}")

    logger.section("CROSS-SURVEY OUTPUTS WRITTEN")
    logger.kv("Log", str(log_path))
    logger.save()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="NCFO EDA pipeline — run all surveys or specific ones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/ncfo_eda.py                              # all surveys, auto-detect raw vs processed
  python src/ncfo_eda.py --from-processed             # all surveys, load from data/processed/ CSVs
  python src/ncfo_eda.py --survey operator            # operator only (auto-detect)
  python src/ncfo_eda.py --survey vessel --from-processed
  python src/ncfo_eda.py --survey cross               # cross-survey join only
  python src/ncfo_eda.py --list-files                 # show expected raw file paths and exit
        """
    )
    parser.add_argument(
        "--survey", nargs="*",
        choices=list(EDA_DISPATCH.keys()) + ["cross"],
        help="Survey(s) to run. Omit to run all.",
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Project root override (default: auto-detected).",
    )
    parser.add_argument(
        "--from-processed", action="store_true",
        help=(
            "Load from data/processed/<survey>_master.csv instead of raw files. "
            "Use when raw ingest has already been done. "
            "This is also the automatic fallback when raw files are not found."
        ),
    )
    parser.add_argument(
        "--list-files", action="store_true",
        help="Print all expected input file paths and exit.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else find_project_root()

    if args.list_files:
        print(f"\nProject root : {root}")
        print(f"\nExpected raw input files:\n")
        sources = build_sources(root)
        for survey, specs in sources.items():
            print(f"  [{survey}]")
            for s in specs:
                status = "OK     " if s.path.exists() else "MISSING"
                print(f"    [{status}]  {s.path}")
        print(f"\nProcessed CSVs (used with --from-processed):\n")
        for survey in EDA_DISPATCH:
            p = root / "data" / "processed" / f"{survey}_master.csv"
            status = "OK     " if p.exists() else "MISSING"
            print(f"    [{status}]  {p}")
        print()
        return 0

    surveys = args.survey or list(EDA_DISPATCH.keys())
    print(f"Project root : {root}")
    print(f"Surveys      : {surveys}")

    # Determine runner per survey:
    # - If --from-processed is set, always use run_survey_from_processed
    # - Otherwise, check if ALL raw files exist for the survey; if not, auto-fallback
    sources = build_sources(root)

    def _pick_runner(survey: str):
        if args.from_processed:
            return run_survey_from_processed
        raw_specs = sources.get(survey, [])
        all_raw_exist = all(s.path.exists() for s in raw_specs)
        if all_raw_exist:
            return run_survey
        # Auto-fallback: check if processed CSV exists
        processed_csv = root / "data" / "processed" / f"{survey}_master.csv"
        if processed_csv.exists():
            print(f"  [AUTO] {survey}: raw files not found, "
                  f"falling back to processed CSV at data/processed/")
            return run_survey_from_processed
        # Neither raw nor processed available — run_survey will produce clean error
        return run_survey

    exit_codes: Dict[str, int] = {}
    for survey in surveys:
        if survey == "cross":
            run_cross_survey(root)
        else:
            runner = _pick_runner(survey)
            exit_codes[survey] = runner(survey, root)

    # Run cross-survey if all individual surveys succeeded
    if not args.survey and all(c == 0 for c in exit_codes.values()):
        print("\n[INFO] All surveys complete — running cross-survey analysis")
        run_cross_survey(root)

    n_failed = sum(1 for c in exit_codes.values() if c != 0)
    if n_failed:
        print(f"\n[WARN] {n_failed} survey(s) failed — check logs for details")
        return 1

    print("\n[SUCCESS] NCFO EDA pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
