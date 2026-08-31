"""
spptpandas.py
===================
Spatial Point Pattern Test (SPPT) -- pandas version.

For a chosen spatial unit of aggregation (e.g. ward, uber hexagon, street), this compares
crime counts in a user-specified date window ("test" period) against a
"base" period that is either:

  - the immediately preceding window of equal length  -> comparison="immediateprior"
  - the same calendar window one year earlier          -> comparison="sameperiodlastyear"

For every combination of spatial unit x crime theme (plus an aggregated
"ALLTHEMES" theme), it returns two complementary "S-index or similar"
measures used in the SPPT literature:

  1. Andresen's classic Index of Similarity (s_index), 0-1, 1 = identical
     spatial pattern. See Andresen (2009, 2016); martin-a-andresen/sppt.aggregated.data
  2. A per-unit chi-square proportion-difference test with false-discovery-rate
     correction (chi2_p / fdr_q), following Wheeler, Steenbeek & Andresen (2018)
     as implemented in apwheele/crimepy (crimepy/sppt.py)

References
----------
Andresen, M.A. (2009). Testing for similarity in area-based spatial patterns:
    a reapplication of the spatial point pattern test to test for spatial
    divergence. Canadian Journal of Regional Science.
Andresen, M.A. (2016). An area-based nonparametric spatial point pattern
    test: the test, its applications, and the future. Methodological
    Innovations, 9.
Wheeler, A.P., Steenbeek, W., & Andresen, M.A. (2018). Testing for
    similarity in area-based spatial patterns: alternative methods to
    Andresen's spatial point pattern test. Transactions in GIS, 22(3).

Dependencies: pandas, numpy, scipy.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control

ALLTHEMES_LABEL = "ALLTHEMES"

_COMPARISON_ALIASES = {
    "immediateprior": "immediateprior",
    "immediate_prior": "immediateprior",
    "prior": "immediateprior",
    "sameperiodlastyear": "sameperiodlastyear",
    "sameaslastyear": "sameperiodlastyear",
    "same_as_last_year": "sameperiodlastyear",
    "yoy": "sameperiodlastyear",
    "lastyear": "sameperiodlastyear",
}


# --------------------------------------------------------------------------
# Date helpers 
# --------------------------------------------------------------------------

def _to_date(d: Union[str, _dt.date, _dt.datetime]) -> _dt.date:
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    return _dt.datetime.strptime(str(d).strip(), "%Y-%m-%d").date()


def _shift_years(d: _dt.date, years: int) -> _dt.date:
    """Shift a date by whole calendar years, handling Feb 29 safely."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)


def _resolve_periods(start_date, end_date, comparison: str):
    """Return (base_start, base_end, test_start, test_end) as date objects,
    all bounds inclusive."""
    test_start = _to_date(start_date)
    test_end = _to_date(end_date)
    if test_end < test_start:
        raise ValueError("end_date must not be before start_date")

    key = _COMPARISON_ALIASES.get(str(comparison).lower().strip())
    if key is None:
        raise ValueError(
            "comparison must be 'immediateprior' or 'sameperiodlastyear' "
            f"(got {comparison!r})"
        )

    n_days = (test_end - test_start).days + 1  # inclusive day count

    if key == "immediateprior":
        base_end = test_start - _dt.timedelta(days=1)
        base_start = base_end - _dt.timedelta(days=n_days - 1)
    else:  # sameperiodlastyear
        base_start = _shift_years(test_start, -1)
        base_end = _shift_years(test_end, -1)

    return base_start, base_end, test_start, test_end


# --------------------------------------------------------------------------
# pandas-side aggregation
# --------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in dataframe: {missing}")


def _period_counts_pdf(
    df: pd.DataFrame,
    date_col: str,
    unit_col: str,
    theme_col: str,
    x_col: Optional[str],
    y_col: Optional[str],
    date_format: Optional[str],
    base_start: _dt.date,
    base_end: _dt.date,
    test_start: _dt.date,
    test_end: _dt.date,
    include_alltheme: bool,
) -> pd.DataFrame:
    d = df.copy()

    if date_format:
        # NOTE: this is a pandas/strftime-style format string (e.g. '%m/%d/%Y'),
        # NOT a Java/Spark SimpleDateFormat pattern (e.g. 'MM/dd/yyyy') --
        # the two syntaxes differ. See the date_format parameter docs on
        # sppt_test() below for common token translations.
        d["_evt_date"] = pd.to_datetime(d[date_col], format=date_format, errors="coerce").dt.date
    else:
        d["_evt_date"] = pd.to_datetime(d[date_col], errors="coerce").dt.date

    mask = d[unit_col].notna() & d[theme_col].notna() & d["_evt_date"].notna()
    if x_col:
        mask &= d[x_col].notna()
    if y_col:
        mask &= d[y_col].notna()
    d = d.loc[mask].copy()

    is_base = (d["_evt_date"] >= base_start) & (d["_evt_date"] <= base_end)
    is_test = (d["_evt_date"] >= test_start) & (d["_evt_date"] <= test_end)
    d = d.loc[is_base | is_test].copy()
    # base and test windows never overlap (by construction in _resolve_periods),
    # so once restricted to their union, ">= test_start" unambiguously picks out test rows
    d["_period"] = np.where(d["_evt_date"] >= test_start, "test", "base")

    by_theme = (
        d.groupby([unit_col, theme_col, "_period"]).size().reset_index(name="count")
        .rename(columns={unit_col: "unit", theme_col: "theme"})
    )

    frames = [by_theme]
    if include_alltheme:
        by_all = (
            d.groupby([unit_col, "_period"]).size().reset_index(name="count")
            .rename(columns={unit_col: "unit"})
        )
        by_all["theme"] = ALLTHEMES_LABEL
        by_all = by_all[["unit", "theme", "_period", "count"]]
        frames.append(by_all)

    return pd.concat(frames, ignore_index=True)


def _to_wide(pdf: pd.DataFrame, all_units, all_themes) -> pd.DataFrame:
    """Pivot to one row per (unit, theme) with n_base/n_test columns, and
    expand to the FULL unit x theme grid (filling 0) so that units with zero
    events for a given theme in both periods are still represented -- this
    is what makes remove_zero=False and the 'standard' global S-index
    meaningful, rather than silently dropping those combinations."""
    wide = pdf.pivot_table(
        index=["unit", "theme"], columns="_period", values="count", aggfunc="sum", fill_value=0
    )
    for col in ("base", "test"):
        if col not in wide.columns:
            wide[col] = 0
    wide = wide[["base", "test"]]

    full_index = pd.MultiIndex.from_product([all_units, all_themes], names=["unit", "theme"])
    wide = wide.reindex(full_index, fill_value=0).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"base": "n_base", "test": "n_test"})
    return wide


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _chi2_row(row, correction: bool) -> float:
    c1, c2, R1, R2 = row["n_base"], row["n_test"], row["R_base"], row["R_test"]
    if (c1 + c2) == 0:
        return np.nan
    table = np.array([[c1, R1], [c2, R2]], dtype=float)
    try:
        _, p, _, _ = chi2_contingency(table, correction=correction)
    except ValueError:
        return np.nan
    return p


def _compute_stats_full(
    wide: pd.DataFrame, alpha: float, fdr_method: str, yates_correction: bool
) -> pd.DataFrame:
    frames = []
    for theme, g in wide.groupby("theme", sort=False):
        g = g.copy()
        N_base = float(g["n_base"].sum())
        N_test = float(g["n_test"].sum())
        g["N_base_total"] = N_base
        g["N_test_total"] = N_test

        if N_base > 0 and N_test > 0:
            g["R_base"] = N_base - g["n_base"]
            g["R_test"] = N_test - g["n_test"]
            g["p_base"] = g["n_base"] / N_base
            g["p_test"] = g["n_test"] / N_test
            g["diff"] = g["p_test"] - g["p_base"]

            denom = g["p_base"] + g["p_test"]
            g["s_index"] = np.where(denom > 0, 1 - (g["diff"].abs() / denom), 1.0)

            g["chi2_p"] = g.apply(lambda r: _chi2_row(r, yates_correction), axis=1)
            valid = g["chi2_p"].notna()
            g["fdr_q"] = np.nan
            if valid.any():
                g.loc[valid, "fdr_q"] = false_discovery_control(
                    g.loc[valid, "chi2_p"].values, method=fdr_method
                )

            g["classification"] = np.select(
                [~valid, g["fdr_q"] < alpha],
                ["no_events", "Dissimilar (sig. change)"],
                default="Similar (no sig. change)",
            )
            g = g.drop(columns=["R_base", "R_test"])
        else:
            g["p_base"] = np.nan
            g["p_test"] = np.nan
            g["diff"] = np.nan
            g["s_index"] = np.nan
            g["chi2_p"] = np.nan
            g["fdr_q"] = np.nan
            g["classification"] = "insufficient_data"

        g["direction"] = np.select(
            [g["diff"].isna(), g["diff"] > 0, g["diff"] < 0],
            ["N/A", "Increase", "Decrease"],
            default="No change",
        )
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def _global_summary(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for theme, g in full.groupby("theme", sort=False):
        nonzero = g[(g["n_base"] + g["n_test"]) > 0]
        n_dissim = int((g["classification"] == "Dissimilar (sig. change)").sum())
        rows.append(
            {
                "theme": theme,
                "n_spatial_units": len(g),
                "n_units_with_events": len(nonzero),
                "N_base_total": g["N_base_total"].iloc[0],
                "N_test_total": g["N_test_total"].iloc[0],
                "global_S_standard": g["s_index"].mean(skipna=True),
                "global_S_robust": nonzero["s_index"].mean(skipna=True) if len(nonzero) else np.nan,
                "n_units_dissimilar": n_dissim,
                "pct_units_dissimilar": (n_dissim / len(nonzero) * 100) if len(nonzero) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def sppt_test(
    df: pd.DataFrame,
    start_date: Union[str, _dt.date],
    end_date: Union[str, _dt.date],
    theme_col: str = "crimetype",
    unit_col: str = "ward",
    date_col: str = "datefrome",
    x_col: Optional[str] = "easting",
    y_col: Optional[str] = "northing",
    comparison: str = "immediateprior",
    alpha: float = 0.05,
    fdr_method: str = "by",
    yates_correction: bool = True,
    remove_zero: bool = True,
    include_alltheme: bool = True,
    date_format: Optional[str] = None,
    unit_universe: Optional[list] = None,
    return_summary: bool = False,
) -> Union[pd.DataFrame, tuple]:
    """
    Run a Spatial Point Pattern Test (SPPT) by crime theme, for a chosen
    spatial unit of aggregation, comparing a user-defined date window against
    a base period. 

    Parameters
    ----------
    df : pandas.DataFrame
        Point-level event data. One row per incident.
    start_date, end_date : str ('YYYY-MM-DD') or datetime.date
        Inclusive bounds of the "test" period the analyst wants to evaluate.
    theme_col : str, default 'crimetype'
        String column to group by (e.g. crime type). Results are produced
        separately for each distinct value, plus a rolled-up 'ALLTHEMES' row
        per spatial unit if include_alltheme=True.
    unit_col : str, default 'ward'
        Spatial unit of aggregation already present on df (e.g. ward, hex id).
        String, integer, or long are all fine.
    date_col : str, default 'datefrome'
        Date/timestamp column (or an ISO 'YYYY-MM-DD' string column) used to
        assign each event to the base or test period.
    x_col, y_col : str or None, default 'easting', 'northing'
        Optional coordinate columns. Not used in the statistical calculation
        itself (aggregation is by unit_col) -- if supplied they are only used
        as a data-quality filter to drop records with missing/null
        coordinates before aggregating. Pass None to skip this filter, e.g.
        if you don't have or don't want to check coordinate columns.
    comparison : {'immediateprior', 'sameperiodlastyear'}, default 'immediateprior'
        How the base period is derived from start_date/end_date:
          - 'immediateprior': the base period is the immediately preceding
            window of equal length in days.
          - 'sameperiodlastyear' (alias 'sameaslastyear'): the base period is
            the same calendar window exactly one year earlier.
    alpha : float, default 0.05
        Significance threshold applied to the FDR-adjusted q-value to flag a
        unit as 'Dissimilar (sig. change)' vs 'Similar (no sig. change)'.
    fdr_method : {'by', 'bh'}, default 'by'
        False discovery rate correction method passed to
        scipy.stats.false_discovery_control. 'by' (Benjamini-Yekutieli) is
        robust to correlated tests (the default, and generally the safer
        choice for spatially adjacent units); 'bh' (Benjamini-Hochberg)
        assumes independence and has more power if that assumption holds.
    yates_correction : bool, default True
        Whether to apply Yates' continuity correction in the underlying
        2x2 chi-square test (scipy.stats.chi2_contingency).
    remove_zero : bool, default True
        If True (default), spatial units with zero events in BOTH the base
        and test period for a given theme are dropped from the detailed
        output (they carry no information about a change). They are still
        included in the global summary's 'standard' S-index if
        return_summary=True, matching the standard-vs-robust S-index
        distinction in the SPPT literature.
    include_alltheme : bool, default True
        If True, also compute a rolled-up 'ALLTHEMES' theme summing across
        every distinct value of theme_col.
    date_format : str or None, default None
        A pandas/strftime-style format string (e.g. '%m/%d/%Y') to parse
        date_col if it is a string not already in ISO 'yyyy-MM-dd' format.
        IMPORTANT: this is different syntax from the Databricks/PySpark
        version's date_format, which uses Java SimpleDateFormat tokens
        (e.g. 'MM/dd/yyyy'). Common translations: 'yyyy-MM-dd' -> '%Y-%m-%d',
        'MM/dd/yyyy' -> '%m/%d/%Y', 'dd/MM/yyyy' -> '%d/%m/%Y'.
        Leave as None if date_col is already a datetime dtype column, or an
        ISO-format string.
    unit_universe : list or None, default None
        Optional. By default, the set of spatial units compared is inferred
        from the data itself: every distinct unit_col value with at least
        one event (of ANY theme) in the base or test window. A unit that
        truly has zero events of every theme across both periods will not
        appear unless you pass its value here (e.g. a full list of ward
        codes/hex ids for your study area) -- useful if you want
        genuinely-empty units included as zero/zero rows when
        remove_zero=False, or reflected in the 'standard' global S-index.
    return_summary : bool, default False
        If True, also returns a second DataFrame with one row per theme:
        total counts, global standard/robust S-index, and the count and
        percent of spatial units flagged as significantly dissimilar.

    Returns
    -------
    pandas.DataFrame
        One row per (unit_col, theme_col) with columns:
        n_base, n_test, N_base_total, N_test_total, p_base, p_test, diff,
        s_index, chi2_p, fdr_q, classification, direction,
        base_period_start, base_period_end, test_period_start,
        test_period_end, comparison_type.
        - s_index: Andresen's classic Index of Similarity (0-1; 1 = identical
          spatial share of events between the two periods).
        - chi2_p / fdr_q: p-value / FDR-adjusted q-value from a per-unit
          2x2 proportion-difference (chi-square) test of whether that unit's
          share of citywide events changed between the two periods.
    (pandas.DataFrame, pandas.DataFrame)
        If return_summary=True: (detail_df, summary_df).

    Examples
    --------
    >>> result = sppt_test(
    ...     df,
    ...     start_date="2026-06-01",
    ...     end_date="2026-06-30",
    ...     theme_col="crimetype",
    ...     unit_col="ward",
    ...     date_col="datefrome",
    ...     comparison="sameperiodlastyear",
    ... )
    >>> result.head()

    >>> detail, summary = sppt_test(
    ...     df,
    ...     start_date="2026-06-01",
    ...     end_date="2026-06-07",
    ...     unit_col="hex",
    ...     comparison="immediateprior",
    ...     return_summary=True,
    ... )
    >>> summary.head()
    """
    check_cols = [theme_col, unit_col, date_col]
    if x_col:
        check_cols.append(x_col)
    if y_col:
        check_cols.append(y_col)
    _validate_columns(df, check_cols)

    base_start, base_end, test_start, test_end = _resolve_periods(start_date, end_date, comparison)

    pdf_counts = _period_counts_pdf(
        df,
        date_col=date_col,
        unit_col=unit_col,
        theme_col=theme_col,
        x_col=x_col,
        y_col=y_col,
        date_format=date_format,
        base_start=base_start,
        base_end=base_end,
        test_start=test_start,
        test_end=test_end,
        include_alltheme=include_alltheme,
    )

    if pdf_counts.empty:
        raise ValueError(
            "No events found in the base or test date window -- check date_col, "
            "start_date/end_date, and that unit_col/theme_col are populated."
        )

    all_units = set(pdf_counts["unit"].unique().tolist())
    if unit_universe:
        all_units |= set(unit_universe)
    all_units = sorted(all_units, key=lambda v: (str(type(v)), v))

    all_themes = sorted(t for t in pdf_counts["theme"].unique().tolist() if t != ALLTHEMES_LABEL)
    if include_alltheme:
        all_themes = all_themes + [ALLTHEMES_LABEL]

    wide = _to_wide(pdf_counts, all_units=all_units, all_themes=all_themes)
    full = _compute_stats_full(wide, alpha=alpha, fdr_method=fdr_method, yates_correction=yates_correction)

    summary_pdf = _global_summary(full) if return_summary else None

    detail = full[(full["n_base"] + full["n_test"]) > 0].copy() if remove_zero else full

    for col_name, val in [
        ("base_period_start", base_start.isoformat()),
        ("base_period_end", base_end.isoformat()),
        ("test_period_start", test_start.isoformat()),
        ("test_period_end", test_end.isoformat()),
        ("comparison_type", comparison),
    ]:
        detail[col_name] = val

    detail = detail.rename(columns={"unit": unit_col, "theme": theme_col})
    ordered_cols = [
        unit_col,
        theme_col,
        "n_base",
        "n_test",
        "N_base_total",
        "N_test_total",
        "p_base",
        "p_test",
        "diff",
        "s_index",
        "chi2_p",
        "fdr_q",
        "classification",
        "direction",
        "base_period_start",
        "base_period_end",
        "test_period_start",
        "test_period_end",
        "comparison_type",
    ]
    detail = detail[ordered_cols].reset_index(drop=True)

    if return_summary:
        summary_pdf = summary_pdf.rename(columns={"theme": theme_col})
        return detail, summary_pdf

    return detail
