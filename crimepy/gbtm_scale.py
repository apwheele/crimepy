"""
gbtmpandas.py
====================
Group-Based Trajectory Modeling (GBTM) -- pandas version.

Given row-level event data with a spatial unit already assigned (ward,
hexagon, etc.), this bins events into fixed-length time periods (e.g. 28
days for a shorter dataset, 365 days for a many-year dataset), builds a
unit-by-period count matrix for each crime theme (plus a rolled-up
'ALLTHEMES' theme), and fits a group-based trajectory model that assigns
every spatial unit to one of a small number of latent trajectory groups --
e.g. "chronic high", "emerging", "declining", "chronic low" -- following
Nagin's group-based trajectory modeling approach as commonly applied to
area-based crime counts (see e.g. Weisburd, Bushway, Lum & Yang (2004) on
crime trajectories at micro places; Andresen, Curman & Linning; and
Wheeler's SPSS/R crimCV-based GBTM tutorials).

Two fitting methods are supported:

  - method="poisson_mixture" (default): a finite mixture of Poisson
    trajectories, one polynomial-in-time curve per latent group, fit by the
    EM algorithm -- this is the standard statistical formulation of GBTM
    for count data (as in Stata/SAS `traj` and R `crimCV`). The number of
    groups K is chosen by BIC over a user-supplied range unless overridden.
  - method="kmeans": a simpler, faster shape-based clustering on
    log-transformed period rates, with K chosen by silhouette score. This
    is a common, more approximate substitute when a fast first pass is
    wanted, or as a sanity check against the Poisson mixture result.

Dependencies: pandas, numpy, scipy, statsmodels, scikit-learn.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

try:
    import statsmodels.api as sm
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "gbtmpandas requires statsmodels. Install it with pip install statsmodels."
    ) from _e

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "gbtmpandas requires scikit-learn. Install it with pip install scikit-learn."
    ) from _e

ALLTHEMES_LABEL = "ALLTHEMES"


# ==========================================================================
# Date / period-binning helpers (pure Python/datetime/numpy)
# ==========================================================================

def _to_date(d: Union[str, _dt.date, _dt.datetime]) -> _dt.date:
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    return _dt.datetime.strptime(str(d).strip(), "%Y-%m-%d").date()


def _build_period_bins(start_date: _dt.date, end_date: _dt.date, period_days: int, drop_partial_period: bool):
    """Return (n_periods, period_starts, period_ends, period_days_actual)."""
    if period_days <= 0:
        raise ValueError("period_days must be a positive integer")
    total_days = (end_date - start_date).days + 1
    n_full = total_days // period_days
    remainder = total_days - n_full * period_days

    n_periods = n_full + (0 if (remainder == 0 or drop_partial_period) else 1)
    if n_periods < 2:
        raise ValueError(
            f"Only {n_periods} period(s) fit between {start_date} and {end_date} with "
            f"period_days={period_days}. Choose a shorter period_days or a wider date range "
            "-- GBTM needs multiple time points per unit."
        )

    period_starts, period_ends, period_len = [], [], []
    for i in range(n_full):
        s = start_date + _dt.timedelta(days=i * period_days)
        e = s + _dt.timedelta(days=period_days - 1)
        period_starts.append(s)
        period_ends.append(e)
        period_len.append(period_days)
    if n_periods > n_full:
        s = start_date + _dt.timedelta(days=n_full * period_days)
        e = end_date
        period_starts.append(s)
        period_ends.append(e)
        period_len.append(remainder)

    return n_periods, period_starts, period_ends, np.array(period_len, dtype=float)


# ==========================================================================
# pandas-side aggregation
# ==========================================================================

def _validate_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in dataframe: {missing}")


def _infer_date_bounds(df: pd.DataFrame, date_col: str, date_format: Optional[str]):
    if date_format:
        parsed = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    else:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        raise ValueError(f"No parseable dates found in column {date_col!r}.")
    return parsed.min().date(), parsed.max().date()


def _period_counts_pdf(
    df: pd.DataFrame,
    unit_col: str,
    theme_col: str,
    date_col: str,
    date_format: Optional[str],
    start_date: _dt.date,
    period_days: int,
    n_periods: int,
    include_alltheme: bool,
) -> pd.DataFrame:
    d = df.copy()
    if date_format:
        # NOTE: pandas/strftime-style format string (e.g. '%m/%d/%Y'), NOT a
        # Java/Spark SimpleDateFormat pattern (e.g. 'MM/dd/yyyy') -- see the
        # date_format parameter docs on gbtm_trajectories() below.
        d["_evt_date"] = pd.to_datetime(d[date_col], format=date_format, errors="coerce").dt.normalize()
    else:
        d["_evt_date"] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()

    mask = d[unit_col].notna() & d[theme_col].notna() & d["_evt_date"].notna()
    d = d.loc[mask].copy()

    start_ts = pd.Timestamp(start_date)
    d["_period_index"] = ((d["_evt_date"] - start_ts).dt.days // period_days).astype("int64")
    d = d[(d["_period_index"] >= 0) & (d["_period_index"] < n_periods)]

    by_theme = (
        d.groupby([unit_col, theme_col, "_period_index"]).size().reset_index(name="count")
        .rename(columns={unit_col: "unit", theme_col: "theme", "_period_index": "period_index"})
    )

    frames = [by_theme]
    if include_alltheme:
        by_all = (
            d.groupby([unit_col, "_period_index"]).size().reset_index(name="count")
            .rename(columns={unit_col: "unit", "_period_index": "period_index"})
        )
        by_all["theme"] = ALLTHEMES_LABEL
        by_all = by_all[["unit", "theme", "period_index", "count"]]
        frames.append(by_all)

    return pd.concat(frames, ignore_index=True)


def _build_matrix(long_pdf: pd.DataFrame, all_units: list, n_periods: int) -> np.ndarray:
    """(n_units, n_periods) count matrix, rows ordered as all_units, zero-filled."""
    unit_index = {u: i for i, u in enumerate(all_units)}
    Y = np.zeros((len(all_units), n_periods), dtype=float)
    rows = long_pdf["unit"].map(unit_index).to_numpy()
    cols = long_pdf["period_index"].to_numpy()
    Y[rows, cols] = long_pdf["count"].to_numpy(dtype=float)
    return Y


# ==========================================================================
# Poisson finite-mixture GBTM (EM) 
# version (apure numpy/statsmodels/sklearn)
# ==========================================================================

def _design_matrix(t_scaled: np.ndarray, degree: int) -> np.ndarray:
    return np.vstack([t_scaled ** p for p in range(degree + 1)]).T


def _fit_poisson_mixture(
    Y: np.ndarray,
    offset_log: np.ndarray,
    K: int,
    degree: int = 2,
    n_init: int = 10,
    max_iter: int = 200,
    tol: float = 1e-7,
    random_state: Optional[int] = None,
) -> dict:
    """Fit a K-group polynomial Poisson mixture trajectory model by EM.

    Returns a dict with: loglik, bic, aic, beta (K x degree+1), pi (K,),
    resp (n_units x K posterior probabilities), t_scaled, degree, n_params.
    """
    n_units, n_periods = Y.shape
    t = np.arange(n_periods, dtype=float)
    t_c = t - t.mean()
    scale = t_c.std() if t_c.std() > 0 else 1.0
    t_scaled = t_c / scale
    Xt = _design_matrix(t_scaled, degree)

    rng = np.random.default_rng(random_state)
    const_term = -gammaln(Y + 1).sum(axis=1)

    rate_feat = np.log1p(Y / np.exp(offset_log)[None, :])
    feat_std = rate_feat.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    feat_norm = (rate_feat - rate_feat.mean(axis=0)) / feat_std

    best = None
    for init_i in range(max(n_init, 1)):
        if init_i == 0:
            k_eff = min(K, n_units)
            km = KMeans(n_clusters=k_eff, n_init=8, random_state=int(rng.integers(1_000_000))).fit(feat_norm)
            labels = km.labels_  # values in [0, k_eff), always valid indices into the K-wide resp array
            resp = np.full((n_units, K), 0.02 / max(K - 1, 1))
            for i, lab in enumerate(labels):
                resp[i, :] = 0.02 / max(K - 1, 1)
                resp[i, lab] = 0.98
        else:
            alpha = rng.uniform(0.3, 3.0, size=K)
            resp = rng.dirichlet(alpha, size=n_units)

        pi = resp.mean(axis=0)
        beta = np.zeros((K, degree + 1))
        prev_ll = -np.inf
        converged_ll = -np.inf

        for _ in range(max_iter):
            for k in range(K):
                w = resp[:, k]
                W = w.sum()
                if W < 1e-6:
                    continue
                S = w @ Y
                rate = S / W
                try:
                    res = sm.GLM(rate, Xt, family=sm.families.Poisson(), offset=offset_log).fit(maxiter=100)
                    beta[k] = res.params
                except Exception:
                    pass

            pi = resp.mean(axis=0)
            pi = np.clip(pi, 1e-10, None)
            pi = pi / pi.sum()

            log_lambda = Xt @ beta.T + offset_log[:, None]
            log_lambda = np.clip(log_lambda, -25, 25)
            lam = np.exp(log_lambda)

            term1 = Y @ log_lambda
            term2 = lam.sum(axis=0)[None, :]
            logp_yk = term1 - term2 + const_term[:, None]
            log_joint = logp_yk + np.log(pi)[None, :]
            log_norm = logsumexp(log_joint, axis=1)
            resp = np.exp(log_joint - log_norm[:, None])

            ll = log_norm.sum()
            converged_ll = ll
            if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1e-9):
                break
            prev_ll = ll

        n_params = K * (degree + 1) + (K - 1)
        bic = -2 * converged_ll + n_params * np.log(n_units)
        aic = -2 * converged_ll + 2 * n_params
        candidate = dict(
            loglik=converged_ll, bic=bic, aic=aic, beta=beta.copy(), pi=pi.copy(), resp=resp.copy(),
            n_params=n_params, t_scaled=t_scaled, degree=degree, scale=scale, method="poisson_mixture",
        )
        if best is None or converged_ll > best["loglik"]:
            best = candidate
    return best


# ==========================================================================
# KMeans alternative
# ==========================================================================

def _fit_kmeans_groups(Y: np.ndarray, offset_log: np.ndarray, K: int, n_init: int, random_state: Optional[int]) -> dict:
    n_units = Y.shape[0]
    rate_feat = np.log1p(Y / np.exp(offset_log)[None, :])
    feat_std = rate_feat.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    feat_norm = (rate_feat - rate_feat.mean(axis=0)) / feat_std

    k_eff = min(K, n_units)
    km = KMeans(n_clusters=k_eff, n_init=max(n_init, 1), random_state=random_state).fit(feat_norm)
    labels = km.labels_

    # pseudo-posterior via softmax of negative distance to each centroid
    dists = np.linalg.norm(feat_norm[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    resp = np.exp(-dists - logsumexp(-dists, axis=1, keepdims=True))

    if k_eff > 1 and n_units > k_eff:
        try:
            sil = silhouette_score(feat_norm, labels)
        except Exception:
            sil = np.nan
    else:
        sil = np.nan

    return dict(labels=labels, resp=resp, inertia=km.inertia_, silhouette=sil, k_eff=k_eff, method="kmeans")


# ==========================================================================
# Group relabeling (canonical order: ascending mean level)
# ==========================================================================

def _relabel_by_level(assigned: np.ndarray, mean_level: np.ndarray) -> tuple:
    """Relabel raw cluster ids 0..K-1 to 1..K ordered by ascending mean_level
    so labels are stable/interpretable across runs ('Group 1' = lowest)."""
    order = np.argsort(mean_level)  # raw id -> rank
    remap = {raw_id: rank + 1 for rank, raw_id in enumerate(order)}
    new_assigned = np.array([remap[a] for a in assigned])
    return new_assigned, remap


# ==========================================================================
# Main entry point
# ==========================================================================

def gbtm_trajectories(
    df: pd.DataFrame,
    unit_col: str = "ward",
    date_col: str = "datefrome",
    theme_col: str = "crimetype",
    period_days: int = 28,
    start_date: Optional[Union[str, _dt.date]] = None,
    end_date: Optional[Union[str, _dt.date]] = None,
    drop_partial_period: bool = True,
    include_alltheme: bool = True,
    method: str = "poisson_mixture",
    k_range: Sequence[int] = range(2, 7),
    n_groups: Optional[int] = None,
    poly_degree: int = 2,
    n_init: int = 10,
    random_state: int = 42,
    unit_universe: Optional[list] = None,
    date_format: Optional[str] = None,
    min_total_events: float = 0,
    return_diagnostics: bool = False,
):
    """
    Group-Based Trajectory Modeling (GBTM) for area-based crime counts.
    Pandas-only version -- no Spark/SparkSession involved.

    Bins point-level events into fixed-length time periods, builds a
    (spatial unit x period) count matrix per crime theme (+ 'ALLTHEMES'),
    and assigns every spatial unit to a latent trajectory group.

    Parameters
    ----------
    df : pandas.DataFrame
        Point-level event data, one row per incident.
    unit_col : str, default 'ward'
        Spatial unit of aggregation already present on df (ward, hex id...).
    date_col : str, default 'datefrome'
        Date/timestamp column (or ISO 'YYYY-MM-DD' string) for each event.
    theme_col : str, default 'crimetype'
        Crime type column. A model is fit separately for each distinct
        value, plus a rolled-up 'ALLTHEMES' theme if include_alltheme=True.
    period_days : int, default 28
        Length of each time bin, in days. Use something short (e.g. 28)
        for a shorter dataset (a couple of years) and something long
        (e.g. 365) for a many-year dataset, so you end up with a sensible
        number of time points (roughly 10-40 is a good range to aim for).
    start_date, end_date : str ('YYYY-MM-DD') or date, optional
        Bounds of the analysis window. Defaults to the min/max of date_col
        found in df.
    drop_partial_period : bool, default True
        If the date range isn't an exact multiple of period_days, the
        final, shorter bin is dropped by default (so every period plotted
        represents the same number of days). Set False to keep it -- it is
        still handled correctly (as a statistical exposure offset / rate
        denominator), it just has fewer underlying days than the rest.
    include_alltheme : bool, default True
        Also fit a rolled-up 'ALLTHEMES' theme summing every theme_col value.
    method : {'poisson_mixture', 'kmeans'}, default 'poisson_mixture'
        'poisson_mixture' fits a finite mixture of Poisson trajectories
        (polynomial-in-time, degree=poly_degree) by EM -- the standard
        statistical formulation of GBTM for count data. 'kmeans' is a
        faster, simpler shape-based clustering on log-transformed period
        rates; useful as a quick pass or a sanity check.
    k_range : sequence of int, default range(2, 7)
        Candidate numbers of trajectory groups to try per theme. The best K
        is chosen by BIC (poisson_mixture) or silhouette score (kmeans),
        unless n_groups overrides it. If the selected K keeps landing on
        the edge of this range, widen it and re-run.
    n_groups : int or None, default None
        If set, forces this exact number of groups for every theme instead
        of searching k_range.
    poly_degree : int, default 2
        Degree of the polynomial time trend per group (poisson_mixture
        only). 1 = linear, 2 = quadratic, 3 = cubic. Keep this well below
        the number of periods.
    n_init : int, default 10
        Number of random restarts (EM) / KMeans re-runs, to reduce the
        chance of settling on a poor local optimum. Higher is more
        reliable but slower.
    random_state : int, default 42
        Seed for reproducibility.
    unit_universe : list or None, default None
        By default, the units modeled are those with at least one event of
        ANY theme in the analysis window. Pass a full list of unit codes
        here to force-include units that are genuinely always empty (they
        will simply form/join a chronic-zero group).
    date_format : str or None, default None
        A pandas/strftime-style format string (e.g. '%m/%d/%Y') for date_col
        if it's a string not already in ISO 'yyyy-MM-dd' format. IMPORTANT:
        this is different syntax from the Databricks/PySpark version's
        date_format, which uses Java SimpleDateFormat tokens (e.g.
        'MM/dd/yyyy'). Common translations: 'yyyy-MM-dd' -> '%Y-%m-%d',
        'MM/dd/yyyy' -> '%m/%d/%Y', 'dd/MM/yyyy' -> '%d/%m/%Y'.
    min_total_events : float, default 0
        Drop units whose total event count across the whole window (for a
        given theme) is below this threshold, before fitting -- useful to
        exclude units too sparse to inform a trajectory shape. 0 = keep all.
    return_diagnostics : bool, default False
        If True, also return model-selection (BIC/AIC by K) and fitted
        group-level trajectory curves for plotting.

    Returns
    -------
    dict[str, pandas.DataFrame]
        One entry per theme (including 'ALLTHEMES'). Each DataFrame has one
        row per unit_col, with columns: unit_col, one column per time
        period (named 'p<index>_<period_start_date>', raw counts), 'Group'
        (1 = lowest average level, ascending), and 'group_confidence' (the
        model's posterior probability / pseudo-probability of that unit's
        assigned group -- higher is a more confident assignment).
    If return_diagnostics=True, returns a 3-tuple instead:
        (trajectories, model_selection, fitted_curves)
        - trajectories: as above.
        - model_selection: dict[str, pandas.DataFrame], one row per K
          tried, with loglik/bic/aic (or silhouette) and a 'selected' flag.
        - fitted_curves: dict[str, pandas.DataFrame], the model's smoothed
          group-level trajectory (one row per Group x period), for
          overlaying a mean trend line on a spaghetti plot of the raw unit
          trajectories.

    Examples
    --------
    >>> traj = gbtm_trajectories(
    ...     df, unit_col="WardCode", date_col="DateCommittedFrom",
    ...     theme_col="CrimeType", period_days=28,
    ... )
    >>> traj["ROBBERY"].head()
    >>> traj["ALLTHEMES"].head()

    >>> traj, model_sel, curves = gbtm_trajectories(
    ...     df, unit_col="HexagonCode", date_col="DateCommittedFrom",
    ...     period_days=365, k_range=range(2, 5), return_diagnostics=True,
    ... )
    >>> model_sel["ALLTHEMES"]
    """
    if method not in ("poisson_mixture", "kmeans"):
        raise ValueError("method must be 'poisson_mixture' or 'kmeans'")

    check_cols = [unit_col, theme_col, date_col]
    _validate_columns(df, check_cols)

    if start_date is None or end_date is None:
        mn, mx = _infer_date_bounds(df, date_col, date_format)
        start_date = _to_date(start_date) if start_date is not None else mn
        end_date = _to_date(end_date) if end_date is not None else mx
    else:
        start_date = _to_date(start_date)
        end_date = _to_date(end_date)
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    n_periods, period_starts, period_ends, period_len_days = _build_period_bins(
        start_date, end_date, period_days, drop_partial_period
    )
    offset_log = np.log(period_len_days)

    long_pdf = _period_counts_pdf(
        df, unit_col=unit_col, theme_col=theme_col, date_col=date_col, date_format=date_format,
        start_date=start_date, period_days=period_days, n_periods=n_periods, include_alltheme=include_alltheme,
    )
    if long_pdf.empty:
        raise ValueError(
            "No events found in the analysis window -- check date_col, start_date/end_date, "
            "and that unit_col/theme_col are populated."
        )

    all_units = set(long_pdf["unit"].unique().tolist())
    if unit_universe:
        all_units |= set(unit_universe)
    all_units = sorted(all_units, key=lambda v: (str(type(v)), v))

    all_themes = sorted(t for t in long_pdf["theme"].unique().tolist() if t != ALLTHEMES_LABEL)
    if include_alltheme:
        all_themes = all_themes + [ALLTHEMES_LABEL]

    period_col_names = [f"p{idx:02d}_{s.isoformat()}" for idx, s in enumerate(period_starts)]

    trajectories, model_selection, fitted_curves = {}, {}, {}

    for theme in all_themes:
        theme_pdf = long_pdf[long_pdf["theme"] == theme]
        Y_full = _build_matrix(theme_pdf, all_units, n_periods)
        units_arr = np.array(all_units, dtype=object)

        keep_mask = Y_full.sum(axis=1) >= min_total_events
        Y = Y_full[keep_mask]
        units_kept = units_arr[keep_mask]

        if Y.shape[0] < 2:
            raise ValueError(
                f"Theme {theme!r} has fewer than 2 units meeting min_total_events={min_total_events}; "
                "cannot fit a trajectory model. Lower min_total_events or check the data."
            )

        sel_rows = []
        if method == "poisson_mixture":
            k_candidates = [n_groups] if n_groups else list(k_range)
            fits = {}
            for K in k_candidates:
                K_eff = min(K, Y.shape[0])
                fits[K] = _fit_poisson_mixture(
                    Y, offset_log, K_eff, degree=poly_degree, n_init=n_init, random_state=random_state
                )
                sel_rows.append({"K": K, "loglik": fits[K]["loglik"], "bic": fits[K]["bic"], "aic": fits[K]["aic"]})
            best_K = n_groups if n_groups else min(fits, key=lambda k: fits[k]["bic"])
            best = fits[best_K]
            assigned_raw = best["resp"].argmax(axis=1)
            confidence = best["resp"][np.arange(len(assigned_raw)), assigned_raw]

            t_scaled = best["t_scaled"]
            Xt = _design_matrix(t_scaled, best["degree"])
            fitted_log_lambda = Xt @ best["beta"].T + offset_log[:, None]
            fitted_lambda = np.exp(np.clip(fitted_log_lambda, -25, 25))  # (n_periods, K_eff)
            mean_level = fitted_lambda.mean(axis=0)

        else:  # kmeans
            k_candidates = [n_groups] if n_groups else list(k_range)
            fits = {}
            for K in k_candidates:
                K_eff = min(K, Y.shape[0])
                fits[K] = _fit_kmeans_groups(Y, offset_log, K_eff, n_init=n_init, random_state=random_state)
                sel_rows.append({"K": K, "inertia": fits[K]["inertia"], "silhouette": fits[K]["silhouette"]})
            valid = {k: v for k, v in fits.items() if not np.isnan(v["silhouette"])}
            best_K = n_groups if n_groups else (max(valid, key=lambda k: valid[k]["silhouette"]) if valid else k_candidates[0])
            best = fits[best_K]
            assigned_raw = best["labels"]
            confidence = best["resp"][np.arange(len(assigned_raw)), assigned_raw]

            rate = Y / np.exp(offset_log)[None, :]
            mean_level = np.array([
                rate[assigned_raw == k].mean() if (assigned_raw == k).sum() > 0 else 0.0
                for k in range(best["k_eff"])
            ])

        group_labels, _remap = _relabel_by_level(assigned_raw, mean_level)

        sel_df = pd.DataFrame(sel_rows)
        sel_df.insert(0, "theme", theme)
        sel_df["selected"] = sel_df["K"] == best_K
        model_selection[theme] = sel_df

        if method == "poisson_mixture":
            curve_rows = []
            for raw_id, rank1 in _remap.items():
                for p_idx in range(n_periods):
                    curve_rows.append({
                        "theme": theme, "Group": rank1, "period_index": p_idx,
                        "period_start": period_starts[p_idx].isoformat(),
                        "period_end": period_ends[p_idx].isoformat(),
                        "fitted_count": float(fitted_lambda[p_idx, raw_id]),
                    })
            fitted_curves[theme] = pd.DataFrame(curve_rows)
        else:
            rate = Y / np.exp(offset_log)[None, :]
            curve_rows = []
            for raw_id, rank1 in _remap.items():
                mask = assigned_raw == raw_id
                mean_rate_by_period = rate[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(n_periods)
                for p_idx in range(n_periods):
                    curve_rows.append({
                        "theme": theme, "Group": rank1, "period_index": p_idx,
                        "period_start": period_starts[p_idx].isoformat(),
                        "period_end": period_ends[p_idx].isoformat(),
                        "fitted_count": float(mean_rate_by_period[p_idx] * period_len_days[p_idx]),
                    })
            fitted_curves[theme] = pd.DataFrame(curve_rows)

        wide = pd.DataFrame(Y, columns=period_col_names)
        wide.insert(0, unit_col, units_kept)
        wide["Group"] = group_labels
        wide["group_confidence"] = confidence
        trajectories[theme] = wide

    if return_diagnostics:
        return trajectories, model_selection, fitted_curves
    return trajectories
