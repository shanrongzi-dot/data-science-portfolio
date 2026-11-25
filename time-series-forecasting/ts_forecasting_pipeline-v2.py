"""Time-Series Forecasting Pipeline for Money Supply, Output, Interest Rates, and Transactions.

Converts and refines `main202309.R` as a modular time-series forecasting workflow:
  - Module 1: Series Preprocessing, Seasonal Adjustment & Missing-Value Imputation
  - Module 2: Time-Series Alignment & Derived Feature Construction
  - Module 3: Dynamic Lag-Lead Regression (DOLS) with BIC Order Selection & AIC Refinement
  - Module 4: Multi-Horizon Out-of-Sample Time-Series Forecasting & Backtesting
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.seasonal import STL


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

QUARTER_TO_MONTH: Dict[int, int] = {1: 3, 2: 6, 3: 9, 4: 12}


# =============================================================================
# MODULE 1: SERIES PREPROCESSING, SEASONAL ADJUSTMENT & IMPUTATION
# Purpose: Clean and prepare raw time-series inputs; extract seasonal-adjusted
#          trend components via STL; and impute missing observations using a
#          local-level Kalman smoother.
# =============================================================================

def parse_quarter_code(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse quarter strings (e.g., '2000Q1' or '2000-1') into year and quarter-end month."""
    s = series.astype(str).str.strip()
    years = pd.to_numeric(s.str.slice(0, 4), errors="coerce").astype("Int64")
    quarters = pd.to_numeric(s.str.slice(5, 6), errors="coerce").astype("Int64")
    months = quarters.map(QUARTER_TO_MONTH).astype("Int64")
    return years, months


def deseasonalize_stl(
    values: Union[pd.Series, np.ndarray],
    period: int = 4,
    seasonal: int = 13,
    robust: bool = True,
) -> np.ndarray:
    """Deseasonalize a quarterly time series via STL and return the trend component."""
    arr = np.asarray(values, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < period * 2:
        return arr.copy()

    filled = pd.Series(arr).interpolate(method="linear", limit_direction="both").to_numpy()
    res = STL(filled, period=period, seasonal=seasonal, robust=robust).fit()
    trend = np.asarray(res.trend, dtype=float)
    trend[~valid] = np.nan
    return trend


def kalman_impute_1d(values: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Impute missing values in a 1D time series using a local-level Kalman smoother."""
    arr = np.asarray(values, dtype=float).copy()
    if not np.isnan(arr).any():
        return arr
    if np.isnan(arr).all():
        return np.zeros_like(arr)

    mod = sm.tsa.UnobservedComponents(arr, level="local level")
    fit = mod.fit(disp=False)
    smoothed = np.asarray(fit.smoothed_state[0], dtype=float)
    out = arr.copy()
    out[np.isnan(out)] = smoothed[np.isnan(out)]
    return out


def prepare_m1_series(path: Union[str, Path]) -> pd.DataFrame:
    """Prepare M012.xls money series, deseasonalize M1, and filter to quarter-end months."""
    df = pd.read_excel(path)
    df["M00"] = pd.to_numeric(df["M0"], errors="coerce")
    df["M0"] = pd.to_numeric(df["M0"], errors="coerce")
    df["M1_raw"] = pd.to_numeric(df["M1"], errors="coerce")
    df["M1"] = deseasonalize_stl(df["M1_raw"], period=4)

    dt = pd.to_datetime(df["Period"], errors="coerce")
    df["year"] = dt.dt.year.astype("Int64")
    df["month"] = dt.dt.month.astype("Int64")
    df["day"] = dt.dt.day.astype("Int64")

    return df[df["month"].isin([3, 6, 9, 12])].copy()


def prepare_gdp_series(path: Union[str, Path], sheet_name: str = "NGDP") -> pd.DataFrame:
    """Prepare quarterly NGDP series, apply STL trend extraction, and map quarter-end months."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df["NGDP0"] = pd.to_numeric(df["NGDP_XDC"], errors="coerce")
    df["NGDP_XDC"] = deseasonalize_stl(df["NGDP0"], period=4)
    df["year"], df["month"] = parse_quarter_code(df["BaseYear"])
    return df


def prepare_interest_rate_series(path: Union[str, Path]) -> pd.DataFrame:
    """Prepare interest rate time series, parse timestamps, and keep quarter-end points."""
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["period"], format="%m/%d/%y", errors="coerce")
    df["year"] = df["date"].dt.year.astype("Int64")
    df["month"] = df["date"].dt.month.astype("Int64")
    df["day"] = df["date"].dt.day.astype("Int64")
    return df[df["month"].isin([3, 6, 9, 12])].copy()


def prepare_retail_series(path: Union[str, Path], sheet_name: str = "Sheet2") -> pd.DataFrame:
    """Convert cumulative within-year monthly retail series into deseasonalized quarterly values."""
    df = pd.read_excel(path, sheet_name=sheet_name)

    fallback_1999 = [f"1999年{m}月" for m in range(11, 0, -1)]
    for offset, label in enumerate(fallback_1999):
        idx = 253 + offset
        if idx < len(df):
            df.loc[idx, "period"] = label

    period_str = df["period"].astype(str).str.strip()
    extracted = period_str.str.extract(r"^(\d{4})\D+(\d{1,2})")
    df["year"] = pd.to_numeric(extracted[0], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(extracted[1], errors="coerce").astype("Int64")
    df["Date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    df = df.sort_values("Date").reset_index(drop=True)

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["cumsum"] = df.groupby("year", dropna=True)["amount"].cumsum()

    q_df = df[df["month"].isin([3, 6, 9, 12])].copy()
    q_df["cumulative2"] = q_df["cumsum"].shift(1)
    q_df["cumulative3"] = q_df["cumsum"] - q_df["cumulative2"]
    q_df.loc[q_df["month"] == 3, "cumulative3"] = q_df.loc[q_df["month"] == 3, "cumsum"]
    q_df = q_df[["year", "month", "Date", "cumulative3"]].dropna().reset_index(drop=True)
    q_df["cumulative3"] = deseasonalize_stl(q_df["cumulative3"], period=4)
    return q_df


def prepare_cpi_series(
    path: Union[str, Path], sheet_name: str = "cpi_china"
) -> pd.DataFrame:
    """Prepare quarterly price index series and compute period-over-period percentage change."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    df["inf"] = np.nan
    df.loc[1:, "inf"] = (np.diff(df["P"].to_numpy()) / df["P"].to_numpy()[:-1]) * 100.0
    df["year"], df["month"] = parse_quarter_code(df["Base Year"])
    return df


def prepare_transactions_series(path: Union[str, Path]) -> pd.DataFrame:
    """Prepare transaction series, parse period keys, and Kalman-impute missing values."""
    df = pd.read_csv(path)
    df["year"], df["month"] = parse_quarter_code(df["Year"])
    for col in ["POS(wan)", "card(yi)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = kalman_impute_1d(df[col])
    return df


# =============================================================================
# MODULE 2: TIME-SERIES ALIGNMENT & DERIVED FEATURE CONSTRUCTION
# Purpose: Align quarterly time series on (year, month) and compute ratios,
#          deflated series, log transforms, and first-difference growth rates.
# =============================================================================

def build_time_series_table(
    base_dir: Union[str, Path] = "/Users/rose/Desktop/money",
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Align prepared quarterly time series and build ratios, logs, and growth features."""
    base = Path(base_dir)
    m1_q = prepare_m1_series(base / "M012.xls")
    ir_q = prepare_interest_rate_series(base / "interestrate.xls")
    gdp_q = prepare_gdp_series(base / "GDP.xlsx", sheet_name="NGDP")
    ret_q = prepare_retail_series(base / "retail.xlsx", sheet_name="Sheet2")
    cpi_q = prepare_cpi_series(base / "cpi_quarterly.xlsx", sheet_name="cpi_china")
    upd_q = prepare_transactions_series(base / "update.csv")

    keys = ["year", "month"]
    merged = (
        m1_q.merge(ir_q, on=keys, how="inner")
        .merge(gdp_q, on=keys, how="inner")
        .merge(ret_q, on=keys, how="inner")
        .merge(cpi_q, on=keys, how="inner")
        .merge(upd_q, on=keys, how="inner")
    )

    df = pd.DataFrame({
        "Date": merged["Date"],
        "M0": pd.to_numeric(merged["M0"], errors="coerce"),
        "M1": pd.to_numeric(merged["M1"], errors="coerce"),
        "M00": pd.to_numeric(merged["M00"], errors="coerce"),
        "NGDP0": pd.to_numeric(merged["NGDP0"], errors="coerce"),
        "ngdp": pd.to_numeric(merged["NGDP_XDC"], errors="coerce"),
        "cpi": pd.to_numeric(merged["P"], errors="coerce"),
        "inf": pd.to_numeric(merged["inf"], errors="coerce"),
        "ir": pd.to_numeric(merged["sr"], errors="coerce"),
        "trans": pd.to_numeric(merged["trans"], errors="coerce"),
        "POS": pd.to_numeric(merged["POS(wan)"], errors="coerce"),
        "card": pd.to_numeric(merged["card(yi)"], errors="coerce"),
        "retail": pd.to_numeric(merged["cumulative3"], errors="coerce"),
    })

    df["pt"] = df["trans"] / df["retail"].replace(0, np.nan)
    df["lnpt"] = np.log(df["pt"].where(df["pt"] > 0))

    df["POS"] = df["POS"] / 100.0
    df["card"] = df["card"] / 10.0

    mg_init = df["M00"] / df["NGDP0"].replace(0, np.nan)
    df["MG"] = deseasonalize_stl(mg_init, period=4)

    df["TT"] = df["trans"] + df["M0"]
    df["lnTT"] = np.log(df["TT"].where(df["TT"] > 0))
    df["gTT"] = np.nan
    df.loc[1:, "gTT"] = np.diff(df["TT"].to_numpy()) / df["TT"].to_numpy()[:-1]

    df["PG"] = df["trans"] / df["ngdp"].replace(0, np.nan)
    df["TM"] = df["trans"] / df["M0"].replace(0, np.nan)
    df["MG"] = df["M0"] / df["ngdp"].replace(0, np.nan)
    df["TG"] = df["PG"] + df["MG"]

    for col in ["PG", "MG", "TG"]:
        g_col = f"{col}_growth"
        df[g_col] = np.nan
        arr = df[col].to_numpy(dtype=float)
        df.loc[1:, g_col] = np.diff(arr) / arr[:-1]

    df["rgdp"] = df["ngdp"] / df["cpi"].replace(0, np.nan)
    df["realm"] = df["M1"] / df["cpi"].replace(0, np.nan)
    df["rm0"] = df["M0"] / df["cpi"].replace(0, np.nan)
    df["lnm"] = np.log(df["realm"].where(df["realm"] > 0))

    df["lnrgdp"] = np.log(df["rgdp"].where(df["rgdp"] > 0))
    df["dlnrgdp"] = np.nan
    df.loc[1:, "dlnrgdp"] = np.diff(df["lnrgdp"].to_numpy())

    df["transs"] = df["trans"] / df["cpi"].replace(0, np.nan)
    df["M0s"] = df["M0"] / df["cpi"].replace(0, np.nan)
    df["lnM0"] = np.log(df["M0s"].where(df["M0s"] > 0))

    df["dlnpt"] = np.nan
    df.loc[1:, "dlnpt"] = np.diff(df["lnpt"].to_numpy())

    df["dlnM0"] = np.nan
    df.loc[1:, "dlnM0"] = np.diff(df["lnM0"].to_numpy())

    df["ir100"] = df["ir"]
    df["ir"] = df["ir"] / 100.0
    df["lnr"] = np.log(df["ir"].where(df["ir"] > 0))

    df["myratio"] = df["M1"] / df["ngdp"].replace(0, np.nan)
    df["m0ratio"] = df["M0"] / df["ngdp"].replace(0, np.nan)
    df["tm0"] = df["trans"] / df["M0"].replace(0, np.nan)
    df["lnmy"] = np.log(df["myratio"].where(df["myratio"] > 0))

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved aligned time-series table to %s", out_path)

    return df


# =============================================================================
# MODULE 3: DYNAMIC LAG-LEAD REGRESSION (DOLS) WITH BIC & AIC SELECTION
# Purpose: Estimate dynamic leads-and-lags specifications on integrated time
#          series, select lag/lead order k by BIC, and prune nuisance terms via AIC.
# =============================================================================

def compute_bic(rss: float, n_obs: int, n_params: int) -> float:
    """Compute BIC criterion matching the R helper `bic(model)`."""
    if n_obs <= 0 or rss <= 0:
        return np.inf
    return float(np.log(rss / n_obs) + n_params * np.log(n_obs) / n_obs)


def build_dols_design_matrix(
    df: pd.DataFrame,
    dependent: str,
    regressors: Sequence[str],
    leads_lags_k: int,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build target series and design matrix with leads and lags [-k, +k] of differences."""
    cols: Dict[str, pd.Series] = {"const": pd.Series(1.0, index=df.index)}
    for reg in regressors:
        cols[reg] = pd.to_numeric(df[reg], errors="coerce")

    for reg in regressors:
        d_reg = pd.to_numeric(df[reg], errors="coerce").diff()
        for shift in range(-leads_lags_k, leads_lags_k + 1):
            label = (
                f"d_{reg}_lead{-shift}"
                if shift < 0
                else (f"d_{reg}_lag{shift}" if shift > 0 else f"d_{reg}_lag0")
            )
            cols[label] = d_reg.shift(-shift)

    design = pd.DataFrame(cols, index=df.index)
    y = pd.to_numeric(df[dependent], errors="coerce")

    combined = pd.concat([y.rename("__y__"), design], axis=1).dropna()
    return combined["__y__"], combined.drop(columns=["__y__"])


def fit_dols(
    df: pd.DataFrame,
    dependent: str,
    regressors: Sequence[str],
    leads_lags_k: int,
):
    """Fit a DOLS regression via OLS and return fitted model with BIC and matrices."""
    y_clean, x_clean = build_dols_design_matrix(df, dependent, regressors, leads_lags_k)
    model = sm.OLS(y_clean, x_clean).fit()
    n_obs = int(model.nobs)
    n_params = int(x_clean.shape[1])
    rss = float(np.sum(np.square(model.resid)))
    bic_val = compute_bic(rss=rss, n_obs=n_obs, n_params=n_params)
    return {
        "fit": model,
        "bic": bic_val,
        "k": leads_lags_k,
        "n_obs": n_obs,
        "n_params": n_params,
        "y": y_clean,
        "X": x_clean,
    }


def select_dols_by_bic(
    df: pd.DataFrame,
    dependent: str,
    regressors: Sequence[str],
    max_k: Optional[int] = None,
) -> Dict[str, object]:
    """Select optimal lead/lag order k in 1..max_k by minimizing BIC."""
    k_cap = max_k if max_k is not None else max(1, int(np.floor(len(df) ** (1.0 / 3.0))))
    candidates: List[Dict[str, object]] = []
    for k_val in range(1, k_cap + 1):
        res = fit_dols(df, dependent=dependent, regressors=regressors, leads_lags_k=k_val)
        candidates.append(res)

    best = min(candidates, key=lambda item: item["bic"])
    return {
        "best": best,
        "bics": [c["bic"] for c in candidates],
        "candidates": candidates,
    }


def stepwise_aic_refine(y: pd.Series, x: pd.DataFrame, keep_cols: Optional[Sequence[str]] = None):
    """Backward-stepwise AIC refinement while keeping protected level regressors fixed."""
    protected = set(keep_cols or [])
    current_cols = list(x.columns)
    current_fit = sm.OLS(y, x[current_cols]).fit()
    current_aic = float(current_fit.aic)

    improved = True
    while improved and len(current_cols) > 1:
        improved = False
        best_candidate_aic = current_aic
        drop_target: Optional[str] = None

        for col in current_cols:
            if col in protected:
                continue
            trial_cols = [c for c in current_cols if c != col]
            if not trial_cols:
                continue
            trial_fit = sm.OLS(y, x[trial_cols]).fit()
            if float(trial_fit.aic) < best_candidate_aic - 1e-9:
                best_candidate_aic = float(trial_fit.aic)
                drop_target = col

        if drop_target is not None:
            current_cols.remove(drop_target)
            current_fit = sm.OLS(y, x[current_cols]).fit()
            current_aic = best_candidate_aic
            improved = True

    return current_fit


def run_dols_specifications(df: pd.DataFrame) -> Dict[str, object]:
    """Run baseline, log-log, and semi-log DOLS specifications with BIC and AIC steps."""
    k_max = max(1, int(np.floor(len(df) ** (1.0 / 3.0))))

    dols5_sel = select_dols_by_bic(df, dependent="lnm", regressors=["lnrgdp", "lnr"], max_k=k_max)
    dols1_sel = select_dols_by_bic(df, dependent="lnm", regressors=["lnrgdp", "lnr", "lnpt"], max_k=k_max)
    dols2_sel = select_dols_by_bic(df, dependent="lnm", regressors=["lnrgdp", "ir", "lnpt"], max_k=k_max)

    dols1_aic = stepwise_aic_refine(
        dols1_sel["best"]["y"],
        dols1_sel["best"]["X"],
        keep_cols=["const", "lnrgdp", "lnr", "lnpt"],
    )
    dols2_aic = stepwise_aic_refine(
        dols2_sel["best"]["y"],
        dols2_sel["best"]["X"],
        keep_cols=["const", "lnrgdp", "ir", "lnpt"],
    )

    return {
        "k_max": k_max,
        "DOLS5_bic": dols5_sel,
        "DOLS1_loglog": dols1_sel,
        "DOLS2_semilog": dols2_sel,
        "DOLS1_aic_fit": dols1_aic,
        "DOLS2_aic_fit": dols2_aic,
    }


# =============================================================================
# MODULE 4: MULTI-HORIZON OUT-OF-SAMPLE TIME-SERIES FORECASTING (EXTENSION)
# Purpose: Generate multi-step time-series forecasts with damped Holt trend
#          and ARIMA models, combine into an ensemble, and evaluate rolling RMSE.
# =============================================================================

def forecast_series_models(
    series: Union[pd.Series, np.ndarray],
    horizon: int = 8,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    damped_trend: bool = True,
) -> Dict[str, np.ndarray]:
    """Forecast a time series h steps ahead using Holt trend smoothing and ARIMA."""
    s = pd.Series(np.asarray(series, dtype=float)).dropna().reset_index(drop=True)
    if len(s) < 6:
        raise ValueError("Need at least 6 non-missing observations to fit forecasting models.")

    holt_fit = Holt(s, damped_trend=damped_trend, initialization_method="estimated").fit()
    holt_fc = np.asarray(holt_fit.forecast(horizon), dtype=float)

    arima_fit = sm.tsa.ARIMA(s, order=arima_order).fit()
    arima_fc = np.asarray(arima_fit.forecast(steps=horizon), dtype=float)

    return {
        "holt_forecast": holt_fc,
        "arima_forecast": arima_fc,
        "ensemble_forecast": 0.5 * (holt_fc + arima_fc),
    }


def evaluate_rolling_forecast_rmse(
    series: Union[pd.Series, np.ndarray],
    initial_window: int = 40,
    horizon: int = 4,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
) -> Dict[str, float]:
    """Compute expanding-window out-of-sample RMSE for Holt, ARIMA, and 50/50 ensemble."""
    s = pd.Series(np.asarray(series, dtype=float)).dropna().reset_index(drop=True)
    n = len(s)
    if n <= initial_window + horizon:
        return {"holt_rmse": np.nan, "arima_rmse": np.nan, "ensemble_rmse": np.nan}

    errs_holt: List[float] = []
    errs_arima: List[float] = []
    errs_ens: List[float] = []

    for end_idx in range(initial_window, n - horizon + 1):
        train = s.iloc[:end_idx]
        actual = s.iloc[end_idx : end_idx + horizon].to_numpy(dtype=float)
        fc = forecast_series_models(train, horizon=horizon, arima_order=arima_order)
        errs_holt.extend(actual - fc["holt_forecast"])
        errs_arima.extend(actual - fc["arima_forecast"])
        errs_ens.extend(actual - fc["ensemble_forecast"])

    return {
        "holt_rmse": float(np.sqrt(np.mean(np.square(errs_holt)))),
        "arima_rmse": float(np.sqrt(np.mean(np.square(errs_arima)))),
        "ensemble_rmse": float(np.sqrt(np.mean(np.square(errs_ens)))),
    }


def run_forecasting_suite(
    df: pd.DataFrame,
    target_cols: Sequence[str] = ("lnm", "lnrgdp", "inf"),
    horizon: int = 8,
) -> Dict[str, Dict[str, object]]:
    """Run out-of-sample forecasts and rolling RMSE evaluation on selected series."""
    results: Dict[str, Dict[str, object]] = {}
    for col in target_cols:
        if col not in df.columns:
            logger.warning("Forecast target '%s' not found in DataFrame; skipping.", col)
            continue
        fc = forecast_series_models(df[col], horizon=horizon)
        metrics = evaluate_rolling_forecast_rmse(df[col], horizon=min(4, max(1, horizon // 2)))
        results[col] = {
            "forecasts": fc,
            "rolling_metrics": metrics,
        }
    return results


def make_synthetic_series_sample(n_quarters: int = 88, seed: int = 2023) -> pd.DataFrame:
    """Create a synthetic quarterly time-series table for smoke-testing without raw files."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1999-03-31", periods=n_quarters, freq="QE")
    t = np.arange(n_quarters, dtype=float)
    season = 0.04 * np.sin(2 * np.pi * t / 4.0)

    lnrgdp = 10.0 + 0.02 * t + 0.05 * np.sin(t / 8.0) + rng.normal(0, 0.01, n_quarters)
    ir_pct = np.clip(3.5 - 0.01 * t + 0.4 * np.cos(t / 10.0) + rng.normal(0, 0.15, n_quarters), 0.5, 10.0)
    lnpt = -1.5 + 0.015 * t + rng.normal(0, 0.05, n_quarters)
    lnm = 1.2 * lnrgdp - 0.08 * (ir_pct / 100.0) + 0.15 * lnpt + season + rng.normal(0, 0.02, n_quarters)

    cpi = 100.0 * np.exp(0.006 * t + rng.normal(0, 0.004, n_quarters))
    rgdp = np.exp(lnrgdp)
    ngdp = rgdp * cpi
    realm = np.exp(lnm)
    m1 = realm * cpi
    m0 = 0.25 * m1
    retail = np.exp(8.5 + 0.025 * t + season + rng.normal(0, 0.02, n_quarters))
    pt = np.exp(lnpt)
    trans = pt * retail
    ir = ir_pct / 100.0

    inf = np.empty(n_quarters)
    inf[0] = np.nan
    inf[1:] = (np.diff(cpi) / cpi[:-1]) * 100.0

    return pd.DataFrame({
        "Date": dates,
        "M0": m0,
        "M1": m1,
        "M00": m0,
        "NGDP0": ngdp,
        "ngdp": ngdp,
        "cpi": cpi,
        "inf": inf,
        "ir": ir,
        "ir100": ir_pct,
        "trans": trans,
        "POS": rng.uniform(1.0, 20.0, n_quarters),
        "card": rng.uniform(5.0, 50.0, n_quarters),
        "retail": retail,
        "pt": pt,
        "lnpt": lnpt,
        "MG": m0 / ngdp,
        "TT": trans + m0,
        "lnTT": np.log(trans + m0),
        "gTT": np.concatenate([[np.nan], np.diff(trans + m0) / (trans + m0)[:-1]]),
        "PG": trans / ngdp,
        "TM": trans / m0,
        "TG": (trans + m0) / ngdp,
        "PG_growth": np.concatenate([[np.nan], np.diff(trans / ngdp) / (trans / ngdp)[:-1]]),
        "MG_growth": np.concatenate([[np.nan], np.diff(m0 / ngdp) / (m0 / ngdp)[:-1]]),
        "TG_growth": np.concatenate([[np.nan], np.diff((trans + m0) / ngdp) / ((trans + m0) / ngdp)[:-1]]),
        "rgdp": rgdp,
        "realm": realm,
        "rm0": m0 / cpi,
        "lnm": lnm,
        "lnrgdp": lnrgdp,
        "dlnrgdp": np.concatenate([[np.nan], np.diff(lnrgdp)]),
        "transs": trans / cpi,
        "M0s": m0 / cpi,
        "lnM0": np.log(m0 / cpi),
        "dlnpt": np.concatenate([[np.nan], np.diff(lnpt)]),
        "dlnM0": np.concatenate([[np.nan], np.diff(np.log(m0 / cpi))]),
        "lnr": np.log(ir),
        "myratio": m1 / ngdp,
        "m0ratio": m0 / ngdp,
        "tm0": trans / m0,
        "lnmy": np.log(m1 / ngdp),
    })


if __name__ == "__main__":
    sample_df = make_synthetic_series_sample()
    dols_out = run_dols_specifications(sample_df)
    print("Selected k (log-log):", dols_out["DOLS1_loglog"]["best"]["k"])
    print(dols_out["DOLS1_aic_fit"].summary().tables[1])

    fc_out = run_forecasting_suite(sample_df, target_cols=("lnm", "lnrgdp", "inf"), horizon=8)
    print("
--- 8-Step Forecast for lnm (Ensemble) ---")
    print(np.round(fc_out["lnm"]["forecasts"]["ensemble_forecast"], 4))
    print("--- Rolling Forecast RMSE for lnm ---")
    print(fc_out["lnm"]["rolling_metrics"])
