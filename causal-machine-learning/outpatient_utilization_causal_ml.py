"""Outpatient Utilization Causal ML Python Pipeline.

Combines and refines CHARLS longitudinal demographic/health data cleaning
(2011, 2013, 2015 waves) and causal machine learning estimation (BLP, GATEs,
CATE correlation/diagnostics, and covariate group means with clustered standard
errors) for estimating the effect of health insurance (e.g., NRPS) on
outpatient utilization.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import base, ensemble, linear_model, neural_network
from sklearn.exceptions import ConvergenceWarning

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. CHARLS DEMOGRAPHIC & LONGITUDINAL DATA CLEANING MODULE
# =============================================================================

def normalize_charls_ids_2011(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize 2011 CHARLS householdID and ID strings."""
    out = df.copy()
    out["householdID"] = out["householdID"].astype(str).str.strip() + "0"
    out["ID"] = out["householdID"] + out["ID"].astype(str).str.strip().str.slice(9, 11)
    return out


def calculate_age_on_date(birthdates: pd.Series, ref_date: str) -> pd.Series:
    """Compute age in completed years as of a reference date string (YYYY-MM-DD)."""
    ref_ts = pd.Timestamp(ref_date)
    bd_ts = pd.to_datetime(birthdates, errors="coerce")
    years = ref_ts.year - bd_ts.dt.year
    before_bday = (
        (ref_ts.month < bd_ts.dt.month)
        | ((ref_ts.month == bd_ts.dt.month) & (ref_ts.day < bd_ts.dt.day))
    )
    age = years - before_bday.astype(int)
    age = age.where(bd_ts.notna(), np.nan)
    return age


def build_birthdate_from_parts(
    df: pd.DataFrame, year_col: str, month_col: str, day_col: str
) -> pd.Series:
    """Assemble datetime series from year, month, and day columns."""
    date_str = (
        df[year_col].astype("Int64").astype(str)
        + "-"
        + df[month_col].astype("Int64").astype(str).str.zfill(2)
        + "-"
        + df[day_col].astype("Int64").astype(str).str.zfill(2)
    )
    return pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")


def clean_charls_2011(path: Union[str, Path], min_age: int = 45) -> pd.DataFrame:
    """Clean CHARLS 2011 demographic background data for respondents aged >= min_age."""
    cols = [
        "ID", "householdID", "communityID",
        "rgender", "ba002_1", "ba002_2", "ba002_3", "bd001", "be001",
    ]
    raw = pd.read_stata(path, columns=cols)
    df = normalize_charls_ids_2011(raw)
    df["birthdate"] = build_birthdate_from_parts(df, "ba002_1", "ba002_2", "ba002_3")
    df = df.dropna(subset=["ID", "householdID", "communityID", "rgender", "birthdate", "bd001", "be001"]).copy()
    df["age"] = calculate_age_on_date(df["birthdate"], "2011-12-31")

    cleaned = pd.DataFrame({
        "ID": df["ID"],
        "householdID": df["householdID"],
        "communityID": df["communityID"],
        "gender": df["rgender"],
        "age": df["age"],
        "birthdate": df["birthdate"],
        "education": df["bd001"],
        "martial": (df["be001"] == 1).astype(float),
    })
    return cleaned[cleaned["age"] >= min_age].reset_index(drop=True)


def clean_charls_2013(
    path: Union[str, Path],
    prev_wave: Optional[pd.DataFrame] = None,
    min_age: int = 45,
) -> pd.DataFrame:
    """Clean CHARLS 2013 demographic data, coalescing missing education/birthdate with prior wave."""
    cols = [
        "ID", "householdID", "communityID",
        "ba000_w2_3", "ba002_1", "ba002_2", "ba002_3", "bd001", "be001",
    ]
    raw = pd.read_stata(path, columns=cols)
    raw["date"] = build_birthdate_from_parts(raw, "ba002_1", "ba002_2", "ba002_3")

    if prev_wave is not None and not prev_wave.empty:
        prior_subset = prev_wave[["ID", "householdID", "communityID", "birthdate", "education"]].rename(
            columns={"birthdate": "prev_birthdate", "education": "prev_education"}
        )
        merged = raw.merge(prior_subset, on=["ID", "householdID", "communityID"], how="left")
        merged["education"] = merged["bd001"].combine_first(merged["prev_education"])
        merged["birthdate"] = merged["date"].combine_first(merged["prev_birthdate"])
    else:
        merged = raw.rename(columns={"bd001": "education", "date": "birthdate"})

    merged = merged.dropna(
        subset=["ID", "householdID", "communityID", "ba000_w2_3", "birthdate", "education", "be001"]
    ).copy()
    merged["age"] = calculate_age_on_date(merged["birthdate"], "2013-12-31")

    cleaned = pd.DataFrame({
        "ID": merged["ID"],
        "householdID": merged["householdID"],
        "communityID": merged["communityID"],
        "gender": merged["ba000_w2_3"],
        "age": merged["age"],
        "birthdate": merged["birthdate"],
        "education": merged["education"],
        "martial": (merged["be001"] == 1).astype(float),
    })
    return cleaned[cleaned["age"] >= min_age].reset_index(drop=True)


def clean_charls_2015(
    path: Union[str, Path],
    prev_wave: Optional[pd.DataFrame] = None,
    min_age: int = 45,
) -> pd.DataFrame:
    """Clean CHARLS 2015 demographic data, coalescing education/birthdate with prior wave."""
    cols = [
        "ID", "householdID", "communityID",
        "ba000_w2_3", "ba004_w3_1", "ba004_w3_2", "ba004_w3_3", "bd001_w2_4", "be001",
    ]
    raw = pd.read_stata(path, columns=cols)
    raw["date"] = build_birthdate_from_parts(raw, "ba004_w3_1", "ba004_w3_2", "ba004_w3_3")
    raw["bd001_w2_4"] = raw["bd001_w2_4"].replace(12, np.nan)

    if prev_wave is not None and not prev_wave.empty:
        prior_subset = prev_wave[["ID", "householdID", "communityID", "birthdate", "education"]].rename(
            columns={"birthdate": "prev_birthdate", "education": "prev_education"}
        )
        merged = raw.merge(prior_subset, on=["ID", "householdID", "communityID"], how="left")
        merged["education"] = merged["bd001_w2_4"].combine_first(merged["prev_education"])
        merged["birthdate"] = merged["date"].combine_first(merged["prev_birthdate"])
    else:
        merged = raw.rename(columns={"bd001_w2_4": "education", "date": "birthdate"})

    merged = merged.dropna(
        subset=["ID", "householdID", "communityID", "ba000_w2_3", "birthdate", "education", "be001"]
    ).copy()
    merged["age"] = calculate_age_on_date(merged["birthdate"], "2015-12-31")

    cleaned = pd.DataFrame({
        "ID": merged["ID"],
        "householdID": merged["householdID"],
        "communityID": merged["communityID"],
        "gender": merged["ba000_w2_3"],
        "age": merged["age"],
        "birthdate": merged["birthdate"],
        "education": merged["education"],
        "martial": (merged["be001"] == 1).astype(float),
    })
    return cleaned[cleaned["age"] >= min_age].reset_index(drop=True)


def run_charls_cleaning_pipeline(
    base_dir: Union[str, Path] = "/Users/shanrongzi/Downloads/CHARLS (2)",
    min_age: int = 45,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run cleaning across 2011, 2013, and 2015 CHARLS waves and optionally export cleaned data."""
    base = Path(base_dir)
    p11 = base / "2011" / "Demographic_Background.dta"
    p13 = base / "2013" / "Demographic_Background.dta"
    p15 = base / "2015" / "Demographic_Background.dta"

    df11 = clean_charls_2011(p11, min_age=min_age)
    df13 = clean_charls_2013(p13, prev_wave=df11, min_age=min_age)
    df15 = clean_charls_2015(p15, prev_wave=df13, min_age=min_age)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df15.to_parquet(out_p, index=False) if out_p.suffix == ".parquet" else df15.to_csv(out_p, index=False)
        logger.info("Saved cleaned CHARLS 2015 demographic dataset to %s", out_p)

    return {"2011": df11, "2013": df13, "2015": df15}


# =============================================================================
# 2. DESIGN MATRIX & CLUSTERED INFERENCE MODULE
# =============================================================================

DEFAULT_FORMULA = """
ip_times ~ C(gender) + age + C(education) + C(martial)
           + NRPS + avg + C(disability) + C(chronic) + C(cancer)
           + C(mental) + C(pain) + C(smoke) + C(alcoholic)
           + ADL + C(self_reported) + C(life_satisfication)
"""


def sanitize_feature_names(columns: Iterable[str]) -> List[str]:
    """Replace brackets in Patsy column names so tree/boosting models remain compatible."""
    return [re.sub(r"[\[\]]", "_", str(col)) for col in columns]


def prepare_analysis_matrices(
    df: pd.DataFrame,
    formula: str = DEFAULT_FORMULA,
    treatment_col: str = "NRPS",
    cluster_col: str = "avg",
    secondary_outcome_col: str = "op_ever",
) -> Dict[str, object]:
    """Build clean design matrices for outcome, treatment, covariates, and cluster IDs."""
    y_mat, x_mat = patsy.dmatrices(formula, df, return_type="dataframe")
    non_empty = x_mat.columns[x_mat.sum(axis=0) > 0]
    x_mat = x_mat.loc[:, non_empty]

    y = y_mat.iloc[:, 0].astype(float)
    treatment = x_mat[treatment_col].astype(float)

    drop_cols = [c for c in ["Intercept", treatment_col] if c in x_mat.columns]
    x_features = x_mat.drop(columns=drop_cols).copy()
    x_features.columns = sanitize_feature_names(x_features.columns)

    cluster_id = df.loc[x_mat.index, cluster_col].astype("category")
    secondary_y = (
        df.loc[x_mat.index, secondary_outcome_col].astype(float)
        if secondary_outcome_col in df.columns
        else None
    )

    return {
        "index": x_mat.index,
        "y": y,
        "treatment": treatment,
        "X": x_features,
        "cluster_id": cluster_id,
        "secondary_y": secondary_y,
        "df_aligned": df.loc[x_mat.index].copy(),
    }


def get_clustered_se(
    fit_result,
    cluster_id: Optional[pd.Series],
    row_mask: Optional[Union[pd.Series, np.ndarray]] = None,
) -> np.ndarray:
    """Compute cluster-robust standard errors for a fitted statsmodels OLS model."""
    if cluster_id is not None:
        if row_mask is None:
            groups = cluster_id
        else:
            mask_arr = np.asarray(row_mask, dtype=bool)
            groups = cluster_id.iloc[mask_arr] if hasattr(cluster_id, "iloc") else np.asarray(cluster_id)[mask_arr]
        vcov = sm.stats.sandwich_covariance.cov_cluster(fit_result, groups)
        return np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return np.asarray(fit_result.HC0_se)


def fit_bivariate_ols_summary(
    y_primary: pd.Series,
    treatment: pd.Series,
    cluster_id: pd.Series,
    y_secondary: Optional[pd.Series] = None,
):
    """Fit baseline OLS regressions with cluster-robust standard errors."""
    df_reg = pd.DataFrame({
        "y_primary": y_primary,
        "treatment": treatment,
    })
    fit_primary = smf.ols("y_primary ~ treatment", data=df_reg).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_id}
    )

    fits = [fit_primary]
    if y_secondary is not None:
        df_reg["y_secondary"] = y_secondary
        fit_secondary = smf.ols("y_secondary ~ treatment", data=df_reg).fit(
            cov_type="cluster", cov_kwds={"groups": cluster_id}
        )
        fits.append(fit_secondary)

    return fits


# =============================================================================
# 3. GENERIC MACHINE LEARNING (CATE, BLP, GATEs) MODULE
# =============================================================================

def get_default_ml_models(random_state: int = 42) -> List[object]:
    """Return candidate ML regressors for nuisance and CATE estimation."""
    models: List[object] = [
        linear_model.LassoCV(cv=10, n_alphas=25, max_iter=500, tol=1e-4, n_jobs=1, random_state=random_state),
        ensemble.RandomForestRegressor(
            n_estimators=200, min_samples_leaf=20, random_state=random_state, n_jobs=-1
        ),
        neural_network.MLPRegressor(
            hidden_layer_sizes=(20, 10),
            max_iter=500,
            activation="logistic",
            solver="adam",
            tol=1e-3,
            early_stopping=True,
            alpha=1e-4,
            random_state=random_state,
        ),
    ]
    if xgb is not None:
        models.insert(
            2,
            xgb.XGBRegressor(
                n_estimators=200,
                max_depth=3,
                reg_lambda=2.0,
                reg_alpha=0.0,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    return models


def run_generic_ml_single_model(
    x: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    estimator: object,
    n_split: int = 11,
    n_group: int = 5,
    cluster_id: Optional[pd.Series] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """Estimate CATE, Best Linear Projection (BLP), and Group Average Treatment Effects (GATEs)."""
    rng = np.random.default_rng(seed)
    nobs = x.shape[0]
    x_reset = x.reset_index(drop=True)
    y_reset = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
    t_reset = pd.Series(np.asarray(treatment, dtype=float)).reset_index(drop=True)
    c_reset = cluster_id.reset_index(drop=True) if cluster_id is not None else None

    blp = np.zeros((n_split, 2))
    blp_se = np.zeros((n_split, 2))
    gate = np.zeros((n_split, n_group))
    gate_se = np.zeros((n_split, n_group))
    baseline = np.zeros((nobs, n_split))
    cate = np.zeros((nobs, n_split))
    lamb = np.zeros((n_split, 2))

    p_treat = float(t_reset.mean())

    for i in range(n_split):
        main_mask = rng.random(nobs) > 0.5
        aux1_mask = (~main_mask) & (t_reset == 1)
        aux0_mask = (~main_mask) & (t_reset == 0)

        if aux1_mask.sum() < 2 or aux0_mask.sum() < 2:
            raise ValueError(f"Split {i} has insufficient treated or control observations in auxiliary sample.")

        mod1 = base.clone(estimator).fit(x_reset.loc[aux1_mask], y_reset.loc[aux1_mask])
        mod0 = base.clone(estimator).fit(x_reset.loc[aux0_mask], y_reset.loc[aux0_mask])

        b_hat = np.asarray(mod0.predict(x_reset), dtype=float)
        s_hat = np.asarray(mod1.predict(x_reset), dtype=float) - b_hat
        baseline[:, i] = b_hat
        cate[:, i] = s_hat

        es_mean = float(np.mean(s_hat))
        reg_df = pd.DataFrame({
            "y": y_reset,
            "B": b_hat,
            "treatment": t_reset,
            "S": s_hat,
            "excess_S": s_hat - es_mean,
        })

        blp_fit = smf.ols("y ~ B + I(treatment - p_treat) + I((treatment - p_treat) * excess_S)", data=reg_df.loc[main_mask]).fit()
        blp[i, :] = blp_fit.params.iloc[2:4].to_numpy()
        blp_se[i, :] = get_clustered_se(blp_fit, c_reset, main_mask)[2:4]
        lamb[i, 0] = (blp_fit.params.iloc[-1] ** 2) * float(np.var(s_hat))

        cutoffs = np.quantile(s_hat, np.linspace(0.0, 1.0, n_group + 1))
        cutoffs[-1] += 1.0
        for k in range(n_group):
            reg_df[f"G{k}"] = ((cutoffs[k] <= s_hat) & (s_hat < cutoffs[k + 1])).astype(float)

        gate_formula = "y ~ B + " + " + ".join(f"I((treatment - p_treat) * G{k})" for k in range(n_group))
        gate_fit = smf.ols(gate_formula, data=reg_df.loc[main_mask]).fit()
        gate[i, :] = gate_fit.params.iloc[2 : 2 + n_group].to_numpy()
        gate_se[i, :] = get_clustered_se(gate_fit, c_reset, main_mask)[2 : 2 + n_group]
        lamb[i, 1] = float(np.mean(gate[i, :] ** 2))

    return {
        "name": type(estimator).__name__,
        "gate": gate,
        "gate_se": gate_se,
        "blp": blp,
        "blp_se": blp_se,
        "Lambda": lamb,
        "baseline": baseline,
        "cate": cate,
    }


def summarize_ml_output(ml_out: Dict[str, object]) -> Dict[str, object]:
    """Aggregate split-level estimates via median across random sample splits."""
    summary: Dict[str, object] = {"name": ml_out["name"]}
    for key in ["blp", "blp_se", "gate", "gate_se", "Lambda"]:
        summary[key] = np.nanmedian(ml_out[key], axis=0)
    return summary


def evaluate_ml_models(
    models: Sequence[object],
    x: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    n_split: int = 11,
    n_group: int = 5,
    cluster_id: Optional[pd.Series] = None,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Evaluate all candidate ML models across sample splits."""
    results: List[Dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for idx, model in enumerate(models):
            logger.info("Evaluating causal ML estimator: %s", type(model).__name__)
            res = run_generic_ml_single_model(
                x=x,
                y=y,
                treatment=treatment,
                estimator=model,
                n_split=n_split,
                n_group=n_group,
                cluster_id=cluster_id,
                seed=seed + idx,
            )
            results.append(res)
    return results


# =============================================================================
# 4. COVARIATE GROUP MEANS & REPORTING MODULE
# =============================================================================

def covariate_mean_by_cate_group(
    covariate: pd.Series,
    ml_out: Dict[str, object],
    cluster_id: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute mean and clustered SE of a covariate across CATE quantile groups."""
    n_group = ml_out["gate"].shape[1]
    n_splits = ml_out["cate"].shape[1]
    cov_reset = pd.Series(np.asarray(covariate, dtype=float)).reset_index(drop=True)
    c_reset = cluster_id.reset_index(drop=True) if cluster_id is not None else None

    group_means = np.zeros((n_splits, n_group))
    group_ses = np.zeros((n_splits, n_group))

    for i in range(n_splits):
        s_hat = ml_out["cate"][:, i]
        cutoffs = np.quantile(s_hat, np.linspace(0.0, 1.0, n_group + 1))
        cutoffs[-1] += 1.0

        dat = pd.DataFrame({"y": cov_reset})
        for k in range(n_group):
            dat[f"G{k}"] = ((cutoffs[k] <= s_hat) & (s_hat < cutoffs[k + 1])).astype(float)

        valid_rows = ~dat["y"].isna()
        sub = dat.loc[valid_rows].astype(float)
        formula = "y ~ -1 + " + " + ".join(f"G{k}" for k in range(n_group))
        fit = smf.ols(formula, data=sub).fit()

        group_means[i, :] = fit.params.to_numpy()
        group_ses[i, :] = get_clustered_se(fit, c_reset, valid_rows.to_numpy())

    return pd.DataFrame({
        "group": np.arange(n_group),
        "mean": np.nanmedian(group_means, axis=0),
        "se": np.nanmedian(group_ses, axis=0),
    })


def compute_covariate_group_profiles(
    df_aligned: pd.DataFrame,
    results: Sequence[Dict[str, object]],
    variables: Sequence[str],
    cluster_id: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute CATE group profiles with 95% confidence intervals across variables and models."""
    frames: List[pd.DataFrame] = []
    for res in results:
        for var in variables:
            if var not in df_aligned.columns:
                logger.warning("Skipping missing covariate '%s' from group profile calculation.", var)
                continue
            prof = covariate_mean_by_cate_group(df_aligned[var], res, cluster_id=cluster_id)
            prof["method"] = res["name"]
            prof["variable"] = var
            prof["plus2sd"] = prof["mean"] + 1.96 * prof["se"]
            prof["minus2sd"] = prof["mean"] - 1.96 * prof["se"]
            frames.append(prof)

    if not frames:
        return pd.DataFrame(columns=["group", "mean", "se", "method", "variable", "plus2sd", "minus2sd"])
    return pd.concat(frames, ignore_index=True)


def generate_summary_tables(results: Sequence[Dict[str, object]]) -> Dict[str, pd.DataFrame]:
    """Build CATE correlation, BLP, and GATE summary DataFrames."""
    summaries = [summarize_ml_output(r) for r in results]
    cate_medians = pd.DataFrame({
        r["name"]: np.median(r["cate"], axis=1) for r in results
    })
    cate_corr = cate_medians.corr()

    blp_rows = []
    gate_rows = []
    for s in summaries:
        m_name = s["name"]
        for idx, val in enumerate(s["blp"]):
            blp_rows.append({"method": m_name, "coef_idx": idx, "blp": val, "se": s["blp_se"][idx]})
        for g_idx, val in enumerate(s["gate"]):
            gate_rows.append({"method": m_name, "group": g_idx, "gate": val, "se": s["gate_se"][g_idx]})

    return {
        "cate_medians": cate_medians,
        "cate_corr": cate_corr,
        "blp_summary": pd.DataFrame(blp_rows),
        "gate_summary": pd.DataFrame(gate_rows),
    }


def plot_diagnostics(
    summary_tables: Dict[str, pd.DataFrame],
    group_profiles: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Create CATE pairplot and covariate group profile facet grid."""
    out_dir = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    cate_medians = summary_tables["cate_medians"]
    if cate_medians.shape[1] >= 2:
        pair_grid = sns.pairplot(cate_medians, diag_kind="kde", kind="reg")
        if out_dir is not None:
            pair_grid.savefig(out_dir / "cate_pairplot.png", dpi=300, bbox_inches="tight")
            plt.close(pair_grid.figure)

    if not group_profiles.empty:
        facet = sns.FacetGrid(
            group_profiles, col="variable", col_wrap=5, hue="method", sharey=False
        )
        facet.map(plt.plot, "group", "mean")
        facet.map(plt.plot, "group", "plus2sd", ls="--")
        facet.map(plt.plot, "group", "minus2sd", ls="--")
        facet.add_legend()
        if out_dir is not None:
            facet.savefig(out_dir / "covariate_group_profiles.png", dpi=300, bbox_inches="tight")
            plt.close(facet.figure)


# =============================================================================
# 5. SYNTHETIC SMOKE TEST / DEMO ENTRYPOINT
# =============================================================================

def make_synthetic_charls_sample(n: int = 600, seed: int = 2026) -> pd.DataFrame:
    """Generate a synthetic dataset matching the econometric schema for testing."""
    rng = np.random.default_rng(seed)
    gender = rng.integers(1, 3, size=n)
    age = rng.integers(45, 86, size=n)
    education = rng.integers(1, 6, size=n)
    martial = rng.integers(0, 2, size=n)
    avg = rng.integers(1, 21, size=n)
    nrps = rng.binomial(1, 0.65, size=n)
    disability = rng.integers(0, 2, size=n)
    chronic = rng.integers(0, 4, size=n)
    cancer = rng.integers(0, 2, size=n)
    mental = rng.integers(0, 2, size=n)
    pain = rng.integers(0, 2, size=n)
    smoke = rng.integers(0, 2, size=n)
    alcoholic = rng.integers(0, 2, size=n)
    adl = rng.normal(0.0, 1.0, size=n)
    self_reported = rng.integers(1, 6, size=n)
    life_satisfication = rng.integers(1, 6, size=n)

    true_cate = 0.15 + 0.01 * (age - 60) + 0.12 * disability - 0.08 * chronic
    baseline_util = 0.5 + 0.02 * chronic + 0.25 * disability + 0.1 * adl + rng.normal(0.0, 0.4, size=n)
    ip_times = np.maximum(0.0, baseline_util + nrps * true_cate + rng.normal(0.0, 0.3, size=n))
    op_ever = (ip_times + rng.normal(0.0, 0.5, size=n) > 0.6).astype(int)

    return pd.DataFrame({
        "gender": gender,
        "age": age,
        "education": education,
        "martial": martial,
        "NRPS": nrps,
        "avg": avg,
        "disability": disability,
        "chronic": chronic,
        "cancer": cancer,
        "mental": mental,
        "pain": pain,
        "smoke": smoke,
        "alcoholic": alcoholic,
        "ADL": adl,
        "self_reported": self_reported,
        "life_satisfication": life_satisfication,
        "ip_times": ip_times,
        "op_ever": op_ever,
    })


def run_causal_ml_pipeline(
    data: pd.DataFrame,
    formula: str = DEFAULT_FORMULA,
    treatment_col: str = "NRPS",
    cluster_col: str = "avg",
    n_split: int = 11,
    n_group: int = 5,
    save_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Execute end-to-end preparation, OLS baseline, Generic ML estimation, and reporting."""
    prep = prepare_analysis_matrices(
        df=data,
        formula=formula,
        treatment_col=treatment_col,
        cluster_col=cluster_col,
    )
    ols_fits = fit_bivariate_ols_summary(
        y_primary=prep["y"],
        treatment=prep["treatment"],
        cluster_id=prep["cluster_id"],
        y_secondary=prep["secondary_y"],
    )
    models = get_default_ml_models()
    results = evaluate_ml_models(
        models=models,
        x=prep["X"],
        y=prep["y"],
        treatment=prep["treatment"],
        n_split=n_split,
        n_group=n_group,
        cluster_id=prep["cluster_id"],
    )
    tables = generate_summary_tables(results)
    profile_vars = [
        "gender", "age", "education", "martial", "avg", "disability",
        "chronic", "cancer", "mental", "pain", "smoke", "alcoholic",
        "ADL", "self_reported", "life_satisfication",
    ]
    profiles = compute_covariate_group_profiles(
        df_aligned=prep["df_aligned"],
        results=results,
        variables=profile_vars,
        cluster_id=prep["cluster_id"],
    )
    plot_diagnostics(tables, profiles, save_dir=save_dir)

    return {
        "prepared": prep,
        "ols_fits": ols_fits,
        "ml_results": results,
        "summary_tables": tables,
        "group_profiles": profiles,
    }


if __name__ == "__main__":
    sample_df = make_synthetic_charls_sample()
    outputs = run_causal_ml_pipeline(sample_df, n_split=5, n_group=5)
    print("
--- CATE Correlation ---")
    print(outputs["summary_tables"]["cate_corr"])
    print("
--- BLP Summary ---")
    print(outputs["summary_tables"]["blp_summary"])
    print("
--- GATE Summary ---")
    print(outputs["summary_tables"]["gate_summary"].head(10))
