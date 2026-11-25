"""
Modular Python port and refinement of BMR post-estimation, prior,
and macroeconomic visualization utilities.

Dependencies:
    pip install numpy scipy pandas matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ==============================================================================
# MODULE 1: GRID LAYOUT & PLOTTING HELPERS
# Mirrors BMR's subplot grid sizing and figure saving logic.
# ==============================================================================

class GridLayoutHelper:
    """Computes subplot (rows, cols, pages) and handles figure saving."""

    @staticmethod
    def grid_shape(n_panels: int, max_per_page: int = 12) -> Tuple[int, int, int]:
        """Return (rows, cols, pages) for a given number of panels."""
        if n_panels <= 0:
            raise ValueError("n_panels must be > 0")

        if n_panels <= max_per_page:
            pages = 1
            count = n_panels
        else:
            pages = math.ceil(n_panels / max_per_page)
            count = max_per_page

        lookup = {
            1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2),
            5: (3, 2), 6: (3, 2), 7: (3, 3), 8: (3, 3), 9: (3, 3),
            10: (4, 3), 11: (4, 3), 12: (4, 3),
        }
        if count in lookup:
            rows, cols = lookup[count]
        else:
            cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)
        return rows, cols, pages

    @staticmethod
    def resolve_names(names: Optional[Sequence[str]], n: int, prefix: str = "VAR") -> List[str]:
        """Use provided names or generate default labels like VAR1, VAR2, ..."""
        if names is not None and len(names) == n:
            return list(names)
        return [f"{prefix}{i + 1}" for i in range(n)]

    @staticmethod
    def finalize_figure(fig: plt.Figure, save: bool, filename: Optional[str], dpi: int = 300) -> None:
        """Save figure if requested, otherwise display it."""
        fig.tight_layout()
        if save and filename:
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ==============================================================================
# MODULE 2: PRIOR DISTRIBUTIONS (from Prior.R)
# Evaluates moments (mean, mode, variance) and plots prior densities.
# ==============================================================================

class BayesianPrior:
    """Analytical moments and density curves for common macroeconometric priors."""

    def __init__(self, prior_form: str, prior_pars: Sequence[float], par_name: Optional[str] = None):
        self.prior_form = prior_form.lower()
        self.prior_pars = list(prior_pars)
        self.par_name = par_name or f"Prior ({self.prior_form})"

    def evaluate(self, n_grid: int = 400, print_moments: bool = True) -> Dict[str, Union[float, str, np.ndarray]]:
        p = self.prior_pars
        form = self.prior_form

        if form == "beta":
            a, b = p[0], p[1]
            x = np.linspace(0.0, 1.0, n_grid)
            density = stats.beta.pdf(x, a, b)
            mean = a / (a + b)
            mode: Union[float, str] = (a - 1.0) / (a + b - 2.0) if (a > 1.0 and b > 1.0) else np.nan
            var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))

        elif form == "uniform":
            low, high = p[0], p[1]
            pad = (high - low) / 10.0
            x = np.linspace(low - pad, high + pad, n_grid)
            density = stats.uniform.pdf(x, loc=low, scale=high - low)
            mean = 0.5 * (low + high)
            mode = f"any value in [{low}, {high}]"
            var = ((high - low) ** 2) / 12.0

        elif form == "igamma":
            # Inverse-Gamma parameterized with shape alpha = p[0], scale beta = p[1]
            # matching BMR's InvGamma(x; a, b) = (b^a / Gamma(a)) * x^(-a-1) * exp(-b/x)
            a, b = p[0], p[1]
            x = np.linspace(0.01, 5.0, n_grid)
            density = stats.invgamma.pdf(x, a=a, scale=b)
            mean = b / (a - 1.0) if a > 1.0 else np.nan
            mode = b / (a + 1.0)
            var = (b ** 2) / (((a - 1.0) ** 2) * (a - 2.0)) if a > 2.0 else np.nan

        elif form == "gamma":
            # BMR passes shape = p[0] and inverse-scale/variance-like scale p[1] -> rate = 1/p[1]
            shape = p[0]
            rate = 1.0 / p[1]
            scale = 1.0 / rate
            upper = max(2.0 * shape / (rate ** 2), shape / rate + 2.0 * rate * shape / (rate ** 2))
            x = np.linspace(0.01, upper, n_grid)
            density = stats.gamma.pdf(x, a=shape, scale=scale)
            mean = shape / rate
            mode = (shape - 1.0) / rate if shape > 1.0 else 0.0
            var = shape / (rate ** 2)

        elif form == "normal":
            # BMR passes mean = p[0] and variance = p[1]
            mu = p[0]
            sigma = math.sqrt(p[1])
            x = np.linspace(mu - 3.0 * sigma, mu + 3.0 * sigma, n_grid)
            density = stats.norm.pdf(x, loc=mu, scale=sigma)
            mean = mu
            mode = mu
            var = p[1]

        else:
            raise ValueError(f"Unknown prior form: '{self.prior_form}'")

        if print_moments:
            print(f"[{self.par_name}] Mean: {mean} | Mode: {mode} | Variance: {var}")

        return {"domain": x, "density": density, "mean": mean, "mode": mode, "variance": var}

    def plot(self, print_moments: bool = True, save: bool = False, filename: Optional[str] = None) -> None:
        res = self.evaluate(print_moments=print_moments)
        x, y = res["domain"], res["density"]
        mean, mode = res["mean"], res["mode"]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, y, color="crimson", lw=1.8, label="Density")
        ax.fill_between(x, 0.0, y, color="royalblue", alpha=0.2)

        if isinstance(mean, (int, float)) and not np.isnan(mean):
            ax.axvline(mean, color="black", lw=1.2, linestyle="-", label="Mean")
        if isinstance(mode, (int, float)) and not np.isnan(mode):
            ax.axvline(mode, color="navy", lw=1.2, linestyle="--", label="Mode")

        ax.set_title(self.par_name)
        ax.set_ylabel("Density")
        ax.grid(True, color="0.89")
        ax.legend()

        out_name = filename or f"prior_{self.prior_form}.png"
        GridLayoutHelper.finalize_figure(fig, save, out_name)


# ==============================================================================
# MODULE 3: ECONOMIC TIME-SERIES PLOTTING (from gtsplot.R)
# ==============================================================================

def plot_economic_series(
    data: Union[np.ndarray, pd.DataFrame],
    dates: Optional[Sequence] = None,
    var_names: Optional[Sequence[str]] = None,
    save: bool = False,
    prefix: str = "TimeSeries",
) -> None:
    """Plot multiple economic time series across paginated subplot grids."""
    if isinstance(data, pd.DataFrame):
        names = list(var_names) if var_names is not None else list(data.columns)
        idx = dates if dates is not None else data.index
        mat = data.to_numpy(dtype=float)
    else:
        mat = np.asarray(data, dtype=float)
        if mat.ndim == 1:
            mat = mat[:, None]
        n_obs, n_series = mat.shape
        names = GridLayoutHelper.resolve_names(var_names, n_series, prefix="Variable")
        idx = dates if dates is not None else np.arange(1, n_obs + 1)

    n_series = mat.shape[1]
    rows, cols, pages = GridLayoutHelper.grid_shape(n_series, max_per_page=12)
    per_page = rows * cols

    for page in range(pages):
        fig, axes = plt.subplots(rows, cols, figsize=(11, 13), squeeze=False)
        for panel in range(per_page):
            series_idx = page * per_page + panel
            r, c = divmod(panel, cols)
            ax = axes[r][c]
            if series_idx < n_series:
                ax.plot(idx, mat[:, series_idx], color="royalblue", lw=1.5)
                ax.set_ylabel(names[series_idx])
                ax.grid(True, color="0.89")
            else:
                ax.axis("off")

        fname = f"{prefix}.png" if pages == 1 else f"{prefix}_{page + 1}.png"
        GridLayoutHelper.finalize_figure(fig, save, fname)


# ==============================================================================
# MODULE 4: MODEL CONTAINER & POST-ESTIMATION ENGINE
# Replaces Rcpp object method calls (IRF, FEVD, forecast, states, mode_check)
# with a clean Python interface that can wrap MCMC draws or simulate a BVAR.
# ==============================================================================

@dataclass
class BayesianMacroModel:
    """
    Container for Bayesian VAR / TVP-BVAR / DSGE / DSGE-VAR posterior draws.

    Expected array shapes:
        - beta_draws: (K, M, n_draws) for BVARs
        - Sigma_draws: (M, M, n_draws) optional covariance draws
        - alpha_draws: (K*M, kT, n_draws) optional TVP-BVAR draws
        - dsge_draws: (n_draws, n_param) optional DSGE parameter draws
        - irf_draws: (M, M, periods, n_draws) precomputed or simulated IRFs
        - fevd_draws: (M, M, periods, n_draws) precomputed or simulated FEVDs
        - forecast_draws: (periods, M, n_draws) forecast trajectories
        - state_draws: (n_obs, n_states, n_draws) filtered state trajectories
        - Y: (n_obs, M) historical economic data
    """
    model_type: str = "bvarm"
    M: int = 3
    p: int = 2
    cons_term: bool = True
    c_int: int = 1
    tau: int = 0
    Y: Optional[np.ndarray] = None
    beta_draws: Optional[np.ndarray] = None
    Sigma_draws: Optional[np.ndarray] = None
    Psi_draws: Optional[np.ndarray] = None
    alpha_draws: Optional[np.ndarray] = None
    dsge_draws: Optional[np.ndarray] = None
    irf_draws: Optional[np.ndarray] = None
    fevd_draws: Optional[np.ndarray] = None
    forecast_draws: Optional[np.ndarray] = None
    state_draws: Optional[np.ndarray] = None
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        if self.beta_draws is not None:
            return self.beta_draws.shape[2]
        if self.alpha_draws is not None:
            return self.alpha_draws.shape[2]
        if self.dsge_draws is not None:
            return self.dsge_draws.shape[0]
        if self.irf_draws is not None:
            return self.irf_draws.shape[3]
        raise ValueError("No posterior draws available on model.")

    def get_irfs(self, periods: int = 10, shock_index: int = 0) -> np.ndarray:
        """Return IRF array of shape (M, M, periods, n_draws)."""
        if self.irf_draws is not None:
            arr = self.irf_draws
            if arr.ndim == 4:
                return arr[:, :, :periods, :]
            raise ValueError("Unexpected irf_draws dimensions.")
        # Fallback synthetic IRF if not explicitly provided
        rng = np.random.default_rng(42 + shock_index)
        draws = self.n_draws
        out = np.zeros((self.M, self.M, periods, draws))
        horizons = np.arange(periods)[None, None, :, None]
        decay = np.exp(-0.25 * horizons)
        amp = rng.normal(0.5, 0.25, size=(self.M, self.M, 1, draws))
        out[:] = amp * decay
        return out

    def get_fevd(self, periods: int = 10) -> np.ndarray:
        """Return FEVD array of shape (M, M, periods, n_draws) summing to 1 across shocks."""
        if self.fevd_draws is not None:
            return self.fevd_draws[:, :, :periods, :]
        irfs = np.square(self.get_irfs(periods=periods))
        cum_irfs = np.cumsum(irfs, axis=2)
        totals = np.sum(cum_irfs, axis=1, keepdims=True)
        totals = np.where(totals == 0, 1.0, totals)
        return cum_irfs / totals

    def get_forecast(self, periods: int = 20, shocks: bool = True) -> np.ndarray:
        """Return forecast array of shape (periods, M, n_draws)."""
        if self.forecast_draws is not None:
            return self.forecast_draws[:periods, :, :]
        rng = np.random.default_rng(100)
        last_y = self.Y[-1] if self.Y is not None and len(self.Y) > 0 else np.zeros(self.M)
        draws = self.n_draws
        fc = np.zeros((periods, self.M, draws))
        curr = np.broadcast_to(last_y[None, :], (draws, self.M)).copy()
        for h in range(periods):
            noise = rng.normal(0.0, 0.3 if shocks else 0.0, size=(draws, self.M))
            curr = 0.85 * curr + noise
            fc[h] = curr.T
        return fc

    def get_states(self) -> np.ndarray:
        """Return filtered state array of shape (n_obs, n_states, n_draws)."""
        if self.state_draws is not None:
            return self.state_draws
        raise ValueError("No filtered state draws available on model.")

    def evaluate_log_posterior_grid(
        self, mode_vals: Sequence[float], grid_size: int = 201, scale_val: float = 1.0
    ) -> np.ndarray:
        """
        Evaluate a 1D slice around each parameter's mode value.
        Returns shape (grid_size, n_param, 2) where [:, k, 0] is grid and [:, k, 1] is log-posterior.
        """
        mode_arr = np.asarray(mode_vals, dtype=float)
        n_param = len(mode_arr)
        out = np.zeros((grid_size, n_param, 2))

        for k in range(n_param):
            center = mode_arr[k]
            span = max(abs(center) * 0.25 * scale_val, 0.1 * scale_val)
            grid = np.linspace(center - span, center + span, grid_size)
            # Quadratic local approximation around mode for demonstration
            log_post = -0.5 * ((grid - center) / (0.4 * span)) ** 2 - 100.0
            out[:, k, 0] = grid
            out[:, k, 1] = log_post
        return out


# ==============================================================================
# MODULE 5: POST-ESTIMATION ANALYSIS & PLOTTING (IRF, FEVD, Forecast, States, Mode)
# ==============================================================================

class BayesianPostProcessor:
    """Computes quantiles and plots IRFs, FEVDs, forecasts, states, and mode checks."""

    def __init__(self, model: BayesianMacroModel, var_names: Optional[Sequence[str]] = None):
        self.model = model
        self.var_names = GridLayoutHelper.resolve_names(var_names, model.M, prefix="VAR")

    @staticmethod
    def _quantiles_3d(arr: np.ndarray, percentiles: Sequence[float], axis: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (lower, median, upper) along the draw axis."""
        q_low, q_mid, q_high = np.quantile(arr, percentiles, axis=axis)
        return q_low, q_mid, q_high

    def plot_irf(
        self,
        periods: int = 10,
        cumulative: bool = False,
        cumul_inds: Optional[Sequence[int]] = None,
        percentiles: Sequence[float] = (0.05, 0.50, 0.95),
        which_shock: Optional[Sequence[int]] = None,
        which_response: Optional[Sequence[int]] = None,
        shocks_row_order: bool = True,
        save: bool = False,
        save_title: Optional[str] = None,
    ) -> np.ndarray:
        if periods <= 0:
            raise ValueError("periods must be > 0")

        M = self.model.M
        shocks = list(range(M)) if which_shock is None else list(which_shock)
        responses = list(range(M)) if which_response is None else list(which_response)

        # irf shape: (response, shock, period, draw)
        irf = self.model.get_irfs(periods=periods).astype(float).copy()
        if cumulative:
            inds = list(range(M)) if cumul_inds is None else list(cumul_inds)
            irf[inds, :, :, :] = np.cumsum(irf[inds, :, :, :], axis=2)

        low, mid, high = self._quantiles_3d(irf, percentiles, axis=-1)
        horizons = np.arange(1, periods + 1)

        # Store plot_vals as (periods, 4, response, shock) -> [low, mid, high, time]
        plot_vals = np.zeros((periods, 4, M, M))
        for r in range(M):
            for s in range(M):
                plot_vals[:, :, r, s] = np.column_stack([low[r, s], mid[r, s], high[r, s], horizons])

        if len(responses) == M and len(shocks) == M:
            fig, axes = plt.subplots(M, M, figsize=(13, 13), squeeze=False)
            for s_idx, s in enumerate(shocks):
                for r_idx, r in enumerate(responses):
                    row, col = (s_idx, r_idx) if shocks_row_order else (r_idx, s_idx)
                    ax = axes[row][col]
                    ax.plot(horizons, mid[r, s], color="red", lw=2.0)
                    ax.fill_between(horizons, low[r, s], high[r, s], color="blue", alpha=0.2)
                    ax.axhline(0.0, color="black", lw=0.8)
                    ax.set_ylabel(f"{self.var_names[s]} -> {self.var_names[r]}")
                    ax.grid(True, color="0.89")
            fname = f"{save_title or 'IRFs'}.png"
            GridLayoutHelper.finalize_figure(fig, save, fname)
        else:
            rows, cols, _ = GridLayoutHelper.grid_shape(len(responses))
            for s in shocks:
                fig, axes = plt.subplots(rows, cols, figsize=(13, 13), squeeze=False)
                for idx, r in enumerate(responses):
                    row, col = divmod(idx, cols)
                    ax = axes[row][col]
                    ax.plot(horizons, mid[r, s], color="red", lw=2.0)
                    ax.fill_between(horizons, low[r, s], high[r, s], color="blue", alpha=0.2)
                    ax.axhline(0.0, color="black", lw=0.8)
                    ax.set_ylabel(f"{self.var_names[s]} -> {self.var_names[r]}")
                    ax.grid(True, color="0.89")
                for idx in range(len(responses), rows * cols):
                    row, col = divmod(idx, cols)
                    axes[row][col].axis("off")
                fname = f"{save_title or self.var_names[s] + '_Shock'}.png"
                GridLayoutHelper.finalize_figure(fig, save, fname)

        return plot_vals

    def compare_irfs(
        self,
        other_model: BayesianMacroModel,
        periods: int = 10,
        cumulative: bool = False,
        cumul_inds: Optional[Sequence[int]] = None,
        percentiles: Sequence[float] = (0.05, 0.50, 0.95),
        shocks_row_order: bool = True,
        save: bool = False,
        save_title: str = "IRF_Comparison",
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model.M != other_model.M:
            raise ValueError("Models must have the same number of variables (M).")

        M = self.model.M
        horizons = np.arange(1, periods + 1)

        def _prep(m: BayesianMacroModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            arr = m.get_irfs(periods=periods).astype(float).copy()
            if cumulative:
                inds = list(range(M)) if cumul_inds is None else list(cumul_inds)
                arr[inds, :, :, :] = np.cumsum(arr[inds, :, :, :], axis=2)
            lo, md, hi = self._quantiles_3d(arr, percentiles, axis=-1)
            pv = np.zeros((periods, 4, M, M))
            for r in range(M):
                for s in range(M):
                    pv[:, :, r, s] = np.column_stack([lo[r, s], md[r, s], hi[r, s], horizons])
            return lo, md, hi, pv

        lo1, md1, hi1, pv1 = _prep(self.model)
        lo2, md2, hi2, pv2 = _prep(other_model)

        fig, axes = plt.subplots(M, M, figsize=(13, 13), squeeze=False)
        for s in range(M):
            for r in range(M):
                row, col = (s, r) if shocks_row_order else (r, s)
                ax = axes[row][col]
                ax.plot(horizons, md1[r, s], color="blue", lw=2.0, label="Model 1")
                ax.plot(horizons, lo1[r, s], color="blue", lw=1.0, ls=":")
                ax.plot(horizons, hi1[r, s], color="blue", lw=1.0, ls=":")

                ax.plot(horizons, md2[r, s], color="darkred", lw=2.0, label="Model 2")
                ax.plot(horizons, lo2[r, s], color="darkred", lw=1.0, ls=":")
                ax.plot(horizons, hi2[r, s], color="darkred", lw=1.0, ls=":")

                ax.axhline(0.0, color="black", lw=0.8)
                ax.set_ylabel(f"{self.var_names[s]} -> {self.var_names[r]}")
                ax.grid(True, color="0.89")

        GridLayoutHelper.finalize_figure(fig, save, f"{save_title}.png")
        return pv1, pv2

    def plot_fevd(
        self,
        periods: int = 10,
        percentiles: Sequence[float] = (0.05, 0.50, 0.95),
        shocks_row_order: bool = True,
        save: bool = False,
        save_title: str = "FEVDs",
    ) -> np.ndarray:
        if periods <= 0:
            raise ValueError("periods must be > 0")

        M = self.model.M
        fevd = self.model.get_fevd(periods=periods)  # (response, shock, period, draw)
        low, mid, high = self._quantiles_3d(fevd, percentiles, axis=-1)
        horizons = np.arange(1, periods + 1)

        plot_vals = np.zeros((periods, 4, M, M))
        fig, axes = plt.subplots(M, M, figsize=(13, 13), squeeze=False)

        for s in range(M):
            for r in range(M):
                plot_vals[:, :, r, s] = np.column_stack([low[r, s], mid[r, s], high[r, s], horizons])
                row, col = (s, r) if shocks_row_order else (r, s)
                ax = axes[row][col]
                ax.plot(horizons, mid[r, s], color="red", lw=2.0)
                ax.fill_between(horizons, low[r, s], high[r, s], color="blue", alpha=0.2)
                ax.axhline(0.0, color="black", lw=0.8)
                ax.set_ylabel(f"Shock {self.var_names[s]} -> {self.var_names[r]}")
                ax.grid(True, color="0.89")

        GridLayoutHelper.finalize_figure(fig, save, f"{save_title}.png")
        return plot_vals

    def plot_forecast(
        self,
        periods: int = 20,
        shocks: bool = True,
        percentiles: Sequence[float] = (0.05, 0.50, 0.95),
        use_mean: bool = False,
        back_data: int = 0,
        save: bool = False,
        filename: str = "Forecast.png",
    ) -> Tuple[np.ndarray, np.ndarray]:
        if periods <= 0:
            raise ValueError("periods must be > 0")

        fc_cube = self.model.get_forecast(periods=periods, shocks=shocks)  # (periods, M, draws)
        low, mid, high = self._quantiles_3d(fc_cube, percentiles, axis=2)
        fc_mean = np.mean(fc_cube, axis=2)
        center = fc_mean if use_mean else mid

        M = self.model.M
        n_obs = self.model.Y.shape[0] if self.model.Y is not None else 0
        start_t = max(n_obs - back_data + 1, 1) if back_data > 0 else n_obs + 1
        end_t = n_obs + periods
        time_axis = np.arange(start_t, end_t + 1)

        plot_vals = np.zeros((periods + back_data, 4, M))
        fig, axes = plt.subplots(M, 1, figsize=(11, 13), squeeze=False)

        for i in range(M):
            c_seg = center[:, i]
            l_seg = low[:, i]
            h_seg = high[:, i]

            if back_data > 0 and self.model.Y is not None:
                hist = self.model.Y[-back_data:, i]
                c_seg = np.concatenate([hist, c_seg])
                l_seg = np.concatenate([hist, l_seg])
                h_seg = np.concatenate([hist, h_seg])

            plot_vals[:, :, i] = np.column_stack([l_seg, c_seg, h_seg, time_axis])

            ax = axes[i][0]
            ax.plot(time_axis, c_seg, color="red", lw=2.0)
            ax.fill_between(time_axis, l_seg, h_seg, color="blue", alpha=0.2)
            ax.axhline(0.0, color="0.3", lw=0.8)
            if back_data > 0:
                ax.axvline(n_obs, color="black", ls="--", lw=1.0)
            ax.set_ylabel(f"Forecast of {self.var_names[i]}")
            ax.grid(True, color="0.89")

        axes[-1][0].set_xlabel("Time")
        GridLayoutHelper.finalize_figure(fig, save, filename)
        return fc_mean, plot_vals

    def plot_states(
        self,
        percentiles: Sequence[float] = (0.05, 0.50, 0.95),
        state_names: Optional[Sequence[str]] = None,
        use_mean: bool = False,
        save: bool = False,
        prefix: str = "States",
    ) -> Tuple[np.ndarray, np.ndarray]:
        states_cube = self.model.get_states()  # (n_obs, n_states, n_draws)
        n_obs, n_states, _ = states_cube.shape
        low, mid, high = self._quantiles_3d(states_cube, percentiles, axis=2)
        states_mean = np.mean(states_cube, axis=2)
        center = states_mean if use_mean else mid

        names = GridLayoutHelper.resolve_names(state_names, n_states, prefix="State")
        t_axis = np.arange(1, n_obs + 1)

        plot_vals = np.zeros((n_obs, 4, n_states))
        for s in range(n_states):
            plot_vals[:, :, s] = np.column_stack([low[:, s], center[:, s], high[:, s], t_axis])

        rows, cols, pages = GridLayoutHelper.grid_shape(n_states, max_per_page=12)
        per_page = rows * cols

        for page in range(pages):
            fig, axes = plt.subplots(rows, cols, figsize=(11, 13), squeeze=False)
            for panel in range(per_page):
                s_idx = page * per_page + panel
                r, c = divmod(panel, cols)
                ax = axes[r][c]
                if s_idx < n_states:
                    ax.plot(t_axis, center[:, s_idx], color="red", lw=1.5)
                    ax.fill_between(t_axis, low[:, s_idx], high[:, s_idx], color="blue", alpha=0.2)
                    ax.set_ylabel(names[s_idx])
                    ax.set_xlabel("Time")
                    ax.grid(True, color="0.89")
                else:
                    ax.axis("off")

            fname = f"{prefix}.png" if pages == 1 else f"{prefix}_{page + 1}.png"
            GridLayoutHelper.finalize_figure(fig, save, fname)

        return states_mean, plot_vals

    def plot_mode_check(
        self,
        mode_vals: Sequence[float],
        grid_size: int = 201,
        scale_val: float = 1.0,
        par_names: Optional[Sequence[str]] = None,
        save: bool = False,
        prefix: str = "DSGEModeCheck",
    ) -> np.ndarray:
        vals = self.model.evaluate_log_posterior_grid(mode_vals, grid_size=grid_size, scale_val=scale_val)
        n_param = len(mode_vals)
        names = GridLayoutHelper.resolve_names(par_names, n_param, prefix="parameter")

        rows, cols, pages = GridLayoutHelper.grid_shape(n_param, max_per_page=12)
        per_page = rows * cols

        for page in range(pages):
            fig, axes = plt.subplots(rows, cols, figsize=(13, 13), squeeze=False)
            for panel in range(per_page):
                p_idx = page * per_page + panel
                r, c = divmod(panel, cols)
                ax = axes[r][c]
                if p_idx < n_param:
                    ax.plot(vals[:, p_idx, 0], vals[:, p_idx, 1], color="seagreen", lw=1.5)
                    ax.axvline(mode_vals[p_idx], color="black", ls="--", lw=1.0)
                    ax.set_title(names[p_idx])
                    ax.set_ylabel("Log Posterior")
                    ax.grid(True, color="0.89")
                else:
                    ax.axis("off")

            fname = f"{prefix}_{page + 1}.png"
            GridLayoutHelper.finalize_figure(fig, save, fname)

        return vals

    def plot_parameter_posteriors(
        self,
        par_names: Optional[Sequence[str]] = None,
        bins: int = 40,
        trace_plot: bool = False,
        save: bool = False,
        prefix: str = "DSGEParameters",
    ) -> None:
        if self.model.dsge_draws is None:
            raise ValueError("No dsge_draws found on model.")

        draws = self.model.dsge_draws  # (n_draws, n_param)
        n_draws, n_param = draws.shape
        names = GridLayoutHelper.resolve_names(par_names, n_param, prefix="parameter")
        rows, cols, pages = GridLayoutHelper.grid_shape(n_param, max_per_page=12)
        per_page = rows * cols

        for page in range(pages):
            fig, axes = plt.subplots(rows, cols, figsize=(13, 13), squeeze=False)
            for panel in range(per_page):
                p_idx = page * per_page + panel
                r, c = divmod(panel, cols)
                ax = axes[r][c]
                if p_idx < n_param:
                    ax.hist(draws[:, p_idx], bins=bins, color="darkblue", edgecolor="white", alpha=0.8)
                    ax.set_title(names[p_idx])
                    ax.grid(True, color="0.89")
                else:
                    ax.axis("off")
            fname = f"{prefix}.png" if pages == 1 else f"{prefix}_{page + 1}.png"
            GridLayoutHelper.finalize_figure(fig, save, fname)

        if trace_plot:
            x_iter = np.arange(1, n_draws + 1)
            for page in range(pages):
                fig, axes = plt.subplots(rows, cols, figsize=(13, 13), squeeze=False)
                for panel in range(per_page):
                    p_idx = page * per_page + panel
                    r, c = divmod(panel, cols)
                    ax = axes[r][c]
                    if p_idx < n_param:
                        ax.plot(x_iter, draws[:, p_idx], color="black", lw=0.8)
                        ax.set_xlabel("Keep Run")
                        ax.set_ylabel(names[p_idx])
                        ax.grid(True, color="0.89")
                    else:
                        ax.axis("off")
                fname = f"MCMCplot.png" if pages == 1 else f"MCMCplot_{page + 1}.png"
                GridLayoutHelper.finalize_figure(fig, save, fname)


# ==============================================================================
# MODULE 6: EXAMPLE WORKFLOW WITH SYNTHETIC MACROECONOMIC DATA
# ==============================================================================

def run_demo() -> None:
    rng = np.random.default_rng(123)
    n_obs, n_vars, n_draws = 120, 3, 500
    var_names = ["Output_Gap", "Inflation", "Policy_Rate"]

    # Simulated quarterly macro series
    dates = pd.date_range("1995-01-01", periods=n_obs, freq="QS")
    Y = np.zeros((n_obs, n_vars))
    for t in range(1, n_obs):
        Y[t] = 0.8 * Y[t - 1] + rng.normal(0.0, 0.5, size=n_vars)

    # Prior inspection example
    beta_prior = BayesianPrior("beta", [2.0, 5.0], par_name="Calvo Price Stickiness (Beta)")
    beta_prior.evaluate(print_moments=True)

    # Time series plot example (set save=True to write PNG files)
    plot_economic_series(Y, dates=dates, var_names=var_names, save=False)

    # Model 1 & Model 2 containers
    model_1 = BayesianMacroModel(
        model_type="bvarm",
        M=n_vars,
        p=2,
        Y=Y,
        beta_draws=rng.normal(0.0, 0.2, size=(1 + 2 * n_vars, n_vars, n_draws)),
        dsge_draws=rng.normal(loc=[0.5, 1.5], scale=[0.1, 0.2], size=(n_draws, 2)),
        state_draws=rng.normal(0.0, 1.0, size=(n_obs, 2, n_draws)),
    )
    model_2 = BayesianMacroModel(
        model_type="bvarinw",
        M=n_vars,
        p=2,
        Y=Y,
        beta_draws=rng.normal(0.05, 0.2, size=(1 + 2 * n_vars, n_vars, n_draws)),
    )

    pp = BayesianPostProcessor(model_1, var_names=var_names)
    pp.plot_irf(periods=12, cumulative=False, save=False)
    pp.compare_irfs(model_2, periods=12, save=False)
    pp.plot_fevd(periods=12, save=False)
    pp.plot_forecast(periods=12, back_data=16, save=False)
    pp.plot_states(state_names=["Natural_Rate", "Potential_Output"], save=False)
    pp.plot_mode_check(mode_vals=[0.5, 1.5], par_names=["theta_p", "phi_pi"], save=False)


if __name__ == "__main__":
    run_demo()