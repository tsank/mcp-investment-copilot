"""
servers/risk_engine/tests/test_garch_forecast.py

Unit tests for the compute_garch_forecast tool.

Test strategy:
    Property-based validation — GARCH parameters do not have closed-form
    analytical solutions, so we validate mathematical properties rather
    than exact values:
        - Parameters must satisfy stationarity conditions (α + β < 1)
        - Volatility estimates must be positive and finite
        - Forecast must mean-revert toward long-run volatility
        - Regime classification must be consistent with vol levels
        - Handoff fields must be present and correctly structured

    All tests use real NSE fixture data loaded directly via pandas.
    This avoids cross-server import conflicts and tests against
    realistic financial return distributions.

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/risk_engine/tests/test_garch_forecast.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
SERVER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"

from tools.garch_forecast import (
    _compute_longrun_vol,
    _compute_regime,
    _compute_vol_forecast,
    _extract_params,
    _fit_garch_model,
    _get_current_vol,
    compute_garch_forecast,
)


# ── Shared fixture data ───────────────────────────────────────────────────────

def load_returns(symbol: str) -> np.ndarray:
    """
    Load log-returns for a single NSE symbol from fixture CSV.
    Reads directly via pandas — avoids cross-server import conflicts.
    """
    df = pd.read_csv(FIXTURES_DIR / f"{symbol}_2y.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].values
    log_returns = np.diff(np.log(close))
    return log_returns


def load_returns_dict(symbols: list[str]) -> dict[str, list[float]]:
    """
    Load log-returns for multiple symbols, date-aligned via intersection.
    """
    frames = {}
    for symbol in symbols:
        df = pd.read_csv(FIXTURES_DIR / f"{symbol}_2y.csv", parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        frames[symbol] = df

    date_sets = [set(df["Date"].astype(str)) for df in frames.values()]
    common_dates = sorted(set.intersection(*date_sets))

    log_returns = {}
    for symbol, df in frames.items():
        mask = df["Date"].astype(str).isin(set(common_dates))
        aligned = df[mask].sort_values("Date")
        close = aligned["Close"].values
        log_returns[symbol] = np.diff(np.log(close)).tolist()

    return log_returns


# ── Tests: _fit_garch_model ───────────────────────────────────────────────────

class TestFitGarchModel:

    def test_garch_fits_without_error(self):
        """GARCH(1,1) fits RELIANCE.NS returns without convergence error."""
        returns = load_returns("RELIANCE.NS")
        result = _fit_garch_model(returns, model="garch", innovations="student_t")
        assert result is not None

    def test_convergence_flag_is_zero(self):
        """Fitted model converges — convergence_flag = 0."""
        returns = load_returns("TCS.NS")
        result = _fit_garch_model(returns, model="garch", innovations="student_t")
        assert result.convergence_flag == 0

    def test_conditional_volatility_length(self):
        """
        Conditional volatility series has same length as return series.
        arch computes σ_t for each observation t = 1..T.
        """
        returns = load_returns("INFY.NS")
        result = _fit_garch_model(returns, model="garch", innovations="student_t")
        assert len(result.conditional_volatility) == len(returns)

    def test_egarch_fits_without_error(self):
        """EGARCH variant fits without error."""
        returns = load_returns("HDFCBANK.NS")
        result = _fit_garch_model(returns, model="egarch", innovations="student_t")
        assert result.convergence_flag == 0

    def test_gaussian_innovations_fits_without_error(self):
        """Gaussian innovations variant fits without error."""
        returns = load_returns("RELIANCE.NS")
        result = _fit_garch_model(returns, model="garch", innovations="gaussian")
        assert result.convergence_flag == 0


# ── Tests: _extract_params ────────────────────────────────────────────────────

class TestExtractParams:

    def _fit_and_extract(self, symbol: str) -> dict:
        returns = load_returns(symbol)
        result = _fit_garch_model(returns, model="garch", innovations="student_t")
        return _extract_params(result, innovations="student_t")

    def test_omega_is_positive(self):
        """ω (long-run variance intercept) must be positive."""
        params = self._fit_and_extract("RELIANCE.NS")
        assert params["omega"] > 0

    def test_alpha_is_between_zero_and_one(self):
        """α (ARCH term) must be between 0 and 1."""
        params = self._fit_and_extract("TCS.NS")
        assert 0 < params["alpha"] < 1

    def test_beta_is_between_zero_and_one(self):
        """β (GARCH term) must be between 0 and 1."""
        params = self._fit_and_extract("INFY.NS")
        assert 0 < params["beta"] < 1

    def test_alpha_plus_beta_less_than_one(self):
        """
        α + β < 1 for a stationary GARCH process.
        If α + β ≥ 1, the long-run variance is undefined.
        NSE large-cap stocks typically have α + β ≈ 0.95-0.99.
        """
        params = self._fit_and_extract("HDFCBANK.NS")
        assert params["alpha"] + params["beta"] < 1.0

    def test_nu_is_positive_for_student_t(self):
        """
        ν (degrees of freedom) must be positive for Student-t innovations.
        Typically ν ∈ (2, 30) for equity returns — fat tails (low ν)
        to near-Gaussian (high ν).
        """
        params = self._fit_and_extract("RELIANCE.NS")
        assert params["nu"] is not None
        assert params["nu"] > 2.0

    def test_nu_is_none_for_gaussian(self):
        """ν is None for Gaussian innovations — no degrees of freedom."""
        returns = load_returns("RELIANCE.NS")
        result = _fit_garch_model(returns, model="garch", innovations="gaussian")
        params = _extract_params(result, innovations="gaussian")
        assert params["nu"] is None

    def test_omega_is_return_scale(self):
        """
        ω must be in return-scale (not percentage-scale).
        For daily equity returns, ω should be very small (< 0.001).
        If omega > 0.1, the percentage scaling was not reversed.
        """
        params = self._fit_and_extract("RELIANCE.NS")
        assert params["omega"] < 0.001


# ── Tests: _compute_vol_forecast ──────────────────────────────────────────────

class TestComputeVolForecast:

    def _get_forecast_inputs(self, symbol: str) -> dict:
        """Fit model and extract parameters for forecast testing."""
        returns = load_returns(symbol)
        result = _fit_garch_model(returns, model="garch", innovations="student_t")
        params = _extract_params(result, innovations="student_t")
        current_vol = _get_current_vol(result)
        return {
            "omega":       params["omega"],
            "alpha":       params["alpha"],
            "beta":        params["beta"],
            "current_vol": current_vol,
        }

    def test_forecast_length_equals_horizon(self):
        """Forecast list length equals horizon_days."""
        inputs = self._get_forecast_inputs("RELIANCE.NS")
        forecast = _compute_vol_forecast(
            inputs["omega"], inputs["alpha"], inputs["beta"],
            inputs["current_vol"], horizon_days=21
        )
        assert len(forecast) == 21

    def test_all_forecast_values_are_positive(self):
        """All forecast volatility values must be positive."""
        inputs = self._get_forecast_inputs("TCS.NS")
        forecast = _compute_vol_forecast(
            inputs["omega"], inputs["alpha"], inputs["beta"],
            inputs["current_vol"], horizon_days=63
        )
        assert all(v > 0 for v in forecast)

    def test_all_forecast_values_are_finite(self):
        """All forecast values must be finite — no NaN or inf."""
        inputs = self._get_forecast_inputs("INFY.NS")
        forecast = _compute_vol_forecast(
            inputs["omega"], inputs["alpha"], inputs["beta"],
            inputs["current_vol"], horizon_days=252
        )
        assert all(math.isfinite(v) for v in forecast)

    def test_forecast_mean_reverts_from_elevated_vol(self):
        """
        When current vol > long-run vol, forecast must decrease toward long-run.
        This is the core mean-reversion property of GARCH.
        """
        # Use a realistic omega, alpha, beta with known long-run vol
        omega = 0.00001
        alpha = 0.08
        beta  = 0.90
        longrun_vol = _compute_longrun_vol(omega, alpha, beta)

        # Set current vol well above long-run
        current_vol = longrun_vol * 2.0

        forecast = _compute_vol_forecast(omega, alpha, beta, current_vol, horizon_days=252)

        # Forecast must decrease (mean-revert) from elevated level
        assert forecast[0] < current_vol
        assert forecast[-1] < forecast[0]
        assert forecast[-1] > longrun_vol * 0.5  # should not overshoot

    def test_forecast_mean_reverts_from_suppressed_vol(self):
        """
        When current vol < long-run vol, forecast must increase toward long-run.
        """
        omega = 0.00001
        alpha = 0.08
        beta  = 0.90
        longrun_vol = _compute_longrun_vol(omega, alpha, beta)

        # Set current vol well below long-run
        current_vol = longrun_vol * 0.5

        forecast = _compute_vol_forecast(omega, alpha, beta, current_vol, horizon_days=252)

        # Forecast must increase (mean-revert upward)
        assert forecast[0] > current_vol
        assert forecast[-1] > forecast[0]


# ── Tests: _compute_longrun_vol ───────────────────────────────────────────────

class TestComputeLongrunVol:

    def test_known_inputs_produce_known_longrun_vol(self):
        """
        Hand-computed long-run vol for known parameters.
        omega=0.00001, alpha=0.05, beta=0.90 → persistence=0.95
        longrun_var = 0.00001 / (1 - 0.95) = 0.0002
        longrun_vol_daily = √0.0002 = 0.01414
        longrun_vol_annual = 0.01414 × √252 = 0.2245
        """
        omega = 0.00001
        alpha = 0.05
        beta  = 0.90
        longrun_vol = _compute_longrun_vol(omega, alpha, beta)
        expected = np.sqrt(omega / (1 - alpha - beta)) * np.sqrt(252)
        assert math.isclose(longrun_vol, expected, rel_tol=1e-6)

    def test_non_stationary_raises_value_error(self):
        """α + β >= 1 raises ValueError — long-run variance undefined."""
        with pytest.raises(ValueError, match="non-stationary"):
            _compute_longrun_vol(omega=0.00001, alpha=0.10, beta=0.95)

    def test_longrun_vol_is_positive(self):
        """Long-run volatility must be positive."""
        vol = _compute_longrun_vol(omega=0.00001, alpha=0.08, beta=0.88)
        assert vol > 0


# ── Tests: _compute_regime ────────────────────────────────────────────────────

class TestComputeRegime:

    def test_elevated_when_current_vol_high(self):
        """current_vol > longrun_vol × 1.25 → 'elevated'."""
        regime = _compute_regime(current_vol=0.30, longrun_vol=0.20)
        assert regime == "elevated"

    def test_suppressed_when_current_vol_low(self):
        """current_vol < longrun_vol × 0.75 → 'suppressed'."""
        regime = _compute_regime(current_vol=0.10, longrun_vol=0.20)
        assert regime == "suppressed"

    def test_normal_when_current_vol_close_to_longrun(self):
        """current_vol within ±25% of longrun_vol → 'normal'."""
        regime = _compute_regime(current_vol=0.20, longrun_vol=0.20)
        assert regime == "normal"

    def test_regime_boundary_elevated(self):
        """Exactly at 1.25× threshold — boundary is elevated."""
        regime = _compute_regime(current_vol=0.251, longrun_vol=0.20)
        assert regime == "elevated"

    def test_regime_boundary_suppressed(self):
        """Exactly at 0.75× threshold — boundary is suppressed."""
        regime = _compute_regime(current_vol=0.15, longrun_vol=0.20)
        assert regime == "suppressed"


# ── Tests: compute_garch_forecast (full tool) ─────────────────────────────────

class TestComputeGarchForecast:

    def test_output_has_all_required_keys(self):
        """Output dict has all keys matching GARCHResult schema."""
        log_returns = load_returns_dict(["RELIANCE.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)

        required_keys = {
            "per_asset", "portfolio_vol_forecast",
            "garch_model", "innovations_used", "horizon_days",
            "garch_params", "current_vols",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_per_asset_has_all_required_keys(self):
        """Each asset entry has all required sub-keys."""
        log_returns = load_returns_dict(["RELIANCE.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)

        asset_keys = {
            "params", "alpha_plus_beta", "current_vol", "longrun_vol",
            "vol_forecast", "regime", "aic", "bic", "persistence_warning",
        }
        for symbol in log_returns:
            assert asset_keys.issubset(set(result["per_asset"][symbol].keys()))

    def test_params_has_required_keys(self):
        """params sub-dict has omega, alpha, beta, nu."""
        log_returns = load_returns_dict(["TCS.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)
        params = result["per_asset"]["TCS.NS"]["params"]
        assert set(params.keys()) == {"omega", "alpha", "beta", "nu"}

    def test_vol_forecast_length_matches_horizon(self):
        """vol_forecast length equals requested horizon_days."""
        log_returns = load_returns_dict(["RELIANCE.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=63)
        forecast = result["per_asset"]["RELIANCE.NS"]["vol_forecast"]
        assert len(forecast) == 63

    def test_portfolio_vol_forecast_length_matches_horizon(self):
        """portfolio_vol_forecast length equals requested horizon_days."""
        log_returns = load_returns_dict(["RELIANCE.NS", "TCS.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)
        assert len(result["portfolio_vol_forecast"]) == 21

    def test_current_vol_is_positive(self):
        """current_vol (σ_T annualised) must be positive for all assets."""
        log_returns = load_returns_dict(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)
        for symbol in log_returns:
            vol = result["per_asset"][symbol]["current_vol"]
            assert vol > 0, f"{symbol}: current_vol not positive: {vol}"

    def test_current_vol_in_realistic_range(self):
        """
        Annualised vol for NSE large-caps should be between 10% and 60%.
        This validates the percentage scaling reversal in _extract_params
        and _get_current_vol.
        """
        log_returns = load_returns_dict(["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)
        for symbol in log_returns:
            vol = result["per_asset"][symbol]["current_vol"]
            assert 0.10 <= vol <= 0.60, \
                f"{symbol}: current_vol {vol:.4f} out of realistic range [0.10, 0.60]"

    def test_alpha_plus_beta_less_than_one(self):
        """α + β < 1 for all assets — stationarity condition."""
        log_returns = load_returns_dict(["RELIANCE.NS", "TCS.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)
        for symbol in log_returns:
            apb = result["per_asset"][symbol]["alpha_plus_beta"]
            assert apb < 1.0, \
                f"{symbol}: alpha+beta={apb:.4f} violates stationarity"

    def test_handoff_fields_present(self):
        """garch_params and current_vols handoff fields are correctly structured."""
        log_returns = load_returns_dict(["RELIANCE.NS", "TCS.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)

        for symbol in log_returns:
            assert symbol in result["garch_params"]
            assert symbol in result["current_vols"]
            assert set(result["garch_params"][symbol].keys()) == \
                {"omega", "alpha", "beta", "nu"}
            assert result["current_vols"][symbol] > 0

    def test_garch_model_echoed(self):
        """garch_model field echoes the requested model."""
        log_returns = load_returns_dict(["RELIANCE.NS"])
        result = compute_garch_forecast(
            log_returns, horizon_days=21, model="garch"
        )
        assert result["garch_model"] == "garch"

    def test_innovations_used_echoed(self):
        """innovations_used field echoes the requested innovations."""
        log_returns = load_returns_dict(["RELIANCE.NS"])
        result = compute_garch_forecast(
            log_returns, horizon_days=21, innovations="student_t"
        )
        assert result["innovations_used"] == "student_t"

    def test_horizon_days_echoed(self):
        """horizon_days field echoes the requested horizon."""
        log_returns = load_returns_dict(["RELIANCE.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=63)
        assert result["horizon_days"] == 63

    def test_regime_is_valid_value(self):
        """Regime classification is one of the three valid values."""
        log_returns = load_returns_dict(["RELIANCE.NS", "TCS.NS"])
        result = compute_garch_forecast(log_returns, horizon_days=21)
        valid_regimes = {"elevated", "normal", "suppressed"}
        for symbol in log_returns:
            regime = result["per_asset"][symbol]["regime"]
            assert regime in valid_regimes, \
                f"{symbol}: invalid regime '{regime}'"

    def test_invalid_model_raises_value_error(self):
        """Invalid model name raises ValueError."""
        log_returns = load_returns_dict(["RELIANCE.NS"])
        with pytest.raises(ValueError, match="Invalid model"):
            compute_garch_forecast(log_returns, model="invalid_model")

    def test_invalid_innovations_raises_value_error(self):
        """Invalid innovations name raises ValueError."""
        log_returns = load_returns_dict(["RELIANCE.NS"])
        with pytest.raises(ValueError, match="Invalid innovations"):
            compute_garch_forecast(log_returns, innovations="invalid_dist")

    def test_empty_log_returns_raises_value_error(self):
        """Empty log_returns dict raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_garch_forecast({})

    def test_insufficient_observations_raises_value_error(self):
        """Series with fewer than 100 observations raises ValueError."""
        log_returns = {"RELIANCE.NS": [0.001] * 50}
        with pytest.raises(ValueError, match="minimum 100"):
            compute_garch_forecast(log_returns)