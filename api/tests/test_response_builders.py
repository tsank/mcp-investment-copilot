"""
api/tests/test_response_builders.py

Tests for the response builders in api/routes/analyse.py that were added
to plumb GARCH forecast and rolling risk data through to the frontend.

These close the "green suite != new path works" gap: they assert the new
data actually survives the state → response mapping, not just that nothing
regressed.

Run from project root:
    OPENAI_API_KEY=sk-dummy pytest api/tests/test_response_builders.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.routes.analyse import _build_garch_forecast, _build_rolling_risk


# ── Minimal stand-ins matching the attributes the builders read ───────────────

class _Asset:
    def __init__(self):
        self.vol_forecast = [0.21, 0.205, 0.20]
        self.alpha_plus_beta = 0.94
        self.current_vol = 0.23
        self.longrun_vol = 0.19
        self.regime = "elevated"
        self.persistence_warning = False


class _GARCH:
    def __init__(self):
        self.per_asset = {"TCS.NS": _Asset()}
        self.horizon_days = 10


class _Window:
    def __init__(self):
        self.rolling_cvar = [0.012, 0.013, 0.011]
        self.rolling_vol = [0.10, 0.11, 0.105]
        self.window_end = [20, 21, 22]
        self.mean_cvar = 0.012
        self.mean_vol = 0.105
        self.window_size = 21
        self.n_points = 3


class _Rolling:
    def __init__(self):
        self.windows = {"21": _Window()}
        self.computation_window = "2y"


# ── _build_garch_forecast (the owed test) ─────────────────────────────────────

def test_build_garch_forecast_maps_real_fields():
    state = {"garch_result": _GARCH()}
    resp = _build_garch_forecast(state)
    assert resp is not None
    asset = resp.per_asset["TCS.NS"]
    assert asset.vol_forecast == [0.21, 0.205, 0.20]
    assert asset.alpha_plus_beta == 0.94
    assert asset.regime == "elevated"
    assert resp.horizon_days == 10


def test_build_garch_forecast_none_when_absent():
    assert _build_garch_forecast({"garch_result": None}) is None
    assert _build_garch_forecast({}) is None


# ── _build_rolling_risk (current + optimal) ───────────────────────────────────

def test_build_rolling_risk_maps_window_series():
    state = {"rolling_risk_current": _Rolling()}
    resp = _build_rolling_risk(state, "rolling_risk_current")
    assert resp is not None
    w = resp.windows["21"]
    assert w.rolling_cvar == [0.012, 0.013, 0.011]
    assert w.rolling_vol == [0.10, 0.11, 0.105]
    assert w.window_end == [20, 21, 22]
    assert w.n_points == 3
    assert resp.computation_window == "2y"


def test_build_rolling_risk_none_when_absent():
    # optimal absent on risk-only queries → builder must return None cleanly
    assert _build_rolling_risk({"rolling_risk_optimal": None}, "rolling_risk_optimal") is None
    assert _build_rolling_risk({}, "rolling_risk_current") is None