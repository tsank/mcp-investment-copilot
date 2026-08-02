"""
servers/risk_engine/tests/test_rolling_cvar.py

Unit tests for the compute_rolling_risk / compute_rolling_cvar tools.

Test strategy:
    - Correctness by construction: the rolling endpoint must equal a manual
      trailing-window CVaR computed with the same primitives.
    - Multi-window output shape and point counts (n - window + 1).
    - Positive-loss sign convention (matches risk_metrics).
    - Annualised-vol convention (std_daily × √252) matches risk_metrics.
    - Short-history: windows longer than history are skipped, not fatal;
      total shortfall raises.

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/risk_engine/tests/test_rolling_cvar.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SERVER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))

from tools.rolling_cvar import (
    DEFAULT_WINDOWS,
    compute_rolling_cvar,
    compute_rolling_risk,
)
from tools.risk_metrics import (
    TRADING_DAYS_PER_YEAR,
    _compute_cvar,
    _compute_portfolio_returns,
    _compute_var,
)


def _synthetic_returns(n: int, seed: int = 7) -> dict[str, list[float]]:
    """Two aligned return series of length n."""
    rng = np.random.default_rng(seed)
    return {
        "AAA.NS": rng.normal(0.0005, 0.012, n).tolist(),
        "BBB.NS": rng.normal(0.0003, 0.015, n).tolist(),
    }


def _weights() -> dict[str, float]:
    return {"AAA.NS": 0.6, "BBB.NS": 0.4}


# ── compute_rolling_cvar (single window) ──────────────────────────────────────

def test_single_window_point_count():
    lr = _synthetic_returns(300)
    res = compute_rolling_cvar(lr, _weights(), window=252)
    assert res["n_points"] == 300 - 252 + 1
    assert len(res["rolling_cvar"]) == res["n_points"]
    assert len(res["rolling_vol"]) == res["n_points"]
    assert len(res["window_end"]) == res["n_points"]


def test_positive_loss_convention():
    lr = _synthetic_returns(300)
    res = compute_rolling_cvar(lr, _weights(), window=252)
    assert all(v > 0 for v in res["rolling_cvar"]), "CVaR must be positive loss"
    assert all(v > 0 for v in res["rolling_vol"]), "vol must be positive"


def test_endpoint_matches_manual_trailing_window():
    """The last rolling point must equal a hand-computed trailing-window CVaR."""
    lr = _synthetic_returns(300)
    w = _weights()
    res = compute_rolling_cvar(lr, w, window=252)

    pr = _compute_portfolio_returns(lr, w)
    tail = pr[-252:]
    var = _compute_var(tail, percentile=5.0)
    cvar = _compute_cvar(tail, var=var)

    assert res["rolling_cvar"][-1] == pytest.approx(cvar, abs=1e-12)


def test_vol_convention_matches_risk_metrics():
    """Rolling vol endpoint = std_daily(ddof=1) × √252 on the trailing window."""
    lr = _synthetic_returns(300)
    w = _weights()
    res = compute_rolling_cvar(lr, w, window=252)

    pr = _compute_portfolio_returns(lr, w)
    tail = np.array(pr[-252:])
    expected = float(tail.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    assert res["rolling_vol"][-1] == pytest.approx(expected, abs=1e-12)


def test_window_end_alignment():
    lr = _synthetic_returns(300)
    res = compute_rolling_cvar(lr, _weights(), window=252)
    # First end-day is window-1; last is n-1.
    assert res["window_end"][0] == 251
    assert res["window_end"][-1] == 299


def test_window_too_long_raises():
    lr = _synthetic_returns(100)
    with pytest.raises(ValueError):
        compute_rolling_cvar(lr, _weights(), window=252)


# ── compute_rolling_risk (multi-window) ───────────────────────────────────────

def test_multi_window_default_windows():
    lr = _synthetic_returns(300)
    res = compute_rolling_risk(lr, _weights())
    assert set(res["windows"].keys()) == {str(w) for w in DEFAULT_WINDOWS}
    # 21-day window yields the most points; 252 the fewest.
    assert res["windows"]["21"]["n_points"] > res["windows"]["252"]["n_points"]


def test_multi_window_shorter_windows_more_reactive():
    """Cluster sensitivity: shorter window has a wider CVaR range."""
    lr = _synthetic_returns(400, seed=3)
    res = compute_rolling_risk(lr, _weights())
    span21 = max(res["windows"]["21"]["rolling_cvar"]) - min(res["windows"]["21"]["rolling_cvar"])
    span252 = max(res["windows"]["252"]["rolling_cvar"]) - min(res["windows"]["252"]["rolling_cvar"])
    assert span21 > span252


def test_multi_window_skips_unsupported_but_keeps_others():
    """History supports 21 and 63 but not 252 — those two still returned."""
    lr = _synthetic_returns(100)
    res = compute_rolling_risk(lr, _weights())
    assert "21" in res["windows"]
    assert "63" in res["windows"]
    assert "252" not in res["windows"]


def test_multi_window_all_unsupported_raises():
    lr = _synthetic_returns(10)
    with pytest.raises(ValueError):
        compute_rolling_risk(lr, _weights())