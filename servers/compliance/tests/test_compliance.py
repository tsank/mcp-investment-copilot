"""
servers/compliance/tests/test_compliance.py

Unit tests for the check_compliance tool.

Test strategy:
    For each rule, construct portfolios that:
        (1) clearly pass — value well below limit
        (2) clearly fail — value well above limit
        (3) sit at the boundary — value exactly at limit

    All tests use retail_conservative_v1.0 ruleset unless noted.
    The YAML rulesets are the source of truth for limits.

    Retail conservative limits (v1.0):
        SINGLE_ASSET_CAP:  0.30 (hard)
        SECTOR_CAP:        0.40 (hard)
        CVAR_THRESHOLD:    0.25 (hard)
        MIN_ASSETS:        3    (soft)
        MIN_POSITION_SIZE: 0.02 (soft)

Run from project root:
    cd ~/genaiprojects/mcp-investment-copilot
    pytest servers/compliance/tests/test_compliance.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
SERVER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))

from tools.check_compliance import (
    _check_cvar_threshold,
    _check_min_assets,
    _check_min_position_size,
    _check_sector_cap,
    _check_single_asset_cap,
    _load_ruleset,
    check_compliance,
)


# ── Shared test data ──────────────────────────────────────────────────────────

# Standard 5-asset portfolio — passes all retail_conservative rules
PASSING_WEIGHTS = {
    "RELIANCE.NS":  0.20,
    "TCS.NS":       0.20,
    "INFY.NS":      0.20,
    "HDFCBANK.NS":  0.20,
    "ICICIBANK.NS": 0.20,
}

# Standard sector map for the 5 symbols above
SECTOR_MAP = {
    "RELIANCE.NS":  "Energy",
    "TCS.NS":       "Technology",
    "INFY.NS":      "Technology",
    "HDFCBANK.NS":  "Financial Services",
    "ICICIBANK.NS": "Financial Services",
}

PROFILE  = "retail_conservative"
VERSION  = "v1.0"
VAR_95   = 0.05   # 5% — context only
CVAR_95  = 0.10   # 10% — well below 25% threshold


# ── Tests: _load_ruleset ──────────────────────────────────────────────────────

class TestLoadRuleset:

    def test_loads_retail_conservative(self):
        """retail_conservative_v1.0 ruleset loads successfully."""
        ruleset = _load_ruleset("retail_conservative", "v1.0")
        assert "rules" in ruleset
        assert len(ruleset["rules"]) > 0

    def test_loads_institutional(self):
        """institutional_v1.0 ruleset loads successfully."""
        ruleset = _load_ruleset("institutional", "v1.0")
        assert "rules" in ruleset

    def test_version_with_v_prefix_accepted(self):
        """Version 'v1.0' and '1.0' both load the same file."""
        r1 = _load_ruleset("retail_conservative", "v1.0")
        r2 = _load_ruleset("retail_conservative", "1.0")
        assert r1["version"] == r2["version"]

    def test_missing_ruleset_raises_file_not_found(self):
        """Non-existent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_ruleset("nonexistent_profile", "v1.0")

    def test_retail_conservative_has_all_required_rules(self):
        """retail_conservative ruleset contains all expected rule IDs."""
        ruleset = _load_ruleset("retail_conservative", "v1.0")
        rule_ids = {r["id"] for r in ruleset["rules"]}
        required = {
            "SINGLE_ASSET_CAP", "SECTOR_CAP",
            "CVAR_THRESHOLD", "MIN_ASSETS", "MIN_POSITION_SIZE"
        }
        assert required.issubset(rule_ids)


# ── Tests: individual rule checkers ──────────────────────────────────────────

class TestRuleCheckers:

    # ── SINGLE_ASSET_CAP ────────────────────────────────────────────────────

    def test_single_asset_cap_passes_when_below_limit(self):
        """All weights below 0.30 — no violations."""
        weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        violations, warnings = _check_single_asset_cap(weights, 0.30, "hard")
        assert len(violations) == 0

    def test_single_asset_cap_fails_when_above_limit(self):
        """Weight of 0.40 exceeds 0.30 limit — hard violation."""
        weights = {"A": 0.40, "B": 0.30, "C": 0.30}
        violations, warnings = _check_single_asset_cap(weights, 0.30, "hard")
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "SINGLE_ASSET_CAP"
        assert violations[0]["severity"] == "hard"
        assert violations[0]["value"] == 0.40
        assert violations[0]["limit"] == 0.30

    def test_single_asset_cap_soft_becomes_warning(self):
        """Soft severity violation appears in warnings, not violations."""
        weights = {"A": 0.40, "B": 0.30, "C": 0.30}
        violations, warnings = _check_single_asset_cap(weights, 0.30, "soft")
        assert len(violations) == 0
        assert len(warnings) == 1

    def test_single_asset_cap_multiple_violations(self):
        """Two assets exceeding limit produces two violations."""
        weights = {"A": 0.40, "B": 0.40, "C": 0.20}
        violations, warnings = _check_single_asset_cap(weights, 0.30, "hard")
        assert len(violations) == 2

    def test_single_asset_cap_exactly_at_limit_passes(self):
        """Weight exactly at limit (0.30) does not violate."""
        weights = {"A": 0.30, "B": 0.35, "C": 0.35}
        violations, _ = _check_single_asset_cap(weights, 0.30, "hard")
        # A is exactly at limit — should not violate (uses > not >=)
        violating = [v for v in violations if v["value"] == 0.30]
        assert len(violating) == 0

    # ── SECTOR_CAP ───────────────────────────────────────────────────────────

    def test_sector_cap_passes_when_below_limit(self):
        """Sector weights below 0.40 — no violations."""
        weights = {
            "RELIANCE.NS": 0.20, "ADANIENT.NS": 0.15,   # Energy: 0.35
            "TCS.NS": 0.20, "INFY.NS": 0.15,             # Technology: 0.35
            "HDFCBANK.NS": 0.10, "SBIN.NS": 0.10,        # Financial: 0.20
        }
        sector_map = {
            "RELIANCE.NS": "Energy", "ADANIENT.NS": "Energy",
            "TCS.NS": "Technology", "INFY.NS": "Technology",
            "HDFCBANK.NS": "Financial Services", "SBIN.NS": "Financial Services",
        }
        violations, warnings = _check_sector_cap(weights, sector_map, 0.40, "hard")
        assert len(violations) == 0

    def test_sector_cap_fails_when_above_limit(self):
        """Financial Services at 0.50 exceeds 0.40 limit — exactly 1 violation."""
        weights = {
            "HDFCBANK.NS":  0.25,
            "ICICIBANK.NS": 0.25,  # Financial Services: 0.50
            "TCS.NS":       0.30,  # Technology: 0.30
            "RELIANCE.NS":  0.20,  # Energy: 0.20
        }
        sector_map = {
            "HDFCBANK.NS":  "Financial Services",
            "ICICIBANK.NS": "Financial Services",
            "TCS.NS":       "Technology",
            "RELIANCE.NS":  "Energy",
        }
        violations, warnings = _check_sector_cap(weights, sector_map, 0.40, "hard")
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "SECTOR_CAP"
        assert violations[0]["value"] == 0.50

    def test_sector_cap_unknown_sector_grouped_together(self):
        """Symbols without sector_map entry grouped under 'Unknown'."""
        weights = {"A": 0.60, "B": 0.40}
        sector_map = {}  # No mapping — both go to Unknown
        violations, warnings = _check_sector_cap(weights, sector_map, 0.40, "hard")
        # A+B = 1.0 under Unknown — exceeds 0.40 limit
        assert len(violations) == 1

    # ── CVAR_THRESHOLD ───────────────────────────────────────────────────────

    def test_cvar_threshold_passes_when_below_limit(self):
        """CVaR 0.10 below 0.25 limit — no violation."""
        violations, warnings = _check_cvar_threshold(0.10, 0.25, "hard")
        assert len(violations) == 0

    def test_cvar_threshold_fails_when_above_limit(self):
        """CVaR 0.30 exceeds 0.25 limit — hard violation."""
        violations, warnings = _check_cvar_threshold(0.30, 0.25, "hard")
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "CVAR_THRESHOLD"
        assert violations[0]["value"] == 0.30
        assert violations[0]["limit"] == 0.25

    def test_cvar_threshold_exactly_at_limit_passes(self):
        """CVaR exactly at limit does not violate."""
        violations, _ = _check_cvar_threshold(0.25, 0.25, "hard")
        assert len(violations) == 0

    # ── MIN_ASSETS ───────────────────────────────────────────────────────────

    def test_min_assets_passes_when_sufficient(self):
        """5 active assets meets minimum of 3 — no warning."""
        weights = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
        violations, warnings = _check_min_assets(weights, 3, "soft")
        assert len(warnings) == 0

    def test_min_assets_warns_when_insufficient(self):
        """2 active assets below minimum of 3 — soft warning."""
        weights = {"A": 0.5, "B": 0.5, "C": 0.0}
        violations, warnings = _check_min_assets(weights, 3, "soft")
        assert len(violations) == 0
        assert len(warnings) == 1
        assert warnings[0]["rule_id"] == "MIN_ASSETS"

    def test_min_assets_zero_weights_not_counted(self):
        """Zero-weight assets are not counted as active."""
        weights = {"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0}
        violations, warnings = _check_min_assets(weights, 3, "soft")
        assert len(warnings) == 1  # only 2 active assets

    # ── MIN_POSITION_SIZE ────────────────────────────────────────────────────

    def test_min_position_size_passes_when_above_limit(self):
        """All non-zero weights above 0.02 — no warning."""
        weights = {"A": 0.50, "B": 0.30, "C": 0.20}
        violations, warnings = _check_min_position_size(weights, 0.02, "soft")
        assert len(warnings) == 0

    def test_min_position_size_warns_for_token_position(self):
        """Weight of 0.01 below 0.02 limit — soft warning."""
        weights = {"A": 0.49, "B": 0.50, "C": 0.01}
        violations, warnings = _check_min_position_size(weights, 0.02, "soft")
        assert len(warnings) == 1
        assert warnings[0]["rule_id"] == "MIN_POSITION_SIZE"

    def test_min_position_size_zero_weight_not_checked(self):
        """Zero-weight assets are excluded from minimum position check."""
        weights = {"A": 0.50, "B": 0.50, "C": 0.00}
        violations, warnings = _check_min_position_size(weights, 0.02, "soft")
        assert len(warnings) == 0  # C is zero — not a violation


# ── Tests: check_compliance (full tool) ──────────────────────────────────────

class TestCheckCompliance:

    def test_passing_portfolio_returns_passed_true(self):
        """Well-diversified portfolio passes all retail_conservative rules."""
        result = check_compliance(
            weights=PASSING_WEIGHTS,
            sector_map=SECTOR_MAP,
            var_95=VAR_95,
            cvar_95=CVAR_95,
            rules_profile=PROFILE,
            rules_version=VERSION,
        )
        assert result["passed"] is True
        assert len(result["violations"]) == 0

    def test_output_has_all_required_keys(self):
        """Output has all keys matching ComplianceResult schema."""
        result = check_compliance(
            weights=PASSING_WEIGHTS,
            sector_map=SECTOR_MAP,
            var_95=VAR_95,
            cvar_95=CVAR_95,
        )
        required = {"passed", "violations", "warnings",
                    "rules_version", "rules_profile"}
        assert required.issubset(set(result.keys()))

    def test_rules_version_echoed(self):
        """rules_version is echoed back in response."""
        result = check_compliance(
            weights=PASSING_WEIGHTS,
            sector_map=SECTOR_MAP,
            var_95=VAR_95,
            cvar_95=CVAR_95,
            rules_version="v1.0",
        )
        assert result["rules_version"] == "v1.0"

    def test_rules_profile_echoed(self):
        """rules_profile is echoed back in response."""
        result = check_compliance(
            weights=PASSING_WEIGHTS,
            sector_map=SECTOR_MAP,
            var_95=VAR_95,
            cvar_95=CVAR_95,
            rules_profile="retail_conservative",
        )
        assert result["rules_profile"] == "retail_conservative"

    def test_single_asset_violation_fails(self):
        """Portfolio with 40% in one asset fails SINGLE_ASSET_CAP."""
        weights = {
            "RELIANCE.NS":  0.40,
            "TCS.NS":       0.20,
            "INFY.NS":      0.20,
            "HDFCBANK.NS":  0.10,
            "ICICIBANK.NS": 0.10,
        }
        result = check_compliance(
            weights=weights,
            sector_map=SECTOR_MAP,
            var_95=VAR_95,
            cvar_95=CVAR_95,
        )
        assert result["passed"] is False
        rule_ids = [v["rule_id"] for v in result["violations"]]
        assert "SINGLE_ASSET_CAP" in rule_ids

    def test_sector_violation_fails(self):
        """
        Portfolio with 60% in Financial Services fails SECTOR_CAP.
        HDFCBANK + ICICIBANK + BAJFINANCE = 0.60 > 0.40 limit.
        """
        weights = {
            "HDFCBANK.NS":   0.25,
            "ICICIBANK.NS":  0.20,
            "BAJFINANCE.NS": 0.15,
            "TCS.NS":        0.20,
            "INFY.NS":       0.20,
        }
        sector_map = {
            "HDFCBANK.NS":   "Financial Services",
            "ICICIBANK.NS":  "Financial Services",
            "BAJFINANCE.NS": "Financial Services",
            "TCS.NS":        "Technology",
            "INFY.NS":       "Technology",
        }
        result = check_compliance(
            weights=weights,
            sector_map=sector_map,
            var_95=VAR_95,
            cvar_95=CVAR_95,
        )
        assert result["passed"] is False
        rule_ids = [v["rule_id"] for v in result["violations"]]
        assert "SECTOR_CAP" in rule_ids

    def test_cvar_violation_fails(self):
        """CVaR of 0.30 exceeds 0.25 retail_conservative limit."""
        result = check_compliance(
            weights=PASSING_WEIGHTS,
            sector_map=SECTOR_MAP,
            var_95=0.10,
            cvar_95=0.30,  # exceeds 0.25 limit
        )
        assert result["passed"] is False
        rule_ids = [v["rule_id"] for v in result["violations"]]
        assert "CVAR_THRESHOLD" in rule_ids

    def test_soft_violation_does_not_fail(self):
        """
        Portfolio with a token position (ICICIBANK.NS at 0.01) triggers
        MIN_POSITION_SIZE warning but passed remains True (soft rule).
        All hard rules pass: no single asset > 0.30, no sector > 0.40,
        CVaR well below threshold.
        """
        weights = {
            "RELIANCE.NS":   0.29,  # Energy: 0.29
            "TCS.NS":        0.29,  # Technology: 0.29
            "BHARTIARTL.NS": 0.29,  # Communication Services: 0.29
            "HDFCBANK.NS":   0.12,  # Financial Services: 0.13
            "ICICIBANK.NS":  0.01,  # Financial Services: 0.13 (token — triggers warning)
        }
        sector_map = {
            "RELIANCE.NS":   "Energy",
            "TCS.NS":        "Technology",
            "BHARTIARTL.NS": "Communication Services",
            "HDFCBANK.NS":   "Financial Services",
            "ICICIBANK.NS":  "Financial Services",
        }
        result = check_compliance(
            weights=weights,
            sector_map=sector_map,
            var_95=VAR_95,
            cvar_95=CVAR_95,
        )
        assert result["passed"] is True
        warning_ids = [w["rule_id"] for w in result["warnings"]]
        assert "MIN_POSITION_SIZE" in warning_ids

    def test_multiple_violations_all_reported(self):
        """
        Portfolio violating both SINGLE_ASSET_CAP and CVAR_THRESHOLD
        reports both violations.
        """
        weights = {
            "RELIANCE.NS":  0.50,
            "TCS.NS":       0.20,
            "INFY.NS":      0.10,
            "HDFCBANK.NS":  0.10,
            "ICICIBANK.NS": 0.10,
        }
        result = check_compliance(
            weights=weights,
            sector_map=SECTOR_MAP,
            var_95=0.15,
            cvar_95=0.35,  # exceeds 0.25 limit
        )
        assert result["passed"] is False
        rule_ids = [v["rule_id"] for v in result["violations"]]
        assert "SINGLE_ASSET_CAP" in rule_ids
        assert "CVAR_THRESHOLD" in rule_ids

    def test_institutional_has_looser_limits(self):
        """
        Portfolio failing retail_conservative passes institutional.
        institutional SINGLE_ASSET_CAP = 0.40, retail = 0.30.
        """
        weights = {
            "RELIANCE.NS":  0.35,   # fails retail (>0.30), passes institutional (<0.40)
            "TCS.NS":       0.25,
            "INFY.NS":      0.20,
            "HDFCBANK.NS":  0.10,
            "ICICIBANK.NS": 0.10,
        }
        retail_result = check_compliance(
            weights=weights,
            sector_map=SECTOR_MAP,
            var_95=VAR_95,
            cvar_95=CVAR_95,
            rules_profile="retail_conservative",
        )
        institutional_result = check_compliance(
            weights=weights,
            sector_map=SECTOR_MAP,
            var_95=VAR_95,
            cvar_95=CVAR_95,
            rules_profile="institutional",
        )
        assert retail_result["passed"] is False
        assert institutional_result["passed"] is True

    def test_invalid_profile_raises_value_error(self):
        """Invalid rules_profile raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rules_profile"):
            check_compliance(
                weights=PASSING_WEIGHTS,
                sector_map=SECTOR_MAP,
                var_95=VAR_95,
                cvar_95=CVAR_95,
                rules_profile="unknown_profile",
            )

    def test_weights_not_summing_to_one_raises(self):
        """Weights not summing to 1 raises ValueError."""
        bad_weights = {"A": 0.4, "B": 0.4}  # sums to 0.8
        with pytest.raises(ValueError, match="sum to 1"):
            check_compliance(
                weights=bad_weights,
                sector_map={},
                var_95=VAR_95,
                cvar_95=CVAR_95,
            )

    def test_deterministic_same_inputs_same_result(self):
        """Same inputs always produce identical results — stateless."""
        kwargs = dict(
            weights=PASSING_WEIGHTS,
            sector_map=SECTOR_MAP,
            var_95=VAR_95,
            cvar_95=CVAR_95,
        )
        result1 = check_compliance(**kwargs)
        result2 = check_compliance(**kwargs)
        assert result1 == result2