"""
servers/compliance/tools/check_compliance.py

Implementation of the check_compliance tool.

Responsibilities:
    - Load versioned YAML ruleset at startup (once, held in memory)
    - Check portfolio weights and risk metrics against rule limits
    - Return pass/fail with specific violation details
    - Echo rules_version in response to close the audit loop

Design decisions:
    Stateless:
        Every request is fully self-contained.
        Ruleset is loaded at module import time — not per request.
        No database reads during request processing.
        If the server restarts, it loads the same YAML and gives
        the same answers — no recovery needed.

    Gatekeeper:
        Always the last computation step before synthesis.
        Compliance checks the user's CURRENT portfolio weights —
        not the optimal weights from the Portfolio Optimiser.
        Reason: we check what the user actually holds,
        not a hypothetical rebalanced position.

    CVaR as gating metric:
        CVaR (not VaR) gates the CVAR_THRESHOLD rule.
        CVaR captures the magnitude of tail losses.
        VaR only identifies the threshold — not how bad the tail is.

    Audit trail:
        rules_version is mandatory in every request.
        Echoed in every response.
        Same portfolio + same rules_version → same result always.
        Enables deterministic replay of historical compliance decisions.

    Rule severity:
        hard: blocking — passed=False, recommendation should not proceed
        soft: warning — passed remains True, surfaced in synthesis node

YAML ruleset location:
    servers/compliance/rules/{profile}_v{version}.yaml
    e.g. retail_conservative_v1.0.yaml

Supported rule IDs (v1):
    SINGLE_ASSET_CAP     — no single asset weight > limit
    SECTOR_CAP           — no single sector weight > limit
    CVAR_THRESHOLD       — portfolio CVaR_95 < limit
    MIN_ASSETS           — active assets (non-zero weight) >= limit
    MIN_POSITION_SIZE    — all non-zero weights >= limit

This file contains pure computation logic only.
No MCP protocol code — that lives in server.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
RULES_DIR = Path(__file__).parent.parent / "rules"

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_PROFILES = {"retail_conservative", "institutional"}
ZERO_WEIGHT_THRESHOLD = 1e-6   # weights below this are treated as zero


# ── Ruleset loader ────────────────────────────────────────────────────────────

def _load_ruleset(profile: str, version: str) -> dict:
    """
    Load a YAML ruleset from the rules directory.

    Filename convention: {profile}_v{version}.yaml
    e.g. retail_conservative_v1.0.yaml

    The ruleset is loaded per request in v1 (simple and correct).
    In production with high load, this could be cached at startup.

    Args:
        profile: str — e.g. "retail_conservative"
        version: str — e.g. "v1.0" or "1.0"

    Returns:
        dict — parsed YAML content

    Raises:
        FileNotFoundError: if ruleset file does not exist
        ValueError: if YAML is malformed or missing required fields
    """
    # Normalise version — accept "v1.0" or "1.0"
    version_normalised = version.lstrip("v")
    filename = f"{profile}_v{version_normalised}.yaml"
    filepath = RULES_DIR / filename

    if not filepath.exists():
        available = [f.name for f in RULES_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Ruleset not found: {filepath}\n"
            f"Available rulesets: {available}"
        )

    with open(filepath, "r") as f:
        ruleset = yaml.safe_load(f)

    if not ruleset or "rules" not in ruleset:
        raise ValueError(
            f"Ruleset {filename} is malformed — missing 'rules' key."
        )

    return ruleset


# ── Rule checkers ─────────────────────────────────────────────────────────────

def _check_single_asset_cap(
    weights: dict[str, float],
    limit: float,
    severity: str,
) -> tuple[list[dict], list[dict]]:
    """
    Check SINGLE_ASSET_CAP: no single asset weight exceeds limit.

    Returns:
        tuple: (violations, warnings) — lists of violation/warning dicts
    """
    violations, warnings = [], []

    for symbol, weight in weights.items():
        if weight > limit + 1e-8:
            entry = {
                "rule_id":     "SINGLE_ASSET_CAP",
                "description": (
                    f"{symbol} weight {weight:.1%} exceeds "
                    f"single asset cap of {limit:.1%}"
                ),
                "value":       weight,
                "limit":       limit,
            }
            if severity == "hard":
                entry["severity"] = "hard"
                violations.append(entry)
            else:
                warnings.append({k: v for k, v in entry.items()
                                  if k != "severity"})

    return violations, warnings


def _check_sector_cap(
    weights: dict[str, float],
    sector_map: dict[str, str],
    limit: float,
    severity: str,
) -> tuple[list[dict], list[dict]]:
    """
    Check SECTOR_CAP: no single sector weight exceeds limit.

    Sector weights are computed by summing weights of all assets
    belonging to the same sector.

    Args:
        weights:    dict[symbol, float]
        sector_map: dict[symbol, str] — from fundamentals.sector
        limit:      float
        severity:   str

    Returns:
        tuple: (violations, warnings)
    """
    violations, warnings = [], []

    # Compute sector weights
    sector_weights: dict[str, float] = {}
    for symbol, weight in weights.items():
        sector = sector_map.get(symbol, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

    for sector, sector_weight in sector_weights.items():
        if sector_weight > limit + 1e-8:
            entry = {
                "rule_id":     "SECTOR_CAP",
                "description": (
                    f"Sector '{sector}' weight {sector_weight:.1%} exceeds "
                    f"sector cap of {limit:.1%}"
                ),
                "value":       sector_weight,
                "limit":       limit,
            }
            if severity == "hard":
                entry["severity"] = "hard"
                violations.append(entry)
            else:
                warnings.append({k: v for k, v in entry.items()
                                  if k != "severity"})

    return violations, warnings


def _check_cvar_threshold(
    cvar_95: float,
    limit: float,
    severity: str,
) -> tuple[list[dict], list[dict]]:
    """
    Check CVAR_THRESHOLD: CVaR_95 must not exceed limit.

    CVaR is the primary gating metric — not VaR.
    CVaR captures the magnitude of tail losses beyond the VaR threshold.

    Args:
        cvar_95:  float — Expected Shortfall at 95% confidence
        limit:    float — maximum acceptable CVaR (e.g. 0.25 = 25% loss)
        severity: str

    Returns:
        tuple: (violations, warnings)
    """
    violations, warnings = [], []

    if cvar_95 > limit + 1e-8:
        entry = {
            "rule_id":     "CVAR_THRESHOLD",
            "description": (
                f"Portfolio CVaR_95 {cvar_95:.1%} exceeds "
                f"maximum CVaR threshold of {limit:.1%}"
            ),
            "value":       cvar_95,
            "limit":       limit,
        }
        if severity == "hard":
            entry["severity"] = "hard"
            violations.append(entry)
        else:
            warnings.append({k: v for k, v in entry.items()
                              if k != "severity"})

    return violations, warnings


def _check_min_assets(
    weights: dict[str, float],
    limit: int,
    severity: str,
) -> tuple[list[dict], list[dict]]:
    """
    Check MIN_ASSETS: portfolio must hold at least `limit` assets
    with non-zero weight.

    Active assets are those with weight > ZERO_WEIGHT_THRESHOLD.
    The Markowitz optimiser may set some weights to zero — this rule
    ensures minimum diversification.

    Args:
        weights:  dict[symbol, float]
        limit:    int — minimum number of active assets
        severity: str

    Returns:
        tuple: (violations, warnings)
    """
    violations, warnings = [], []

    active_count = sum(
        1 for w in weights.values() if w > ZERO_WEIGHT_THRESHOLD
    )

    if active_count < limit:
        entry = {
            "rule_id":     "MIN_ASSETS",
            "description": (
                f"Portfolio holds {active_count} active assets, "
                f"minimum required is {int(limit)}"
            ),
            "value":       float(active_count),
            "limit":       float(limit),
        }
        if severity == "hard":
            entry["severity"] = "hard"
            violations.append(entry)
        else:
            warnings.append({k: v for k, v in entry.items()
                              if k != "severity"})

    return violations, warnings


def _check_min_position_size(
    weights: dict[str, float],
    limit: float,
    severity: str,
) -> tuple[list[dict], list[dict]]:
    """
    Check MIN_POSITION_SIZE: all non-zero weights must be at least `limit`.

    Prevents token positions that add negligible diversification benefit.
    Only checks assets with non-zero weight — zero weights are allowed.

    Args:
        weights:  dict[symbol, float]
        limit:    float — minimum non-zero weight (e.g. 0.02 = 2%)
        severity: str

    Returns:
        tuple: (violations, warnings)
    """
    violations, warnings = [], []

    for symbol, weight in weights.items():
        if weight > ZERO_WEIGHT_THRESHOLD and weight < limit - 1e-8:
            entry = {
                "rule_id":     "MIN_POSITION_SIZE",
                "description": (
                    f"{symbol} weight {weight:.2%} is below "
                    f"minimum position size of {limit:.2%}"
                ),
                "value":       weight,
                "limit":       limit,
            }
            if severity == "hard":
                entry["severity"] = "hard"
                violations.append(entry)
            else:
                warnings.append({k: v for k, v in entry.items()
                                  if k != "severity"})

    return violations, warnings


# ── Rule dispatcher ───────────────────────────────────────────────────────────

def _apply_rule(
    rule: dict,
    weights: dict[str, float],
    sector_map: dict[str, str],
    cvar_95: float,
) -> tuple[list[dict], list[dict]]:
    """
    Dispatch a single rule to the appropriate checker function.

    Args:
        rule:       dict — one rule entry from the YAML ruleset
        weights:    dict[symbol, float]
        sector_map: dict[symbol, str]
        cvar_95:    float

    Returns:
        tuple: (violations, warnings)
    """
    rule_id  = rule["id"]
    limit    = rule["limit"]
    severity = rule["severity"]

    if rule_id == "SINGLE_ASSET_CAP":
        return _check_single_asset_cap(weights, limit, severity)

    elif rule_id == "SECTOR_CAP":
        return _check_sector_cap(weights, sector_map, limit, severity)

    elif rule_id == "CVAR_THRESHOLD":
        return _check_cvar_threshold(cvar_95, limit, severity)

    elif rule_id == "MIN_ASSETS":
        return _check_min_assets(weights, limit, severity)

    elif rule_id == "MIN_POSITION_SIZE":
        return _check_min_position_size(weights, limit, severity)

    else:
        # Unknown rule ID — skip silently with a warning
        return [], [{
            "rule_id":     rule_id,
            "description": f"Unknown rule ID '{rule_id}' — skipped",
            "value":       0.0,
        }]


# ── Main tool function ────────────────────────────────────────────────────────

def check_compliance(
    weights: dict[str, float],
    sector_map: dict[str, str],
    var_95: float,
    cvar_95: float,
    rules_profile: str = "retail_conservative",
    rules_version: str = "v1.0",
) -> dict:
    """
    Validate portfolio allocation against a versioned compliance ruleset.

    This is the implementation of the check_compliance MCP tool.
    Called by server.py — never called directly by other servers.

    Always the last computation step before synthesis.
    Checks the user's CURRENT portfolio weights — not optimal weights.

    Args:
        weights:       dict[symbol, float] — current portfolio weights
        sector_map:    dict[symbol, str]   — symbol → sector
                       from MarketDataResult.fundamentals[symbol].sector
        var_95:        float — Historical VaR at 95% (context only, not gating)
        cvar_95:       float — CVaR at 95% (primary gating metric)
        rules_profile: str  — "retail_conservative" | "institutional"
        rules_version: str  — e.g. "v1.0" — mandatory for audit trail

    Returns:
        dict matching ComplianceResult schema in orchestrator/state.py:
            passed:        bool   — True if no hard violations
            violations:    list   — hard violations with rule_id, description,
                                    severity, value, limit
            warnings:      list   — soft violations with rule_id, description, value
            rules_version: str    — echoed for audit trail
            rules_profile: str    — echoed for audit trail

    Raises:
        ValueError: if weights do not sum to 1
        ValueError: if rules_profile is invalid
        FileNotFoundError: if ruleset file does not exist
    """
    # ── Input validation ──────────────────────────────────────────
    if not weights:
        raise ValueError("weights dict must not be empty")

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-4:
        raise ValueError(
            f"Weights must sum to 1.0, got {weight_sum:.6f}"
        )

    if rules_profile not in VALID_PROFILES:
        raise ValueError(
            f"Invalid rules_profile '{rules_profile}'. "
            f"Must be one of: {VALID_PROFILES}"
        )

    if cvar_95 < 0:
        raise ValueError(
            f"cvar_95 must be non-negative (expressed as positive loss), "
            f"got {cvar_95}"
        )

    # ── Load ruleset ──────────────────────────────────────────────
    ruleset = _load_ruleset(rules_profile, rules_version)

    # ── Apply each rule ───────────────────────────────────────────
    all_violations: list[dict] = []
    all_warnings:   list[dict] = []

    for rule in ruleset["rules"]:
        v, w = _apply_rule(rule, weights, sector_map, cvar_95)
        all_violations.extend(v)
        all_warnings.extend(w)

    # ── Determine pass/fail ───────────────────────────────────────
    # passed = True only if no hard violations
    # Soft violations (warnings) do not affect passed status
    passed = len(all_violations) == 0

    return {
        "passed":        passed,
        "violations":    all_violations,
        "warnings":      all_warnings,
        "rules_version": rules_version,
        "rules_profile": rules_profile,
    }