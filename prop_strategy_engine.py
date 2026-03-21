from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineConfig:
    n_sims: int = 50_000
    max_steps_cap: int = 1_200  # keep app responsive at 250k simulations


def _prepare_r_ratio(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Compute r_ratio = rPnL / BaselineRisk where:
      - BaselineRisk is the median of absolute values of all negative rPnL trades.
      - r_ratio preserves sign (profits are +R, losses are -R).

    Returns:
      r_ratio_arr: float64 array aligned with rows
      day_codes: int64 array of calendar-day codes aligned with rows
      baseline_risk: float baseline risk ($1R)
      n_trades: number of trade rows
    """
    if "rPnL" not in df.columns or "dateStart" not in df.columns:
        missing = {"rPnL", "dateStart"} - set(df.columns)
        raise ValueError(f"CSV must include columns: {sorted(missing)}")

    rpnls = pd.to_numeric(df["rPnL"], errors="coerce")
    if rpnls.isna().any():
        bad = int(rpnls.isna().sum())
        raise ValueError(f"Column 'rPnL' has {bad} invalid rows (non-numeric).")

    dt = pd.to_datetime(df["dateStart"], errors="coerce", utc=False)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Column 'dateStart' has {bad} invalid timestamps.")

    neg_abs = (-rpnls[rpnls < 0.0]).astype(float).abs()
    if len(neg_abs) == 0:
        raise ValueError("No negative trades found in 'rPnL'; cannot compute $1R baseline risk.")

    baseline_risk = float(np.median(neg_abs.to_numpy(dtype=float)))
    if not np.isfinite(baseline_risk) or baseline_risk <= 0:
        raise ValueError("Computed $1R baseline risk is non-positive; check negative trades in 'rPnL'.")

    r_ratio_arr = (rpnls.to_numpy(dtype=float) / baseline_risk).astype(float, copy=False)

    # Calendar day codes
    day_vals = dt.dt.normalize()
    day_codes = pd.factorize(day_vals, sort=False)[0].astype(np.int64, copy=False)

    return r_ratio_arr, day_codes, baseline_risk, int(len(df))


def run_prop_strategy_monte_carlo(
    *,
    df: pd.DataFrame,
    starting_balance: float,
    risk_per_trade_dollar: float,
    profit_target_dollar: float,
    max_loss_dollar: float,
    daily_loss_dollar: float,
    drawdown_type: Literal["Static", "Trailing"],
    challenge_fee: float,
    consistency_rule_pct: float,
    seed: int = 12345,
    config: EngineConfig = EngineConfig(),
    equity_paths: int = 50,
) -> Dict[str, object]:
    """
    Vectorized Monte Carlo:
      - 10,000 simulations hardcoded via EngineConfig
      - sample trades with replacement from r_ratio pool
      - $Risk = current equity * (risk_pct_per_trade/100)
      - PnL = $Risk * r_ratio_sample
      - Daily loss limit: track cumulative daily PnL per calendar day; reset on day change
      - Success: equity reaches profit target before loss limits
      - Failure: daily loss OR max total loss breached

    Returns plot-ready equity curves (50 random paths) + scalar metrics.
    """
    if starting_balance <= 0:
        raise ValueError("Account Size must be > 0.")
    if risk_per_trade_dollar < 0:
        raise ValueError("Risk per trade must be >= 0.")
    if profit_target_dollar < 0 or max_loss_dollar < 0 or daily_loss_dollar < 0:
        raise ValueError("Target, Max Loss, and Daily Loss must be >= 0.")
    if drawdown_type not in {"Static", "Trailing"}:
        raise ValueError("drawdown_type must be 'Static' or 'Trailing'.")
    if challenge_fee < 0:
        raise ValueError("Challenge Fee must be >= 0.")
    if consistency_rule_pct < 0:
        raise ValueError("Consistency Rule % must be >= 0.")
    if equity_paths <= 0:
        raise ValueError("equity_paths must be positive.")

    r_ratio_arr, day_codes, _, n_trades = _prepare_r_ratio(df)
    if n_trades < 1:
        raise ValueError("CSV has no rows.")

    n_sims = int(config.n_sims)
    max_steps = int(min(n_trades, config.max_steps_cap))
    if max_steps < 1:
        raise ValueError("Not enough data to simulate.")

    rng = np.random.default_rng(seed)

    target_profit_dollars = float(profit_target_dollar)
    max_loss_dollars = float(max_loss_dollar)
    daily_loss_dollars_static = float(daily_loss_dollar)
    target_balance = starting_balance + target_profit_dollars
    max_total_loss_balance_static = starting_balance - max_loss_dollars
    risk_dollars_fixed = float(risk_per_trade_dollar)
    consistency_day_cap = (consistency_rule_pct / 100.0) * target_profit_dollars

    # Simulation state
    equity = np.full(n_sims, float(starting_balance), dtype=float)
    high_water = np.full(n_sims, float(starting_balance), dtype=float)
    outcome = np.full(n_sims, -1, dtype=np.int8)  # -1 ongoing; 1 success; 0 fail
    current_day = np.full(n_sims, -1, dtype=np.int64)
    day_start_equity = np.full(n_sims, float(starting_balance), dtype=float)
    daily_pnl = np.zeros(n_sims, dtype=float)
    trades_taken = np.zeros(n_sims, dtype=np.int32)

    # Plot sample
    n_paths = int(min(max(1, equity_paths), n_sims))
    plot_idx = rng.choice(n_sims, size=n_paths, replace=False)
    plot_done = np.zeros(n_paths, dtype=bool)
    plot_equity = np.full((n_paths, max_steps + 1), np.nan, dtype=float)
    plot_equity[:, 0] = float(starting_balance)
    # For labeling:
    plot_outcome = np.full(n_paths, -2, dtype=np.int8)

    # Main Monte Carlo loop (time is steps, vectorization is over sims)
    trade_pool_len = n_trades
    for step in range(max_steps):
        active = outcome == -1
        if not np.any(active):
            break

        # Sample trade indices for all sims.
        sampled_idx = rng.integers(0, trade_pool_len, size=n_sims)
        sampled_r = r_ratio_arr[sampled_idx]
        sampled_day = day_codes[sampled_idx]

        # Day transitions for active sims
        transitions = active & (sampled_day != current_day)
        if np.any(transitions):
            current_day[transitions] = sampled_day[transitions]
            day_start_equity[transitions] = equity[transitions]
            daily_pnl[transitions] = 0.0

        # Apply trade PnL to active sims
        base_risk_dollars = np.full(n_sims, risk_dollars_fixed, dtype=float)
        pnl = base_risk_dollars * sampled_r

        # Update only active
        equity[active] += pnl[active]
        daily_pnl[active] += pnl[active]
        trades_taken[active] += 1
        np.maximum(high_water, equity, out=high_water)

        # Check success first (success wins over fail on same trade)
        success_mask = active & (equity >= target_balance)
        if np.any(success_mask):
            outcome[success_mask] = 1

        # Fail conditions
        fail_daily_mask = active & (daily_pnl <= -daily_loss_dollars_static)

        if drawdown_type == "Static":
            max_floor = max_total_loss_balance_static
        else:
            max_floor = high_water - max_loss_dollars

        fail_max_mask = active & (equity <= max_floor)
        fail_consistency_mask = active & (daily_pnl > consistency_day_cap)
        fail_mask = (fail_daily_mask | fail_max_mask | fail_consistency_mask) & (~success_mask)
        if np.any(fail_mask):
            outcome[fail_mask] = 0

        # Record plot equity for paths still unresolved at the start of this step.
        if n_paths > 0:
            plot_active = ~plot_done
            if np.any(plot_active):
                record_mask = plot_active
                plot_equity[record_mask, step + 1] = equity[plot_idx[record_mask]]

            # Mark plot_done after checks
            resolved_now = outcome[plot_idx] != -1
            plot_done |= resolved_now
            plot_outcome[resolved_now] = outcome[plot_idx[resolved_now]]

    # Unresolved at max_steps are treated as FAIL (conservative for pass rate).
    unresolved = outcome == -1
    if np.any(unresolved):
        outcome[unresolved] = 0

    passed_mask = outcome == 1
    pass_rate = float(passed_mask.mean())
    ruin_rate = float((outcome == 0).mean())
    avg_trades_to_pass = float(np.mean(trades_taken[passed_mask])) if np.any(passed_mask) else float("nan")

    efficiency = float("inf") if pass_rate <= 0 else float(challenge_fee) / pass_rate

    target_drawdown_balance = float(max_total_loss_balance_static)
    if drawdown_type == "Trailing":
        target_drawdown_balance = float(starting_balance - max_loss_dollars)

    # Build list of plot paths for Plotly (slice until last non-nan)
    passed_paths = np.array([bool(o == 1) for o in plot_outcome], dtype=bool)
    paths_list = []
    for i in range(n_paths):
        row = plot_equity[i]
        if np.all(np.isnan(row)):
            paths_list.append(np.array([starting_balance], dtype=float))
            continue
        valid_idx = np.where(~np.isnan(row))[0]
        last_valid = int(valid_idx.max()) if valid_idx.size else 0
        paths_list.append(row[: last_valid + 1].copy())

    return {
        "pass_rate": pass_rate,
        "ruin_rate": ruin_rate,
        "avg_trades_to_pass": avg_trades_to_pass,
        "efficiency_total_cost_to_fund": efficiency,
        "target_balance": float(target_balance),
        "max_drawdown_balance": target_drawdown_balance,
        "plot_paths_equity": paths_list,
        "plot_paths_passed": passed_paths.tolist(),
        "max_steps": max_steps,
        "n_sims": n_sims,
    }


def run_firm_comparison_batch(
    *,
    df: pd.DataFrame,
    firms: List[Dict[str, Any]],
    n_sims: int = 10_000,
    seed: int = 42,
    max_steps_cap: int = 700,
) -> pd.DataFrame:
    """
    Vectorized multi-firm Monte Carlo: shape (n_firms, n_sims) per step.
    Each firm dict must include: name, starting_equity, profit_target, max_loss, daily_loss,
    drawdown_type ('Static'|'Trailing'), challenge_fee, consistency_rule_pct.
    Optional: monthly_recurring (bool), risk_per_trade_dollar (float).

    Returns a DataFrame with pass_rate, avg_days_to_pass, budget_95, etc.
    """
    if not firms:
        raise ValueError("firms list is empty.")
    n_sims = int(n_sims)
    if n_sims < 1:
        raise ValueError("n_sims must be positive.")

    r_ratio_arr, day_codes, _, n_trades = _prepare_r_ratio(df)
    if n_trades < 1:
        raise ValueError("CSV has no rows.")

    max_steps = int(min(n_trades, max_steps_cap))
    if max_steps < 1:
        raise ValueError("Not enough data to simulate.")

    F = len(firms)
    starting = np.array([float(f["starting_equity"]) for f in firms], dtype=float)
    profit_target = np.array([float(f["profit_target"]) for f in firms], dtype=float)
    max_loss = np.array([float(f["max_loss"]) for f in firms], dtype=float)
    daily_loss = np.array([float(f["daily_loss"]) for f in firms], dtype=float)
    consistency_pct = np.array([float(f["consistency_rule_pct"]) for f in firms], dtype=float)
    challenge_fee = np.array([float(f["challenge_fee"]) for f in firms], dtype=float)
    trailing = np.array([f.get("drawdown_type", "Static") == "Trailing" for f in firms], dtype=bool)

    risk = np.array(
        [
            float(
                f["risk_per_trade_dollar"]
                if f.get("risk_per_trade_dollar") is not None
                else max(1.0, int(0.005 * float(f["starting_equity"])))
            )
            for f in firms
        ],
        dtype=float,
    )

    target_balance = starting + profit_target
    max_static_floor = starting - max_loss
    consistency_cap = (consistency_pct / 100.0) * profit_target

    rng = np.random.default_rng(seed)
    S = n_sims

    equity = np.tile(starting[:, np.newaxis], (1, S))
    high_water = equity.copy()
    outcome = np.full((F, S), -1, dtype=np.int8)
    current_day = np.full((F, S), -1, dtype=np.int64)
    day_start_equity = np.tile(starting[:, np.newaxis], (1, S))
    daily_pnl = np.zeros((F, S), dtype=float)
    trades_taken = np.zeros((F, S), dtype=np.int32)

    target_bal = target_balance[:, np.newaxis]
    daily_lim = daily_loss[:, np.newaxis]
    cons_lim = consistency_cap[:, np.newaxis]
    max_loss_b = max_loss[:, np.newaxis]
    static_floor = max_static_floor[:, np.newaxis]

    trade_pool_len = n_trades
    for _step in range(max_steps):
        active = outcome == -1
        if not np.any(active):
            break

        sampled_idx = rng.integers(0, trade_pool_len, size=(F, S))
        sampled_r = r_ratio_arr[sampled_idx]
        sampled_day = day_codes[sampled_idx]

        transitions = active & (sampled_day != current_day)
        current_day[transitions] = sampled_day[transitions]
        day_start_equity[transitions] = equity[transitions]
        daily_pnl[transitions] = 0.0

        pnl = risk[:, np.newaxis] * sampled_r
        equity[active] = equity[active] + pnl[active]
        daily_pnl[active] = daily_pnl[active] + pnl[active]
        trades_taken[active] = trades_taken[active] + 1
        np.maximum(high_water, equity, out=high_water)

        success_mask = active & (equity >= target_bal)
        outcome[success_mask] = 1

        fail_daily = active & (daily_pnl <= -daily_lim)
        trail_floor = high_water - max_loss_b
        max_floor = np.where(trailing[:, np.newaxis], trail_floor, static_floor)
        fail_max = active & (equity <= max_floor)
        fail_cons = active & (daily_pnl > cons_lim)
        fail_mask = (fail_daily | fail_max | fail_cons) & (~success_mask)
        outcome[fail_mask] = 0

    unresolved = outcome == -1
    outcome[unresolved] = 0

    rows: List[Dict[str, Any]] = []
    names = [str(f["name"]) for f in firms]

    for fi in range(F):
        out = outcome[fi]
        passed = out == 1
        pr = float(np.mean(out == 1))
        rr = float(np.mean(out == 0))
        if np.any(passed):
            avg_trades = float(np.mean(trades_taken[fi, passed]))
        else:
            avg_trades = float("nan")

        fee = float(challenge_fee[fi])
        if pr <= 0:
            n95 = float("inf")
        elif pr >= 1:
            n95 = 1.0
        else:
            n95 = float(np.log(0.05) / np.log(1.0 - pr))

        rows.append(
            {
                "Firm": names[fi],
                "Pass Rate": pr,
                "Fail Rate": rr,
                "Avg Trades to Pass": avg_trades,
                "Challenge Fee": fee,
                "n95": n95,
            }
        )

    return pd.DataFrame(rows)

