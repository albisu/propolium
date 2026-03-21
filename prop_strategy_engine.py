from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineConfig:
    n_sims: int = 250_000
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
    risk_value: float,
    risk_mode: Literal["percent", "dollar"],
    target_value: float,
    max_loss_value: float,
    daily_loss_value: float,
    value_mode: Literal["percent", "dollar"],
    drawdown_type: Literal["Static", "Trailing"],
    challenge_fee: float,
    payout_split_pct: float,
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
    if risk_value < 0:
        raise ValueError("Risk per trade must be >= 0.")
    if risk_mode not in {"percent", "dollar"}:
        raise ValueError("risk_mode must be 'percent' or 'dollar'.")
    if target_value < 0 or max_loss_value < 0 or daily_loss_value < 0:
        raise ValueError("Target, Max Loss, and Daily Loss must be >= 0.")
    if value_mode not in {"percent", "dollar"}:
        raise ValueError("value_mode must be 'percent' or 'dollar'.")
    if drawdown_type not in {"Static", "Trailing"}:
        raise ValueError("drawdown_type must be 'Static' or 'Trailing'.")
    if challenge_fee < 0:
        raise ValueError("Challenge Fee must be >= 0.")
    if not (0 <= payout_split_pct <= 100):
        raise ValueError("Payout Split % must be between 0 and 100.")
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

    if value_mode == "percent":
        target_profit_dollars = starting_balance * (target_value / 100.0)
        max_loss_dollars = starting_balance * (max_loss_value / 100.0)
        daily_loss_dollars_static = starting_balance * (daily_loss_value / 100.0)
        daily_loss_is_percent = True
    else:
        target_profit_dollars = float(target_value)
        max_loss_dollars = float(max_loss_value)
        daily_loss_dollars_static = float(daily_loss_value)
        daily_loss_is_percent = False

    target_balance = starting_balance + target_profit_dollars
    max_total_loss_balance_static = starting_balance - max_loss_dollars
    risk_frac = risk_value / 100.0
    risk_dollars_fixed = float(risk_value)
    consistency_day_cap = (consistency_rule_pct / 100.0) * target_profit_dollars

    # Simulation state
    equity = np.full(n_sims, float(starting_balance), dtype=float)
    high_water = np.full(n_sims, float(starting_balance), dtype=float)
    outcome = np.full(n_sims, -1, dtype=np.int8)  # -1 ongoing; 1 success; 0 fail
    current_day = np.full(n_sims, -1, dtype=np.int64)
    day_start_equity = np.full(n_sims, float(starting_balance), dtype=float)
    daily_pnl = np.zeros(n_sims, dtype=float)

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
        if risk_mode == "percent":
            base_risk_dollars = equity * risk_frac
        else:
            base_risk_dollars = np.full(n_sims, risk_dollars_fixed, dtype=float)
        pnl = base_risk_dollars * sampled_r

        # Update only active
        equity[active] += pnl[active]
        daily_pnl[active] += pnl[active]
        np.maximum(high_water, equity, out=high_water)

        # Check success first (success wins over fail on same trade)
        success_mask = active & (equity >= target_balance)
        if np.any(success_mask):
            outcome[success_mask] = 1

        # Fail conditions
        if daily_loss_is_percent:
            fail_daily_mask = active & (daily_pnl <= -(day_start_equity * (daily_loss_value / 100.0)))
        else:
            fail_daily_mask = active & (daily_pnl <= -daily_loss_dollars_static)

        if drawdown_type == "Static":
            max_floor = max_total_loss_balance_static
        else:
            # trailing floor moves with high-water mark
            if value_mode == "percent":
                max_floor = high_water * (1.0 - max_loss_value / 100.0)
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

    # Payout model: payout split of realized profit (equity - starting_balance) on pass.
    profits_on_pass = equity[passed_mask] - float(starting_balance)
    payout_split = float(payout_split_pct) / 100.0
    payouts_pass = payout_split * profits_on_pass
    avg_payout = float(payouts_pass.mean()) if len(payouts_pass) else 0.0

    net_ev = (pass_rate * avg_payout) - (ruin_rate * float(challenge_fee))
    efficiency = float("inf") if pass_rate <= 0 else float(challenge_fee) / pass_rate

    target_drawdown_balance = float(max_total_loss_balance_static)
    if drawdown_type == "Trailing":
        # Chart marker for trailing mode: initial floor (will trail upward intrapath).
        if value_mode == "percent":
            target_drawdown_balance = float(starting_balance * (1.0 - max_loss_value / 100.0))
        else:
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
        "avg_payout": avg_payout,
        "net_ev": net_ev,
        "efficiency_total_cost_to_fund": efficiency,
        "target_balance": float(target_balance),
        "max_drawdown_balance": target_drawdown_balance,
        "plot_paths_equity": paths_list,
        "plot_paths_passed": passed_paths.tolist(),
        "max_steps": max_steps,
        "n_sims": n_sims,
    }

