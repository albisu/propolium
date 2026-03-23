from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# Resampling: how the trade pool is built before each Monte Carlo realization.
# - step_iid: legacy — pool = full history; each step draws a random trade with replacement (default).
# - pool_bootstrap_iid: one nonparametric bootstrap resample of rows (length n) for the pool; then step_iid sampling.
# - pool_bootstrap_block: concatenate random contiguous blocks to length n; then step_iid sampling.
ResampleMode = Literal["step_iid", "pool_bootstrap_iid", "pool_bootstrap_block"]


def _winsorize_r_ratios(r_ratio_arr: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Clip r_ratio values to [low_pct, high_pct] empirical percentiles (inclusive)."""
    x = np.asarray(r_ratio_arr, dtype=float).copy()
    if x.size == 0:
        return x
    lo = float(np.percentile(x, low_pct))
    hi = float(np.percentile(x, high_pct))
    if lo > hi:
        lo, hi = hi, lo
    return np.clip(x, lo, hi)


def _build_pool_bootstrap_iid(
    r_ratio_arr: np.ndarray, day_codes: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(len(r_ratio_arr))
    idx = rng.integers(0, n, size=n, endpoint=False)
    return r_ratio_arr[idx].copy(), day_codes[idx].copy()


def _build_pool_bootstrap_block(
    r_ratio_arr: np.ndarray, day_codes: np.ndarray, block_size: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate random contiguous blocks until length n (truncate). Preserves short-range structure in the pool."""
    n = int(len(r_ratio_arr))
    r_ratio_arr = np.asarray(r_ratio_arr, dtype=float)
    day_codes = np.asarray(day_codes, dtype=np.int64)
    bs = max(1, int(block_size))
    if n <= 1 or bs >= n:
        return _build_pool_bootstrap_iid(r_ratio_arr, day_codes, rng)

    out_r: List[float] = []
    out_d: List[int] = []
    max_start = n - bs
    while len(out_r) < n:
        start = int(rng.integers(0, max_start + 1, endpoint=False))
        out_r.extend(r_ratio_arr[start : start + bs].tolist())
        out_d.extend(day_codes[start : start + bs].tolist())
    return np.asarray(out_r[:n], dtype=float), np.asarray(out_d[:n], dtype=np.int64)


def _apply_resample_mode(
    r_ratio_arr: np.ndarray,
    day_codes: np.ndarray,
    mode: ResampleMode,
    block_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "step_iid":
        return r_ratio_arr, day_codes
    if mode == "pool_bootstrap_iid":
        return _build_pool_bootstrap_iid(r_ratio_arr, day_codes, rng)
    if mode == "pool_bootstrap_block":
        return _build_pool_bootstrap_block(r_ratio_arr, day_codes, block_size, rng)
    raise ValueError(f"Unknown resample_mode: {mode!r}")


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


ExpressFundedPath = Literal["standard", "consistency"]


def _run_funded_phase_vectorized(
    rng: np.random.Generator,
    r_ratio_arr: np.ndarray,
    day_codes: np.ndarray,
    *,
    n_sims: int,
    funded_start_balance: float,
    risk_dollars_funded: float,
    max_loss_dollars: float,
    daily_loss_dollars: float,
    drawdown_type: Literal["Static", "Trailing"],
    passed_mask: np.ndarray,
    min_payout_buffer: float,
    profit_split_pct: float,
    funded_consistency_max_pct: float,
    winning_day_profit_threshold: float,
    min_winning_days: int,
    min_trading_days_for_payout: int,
    max_steps: int,
    max_payout_cap_dollars: float,
    max_payout_frac_of_equity: float = 0.5,
    apply_consistency_gate: bool = True,
    payout_withdrawal_request_model: bool = False,
    express_funded_path: Optional[ExpressFundedPath] = None,
    min_consistency_calendar_days: int = 3,
    payout_processing_fee_dollars: float = 0.0,
    min_payout_request_dollars: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Funded account: no profit target; exit on daily loss or max loss.
    After first payout, max loss floor locks to funded start balance (cannot dip below start).

    Legacy mode (``express_funded_path is None``): optional buffer, cumulative winning days,
    and optional consistency on *lifetime* funded profit (simplified).

    Topstep Express mode (``express_funded_path`` is ``\"standard\"`` or ``\"consistency\"``):
    models repeat payout *cycles* — counters reset after each payout; uses withdrawal request
    (min of % of balance and cap); optional ACH/Wire-style processing fee (deducted from
    trader share); minimum request size gate. Standard: N winning days + profit since last payout
    (first payout: profit vs funded start only). Consistency: M calendar days in cycle +
    largest-day <= cap × cycle net profit, plus profit since last payout.
    """
    n = int(n_sims)
    active = passed_mask.astype(bool).copy()
    if not np.any(active):
        z = np.zeros(n, dtype=float)
        return {
            "total_payout": z.copy(),
            "had_payout": np.zeros(n, dtype=bool),
            "funded_calendar_days": z.copy(),
            "funded_survival_days": z.copy(),
            "funded_failed": np.zeros(n, dtype=bool),
        }

    start = float(funded_start_balance)
    equity = np.full(n, np.nan, dtype=float)
    equity[active] = start
    high_water = np.full(n, np.nan, dtype=float)
    high_water[active] = start
    current_day = np.full(n, -1, dtype=np.int64)
    day_start_equity = np.full(n, np.nan, dtype=float)
    day_start_equity[active] = start
    daily_pnl = np.zeros(n, dtype=float)

    payout_lock = np.zeros(n, dtype=bool)
    total_payout = np.zeros(n, dtype=float)
    had_payout = np.zeros(n, dtype=bool)
    winning_days = np.zeros(n, dtype=np.int32)
    max_daily_profit = np.zeros(n, dtype=float)
    funded_calendar_days = np.zeros(n, dtype=np.float32)
    funded_failed = np.zeros(n, dtype=bool)

    use_express = express_funded_path is not None
    if use_express:
        # Per-cycle state for Topstep Express (resets after each simulated payout).
        payout_count = np.zeros(n, dtype=np.int32)
        equity_ref_after_last = np.full(n, np.nan, dtype=float)
        cycle_winning_days = np.zeros(n, dtype=np.int32)
        cycle_calendar_days = np.zeros(n, dtype=np.int32)
        cycle_net = np.zeros(n, dtype=float)
        cycle_max_daily = np.full(n, -np.inf, dtype=float)
        wd_model = True
    else:
        wd_model = bool(payout_withdrawal_request_model)

    static_floor_val = float(start - max_loss_dollars)
    daily_lim = float(daily_loss_dollars)
    split_frac = float(profit_split_pct) / 100.0
    cons_cap = float(funded_consistency_max_pct) / 100.0
    trailing = drawdown_type == "Trailing"
    proc_fee = float(payout_processing_fee_dollars)
    min_req = float(min_payout_request_dollars)
    min_cons_days = max(0, int(min_consistency_calendar_days))

    trade_pool_len = int(len(r_ratio_arr))

    for _step in range(max_steps):
        if not np.any(active):
            break

        sampled_idx = rng.integers(0, trade_pool_len, size=n)
        sampled_r = r_ratio_arr[sampled_idx]
        sampled_day = day_codes[sampled_idx]

        transitions = active & (sampled_day != current_day)
        if np.any(transitions):
            old_close_mask = transitions & (current_day >= 0)
            if np.any(old_close_mask):
                closing = daily_pnl[old_close_mask]
                funded_calendar_days[old_close_mask] += 1.0
                if use_express:
                    cycle_net[old_close_mask] += closing
                    cycle_max_daily[old_close_mask] = np.maximum(
                        cycle_max_daily[old_close_mask], closing
                    )
                    cycle_calendar_days[old_close_mask] += 1
                    if express_funded_path == "standard":
                        cycle_winning_days[old_close_mask] += (
                            closing > winning_day_profit_threshold
                        ).astype(np.int32)
                else:
                    winning_days[old_close_mask] += (
                        closing > winning_day_profit_threshold
                    ).astype(np.int32)
                    max_daily_profit[old_close_mask] = np.maximum(
                        max_daily_profit[old_close_mask], closing
                    )

            current_day[transitions] = sampled_day[transitions]
            day_start_equity[transitions] = equity[transitions]
            daily_pnl[transitions] = 0.0

        pnl = risk_dollars_funded * sampled_r
        equity[active] = equity[active] + pnl[active]
        daily_pnl[active] = daily_pnl[active] + pnl[active]
        np.maximum(high_water, equity, out=high_water)

        fail_daily = active & (daily_pnl <= -daily_lim)

        trail_floor = high_water - max_loss_dollars
        max_floor = np.where(trailing, trail_floor, static_floor_val)
        max_floor = np.where(payout_lock, start, max_floor)
        fail_max = active & (equity <= max_floor)

        fail_mask = fail_daily | fail_max
        if np.any(fail_mask):
            funded_failed[fail_mask] = True
            active[fail_mask] = False

        if np.any(active):
            profit = equity - start

            if use_express:
                first_cycle = payout_count == 0
                profit_since_ok = first_cycle & (equity > start + 1e-9)
                profit_since_ok |= (~first_cycle) & (equity > equity_ref_after_last + 1e-9)

                if express_funded_path == "standard":
                    gate_ok = (
                        profit_since_ok
                        & (cycle_winning_days >= int(min_winning_days))
                        & (equity > start + 1e-9)
                    )
                else:
                    # Consistency path: calendar days in current cycle + 40% style rule on cycle PnL.
                    cons_ratio_ok = (cycle_net > 1e-9) & (
                        cycle_max_daily <= cons_cap * cycle_net + 1e-9
                    )
                    gate_ok = (
                        profit_since_ok
                        & (cycle_calendar_days >= min_cons_days)
                        & cons_ratio_ok
                        & (equity > start + 1e-9)
                    )

                withdrawal = np.minimum(
                    float(max_payout_frac_of_equity) * equity,
                    float(max_payout_cap_dollars),
                )
                gross_trader = withdrawal * split_frac
                user_share_arr = gross_trader - proc_fee
                size_ok = withdrawal >= min_req - 1e-9
                share_ok = user_share_arr > 1e-9
                payout_ok = active & gate_ok & size_ok & share_ok
            else:
                ratio = max_daily_profit / np.maximum(profit, 1e-9)
                if apply_consistency_gate:
                    if payout_withdrawal_request_model:
                        consistency_ok = ratio <= cons_cap + 1e-12
                    else:
                        consistency_ok = ratio < cons_cap
                else:
                    consistency_ok = np.ones(n, dtype=bool)
                buffer_ok = equity >= (start + min_payout_buffer)
                if min_winning_days > 0:
                    win_ok = winning_days >= int(min_winning_days)
                else:
                    win_ok = np.ones(n, dtype=bool)
                if min_trading_days_for_payout > 0:
                    trade_ok = funded_calendar_days >= float(min_trading_days_for_payout)
                else:
                    trade_ok = np.ones(n, dtype=bool)
                payout_ok = (
                    active
                    & buffer_ok
                    & win_ok
                    & trade_ok
                    & consistency_ok
                    & (profit > 0.0)
                )

            if np.any(payout_ok):
                if wd_model:
                    eq_sub = equity[payout_ok]
                    withdrawal = np.minimum(
                        float(max_payout_frac_of_equity) * eq_sub,
                        float(max_payout_cap_dollars),
                    )
                    if use_express:
                        user_share = withdrawal * split_frac - proc_fee
                    else:
                        user_share = withdrawal * split_frac
                    total_payout[payout_ok] += user_share
                    had_payout[payout_ok] = True
                    equity[payout_ok] = eq_sub - withdrawal
                    np.maximum(high_water, equity, out=high_water)
                    payout_lock[payout_ok] = True
                    daily_pnl[payout_ok] = 0.0
                    if use_express:
                        payout_count[payout_ok] += 1
                        equity_ref_after_last[payout_ok] = equity[payout_ok]
                        cycle_winning_days[payout_ok] = 0
                        cycle_calendar_days[payout_ok] = 0
                        cycle_net[payout_ok] = 0.0
                        cycle_max_daily[payout_ok] = -np.inf
                    hit_floor = payout_ok & (equity <= start)
                    if np.any(hit_floor):
                        funded_failed[hit_floor] = True
                        active[hit_floor] = False
                else:
                    trader_gross = split_frac * profit[payout_ok]
                    cap_raw = np.minimum(
                        float(max_payout_cap_dollars),
                        float(max_payout_frac_of_equity) * equity[payout_ok],
                    )
                    withdrawal = np.minimum(trader_gross, cap_raw)
                    total_payout[payout_ok] += withdrawal
                    had_payout[payout_ok] = True
                    equity[payout_ok] = start
                    high_water[payout_ok] = start
                    payout_lock[payout_ok] = True
                    daily_pnl[payout_ok] = 0.0

    funded_survival_days = np.zeros(n, dtype=float)
    funded_survival_days[passed_mask] = funded_calendar_days[passed_mask].astype(float)

    return {
        "total_payout": total_payout,
        "had_payout": had_payout,
        "funded_calendar_days": funded_calendar_days.astype(float),
        "funded_survival_days": funded_survival_days,
        "funded_failed": funded_failed,
    }


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
    funded_params: Optional[Dict[str, Any]] = None,
    funded_risk_per_trade_dollar: Optional[float] = None,
    winsorize_percentiles: Optional[Tuple[float, float]] = None,
    resample_mode: ResampleMode = "step_iid",
    block_size: int = 10,
    bootstrap_outer_replicates: int = 0,
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

    Optional data / uncertainty layers (applied to prepared r_ratio series, in order):
      - winsorize_percentiles: clip r_ratio to empirical percentiles, e.g. (1.0, 99.0).
      - resample_mode: ``step_iid`` (default) uses the full history as the pool; ``pool_bootstrap_iid``
        replaces the pool once with n rows sampled with replacement; ``pool_bootstrap_block`` builds
        the pool by concatenating random contiguous blocks (dependence is still broken by per-step
        random index sampling unless you model sequential paths separately).
      - bootstrap_outer_replicates: if > 0, repeat the full Monte Carlo that many times with
        independent RNG streams (and new pool draws when pool bootstrap modes are on). Adds
        ``*_bootstrap_p*`` percentile keys for pass_rate and net_ev (when funded).

    Returns plot-ready equity curves (50 random paths) + scalar metrics.
    """
    if starting_balance <= 0:
        raise ValueError("Account Size must be > 0.")
    if risk_per_trade_dollar < 0:
        raise ValueError("Risk per trade must be >= 0.")
    if funded_risk_per_trade_dollar is not None and funded_risk_per_trade_dollar < 0:
        raise ValueError("Funded risk per trade must be >= 0 when set.")
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
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if winsorize_percentiles is not None:
        lo_w, hi_w = winsorize_percentiles
        if not (0.0 <= lo_w < hi_w <= 100.0):
            raise ValueError("winsorize_percentiles must be (low, high) with 0 <= low < high <= 100.")

    r_base, day_base, _, n_trades = _prepare_r_ratio(df)
    if n_trades < 1:
        raise ValueError("CSV has no rows.")

    r_base = np.asarray(r_base, dtype=float).copy()
    day_base = np.asarray(day_base, dtype=np.int64).copy()
    if winsorize_percentiles is not None:
        lo_w, hi_w = winsorize_percentiles
        r_base = _winsorize_r_ratios(r_base, lo_w, hi_w)

    max_steps_cap_check = int(min(len(r_base), config.max_steps_cap))
    if max_steps_cap_check < 1:
        raise ValueError("Not enough data to simulate.")

    target_profit_dollars = float(profit_target_dollar)
    max_loss_dollars = float(max_loss_dollar)
    daily_loss_dollars_static = float(daily_loss_dollar)
    target_balance = starting_balance + target_profit_dollars
    max_total_loss_balance_static = starting_balance - max_loss_dollars
    risk_dollars_fixed = float(risk_per_trade_dollar)
    consistency_day_cap = (consistency_rule_pct / 100.0) * target_profit_dollars

    def _sim_realization(
        rng_local: np.random.Generator, r_ratio_arr: np.ndarray, day_codes: np.ndarray
    ) -> Dict[str, Any]:
        n_trades_local = int(len(r_ratio_arr))
        n_sims = int(config.n_sims)
        max_steps = int(min(n_trades_local, config.max_steps_cap))
        if max_steps < 1:
            raise ValueError("Not enough data to simulate.")

        # Simulation state
        equity = np.full(n_sims, float(starting_balance), dtype=float)
        high_water = np.full(n_sims, float(starting_balance), dtype=float)
        outcome = np.full(n_sims, -1, dtype=np.int8)  # -1 ongoing; 1 success; 0 fail
        current_day = np.full(n_sims, -1, dtype=np.int64)
        day_start_equity = np.full(n_sims, float(starting_balance), dtype=float)
        daily_pnl = np.zeros(n_sims, dtype=float)
        trades_taken = np.zeros(n_sims, dtype=np.int32)
        challenge_pnl_dollars = np.full(n_sims, np.nan, dtype=float)

        n_paths = int(min(max(1, equity_paths), n_sims))
        plot_idx = rng_local.choice(n_sims, size=n_paths, replace=False)
        plot_done = np.zeros(n_paths, dtype=bool)
        plot_equity = np.full((n_paths, max_steps + 1), np.nan, dtype=float)
        plot_equity[:, 0] = float(starting_balance)
        plot_outcome = np.full(n_paths, -2, dtype=np.int8)

        trade_pool_len = n_trades_local
        for step in range(max_steps):
            active = outcome == -1
            if not np.any(active):
                break

            sampled_idx = rng_local.integers(0, trade_pool_len, size=n_sims)
            sampled_r = r_ratio_arr[sampled_idx]
            sampled_day = day_codes[sampled_idx]

            transitions = active & (sampled_day != current_day)
            if np.any(transitions):
                current_day[transitions] = sampled_day[transitions]
                day_start_equity[transitions] = equity[transitions]
                daily_pnl[transitions] = 0.0

            base_risk_dollars = np.full(n_sims, risk_dollars_fixed, dtype=float)
            pnl = base_risk_dollars * sampled_r

            equity[active] += pnl[active]
            daily_pnl[active] += pnl[active]
            trades_taken[active] += 1
            np.maximum(high_water, equity, out=high_water)

            success_mask = active & (equity >= target_balance)
            if np.any(success_mask):
                challenge_pnl_dollars[success_mask] = equity[success_mask] - float(starting_balance)
                outcome[success_mask] = 1

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

            if n_paths > 0:
                plot_active = ~plot_done
                if np.any(plot_active):
                    record_mask = plot_active
                    plot_equity[record_mask, step + 1] = equity[plot_idx[record_mask]]

                resolved_now = outcome[plot_idx] != -1
                plot_done |= resolved_now
                plot_outcome[resolved_now] = outcome[plot_idx[resolved_now]]

        unresolved = outcome == -1
        if np.any(unresolved):
            outcome[unresolved] = 0

        passed_mask = outcome == 1
        failed_mask = outcome == 0
        pass_rate = float(passed_mask.mean())
        ruin_rate = float((outcome == 0).mean())
        avg_trades_to_pass = float(np.mean(trades_taken[passed_mask])) if np.any(passed_mask) else float("nan")
        avg_trades_to_fail = float(np.mean(trades_taken[failed_mask])) if np.any(failed_mask) else float("nan")

        p = pass_rate
        if p <= 0.0:
            expected_trades_to_first_pass = float("inf")
        elif p >= 1.0:
            expected_trades_to_first_pass = float(avg_trades_to_pass) if np.isfinite(avg_trades_to_pass) else float("nan")
        elif np.isfinite(avg_trades_to_fail):
            expected_trades_to_first_pass = float(((1.0 - p) / p) * avg_trades_to_fail + avg_trades_to_pass)
        else:
            expected_trades_to_first_pass = float(avg_trades_to_pass) if np.isfinite(avg_trades_to_pass) else float("nan")

        funded_metrics: Optional[Dict[str, Any]] = None
        if funded_params is not None:
            max_f = int(
                min(
                    n_trades_local,
                    int(funded_params.get("max_steps_funded", config.max_steps_cap * 2)),
                )
            )
            max_f = max(1, max_f)
            funded_start_bal = float(funded_params.get("funded_starting_balance", starting_balance))
            risk_funded = (
                float(funded_risk_per_trade_dollar) if funded_risk_per_trade_dollar is not None else risk_dollars_fixed
            )
            ex_path = funded_params.get("express_funded_path")
            if ex_path is not None and ex_path not in ("standard", "consistency"):
                raise ValueError("express_funded_path must be None, 'standard', or 'consistency'.")
            funded_out = _run_funded_phase_vectorized(
                rng_local,
                r_ratio_arr,
                day_codes,
                n_sims=n_sims,
                funded_start_balance=funded_start_bal,
                risk_dollars_funded=risk_funded,
                max_loss_dollars=max_loss_dollars,
                daily_loss_dollars=daily_loss_dollars_static,
                drawdown_type=drawdown_type,
                passed_mask=passed_mask,
                min_payout_buffer=float(funded_params["min_payout_buffer"]),
                profit_split_pct=float(funded_params["profit_split_pct"]),
                funded_consistency_max_pct=float(funded_params["funded_consistency_max_pct"]),
                winning_day_profit_threshold=float(funded_params.get("winning_day_profit_threshold", 150.0)),
                min_winning_days=int(funded_params.get("min_winning_days", 5)),
                min_trading_days_for_payout=int(funded_params.get("min_trading_days_for_payout", 0)),
                max_steps=max_f,
                max_payout_cap_dollars=float(funded_params.get("max_payout_cap_dollars", 1.0e12)),
                max_payout_frac_of_equity=float(funded_params.get("max_payout_frac_of_equity", 0.5)),
                apply_consistency_gate=bool(funded_params.get("funded_payout_consistency_gate", True)),
                payout_withdrawal_request_model=bool(
                    funded_params.get("payout_withdrawal_request_model", False)
                ),
                express_funded_path=ex_path,
                min_consistency_calendar_days=int(funded_params.get("min_consistency_calendar_days", 3)),
                payout_processing_fee_dollars=float(funded_params.get("payout_processing_fee_dollars", 0.0)),
                min_payout_request_dollars=float(funded_params.get("min_payout_request_dollars", 0.0)),
            )
            total_payout = funded_out["total_payout"]
            had_payout = funded_out["had_payout"]
            fund_days = funded_out["funded_calendar_days"]
            funded_failed = funded_out["funded_failed"]
            funded_failed_before_first_payout = funded_failed & (~had_payout)

            n_passed = int(np.sum(passed_mask))
            avg_total_payout = float(np.sum(total_payout) / max(1, n_passed))
            payouts_achieved = int(np.sum(had_payout))
            payout_success_rate_conditional = float(payouts_achieved / max(1, n_passed))
            payout_success_rate_absolute = float(payouts_achieved / float(n_sims))
            payout_efficiency_pct = float(100.0 * payout_success_rate_absolute)
            funded_blowup_before_payout_rate_conditional = float(
                np.sum(funded_failed_before_first_payout[passed_mask]) / max(1, n_passed)
            )
            funded_blowup_before_payout_rate_absolute = float(
                np.sum(funded_failed_before_first_payout) / float(n_sims)
            )

            avg_funded_pnl = float(np.mean(total_payout[passed_mask])) if np.any(passed_mask) else float("nan")
            avg_challenge_pnl = float(np.nanmean(challenge_pnl_dollars[passed_mask])) if np.any(passed_mask) else float("nan")
            if np.isfinite(avg_challenge_pnl) and avg_challenge_pnl > 1e-12:
                survival_factor = float(avg_funded_pnl / avg_challenge_pnl)
            else:
                survival_factor = float("nan")

            net_ev = float(pass_rate * avg_total_payout - ruin_rate * float(challenge_fee))
            roi_pct = float((net_ev / float(challenge_fee)) * 100.0) if float(challenge_fee) > 0 else float("nan")

            failed_funded = passed_mask & funded_failed
            if np.any(failed_funded):
                avg_longevity = float(np.mean(fund_days[failed_funded]))
            elif np.any(passed_mask):
                avg_longevity = float(np.mean(fund_days[passed_mask]))
            else:
                avg_longevity = float("nan")

            fund_surv = funded_out["funded_survival_days"]
            avg_funded_survival_months = (
                float(np.mean(fund_surv[passed_mask]) / 30.0) if np.any(passed_mask) else float("nan")
            )

            funded_metrics = {
                "payout_success_rate_conditional": payout_success_rate_conditional,
                "payout_success_rate_absolute": payout_success_rate_absolute,
                "payout_efficiency_pct": payout_efficiency_pct,
                "funded_blowup_before_payout_rate_conditional": funded_blowup_before_payout_rate_conditional,
                "funded_blowup_before_payout_rate_absolute": funded_blowup_before_payout_rate_absolute,
                "survival_factor": survival_factor,
                "avg_funded_pnl": avg_funded_pnl,
                "avg_challenge_pnl": avg_challenge_pnl,
                "avg_total_payout_per_challenge_pass": avg_total_payout,
                "net_ev": net_ev,
                "roi_pct": roi_pct,
                "avg_account_longevity_days": avg_longevity,
                "avg_funded_survival_months": avg_funded_survival_months,
                "funded_total_payout_per_sim": total_payout,
                "funded_survival_days_per_sim": fund_surv.astype(float).copy(),
                "payout_histogram_values": total_payout[passed_mask].astype(float).copy(),
            }

        efficiency = float("inf") if pass_rate <= 0 else float(challenge_fee) / pass_rate

        target_drawdown_balance = float(max_total_loss_balance_static)
        if drawdown_type == "Trailing":
            target_drawdown_balance = float(starting_balance - max_loss_dollars)

        passed_paths = np.array([bool(o == 1) for o in plot_outcome], dtype=bool)
        paths_list: List[np.ndarray] = []
        for i in range(n_paths):
            row = plot_equity[i]
            if np.all(np.isnan(row)):
                paths_list.append(np.array([starting_balance], dtype=float))
                continue
            valid_idx = np.where(~np.isnan(row))[0]
            last_valid = int(valid_idx.max()) if valid_idx.size else 0
            paths_list.append(row[: last_valid + 1].copy())

        out_inner: Dict[str, Any] = {
            "pass_rate": pass_rate,
            "ruin_rate": ruin_rate,
            "avg_trades_to_pass": avg_trades_to_pass,
            "avg_trades_to_fail": avg_trades_to_fail,
            "expected_trades_to_first_pass": expected_trades_to_first_pass,
            "efficiency_total_cost_to_fund": efficiency,
            "target_balance": float(target_balance),
            "max_drawdown_balance": target_drawdown_balance,
            "plot_paths_equity": paths_list,
            "plot_paths_passed": passed_paths.tolist(),
            "max_steps": max_steps,
            "n_sims": n_sims,
        }
        if funded_metrics is not None:
            out_inner.update(funded_metrics)
        return out_inner

    B = int(bootstrap_outer_replicates)
    if B < 0:
        raise ValueError("bootstrap_outer_replicates must be >= 0.")

    if B == 0:
        rng = np.random.default_rng(seed)
        r_pool, d_pool = _apply_resample_mode(r_base, day_base, resample_mode, block_size, rng)
        out = _sim_realization(rng, r_pool, d_pool)
        out["resample_mode"] = resample_mode
        out["block_size"] = int(block_size)
        out["bootstrap_outer_replicates"] = 0
        if winsorize_percentiles is not None:
            out["winsorize_percentiles"] = winsorize_percentiles
        return out

    pass_rates: List[float] = []
    net_evs: List[float] = []
    out_primary: Optional[Dict[str, Any]] = None
    for b in range(B):
        rng_b = np.random.default_rng(seed + 10007 * b)
        r_pool, d_pool = _apply_resample_mode(r_base, day_base, resample_mode, block_size, rng_b)
        o = _sim_realization(rng_b, r_pool, d_pool)
        pass_rates.append(float(o["pass_rate"]))
        if funded_params is not None and "net_ev" in o:
            net_evs.append(float(o["net_ev"]))
        if b == 0:
            out_primary = o

    if out_primary is None:
        raise RuntimeError("outer bootstrap produced no output")

    out = dict(out_primary)
    pr_arr = np.asarray(pass_rates, dtype=float)
    out["pass_rate_bootstrap_mean"] = float(np.mean(pr_arr))
    out["pass_rate_bootstrap_p2_5"] = float(np.percentile(pr_arr, 2.5))
    out["pass_rate_bootstrap_p97_5"] = float(np.percentile(pr_arr, 97.5))
    out["pass_rate"] = float(out["pass_rate_bootstrap_mean"])
    out["pass_rate_replicate_0"] = float(pass_rates[0])

    if net_evs:
        ne_arr = np.asarray(net_evs, dtype=float)
        out["net_ev_bootstrap_mean"] = float(np.mean(ne_arr))
        out["net_ev_bootstrap_p2_5"] = float(np.percentile(ne_arr, 2.5))
        out["net_ev_bootstrap_p97_5"] = float(np.percentile(ne_arr, 97.5))
        out["net_ev"] = float(out["net_ev_bootstrap_mean"])
        out["net_ev_replicate_0"] = float(net_evs[0])
        if float(challenge_fee) > 0:
            out["roi_pct"] = float((out["net_ev"] / float(challenge_fee)) * 100.0)

    out["bootstrap_outer_replicates"] = B
    out["resample_mode"] = resample_mode
    out["block_size"] = int(block_size)
    if winsorize_percentiles is not None:
        out["winsorize_percentiles"] = winsorize_percentiles
    return out


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

