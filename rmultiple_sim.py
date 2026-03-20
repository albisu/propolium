from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationPath:
    balances: np.ndarray  # equity curve, includes starting balance
    passed: bool


def prepare_rmultiple_trades(
    df: pd.DataFrame,
    *,
    standard_risk_amount: float,
) -> Tuple[List[np.ndarray], Optional[pd.Timestamp]]:
    """
    Prepare trades grouped by calendar day.

    Returns:
      trades_by_day: list of arrays, each containing R-multiples for one calendar day
      date_start_range: (optional) min timestamp used for labeling/debug
    """
    required = {"rPnL", "dateStart", "initialBalance"}
    missing = required - set(df.columns)
    # initialBalance is used only for default balance convenience in UI; allow missing.
    missing_nonoptional = {"rPnL", "dateStart"} - set(df.columns)
    if missing_nonoptional:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_nonoptional)}")

    if standard_risk_amount <= 0:
        raise ValueError("standard_risk_amount must be > 0.")

    if len(df) == 0:
        raise ValueError("CSV has no rows.")

    date_start_range: Optional[pd.Timestamp] = None
    dt = pd.to_datetime(df["dateStart"], errors="coerce", utc=False)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"dateStart contains {bad} invalid timestamps.")
    date_start_range = dt.min()

    # Stable sort to keep intra-day order deterministic.
    work = df.copy()
    work["_dateStart_dt"] = dt
    work["_day"] = work["_dateStart_dt"].dt.normalize()
    work["_rPnL"] = pd.to_numeric(work["rPnL"], errors="coerce")
    if work["_rPnL"].isna().any():
        bad = int(work["_rPnL"].isna().sum())
        raise ValueError(f"rPnL contains {bad} invalid/non-numeric values.")

    work = work.sort_values(["_dateStart_dt"], kind="mergesort")
    work["_r_multiple"] = work["_rPnL"].to_numpy(dtype=float) / float(standard_risk_amount)

    grouped = work.groupby("_day", sort=False)["_r_multiple"]
    trades_by_day: List[np.ndarray] = []
    for _, grp in grouped:
        arr = grp.to_numpy(dtype=float)
        if arr.size == 0:
            continue
        trades_by_day.append(arr)

    if len(trades_by_day) == 0:
        raise ValueError("No trades could be grouped by calendar day.")

    return trades_by_day, date_start_range


def run_rmultiple_monte_carlo(
    *,
    trades_by_day: List[np.ndarray],
    n_sims: int,
    seed: int,
    starting_balance: float,
    challenge_fee: float,
    risk_pct_per_trade: float,
    static_risk: bool,
    profit_target_pct: float,
    daily_loss_limit_pct: float,
    max_total_loss_pct: float,
    equity_paths: int = 50,
) -> Dict[str, object]:
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")
    if starting_balance <= 0:
        raise ValueError("starting_balance must be > 0.")
    if challenge_fee < 0:
        raise ValueError("challenge_fee must be >= 0.")
    if not np.isfinite(risk_pct_per_trade):
        raise ValueError("risk_pct_per_trade must be finite.")
    if risk_pct_per_trade < 0:
        raise ValueError("risk_pct_per_trade must be >= 0.")
    if daily_loss_limit_pct < 0 or max_total_loss_pct < 0:
        raise ValueError("daily_loss_limit_pct and max_total_loss_pct must be >= 0.")

    if profit_target_pct < 0:
        raise ValueError("profit_target_pct must be >= 0.")

    n_days = len(trades_by_day)
    if n_days == 0:
        raise ValueError("No day-groups available.")

    rng = np.random.default_rng(seed)

    sample_n = min(max(1, equity_paths), n_sims)
    sample_sims = set(map(int, rng.choice(n_sims, size=sample_n, replace=False)))

    profit_target_balance = starting_balance * (1.0 + profit_target_pct / 100.0)
    max_total_loss_dollars = starting_balance * (max_total_loss_pct / 100.0)
    max_total_loss_balance = starting_balance - max_total_loss_dollars

    risk_fraction = risk_pct_per_trade / 100.0

    pass_count = 0
    ruin_count = 0
    trades_to_pass: List[int] = []
    payout_on_pass: List[float] = []

    sample_paths: List[SimulationPath] = []
    # Pre-create empty placeholders so we can append in order.
    sim_to_path: Dict[int, List[float]] = {}
    sim_to_status: Dict[int, bool] = {}

    for sim in range(n_sims):
        # Shuffle "days" to model the daily reset behavior without relying on intra-day ordering.
        day_order = rng.permutation(n_days)

        equity = float(starting_balance)
        total_pnl = 0.0
        trade_count = 0

        day_pnl = 0.0
        day_start_balance = equity
        passed = False
        ruined = False

        if sim in sample_sims:
            sim_to_path[sim] = [equity]

        for day_idx in day_order:
            day_start_balance = equity
            day_pnl = 0.0
            daily_loss_limit_dollars = day_start_balance * (daily_loss_limit_pct / 100.0)

            r_mults = trades_by_day[int(day_idx)]

            for r_mult in r_mults:
                # Compute the user's $Risk for this trade based on the current balance (compounding)
                # or the initial balance (static).
                base_for_risk = starting_balance if static_risk else equity
                risk_dollars = base_for_risk * risk_fraction
                pnl = float(r_mult) * risk_dollars

                equity += pnl
                total_pnl += pnl
                day_pnl += pnl
                trade_count += 1

                if sim in sample_sims:
                    sim_to_path[sim].append(equity)

                # SUCCESS
                if equity >= profit_target_balance:
                    passed = True
                    break

                # FAILURE: daily loss limit breached (cumulative day PnL <= -limit)
                if day_pnl <= -daily_loss_limit_dollars:
                    ruined = True
                    break

                # FAILURE: max total loss limit breached
                if equity <= max_total_loss_balance:
                    ruined = True
                    break

            if passed or ruined:
                break

        if passed:
            pass_count += 1
            trades_to_pass.append(trade_count)
            payout = equity - starting_balance
            payout_on_pass.append(float(payout))
            if sim in sample_sims:
                sim_to_status[sim] = True
        else:
            if ruined:
                ruin_count += 1
            if sim in sample_sims:
                sim_to_status[sim] = False

    pass_rate = pass_count / n_sims
    ruin_rate = ruin_count / n_sims
    fail_rate = 1.0 - pass_rate

    avg_trades_to_pass = float(np.mean(trades_to_pass)) if trades_to_pass else float("nan")
    avg_payout = float(np.mean(payout_on_pass)) if payout_on_pass else 0.0

    ev = pass_rate * avg_payout - fail_rate * float(challenge_fee)

    total_cost_to_fund = float("inf") if pass_rate == 0 else float(challenge_fee) / pass_rate

    # Assemble sampled equity paths.
    for sim in sorted(sample_sims):
        balances = np.asarray(sim_to_path[sim], dtype=float)
        passed_flag = bool(sim_to_status.get(sim, False))
        sample_paths.append(SimulationPath(balances=balances, passed=passed_flag))

    return {
        "pass_rate": float(pass_rate),
        "ruin_rate": float(ruin_rate),
        "fail_rate": float(fail_rate),
        "avg_trades_to_pass": avg_trades_to_pass,
        "avg_payout": float(avg_payout),
        "ev": float(ev),
        "total_cost_to_fund": float(total_cost_to_fund),
        "payout_on_pass": payout_on_pass,
        "trades_to_pass": trades_to_pass,
        "sample_paths": sample_paths,
    }

