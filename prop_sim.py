from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_dates_to_calendar_days(date_series: pd.Series) -> np.ndarray:
    """
    Convert timestamps to calendar-day resolution (midnight) so that daily loss
    is evaluated per calendar day.
    """
    dt = pd.to_datetime(date_series, errors="coerce", utc=False)
    if dt.isna().any():
        bad = dt.isna().sum()
        raise ValueError(f"dateStart contains {bad} invalid timestamps.")
    # datetime64[D] = day-level resolution
    return dt.dt.normalize().to_numpy(dtype="datetime64[D]")


def prepare_trades(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pnls: float64 array of rPnL values (balance adjustments)
      dates: datetime64[D] array of calendar days derived from dateStart
    """
    required = {"rPnL", "dateStart"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    pnls = pd.to_numeric(df["rPnL"], errors="coerce")
    if pnls.isna().any():
        bad = pnls.isna().sum()
        raise ValueError(f"rPnL contains {bad} invalid/non-numeric values.")

    dates = _normalize_dates_to_calendar_days(df["dateStart"])
    if len(pnls) == 0:
        raise ValueError("CSV has no trade rows.")
    if len(pnls) != len(dates):
        raise ValueError("Trade arrays have inconsistent lengths.")

    return pnls.to_numpy(dtype="float64"), dates


@dataclass(frozen=True)
class Phase1Path:
    trades: np.ndarray  # integer trade index within the simulated sequence
    balances: np.ndarray  # balance values at each trade index


def _run_phase(
    *,
    pnls: np.ndarray,
    dates: np.ndarray,
    rng: np.random.Generator,
    n_sims: int,
    start_balance: float,
    profit_target_pct: float,
    daily_loss_pct: float,
    max_total_loss_pct: float,
    max_total_loss_mode: str,
    max_trades: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    n_trades = len(pnls)

    # Precompute thresholds relative to start, for speed.
    profit_target_balance = start_balance * (1.0 + profit_target_pct / 100.0)
    static_max_total_threshold = start_balance * (1.0 - max_total_loss_pct / 100.0)
    daily_loss_threshold_ratio = 1.0 - daily_loss_pct / 100.0

    # Outputs.
    passed = np.zeros(n_sims, dtype=bool)
    failed = np.zeros(n_sims, dtype=bool)
    trades_to_pass = np.full(n_sims, np.nan, dtype="float64")
    days_to_pass = np.full(n_sims, np.nan, dtype="float64")

    # For the balance-curve visualization we track a small sample of sims separately.
    # This avoids storing full paths for all sims.
    sim_path_indices = np.arange(min(100, n_sims))
    paths = [Phase1Path(trades=np.array([], dtype=int), balances=np.array([], dtype=float)) for _ in range(len(sim_path_indices))]
    path_by_sim = {int(sim_path_indices[i]): i for i in range(len(sim_path_indices))}

    for sim in range(n_sims):
        current_balance = float(start_balance)
        high_water_mark = float(start_balance)

        first_day_seen = True
        current_day = None
        day_count = 0
        day_start_balance = float(start_balance)

        # Draw a full sequence up-front for repeatable inner-loop.
        idxs = rng.integers(0, n_trades, size=max_trades)

        if sim in path_by_sim:
            pidx = path_by_sim[sim]
            trades_hist = [0]
            balances_hist = [current_balance]

        for t, idx in enumerate(idxs):
            trade_date = dates[idx]
            pnl = float(pnls[idx])

            if current_day is None or trade_date != current_day:
                current_day = trade_date
                if first_day_seen:
                    first_day_seen = False
                else:
                    day_count += 1
                day_start_balance = current_balance

            new_balance = current_balance + pnl

            # Update high-water mark based on new equity.
            if new_balance > high_water_mark:
                high_water_mark = new_balance

            daily_threshold = day_start_balance * daily_loss_threshold_ratio

            if max_total_loss_mode == "static":
                max_total_threshold = static_max_total_threshold
            elif max_total_loss_mode == "trailing":
                max_total_threshold = high_water_mark * (1.0 - max_total_loss_pct / 100.0)
            else:
                raise ValueError("max_total_loss_mode must be 'static' or 'trailing'")

            # Profit check is based on end-of-trade balance.
            current_balance = new_balance

            if current_balance >= profit_target_balance:
                passed[sim] = True
                trades_to_pass[sim] = t + 1
                # day_count starts at 0 for the first day we see; incremented per day rollover.
                days_to_pass[sim] = float(day_count + 1)
                if sim in path_by_sim:
                    pidx = path_by_sim[sim]
                    trades_hist.append(t + 1)
                    balances_hist.append(current_balance)
                    paths[pidx] = Phase1Path(
                        trades=np.asarray(trades_hist, dtype=int),
                        balances=np.asarray(balances_hist, dtype=float),
                    )
                break

            # Violation check.
            if (current_balance <= daily_threshold) or (current_balance <= max_total_threshold):
                failed[sim] = True
                if sim in path_by_sim:
                    pidx = path_by_sim[sim]
                    trades_hist.append(t + 1)
                    balances_hist.append(current_balance)
                    paths[pidx] = Phase1Path(
                        trades=np.asarray(trades_hist, dtype=int),
                        balances=np.asarray(balances_hist, dtype=float),
                    )
                break

            if sim in path_by_sim:
                pidx = path_by_sim[sim]
                trades_hist.append(t + 1)
                balances_hist.append(current_balance)
        else:
            # Reached max_trades without pass. Mark as neither passed nor failed.
            # (The "ruin" metric typically treats only violations as ruin.)
            if sim in path_by_sim:
                pidx = path_by_sim[sim]
                paths[pidx] = Phase1Path(
                    trades=np.asarray(trades_hist, dtype=int) if "trades_hist" in locals() else np.array([], dtype=int),
                    balances=np.asarray(balances_hist, dtype=float) if "balances_hist" in locals() else np.array([], dtype=float),
                )

    return {
        "passed": passed,
        "failed": failed,
        "trades_to_pass": trades_to_pass,
        "days_to_pass": days_to_pass,
        "paths": paths,
        "path_sims": sim_path_indices,
    }


def run_monte_carlo(
    *,
    pnls: np.ndarray,
    dates: np.ndarray,
    n_sims: int,
    seed: int,
    start_balance: float,
    profit_target_pct: float,
    daily_loss_pct: float,
    max_total_loss_pct: float,
    max_total_loss_mode: str,
    challenge_fee: float,
    payout_split_pct: float,
    funded_phase_target_pct: float,
    max_trades_phase1: int,
    max_trades_funded: int,
    max_plot_paths: int = 100,
) -> Dict[str, object]:
    """
    Phase 1: try to hit `profit_target_pct` before any daily/max-total violation.
    Phase 2 ("Funded Phase"): if Phase 1 passes, continue from the *same sampled*
    trade sequence to hit `funded_phase_target_pct` (first payout). If a violation
    occurs first, payout=0.
    """
    if max_trades_phase1 <= 0 or max_trades_funded <= 0:
        raise ValueError("max_trades_phase1 and max_trades_funded must be positive.")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")
    if start_balance <= 0:
        raise ValueError("start_balance must be positive.")
    if not np.isfinite(pnls).all():
        raise ValueError("pnls contains non-finite values.")

    n_trades = len(pnls)
    if n_trades == 0:
        raise ValueError("No trades available.")

    rng = np.random.default_rng(seed)
    rng_plot = np.random.default_rng(seed + 42_424)
    plot_sims = rng_plot.choice(n_sims, size=min(max_plot_paths, n_sims), replace=False)
    plot_sims_set = set(map(int, plot_sims))

    # Outputs.
    passed = np.zeros(n_sims, dtype=bool)
    failed = np.zeros(n_sims, dtype=bool)
    trades_to_pass = np.full(n_sims, np.nan, dtype="float64")
    days_to_pass = np.full(n_sims, np.nan, dtype="float64")
    payout_gross = np.zeros(n_sims, dtype="float64")
    payout_hit = np.zeros(n_sims, dtype=bool)

    # Balance-path visualization (Phase 1 only).
    balance_paths: Dict[int, Phase1Path] = {}

    profit_target_balance_phase1 = start_balance * (1.0 + profit_target_pct / 100.0)
    daily_loss_threshold_ratio = 1.0 - daily_loss_pct / 100.0
    static_max_total_threshold_phase1 = start_balance * (1.0 - max_total_loss_pct / 100.0)
    static_max_total_threshold_funded_multiplier = 1.0 - max_total_loss_pct / 100.0

    seq_len = max_trades_phase1 + max_trades_funded

    for sim in range(n_sims):
        idxs = rng.integers(0, n_trades, size=seq_len)

        current_balance = float(start_balance)
        high_water_mark = float(start_balance)

        current_day = None
        day_start_balance = float(start_balance)
        day_count = 0
        first_day_seen = False

        if sim in plot_sims_set:
            trades_hist = [0]
            balances_hist = [current_balance]

        pass_t = None

        # ---- Phase 1
        for t in range(max_trades_phase1):
            idx = int(idxs[t])
            trade_date = dates[idx]
            pnl = float(pnls[idx])

            if current_day is None or trade_date != current_day:
                current_day = trade_date
                if first_day_seen:
                    day_count += 1
                else:
                    first_day_seen = True
                day_start_balance = current_balance

            new_balance = current_balance + pnl
            if new_balance > high_water_mark:
                high_water_mark = new_balance

            daily_threshold = day_start_balance * daily_loss_threshold_ratio
            if max_total_loss_mode == "static":
                max_total_threshold = static_max_total_threshold_phase1
            elif max_total_loss_mode == "trailing":
                max_total_threshold = high_water_mark * (1.0 - max_total_loss_pct / 100.0)
            else:
                raise ValueError("max_total_loss_mode must be 'static' or 'trailing'")

            current_balance = new_balance

            if current_balance >= profit_target_balance_phase1:
                passed[sim] = True
                pass_t = t
                trades_to_pass[sim] = float(t + 1)
                days_to_pass[sim] = float(day_count + 1)
                if sim in plot_sims_set:
                    trades_hist.append(t + 1)
                    balances_hist.append(current_balance)
                    balance_paths[sim] = Phase1Path(
                        trades=np.asarray(trades_hist, dtype=int),
                        balances=np.asarray(balances_hist, dtype=float),
                    )
                break

            if (current_balance <= daily_threshold) or (current_balance <= max_total_threshold):
                failed[sim] = True
                if sim in plot_sims_set:
                    trades_hist.append(t + 1)
                    balances_hist.append(current_balance)
                    balance_paths[sim] = Phase1Path(
                        trades=np.asarray(trades_hist, dtype=int),
                        balances=np.asarray(balances_hist, dtype=float),
                    )
                break

            if sim in plot_sims_set:
                trades_hist.append(t + 1)
                balances_hist.append(current_balance)

        if pass_t is None:
            continue

        # ---- Funded Phase (continue from pass_t + 1)
        funded_start_balance = current_balance
        funded_target_balance = funded_start_balance * (1.0 + funded_phase_target_pct / 100.0)

        high_water_mark_funded = float(high_water_mark)
        # current_day/day_start_balance/high_water_mark_funded carry over.

        for t2 in range(pass_t + 1, seq_len):
            idx = int(idxs[t2])
            trade_date = dates[idx]
            pnl = float(pnls[idx])

            if trade_date != current_day:
                current_day = trade_date
                day_start_balance = current_balance

            new_balance = current_balance + pnl
            if new_balance > high_water_mark_funded:
                high_water_mark_funded = new_balance

            daily_threshold = day_start_balance * daily_loss_threshold_ratio
            if max_total_loss_mode == "static":
                max_total_threshold = funded_start_balance * static_max_total_threshold_funded_multiplier
            else:  # trailing
                max_total_threshold = high_water_mark_funded * (1.0 - max_total_loss_pct / 100.0)

            current_balance = new_balance

            if current_balance >= funded_target_balance:
                payout_hit[sim] = True
                payout_gross[sim] = (payout_split_pct / 100.0) * (current_balance - funded_start_balance)
                break

            if (current_balance <= daily_threshold) or (current_balance <= max_total_threshold):
                payout_hit[sim] = False
                payout_gross[sim] = 0.0
                break

    passed_count = int(passed.sum())
    failed_count = int(failed.sum())

    pass_probability = float(passed_count / n_sims)
    ruin_probability = float(failed_count / n_sims)

    avg_trades_to_pass = float(np.nanmean(trades_to_pass[passed])) if passed_count else float("nan")
    avg_days_to_pass = float(np.nanmean(days_to_pass[passed])) if passed_count else float("nan")

    payout_gross_mean_overall = float(np.mean(payout_gross))
    avg_payout_on_successful_accounts = float(np.mean(payout_gross[payout_hit])) if np.any(payout_hit) else 0.0

    expected_value_net = payout_gross_mean_overall - float(challenge_fee)

    # Make a stable list for plotting.
    balance_paths_list = [balance_paths[i] for i in sorted(balance_paths.keys())]

    return {
        "n_sims": n_sims,
        "pass_probability": pass_probability,
        "ruin_probability": ruin_probability,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "avg_trades_to_pass": avg_trades_to_pass,
        "avg_days_to_pass": avg_days_to_pass,
        "payout_gross_mean_overall": payout_gross_mean_overall,
        "avg_payout_on_successful_accounts": avg_payout_on_successful_accounts,
        "expected_value_net": expected_value_net,
        "payout_gross": payout_gross,
        "payout_hit": payout_hit,
        "trades_to_pass": trades_to_pass,
        "days_to_pass": days_to_pass,
        "passed": passed,
        "failed": failed,
        "balance_paths": balance_paths_list,
    }

