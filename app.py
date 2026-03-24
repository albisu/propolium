import inspect
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure we load `prop_strategy_engine.py` from this repo (same folder as app.py), not an older
# package on PYTHONPATH — a common cause of "unexpected keyword argument 'funded_params'" on hosts.
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from prop_strategy_engine import run_prop_strategy_monte_carlo

_ENGINE_SUPPORTS_FUNDED = "funded_params" in inspect.signature(run_prop_strategy_monte_carlo).parameters


def _run_prop_strategy_monte_carlo(**kwargs):
    """
    Call the engine with only arguments accepted by the installed `run_prop_strategy_monte_carlo`.

    Streamlit Community Cloud occasionally serves a stale checkout; an older engine without
    `funded_params` would otherwise raise: unexpected keyword argument 'funded_params'.
    """
    sig = inspect.signature(run_prop_strategy_monte_carlo)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return run_prop_strategy_monte_carlo(**allowed)


st.set_page_config(page_title="Prop Firm Strategy Simulation Engine", layout="wide")

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* Light-touch polish: Streamlit dark defaults + tight spacing */
    :root {
        --pp-font: "Inter", "Source Sans Pro", sans-serif;
        --pp-accent: #7ce38f;
        --page-pad-x: clamp(0.75rem, 2vw, 1.5rem);
        --page-pad-y: 0.65rem;
        --col-gap: 1rem;
    }
    html { scroll-behavior: smooth; }
    html, body, .stApp {
        font-family: var(--pp-font), sans-serif;
    }
    .block-container {
        padding: 0.75rem var(--page-pad-x) 1rem var(--page-pad-x) !important;
        max-width: min(1400px, 100%) !important;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stDecoration"] { display: none; }
    /* Page title */
    .pp-title {
        font-weight: 600;
        font-size: clamp(1.1rem, 1.8vw, 1.35rem);
        line-height: 1.25;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .pp-subtitle {
        font-size: 0.8rem;
        opacity: 0.85;
        margin: 0.2rem 0 0 0;
    }
    p.pp-column-title {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.7;
        margin: 0 0 0.35rem 0 !important;
    }
    /* Section labels — minimal, no heavy boxes */
    p.pp-input-section-title {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 0 0.45rem 0 !important;
        padding: 0 0 0.2rem 0.5rem !important;
        border-left: 3px solid #3d9df0;
        line-height: 1.3;
    }
    p.pp-input-section-title.pp-funded {
        border-left-color: var(--pp-accent);
    }
    /* Bordered panels: compact padding */
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.5rem 0.65rem 0.6rem 0.65rem !important;
        border-radius: 0.5rem !important;
    }
    [data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
    [data-testid="stHorizontalBlock"] {
        gap: var(--col-gap) !important;
        align-items: flex-start !important;
    }
    /* Inputs: small gap between challenge / funded blocks */
    section[data-testid="stMain"] [data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stVerticalBlockBorderWrapper"]:first-of-type {
        margin-bottom: 0.65rem !important;
    }
    section[data-testid="stMain"] [data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stVerticalBlockBorderWrapper"] + [data-testid="stVerticalBlockBorderWrapper"] {
        border-left: 2px solid rgba(124, 227, 143, 0.45) !important;
    }
    hr {
        margin: 0.5rem 0 0.65rem 0 !important;
        opacity: 0.35;
    }
    /* Charts: shorter default to reduce scroll */
    [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] iframe {
        min-height: 220px !important;
    }
    /* Tabs: compact, keep Streamlit behavior */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.35rem !important;
        padding: 0.15rem 0 0.35rem 0 !important;
        min-height: unset !important;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"] {
        padding: 0.35rem 0.75rem !important;
        min-height: unset !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stTabContent"] {
        padding-top: 0.45rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div id="propolium-page-top">
        <div class="pp-title">Prop Firm Strategy Simulation Engine</div>
        <div class="pp-subtitle">Monte Carlo · prop firm rules · funding economics</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ─── Main work area: Inputs | Results ───────────────────────────────────────


def _fmt_percent(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100.0:.2f}%"


def _fmt_dollars(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def _fmt_days(x: float | None) -> str:
    """Format calendar days; +inf prints as ∞."""
    if x is None:
        return "n/a"
    if isinstance(x, (float, np.floating)) and np.isposinf(float(x)):
        return "∞"
    if np.isfinite(x):
        return f"{float(x):.2f}"
    return "n/a"


@st.cache_data(show_spinner=False)
def _load_df(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _strategy_metrics_from_df(df: pd.DataFrame) -> dict[str, float]:
    """
    Historical stats from uploaded CSV (not Monte Carlo): win rate, profit factor, win/loss ratio, trade frequency.
    """
    out: dict[str, float] = {
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
        "win_loss_ratio": float("nan"),
        "trades_per_calendar_day": float("nan"),
    }
    if df is None or len(df) == 0 or "rPnL" not in df.columns:
        return out
    r = pd.to_numeric(df["rPnL"], errors="coerce").dropna()
    if len(r) == 0:
        return out
    wins = r[r > 0]
    losses = r[r < 0]
    out["win_rate"] = float(len(wins) / len(r))
    sum_w = float(wins.sum()) if len(wins) else 0.0
    sum_l = float(abs(losses.sum())) if len(losses) else 0.0
    if sum_l > 1e-12:
        out["profit_factor"] = float(sum_w / sum_l)
    elif sum_w > 1e-12:
        out["profit_factor"] = float("inf")
    if len(wins) and len(losses):
        al = float(losses.mean())
        if abs(al) > 1e-12:
            out["win_loss_ratio"] = float(wins.mean() / abs(al))
    if "dateStart" in df.columns:
        dt = pd.to_datetime(df["dateStart"], errors="coerce")
        n_days = int(dt.dt.normalize().nunique(dropna=True))
        if n_days > 0:
            out["trades_per_calendar_day"] = float(len(r) / n_days)
    return out


# Configuration templates (challenge + funded engine presets).
TEMPLATE_CUSTOM = "custom"
TEMPLATE_TOPSTEP_STANDARD = "topstep_express_standard"
TEMPLATE_TOPSTEP_CONSISTENCY = "topstep_express_consistency"
TEMPLATE_FOREX = "forex"

TEMPLATE_LABELS: dict[str, str] = {
    TEMPLATE_CUSTOM: "Custom (manual)",
    TEMPLATE_TOPSTEP_STANDARD: "Topstep Express — Standard payout",
    TEMPLATE_TOPSTEP_CONSISTENCY: "Topstep Express — Consistency payout",
    TEMPLATE_FOREX: "Forex (Phase 1 & 2)",
}


def _apply_template_fields(template_id: str) -> None:
    """Overwrite input session keys when user picks a non-custom template."""
    if template_id == TEMPLATE_CUSTOM:
        return
    st.session_state.starting_equity = 50000
    st.session_state.risk_per_trade = 500
    st.session_state.profit_target = 3000
    st.session_state.max_loss = 2000
    st.session_state.daily_loss = 2000
    st.session_state.drawdown_type = "Trailing"
    st.session_state.consistency_rule_pct = 50
    st.session_state.challenge_fee = 49
    st.session_state.activation_fee = 149
    st.session_state.monthly_recurring = True
    st.session_state.funded_risk_per_trade = 250
    st.session_state.funded_consistency_max = 40
    if template_id == TEMPLATE_TOPSTEP_STANDARD:
        st.session_state.funded_min_days = 5
    elif template_id == TEMPLATE_TOPSTEP_CONSISTENCY:
        st.session_state.funded_min_days = 3
    elif template_id == TEMPLATE_FOREX:
        st.session_state.starting_equity = 100000
        st.session_state.risk_per_trade = 500
        st.session_state.profit_target_phase1 = 10000
        st.session_state.profit_target_phase2 = 5000
        st.session_state.max_loss = 10000
        st.session_state.daily_loss = 5000
        st.session_state.profit_target = 10000  # fallback when switching away
        st.session_state.activation_fee = 0
        st.session_state.consistency_rule_pct = 0
        st.session_state.monthly_recurring = False


def _sync_template_on_change() -> None:
    cur = st.session_state.get("config_template", TEMPLATE_CUSTOM)
    prev = st.session_state.get("_config_template_prev", cur)
    if cur != prev:
        _apply_template_fields(cur)
    st.session_state._config_template_prev = cur


def _funded_params_for_template(template_id: str, ui_funded_min_days: int, ui_funded_consistency: float) -> dict:
    """Engine `funded_params` — Topstep Express templates use repeat payout cycles + fees."""
    base = {
        "min_payout_buffer": 2000.0,
        "profit_split_pct": 90.0,
        "funded_consistency_max_pct": float(ui_funded_consistency),
        "winning_day_profit_threshold": 150.0,
        "min_winning_days": int(ui_funded_min_days),
        "min_trading_days_for_payout": 0,
        "max_steps_funded": 2400,
        "max_payout_cap_dollars": 1.0e12,
        "max_payout_frac_of_equity": 0.5,
        "funded_payout_consistency_gate": True,
        "payout_withdrawal_request_model": False,
        "express_funded_path": None,
        "min_consistency_calendar_days": 3,
        "payout_processing_fee_dollars": 0.0,
        "min_payout_request_dollars": 0.0,
    }
    if template_id == TEMPLATE_TOPSTEP_STANDARD:
        base.update(
            {
                "min_payout_buffer": 0.0,
                "funded_consistency_max_pct": 40.0,
                "winning_day_profit_threshold": 150.0,
                "min_winning_days": int(ui_funded_min_days),
                "max_payout_cap_dollars": 5000.0,
                "max_payout_frac_of_equity": 0.5,
                "funded_payout_consistency_gate": False,
                "payout_withdrawal_request_model": True,
                "express_funded_path": "standard",
                "min_consistency_calendar_days": 3,
                "payout_processing_fee_dollars": 30.0,
                "min_payout_request_dollars": 125.0,
            }
        )
    elif template_id == TEMPLATE_TOPSTEP_CONSISTENCY:
        base.update(
            {
                "min_payout_buffer": 0.0,
                "funded_consistency_max_pct": float(ui_funded_consistency),
                "winning_day_profit_threshold": 150.0,
                "min_winning_days": 0,
                "max_payout_cap_dollars": 6000.0,
                "max_payout_frac_of_equity": 0.5,
                "funded_payout_consistency_gate": True,
                "payout_withdrawal_request_model": True,
                "express_funded_path": "consistency",
                "min_consistency_calendar_days": max(3, int(ui_funded_min_days)),
                "payout_processing_fee_dollars": 30.0,
                "min_payout_request_dollars": 125.0,
            }
        )
    elif template_id == TEMPLATE_FOREX:
        base.update(
            {
                "min_payout_buffer": 0.0,
                "funded_consistency_max_pct": 100.0,
                "winning_day_profit_threshold": 1.0,
                "min_winning_days": 0,
                "min_trading_days_for_payout": 0,
                "max_payout_cap_dollars": 1.0e12,
                "max_payout_frac_of_equity": 0.5,
                "funded_payout_consistency_gate": False,
                "payout_withdrawal_request_model": False,
                "express_funded_path": None,
                "min_consistency_calendar_days": 0,
                "payout_processing_fee_dollars": 0.0,
                "min_payout_request_dollars": 0.0,
            }
        )
    return base


def _init_defaults() -> None:
    st.session_state.setdefault("config_template", TEMPLATE_CUSTOM)
    st.session_state.setdefault("starting_equity", 50000)
    st.session_state.setdefault("risk_per_trade", 500)
    st.session_state.setdefault("profit_target", 3000)
    st.session_state.setdefault("max_loss", 2000)
    st.session_state.setdefault("daily_loss", 2000)
    st.session_state.setdefault("drawdown_type", "Trailing")
    st.session_state.setdefault("challenge_fee", 49)
    st.session_state.setdefault("activation_fee", 149)
    st.session_state.setdefault("consistency_rule_pct", 50)
    st.session_state.setdefault("monthly_recurring", True)
    st.session_state.setdefault("funded_consistency_max", 40)
    st.session_state.setdefault("funded_min_days", 5)
    st.session_state.setdefault("funded_risk_per_trade", 250)
    st.session_state.setdefault("profit_target_phase1", 8000)
    st.session_state.setdefault("profit_target_phase2", 5000)
    st.session_state.setdefault("last_run_template", None)


_init_defaults()

run = False

col_inputs, col_outputs = st.columns([1, 1.2], gap="medium")

with col_inputs:
    st.markdown('<p class="pp-column-title">Inputs</p>', unsafe_allow_html=True)

    st.selectbox(
        "Configuration template",
        options=[TEMPLATE_CUSTOM, TEMPLATE_TOPSTEP_STANDARD, TEMPLATE_TOPSTEP_CONSISTENCY, TEMPLATE_FOREX],
        format_func=lambda x: TEMPLATE_LABELS[x],
        key="config_template",
        help=(
            "Topstep Express: payout policy (Standard vs Consistency). "
            "Forex: two-phase challenge with separate Phase 1 & Phase 2 profit targets."
        ),
    )
    _sync_template_on_change()

    with st.container(border=True):
        st.markdown(
            '<p class="pp-input-section-title pp-challenge">Challenge evaluation</p>',
            unsafe_allow_html=True,
        )
        st.caption("Everything below simulates **passing the firm challenge** (account, limits, rules, fees).")

        st.markdown("**Data upload**")
        uploaded_file = st.file_uploader("Upload trade history CSV", type=["csv"], label_visibility="collapsed")

        st.markdown("**1 · Account & risk**")
        ar1, ar2 = st.columns(2)
        with ar1:
            account_size = st.number_input("Equity ($)", min_value=1, step=1000, format="%d", key="starting_equity")
        with ar2:
            risk_per_trade = st.number_input(
                "Risk per Trade ($)",
                min_value=0,
                step=10,
                format="%d",
                key="risk_per_trade",
                help="Dollar risk per trade during the **challenge** phase (funded phase uses Funded Risk below).",
            )

        st.markdown("**2 · Challenge limits**")
        _tpl = st.session_state.get("config_template", TEMPLATE_CUSTOM)
        if _tpl == TEMPLATE_FOREX:
            lm1, lm2, lm3, lm4 = st.columns(4)
            with lm1:
                profit_target_phase1 = st.number_input(
                    "Phase 1 target ($)",
                    min_value=0,
                    step=100,
                    format="%d",
                    key="profit_target_phase1",
                    help="Profit required to pass Phase 1 (then Phase 2 starts from starting balance).",
                )
            with lm2:
                profit_target_phase2 = st.number_input(
                    "Phase 2 target ($)",
                    min_value=0,
                    step=100,
                    format="%d",
                    key="profit_target_phase2",
                    help="Profit required in Phase 2 (verification) to pass the full challenge.",
                )
            with lm3:
                max_loss = st.number_input("Max loss ($)", min_value=0, step=100, format="%d", key="max_loss")
            with lm4:
                daily_loss = st.number_input("Daily loss ($)", min_value=0, step=100, format="%d", key="daily_loss")
            profit_target = profit_target_phase1  # fallback for any legacy path
            st.caption(
                "Phase 1 & 2: hit Phase 1 target, then Phase 2 (verification) starts from starting balance."
            )
        else:
            lm1, lm2, lm3 = st.columns(3)
            with lm1:
                profit_target = st.number_input(
                    "Profit target ($)",
                    min_value=0,
                    step=100,
                    format="%d",
                    key="profit_target",
                    help="Profit required to pass the challenge.",
                )
            with lm2:
                max_loss = st.number_input("Max loss ($)", min_value=0, step=100, format="%d", key="max_loss")
            with lm3:
                daily_loss = st.number_input("Daily loss ($)", min_value=0, step=100, format="%d", key="daily_loss")

        st.markdown("**3 · Rules**")
        _tpl_rules = st.session_state.get("config_template", TEMPLATE_CUSTOM)
        ru1, ru2 = st.columns(2)
        with ru1:
            drawdown_type = st.selectbox("Drawdown type", options=["Static", "Trailing"], key="drawdown_type")
        with ru2:
            if _tpl_rules == TEMPLATE_FOREX:
                st.caption("Consistency rule: **off** (forex challenge).")
                consistency_rule_pct = int(st.session_state.get("consistency_rule_pct", 0))
            else:
                consistency_rule_pct = st.number_input(
                    "Consistency rule (%)", min_value=0, step=1, format="%d", key="consistency_rule_pct"
                )

        st.markdown("**4 · Fees**")
        _tpl_fees = st.session_state.get("config_template", TEMPLATE_CUSTOM)
        if _tpl_fees == TEMPLATE_FOREX:
            fe1, fe2 = st.columns(2)
            with fe1:
                challenge_fee = st.number_input("Challenge fee ($)", min_value=0, step=1, format="%d", key="challenge_fee")
            with fe2:
                st.caption("Activation fee: **none** (forex). Cost to pass = challenge attempts only.")
            activation_fee = 0
            monthly_recurring = st.checkbox(
                "Monthly recurring",
                key="monthly_recurring",
                help="If on, each challenge attempt is billed as fee × max(1, months to pass) for budgeting.",
            )
        else:
            fe1, fe2, fe3 = st.columns(3)
            with fe1:
                challenge_fee = st.number_input("Challenge fee ($)", min_value=0, step=1, format="%d", key="challenge_fee")
            with fe2:
                activation_fee = st.number_input(
                    "Activation fee ($)",
                    min_value=0,
                    step=1,
                    format="%d",
                    key="activation_fee",
                    help="Paid after you pass the challenge. Included in expected cost to reach funded.",
                )
            with fe3:
                monthly_recurring = st.checkbox(
                    "Monthly recurring",
                    key="monthly_recurring",
                    help="If on, each challenge attempt is billed as fee × max(1, months to pass) for budgeting.",
                )

    with st.container(border=True):
        st.markdown(
            '<p class="pp-input-section-title pp-funded">Funded account</p>',
            unsafe_allow_html=True,
        )
        st.caption("Parameters used **after** you pass the challenge — funded-phase simulation only.")

        _tpl_fd = st.session_state.get("config_template", TEMPLATE_CUSTOM)
        if _tpl_fd == TEMPLATE_FOREX:
            funded_risk_per_trade = st.number_input(
                "Funded Risk per Trade ($)",
                min_value=0,
                step=10,
                format="%d",
                key="funded_risk_per_trade",
                help="Used in the funded phase only; challenge phase uses Risk per Trade.",
            )
            funded_min_days = 0
            funded_consistency_max = 100
            st.caption(
                "Forex: no minimum winning days or funded consistency gate — only **funded risk per trade** applies."
            )
        else:
            fp1, fp2, fp3 = st.columns(3)
            with fp1:
                funded_risk_per_trade = st.number_input(
                    "Funded Risk per Trade ($)",
                    min_value=0,
                    step=10,
                    format="%d",
                    key="funded_risk_per_trade",
                    help="Used in the funded phase only; challenge phase uses Risk per Trade.",
                )
            with fp2:
                funded_min_days = st.number_input(
                    "Minimum days",
                    min_value=0,
                    max_value=365,
                    step=1,
                    format="%d",
                    key="funded_min_days",
                    help=(
                        "Custom: winning-day gate (min_winning_days). "
                        "Topstep Standard: five $150+ winning days per payout cycle (default 5). "
                        "Topstep Consistency: minimum calendar days per cycle with 40% consistency (default 3, at least 3)."
                    ),
                )
            with fp3:
                funded_consistency_max = st.number_input(
                    "Funded consistency cap (%)",
                    min_value=1,
                    max_value=100,
                    step=1,
                    format="%d",
                    key="funded_consistency_max",
                    help="Max daily profit ÷ total profit must stay below this value in funded phase.",
                )

        _ct = st.session_state.get("config_template", TEMPLATE_CUSTOM)
        if _ct == TEMPLATE_CUSTOM:
            st.caption(
                "Custom template: legacy funded payout model (buffer, cumulative winning days, optional consistency on "
                "lifetime funded profit). 90/10 split and withdrawal-style caps can be enabled in code via `funded_params`."
            )
        elif _ct == TEMPLATE_FOREX:
            st.caption(
                "Forex template: funded phase uses your **risk per trade** and standard payout split/caps only "
                "(no Topstep Express payout policy)."
            )
        else:
            st.caption(
                "Topstep Express template: repeat payout cycles (counters reset after each payout), "
                "profit-since-last-payout after the first payout, min request $125, $30 withdrawal processing fee, "
                "caps $5k Standard / $6k Consistency. Policy reference: "
                "https://help.topstep.com/en/articles/8284233-topstep-payout-policy"
            )

        run = st.button("▶️", type="primary", help="Run Simulation", key="run_single")

with col_outputs:
    st.markdown('<p class="pp-column-title">Results</p>', unsafe_allow_html=True)
    with st.container(border=True):
        if not _ENGINE_SUPPORTS_FUNDED:
            st.warning(
                "This deployment is using an older `prop_strategy_engine` without `funded_params`. "
                "Redeploy from the latest `main` so funded EV / payout metrics match localhost."
            )
        if "last_result" not in st.session_state:
            st.session_state.last_result = None
        if "last_elapsed" not in st.session_state:
            st.session_state.last_elapsed = None
        if "last_time_to_pass_days" not in st.session_state:
            st.session_state.last_time_to_pass_days = None
        if "last_expected_days_to_pass_retries" not in st.session_state:
            st.session_state.last_expected_days_to_pass_retries = None
        if "last_cost_metrics" not in st.session_state:
            st.session_state.last_cost_metrics = None
        if "last_projected_challenge_fees" not in st.session_state:
            st.session_state.last_projected_challenge_fees = None
        if "last_error" not in st.session_state:
            st.session_state.last_error = None
        if run:
            if uploaded_file is None:
                st.session_state.last_error = "Please upload a CSV with 'rPnL' and 'dateStart'."
                st.session_state.last_result = None
            else:
                try:
                    df = _load_df(uploaded_file.getvalue())
                    dt = pd.to_datetime(df["dateStart"], errors="coerce", utc=False)
                    if dt.isna().any():
                        raise ValueError("CSV has invalid dateStart values.")

                    trades_per_day = df.groupby(dt.dt.normalize()).size()
                    avg_trades_per_day = float(trades_per_day.mean()) if len(trades_per_day) else float("nan")

                    _tpl_run = st.session_state.get("config_template", TEMPLATE_CUSTOM)
                    mc_kw = dict(
                        df=df,
                        starting_balance=float(account_size),
                        risk_per_trade_dollar=float(risk_per_trade),
                        profit_target_dollar=float(profit_target),
                        max_loss_dollar=float(max_loss),
                        daily_loss_dollar=float(daily_loss),
                        drawdown_type=drawdown_type,
                        challenge_fee=float(challenge_fee),
                        consistency_rule_pct=float(consistency_rule_pct),
                        equity_paths=50,
                        funded_risk_per_trade_dollar=float(funded_risk_per_trade),
                    )
                    if _tpl_run == TEMPLATE_FOREX:
                        p1 = float(st.session_state.get("profit_target_phase1", 0))
                        p2 = float(st.session_state.get("profit_target_phase2", 0))
                        mc_kw["profit_target_phase1_dollar"] = p1
                        mc_kw["profit_target_phase2_dollar"] = p2
                        mc_kw["profit_target_dollar"] = p1  # validation fallback
                        mc_kw["apply_challenge_consistency_gate"] = False

                    _tpl = _tpl_run
                    funded_params = _funded_params_for_template(
                        _tpl,
                        int(funded_min_days),
                        float(funded_consistency_max),
                    )
                    with st.spinner("Running challenge + funded Monte Carlo (50,000 sims)..."):
                        start = time.time()
                        result = _run_prop_strategy_monte_carlo(
                            **mc_kw,
                            funded_params=funded_params,
                        )
                        elapsed = time.time() - start

                    time_to_pass_days = (
                        float(result["avg_trades_to_pass"]) / avg_trades_per_day
                        if np.isfinite(result["avg_trades_to_pass"]) and np.isfinite(avg_trades_per_day) and avg_trades_per_day > 0
                        else float("nan")
                    )
                    _et = float(result.get("expected_trades_to_first_pass", float("nan")))
                    if np.isfinite(_et) and np.isfinite(avg_trades_per_day) and avg_trades_per_day > 0:
                        expected_days_to_pass_retries = float(_et) / avg_trades_per_day
                    elif np.isposinf(_et):
                        expected_days_to_pass_retries = float("inf")
                    else:
                        expected_days_to_pass_retries = float("nan")
                    pass_rate = float(result["pass_rate"])
                    fee = float(challenge_fee)

                    # 95% confidence budgeting:
                    # n = log(1 - 0.95) / log(1 - pass_rate)
                    # budget = n * cost_per_attempt
                    if pass_rate <= 0:
                        n95 = float("inf")
                    elif pass_rate >= 1:
                        n95 = 1.0
                    else:
                        n95 = float(np.log(0.05) / np.log(1.0 - pass_rate))

                    time_to_pass_months = (
                        float(time_to_pass_days) / 30.0
                        if np.isfinite(time_to_pass_days)
                        else float("nan")
                    )
                    if monthly_recurring and np.isfinite(time_to_pass_months):
                        cost_per_attempt = fee * max(1.0, time_to_pass_months)
                    elif monthly_recurring and not np.isfinite(time_to_pass_months):
                        cost_per_attempt = float("nan")
                    else:
                        cost_per_attempt = fee

                    confidence_budget_95 = (
                        float("inf")
                        if (not np.isfinite(n95) or not np.isfinite(cost_per_attempt))
                        else n95 * cost_per_attempt
                    )
                    if pass_rate > 1e-12 and np.isfinite(cost_per_attempt):
                        attempts_needed = int(np.ceil(1.0 / pass_rate))
                        projected_challenge_fees = float(attempts_needed) * float(cost_per_attempt)
                    else:
                        projected_challenge_fees = float("inf")

                    activation_fee_f = 0.0 if _tpl_run == TEMPLATE_FOREX else float(activation_fee)
                    if np.isfinite(projected_challenge_fees):
                        expected_cost_to_funded = float(projected_challenge_fees) + activation_fee_f
                    else:
                        expected_cost_to_funded = float("inf")

                    st.session_state.last_run_template = _tpl_run
                    st.session_state.last_result = result
                    st.session_state.last_elapsed = elapsed
                    st.session_state.last_time_to_pass_days = time_to_pass_days
                    st.session_state.last_expected_days_to_pass_retries = expected_days_to_pass_retries
                    st.session_state.last_cost_metrics = {
                        "confidence_budget_95": confidence_budget_95,
                        "time_to_pass_months": time_to_pass_months,
                        "cost_per_attempt": cost_per_attempt,
                        "expected_challenge_spend": projected_challenge_fees,
                        "expected_cost_to_funded": expected_cost_to_funded,
                        "activation_fee": activation_fee_f,
                    }
                    st.session_state.last_projected_challenge_fees = projected_challenge_fees
                    st.session_state.last_error = None
                except Exception as e:
                    st.session_state.last_error = str(e)
                    st.session_state.last_result = None
                    st.session_state.last_run_template = None
                    st.session_state.last_cost_metrics = None
                    st.session_state.last_expected_days_to_pass_retries = None
                    st.session_state.last_projected_challenge_fees = None
        if st.session_state.last_error:
            st.error(st.session_state.last_error)
        elif st.session_state.last_elapsed is not None:
            st.caption(f"Simulation complete in {st.session_state.last_elapsed:.2f}s.")
        else:
            st.caption("Run a simulation to populate metrics and paths.")

        budget_help = (
            "The total capital required to have a 95% chance of passing at least one account, "
            "accounting for the length of time it takes to pass and your specific failure rate."
        )
        projected_fees_help = (
            "Cost of attempts to pass using rounded-up attempts: ceil(1 / pass rate) × cost per attempt. "
            "Cost per attempt = challenge fee × months if Monthly Recurring is on, else flat fee."
        )
        expected_cost_help = (
            "Expected $ to get **funded**: cost of attempts to pass + **activation fee** "
            "after pass. Same as Cost of Attempts to Pass + Activation Fee from inputs."
        )
        expected_cost_help_forex = (
            "Expected $ to reach funded (forex): **challenge attempt costs only** — no activation fee."
        )

        tab_challenge, tab_funded, tab_metrics = st.tabs(["Challenge", "Funded", "Metrics"])

        with tab_challenge:
            st.markdown("##### Challenge pass dynamics")
            c0, c1, c2 = st.columns(3)
            if st.session_state.last_result is None:
                c0.metric("PASS RATE (%)", "—")
                c1.metric("FAIL RATE (%)", "—")
                c2.metric("AVG TIME TO PASS (DAYS)", "—")
                c3, c4 = st.columns(2)
                c3.metric("EXPECTED TIME TO PASS (DAYS)", "—")
                c4.metric("COST OF ATTEMPTS TO PASS ($)", "—")
            else:
                result = st.session_state.last_result
                c0.metric("PASS RATE (%)", _fmt_percent(float(result["pass_rate"])))
                c1.metric("FAIL RATE (%)", _fmt_percent(float(result["ruin_rate"])))
                ttp = st.session_state.last_time_to_pass_days
                c2.metric(
                    "AVG TIME TO PASS (DAYS)",
                    _fmt_days(ttp) if ttp is not None else "n/a",
                    help=(
                        "Average calendar days for one successful challenge attempt only "
                        "(does not include failed attempts)."
                    ),
                )
                c3, c4 = st.columns(2)
                etr = st.session_state.last_expected_days_to_pass_retries
                c3.metric(
                    "EXPECTED TIME TO PASS (DAYS)",
                    _fmt_days(etr),
                    help=(
                        "Expected total calendar days until your first challenge pass, including retries. "
                        "Model: independent attempts; ((1 − pass rate) / pass rate) × avg trades to fail "
                        "+ avg trades to pass, ÷ trades per day from CSV."
                    ),
                )
                _pf = st.session_state.last_projected_challenge_fees
                c4.metric(
                    "COST OF ATTEMPTS TO PASS ($)",
                    _fmt_dollars(_pf) if _pf is not None and np.isfinite(float(_pf)) else ("∞" if _pf is not None and np.isposinf(float(_pf)) else "n/a"),
                    help=projected_fees_help,
                )
                st.caption(f"Monte Carlo: **{int(result.get('n_sims', 0)):,}** simulations.")

            fig = go.Figure()
            if st.session_state.last_result is not None:
                result = st.session_state.last_result
                target_balance = float(result["target_balance"])
                max_drawdown_balance = float(result["max_drawdown_balance"])
                plot_paths = result["plot_paths_equity"]
                plot_passed = result["plot_paths_passed"]
                for i, equity_path in enumerate(plot_paths):
                    passed = bool(plot_passed[i]) if i < len(plot_passed) else False
                    color = "#00FF41" if passed else "#FF5C5C"
                    x = np.arange(len(equity_path), dtype=int)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=equity_path,
                            mode="lines",
                            line=dict(color=color, width=2),
                            opacity=0.55,
                            showlegend=False,
                        )
                    )

                max_path_steps = max((len(p) - 1 for p in plot_paths), default=0)
                x_line_end = max(1, max_path_steps)
                fig.add_trace(
                    go.Scatter(
                        x=[0, x_line_end],
                        y=[target_balance, target_balance],
                        mode="lines",
                        line=dict(color="#38BDF8", width=2),
                        name="Target",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, x_line_end],
                        y=[max_drawdown_balance, max_drawdown_balance],
                        mode="lines",
                        line=dict(color="#F59E0B", width=2),
                        name="Loss Limit",
                    )
                )
            else:
                x_line_end = 100
                fig.add_trace(
                    go.Scatter(
                        x=[0, x_line_end],
                        y=[0, 0],
                        mode="lines",
                        line=dict(color="rgba(255,255,255,0.12)", width=1),
                        name="No data",
                    )
                )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6E6E6"),
                height=420,
                margin=dict(l=8, r=12, t=40, b=8),
                xaxis_title="Step # (simulated trades)",
                yaxis_title="Equity / Balance",
                xaxis_range=[0, x_line_end],
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False, automargin=True),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False, automargin=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_funded:
            st.markdown("##### Funded economics")
            _fund_tpl = st.session_state.get("last_run_template") or st.session_state.get(
                "config_template", TEMPLATE_CUSTOM
            )
            st.caption(
                "Funded phase uses the **same** max loss ($), daily loss ($), and drawdown type (static / trailing) "
                "as the challenge. **90/10** split on withdrawals unless your template says otherwise."
            )
            if _fund_tpl == TEMPLATE_FOREX:
                st.caption("Forex: no Topstep payout rules. Drawdown lock after first payout still applies in the engine.")
            else:
                st.caption(
                    "**Drawdown lock:** After the first funded payout request, max-loss floor locks to the funded "
                    "starting balance (e.g. $50k Topstep) and does not trail lower."
                )
            if st.session_state.last_result is None or "net_ev" not in st.session_state.last_result:
                if _fund_tpl == TEMPLATE_FOREX:
                    fx0, fx1, fx2 = st.columns(3)
                    fx0.metric("PAYOUT SUCCESS (CONDITIONAL)", "—")
                    fx1.metric("NET EV ($)", "—")
                    fx2.metric("AVG TOTAL PAYOUT ($)", "—")
                    fxa0, fxa1 = st.columns(2)
                    fxa0.metric("BREAKEVEN ATTEMPT COST (FUNDED START, DAYS)", "—")
                    fxa1.metric("BREAKEVEN ATTEMPT COST (CHALLENGE START, DAYS)", "—")
                else:
                    f0, f1, f2 = st.columns(3)
                    f0.metric("PAYOUT SUCCESS (CONDITIONAL)", "—")
                    f1.metric("FUNDED BLOWUP BEFORE PAYOUT", "—")
                    f2.metric("PAYOUT SUCCESS (ABSOLUTE)", "—")
                    g0, g1, g2 = st.columns(3)
                    g0.metric("AVG TOTAL PAYOUT ($)", "—")
                    g1.metric("NET EV ($)", "—")
                    g2.metric("EXPECTED COST TO FUNDED ($)", "—")
                    t0, t1 = st.columns(2)
                    t0.metric("TIME TO BREAKEVEN (FUNDED START, DAYS)", "—")
                    t1.metric("TIME TO BREAKEVEN (CHALLENGE START, DAYS)", "—")
                    ta0, ta1 = st.columns(2)
                    ta0.metric("BREAKEVEN ATTEMPT COST (FUNDED START, DAYS)", "—")
                    ta1.metric("BREAKEVEN ATTEMPT COST (CHALLENGE START, DAYS)", "—")
                st.caption("Run the simulation to populate funded metrics and the payout chart.")
            else:
                fr = st.session_state.last_result
                costs = st.session_state.last_cost_metrics or {}
                _ecf = costs.get("expected_cost_to_funded", float("nan"))
                ev_help = (
                    "Net EV = (pass rate × avg. total user payout) − (challenge fail rate × challenge fee). "
                    "Avg. payout is over simulations that passed the challenge."
                )
                _p_abs = fr.get("payout_success_rate_absolute", fr.get("payout_probability"))
                _p_cond = fr.get("payout_success_rate_conditional")
                if _p_cond is None and _p_abs is not None:
                    p_pass = float(fr.get("pass_rate", 0.0))
                    _p_cond = float(_p_abs) / p_pass if p_pass > 1e-12 else 0.0
                _p_abs_f = float(_p_abs) if _p_abs is not None else 0.0
                _p_cond_f = float(_p_cond) if _p_cond is not None else 0.0
                _blow_cond = float(fr.get("funded_blowup_before_payout_rate_conditional", float("nan")))
                _net_ev = float(fr.get("net_ev", float("nan")))
                _avg_pay = float(fr.get("avg_total_payout_per_challenge_pass", float("nan")))
                _etr_days = st.session_state.last_expected_days_to_pass_retries
                _breakeven_days_from_funded = float("nan")
                _breakeven_days_from_challenge = float("nan")
                _breakeven_help_funded = (
                    "Conditional mean funded survival days until total funded payouts ≥ **expected cost to funded** "
                    "(attempt fees + activation when applicable), among simulations that reach that threshold."
                )
                _breakeven_help_challenge = (
                    "Same as funded-start full-cost breakeven, plus expected days to first pass with retries "
                    "(E[attempts] × challenge fee path from the Challenge tab)."
                )
                _eca = costs.get("expected_challenge_spend", float("nan"))
                _breakeven_attempts_funded = float("nan")
                _breakeven_attempts_challenge = float("nan")
                _breakeven_attempt_help_funded = (
                    "Conditional mean funded survival days until total funded payouts ≥ **cost of attempts to pass** "
                    "only (Challenge tab: COST OF ATTEMPTS TO PASS ($)), among simulations that reach that threshold."
                )
                _breakeven_attempt_help_challenge = (
                    "Attempt-cost breakeven from funded start, plus expected days to first pass with retries."
                )
                _payout_arr = fr.get("funded_total_payout_per_sim")
                _fund_days_arr = fr.get("funded_survival_days_per_sim")
                if _payout_arr is not None and _fund_days_arr is not None:
                    payout_arr = np.asarray(_payout_arr, dtype=float)
                    fund_days_arr = np.asarray(_fund_days_arr, dtype=float)
                    valid = np.isfinite(payout_arr) & np.isfinite(fund_days_arr)
                    if _ecf is not None and np.isfinite(float(_ecf)):
                        mask_full = valid & (payout_arr >= float(_ecf))
                        if np.any(mask_full):
                            _breakeven_days_from_funded = float(np.mean(fund_days_arr[mask_full]))
                            if _etr_days is not None and np.isfinite(float(_etr_days)):
                                _breakeven_days_from_challenge = float(_etr_days) + _breakeven_days_from_funded
                    if _eca is not None and np.isfinite(float(_eca)):
                        mask_att = valid & (payout_arr >= float(_eca))
                        if np.any(mask_att):
                            _breakeven_attempts_funded = float(np.mean(fund_days_arr[mask_att]))
                            if _etr_days is not None and np.isfinite(float(_etr_days)):
                                _breakeven_attempts_challenge = float(_etr_days) + _breakeven_attempts_funded

                _run_tpl = st.session_state.get("last_run_template") or st.session_state.get(
                    "config_template", TEMPLATE_CUSTOM
                )
                if _run_tpl == TEMPLATE_FOREX:
                    fx0, fx1, fx2 = st.columns(3)
                    fx0.metric(
                        "PAYOUT SUCCESS (CONDITIONAL)",
                        _fmt_percent(_p_cond_f),
                        help="Among challenge passes, share that reach ≥1 funded payout.",
                    )
                    fx1.metric(
                        "NET EV ($)",
                        _fmt_dollars(_net_ev) if np.isfinite(_net_ev) else "n/a",
                        help=ev_help,
                    )
                    fx2.metric(
                        "AVG TOTAL PAYOUT ($)",
                        _fmt_dollars(_avg_pay) if np.isfinite(_avg_pay) else "n/a",
                        help="User withdrawals in funded phase ÷ challenge passes (after split rules).",
                    )
                    fxa0, fxa1 = st.columns(2)
                    fxa0.metric(
                        "BREAKEVEN ATTEMPT COST (FUNDED START, DAYS)",
                        _fmt_days(_breakeven_attempts_funded),
                        help=_breakeven_attempt_help_funded,
                    )
                    fxa1.metric(
                        "BREAKEVEN ATTEMPT COST (CHALLENGE START, DAYS)",
                        _fmt_days(_breakeven_attempts_challenge),
                        help=_breakeven_attempt_help_challenge,
                    )
                else:
                    f0, f1, f2 = st.columns(3)
                    f0.metric(
                        "PAYOUT SUCCESS (CONDITIONAL)",
                        _fmt_percent(_p_cond_f),
                        help="Among challenge passes, share that reach ≥1 funded payout.",
                    )
                    f1.metric(
                        "FUNDED BLOWUP BEFORE PAYOUT",
                        _fmt_percent(_blow_cond) if np.isfinite(_blow_cond) else "n/a",
                        help=(
                            "Among challenge passes, probability of failing the funded account "
                            "before receiving the first payout. Sensitive to funded risk per trade."
                        ),
                    )
                    f2.metric(
                        "PAYOUT SUCCESS (ABSOLUTE)",
                        _fmt_percent(_p_abs_f),
                        help="(Simulations with ≥1 funded payout) ÷ all simulations.",
                    )
                    g0, g1, g2 = st.columns(3)
                    g0.metric(
                        "AVG TOTAL PAYOUT ($)",
                        _fmt_dollars(_avg_pay) if np.isfinite(_avg_pay) else "n/a",
                        help="User withdrawals in funded phase ÷ challenge passes (after split rules).",
                    )
                    g1.metric(
                        "NET EV ($)",
                        _fmt_dollars(_net_ev) if np.isfinite(_net_ev) else "n/a",
                        help=ev_help,
                    )
                    g2.metric(
                        "EXPECTED COST TO FUNDED ($)",
                        _fmt_dollars(_ecf) if _ecf is not None and np.isfinite(float(_ecf)) else ("∞" if _ecf is not None and np.isposinf(float(_ecf)) else "n/a"),
                        help=expected_cost_help,
                    )
                    t0, t1 = st.columns(2)
                    t0.metric(
                        "TIME TO BREAKEVEN (FUNDED START, DAYS)",
                        _fmt_days(_breakeven_days_from_funded),
                        help=_breakeven_help_funded,
                    )
                    t1.metric(
                        "TIME TO BREAKEVEN (CHALLENGE START, DAYS)",
                        _fmt_days(_breakeven_days_from_challenge),
                        help=_breakeven_help_challenge,
                    )
                    ta0, ta1 = st.columns(2)
                    ta0.metric(
                        "BREAKEVEN ATTEMPT COST (FUNDED START, DAYS)",
                        _fmt_days(_breakeven_attempts_funded),
                        help=_breakeven_attempt_help_funded,
                    )
                    ta1.metric(
                        "BREAKEVEN ATTEMPT COST (CHALLENGE START, DAYS)",
                        _fmt_days(_breakeven_attempts_challenge),
                        help=_breakeven_attempt_help_challenge,
                    )

                    _ecs = costs.get("expected_challenge_spend", float("nan"))
                    _act = costs.get("activation_fee", float("nan"))
                    _b95 = float(costs.get("confidence_budget_95", float("nan")))
                    b95_txt = _fmt_dollars(_b95) if np.isfinite(_b95) else "inf"
                    if np.isfinite(float(_ecs)) and _act is not None and np.isfinite(float(_act)):
                        st.caption(
                            f"**Breakdown:** challenge spend (E[attempts] × fee) **{_fmt_dollars(float(_ecs))}** + activation **{_fmt_dollars(float(_act))}** "
                            f"· *95% confidence budget (reference):* **{b95_txt}**"
                        )
                    else:
                        st.caption(f"*95% confidence budget (reference):* **{b95_txt}** — {budget_help}")

                ph = fr.get("payout_histogram_values")
                if ph is not None and len(ph) > 0:
                    fig_hist = go.Figure(
                        go.Histogram(
                            x=np.asarray(ph, dtype=float),
                            nbinsx=30,
                            marker=dict(color="#00FF41", line=dict(width=0)),
                            opacity=0.85,
                        )
                    )
                    fig_hist.update_layout(
                        title=dict(
                            text="Payout distribution ($ total withdrawn, funded phase)",
                            font=dict(color="#E6E6E6", size=14),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E6E6E6"),
                        height=240,
                        margin=dict(l=8, r=8, t=28, b=6),
                        xaxis_title="Total payout ($)",
                        yaxis_title="Count",
                        showlegend=False,
                    )
                    fig_hist.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
                    fig_hist.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
                    st.plotly_chart(fig_hist, use_container_width=True)

        with tab_metrics:
            st.markdown("##### CSV trade profile (historical)")
            st.caption("From **rPnL** / **dateStart** — not Monte Carlo. Win/loss ratio uses mean win ÷ |mean loss| (scale-free).")
            if uploaded_file is None:
                m0, m1, m2, m3 = st.columns(4)
                m0.metric("WIN RATE (%)", "—")
                m1.metric("PROFIT FACTOR", "—")
                m2.metric("WIN / LOSS RATIO", "—")
                m3.metric("TRADES / DAY", "—")
                m4, = st.columns(1)
                m4.metric("TRADE COUNT", "—")
            else:
                _df_edge = _load_df(uploaded_file.getvalue())
                sm = _strategy_metrics_from_df(_df_edge)
                if "rPnL" in _df_edge.columns:
                    r = pd.to_numeric(_df_edge["rPnL"], errors="coerce").dropna()
                    n_tr = int(len(r))
                else:
                    n_tr = 0
                pf = sm["profit_factor"]
                if isinstance(pf, float) and np.isposinf(pf):
                    pf_s = "∞"
                elif np.isfinite(pf):
                    pf_s = f"{pf:.2f}"
                else:
                    pf_s = "n/a"
                wlr = sm["win_loss_ratio"]
                wlr_s = f"{wlr:.2f}" if np.isfinite(wlr) else "n/a"

                m0, m1, m2, m3 = st.columns(4)
                m0.metric(
                    "WIN RATE (%)",
                    _fmt_percent(sm["win_rate"]) if np.isfinite(sm["win_rate"]) else "n/a",
                    help="Share of rows with rPnL > 0.",
                )
                m1.metric(
                    "PROFIT FACTOR",
                    pf_s,
                    help="Sum of winning rPnL ÷ |sum of losing rPnL|.",
                )
                m2.metric(
                    "WIN / LOSS RATIO",
                    wlr_s,
                    help="Mean winning rPnL ÷ |mean losing rPnL| (no $ emphasis — useful with compounded series).",
                )
                m3.metric(
                    "TRADES / DAY",
                    f"{sm['trades_per_calendar_day']:.2f}" if np.isfinite(sm["trades_per_calendar_day"]) else "n/a",
                    help="Total trades ÷ distinct calendar days in dateStart.",
                )
                m4, = st.columns(1)
                m4.metric("TRADE COUNT", f"{n_tr:,}" if n_tr else "—")

