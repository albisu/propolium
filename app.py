import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prop_strategy_engine import run_prop_strategy_monte_carlo


st.set_page_config(page_title="Prop Firm Strategy Simulation Engine", layout="wide")


def _fmt_percent(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100.0:.2f}%"


def _fmt_dollars(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    :root {
        --bg-main: #000000;
        --bg-secondary: #111111;
        --text-main: #E6E6E6;
        --accent: #00FF41;
        --danger: #FF4D4D;
        --page-pad-x: clamp(1.25rem, 4vw, 2.75rem);
        --page-pad-y-top: 1rem;
        --page-pad-y-bottom: 1.5rem;
        --col-gap: 1.5rem;
    }
    html, body {
        font-family: "Inter", system-ui, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
        background-color: #000000 !important;
    }
    .stApp {
        font-family: "Inter", system-ui, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
        background-color: #000000 !important;
        color: var(--text-main);
    }
    /* Remove default white wrappers and borders */
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="stExpander"],
    [data-testid="stAppViewContainer"],
    section.main,
    section.main > div,
    .block-container,
    div[data-testid="stMetric"],
    [data-testid="stForm"] {
        background: #000000 !important;
        border: none !important;
        box-shadow: none !important;
    }
    .block-container {
        padding: var(--page-pad-y-top) var(--page-pad-x) var(--page-pad-y-bottom) var(--page-pad-x) !important;
        max-width: 100% !important;
    }
    /* Main content column: consistent horizontal rhythm */
    section.main > div {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    /* Wider gap between Inputs | Outputs columns */
    [data-testid="stHorizontalBlock"] {
        gap: var(--col-gap) !important;
        align-items: flex-start !important;
    }
    .stMarkdown, .stText, .stCaption, label, p, span {
        color: var(--text-main) !important;
    }
    .subcard-heading {
        margin: 0 0 8px 0;
        font-weight: 600;
        color: #D9E0EE;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #1f1f1f !important;
        border-radius: 8px;
        background: #101010 !important;
        padding: 8px 10px;
    }
    div[data-testid="stMetricValue"] {
        color: var(--accent) !important;
    }
    div.stButton > button {
        width: 56px;
        height: 40px;
        min-width: 56px;
        padding: 0 !important;
        border-radius: 8px;
        background: #00FF41 !important;
        color: #000000 !important;
        border: 1px solid #00FF41 !important;
        font-weight: 700;
        font-size: 18px;
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
    }
    /* Hide number input + / - controls (spinners) */
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type=number] {
        -moz-appearance: textfield;
    }
    input, textarea, select, button {
        font-family: "Inter", system-ui, ui-sans-serif, sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_df(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


TEMPLATES = {
    "Custom": None,
    "Topstep 50K": {
        "starting_equity": 50000,
        "profit_target": 3000,
        "max_loss": 2000,
        "daily_loss": 1000,
        "drawdown_type": "Trailing",
        "challenge_fee": 165,
        "consistency_rule_pct": 40,
    },
    "MFFU 50K (Flex)": {
        "starting_equity": 50000,
        "profit_target": 3000,
        "max_loss": 2000,
        "daily_loss": 0,
        "drawdown_type": "Trailing",
        "challenge_fee": 107,
        "consistency_rule_pct": 50,
    },
    "Alpha Futures 50K": {
        "starting_equity": 50000,
        "profit_target": 3000,
        "max_loss": 2000,
        "daily_loss": 1000,
        "drawdown_type": "Static",
        "challenge_fee": 79,
        "consistency_rule_pct": 50,
    },
    "FTMO 100K (2-Step)": {
        "starting_equity": 100000,
        "profit_target": 10000,
        "max_loss": 10000,
        "daily_loss": 5000,
        "drawdown_type": "Static",
        "challenge_fee": 580,
        "consistency_rule_pct": 0,
    },
    "FTMO 100K (1-Step)": {
        "starting_equity": 100000,
        "profit_target": 10000,
        "max_loss": 6000,
        "daily_loss": 3000,
        "drawdown_type": "Static",
        "challenge_fee": 650,
        "consistency_rule_pct": 0,
    },
    "The5ers 100K (High Stake)": {
        "starting_equity": 100000,
        "profit_target": 10000,
        "max_loss": 10000,
        "daily_loss": 5000,
        "drawdown_type": "Static",
        "challenge_fee": 495,
        "consistency_rule_pct": 0,
    },
    "FundingPips 100K (1-Step)": {
        "starting_equity": 100000,
        "profit_target": 12000,
        "max_loss": 6000,
        "daily_loss": 4000,
        "drawdown_type": "Static",
        "challenge_fee": 399,
        "consistency_rule_pct": 0,
    },
}


def _init_defaults() -> None:
    st.session_state.setdefault("starting_equity", 100000)
    st.session_state.setdefault("risk_per_trade", 500)
    st.session_state.setdefault("profit_target", 8000)
    st.session_state.setdefault("max_loss", 10000)
    st.session_state.setdefault("daily_loss", 5000)
    st.session_state.setdefault("drawdown_type", "Static")
    st.session_state.setdefault("challenge_fee", 500)
    st.session_state.setdefault("consistency_rule_pct", 40)
    st.session_state.setdefault("monthly_recurring", False)
    st.session_state.setdefault("template_name", "Custom")
    st.session_state.setdefault("_last_template_applied", "Custom")


_init_defaults()


col_inputs, col_outputs = st.columns([1, 1.2], gap="large")

with col_inputs:
    st.markdown("**Data Upload**")
    uploaded_file = st.file_uploader("Upload trade history CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("**Challenge Parameters**")
    selected_template = st.selectbox("Load Template", list(TEMPLATES.keys()), key="template_name")
    if selected_template != "Custom" and st.session_state["_last_template_applied"] != selected_template:
        tpl = TEMPLATES[selected_template]
        st.session_state["starting_equity"] = tpl["starting_equity"]
        st.session_state["profit_target"] = tpl["profit_target"]
        st.session_state["max_loss"] = tpl["max_loss"]
        st.session_state["daily_loss"] = tpl["daily_loss"]
        st.session_state["drawdown_type"] = tpl["drawdown_type"]
        st.session_state["challenge_fee"] = tpl["challenge_fee"]
        st.session_state["consistency_rule_pct"] = tpl["consistency_rule_pct"]
        st.session_state["_last_template_applied"] = selected_template
        st.rerun()
    elif selected_template == "Custom":
        st.session_state["_last_template_applied"] = "Custom"

    # 4 columns x 2 rows (dollar-only inputs)
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        account_size = st.number_input("Starting Equity ($)", min_value=1, step=1000, format="%d", key="starting_equity")
    with r1c2:
        risk_per_trade = st.number_input("Risk per Trade ($)", min_value=0, step=10, format="%d", key="risk_per_trade")
    with r1c3:
        profit_target = st.number_input("Profit Target ($)", min_value=0, step=100, format="%d", key="profit_target")
    with r1c4:
        max_loss = st.number_input("Max Loss ($)", min_value=0, step=100, format="%d", key="max_loss")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        daily_loss = st.number_input("Daily Loss ($)", min_value=0, step=100, format="%d", key="daily_loss")
    with r2c2:
        drawdown_type = st.selectbox("Drawdown Type", options=["Static", "Trailing"], key="drawdown_type")
    with r2c3:
        challenge_fee = st.number_input("Challenge Fee ($)", min_value=0, step=50, format="%d", key="challenge_fee")
        monthly_recurring = st.checkbox("Monthly Recurring", key="monthly_recurring")
    with r2c4:
        consistency_rule_pct = st.number_input("Consistency Rule (%)", min_value=0, step=1, format="%d", key="consistency_rule_pct")

    run = st.button("▶️", type="primary", help="Run Simulation")

with col_outputs:
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_elapsed" not in st.session_state:
        st.session_state.last_elapsed = None
    if "last_time_to_pass_days" not in st.session_state:
        st.session_state.last_time_to_pass_days = None
    if "last_cost_metrics" not in st.session_state:
        st.session_state.last_cost_metrics = None
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

                with st.spinner("Running 50,000 Monte Carlo simulations..."):
                    start = time.time()
                    result = run_prop_strategy_monte_carlo(
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
                    )
                    elapsed = time.time() - start

                time_to_pass_days = (
                    float(result["avg_trades_to_pass"]) / avg_trades_per_day
                    if np.isfinite(result["avg_trades_to_pass"]) and np.isfinite(avg_trades_per_day) and avg_trades_per_day > 0
                    else float("nan")
                )
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

                st.session_state.last_result = result
                st.session_state.last_elapsed = elapsed
                st.session_state.last_time_to_pass_days = time_to_pass_days
                st.session_state.last_cost_metrics = {
                    "confidence_budget_95": confidence_budget_95,
                    "time_to_pass_months": time_to_pass_months,
                    "cost_per_attempt": cost_per_attempt,
                }
                st.session_state.last_error = None
            except Exception as e:
                st.session_state.last_error = str(e)
                st.session_state.last_result = None
                st.session_state.last_cost_metrics = None

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
    metric_row_left, metric_row_right = st.columns([2.6, 1.2], gap="small")
    with metric_row_left:
        c0, c1, c2 = st.columns(3, gap="small")
        if st.session_state.last_result is None:
            c0.metric("PASS RATE", "-")
            c1.metric("FAIL RATE", "-")
            c2.metric("TIME TO PASS (DAYS)", "-")
        else:
            result = st.session_state.last_result
            c0.metric("PASS RATE", _fmt_percent(float(result["pass_rate"])))
            c1.metric("FAIL RATE", _fmt_percent(float(result["ruin_rate"])))
            ttp = st.session_state.last_time_to_pass_days
            c2.metric("TIME TO PASS (DAYS)", f"{ttp:.2f}" if ttp is not None and np.isfinite(ttp) else "n/a")
    with metric_row_right:
        if st.session_state.last_result is None:
            st.metric("95% CONFIDENCE BUDGET ($)", "-", help=budget_help)
        else:
            costs = st.session_state.last_cost_metrics or {}
            budget95 = costs.get("confidence_budget_95", float("nan"))
            st.metric(
                "95% CONFIDENCE BUDGET ($)",
                _fmt_dollars(budget95) if np.isfinite(budget95) else "inf",
                help=budget_help,
            )

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
        fig.add_trace(go.Scatter(x=[0, x_line_end], y=[target_balance, target_balance], mode="lines", line=dict(color="#38BDF8", width=2), name="Target"))
        fig.add_trace(go.Scatter(x=[0, x_line_end], y=[max_drawdown_balance, max_drawdown_balance], mode="lines", line=dict(color="#F59E0B", width=2), name="Loss Limit"))
    else:
        x_line_end = 100
        fig.add_trace(go.Scatter(x=[0, x_line_end], y=[0, 0], mode="lines", line=dict(color="rgba(255,255,255,0.12)", width=1), name="No data"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E6E6"),
        height=560,
        xaxis_title="Step # (simulated trades)",
        yaxis_title="Equity / Balance",
        xaxis_range=[0, x_line_end],
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

