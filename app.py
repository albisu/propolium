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


st.title("Prop Firm Strategy Simulation Engine")
st.caption("Monte Carlo challenge-pass rates using automatic R-multiples and daily reset stop rules.")

st.markdown(
    """
    <style>
    :root {
        --bg-main: #0E1117;
        --bg-secondary: #1A1C24;
        --text-main: #E6E6E6;
        --accent: #00FF41;
        --danger: #FF5C5C;
    }
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-main);
    }
    /* Remove default white wrappers from streamlit blocks */
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="stExpander"],
    div[data-testid="stMetric"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stMarkdown, .stText, .stCaption, label, p, span {
        color: var(--text-main) !important;
    }
    .big-card {
        border: 1px solid #2B2F3A;
        border-radius: 16px;
        padding: 14px;
        background: var(--bg-secondary);
        box-shadow: 0 8px 30px rgba(0,0,0,0.30);
    }
    .sub-card {
        border: 1px solid #313645;
        border-radius: 12px;
        padding: 12px;
        background: #161923;
        height: 100%;
    }
    .card-title {
        font-weight: 700;
        letter-spacing: 0.4px;
        margin-bottom: 10px;
        color: #FFFFFF;
    }
    .subcard-heading {
        margin: 0 0 8px 0;
        font-weight: 600;
        color: #D9E0EE;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #2E3443 !important;
        border-radius: 12px;
        background: #141824 !important;
        padding: 10px 12px;
    }
    div[data-testid="stMetricValue"] {
        color: var(--accent) !important;
    }
    div.stButton > button {
        width: 100%;
        font-size: 20px;
        padding: 14px 0;
        border-radius: 12px;
        background: var(--accent) !important;
        color: #06120A !important;
        border: none !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_df(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


col_inputs, col_outputs = st.columns([1, 1.5], gap="large")

with col_inputs:
    st.markdown('<div class="big-card">', unsafe_allow_html=True)
    # Sub-cards arranged vertically
    st.markdown('<div class="sub-card">', unsafe_allow_html=True)
    st.markdown('<p class="subcard-heading">Data Upload</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload trade history CSV", type=["csv"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-card">', unsafe_allow_html=True)
    head_l, head_r = st.columns([2, 1])
    with head_l:
        st.markdown('<p class="subcard-heading">Challenge Parameters</p>', unsafe_allow_html=True)
    with head_r:
        value_mode_label = st.radio("Input Mode", ["Percentage (%)", "Dollar ($)"], horizontal=True, label_visibility="collapsed")
    value_mode = "percent" if value_mode_label.startswith("Percentage") else "dollar"

    g1, g2 = st.columns(2)
    with g1:
        account_size = st.number_input("Starting Equity ($)", value=100_000.0, min_value=1.0, step=1000.0)
        risk_mode_label = st.radio("Risk Mode", ["Risk (%)", "Risk ($)"], horizontal=True)
        risk_mode = "percent" if risk_mode_label.startswith("Risk (%)") else "dollar"
        risk_label = "Risk per Trade (%)" if risk_mode == "percent" else "Risk per Trade ($)"
        risk_default = 0.5 if risk_mode == "percent" else 500.0
        risk_step = 0.05 if risk_mode == "percent" else 10.0
        if risk_mode == "percent":
            risk_value = st.number_input(risk_label, value=risk_default, min_value=0.0, step=risk_step, format="%.4f")
        else:
            risk_value = st.number_input(risk_label, value=risk_default, min_value=0.0, step=risk_step)
        drawdown_type = st.selectbox("Drawdown Type", options=["Static", "Trailing"], index=0)

    with g2:
        target_label = "Target (%)" if value_mode == "percent" else "Target ($)"
        max_loss_label = "Max Loss (%)" if value_mode == "percent" else "Max Loss ($)"
        daily_loss_label = "Daily Loss (%)" if value_mode == "percent" else "Daily Loss ($)"
        target_value = st.number_input(target_label, value=8.0 if value_mode == "percent" else 8_000.0, min_value=0.0, step=0.25 if value_mode == "percent" else 100.0)
        max_loss_value = st.number_input(max_loss_label, value=10.0 if value_mode == "percent" else 10_000.0, min_value=0.0, step=0.25 if value_mode == "percent" else 100.0)
        daily_loss_value = st.number_input(daily_loss_label, value=5.0 if value_mode == "percent" else 5_000.0, min_value=0.0, step=0.25 if value_mode == "percent" else 100.0)
        challenge_fee = st.number_input("Challenge Fee ($)", value=500.0, min_value=0.0, step=50.0)
        consistency_rule_pct = st.number_input("Consistency Rule (%)", value=40.0, min_value=0.0, step=1.0)
        payout_split_pct = st.number_input("Payout Split (%)", value=80.0, min_value=0.0, max_value=100.0, step=1.0)
        seed = st.number_input("Random Seed", value=12345, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    run = st.button("Run Simulation", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with col_outputs:
    st.markdown('<div class="big-card">', unsafe_allow_html=True)

    if not run:
        st.info("Configure inputs and click Run Simulation.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    if uploaded_file is None:
        st.error("Please upload a CSV with 'rPnL' and 'dateStart'.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    try:
        df = _load_df(uploaded_file.getvalue())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    with st.spinner("Running 250,000 Monte Carlo simulations..."):
        start = time.time()
        result = run_prop_strategy_monte_carlo(
            df=df,
            starting_balance=float(account_size),
            risk_value=float(risk_value),
            risk_mode=risk_mode,
            target_value=float(target_value),
            max_loss_value=float(max_loss_value),
            daily_loss_value=float(daily_loss_value),
            value_mode=value_mode,
            drawdown_type=drawdown_type,
            challenge_fee=float(challenge_fee),
            payout_split_pct=float(payout_split_pct),
            consistency_rule_pct=float(consistency_rule_pct),
            seed=int(seed),
            equity_paths=50,
        )
        elapsed = time.time() - start

    st.caption(f"Simulation complete in {elapsed:.2f}s.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("PASS RATE", _fmt_percent(float(result["pass_rate"])))
    metric_cols[1].metric("FAIL RATE", _fmt_percent(float(result["ruin_rate"])))
    metric_cols[2].metric("NET EV ($)", _fmt_dollars(float(result["net_ev"])))
    metric_cols[3].metric("COST TO FUND ($)", _fmt_dollars(float(result["efficiency_total_cost_to_fund"])) if np.isfinite(result["efficiency_total_cost_to_fund"]) else "inf")

    target_balance = float(result["target_balance"])
    max_drawdown_balance = float(result["max_drawdown_balance"])

    fig = go.Figure()
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

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E6E6"),
        height=640,
        xaxis_title="Step # (simulated trades)",
        yaxis_title="Equity / Balance",
        xaxis_range=[0, x_line_end],
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

