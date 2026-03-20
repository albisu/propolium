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


@st.cache_data(show_spinner=False)
def _load_df(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


uploaded_file = st.file_uploader("Upload backtest trade CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()
try:
    df = _load_df(file_bytes)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.divider()

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    account_size = st.number_input("Account Size ($)", value=100_000.0, min_value=1.0, step=1000.0)
    challenge_fee = st.number_input("Challenge Fee ($)", value=500.0, min_value=0.0, step=50.0)
    payout_split_pct = st.number_input("Payout Split %", value=80.0, min_value=0.0, max_value=100.0, step=1.0)

with c2:
    risk_pct_per_trade = st.number_input("Risk % per Trade", value=0.5, min_value=0.0, step=0.05, format="%.4f")

with c3:
    profit_target_pct = st.number_input("Profit Target %", value=8.0, min_value=0.0, step=0.25)
    daily_loss_limit_pct = st.number_input("Daily Loss Limit %", value=5.0, min_value=0.0, step=0.25)
    max_total_loss_pct = st.number_input("Max Total Loss % (Static)", value=10.0, min_value=0.0, step=0.25)

st.markdown(
    """
    <style>
    div.stButton > button {
        width: 100%;
        font-size: 18px;
        padding: 14px 0;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

run = st.button("Run Simulation", type="primary", use_container_width=True)
if not run:
    st.stop()

with st.spinner("Running 10,000 Monte Carlo simulations..."):
    start = time.time()
    try:
        # equity_paths fixed to 50 by spec, but we keep it configurable internally.
        result = run_prop_strategy_monte_carlo(
            df=df,
            starting_balance=float(account_size),
            risk_pct_per_trade=float(risk_pct_per_trade),
            profit_target_pct=float(profit_target_pct),
            daily_loss_limit_pct=float(daily_loss_limit_pct),
            max_total_loss_pct=float(max_total_loss_pct),
            challenge_fee=float(challenge_fee),
            payout_split_pct=float(payout_split_pct),
            seed=12345,
            equity_paths=50,
        )
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.stop()
    elapsed = time.time() - start

st.success(f"Simulation complete in {elapsed:.2f}s.")

metric_cols = st.columns(5)
metric_cols[0].metric("PASS RATE", _fmt_percent(float(result["pass_rate"])))
metric_cols[1].metric("RUIN RATE", _fmt_percent(float(result["ruin_rate"])))
metric_cols[2].metric("EXPECTED PAYOUT ($)", _fmt_dollars(float(result["avg_payout"])))
metric_cols[3].metric("NET STRATEGY EV ($)", _fmt_dollars(float(result["net_ev"])))

eff = result["efficiency_total_cost_to_fund"]
metric_cols[4].metric("EFFICIENCY (Fee/Pass Rate)", _fmt_dollars(float(eff)) if np.isfinite(eff) else "inf")

target_balance = float(result["target_balance"])
max_drawdown_balance = float(result["max_drawdown_balance"])

st.subheader("Simulation Equity Curves (50 random paths)")

fig = go.Figure()
plot_paths = result["plot_paths_equity"]
plot_passed = result["plot_paths_passed"]
for i, equity_path in enumerate(plot_paths):
    passed = bool(plot_passed[i]) if i < len(plot_passed) else False
    color = "green" if passed else "red"
    x = np.arange(len(equity_path), dtype=int)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=equity_path,
            mode="lines",
            line=dict(color=color, width=2),
            opacity=0.5,
            showlegend=False,
        )
    )

# Horizontal reference lines
fig.add_trace(go.Scatter(x=[0, result["max_steps"]], y=[target_balance, target_balance], mode="lines", line=dict(color="blue", width=2), name="Target"))
fig.add_trace(go.Scatter(x=[0, result["max_steps"]], y=[max_drawdown_balance, max_drawdown_balance], mode="lines", line=dict(color="orange", width=2), name="Max Drawdown"))

fig.update_layout(
    template="plotly_white",
    height=600,
    xaxis_title="Step # (simulated trades)",
    yaxis_title="Equity / Balance",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True)

