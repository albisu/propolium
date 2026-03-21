import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prop_strategy_engine import run_firm_comparison_batch, run_prop_strategy_monte_carlo


st.set_page_config(page_title="Prop Firm Strategy Simulation Engine", layout="wide")

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* ─── Design tokens ───────────────────────────────────────── */
    :root {
        --pp-bg: #000000;
        --pp-surface: #0a0a0a;
        --pp-border: #1f1f1f;
        --pp-text: #E6E6E6;
        --pp-muted: #9ca3af;
        --pp-accent: #00FF41;
        --pp-radius: 12px;
        --page-pad-x: clamp(1rem, 3.5vw, 2.75rem);
        --page-pad-y-top: max(1.25rem, env(safe-area-inset-top, 0px));
        --page-pad-y-bottom: 2rem;
        --col-gap: 1.5rem;
    }
    html { scroll-behavior: smooth; }
    /* Scroll: do not trap overflow on nested flex — let the page scroll naturally */
    html, body {
        font-family: "SF UI Display", "Inter", system-ui, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
        background-color: var(--pp-bg) !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        height: auto !important;
        min-height: 100% !important;
        max-height: none !important;
    }
    .stApp {
        font-family: "SF UI Display", "Inter", system-ui, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
        background-color: var(--pp-bg) !important;
        color: var(--pp-text) !important;
        overflow: visible !important;
        min-height: 100% !important;
    }
    header[data-testid="stHeader"] {
        background-color: var(--pp-bg) !important;
        border-bottom: 1px solid #1a1a1a !important;
        position: relative !important;
        z-index: 2 !important;
    }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stAppViewContainer"] {
        background-color: var(--pp-bg) !important;
        overflow: visible !important;
        min-height: unset !important;
    }
    [data-testid="stMain"], section.main {
        background-color: var(--pp-bg) !important;
        overflow: visible !important;
        min-height: unset !important;
    }
    .block-container {
        padding: var(--page-pad-y-top) var(--page-pad-x) var(--page-pad-y-bottom) var(--page-pad-x) !important;
        max-width: min(1680px, 100%) !important;
        box-sizing: border-box !important;
        overflow: visible !important;
    }
    /* ─── Page shell: header band → divider → two columns ─────── */
    .pp-title {
        color: var(--pp-text) !important;
        font-weight: 700 !important;
        font-size: clamp(1.25rem, 2.2vw, 1.6rem) !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
        scroll-margin-top: 1rem !important;
    }
    .pp-subtitle {
        color: var(--pp-muted) !important;
        font-size: 0.82rem !important;
        margin: 0.35rem 0 0 0 !important;
        letter-spacing: 0.02em;
    }
    p.pp-column-title {
        color: var(--pp-muted) !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 0 0 0.5rem 0 !important;
    }
    /* Bordered panels (st.container(border=True)) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--pp-surface) !important;
        border: 1px solid var(--pp-border) !important;
        border-radius: var(--pp-radius) !important;
        padding: 0.85rem 1rem 1rem 1rem !important;
        overflow: visible !important;
        max-height: none !important;
    }
    [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    [data-testid="stVerticalBlock"] [data-testid="stElementContainer"] { overflow: visible !important; }
    [data-testid="stHorizontalBlock"] {
        gap: var(--col-gap) !important;
        align-items: flex-start !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--pp-text) !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
    }
    .stMarkdown, .stText, .stCaption, label, p, span { color: var(--pp-text) !important; }
    hr {
        border: none !important;
        border-top: 1px solid #2a2a2a !important;
        margin: 0.85rem 0 1.15rem 0 !important;
    }
    [data-testid="stExpander"], [data-testid="stForm"] {
        background: var(--pp-bg) !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--pp-border) !important;
        border-radius: 8px;
        background: #101010 !important;
        padding: 8px 10px;
    }
    div[data-testid="stMetricValue"] { color: var(--pp-accent) !important; }
    div.stButton > button {
        width: 56px;
        height: 40px;
        min-width: 56px;
        padding: 0 !important;
        border-radius: 8px;
        background: var(--pp-accent) !important;
        color: #000000 !important;
        border: 1px solid var(--pp-accent) !important;
        font-weight: 700;
        font-size: 18px;
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
    }
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
    input[type=number] { -moz-appearance: textfield; }
    input, textarea, select, button {
        font-family: "SF UI Display", "Inter", system-ui, ui-sans-serif, sans-serif !important;
    }
    [data-testid="stPlotlyChart"] {
        width: 100% !important;
        min-height: 260px !important;
        overflow: visible !important;
    }
    [data-testid="stPlotlyChart"] > div { overflow: visible !important; }
    [data-testid="stPlotlyChart"] iframe {
        width: 100% !important;
        min-height: 260px !important;
    }
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] [data-testid="stTable"] {
        background-color: var(--pp-surface) !important;
        border: none !important;
    }
    [data-testid="stDataFrame"] table td,
    [data-testid="stDataFrame"] table th {
        border-color: #222222 !important;
        color: #e6e6e6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header band (title + mode) ─────────────────────────────────────────────
_hdr_l, _hdr_r = st.columns([1, 0.32], gap="large")
with _hdr_l:
    st.markdown(
        """
        <div id="propolium-page-top">
            <div class="pp-title">Prop Firm Strategy Simulation Engine</div>
            <div class="pp-subtitle">Monte Carlo · prop firm rules · funding economics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with _hdr_r:
    comparison_mode = st.toggle("Comparison Mode", value=False, key="comparison_mode")

st.divider()

# ─── Main work area: Inputs | Results (same logical structure as before) ───


def _fmt_percent(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100.0:.2f}%"


def _fmt_dollars(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


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


def _comparison_firm_payloads() -> list[dict]:
    """All non-Custom templates as firm dicts for batch simulation."""
    rows: list[dict] = []
    for name, tpl in TEMPLATES.items():
        if name == "Custom" or tpl is None:
            continue
        rows.append({"name": name, **tpl})
    return rows


def _comparison_plotly_layout(fig: go.Figure, *, margin_right: float = 32) -> None:
    """Transparent Plotly theme; generous margins so outside labels are not clipped."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E6E6", family="SF UI Display, Inter, system-ui, sans-serif"),
        margin=dict(l=4, r=margin_right, t=16, b=12),
        showlegend=False,
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.2)",
        tickfont=dict(color="#E6E6E6"),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.2)",
        tickfont=dict(color="#E6E6E6"),
        automargin=True,
    )


def _enrich_comparison_df(raw: pd.DataFrame, avg_trades_per_day: float, monthly_recurring: bool) -> pd.DataFrame:
    """Add time-based fields, cost per attempt, 95% budget; sort cheapest to fund."""
    df = raw.copy()
    if not np.isfinite(avg_trades_per_day) or avg_trades_per_day <= 0:
        df["Avg Days to Pass"] = np.nan
        df["Months to Pass"] = np.nan
    else:
        df["Avg Days to Pass"] = df["Avg Trades to Pass"] / avg_trades_per_day
        df["Months to Pass"] = df["Avg Days to Pass"] / 30.0

    def _cost_attempt(row: pd.Series) -> float:
        if not monthly_recurring:
            return float(row["Challenge Fee"])
        m = row["Months to Pass"]
        if not np.isfinite(m):
            return np.nan
        return float(row["Challenge Fee"]) * max(1.0, float(m))

    df["Cost per Attempt"] = df.apply(_cost_attempt, axis=1)

    def _budget(row: pd.Series) -> float:
        n95 = row["n95"]
        ca = row["Cost per Attempt"]
        if not np.isfinite(n95) or not np.isfinite(ca):
            return np.inf
        return float(n95) * float(ca)

    df["95% Confidence Budget ($)"] = df.apply(_budget, axis=1)
    df = df.sort_values("95% Confidence Budget ($)", ascending=True).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


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
    st.session_state.setdefault("comparison_df", None)
    st.session_state.setdefault("comparison_elapsed", None)
    st.session_state.setdefault("comparison_error", None)
    st.session_state.setdefault("comparison_monthly", False)


_init_defaults()

run = False
run_compare = False

col_inputs, col_outputs = st.columns([1, 1.25], gap="large")

with col_inputs:
    st.markdown('<p class="pp-column-title">Inputs</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Data Upload**")
        uploaded_file = st.file_uploader("Upload trade history CSV", type=["csv"], label_visibility="collapsed")

        if not comparison_mode:
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

            run = st.button("▶️", type="primary", help="Run Simulation", key="run_single")
        else:
            st.markdown("**Firm Comparison**")
            st.caption("10,000 simulations per firm · vectorized batch · ranked by 95% funding budget")
            st.checkbox("Monthly Recurring", key="comparison_monthly")
            run_compare = st.button("▶️", type="primary", help="Run All-Firm Comparison", key="run_compare")

with col_outputs:
    st.markdown('<p class="pp-column-title">Results</p>', unsafe_allow_html=True)
    with st.container(border=True):
        if comparison_mode:
            if run_compare:
                if uploaded_file is None:
                    st.session_state.comparison_error = "Please upload a CSV with 'rPnL' and 'dateStart'."
                    st.session_state.comparison_df = None
                else:
                    try:
                        df = _load_df(uploaded_file.getvalue())
                        dt = pd.to_datetime(df["dateStart"], errors="coerce", utc=False)
                        if dt.isna().any():
                            raise ValueError("CSV has invalid dateStart values.")

                        trades_per_day = df.groupby(dt.dt.normalize()).size()
                        avg_trades_per_day = float(trades_per_day.mean()) if len(trades_per_day) else float("nan")

                        firms = _comparison_firm_payloads()
                        comp_monthly = bool(st.session_state.get("comparison_monthly", False))

                        with st.spinner("Running all-firm comparison (10,000 sims × firm)..."):
                            t0 = time.time()
                            raw = run_firm_comparison_batch(df=df, firms=firms, n_sims=10_000, seed=42)
                            enriched = _enrich_comparison_df(raw, avg_trades_per_day, comp_monthly)
                            st.session_state.comparison_elapsed = time.time() - t0

                        st.session_state.comparison_df = enriched
                        st.session_state.comparison_error = None
                    except Exception as e:
                        st.session_state.comparison_error = str(e)
                        st.session_state.comparison_df = None
                        st.session_state.comparison_elapsed = None

            if st.session_state.comparison_error:
                st.error(st.session_state.comparison_error)

            if st.session_state.comparison_elapsed is not None and st.session_state.comparison_df is not None:
                st.caption(f"Comparison complete in {st.session_state.comparison_elapsed:.2f}s.")
            elif st.session_state.comparison_df is None and not st.session_state.comparison_error:
                st.caption("Upload CSV and run All-Firm Comparison.")

            cdf = st.session_state.comparison_df
            if cdf is not None and len(cdf):
                st.markdown("### Leaderboard")
                st.caption("Ranked **cheapest → most expensive** by **95% Confidence Budget** (for this backtest).")
                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

                display_cols = [c for c in cdf.columns if c != "n95"]
                disp = cdf[display_cols].copy()
                styler = (
                    disp.style.format(
                        {
                            "Rank": "{:.0f}",
                            "Pass Rate": "{:.2%}",
                            "Fail Rate": "{:.2%}",
                            "Avg Trades to Pass": "{:.2f}",
                            "Challenge Fee": "${:,.0f}",
                            "Avg Days to Pass": "{:.2f}",
                            "Months to Pass": "{:.2f}",
                            "Cost per Attempt": "${:,.2f}",
                            "95% Confidence Budget ($)": "${:,.2f}",
                        },
                        na_rep="n/a",
                    )
                    .set_properties(**{"background-color": "#0a0a0a", "color": "#e6e6e6", "border-color": "#222"})
                    .hide(axis="index")
                )
                st.dataframe(styler, use_container_width=True)

                firms_list = disp["Firm"].astype(str).tolist()
                pass_rates = disp["Pass Rate"].astype(float).values
                budgets = disp["95% Confidence Budget ($)"].astype(float).values

                fig_pass = go.Figure(
                    go.Bar(
                        x=pass_rates,
                        y=firms_list,
                        orientation="h",
                        marker=dict(color="#00FF41", line=dict(width=0)),
                    )
                )
                fig_pass.update_layout(
                    xaxis_title="Pass Rate",
                    yaxis_title="",
                    height=max(300, 44 * len(firms_list)),
                )
                fig_pass.update_xaxes(tickformat=".0%")
                fig_pass.update_yaxes(autorange="reversed")
                _comparison_plotly_layout(fig_pass, margin_right=24)

                finite_mask = np.isfinite(budgets)
                best_i = int(np.argmin(np.where(finite_mask, budgets, np.inf))) if finite_mask.any() else 0
                bar_colors = ["#00FF41" if i == best_i else "#2d4a2d" for i in range(len(firms_list))]
                x_budget = np.where(finite_mask, budgets, np.nan)

                fig_budget = go.Figure(
                    go.Bar(
                        x=x_budget,
                        y=firms_list,
                        orientation="h",
                        marker=dict(color=bar_colors, line=dict(width=0)),
                        text=[
                            "Most capital-efficient" if i == best_i and finite_mask[i] else ""
                            for i in range(len(firms_list))
                        ],
                        textposition="outside",
                        textfont=dict(color="#00FF41", size=11),
                    )
                )
                fig_budget.update_layout(
                    xaxis_title="USD (95% confidence budget)",
                    yaxis_title="",
                    height=max(300, 44 * len(firms_list)),
                )
                fig_budget.update_yaxes(autorange="reversed")
                fig_budget.update_xaxes(tickprefix="$")
                # Wide right margin so "Most capital-efficient" outside labels are not clipped
                _comparison_plotly_layout(fig_budget, margin_right=160)

                st.divider()
                st.markdown("#### Pass rate comparison")
                st.plotly_chart(fig_pass, use_container_width=True)
                st.markdown("#### 95% confidence budget")
                st.caption("Lowest bar = most capital-efficient for this backtest.")
                st.plotly_chart(fig_budget, use_container_width=True)

        else:
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
                margin=dict(l=8, r=12, t=56, b=12),
                xaxis_title="Step # (simulated trades)",
                yaxis_title="Equity / Balance",
                xaxis_range=[0, x_line_end],
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False, automargin=True),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False, automargin=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

