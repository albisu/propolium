# Prop Firm Monte Carlo Strategy Modeler (Streamlit)

This app runs a Monte Carlo simulation over your trade history CSV to estimate:
- Pass rate (hit the profit target before breaching loss limits)
- Ruin rate (hit the daily loss or max total loss stop)
- Expected value (EV) per challenge purchase, net of the challenge fee
- Average trades required to pass
- "Total cost to fund" (fee divided by pass rate)

## CSV format
Required columns:
- `rPnL`: realized profit/loss for the trade in your backtest currency
- `dateStart`: timestamp for the trade (used to group trades by calendar day)

Optional:
- `initialBalance`: used only to auto-fill the default "Starting Balance" in the UI.

R-multiple modeling:
- User inputs the backtest "$1R" (standard risk amount).
- For each trade: `R_multiple = rPnL / $1R`.
- In simulation, each trade's $ PnL is computed as: `PnL = R_multiple * $Risk`, where `$Risk` depends on the user's risk model:
  - Static Risk: `%Risk` of the initial balance
  - Compounding Risk: `%Risk` of the current balance at that moment

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints in your terminal.

## Stop rules & daily reset logic
- Simulations shuffle calendar-day groups (so daily resets use real calendar-day boundaries).
- The simulator iterates day-by-day; within each day, the trade order from the CSV is preserved.
- Stop conditions:
  - SUCCESS: equity reaches `Starting Balance * (1 + ProfitTarget%)`
  - FAILURE (Ruin): cumulative day PnL breaches `Daily Loss Limit% * (starting equity of that day)`
  - FAILURE (Ruin): equity breaches `Starting Balance - (MaxTotalLoss% * Starting Balance)`

## Expected value
- `pass_rate = passed_sims / n_sims`
- `ruin_rate = ruin_sims / n_sims`
- `EV = (pass_rate * avg_payout_on_pass) - ((1 - pass_rate) * challenge_fee)`
- `total_cost_to_fund = challenge_fee / pass_rate` (inf if pass_rate is 0)
