# Daily Extreme-Returns Hawkes

This module contains Hawkes workflows built from daily close-price returns.

## Script

- `scripts/sanofi_hawkes_yahoo_split_LRtest_PRFIB.py`:
  Builds event times from extreme daily returns, performs two-half LR stability test,
  runs PRFIB bootstrap, and generates time-rescaling diagnostics.

## Default I/O

- Input daily CSV default: `data/input/sanofi_SAN.PA_yahoo_daily.csv`.
- If local CSV is unavailable and `USE_YFINANCE_IF_NO_CSV=True`, data is fetched from Yahoo.
- Numeric outputs are saved to `results/`.
- Figures are saved to `figures/`.

