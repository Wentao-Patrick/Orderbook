# High-Frequency Order-Flow Hawkes

This module contains Hawkes modeling pipelines on buy/sell event times.

## Scripts

- `scripts/export_hawkes_sequences.py`:
  Extracts `t_seconds_jitter` buy/sell sequences from decoded trade records.
- `scripts/sanofi_hawkes_change_test.py`:
  Two-half LR change test with fixed-beta symmetric bivariate Hawkes.
- `scripts/sanofi_hawkes_FIB_and_rescaling.py`:
  Fixed-beta symmetric bivariate Hawkes, FIB bootstrap, and time-rescaling diagnostics.
- `scripts/sanofi_hawkes_FIB_full_opt_with_beta.py`:
  Full MLE (`mu_buy`, `mu_sell`, `a_self`, `a_cross`, `beta`) with FIB bootstrap and diagnostics.
- `scripts/sanofi_hawkes_FIB_univariate_no_cross.py`:
  Univariate Hawkes version on merged buy+sell flow (no cross-mark interaction).

## Default I/O

- Inputs are read from `data/input/` by default.
- Numeric outputs are written to `results/` by default.
- Plots are written to `figures/` by default.

