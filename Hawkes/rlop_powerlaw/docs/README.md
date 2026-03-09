# RLOP Power-Law Hawkes

This module applies a univariate Hawkes model with a power-law kernel to
thresholded `RLOP` events extracted from `causal_zovko/data/rlop_events.csv`.

## Workflow

1. Export threshold exceedance events from `rlop_events.csv`.
2. Fit a power-law Hawkes model on the exceedance times.
3. Evaluate the fit using:
   - two-half likelihood-ratio comparison
   - time-rescaling diagnostics
   - one simulated path from the fitted full-sample model

## Scripts

- `scripts/export_rlop_threshold_events.py`
  Builds thresholded event-time CSVs for one side (`bid` or `ask`).
- `scripts/sanofi_rlop_powerlaw_hawkes.py`
  Fits the power-law Hawkes model, saves diagnostics, and simulates one path.

## Default event definition

- source: `causal_zovko/data/rlop_events.csv`
- side: `bid`
- threshold mode: `quantile`
- threshold value: `0.99`

That means an event is defined as:

- keep one side of the event-level RLOP stream
- mark an event whenever `delta >= q99(side-specific delta)`

## Example

```powershell
C:\Users\Wentao\anaconda3\python.exe Hawkes/rlop_powerlaw/scripts/export_rlop_threshold_events.py --side bid --threshold_mode quantile --threshold_value 0.99
C:\Users\Wentao\anaconda3\python.exe Hawkes/rlop_powerlaw/scripts/sanofi_rlop_powerlaw_hawkes.py --events_csv Hawkes/rlop_powerlaw/data/derived/rlop_bid_events_q99.csv
```

## Main outputs

- `results/powerlaw_fit_summary_*.csv`
- `results/powerlaw_rescaling_summary_*.csv`
- `results/powerlaw_simulated_events_*.csv`
- `figures/powerlaw_cumulative_counts_*.png`
- `figures/half1_*_hist_exp.png`, `figures/half1_*_qq_exp.png`, `figures/half1_*_qq_unif.png`
- `figures/half2_*_hist_exp.png`, `figures/half2_*_qq_exp.png`, `figures/half2_*_qq_unif.png`
