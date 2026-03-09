# rlop_1min_kde

This folder runs the time-varying KDE + rolling KL experiment on the
1-minute RLOP mean series from `causal_zovko/data/rlop_{side}_1min.csv`.

## Structure

- `scripts/time_varying_kde_rolling_kl.py`: dynamic KDE, parameter selection, rolling KL, PDF snapshots, optional video
- `results/`: generated plots, CSVs, and optional MP4 outputs

## Input

By default the script reads:

- `../causal_zovko/data/rlop_bid_1min.csv`
- `../causal_zovko/data/rlop_ask_1min.csv`

The expected columns are:

- `time_paris`
- `delta_mean`
- `count`

## Defaults adapted from log_volume

- Reference window: first day `09:00-10:00`
- Dynamic recursion starts at `10:00`
- `nu=10` by default because the 1-minute reference window only has 60 samples
- `kl-step-points=1` so the KL curve is sampled every minute

## Example runs

```powershell
python rlop_1min_kde/scripts/time_varying_kde_rolling_kl.py --side bid --skip-video
python rlop_1min_kde/scripts/time_varying_kde_rolling_kl.py --side ask --skip-video
```

## Main outputs

For each side, the script writes side-tagged files under `results/`, including:

- rolling KL plot PNG
- rolling KL values CSV
- parameter-selection summary CSV
- 10:00 / 12:00 / 14:00 / 16:00 PDF comparison PNG
- optional MP4 dynamic density video
