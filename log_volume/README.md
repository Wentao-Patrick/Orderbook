# log_volume

This folder contains log-volume experiments and outputs organized by role.

## Structure

- `scripts/`: analysis scripts
  - `log_volume1_histgram.py`
  - `loggamma_fit.py`
  - `relative_entropy_test_1h.py`
  - `time_varying_kde_rolling_kl.py`
- `results/`: generated figures and videos
- `notebooks/`: exploratory notebooks
- `cir_process/`: CIR process experiments
  - `verify_cir_stationary_gamma.py`: validates CIR stationary Gamma law by simulation

## Quick run

```powershell
python log_volume/cir_process/verify_cir_stationary_gamma.py
```
