# Log-volume 1s stationarity report

- Input: `C:\Users\Wentao\Desktop\EA_recherche\sanofi_book_snapshots_1s.parquet`
- Figure: `C:\Users\Wentao\Desktop\EA_recherche\log_volume\results\log_volume_1s_stationarity_overview.png`
- CSV: `C:\Users\Wentao\Desktop\EA_recherche\log_volume\results\log_volume_1s_stationarity_tests.csv`
- Rolling window for visualization: `900` observations

## Sample information
- `bid`: 30582 observations from 2019-10-01 09:00:18.178966266+02:00 to 2019-10-01 17:29:59.178966266+02:00, mean=5.647926, std=1.024338.
- `ask`: 30582 observations from 2019-10-01 09:00:18.178966266+02:00 to 2019-10-01 17:29:59.178966266+02:00, mean=5.794033, std=0.973792.

## Test summary
- `bid` `level`: level-stationary view -> mixed: likely structural change or time-varying moments (ADF(c) p=0, KPSS(c) p=0.01); trend-stationary view -> mixed: likely structural change or time-varying moments (ADF(ct) p=0, KPSS(ct) p=0.01).
- `bid` `diff1`: level-stationary view -> evidence supports stationarity (ADF(c) p=0, KPSS(c) p=0.1); trend-stationary view -> evidence supports stationarity (ADF(ct) p=0, KPSS(ct) p=0.1).
- `ask` `level`: level-stationary view -> mixed: likely structural change or time-varying moments (ADF(c) p=0, KPSS(c) p=0.01); trend-stationary view -> mixed: likely structural change or time-varying moments (ADF(ct) p=0, KPSS(ct) p=0.01).
- `ask` `diff1`: level-stationary view -> evidence supports stationarity (ADF(c) p=0, KPSS(c) p=0.1); trend-stationary view -> evidence supports stationarity (ADF(ct) p=0, KPSS(ct) p=0.1).
