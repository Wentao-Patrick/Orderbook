# RLOP 1min stationarity report

- Bid input: `C:\Users\Wentao\Desktop\EA_recherche\causal_zovko\data\rlop_bid_1min.csv`
- Ask input: `C:\Users\Wentao\Desktop\EA_recherche\causal_zovko\data\rlop_ask_1min.csv`
- Figure: `C:\Users\Wentao\Desktop\EA_recherche\rlop_1min_kde\results\rlop_1min_stationarity_overview.png`
- CSV: `C:\Users\Wentao\Desktop\EA_recherche\rlop_1min_kde\results\rlop_1min_stationarity_tests.csv`
- Rolling window for visualization: `15` observations

## Sample information
- `bid`: 510 observations from 2019-10-01 09:00:00+02:00 to 2019-10-01 17:29:00+02:00, mean=0.080128, std=0.119238.
- `ask`: 510 observations from 2019-10-01 09:00:00+02:00 to 2019-10-01 17:29:00+02:00, mean=0.101306, std=0.137400.

## Test summary
- `bid` `level`: level-stationary view -> inconclusive (ADF(c) p=0.4777, KPSS(c) p=0.1); trend-stationary view -> inconclusive (ADF(ct) p=0.9264, KPSS(ct) p=0.09618).
- `bid` `diff1`: level-stationary view -> evidence supports stationarity (ADF(c) p=0, KPSS(c) p=0.1); trend-stationary view -> evidence supports stationarity (ADF(ct) p=0, KPSS(ct) p=0.08548).
- `ask` `level`: level-stationary view -> mixed: likely structural change or time-varying moments (ADF(c) p=0.002028, KPSS(c) p=0.01); trend-stationary view -> mixed: likely structural change or time-varying moments (ADF(ct) p=1.539e-09, KPSS(ct) p=0.01).
- `ask` `diff1`: level-stationary view -> evidence supports stationarity (ADF(c) p=4.962e-13, KPSS(c) p=0.1); trend-stationary view -> evidence supports stationarity (ADF(ct) p=1.023e-11, KPSS(ct) p=0.1).
