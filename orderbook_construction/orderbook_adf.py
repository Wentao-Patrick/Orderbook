"""
Pipeline minimal pour reconstruire top-K depth, agréger en minute (EOM),
déseasonnaliser (minute-of-day), et effectuer un test ADF avec diagnostics.
Usage: python orderbook_adf.py --demo

Attendu: fichier CSV d'events/snapshots avec colonnes au minimum:
- timestamp: ISO ou convertible en datetime
- side: 'bid' ou 'ask' (optionnel si fichier séparé par side)
- p1,q1,p2,q2,...,pK,qK  (snapshot format)

Le script fournit aussi une démonstration synthétique.
"""

from datetime import timedelta
import argparse
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox


def read_snapshot_csv(path, K=None, timestamp_col='timestamp', side_col='side'):
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    # infer K from columns if not provided
    if K is None:
        q_cols = [c for c in df.columns if c.lower().startswith('q')]
        K = len(q_cols) // 1
    return df, K


def compute_topk_depth_from_snapshot_row(row, K, side_prefix='q'):
    # expects q1,q2,... numeric
    qs = []
    for i in range(1, K + 1):
        col = f"q{i}"
        if col in row:
            qs.append(float(row[col]) if pd.notnull(row[col]) else 0.0)
        else:
            qs.append(0.0)
    return sum(qs)


def compute_event_depths(df, K):
    # add column 'depth_K' computed from q1..qK
    depths = []
    for _, row in df.iterrows():
        depths.append(compute_topk_depth_from_snapshot_row(row, K))
    df = df.copy()
    df['depth_K'] = depths
    return df


def aggregate_eom_minute(df, ts_col='timestamp', depth_col='depth_K', trading_hours=None):
    # df must be sorted by timestamp
    df = df.set_index(ts_col).sort_index()
    # build minute index covering trading hours if provided else from data
    start = df.index[0].floor('T')
    end = df.index[-1].ceil('T')
    minute_index = pd.date_range(start, end, freq='T')

    # take last observation within each minute
    last_per_min = df[depth_col].groupby(pd.Grouper(freq='T')).last()
    last_per_min = last_per_min.reindex(minute_index)
    # carry-forward fill
    last_per_min = last_per_min.fillna(method='ffill')
    # drop leading NaNs (before first observation)
    last_per_min = last_per_min.dropna()
    return last_per_min


def log_transform(series):
    return np.log1p(series)


def deseasonalize_minute_of_day(series):
    # series index is datetime at 1-minute frequency
    df = series.to_frame(name='x')
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    mu = df.groupby('minute_of_day')['x'].mean()
    df['x_deseas'] = df.apply(lambda r: r['x'] - mu.loc[r['minute_of_day']], axis=1)
    return df['x_deseas'], mu


def run_adf(series, regression='c', maxlag=30, autolag='AIC'):
    # uses statsmodels adfuller
    # returns dict with key results
    res = adfuller(series.values, maxlag=maxlag, regression=regression, autolag=autolag)
    adf_stat, pvalue, usedlag, nobs, crit_vals, icbest = res[0], res[1], res[2], res[3], res[4], res[5]
    return {
        'adf_stat': adf_stat,
        'pvalue': pvalue,
        'usedlag': usedlag,
        'nobs': nobs,
        'crit_vals': crit_vals,
        'icbest': icbest,
        'regression': regression,
    }


def diagnostic_ljung_box_on_adf_residuals(series, usedlag, regression='c'):
    # Fit AutoReg with usedlag and same deterministic term to get residuals
    trend_map = {'c': 'c', 'ct': 'ct', 'nc': 'n'}
    trend = trend_map.get(regression, 'c')
    if usedlag <= 0:
        usedlag = 0
    if usedlag == 0:
        # fit constant/trend only model
        # residuals are series minus mean (or trend) — for simplicity subtract mean
        resid = series - series.mean()
    else:
        model = AutoReg(series, lags=usedlag, trend=trend, old_names=False).fit()
        resid = model.resid
    lb = acorr_ljungbox(resid, lags=[min(10, max(1, int(len(resid)/10)))], return_df=True)
    return lb


# --- Synthetic demo to validate pipeline ---

def generate_synthetic_snapshots(days=5, minutes_per_day=390, K=5, seed=0):
    # produce per-minute synthetic snapshots at random intra-minute times
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp('2020-01-02 09:30')
    for d in range(days):
        day_base = base + pd.Timedelta(days=d)
        for m in range(minutes_per_day):
            t_min = day_base + pd.Timedelta(minutes=m)
            # simulate random number of events in the minute, pick last event time slightly before next minute
            t_event = t_min + pd.Timedelta(milliseconds=int(rng.integers(0, 60000)))
            # simulate depth quantities with intraday pattern: higher near open
            pattern = 1 + 0.5 * np.cos(2 * np.pi * m / minutes_per_day)
            qs = (np.abs(rng.normal(loc=50 * pattern, scale=10, size=K))).astype(int)
            row = {'timestamp': t_event}
            for i, q in enumerate(qs, start=1):
                row[f'q{i}'] = int(q)
                row[f'p{i}'] = 100.0 + i * 0.01 + rng.normal(0, 0.001)
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def pipeline_from_snapshot_df(df_snapshot, K=5, demo=False):
    df_events = compute_event_depths(df_snapshot, K)
    minute_series = aggregate_eom_minute(df_events)
    X = log_transform(minute_series)
    X_deseas, mu = deseasonalize_minute_of_day(X)
    # ADF on deseasonalized
    adf_res = run_adf(X_deseas, regression='c', maxlag=30, autolag='AIC')
    lb = diagnostic_ljung_box_on_adf_residuals(X_deseas, adf_res['usedlag'], regression=adf_res['regression'])
    return {
        'minute_series': minute_series,
        'X': X,
        'X_deseas': X_deseas,
        'mu_minute_of_day': mu,
        'adf': adf_res,
        'ljung_box': lb,
    }


def run_demo():
    print('Génération de données synthétiques...')
    df_snap = generate_synthetic_snapshots(days=3, minutes_per_day=60, K=5)
    print('Exécute pipeline...')
    res = pipeline_from_snapshot_df(df_snap, K=5)
    print('\n--- Résultats ADF ---')
    for k, v in res['adf'].items():
        if k == 'crit_vals':
            print('crit_vals:', v)
        else:
            print(f'{k}:', v)
    print('\n--- Ljung-Box (résumé) ---')
    print(res['ljung_box'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run synthetic demo')
    args = parser.parse_args()
    if args.demo:
        run_demo()
    else:
        print('Aucun fichier fourni. Lancez avec --demo pour une démonstration.')
