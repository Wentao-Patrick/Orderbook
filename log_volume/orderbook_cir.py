"""
CIR Forecast Script (Based on 1-minute top-K depth D_m)
Usage: python orderbook_cir.py --demo

Pipeline:
- Generate or read minute-level depth `D_m`
- NO log transformation (predict raw volume D, not log(D))
- Split into estimation(T0) and test samples
- Fit OLS on estimation: D_{t+1} = a + b D_t
- Convert to CIR parameters: kappa=-ln(b), theta=a/(1-b)
- Estimate sigma via standardized residuals
- Rolling one-step forecast, compute MSE (level) and hit ratio
- Plot and save to cir_outputs/
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

OUT_DIR = Path(__file__).resolve().parent / 'cir_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_minute_depth(days=3, minutes_per_day=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp('2020-01-02 09:30')
    for d in range(days):
        for m in range(minutes_per_day):
            t = base + pd.Timedelta(days=d, minutes=m)
            # mean depth pattern: U-shape simplified
            pattern = 40 + 20 * np.cos(2 * np.pi * m / minutes_per_day)
            # generate positive depth with some persistence
            val = max(0.0, rng.normal(loc=pattern, scale=8.0))
            rows.append((t, val))
    df = pd.DataFrame(rows, columns=['timestamp', 'D'])
    df = df.set_index('timestamp')
    # add some AR(1) persistence noise to mimic realistic changes
    D = df['D'].values
    for i in range(1, len(D)):
        D[i] = max(0.0, 0.8 * D[i-1] + 0.2 * D[i] + rng.normal(0, 3.0))
    df['D'] = D
    return df


def load_real_snapshot_data(parquet_file, top_k=5):
    """
    Load real 1-second snapshots and compute top-K depth aggregated to 1-minute level.
    
    Args:
        parquet_file: Path to sanofi_book_snapshots_1s.parquet
        top_k: Number of levels to include (default 5)
    
    Returns:
        df: DataFrame with 1-minute index containing columns:
            'bid_depth','ask_depth','D' (average of two sides)
    """
    df = pd.read_parquet(parquet_file)
    # df has timestamp index, columns: bid1..ask5, bidvolume1..askvolume5, etc.
    
    # Compute top-K depth as sum of bid + ask volumes
    bid_cols = [f'bidvolume{i}' for i in range(1, top_k+1)]
    ask_cols = [f'askvolume{i}' for i in range(1, top_k+1)]
    
    # Sum across all levels
    df['bid_depth'] = df[bid_cols].sum(axis=1)
    df['ask_depth'] = df[ask_cols].sum(axis=1)
    
    # Total depth (average of bid and ask side)
    df['D'] = (df['bid_depth'] + df['ask_depth']) / 2.0
    
    # Aggregate to 1-minute level (use end-of-minute aggregation)
    # keep bid and ask separate so we can fit both sides later
    df_1min = df[['bid_depth', 'ask_depth', 'D']].resample('1T').last()
    df_1min = df_1min.dropna()
    
    # rename columns to ensure consistency
    df_1min = df_1min.rename(columns={'bid_depth': 'bid_depth', 'ask_depth': 'ask_depth'})

    return df_1min


def load_quoted_depth(parquet_file):
    """
    Load snapshots and compute quoted depth = best bid volume + best ask volume,
    aggregated to 1-minute periods.

    Returns a DataFrame with a single column 'D' (quoted depth).
    """
    df = pd.read_parquet(parquet_file)
    # assume columns 'bidvolume1' and 'askvolume1' exist
    df['quoted'] = df['bidvolume1'] + df['askvolume1']
    df_1min = df['quoted'].resample('1T').last().dropna()
    return pd.DataFrame({'D': df_1min})


def estimate_cir_params(D_series, epsilon=1e-6):
    # D_series: pandas Series indexed by time
    D = D_series.values
    T = len(D)
    # build regression D_{t+1} = a + b D_t
    X = D[:-1]
    Y = D[1:]
    Xmat = sm.add_constant(X)
    model = OLS(Y, Xmat).fit()
    a_hat = model.params[0]
    b_hat = model.params[1]

    # convert
    # guard b_hat positive; if b_hat<=0, kappa negative -> mark
    if b_hat <= 0:
        kappa = -np.log(max(b_hat, 1e-8))
    else:
        kappa = -np.log(b_hat)
    theta = a_hat / (1 - b_hat) if (1 - b_hat) != 0 else np.nan

    # residuals
    eps_hat = Y - model.predict(Xmat)
    # compute z_t = eps / sqrt(max(D_t, epsilon))
    denom = np.sqrt(np.maximum(X, epsilon))
    z = eps_hat / denom
    sigma = np.sqrt(np.var(z, ddof=1))

    # Feller condition
    feller = 2 * kappa * theta >= sigma ** 2

    res = {
        'a_hat': float(a_hat),
        'b_hat': float(b_hat),
        'kappa': float(kappa),
        'theta': float(theta),
        'sigma': float(sigma),
        'feller': bool(feller),
        'ols_summary': model.summary().as_text(),
    }
    return res


def rolling_one_step_predict(D_series, kappa, theta):
    # for t in test, predict D_{t+1} = D_t + kappa*(theta - D_t)
    D = D_series.values
    preds = D[:-1] + kappa * (theta - D[:-1])
    # return Series aligned with next-time index
    idx_pred = D_series.index[1:]
    pred_series = pd.Series(preds, index=idx_pred)
    return pred_series


def compute_metrics(D_test, D_pred):
    # D_test and D_pred are pandas Series aligned on same index
    # MSE at level (raw volume)
    mse_D = np.mean((D_test - D_pred) ** 2)
    # hit ratio on sign of change
    D_t = D_test.shift(1)  # D_t corresponds to previous time for D_{t+1}
    # align
    common_idx = D_pred.index.intersection(D_test.index)
    D_t = D_test.shift(1).loc[common_idx]
    D_tp1 = D_test.loc[common_idx]
    D_pred_loc = D_pred.loc[common_idx]
    delta_real = D_tp1 - D_t
    delta_pred = D_pred_loc - D_t
    # sign compare (treat zeros as zero)
    sign_real = np.sign(delta_real)
    sign_pred = np.sign(delta_pred)
    hits = (sign_real == sign_pred).astype(int)
    hr = np.mean(hits)
    return {
        'mse_D': float(mse_D),
        'hit_ratio': float(hr),
        'n': int(len(common_idx)),
    }


def plot_results(D_test_full, D_pred_full, outdir=OUT_DIR):
    idx = D_pred_full.index
    # 1 True vs Predicted path
    plt.figure(figsize=(10,4))
    plt.plot(D_test_full.loc[idx], label='D_true')
    plt.plot(D_pred_full, label='D_pred')
    plt.legend()
    plt.title('True vs Predicted (test period)')
    plt.tight_layout()
    p1 = outdir / 'true_vs_pred.png'
    plt.savefig(p1)
    plt.close()

    # 2 Error time series
    e = D_test_full.loc[idx] - D_pred_full
    plt.figure(figsize=(10,3))
    plt.plot(e)
    plt.title('Forecast error e_t')
    plt.tight_layout()
    p2 = outdir / 'error_ts.png'
    plt.savefig(p2)
    plt.close()

    # 3 Predicted vs Real scatter
    plt.figure(figsize=(5,5))
    plt.scatter(D_pred_full, D_test_full.loc[idx], s=8)
    mx = max(D_pred_full.max(), D_test_full.loc[idx].max())
    plt.plot([0,mx],[0,mx],'r--')
    plt.xlabel('D_pred')
    plt.ylabel('D_true')
    plt.title('Predicted vs Real')
    plt.tight_layout()
    p3 = outdir / 'pred_vs_real.png'
    plt.savefig(p3)
    plt.close()

    # 4 Delta scatter
    D_t = D_test_full.shift(1).loc[idx]
    delta_real = D_test_full.loc[idx] - D_t
    delta_pred = D_pred_full - D_t
    plt.figure(figsize=(5,5))
    plt.scatter(delta_pred, delta_real, s=8)
    mx = max(np.nanmax(delta_pred), np.nanmax(delta_real))
    mn = min(np.nanmin(delta_pred), np.nanmin(delta_real))
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.xlabel('Delta pred')
    plt.ylabel('Delta real')
    plt.title('Delta predicted vs Delta real')
    plt.tight_layout()
    p4 = outdir / 'delta_scatter.png'
    plt.savefig(p4)
    plt.close()

    # 5 Error ACF
    plt.figure(figsize=(8,3))
    plot_acf(e.dropna(), lags=40)
    plt.title('ACF of forecast error')
    plt.tight_layout()
    p5 = outdir / 'error_acf.png'
    plt.savefig(p5)
    plt.close()

    return [p1, p2, p3, p4, p5]


def pipeline_demo():
    df = generate_synthetic_minute_depth(days=6, minutes_per_day=60, seed=42)
    D = df['D']
    T = len(D)
    T0 = int(np.floor(0.8 * T))
    D_est = D.iloc[:T0]
    D_test_all = D.iloc[T0:]

    # estimate on estimation sample (use D_est)
    params = estimate_cir_params(D_est)

    # rolling one-step predictions over test starting at T0
    # Generate predictions using CIR formula for each t in [T0, T-1]: predict D_{t+1}
    pred_idx = []
    pred_vals = []
    for i in range(T0, T):
        if i >= T:
            break
        D_t = D.iloc[i]
        Dhat = D_t + params['kappa'] * (params['theta'] - D_t)
        # predicted corresponds to time index i+1 if exists
        if i + 1 < T:
            pred_idx.append(D.index[i+1])
            pred_vals.append(Dhat)
    D_pred = pd.Series(pred_vals, index=pred_idx)
    D_test = D.loc[D_pred.index]

    # compute metrics
    metrics = compute_metrics(D_test, D_pred)

    # plots
    plot_paths = plot_results(D, D_pred)

    # save summary
    summary = {
        'params': params,
        'metrics': metrics,
        'plot_paths': [str(p) for p in plot_paths],
    }
    # write json
    import json
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # print concise results
    print('CIR Parameter Estimation:')
    print(f"kappa={params['kappa']:.6f}, theta={params['theta']:.6f}, sigma={params['sigma']:.6f}, Feller={params['feller']}")
    print('\nForecast Metrics (Predicting Raw Volume D):')
    print(f"MSE_D={metrics['mse_D']:.6f}, HitRatio={metrics['hit_ratio']:.4f}, N={metrics['n']}")
    print('\nPlots saved:')
    for p in plot_paths:
        print(p)


def pipeline_real_data():
    """Run CIR pipeline on real snapshot data"""
    parquet_file = Path(__file__).resolve().parent / 'sanofi_book_snapshots_1s.parquet'
    if not parquet_file.exists():
        print(f"Error: {parquet_file} not found")
        return
    
    print(f"Loading real data from {parquet_file}...")
    df = load_real_snapshot_data(parquet_file, top_k=5)
    D = df['D']
    T = len(D)
    print(f"Loaded {T} 1-minute depth observations")
    
    T0 = int(np.floor(0.8 * T))
    D_est = D.iloc[:T0]
    D_test_all = D.iloc[T0:]
    
    # estimate on estimation sample
    params = estimate_cir_params(D_est)
    
    # rolling one-step predictions over test period
    pred_idx = []
    pred_vals = []
    for i in range(T0, T):
        if i >= T:
            break
        D_t = D.iloc[i]
        Dhat = D_t + params['kappa'] * (params['theta'] - D_t)
        if i + 1 < T:
            pred_idx.append(D.index[i+1])
            pred_vals.append(Dhat)
    D_pred = pd.Series(pred_vals, index=pred_idx)
    D_test = D.loc[D_pred.index]
    
    # compute metrics
    metrics = compute_metrics(D_test, D_pred)
    
    # plots
    plot_paths = plot_results(D, D_pred)
    
    # save summary
    summary = {
        'params': params,
        'metrics': metrics,
        'data_source': 'real_snapshots',
        'plot_paths': [str(p) for p in plot_paths],
    }
    import json
    with open(OUT_DIR / 'summary_real.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # print concise results
    print('=' * 70)
    print('CIR FORECAST ON REAL SNAPSHOT DATA (1-second aggregated to 1-minute)')
    print('=' * 70)
    print('CIR Parameter Estimation:')
    print(f"kappa={params['kappa']:.6f}, theta={params['theta']:.6f}, sigma={params['sigma']:.6f}, Feller={params['feller']}")
    print('\nForecast Metrics (Predicting Raw Volume D):')
    print(f"MSE_D={metrics['mse_D']:.6f}, HitRatio={metrics['hit_ratio']:.4f}, N={metrics['n']}")
    print('\nDepth Statistics (full period):')
    print(f"Mean depth: {D.mean():.2f}, Std: {D.std():.2f}")
    print(f"Min: {D.min():.2f}, Max: {D.max():.2f}")
    print('\nPlots saved to cir_outputs/:')
    for p in plot_paths:
        print(f"  {p.name}")
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo on synthetic data')
    parser.add_argument('--real', action='store_true', help='Run on real snapshot data')
    args = parser.parse_args()
    if args.demo:
        pipeline_demo()
    elif args.real:
        pipeline_real_data()
    else:
        print('Usage:')
        print('  --demo: Run demo on synthetic data')
        print('  --real: Run on real snapshot data')
