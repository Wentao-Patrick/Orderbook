"""
Improved CIR Model with Moment Estimator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json


OUT_DIR = Path(__file__).resolve().parent / 'cir_improved_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_cir_params_moment(D_series, epsilon=1e-6):
    """
    Moment-based approximation to CIR parameters derived from
    ∫ X dX = κ ∫ X(θ-X) dt + σ ∫ X^{3/2} dW.

    Uses the discrete formulas described in the request:
        θ̂ = (1/m)∑X_i
        κ̂ = [ (1/(m-1))∑ X_i (X_{i+1}-X_i) ] /
              [ (1/m)∑ X_i (θ̂ - X_i) ]

    Returns a dictionary with CIR parameter estimates.
    """
    if isinstance(D_series, np.ndarray):
        D = D_series
    else:
        D = D_series.values if hasattr(D_series, 'values') else D_series
    m = len(D)
    if m < 2:
        return None
    theta_hat = np.mean(D)
    # numerator and denominator
    num = np.sum(D[:-1] * (D[1:] - D[:-1])) / (m - 1)
    den = np.sum(D * (theta_hat - D)) / m
    kappa_hat = num / den if den != 0 else np.nan
    if np.isfinite(kappa_hat):
        kappa_hat = max(kappa_hat, 0.0)
    # estimate sigma using residuals
    # ΔX_i = kappa(θ - X_i) + σ√X_i * ε
    delta = D[1:] - D[:-1]
    eps_hat = delta - kappa_hat * (theta_hat - D[:-1])
    denom = np.sqrt(np.maximum(D[:-1], epsilon))
    z = eps_hat / denom
    sigma_hat = np.sqrt(np.var(z, ddof=1))
    feller = 2 * kappa_hat * theta_hat >= sigma_hat ** 2 if np.isfinite(sigma_hat) else False

    return {
        'kappa': float(kappa_hat),
        'theta': float(theta_hat),
        'sigma': float(sigma_hat),
        'feller': bool(feller),
        'sample_size': int(m),
    }


def compute_metrics(D_test, D_pred):
    """Compute MSE and hit ratio."""
    mse_D = np.mean((D_test - D_pred) ** 2)
    D_t = D_test.shift(1)
    common_idx = D_pred.index.intersection(D_test.index)
    D_t = D_test.shift(1).loc[common_idx]
    D_tp1 = D_test.loc[common_idx]
    D_pred_loc = D_pred.loc[common_idx]
    delta_real = D_tp1 - D_t
    delta_pred = D_pred_loc - D_t
    sign_real = np.sign(delta_real)
    sign_pred = np.sign(delta_pred)
    hits = (sign_real == sign_pred).astype(int)
    hr = np.mean(hits)
    return {
        'mse_D': float(mse_D),
        'hit_ratio': float(hr),
        'n': int(len(common_idx)),
    }


def pipeline_bidask(df):
    """Estimate and predict separately on bid and ask depth series.

    Returns a dictionary keyed by side ('bid_depth','ask_depth') containing
    estimation results, forecast series and metrics.
    """
    results = {}
    for side in ['bid_depth', 'ask_depth']:
        if side not in df.columns:
            continue
        D = df[side]
        T = len(D)
        T0 = int(np.floor(0.8 * T))
        D_est = D.iloc[:T0]
        est = estimate_cir_params_moment(D_est)
        if est is None:
            results[side] = None
            continue

        # rolling one-step forecast on this side
        pred_idx = []
        pred_vals = []
        for i in range(T0, T - 1):
            D_t = D.iloc[i]
            Dhat = D_t + est['kappa'] * (est['theta'] - D_t)
            pred_idx.append(D.index[i + 1])
            pred_vals.append(Dhat)
        D_pred = pd.Series(pred_vals, index=pred_idx)
        D_test = D.loc[pred_idx]
        metrics = compute_metrics(D_test, D_pred)

        results[side] = {
            'est': est,
            'metrics': metrics,
            'pred': D_pred,
            'test': D_test,
        }

    # persist results to JSON (parameters + metrics; series cannot be serialized)
    out = {}
    for side, info in results.items():
        if info is None:
            out[side] = None
            continue
        est = info['est']
        met = info['metrics']
        out[side] = {
            'parameters': {
                'kappa': est['kappa'],
                'theta': est['theta'],
                'sigma': est['sigma'],
                'feller': est['feller'],
            },
            'metrics': met,
        }
    with open(OUT_DIR / 'bidask_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    return results


def plot_bidask_results(results, outdir=OUT_DIR):
    """Create simple diagnostic plots for each side."""
    paths = []
    for side, info in results.items():
        if info is None:
            continue
        D_test = info['test']
        D_pred = info['pred']
        plt.figure(figsize=(10,4))
        plt.plot(D_test, label=f'{side} true')
        plt.plot(D_pred, label=f'{side} pred')
        plt.title(f'{side} True vs Predicted')
        plt.legend()
        plt.tight_layout()
        p = outdir / f'{side}_true_vs_pred.png'
        plt.savefig(p)
        plt.close()
        paths.append(p)

        plt.figure(figsize=(5,5))
        plt.scatter(D_pred, D_test, s=8)
        mx = max(D_pred.max(), D_test.max())
        plt.plot([0, mx], [0, mx], 'r--')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{side} Scatter')
        plt.tight_layout()
        p2 = outdir / f'{side}_scatter.png'
        plt.savefig(p2)
        plt.close()
        paths.append(p2)
    return paths


def main():
    """Run bid/ask separate analysis on real data."""
    from orderbook_cir import load_real_snapshot_data
    
    parquet_file = Path(__file__).resolve().parent / 'sanofi_book_snapshots_1s.parquet'
    
    if not parquet_file.exists():
        print(f"Error: {parquet_file} not found")
        return
    
    print("Loading real snapshot data...")
    df = load_real_snapshot_data(parquet_file, top_k=5)
    
    print(f"Loaded {len(df)} observations")
    
    print('\nRunning bid/ask separate pipeline...')
    results = pipeline_bidask(df)
    for side, info in results.items():
        if info is None:
            print(f"{side}: estimation failed")
            continue
        est = info['est']
        met = info['metrics']
        print(f"\nSide: {side}")
        print(f"  kappa={est['kappa']:.6f}")
        print(f"  theta={est['theta']:.6f}, sigma={est['sigma']:.6f}, Feller={est['feller']}")
        print(f"  Forecast hit ratio={met['hit_ratio']:.4f}, mse={met['mse_D']:.2f}, n={met['n']}")
    plot_paths = plot_bidask_results(results)
    print('\nPlots saved:')
    for p in plot_paths:
        print(f'  {p.name}')

if __name__ == '__main__':
    main()
