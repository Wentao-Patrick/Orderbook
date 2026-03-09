import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loggamma_fit import fit_loggamma_positive, loggamma_pdf


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / 'sanofi_book_snapshots_1s.parquet'
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / 'results'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether log(sum of top-k bid/ask volumes) follows log-gamma."
        )
    )
    parser.add_argument(
        '--in_parquet',
        default=str(DEFAULT_INPUT_PATH),
        help='Input parquet path.',
    )
    parser.add_argument(
        '--out_dir',
        default=str(DEFAULT_OUT_DIR),
        help='Output directory for figures and summary CSV.',
    )
    parser.add_argument(
        '--side',
        choices=['bid', 'ask', 'both'],
        default='both',
        help='Which side to test.',
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Top-k levels to sum, e.g. 5 means volume1..volume5.',
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=100,
        help='Histogram bins.',
    )
    return parser.parse_args()


def get_timestamp(df: pd.DataFrame) -> pd.Series:
    if 'timestamp' in df.columns:
        return pd.to_datetime(df['timestamp'])
    return pd.to_datetime(df.index)


def selected_sides(side_arg: str):
    if side_arg == 'both':
        return ['bid', 'ask']
    return [side_arg]


def topk_volume_columns(side: str, k: int):
    if k < 1:
        raise ValueError('--k must be >= 1')
    return [f'{side}volume{i}' for i in range(1, k + 1)]


def build_log_sum_series(df: pd.DataFrame, side: str, k: int):
    cols = topk_volume_columns(side, k)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns for {side} top-{k}: {missing}')

    sum_name = f'{side}volume1_to_{k}_sum'
    log_name = f'log_{sum_name}'

    sum_vol = df[cols].sum(axis=1, min_count=1)
    sum_vol = sum_vol[sum_vol.notna() & (sum_vol > 0)]
    if sum_vol.empty:
        raise ValueError(f'No positive samples after summing {side} top-{k} volumes.')

    # log-gamma fit in this project uses Gamma on positive log-values only.
    log_series = np.log(sum_vol)
    log_series = log_series[log_series > 0]
    log_series = log_series[np.isfinite(log_series)]
    if log_series.empty:
        raise ValueError(
            f'No positive log samples for {side} top-{k}. '
            'This means summed volume is mostly <= 1.'
        )

    return sum_name, log_name, log_series


def make_hourly_plot(df_work: pd.DataFrame, log_name: str, bins, out_path: Path):
    hours = sorted(df_work['hour'].unique())
    if not hours:
        raise ValueError('No hourly data found.')

    n = len(hours)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for ax, hour in zip(axes, hours):
        data = df_work.loc[df_work['hour'] == hour, log_name]
        ax.hist(
            data,
            bins=bins,
            color='#2C7FB8',
            alpha=0.85,
            edgecolor='white',
            linewidth=0.3,
        )
        ax.set_title(f'{hour:02d}:00')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    for ax in axes[len(hours):]:
        ax.set_visible(False)

    fig.suptitle(f'Distribution of {log_name} by hour', y=1.02)
    fig.supxlabel(log_name)
    fig.supylabel('Count')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_full_day_plot(df_work: pd.DataFrame, log_name: str, bins, fit_stats: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    _, edges, _ = ax.hist(
        df_work[log_name],
        bins=bins,
        density=True,
        color='#41AB5D',
        alpha=0.6,
        edgecolor='white',
        linewidth=0.3,
        label='Empirical density',
    )

    x_plot = np.linspace(edges[0], edges[-1], 500)
    y_plot = loggamma_pdf(x_plot, fit_stats['k'], fit_stats['theta'])
    ax.plot(x_plot, y_plot, color='#D7301F', linewidth=2.0, label='Log-Gamma fit')

    ax.set_title(f'Distribution of {log_name} - Full Day')
    ax.set_xlabel(log_name)
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.legend(loc='best')

    fit_text = (
        f"k={fit_stats['k']:.4f}\n"
        f"theta={fit_stats['theta']:.4f}\n"
        f"KS p={fit_stats['ks_pvalue']:.4g}\n"
        f"n_fit={fit_stats['n_fit']}"
    )
    ax.text(
        0.98,
        0.98,
        fit_text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_single_side(df: pd.DataFrame, side: str, k: int, bins_count: int, out_dir: Path):
    sum_name, log_name, log_series = build_log_sum_series(df=df, side=side, k=k)

    df_work = pd.DataFrame(
        {
            'timestamp': df['timestamp'],
            log_name: np.nan,
        },
        index=df.index,
    )
    df_work.loc[log_series.index, log_name] = log_series
    df_work = df_work.dropna(subset=[log_name]).copy()
    df_work['hour'] = df_work['timestamp'].dt.hour

    minv = df_work[log_name].min()
    maxv = df_work[log_name].max()
    bins = np.linspace(minv, maxv, bins_count) if minv < maxv else bins_count

    fit_stats = fit_loggamma_positive(df_work[log_name].to_numpy())

    hourly_path = out_dir / f'{log_name}_hist_by_hour.png'
    full_day_path = out_dir / f'{log_name}_hist_full_day.png'

    make_hourly_plot(df_work=df_work, log_name=log_name, bins=bins, out_path=hourly_path)
    make_full_day_plot(
        df_work=df_work,
        log_name=log_name,
        bins=bins,
        fit_stats=fit_stats,
        out_path=full_day_path,
    )

    return {
        'side': side,
        'k': k,
        'sum_name': sum_name,
        'log_name': log_name,
        'k_shape': fit_stats['k'],
        'theta_scale': fit_stats['theta'],
        'ks_stat': fit_stats['ks_stat'],
        'ks_pvalue': fit_stats['ks_pvalue'],
        'n_fit': fit_stats['n_fit'],
        'hourly_plot': str(hourly_path),
        'full_day_plot': str(full_day_path),
    }


def main():
    args = parse_args()
    in_path = Path(args.in_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise SystemExit(f'File not found: {in_path}')

    df = pd.read_parquet(str(in_path))
    df = df.copy()
    df['timestamp'] = get_timestamp(df)

    summaries = []
    for side in selected_sides(args.side):
        summary = run_single_side(
            df=df,
            side=side,
            k=args.k,
            bins_count=args.bins,
            out_dir=out_dir,
        )
        summaries.append(summary)
        print(
            f"[log_gamma] {summary['log_name']}: "
            f"k={summary['k_shape']:.6f}, theta={summary['theta_scale']:.6f}, "
            f"KS={summary['ks_stat']:.6f}, p={summary['ks_pvalue']:.6g}, n_fit={summary['n_fit']}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / f'log_gamma_{args.side}_top{args.k}_sum_volume_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved summary: {summary_path}')


if __name__ == '__main__':
    main()
