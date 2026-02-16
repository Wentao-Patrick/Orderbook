import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loggamma_fit import fit_loggamma_positive, loggamma_pdf

# Paths
BASE_DIR = r'C:\Users\Wentao\Desktop\EA_recherche'
INPUT_PATH = os.path.join(BASE_DIR, 'sanofi_book_snapshots_1s.parquet')
OUT_DIR = os.path.join(BASE_DIR, 'log_volume')

# Choose side: "bid" or "ask"
VOLUME_SIDE = 'ask'

if not os.path.exists(INPUT_PATH):
    raise SystemExit(f'File not found: {INPUT_PATH}')

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
# This parquet contains continuous auction data only, per user.
df = pd.read_parquet(INPUT_PATH)

# Handle timestamp as column or index
if 'timestamp' in df.columns:
    ts = pd.to_datetime(df['timestamp'])
else:
    ts = pd.to_datetime(df.index)

df = df.copy()
df['timestamp'] = ts

volume_col = f'{VOLUME_SIDE}volume1'
if volume_col not in df.columns:
    raise SystemExit(f'Missing column: {volume_col}')

# Keep valid, positive volume for log
mask = df[volume_col].notna() & (df[volume_col] > 0)
df = df.loc[mask]

if df.empty:
    raise SystemExit(f'No data left after filtering {volume_col} > 0')

# Compute log volume
# Natural log
log_col = np.log(df[volume_col])
log_name = f'log_{volume_col}'
df[log_name] = log_col

# Hourly distributions
# Hour of day (0-23)
df['hour'] = df['timestamp'].dt.hour
hours = sorted(df['hour'].unique())

if not hours:
    raise SystemExit('No hourly data found.')

minv = df[log_name].min()
maxv = df[log_name].max()
if minv < maxv:
    bins = np.linspace(minv, maxv, 100)
else:
    bins = 100

n = len(hours)
ncols = 4
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
axes = np.array(axes).reshape(-1)

for ax, hour in zip(axes, hours):
    data = df.loc[df['hour'] == hour, log_name]
    ax.hist(data, bins=bins, color='#2C7FB8', alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.set_title(f'{hour:02d}:00')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

for ax in axes[len(hours):]:
    ax.set_visible(False)

fig.suptitle(f'Distribution of log({volume_col}) by hour', y=1.02)
fig.supxlabel(f'log({volume_col})')
fig.supylabel('Count')
fig.tight_layout()

hourly_path = os.path.join(OUT_DIR, f'log_{volume_col}_hist_by_hour.png')
fig.savefig(hourly_path, dpi=200, bbox_inches='tight')
plt.close(fig)

# Full-day distribution
fig_all, ax_all = plt.subplots(figsize=(6, 4))
hist_density, edges, _ = ax_all.hist(
    df[log_name],
    bins=bins,
    density=True,
    color='#41AB5D',
    alpha=0.6,
    edgecolor='white',
    linewidth=0.3,
    label='Empirical density'
)

fit_stats = fit_loggamma_positive(df[log_name].to_numpy())
x_plot = np.linspace(edges[0], edges[-1], 500)
y_plot = loggamma_pdf(x_plot, fit_stats['k'], fit_stats['theta'])
ax_all.plot(x_plot, y_plot, color='#D7301F', linewidth=2.0, label='Log-Gamma fit')

ax_all.set_title(f'Distribution of log({volume_col}) - Full Day')
ax_all.set_xlabel(f'log({volume_col})')
ax_all.set_ylabel('Density')
ax_all.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax_all.legend(loc='best')

fit_text = (
    f"k={fit_stats['k']:.4f}\n"
    f"theta={fit_stats['theta']:.4f}\n"
    f"KS p={fit_stats['ks_pvalue']:.4g}\n"
    f"n_fit={fit_stats['n_fit']}"
)
ax_all.text(
    0.98, 0.98, fit_text,
    transform=ax_all.transAxes,
    ha='right', va='top',
    fontsize=8,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
)
fig_all.tight_layout()

all_path = os.path.join(OUT_DIR, f'log_{volume_col}_hist_full_day.png')
fig_all.savefig(all_path, dpi=200, bbox_inches='tight')
plt.close(fig_all)

print(hourly_path)
print(all_path)
print(
    f"log-gamma fit for {log_name}: "
    f"k={fit_stats['k']:.6f}, theta={fit_stats['theta']:.6f}, "
    f"KS={fit_stats['ks_stat']:.6f}, p={fit_stats['ks_pvalue']:.6g}, "
    f"n_fit={fit_stats['n_fit']}"
)
