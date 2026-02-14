"""
Step 2: Empirical Distribution Visualization

This script demonstrates how to visualize the empirical distributions of log-volumes
for bid and ask sides using the LOBSampleConstructor and LOBVisualization modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lob_sample_construction import LOBSampleConstructor
from visualization import LOBVisualization


def example_empirical_distribution():
    """
    Example: Generate empirical distribution histograms.
    
    This matches the reference code pattern:
    
    K = 10
    bid_vol = []
    ask_vol = []
    
    for snap in lob_snapshots:
        bid_vol.extend(snap.bid_volumes[:K])
        ask_vol.extend(snap.ask_volumes[:K])
    
    bid_vol = np.array(bid_vol)
    ask_vol = np.array(ask_vol)
    
    bid_log = np.log(bid_vol[bid_vol > 0])
    ask_log = np.log(ask_vol[ask_vol > 0])
    
    plt.figure()
    plt.hist(bid_log, bins=100, density=True)
    plt.title("Sanofi LOB: empirical density of log-volumes (BID)")
    
    plt.figure()
    plt.hist(ask_log, bins=100, density=True)
    plt.title("Sanofi LOB: empirical density of log-volumes (ASK)")
    """
    
    print("="*70)
    print("STEP 2: EMPIRICAL DISTRIBUTION VISUALIZATION")
    print("="*70)
    
    # Step 1: Initialize LOB sample constructor
    orderbook_file = Path(__file__).parent.parent / "orderbook_construction" / "sample_data.csv"
    
    if not orderbook_file.exists():
        print(f"Error: Orderbook file not found at {orderbook_file}")
        print("Please provide a valid orderbook data file.")
        return
    
    print(f"\n[1] Loading orderbook data from {orderbook_file.name}...")
    constructor = LOBSampleConstructor(str(orderbook_file), max_depth=10)
    
    # Step 2: Generate timestamps (example: every 5 seconds for 1 hour)
    print("[2] Generating sample timestamps...")
    start_time = 0
    end_time = 3_600_000_000_000  # 1 hour in nanoseconds
    step = 5_000_000_000  # 5 seconds in nanoseconds
    
    timestamps = np.arange(start_time, end_time, step)
    print(f"    Total snapshots to process: {len(timestamps)}")
    
    # Step 3: Construct samples
    print("[3] Constructing log-volume samples...")
    constructor.construct_samples(timestamps)
    
    # Step 4: Get samples
    print("[4] Extracting samples...")
    log_volumes_bid, log_volumes_ask = constructor.get_samples()
    
    # Step 5: Display statistics
    stats = constructor.get_summary_stats()
    print("\n[5] Sample Statistics:")
    print(f"\n    BID side:")
    print(f"      Sample size: {stats['bid']['count']:,}")
    print(f"      Mean: {stats['bid']['mean']:.4f}")
    print(f"      Std Dev: {stats['bid']['std']:.4f}")
    print(f"      Range: [{stats['bid']['min']:.4f}, {stats['bid']['max']:.4f}]")
    
    print(f"\n    ASK side:")
    print(f"      Sample size: {stats['ask']['count']:,}")
    print(f"      Mean: {stats['ask']['mean']:.4f}")
    print(f"      Std Dev: {stats['ask']['std']:.4f}")
    print(f"      Range: [{stats['ask']['min']:.4f}, {stats['ask']['max']:.4f}]")
    
    # Step 6: Visualization - Combined histograms
    print("\n[6] Generating empirical distribution histograms...")
    LOBVisualization.plot_empirical_histograms(
        log_volumes_bid, 
        log_volumes_ask,
        bins=100,
        figsize=(14, 5),
        save_path=None  # Change to "histograms_combined.png" to save
    )
    
    # Step 7: Visualization - Separate histograms
    print("[7] Generating separate histograms...")
    # Uncomment to save:
    # LOBVisualization.plot_separate_histograms(
    #     log_volumes_bid,
    #     log_volumes_ask,
    #     bins=100,
    #     save_dir="."
    # )
    
    # Step 8: Visualization - Empirical CDF
    print("[8] Generating empirical CDFs...")
    LOBVisualization.plot_empirical_cdf(
        log_volumes_bid,
        log_volumes_ask,
        figsize=(14, 5),
        save_path=None  # Change to filename to save
    )
    
    # Step 9: Visualization - QQ plots
    print("[9] Generating Q-Q plots (vs Normal)...")
    LOBVisualization.plot_qq_against_normal(
        log_volumes_bid,
        log_volumes_ask,
        figsize=(14, 5),
        save_path=None  # Change to filename to save
    )
    
    # Step 10: Compute detailed statistics
    print("\n[10] Computing detailed statistics...")
    detailed_stats = LOBVisualization.get_comparison_stats(
        log_volumes_bid,
        log_volumes_ask
    )
    
    print("\n     BID side statistics:")
    for key, value in detailed_stats['bid'].items():
        if isinstance(value, float):
            print(f"       {key:12s}: {value:12.6f}")
        else:
            print(f"       {key:12s}: {value:12,}")
    
    print("\n     ASK side statistics:")
    for key, value in detailed_stats['ask'].items():
        if isinstance(value, float):
            print(f"       {key:12s}: {value:12.6f}")
        else:
            print(f"       {key:12s}: {value:12,}")
    
    print("\n" + "="*70)
    print("Step 2 complete: Empirical distributions visualized")
    print("="*70)
    
    return log_volumes_bid, log_volumes_ask, stats, detailed_stats


if __name__ == "__main__":
    bid_data, ask_data, basic_stats, detailed_stats = example_empirical_distribution()
