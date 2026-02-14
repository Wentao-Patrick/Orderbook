"""
Step 3-5: Complete Gamma Distribution Fitting Pipeline

Workflow:
1. Construct LOB samples from orderbook
2. Visualize empirical distributions
3. Fit Gamma distribution (MLE)
4. Plot theoretical curves overlay
5. Output parameters and statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from lob_sample_construction import LOBSampleConstructor
from visualization import LOBVisualization
from gamma_fitting import GammaDistributionFitter


def complete_analysis_pipeline(orderbook_file: str,
                              max_depth: int = 10,
                              snapshot_interval_ns: int = 5_000_000_000,
                              analysis_duration_ns: int = 3_600_000_000_000):
    """
    Execute complete analysis pipeline from LOB data to Gamma fitting.
    
    Parameters
    ----------
    orderbook_file : str
        Path to orderbook data file
    max_depth : int
        Maximum LOB depth to consider (default: 10)
    snapshot_interval_ns : int
        Time interval between snapshots in nanoseconds (default: 5 seconds)
    analysis_duration_ns : int
        Total duration of analysis in nanoseconds (default: 1 hour)
    """
    
    print("\n" + "="*70)
    print("COMPLETE GAMMA DISTRIBUTION ANALYSIS PIPELINE")
    print("Sanofi (FR0000120578) - Limit Order Book Analysis")
    print("="*70)
    
    # ========================================================================
    # STEP 1: CONSTRUCT LOB SAMPLES
    # ========================================================================
    print("\n" + "─"*70)
    print("STEP 1: SAMPLE CONSTRUCTION")
    print("─"*70)
    
    print(f"\n[1.1] Loading orderbook data...")
    print(f"      File: {Path(orderbook_file).name}")
    
    try:
        constructor = LOBSampleConstructor(orderbook_file, max_depth=max_depth)
    except Exception as e:
        print(f"ERROR: Failed to load orderbook: {e}")
        return
    
    print(f"[1.2] Generating snapshot timestamps...")
    start_time = 0
    end_time = analysis_duration_ns
    step = snapshot_interval_ns
    
    timestamps = np.arange(start_time, end_time, step)
    print(f"      Total snapshots: {len(timestamps)}")
    print(f"      Time range: {start_time/1e9:.0f}s to {end_time/1e9:.0f}s")
    print(f"      Interval: {step/1e9:.1f}s")
    
    print(f"[1.3] Constructing log-volume samples...")
    constructor.construct_samples(timestamps)
    
    log_volumes_bid, log_volumes_ask = constructor.get_samples()
    basic_stats = constructor.get_summary_stats()
    
    print(f"      BID samples: {len(log_volumes_bid):,}")
    print(f"      ASK samples: {len(log_volumes_ask):,}")
    
    # ========================================================================
    # STEP 2: VISUALIZE EMPIRICAL DISTRIBUTIONS
    # ========================================================================
    print("\n" + "─"*70)
    print("STEP 2: EMPIRICAL DISTRIBUTION VISUALIZATION")
    print("─"*70)
    
    print("\n[2.1] Plotting empirical histograms...")
    LOBVisualization.plot_empirical_histograms(
        log_volumes_bid,
        log_volumes_ask,
        bins=100,
        figsize=(14, 5),
        save_path=None
    )
    
    print("[2.2] Plotting empirical CDFs...")
    LOBVisualization.plot_empirical_cdf(
        log_volumes_bid,
        log_volumes_ask,
        figsize=(14, 5),
        save_path=None
    )
    
    # ========================================================================
    # STEP 3: FIT GAMMA DISTRIBUTION
    # ========================================================================
    print("\n" + "─"*70)
    print("STEP 3: GAMMA DISTRIBUTION FITTING")
    print("─"*70)
    
    print("\n[3.1] Initializing Gamma fitter...")
    fitter = GammaDistributionFitter(
        log_volumes_bid,
        log_volumes_ask,
        negative_threshold=0.1  # Switch to volume fitting if >10% negative
    )
    
    print("[3.2] Analyzing data characteristics and deciding fit strategy...")
    bid_result, ask_result = fitter.fit()
    
    # ========================================================================
    # STEP 4: PLOT FITTED DISTRIBUTIONS
    # ========================================================================
    print("\n" + "─"*70)
    print("STEP 4: VISUALIZATION WITH FITTED CURVES")
    print("─"*70)
    
    print("\n[4.1] Plotting fitted histograms (combined)...")
    fitter.plot_fitted_histograms(
        bins=100,
        figsize=(14, 5),
        save_path=None
    )
    
    print("[4.2] Plotting fitted histograms (separate)...")
    # Uncomment to save to files:
    # fitter.plot_separate_fitted_histograms(
    #     bins=100,
    #     save_dir="."
    # )
    
    # ========================================================================
    # STEP 5: OUTPUT RESULTS
    # ========================================================================
    print("\n" + "─"*70)
    print("STEP 5: RESULTS SUMMARY")
    print("─"*70)
    
    print("\n[5.1] Fitting Results:")
    print(fitter.get_summary_report())
    
    # ========================================================================
    # ADDITIONAL: PARAMETER COMPARISON
    # ========================================================================
    print("\n" + "─"*70)
    print("PARAMETER COMPARISON (BID vs ASK)")
    print("─"*70)
    
    print(f"\n{'Metric':<20} {'BID':<20} {'ASK':<20}")
    print("─"*60)
    
    print(f"{'k (shape)':<20} {bid_result.k:<20.6f} {ask_result.k:<20.6f}")
    print(f"{'θ (scale)':<20} {bid_result.theta:<20.6f} {ask_result.theta:<20.6f}")
    print(f"{'Mean':<20} {bid_result.mean:<20.6f} {ask_result.mean:<20.6f}")
    print(f"{'Std Dev':<20} {bid_result.std:<20.6f} {ask_result.std:<20.6f}")
    print(f"{'KS statistic':<20} {bid_result.ks_statistic:<20.6f} {ask_result.ks_statistic:<20.6f}")
    print(f"{'KS p-value':<20} {bid_result.ks_pvalue:<20.6f} {ask_result.ks_pvalue:<20.6f}")
    
    # Interpretation
    print("\n" + "─"*70)
    print("INTERPRETATION")
    print("─"*70)
    
    k_ratio = ask_result.k / bid_result.k
    theta_ratio = ask_result.theta / bid_result.theta
    
    print(f"\nShape Parameter Comparison (k_ask / k_bid): {k_ratio:.4f}")
    if k_ratio > 1.1:
        print("  → ASK has HIGHER shape → more concentrated distribution (less variability)")
    elif k_ratio < 0.9:
        print("  → ASK has LOWER shape → more dispersed distribution (more variability)")
    else:
        print("  → Similar shape parameters between sides")
    
    print(f"\nScale Parameter Comparison (θ_ask / θ_bid): {theta_ratio:.4f}")
    if theta_ratio > 1.1:
        print("  → ASK has LARGER scale → volumes tend to be larger")
    elif theta_ratio < 0.9:
        print("  → ASK has SMALLER scale → volumes tend to be smaller")
    else:
        print("  → Similar scale parameters between sides")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    return {
        'constructor': constructor,
        'fitter': fitter,
        'bid_result': bid_result,
        'ask_result': ask_result,
    }


if __name__ == "__main__":
    # Example: Specify your orderbook file path
    orderbook_file = Path(__file__).parent.parent / "orderbook_construction" / "sample_data.csv"
    
    if not orderbook_file.exists():
        print(f"Error: Orderbook file not found at {orderbook_file}")
        print("Please provide a valid orderbook data file.")
        sys.exit(1)
    
    # Run complete pipeline
    results = complete_analysis_pipeline(
        str(orderbook_file),
        max_depth=10,
        snapshot_interval_ns=5_000_000_000,  # 5 seconds
        analysis_duration_ns=3_600_000_000_000,  # 1 hour
    )
