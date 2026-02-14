#!/usr/bin/env python3
"""
Complete Gamma Distribution Analysis Pipeline for Sanofi LOB

Processes OrderUpdate data and generates empirical and fitted distribution plots.
"""

import sys
import numpy as np
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "orderbook_construction"))

from lob_sample_construction import LOBSampleConstructor
from visualization import LOBVisualization
from gamma_fitting import GammaDistributionFitter

# 使用真实的OrderUpdate数据
ORDERBOOK_FILE = "/Users/charles/Documents/Tâches/Mathématiques/EA/EA_recherche/euronextparis/EuronextParis/EuronextParis_20191001_FR0000120578/FR0000120578/OrderUpdate_20191001_FR0000120578.csv"

OUTPUT_DIR = Path(__file__).parent

def main():
    print("\n" + "="*70)
    print("GAMMA DISTRIBUTION ANALYSIS - SANOFI LOB")
    print("Real OrderBook Data from 2019-10-01")
    print("="*70)
    
    # ====================================================================
    # STEP 1: CONSTRUCT LOB SAMPLES
    # ====================================================================
    print("\n[STEP 1] Constructing LOB samples from OrderUpdate data...")
    
    constructor = LOBSampleConstructor(ORDERBOOK_FILE, max_depth=1)
    
    # Find actual time range in the data
    print("  Scanning data for time range...")
    min_time_ns = float('inf')
    max_time_ns = 0
    
    # Import directly
    import sys
    import os
    orderbook_path = os.path.join(os.path.dirname(ORDERBOOK_FILE), '..', '..', 'orderbook_construction')
    sys.path.insert(0, orderbook_path)
    from orderbook import iter_orderupdate_file
    
    for u in iter_orderupdate_file(ORDERBOOK_FILE):
        if u.event_time < min_time_ns:
            min_time_ns = u.event_time
        if u.event_time > max_time_ns:
            max_time_ns = u.event_time
    
    print(f"  Data range: {min_time_ns} to {max_time_ns} ns")
    
    # Generate timestamps from actual data range
    # Using 1 second intervals for good performance while maintaining data density
    step = 1_000_000_000  # 1 second
    timestamps = np.arange(min_time_ns, max_time_ns, step)
    
    print(f"  Processing {len(timestamps)} snapshots")
    
    constructor.construct_samples(timestamps)
    log_volumes_bid, log_volumes_ask = constructor.get_samples()
    
    print(f"  ✓ BID samples collected: {len(log_volumes_bid):,}")
    print(f"  ✓ ASK samples collected: {len(log_volumes_ask):,}")
    
    # ====================================================================
    # STEP 2: EMPIRICAL DISTRIBUTION VISUALIZATION
    # ====================================================================
    print("\n[STEP 2] Plotting empirical distributions...")
    
    hist_path = OUTPUT_DIR / "01_empirical_histograms.png"
    LOBVisualization.plot_empirical_histograms(
        log_volumes_bid,
        log_volumes_ask,
        bins=100,
        figsize=(14, 5),
        save_path=str(hist_path)
    )
    
    # ====================================================================
    # STEP 3: GAMMA DISTRIBUTION FITTING (MLE)
    # ====================================================================
    print("\n[STEP 3] Fitting Gamma distribution (Maximum Likelihood Estimation)...")
    
    fitter = GammaDistributionFitter(log_volumes_bid, log_volumes_ask, negative_threshold=0.1)
    bid_result, ask_result = fitter.fit()
    
    # ====================================================================
    # STEP 4: PLOT FITTED DISTRIBUTIONS
    # ====================================================================
    print("\n[STEP 4] Plotting fitted distributions with theoretical curves...")
    
    fitted_path = OUTPUT_DIR / "02_gamma_fitted_histograms.png"
    fitter.plot_fitted_histograms(
        bins=100,
        figsize=(14, 5),
        save_path=str(fitted_path)
    )
    
    # ====================================================================
    # STEP 5: RESULTS SUMMARY
    # ====================================================================
    print("\n[STEP 5] Results Summary:")
    print(fitter.get_summary_report())
    
    # Additional comparison
    print("\nPARAMETER COMPARISON (BID vs ASK):")
    print("-" * 70)
    print(f"{'Metric':<25} {'BID':<20} {'ASK':<20}")
    print("-" * 70)
    print(f"{'k (shape parameter)':<25} {bid_result.k:<20.6f} {ask_result.k:<20.6f}")
    print(f"{'θ (scale parameter)':<25} {bid_result.theta:<20.6f} {ask_result.theta:<20.6f}")
    print(f"{'Mean (data)':<25} {bid_result.mean:<20.6f} {ask_result.mean:<20.6f}")
    print(f"{'Std Dev (data)':<25} {bid_result.std:<20.6f} {ask_result.std:<20.6f}")
    print(f"{'KS Statistic':<25} {bid_result.ks_statistic:<20.6f} {ask_result.ks_statistic:<20.6f}")
    print(f"{'KS p-value':<25} {bid_result.ks_pvalue:<20.6f} {ask_result.ks_pvalue:<20.6f}")
    print("-" * 70)
    
    # Interpretation
    print("\nINTERPRETATION:")
    print("-" * 70)
    
    k_ratio = ask_result.k / bid_result.k
    theta_ratio = ask_result.theta / bid_result.theta
    
    print(f"\n1. Shape Parameter Ratio (k_ask / k_bid): {k_ratio:.4f}")
    if k_ratio > 1.1:
        print("   → ASK side has HIGHER shape: more concentrated, less variability")
    elif k_ratio < 0.9:
        print("   → ASK side has LOWER shape: more dispersed, higher variability")
    else:
        print("   → Similar shape parameters (balanced liquidity distribution)")
    
    print(f"\n2. Scale Parameter Ratio (θ_ask / θ_bid): {theta_ratio:.4f}")
    if theta_ratio > 1.1:
        print("   → ASK side has LARGER scale: higher typical volumes")
    elif theta_ratio < 0.9:
        print("   → ASK side has SMALLER scale: lower typical volumes")
    else:
        print("   → Similar scale parameters (symmetric volume distribution)")
    
    print("\n3. Goodness-of-Fit (Kolmogorov-Smirnov test):")
    print(f"   BID: p-value = {bid_result.ks_pvalue:.6f}", end="")
    if bid_result.ks_pvalue > 0.05:
        print(" ✓ (Gamma fit is acceptable, p > 0.05)")
    else:
        print(" ✗ (Gamma fit rejected, p < 0.05)")
    
    print(f"   ASK: p-value = {ask_result.ks_pvalue:.6f}", end="")
    if ask_result.ks_pvalue > 0.05:
        print(" ✓ (Gamma fit is acceptable, p > 0.05)")
    else:
        print(" ✗ (Gamma fit rejected, p < 0.05)")
    
    # ====================================================================
    # FINAL OUTPUT
    # ====================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\n✓ Generated visualizations:")
    print(f"  1. {hist_path.name}")
    print(f"     - Empirical histograms (BID and ASK)")
    print(f"     - Sample sizes and basic statistics")
    print(f"\n  2. {fitted_path.name}")
    print(f"     - Empirical histograms with Gamma PDF overlay")
    print(f"     - Fitted parameters (k, θ)")
    print(f"     - Goodness-of-fit statistics")
    
    print(f"\n✓ All files saved to:")
    print(f"  {OUTPUT_DIR}")
    
    return {
        'bid_result': bid_result,
        'ask_result': ask_result,
        'fitter': fitter,
        'constructor': constructor,
    }


if __name__ == "__main__":
    results = main()
