"""
Traditional ACF Analysis Pipeline
Focus: Queue Size Persistence in LOB
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "orderbook_construction"))

from lob_sample_construction import LOBSampleConstructor
from traditional_acf_analysis import TraditionalACFAnalyzer
from orderbook import iter_orderupdate_file


# Configuration
ORDERBOOK_FILE = "/Users/charles/Documents/Tâches/Mathématiques/EA/EA_recherche/euronextparis/EuronextParis/EuronextParis_20191001_FR0000120578/FR0000120578/OrderUpdate_20191001_FR0000120578.csv"

MAX_DEPTH = 1  # K=1 (best bid/ask)
TIME_STEP = 1_000_000_000  # 1 second


def main():
    """Execute traditional ACF analysis."""
    
    print("\n" + "="*70)
    print("TRADITIONAL ACF ANALYSIS PIPELINE")
    print("Queue Size Persistence in Limit Order Book")
    print("="*70)
    
    # [STEP 1] Construct LOB samples
    print("\n[STEP 1] Constructing LOB log-volume samples...")
    constructor = LOBSampleConstructor(ORDERBOOK_FILE, max_depth=MAX_DEPTH)
    
    # Scan for timestamp range
    print("  Scanning data...")
    min_time_ns = float('inf')
    max_time_ns = float('-inf')
    
    for u in iter_orderupdate_file(ORDERBOOK_FILE):
        min_time_ns = min(min_time_ns, u.event_time)
        max_time_ns = max(max_time_ns, u.event_time)
    
    timestamps = np.arange(min_time_ns, max_time_ns, TIME_STEP)
    print(f"  {len(timestamps):,} snapshots generated")
    
    # Construct samples
    print(f"  Constructing samples...")
    constructor.construct_samples(timestamps)
    log_volumes_bid, log_volumes_ask = constructor.get_samples()
    
    print(f"  ✓ Samples collected: BID={len(log_volumes_bid):,}, ASK={len(log_volumes_ask):,}")
    
    # [STEP 2] Traditional ACF Analysis
    print("\n[STEP 2] Computing Traditional Autocorrelation (nlags=100)...")
    analyzer = TraditionalACFAnalyzer(log_volumes_bid, log_volumes_ask)
    
    result_bid, result_ask = analyzer.compute_acf(nlags=100, method='fft')
    
    # [STEP 3] Visualizations
    print("\n[STEP 3] Generating visualizations...")
    analyzer.plot_acf_detailed(
        save_path="/Users/charles/Documents/Tâches/Mathématiques/EA/EA_recherche/gamma_distribution_analysis/05_traditional_acf.png"
    )
    
    analyzer.plot_ljung_box(
        save_path="/Users/charles/Documents/Tâches/Mathématiques/EA/EA_recherche/gamma_distribution_analysis/06_ljung_box_test.png"
    )
    
    # [STEP 4] LOB Interpretation
    print("\n[STEP 4] LOB Microstructure Interpretation...")
    analyzer.print_lob_interpretation()
    
    # [STEP 5] Summary Report
    print("\n" + "="*70)
    print("TRADITIONAL ACF ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nKEY METRICS:")
    print("-" * 70)
    
    for result in [result_bid, result_ask]:
        print(f"\n{result.side.upper()} SIDE:")
        print(f"  ACF(1): {result.acf_values[1]:.4f} ← Order splitting indicator")
        print(f"  ACF(5): {result.acf_values[5]:.4f}")
        print(f"  ACF(10): {result.acf_values[10]:.4f}")
        print(f"  ACF(20): {result.acf_values[20]:.4f}")
        half_life_str = f"{result.half_life:.2f}" if result.half_life else "N/A"
        print(f"  Half-life: {half_life_str} seconds")
        print(f"  Significant lags: {len(result.significant_lags)}/100")
        
        # Ljung-Box summary
        rejections = np.sum(result.ljung_box_pvalues < 0.05)
        print(f"  Ljung-Box rejections (α=0.05): {rejections}/100")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    
    print("""
1. ORDER SPLITTING (Lag 1-5):
   • High ACF(1) > 0.8 → Orders are fragmented
   • Liquidity providers break large orders into pieces
   • Each piece executed sequentially → persistent depth
   
2. LIQUIDITY PERSISTENCE (Lag 5-20):
   • Slow decay → Market in stable regime
   • Fast decay → Active mean reversion
   • Shape indicates supply/demand elasticity
   
3. LJUNG-BOX TEST:
   • H₀: No serial correlation
   • p-value < 0.05 → Reject H₀, significant dependence
   • Tests whether information can be extracted from past
   
4. PRACTICAL IMPLICATIONS:
   • Queue size NOT random → Predictable component
   • Can use AR/ARMA models for volume forecasting
   • Market efficiency violated (weak form)
   """)
    
    print("="*70)
    print("FILES GENERATED:")
    print("  1. 05_traditional_acf.png - ACF plots with zones")
    print("  2. 06_ljung_box_test.png - Statistical test results")
    print("="*70)


if __name__ == "__main__":
    main()
