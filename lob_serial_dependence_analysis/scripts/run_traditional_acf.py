"""
Traditional ACF Analysis Pipeline
Focus: Queue Size Persistence in LOB
"""

import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
RESULTS_DIR = ANALYSIS_ROOT / "results"

from traditional_acf_analysis import TraditionalACFAnalyzer


# Configuration
SNAPSHOT_FILE = PROJECT_ROOT / "sanofi_book_snapshots_1s.parquet"


def load_log_volumes_from_snapshot(snapshot_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and construct log-volume samples directly from 1s snapshot parquet."""
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    df = pd.read_parquet(snapshot_path)
    required_cols = ["bidvolume1", "askvolume1"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in snapshot parquet: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    bid_vol = pd.to_numeric(df["bidvolume1"], errors="coerce")
    ask_vol = pd.to_numeric(df["askvolume1"], errors="coerce")

    log_volumes_bid = np.log(bid_vol[bid_vol > 0].to_numpy(dtype=float))
    log_volumes_ask = np.log(ask_vol[ask_vol > 0].to_numpy(dtype=float))

    return log_volumes_bid, log_volumes_ask


def main():
    """Execute traditional ACF analysis."""
    
    print("\n" + "="*70)
    print("TRADITIONAL ACF ANALYSIS PIPELINE")
    print("Queue Size Persistence in Limit Order Book")
    print("="*70)
    
    # [STEP 1] Load LOB samples from snapshot parquet
    print("\n[STEP 1] Loading LOB log-volume samples from snapshot parquet...")
    print(f"  Source: {SNAPSHOT_FILE}")
    log_volumes_bid, log_volumes_ask = load_log_volumes_from_snapshot(SNAPSHOT_FILE)
    
    print(f"  ✓ Samples collected: BID={len(log_volumes_bid):,}, ASK={len(log_volumes_ask):,}")
    
    # [STEP 2] Traditional ACF Analysis
    print("\n[STEP 2] Computing Traditional Autocorrelation (nlags=100)...")
    analyzer = TraditionalACFAnalyzer(log_volumes_bid, log_volumes_ask)
    
    result_bid, result_ask = analyzer.compute_acf(nlags=100, method='fft')
    
    # [STEP 3] Visualizations
    print("\n[STEP 3] Generating visualizations...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    analyzer.plot_acf_detailed(
        save_path=str(RESULTS_DIR / "05_traditional_acf.png")
    )
    
    analyzer.plot_ljung_box(
        save_path=str(RESULTS_DIR / "06_ljung_box_test.png")
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
