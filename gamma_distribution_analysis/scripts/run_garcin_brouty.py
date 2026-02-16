#!/usr/bin/env python3
"""
Run Garcin-Brouty (2023) Information-Theoretic Analysis
========================================================

End-to-end pipeline for analyzing binary symbolic volume sequences
using information-theoretic measures from Garcin-Brouty paper.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
RESULTS_DIR = ANALYSIS_ROOT / "results"

# Import analysis module
from garcin_brouty_method import (
    GarcinBroutyAnalyzer,
    GarcinBroutyVisualizer,
    GarcinBroutyInterpretation
)

# Configuration
SNAPSHOT_FILE = PROJECT_ROOT / "sanofi_book_snapshots_1s.parquet"

MAX_L = 20  # Maximum pattern length


def load_log_volumes_from_snapshot(snapshot_path: Path):
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
    
    print(f"  ✓ Samples collected: BID={len(log_volumes_bid):,}, ASK={len(log_volumes_ask):,}")
    
    return log_volumes_bid, log_volumes_ask



def main():
    """Main execution pipeline"""
    
    print("\n" + "="*72)
    print("GARCIN-BROUTY INFORMATION-THEORETIC ANALYSIS PIPELINE")
    print("Symbolic Representation + Shannon Entropy")
    print("="*72)
    
    # Step 1: Load LOB samples
    print(f"\n[STEP 1] Loading LOB log-volume samples from snapshot parquet...")
    print(f"  Source: {SNAPSHOT_FILE}")
    bid_volumes, ask_volumes = load_log_volumes_from_snapshot(SNAPSHOT_FILE)
    
    # Step 2: Perform Garcin-Brouty analysis
    print(f"\n[STEP 2] Computing Garcin-Brouty analysis (max_L={MAX_L})...")
    
    analyzer = GarcinBroutyAnalyzer(bid_volumes, max_L=MAX_L)
    bid_result, ask_result = analyzer.analyze_by_side(bid_volumes, ask_volumes)
    
    print(f"  ✓ BID analysis complete:")
    print(f"    Max information: {bid_result.max_information:.6f} nats")
    print(f"    Half-life L: {bid_result.half_life_L}")
    
    print(f"  ✓ ASK analysis complete:")
    print(f"    Max information: {ask_result.max_information:.6f} nats")
    print(f"    Half-life L: {ask_result.half_life_L}")
    
    # Step 3: Generate visualizations
    print(f"\n[STEP 3] Generating visualizations...")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Information quantity plot
    fig1, _ = GarcinBroutyVisualizer.plot_information_quantity(
        bid_result, ask_result,
        save_path=str(RESULTS_DIR / "07_gb_information_quantity.png")
    )
    
    # Entropy comparison plot
    fig2, _ = GarcinBroutyVisualizer.plot_entropy_comparison(
        bid_result, ask_result,
        save_path=str(RESULTS_DIR / "08_gb_entropy_comparison.png")
    )
    
    # Conditional probabilities plot
    fig3, _ = GarcinBroutyVisualizer.plot_conditional_probabilities(
        bid_result, ask_result,
        max_L=5,
        save_path=str(RESULTS_DIR / "09_gb_conditional_probs.png")
    )
    
    # Step 4: Print interpretation
    print(f"\n[STEP 4] LOB Microstructure Interpretation...")
    GarcinBroutyInterpretation.print_interpretation(bid_result, ask_result, symbol="SANOFI")
    
    print("\n" + "="*72)
    print("FILES GENERATED:")
    print("  1. 07_gb_information_quantity.png")
    print("  2. 08_gb_entropy_comparison.png")
    print("  3. 09_gb_conditional_probs.png")
    print("="*72)


if __name__ == "__main__":
    main()
