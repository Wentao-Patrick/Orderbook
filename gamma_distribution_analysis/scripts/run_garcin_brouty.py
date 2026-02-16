#!/usr/bin/env python3
"""
Run Garcin-Brouty (2023) Information-Theoretic Analysis
========================================================

End-to-end pipeline for analyzing binary symbolic volume sequences
using information-theoretic measures from Garcin-Brouty paper.
"""

import numpy as np
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
RESULTS_DIR = ANALYSIS_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT / "orderbook_construction"))

from lob_sample_construction import LOBSampleConstructor
from orderbook import iter_orderupdate_file

# Import analysis module
from garcin_brouty_method import (
    GarcinBroutyAnalyzer,
    GarcinBroutyVisualizer,
    GarcinBroutyInterpretation
)

# Configuration
ORDERBOOK_FILE = "/Users/charles/Documents/Tâches/Mathématiques/EA/EA_recherche/euronextparis/EuronextParis/EuronextParis_20191001_FR0000120578/FR0000120578/OrderUpdate_20191001_FR0000120578.csv"

MAX_DEPTH = 1  # K=1 (best bid/ask)
TIME_STEP = 1_000_000_000  # 1 second
MAX_L = 20  # Maximum pattern length


def construct_lob_samples():
    """Construct LOB log-volume samples - optimized for speed"""
    print(f"\n[STEP 1] Constructing LOB log-volume samples (K={MAX_DEPTH}, Δt=1s)...")
    
    constructor = LOBSampleConstructor(ORDERBOOK_FILE, max_depth=MAX_DEPTH)
    
    print(f"  Scanning data...")
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
    
    return log_volumes_bid, log_volumes_ask



def main():
    """Main execution pipeline"""
    
    print("\n" + "="*72)
    print("GARCIN-BROUTY INFORMATION-THEORETIC ANALYSIS PIPELINE")
    print("Symbolic Representation + Shannon Entropy")
    print("="*72)
    
    # Step 1: Construct LOB samples
    bid_volumes, ask_volumes = construct_lob_samples()
    
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
