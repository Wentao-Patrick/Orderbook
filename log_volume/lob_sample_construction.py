"""
LOB Log-Volumes Sample Construction for Gamma Distribution Analysis

This module implements the construction of log-volume samples from the 
Limit Order Book (LOB) for Sanofi, separated by side (bid/ask).

Reference implementation following the mathematical specification:
- X_bid = {log(V_i^bid(t_j)) : V_i^bid(t_j) > 0, i ≤ K}
- X_ask = {log(V_i^ask(t_j)) : V_i^ask(t_j) > 0, i ≤ K}

where:
- V_i^bid/ask(t_j): Volume at depth level i for snapshot at time t_j
- K: Maximum depth level (default K=10)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import sys

# Add parent directory to path to import orderbook module
sys.path.insert(0, str(Path(__file__).parent.parent / "orderbook_construction"))
from log_volume.orderbook_1 import OrderBookReplayer


class LOBSampleConstructor:
    """Constructs log-volume samples from LOB snapshots."""
    
    def __init__(self, orderbook_path: str, max_depth: int = 10):
        """
        Initialize the LOB sample constructor.
        
        Parameters
        ----------
        orderbook_path : str
            Path to the orderbook data file
        max_depth : int
            Maximum depth level K to consider (default: 10)
        """
        self.orderbook_path = orderbook_path
        self.max_depth = max_depth
        self.replayer = OrderBookReplayer(orderbook_path)
        
        # Storage for samples
        self.log_volumes_bid: List[float] = []
        self.log_volumes_ask: List[float] = []
        
    def extract_depth_volumes(self, t_ns: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract volume arrays at each depth level for a given snapshot.
        
        Parameters
        ----------
        t_ns : int
            Timestamp in nanoseconds
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - bid_volumes: Array of volumes at bid side, level 1 to K
            - ask_volumes: Array of volumes at ask side, level 1 to K
        """
        # Get snapshot at time t
        self.replayer.replay_until(t_ns)
        
        # Build depth from non-cumulative levels
        bid_levels: Dict[float, float] = {}
        ask_levels: Dict[float, float] = {}
        
        for o in self.replayer.book.orders.values():
            if o.qty <= 0:
                continue
            # Allow price <= 0 (market orders) but still collect them
            if o.price < 0:
                continue
                
            if o.side == 1:  # Bid
                bid_levels[o.price] = bid_levels.get(o.price, 0.0) + o.qty
            elif o.side == 2:  # Ask
                ask_levels[o.price] = ask_levels.get(o.price, 0.0) + o.qty
        
        # Sort and extract top K levels
        bid_prices = sorted(bid_levels.keys(), reverse=True)[:self.max_depth]
        ask_prices = sorted(ask_levels.keys())[:self.max_depth]
        
        bid_volumes = np.array([bid_levels[p] for p in bid_prices], dtype=float) if len(bid_prices) > 0 else np.array([])
        ask_volumes = np.array([ask_levels[p] for p in ask_prices], dtype=float) if len(ask_prices) > 0 else np.array([])
        
        return bid_volumes, ask_volumes
    
    def construct_samples(self, timestamps: np.ndarray) -> None:
        """
        Construct log-volume samples from a sequence of snapshots.
        
        Parameters
        ----------
        timestamps : np.ndarray
            Array of timestamps (in nanoseconds) for snapshots
        """
        self.log_volumes_bid.clear()
        self.log_volumes_ask.clear()
        
        self.replayer.reset()
        
        # Optimize: batch process timestamps to avoid repeated replayer resets
        last_replay_time = 0
        
        for i, t_ns in enumerate(timestamps):
            t_ns = int(t_ns)
            
            # Only replay if we haven't passed this timestamp yet
            if t_ns >= last_replay_time:
                try:
                    bid_vols, ask_vols = self.extract_depth_volumes(t_ns)
                    last_replay_time = t_ns
                    
                    # Append log-volumes for all positive volumes
                    # Bid side
                    for vol in bid_vols:
                        if vol > 0:
                            self.log_volumes_bid.append(np.log(vol))
                    
                    # Ask side
                    for vol in ask_vols:
                        if vol > 0:
                            self.log_volumes_ask.append(np.log(vol))
                    
                    # Progress indicator
                    if (i + 1) % 1000 == 0:
                        print(f"    Processed {i + 1}/{len(timestamps)} snapshots...")
                        
                except Exception as e:
                    # Silently skip problematic timestamps
                    continue
    
    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the constructed log-volume samples.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - log_volumes_bid: Log-volumes for bid side
            - log_volumes_ask: Log-volumes for ask side
        """
        return np.array(self.log_volumes_bid), np.array(self.log_volumes_ask)
    
    def get_summary_stats(self) -> Dict[str, dict]:
        """
        Get summary statistics for both sides.
        
        Returns
        -------
        Dict[str, dict]
            Dictionary containing statistics for bid and ask sides
        """
        bid_array = np.array(self.log_volumes_bid)
        ask_array = np.array(self.log_volumes_ask)
        
        stats = {
            'bid': {
                'count': len(bid_array),
                'mean': float(np.mean(bid_array)) if len(bid_array) > 0 else np.nan,
                'std': float(np.std(bid_array)) if len(bid_array) > 0 else np.nan,
                'min': float(np.min(bid_array)) if len(bid_array) > 0 else np.nan,
                'max': float(np.max(bid_array)) if len(bid_array) > 0 else np.nan,
            },
            'ask': {
                'count': len(ask_array),
                'mean': float(np.mean(ask_array)) if len(ask_array) > 0 else np.nan,
                'std': float(np.std(ask_array)) if len(ask_array) > 0 else np.nan,
                'min': float(np.min(ask_array)) if len(ask_array) > 0 else np.nan,
                'max': float(np.max(ask_array)) if len(ask_array) > 0 else np.nan,
            }
        }
        
        return stats


def main():
    """
    Example usage of LOBSampleConstructor.
    """
    # Example with sample data
    orderbook_file = Path(__file__).parent.parent / "orderbook_construction" / "sample_data.csv"
    
    if not orderbook_file.exists():
        print(f"Error: Orderbook file not found at {orderbook_file}")
        print("Please provide a valid orderbook data file.")
        return
    
    # Initialize constructor
    constructor = LOBSampleConstructor(str(orderbook_file), max_depth=10)
    
    # Generate example timestamps (every 5 seconds for 1 minute)
    # In practice, these would come from your actual data
    start_time = 0
    end_time = 60_000_000_000  # 60 seconds in nanoseconds
    step = 5_000_000_000  # 5 seconds in nanoseconds
    
    timestamps = np.arange(start_time, end_time, step)
    
    # Construct samples
    print(f"Constructing log-volume samples for {len(timestamps)} snapshots...")
    constructor.construct_samples(timestamps)
    
    # Get results
    log_vols_bid, log_vols_ask = constructor.get_samples()
    stats = constructor.get_summary_stats()
    
    # Display statistics
    print("\n" + "="*60)
    print("LOG-VOLUME SAMPLE SUMMARY")
    print("="*60)
    print(f"\nBID SIDE:")
    print(f"  Sample size: {stats['bid']['count']}")
    print(f"  Mean (log-volume): {stats['bid']['mean']:.4f}")
    print(f"  Std Dev: {stats['bid']['std']:.4f}")
    print(f"  Range: [{stats['bid']['min']:.4f}, {stats['bid']['max']:.4f}]")
    
    print(f"\nASK SIDE:")
    print(f"  Sample size: {stats['ask']['count']}")
    print(f"  Mean (log-volume): {stats['ask']['mean']:.4f}")
    print(f"  Std Dev: {stats['ask']['std']:.4f}")
    print(f"  Range: [{stats['ask']['min']:.4f}, {stats['ask']['max']:.4f}]")
    print("="*60)


if __name__ == "__main__":
    main()
