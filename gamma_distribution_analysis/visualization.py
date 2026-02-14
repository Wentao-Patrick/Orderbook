"""
Visualization module for LOB log-volumes empirical distribution analysis.

Provides utilities to visualize:
- Empirical histograms (bid/ask separated)
- Empirical CDF
- QQ plots against Gamma distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from scipy import stats


class LOBVisualization:
    """Visualization utilities for LOB log-volume analysis."""
    
    @staticmethod
    def plot_empirical_histograms(
        log_volumes_bid: np.ndarray,
        log_volumes_ask: np.ndarray,
        bins: int = 100,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot empirical histograms of log-volumes for bid and ask sides.
        
        Parameters
        ----------
        log_volumes_bid : np.ndarray
            Log-volumes for bid side
        log_volumes_ask : np.ndarray
            Log-volumes for ask side
        bins : int
            Number of histogram bins (default: 100)
        figsize : Tuple[int, int]
            Figure size (width, height)
        save_path : Optional[str]
            Path to save the figure. If None, displays plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Bid side histogram
        axes[0].hist(log_volumes_bid, bins=bins, density=True, 
                     alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('log(Volume)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Sanofi LOB: Empirical Density of Log-Volumes (BID)', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].text(0.02, 0.98, f'n = {len(log_volumes_bid):,}', 
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
        
        # Ask side histogram
        axes[1].hist(log_volumes_ask, bins=bins, density=True,
                     alpha=0.7, color='red', edgecolor='black')
        axes[1].set_xlabel('log(Volume)', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title('Sanofi LOB: Empirical Density of Log-Volumes (ASK)',
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].text(0.02, 0.98, f'n = {len(log_volumes_ask):,}',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_separate_histograms(
        log_volumes_bid: np.ndarray,
        log_volumes_ask: np.ndarray,
        bins: int = 100,
        figsize: Tuple[int, int] = (10, 6),
        save_dir: Optional[str] = None
    ) -> None:
        """
        Plot separate histograms for bid and ask sides (full page each).
        
        Parameters
        ----------
        log_volumes_bid : np.ndarray
            Log-volumes for bid side
        log_volumes_ask : np.ndarray
            Log-volumes for ask side
        bins : int
            Number of histogram bins
        figsize : Tuple[int, int]
            Figure size (width, height)
        save_dir : Optional[str]
            Directory to save figures. If None, displays plots.
        """
        # BID histogram
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(log_volumes_bid, bins=bins, density=True,
               alpha=0.7, color='blue', edgecolor='black', linewidth=1.2)
        ax.set_xlabel('log(Volume)', fontsize=13)
        ax.set_ylabel('Density', fontsize=13)
        ax.set_title('Sanofi LOB: Empirical Density of Log-Volumes (BID SIDE)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics box
        stats_text = f'n = {len(log_volumes_bid):,}\n' \
                    f'μ = {np.mean(log_volumes_bid):.4f}\n' \
                    f'σ = {np.std(log_volumes_bid):.4f}\n' \
                    f'min = {np.min(log_volumes_bid):.4f}\n' \
                    f'max = {np.max(log_volumes_bid):.4f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/histogram_bid.png', dpi=300, bbox_inches='tight')
            print(f"BID histogram saved to {save_dir}/histogram_bid.png")
        else:
            plt.show()
        
        # ASK histogram
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(log_volumes_ask, bins=bins, density=True,
               alpha=0.7, color='red', edgecolor='black', linewidth=1.2)
        ax.set_xlabel('log(Volume)', fontsize=13)
        ax.set_ylabel('Density', fontsize=13)
        ax.set_title('Sanofi LOB: Empirical Density of Log-Volumes (ASK SIDE)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics box
        stats_text = f'n = {len(log_volumes_ask):,}\n' \
                    f'μ = {np.mean(log_volumes_ask):.4f}\n' \
                    f'σ = {np.std(log_volumes_ask):.4f}\n' \
                    f'min = {np.min(log_volumes_ask):.4f}\n' \
                    f'max = {np.max(log_volumes_ask):.4f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/histogram_ask.png', dpi=300, bbox_inches='tight')
            print(f"ASK histogram saved to {save_dir}/histogram_ask.png")
        else:
            plt.show()
    
    @staticmethod
    def plot_empirical_cdf(
        log_volumes_bid: np.ndarray,
        log_volumes_ask: np.ndarray,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot empirical CDFs for bid and ask sides.
        
        Parameters
        ----------
        log_volumes_bid : np.ndarray
            Log-volumes for bid side
        log_volumes_ask : np.ndarray
            Log-volumes for ask side
        figsize : Tuple[int, int]
            Figure size
        save_path : Optional[str]
            Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Bid CDF
        bid_sorted = np.sort(log_volumes_bid)
        bid_cdf = np.arange(1, len(bid_sorted) + 1) / len(bid_sorted)
        axes[0].plot(bid_sorted, bid_cdf, linewidth=2, color='blue', label='Empirical CDF')
        axes[0].set_xlabel('log(Volume)', fontsize=12)
        axes[0].set_ylabel('Cumulative Probability', fontsize=12)
        axes[0].set_title('Sanofi LOB: Empirical CDF (BID)', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Ask CDF
        ask_sorted = np.sort(log_volumes_ask)
        ask_cdf = np.arange(1, len(ask_sorted) + 1) / len(ask_sorted)
        axes[1].plot(ask_sorted, ask_cdf, linewidth=2, color='red', label='Empirical CDF')
        axes[1].set_xlabel('log(Volume)', fontsize=12)
        axes[1].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1].set_title('Sanofi LOB: Empirical CDF (ASK)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_qq_against_normal(
        log_volumes_bid: np.ndarray,
        log_volumes_ask: np.ndarray,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot QQ plots against normal distribution (preliminary check).
        
        Parameters
        ----------
        log_volumes_bid : np.ndarray
            Log-volumes for bid side
        log_volumes_ask : np.ndarray
            Log-volumes for ask side
        figsize : Tuple[int, int]
            Figure size
        save_path : Optional[str]
            Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Bid QQ plot
        stats.probplot(log_volumes_bid, dist="norm", plot=axes[0])
        axes[0].set_title('Q-Q Plot vs Normal Distribution (BID)', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Ask QQ plot
        stats.probplot(log_volumes_ask, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot vs Normal Distribution (ASK)',
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def get_comparison_stats(
        log_volumes_bid: np.ndarray,
        log_volumes_ask: np.ndarray
    ) -> Dict[str, dict]:
        """
        Compute comparison statistics (skewness, kurtosis, etc.).
        
        Parameters
        ----------
        log_volumes_bid : np.ndarray
            Log-volumes for bid side
        log_volumes_ask : np.ndarray
            Log-volumes for ask side
            
        Returns
        -------
        Dict[str, dict]
            Dictionary with skewness, kurtosis, and other statistics
        """
        stats_dict = {
            'bid': {
                'n': len(log_volumes_bid),
                'mean': float(np.mean(log_volumes_bid)),
                'std': float(np.std(log_volumes_bid)),
                'skewness': float(stats.skew(log_volumes_bid)),
                'kurtosis': float(stats.kurtosis(log_volumes_bid)),
                'min': float(np.min(log_volumes_bid)),
                'max': float(np.max(log_volumes_bid)),
                'q1': float(np.percentile(log_volumes_bid, 25)),
                'median': float(np.percentile(log_volumes_bid, 50)),
                'q3': float(np.percentile(log_volumes_bid, 75)),
            },
            'ask': {
                'n': len(log_volumes_ask),
                'mean': float(np.mean(log_volumes_ask)),
                'std': float(np.std(log_volumes_ask)),
                'skewness': float(stats.skew(log_volumes_ask)),
                'kurtosis': float(stats.kurtosis(log_volumes_ask)),
                'min': float(np.min(log_volumes_ask)),
                'max': float(np.max(log_volumes_ask)),
                'q1': float(np.percentile(log_volumes_ask, 25)),
                'median': float(np.percentile(log_volumes_ask, 50)),
                'q3': float(np.percentile(log_volumes_ask, 75)),
            }
        }
        return stats_dict


def main():
    """Example usage of visualization functions."""
    # This would be used after constructing samples with LOBSampleConstructor
    print("Visualization module loaded.")
    print("Usage:")
    print("  from visualization import LOBVisualization")
    print("  LOBVisualization.plot_empirical_histograms(log_volumes_bid, log_volumes_ask)")


if __name__ == "__main__":
    main()
