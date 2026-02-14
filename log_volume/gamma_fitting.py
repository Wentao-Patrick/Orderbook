"""
Step 3-5: Gamma Distribution Parameter Estimation and Fitting Visualization

This module handles:
1. Gamma parameter estimation (MLE) for log-volumes
2. Automatic detection of negative log-volumes and adaptive fitting strategy
3. Overlay of theoretical Gamma PDF on empirical histograms
4. Goodness-of-fit statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, kstest, anderson
from typing import Tuple, Dict, Optional, NamedTuple
import warnings

warnings.filterwarnings('ignore')


class GammaFitResult(NamedTuple):
    """Container for Gamma fitting results."""
    side: str  # 'bid' or 'ask'
    fit_type: str  # 'logvolume' or 'volume'
    k: float  # shape parameter
    theta: float  # scale parameter
    loc: float  # location parameter (usually 0)
    n_samples: int  # sample size
    n_positive: int  # number of positive samples (for logvolume)
    pct_negative: float  # percentage of negative values
    mean: float
    std: float
    ks_statistic: float  # Kolmogorov-Smirnov test statistic
    ks_pvalue: float


class GammaDistributionFitter:
    """Handles Gamma distribution fitting with adaptive strategies."""
    
    def __init__(self, 
                 log_volumes_bid: np.ndarray,
                 log_volumes_ask: np.ndarray,
                 negative_threshold: float = 0.1):
        """
        Initialize Gamma fitter.
        
        Parameters
        ----------
        log_volumes_bid : np.ndarray
            Log-volumes for bid side
        log_volumes_ask : np.ndarray
            Log-volumes for ask side
        negative_threshold : float
            If percentage of negative values exceeds this threshold (0-1),
            switch from fitting log-volumes to fitting raw volumes (default: 0.1 = 10%)
        """
        self.log_volumes_bid = log_volumes_bid
        self.log_volumes_ask = log_volumes_ask
        self.negative_threshold = negative_threshold
        
        self.fit_results_bid: Optional[GammaFitResult] = None
        self.fit_results_ask: Optional[GammaFitResult] = None
        
        # Store fitting data (after filtering/transformation)
        self.data_bid: np.ndarray = None
        self.data_ask: np.ndarray = None
        self.fit_type_bid: str = None
        self.fit_type_ask: str = None
    
    def _analyze_negative_values(self) -> Dict[str, dict]:
        """
        Analyze the presence and extent of negative values.
        
        Returns
        -------
        Dict[str, dict]
            Analysis for bid and ask sides
        """
        analysis = {}
        
        for side, data in [('bid', self.log_volumes_bid), ('ask', self.log_volumes_ask)]:
            n_total = len(data)
            if n_total == 0:
                analysis[side] = {
                    'n_total': 0,
                    'n_negative': 0,
                    'n_positive': 0,
                    'n_zero': 0,
                    'pct_negative': 0.0,
                    'min': np.nan,
                    'max': np.nan,
                    'mean': np.nan,
                }
            else:
                n_negative = np.sum(data < 0)
                n_positive = np.sum(data > 0)
                pct_negative = n_negative / n_total if n_total > 0 else 0
                
                analysis[side] = {
                    'n_total': n_total,
                    'n_negative': n_negative,
                    'n_positive': n_positive,
                    'n_zero': n_total - n_negative - n_positive,
                    'pct_negative': pct_negative,
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                }
        
        return analysis
    
    def _decide_fit_strategy(self, side: str) -> Tuple[str, np.ndarray]:
        """
        Decide whether to fit log-volumes or raw volumes based on data characteristics.
        
        Parameters
        ----------
        side : str
            'bid' or 'ask'
            
        Returns
        -------
        Tuple[str, np.ndarray]
            (fit_type, data_to_fit) where fit_type is 'logvolume' or 'volume'
        """
        log_data = self.log_volumes_bid if side == 'bid' else self.log_volumes_ask
        
        # Count negative values
        n_negative = np.sum(log_data < 0)
        n_total = len(log_data)
        pct_negative = n_negative / n_total if n_total > 0 else 0
        
        print(f"\n  {side.upper()} side:")
        print(f"    Total samples: {n_total:,}")
        print(f"    Negative log-volumes: {n_negative:,} ({pct_negative*100:.2f}%)")
        
        if pct_negative > self.negative_threshold:
            # Too many negative values - fit raw volumes instead
            print(f"    → Detected {pct_negative*100:.2f}% negative log-volumes (threshold: {self.negative_threshold*100:.1f}%)")
            print(f"    → Switching to VOLUME fitting (instead of log-volume)")
            
            # Reconstruct raw volumes from log-volumes
            volumes = np.exp(log_data)
            volumes = volumes[volumes > 0]  # Remove any numerical issues
            
            return 'volume', volumes
        else:
            # Few negative values - can safely fit log-volumes
            print(f"    → {pct_negative*100:.2f}% negative values is acceptable")
            print(f"    → Fitting LOG-VOLUME (only positive values)")
            
            # Use only positive log-volumes
            positive_log = log_data[log_data > 0]
            
            return 'logvolume', positive_log
    
    def fit(self) -> Tuple[GammaFitResult, GammaFitResult]:
        """
        Perform Gamma parameter estimation for both sides.
        
        Returns
        -------
        Tuple[GammaFitResult, GammaFitResult]
            Fitting results for bid and ask sides
        """
        print("="*70)
        print("GAMMA DISTRIBUTION PARAMETER ESTIMATION (MLE)")
        print("="*70)
        
        # Check if we have any data
        if len(self.log_volumes_bid) == 0 and len(self.log_volumes_ask) == 0:
            print("\n⚠ WARNING: No log-volume samples collected!")
            print("The orderbook may be empty or no orders match the criteria.")
            raise ValueError("No valid samples to fit")
        
        # Analyze negative values
        print("\n[1] Analyzing negative values in log-volumes...")
        analysis = self._analyze_negative_values()
        
        for side in ['bid', 'ask']:
            info = analysis[side]
            print(f"\n  {side.upper()}:")
            if info['n_total'] == 0:
                print(f"    No samples collected")
            else:
                print(f"    Min: {info['min']:.4f}, Max: {info['max']:.4f}")
                print(f"    Mean: {info['mean']:.4f}")
                print(f"    Negative count: {info['n_negative']:,} / {info['n_total']:,}")
        
        # Decide strategies and prepare data
        print("\n[2] Deciding fitting strategy...")
        
        fit_type_bid, data_bid = self._decide_fit_strategy('bid')
        fit_type_ask, data_ask = self._decide_fit_strategy('ask')
        
        self.fit_type_bid = fit_type_bid
        self.fit_type_ask = fit_type_ask
        
        # Check if we still have data after filtering
        if len(data_bid) == 0 or len(data_ask) == 0:
            raise ValueError("No positive samples after filtering negative values")
        
        # Remove extreme outliers (highest frequency values)
        print("\n[2.5] Removing extreme outliers (highest frequency values)...")
        
        n_outliers = 8
        
        def remove_highest_frequency_values(data: np.ndarray, n_remove: int) -> np.ndarray:
            """Remove the n_remove most frequent values from the data."""
            unique_vals, counts = np.unique(data, return_counts=True)
            
            # Sort by frequency (descending)
            sorted_idx = np.argsort(-counts)
            
            # Get the top n_remove most frequent values
            top_n_values = unique_vals[sorted_idx[:n_remove]]
            
            # Create mask: keep only values NOT in top_n_values
            mask = ~np.isin(data, top_n_values)
            
            return data[mask], len(data) - np.sum(mask)
        
        # BID side
        if len(data_bid) > n_outliers:
            data_bid, n_removed_bid = remove_highest_frequency_values(data_bid, n_outliers)
            print(f"  BID: Removed {n_removed_bid} samples (highest frequency values). Remaining: {len(data_bid)}")
        
        # ASK side
        if len(data_ask) > n_outliers:
            data_ask, n_removed_ask = remove_highest_frequency_values(data_ask, n_outliers)
            print(f"  ASK: Removed {n_removed_ask} samples (highest frequency values). Remaining: {len(data_ask)}")
        
        self.data_bid = data_bid
        self.data_ask = data_ask
        
        # Perform MLE estimation
        print("\n[3] Estimating Gamma parameters (MLE)...")
        
        # BID side
        print(f"\n  BID side ({fit_type_bid}):")
        try:
            k_b, loc_b, theta_b = gamma.fit(data_bid, floc=0)
            print(f"    k (shape): {k_b:.6f}")
            print(f"    θ (scale): {theta_b:.6f}")
            print(f"    loc (location): {loc_b:.6f}")
            
            # Compute goodness-of-fit statistics for BID
            ks_stat_b, ks_pval_b = kstest(data_bid, lambda x: gamma.cdf(x, k_b, loc=0, scale=theta_b))
        except Exception as e:
            print(f"    Error fitting BID: {e}")
            k_b, loc_b, theta_b = np.nan, np.nan, np.nan
            ks_stat_b, ks_pval_b = np.nan, np.nan
        
        self.fit_results_bid = GammaFitResult(
            side='bid',
            fit_type=fit_type_bid,
            k=k_b,
            theta=theta_b,
            loc=loc_b,
            n_samples=len(self.log_volumes_bid),
            n_positive=len(data_bid),
            pct_negative=analysis['bid']['pct_negative'],
            mean=float(np.mean(data_bid)) if len(data_bid) > 0 else np.nan,
            std=float(np.std(data_bid)) if len(data_bid) > 0 else np.nan,
            ks_statistic=ks_stat_b,
            ks_pvalue=ks_pval_b,
        )
        
        # ASK side
        print(f"\n  ASK side ({fit_type_ask}):")
        try:
            k_a, loc_a, theta_a = gamma.fit(data_ask, floc=0)
            print(f"    k (shape): {k_a:.6f}")
            print(f"    θ (scale): {theta_a:.6f}")
            print(f"    loc (location): {loc_a:.6f}")
            
            # Compute goodness-of-fit statistics for ASK
            ks_stat_a, ks_pval_a = kstest(data_ask, lambda x: gamma.cdf(x, k_a, loc=0, scale=theta_a))
        except Exception as e:
            print(f"    Error fitting ASK: {e}")
            k_a, loc_a, theta_a = np.nan, np.nan, np.nan
            ks_stat_a, ks_pval_a = np.nan, np.nan
        
        self.fit_results_ask = GammaFitResult(
            side='ask',
            fit_type=fit_type_ask,
            k=k_a,
            theta=theta_a,
            loc=loc_a,
            n_samples=len(self.log_volumes_ask),
            n_positive=len(data_ask),
            pct_negative=analysis['ask']['pct_negative'],
            mean=float(np.mean(data_ask)) if len(data_ask) > 0 else np.nan,
            std=float(np.std(data_ask)) if len(data_ask) > 0 else np.nan,
            ks_statistic=ks_stat_a,
            ks_pvalue=ks_pval_a,
        )
        
        print("\n[4] Goodness-of-fit (Kolmogorov-Smirnov test):")
        print(f"\n  BID: KS statistic = {ks_stat_b:.6f}, p-value = {ks_pval_b:.6f}")
        print(f"  ASK: KS statistic = {ks_stat_a:.6f}, p-value = {ks_pval_a:.6f}")
        
        print("\n" + "="*70)
        
        return self.fit_results_bid, self.fit_results_ask
    
    def plot_fitted_histograms(self,
                               bins: int = 100,
                               figsize: Tuple[int, int] = (14, 5),
                               save_path: Optional[str] = None) -> None:
        """
        Plot empirical histograms with overlaid Gamma PDF curves.
        
        Parameters
        ----------
        bins : int
            Number of histogram bins
        figsize : Tuple[int, int]
            Figure size
        save_path : Optional[str]
            Path to save figure
        """
        if self.fit_results_bid is None or self.fit_results_ask is None:
            raise ValueError("Must call fit() before plotting")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # BID side
        ax = axes[0]
        ax.hist(self.data_bid, bins=bins, density=True, alpha=0.6,
               color='blue', edgecolor='black', label='Empirical')
        
        # Plot Gamma PDF
        x_bid = np.linspace(self.data_bid.min(), self.data_bid.max(), 500)
        pdf_bid = gamma.pdf(x_bid, self.fit_results_bid.k, 
                           loc=self.fit_results_bid.loc,
                           scale=self.fit_results_bid.theta)
        ax.plot(x_bid, pdf_bid, 'r-', linewidth=2.5, label='Gamma PDF')
        
        # Labels and title
        xlabel = 'log(Volume)' if self.fit_type_bid == 'logvolume' else 'Volume'
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Sanofi LOB: BID ({self.fit_type_bid})\n' +
                    f'k={self.fit_results_bid.k:.4f}, θ={self.fit_results_bid.theta:.4f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        # Add stats box
        stats_text = f'n = {self.fit_results_bid.n_positive:,}\n' \
                    f'KS p-val = {self.fit_results_bid.ks_pvalue:.4f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # ASK side
        ax = axes[1]
        ax.hist(self.data_ask, bins=bins, density=True, alpha=0.6,
               color='red', edgecolor='black', label='Empirical')
        
        # Plot Gamma PDF
        x_ask = np.linspace(self.data_ask.min(), self.data_ask.max(), 500)
        pdf_ask = gamma.pdf(x_ask, self.fit_results_ask.k,
                           loc=self.fit_results_ask.loc,
                           scale=self.fit_results_ask.theta)
        ax.plot(x_ask, pdf_ask, 'darkred', linewidth=2.5, label='Gamma PDF')
        
        # Labels and title
        xlabel = 'log(Volume)' if self.fit_type_ask == 'logvolume' else 'Volume'
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Sanofi LOB: ASK ({self.fit_type_ask})\n' +
                    f'k={self.fit_results_ask.k:.4f}, θ={self.fit_results_ask.theta:.4f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        # Add stats box
        stats_text = f'n = {self.fit_results_ask.n_positive:,}\n' \
                    f'KS p-val = {self.fit_results_ask.ks_pvalue:.4f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to {save_path}")
        else:
            plt.show()
    
    def plot_separate_fitted_histograms(self,
                                        bins: int = 100,
                                        figsize: Tuple[int, int] = (12, 6),
                                        save_dir: Optional[str] = None) -> None:
        """
        Plot separate full-page histograms with Gamma PDF overlay.
        
        Parameters
        ----------
        bins : int
            Number of histogram bins
        figsize : Tuple[int, int]
            Figure size
        save_dir : Optional[str]
            Directory to save figures
        """
        if self.fit_results_bid is None or self.fit_results_ask is None:
            raise ValueError("Must call fit() before plotting")
        
        # BID side
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self.data_bid, bins=bins, density=True, alpha=0.6,
               color='blue', edgecolor='black', linewidth=1.2, label='Empirical Histogram')
        
        x_bid = np.linspace(self.data_bid.min(), self.data_bid.max(), 500)
        pdf_bid = gamma.pdf(x_bid, self.fit_results_bid.k,
                           loc=self.fit_results_bid.loc,
                           scale=self.fit_results_bid.theta)
        ax.plot(x_bid, pdf_bid, 'r-', linewidth=3, label='Gamma PDF (MLE fit)')
        
        xlabel = 'log(Volume)' if self.fit_type_bid == 'logvolume' else 'Volume'
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel('Probability Density', fontsize=13)
        ax.set_title(f'Sanofi LOB: BID Side ({self.fit_type_bid})\n' +
                    f'Gamma Distribution Fit (k={self.fit_results_bid.k:.4f}, θ={self.fit_results_bid.theta:.4f})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best')
        
        stats_text = f'Sample size: {self.fit_results_bid.n_positive:,}\n' \
                    f'KS statistic: {self.fit_results_bid.ks_statistic:.6f}\n' \
                    f'KS p-value: {self.fit_results_bid.ks_pvalue:.6f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/gamma_fit_bid.png', dpi=300, bbox_inches='tight')
            print(f"BID histogram saved to {save_dir}/gamma_fit_bid.png")
        else:
            plt.show()
        
        # ASK side
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self.data_ask, bins=bins, density=True, alpha=0.6,
               color='red', edgecolor='black', linewidth=1.2, label='Empirical Histogram')
        
        x_ask = np.linspace(self.data_ask.min(), self.data_ask.max(), 500)
        pdf_ask = gamma.pdf(x_ask, self.fit_results_ask.k,
                           loc=self.fit_results_ask.loc,
                           scale=self.fit_results_ask.theta)
        ax.plot(x_ask, pdf_ask, 'darkred', linewidth=3, label='Gamma PDF (MLE fit)')
        
        xlabel = 'log(Volume)' if self.fit_type_ask == 'logvolume' else 'Volume'
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel('Probability Density', fontsize=13)
        ax.set_title(f'Sanofi LOB: ASK Side ({self.fit_type_ask})\n' +
                    f'Gamma Distribution Fit (k={self.fit_results_ask.k:.4f}, θ={self.fit_results_ask.theta:.4f})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best')
        
        stats_text = f'Sample size: {self.fit_results_ask.n_positive:,}\n' \
                    f'KS statistic: {self.fit_results_ask.ks_statistic:.6f}\n' \
                    f'KS p-value: {self.fit_results_ask.ks_pvalue:.6f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/gamma_fit_ask.png', dpi=300, bbox_inches='tight')
            print(f"ASK histogram saved to {save_dir}/gamma_fit_ask.png")
        else:
            plt.show()
    
    def get_summary_report(self) -> str:
        """
        Generate a formatted summary report of fitting results.
        
        Returns
        -------
        str
            Formatted report
        """
        if self.fit_results_bid is None or self.fit_results_ask is None:
            return "No fitting results available. Call fit() first."
        
        report = []
        report.append("\n" + "="*70)
        report.append("GAMMA DISTRIBUTION FITTING SUMMARY")
        report.append("="*70)
        
        for result in [self.fit_results_bid, self.fit_results_ask]:
            side = result.side.upper()
            report.append(f"\n{side} SIDE ({result.fit_type}):")
            report.append(f"  Original samples: {result.n_samples:,}")
            report.append(f"  Samples used in fitting: {result.n_positive:,}")
            
            if result.fit_type == 'logvolume':
                report.append(f"  Negative log-volumes: {result.pct_negative*100:.2f}%")
            
            report.append(f"\n  Gamma Parameters (MLE):")
            report.append(f"    k (shape):  {result.k:.6f}")
            report.append(f"    θ (scale):  {result.theta:.6f}")
            report.append(f"    loc:        {result.loc:.6f}")
            
            report.append(f"\n  Data Statistics:")
            report.append(f"    Mean: {result.mean:.6f}")
            report.append(f"    Std:  {result.std:.6f}")
            
            report.append(f"\n  Goodness-of-Fit (Kolmogorov-Smirnov):")
            report.append(f"    Statistic: {result.ks_statistic:.6f}")
            report.append(f"    p-value:   {result.ks_pvalue:.6f}")
            
            if result.ks_pvalue > 0.05:
                report.append(f"    ✓ Cannot reject null hypothesis (p > 0.05)")
            else:
                report.append(f"    ✗ Rejects null hypothesis (p < 0.05)")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


def main():
    """Example usage of GammaDistributionFitter."""
    print("Gamma Distribution Fitting module loaded.")
    print("Usage:")
    print("  from gamma_fitting import GammaDistributionFitter")
    print("  fitter = GammaDistributionFitter(log_volumes_bid, log_volumes_ask)")
    print("  fitter.fit()")
    print("  fitter.plot_fitted_histograms()")


if __name__ == "__main__":
    main()
