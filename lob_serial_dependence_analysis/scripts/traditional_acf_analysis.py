"""
Traditional Autocorrelation Analysis (Method 1)
Focus on LOB queue size persistence and order book depth characteristics

This module analyzes:
1. ACF of log-volumes (how long does liquidity persist?)
2. Order splitting detection (small lag strong correlation)
3. Ljung-Box statistical tests
4. LOB microstructure interpretation

Key concept: ACF measures whether large order book depths tend to persist
- Small lags: Order splitting (fragmented orders maintain depth)
- Medium lags: Liquidity persistence (supply/demand conditions stable)
- Large lags: Mean reversion (market returns to equilibrium)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from typing import Tuple, Dict, NamedTuple, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import acf, ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available, using manual ACF computation")


class ACFAnalysisResult(NamedTuple):
    """Container for ACF analysis results."""
    side: str
    acf_values: np.ndarray  # ACF coefficients
    lags: np.ndarray  # Lag indices
    confidence_95: float  # 95% CI bound
    significant_lags: list  # Lags exceeding CI
    ljung_box_stats: np.ndarray  # Ljung-Box test statistics
    ljung_box_pvalues: np.ndarray  # Ljung-Box p-values
    half_life: Optional[float]  # Estimated half-life (if ACF decays)
    n_samples: int
    mean: float
    std: float


class TraditionalACFAnalyzer:
    """
    Traditional autocorrelation analysis for LOB log-volumes.
    
    Focus: Queue size persistence in the limit order book
    """
    
    def __init__(self, log_volumes_bid: np.ndarray, 
                 log_volumes_ask: np.ndarray):
        """Initialize analyzer."""
        self.log_volumes_bid = np.asarray(log_volumes_bid, dtype=float)
        self.log_volumes_ask = np.asarray(log_volumes_ask, dtype=float)
        
        self.result_bid: Optional[ACFAnalysisResult] = None
        self.result_ask: Optional[ACFAnalysisResult] = None
    
    def compute_acf(self, nlags: int = 100, method: str = 'fft') -> Tuple[ACFAnalysisResult, ACFAnalysisResult]:
        """
        Compute autocorrelation function for log-volumes.
        
        Parameters
        ----------
        nlags : int
            Maximum number of lags to compute (default: 100)
        method : str
            'fft' (fast, Fourier) or 'direct' (standard, slower)
            
        Returns
        -------
        Tuple[ACFAnalysisResult, ACFAnalysisResult]
            ACF results for bid and ask sides
        """
        print("="*70)
        print("TRADITIONAL AUTOCORRELATION ANALYSIS")
        print("(Queue Size Persistence in LOB)")
        print("="*70)
        
        self.result_bid = self._compute_acf_single(
            self.log_volumes_bid, 'bid', nlags, method
        )
        self.result_ask = self._compute_acf_single(
            self.log_volumes_ask, 'ask', nlags, method
        )
        
        return self.result_bid, self.result_ask
    
    def _compute_acf_single(self, x: np.ndarray, side: str, nlags: int, 
                           method: str) -> ACFAnalysisResult:
        """Compute ACF for a single series."""
        n = len(x)
        
        # Compute ACF
        if STATSMODELS_AVAILABLE:
            acf_vals = acf(x, nlags=nlags, fft=(method=='fft'))
        else:
            acf_vals = self._compute_acf_manual(x, nlags)
        
        lags = np.arange(len(acf_vals))
        
        # 95% confidence interval
        ci_95 = 1.96 / np.sqrt(n)
        
        # Significant lags
        significant_lags = [k for k in range(1, len(acf_vals)) 
                           if abs(acf_vals[k]) > ci_95]
        
        # Ljung-Box test
        lb_stats, lb_pvalues = self._ljung_box_test(x, nlags)
        
        # Estimate half-life
        half_life = self._estimate_half_life(acf_vals)
        
        # Print summary
        print(f"\n{side.upper()} SIDE:")
        print(f"  Sample size: {n:,}")
        print(f"  Mean log-volume: {np.mean(x):.4f}")
        print(f"  Std deviation: {np.std(x):.4f}")
        print(f"\n  ACF Analysis:")
        print(f"    Lag-0 (ACF_0): {acf_vals[0]:.6f} (always = 1)")
        print(f"    Lag-1 (ρ₁): {acf_vals[1]:.6f} ← Order splitting signal")
        print(f"    Lag-5 (ρ₅): {acf_vals[5]:.6f}")
        print(f"    Lag-10 (ρ₁₀): {acf_vals[10]:.6f}")
        print(f"    Lag-20 (ρ₂₀): {acf_vals[20]:.6f}")
        
        print(f"\n  Statistical Significance:")
        print(f"    95% CI bounds: ±{ci_95:.4f}")
        print(f"    Significant lags: {len(significant_lags)} / {nlags}")
        if significant_lags:
            print(f"    First 10: {significant_lags[:10]}")
        
        print(f"\n  Ljung-Box Test (Serial Correlation):")
        print(f"    Lag 5:  χ² = {lb_stats[4]:.4f}, p-value = {lb_pvalues[4]:.6f}")
        print(f"    Lag 10: χ² = {lb_stats[9]:.4f}, p-value = {lb_pvalues[9]:.6f}")
        print(f"    Lag 20: χ² = {lb_stats[19]:.4f}, p-value = {lb_pvalues[19]:.6f}")
        print(f"    Interpretation: p < 0.05 → Significant serial dependence")
        
        if half_life is not None:
            print(f"\n  Half-Life Estimate:")
            print(f"    τ₁/₂ ≈ {half_life:.2f} lags")
            print(f"    (Time for ACF to decay to 50% of initial value)")
        
        print(f"\n  LOB Interpretation:")
        print(f"    ρ₁ = {acf_vals[1]:.4f} indicates:")
        if acf_vals[1] > 0.8:
            print(f"      → STRONG order splitting (fragmented orders)")
            print(f"      → Liquidity providers maintain multi-leg orders")
            print(f"      → Depth is sticky in short term (< 1 second)")
        elif acf_vals[1] > 0.5:
            print(f"      → MODERATE order splitting")
            print(f"      → Some order persistence but with high turnover")
        else:
            print(f"      → WEAK order splitting")
            print(f"      → Depth highly volatile, low persistence")
        
        return ACFAnalysisResult(
            side=side,
            acf_values=acf_vals,
            lags=lags,
            confidence_95=ci_95,
            significant_lags=significant_lags,
            ljung_box_stats=lb_stats,
            ljung_box_pvalues=lb_pvalues,
            half_life=half_life,
            n_samples=n,
            mean=float(np.mean(x)),
            std=float(np.std(x))
        )
    
    def _compute_acf_manual(self, x: np.ndarray, nlags: int) -> np.ndarray:
        """Manual ACF computation if statsmodels unavailable."""
        n = len(x)
        x_centered = x - np.mean(x)
        c0 = np.dot(x_centered, x_centered) / n
        
        acf_vals = np.ones(nlags + 1)
        for k in range(1, nlags + 1):
            c_k = np.dot(x_centered[:-k], x_centered[k:]) / n
            acf_vals[k] = c_k / c0
        
        return acf_vals
    
    def _ljung_box_test(self, x: np.ndarray, nlags: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ljung-Box test for serial correlation.
        
        H₀: No serial correlation in first nlags
        H₁: At least one lag has correlation ≠ 0
        
        Test statistic: Q = n(n+2) Σ_{k=1}^h [ρ²_k / (n-k)]
        Under H₀: Q ~ χ²(h)
        """
        if STATSMODELS_AVAILABLE:
            try:
                lb_result = ljungbox(x, lags=list(range(1, nlags+1)), return_df=False)
                return lb_result[0], lb_result[1]
            except:
                pass
        
        # Manual Ljung-Box implementation
        n = len(x)
        acf_vals = self._compute_acf_manual(x, nlags)
        
        lb_stats = np.zeros(nlags)
        lb_pvalues = np.zeros(nlags)
        
        for h in range(1, nlags + 1):
            # Ljung-Box statistic
            rho_squared_sum = np.sum(acf_vals[1:h+1]**2 / (n - np.arange(1, h+1)))
            Q = n * (n + 2) * rho_squared_sum
            
            lb_stats[h-1] = Q
            # P-value from chi-squared distribution with h degrees of freedom
            lb_pvalues[h-1] = 1 - chi2.cdf(Q, h)
        
        return lb_stats, lb_pvalues
    
    def _estimate_half_life(self, acf_vals: np.ndarray, min_lag: int = 2) -> Optional[float]:
        """
        Estimate half-life of ACF decay.
        
        Assumes exponential decay: ACF(k) ≈ ρ^k
        Then half-life τ₁/₂ = ln(0.5) / ln(ρ)
        """
        # Use early lags (where exponential assumption is best)
        if len(acf_vals) < min_lag + 1:
            return None
        
        # Fit exponential to first 5-10 lags
        valid_acf = acf_vals[min_lag:min(15, len(acf_vals))]
        
        if np.any(valid_acf <= 0):
            return None
        
        # Estimate ρ from lag-1 decay
        rho = acf_vals[1] if acf_vals[1] > 0 else acf_vals[2]
        
        if rho <= 0 or rho >= 1:
            return None
        
        # Half-life
        half_life = np.log(0.5) / np.log(rho)
        
        return half_life if half_life > 0 else None
    
    def plot_acf_detailed(self, save_path: str = "05_traditional_acf.png") -> None:
        """
        Create detailed ACF plot with interpretation zones.
        """
        if self.result_bid is None or self.result_ask is None:
            raise ValueError("Must run compute_acf() first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Left column: BID, Right column: ASK
        for col, result in enumerate([self.result_bid, self.result_ask]):
            
            # Full ACF (top)
            ax = axes[0, col]
            lags = result.lags[:min(50, len(result.lags))]
            acf_vals = result.acf_values[:min(50, len(result.acf_values))]
            ci = result.confidence_95
            
            # Plot ACF
            ax.stem(lags, acf_vals, basefmt=' ', linefmt='C0-', markerfmt='C0o')
            
            # Confidence interval
            ax.axhline(y=ci, color='red', linestyle='--', linewidth=1.5, label=f'95% CI (±{ci:.4f})')
            ax.axhline(y=-ci, color='red', linestyle='--', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Shade zones
            ax.axvspan(0.5, 5, alpha=0.1, color='yellow', label='Order splitting zone')
            ax.axvspan(5, 20, alpha=0.1, color='green', label='Persistence zone')
            ax.axvspan(20, 50, alpha=0.1, color='blue', label='Mean reversion zone')
            
            ax.set_xlabel('Lag (seconds)', fontsize=11)
            ax.set_ylabel('ACF ρ(k)', fontsize=11)
            ax.set_title(f'{result.side.upper()} - Autocorrelation Function\n(Queue Size Persistence)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_ylim([-0.2, 1.05])
            
            # Zoomed in (0-20 lags, bottom)
            ax = axes[1, col]
            lags_zoom = result.lags[:21]
            acf_zoom = result.acf_values[:21]
            
            ax.bar(lags_zoom, acf_zoom, width=0.6, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axhline(y=ci, color='red', linestyle='--', linewidth=1.5, label=f'95% CI')
            ax.axhline(y=-ci, color='red', linestyle='--', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Annotate key lags
            for lag in [1, 5, 10]:
                if lag < len(result.acf_values):
                    ax.annotate(f'{result.acf_values[lag]:.3f}',
                              xy=(lag, result.acf_values[lag]),
                              xytext=(0, 10), textcoords='offset points',
                              ha='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Lag (seconds)', fontsize=11)
            ax.set_ylabel('ACF ρ(k)', fontsize=11)
            ax.set_title(f'{result.side.upper()} - Short-Term Persistence (0-20s)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([-0.2, 1.05])
            ax.set_xlim([-0.5, 20.5])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ ACF plot saved to {save_path}")
        plt.close()
    
    def plot_ljung_box(self, save_path: str = "06_ljung_box_test.png") -> None:
        """
        Plot Ljung-Box test results.
        """
        if self.result_bid is None or self.result_ask is None:
            raise ValueError("Must run compute_acf() first")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for col, result in enumerate([self.result_bid, self.result_ask]):
            ax = axes[col]
            
            # Plot Ljung-Box statistics
            lags = np.arange(1, min(51, len(result.ljung_box_stats) + 1))
            stats = result.ljung_box_stats[:len(lags)]
            
            ax.bar(lags, stats, width=0.6, alpha=0.7, color='coral', edgecolor='black')
            
            # Critical value at α=0.05 (varies with lag)
            critical_95 = chi2.ppf(0.95, lags)
            ax.plot(lags, critical_95, 'r--', linewidth=2, label='Critical value (α=0.05)')
            
            ax.set_xlabel('Lag', fontsize=11)
            ax.set_ylabel('Ljung-Box Q Statistic', fontsize=11)
            ax.set_title(f'{result.side.upper()} - Ljung-Box Test\n(H₀: No serial correlation)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=10)
            
            # Add text with rejection count
            rejections = np.sum(result.ljung_box_pvalues[:len(lags)] < 0.05)
            ax.text(0.98, 0.97, f'Rejections at α=0.05: {rejections}/{len(lags)}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ljung-Box plot saved to {save_path}")
        plt.close()
    
    def print_lob_interpretation(self) -> None:
        """
        Print detailed LOB microstructure interpretation.
        """
        if self.result_bid is None or self.result_ask is None:
            return
        
        print("\n" + "="*70)
        print("LOB MICROSTRUCTURE INTERPRETATION")
        print("="*70)
        
        for result in [self.result_bid, self.result_ask]:
            print(f"\n{result.side.upper()} SIDE:")
            print("-" * 70)
            
            rho_1 = result.acf_values[1]
            rho_5 = result.acf_values[5]
            rho_10 = result.acf_values[10]
            
            # Interpretation based on ACF profile
            print("\n1. ORDER SPLITTING (Lag 1-5):")
            print(f"   ρ₁ = {rho_1:.4f}")
            
            if rho_1 > 0.85:
                print("   → EXTREME order splitting observed")
                print("     - Liquidity providers break large orders into fragments")
                print("     - Each fragment executed at slightly different times")
                print("     - Suggests potential anti-gaming strategies by market makers")
            elif rho_1 > 0.70:
                print("   → STRONG order splitting")
                print("     - Orders are fragmented to avoid market impact")
                print("     - Typical for large institutional orders")
            elif rho_1 > 0.50:
                print("   → MODERATE order splitting")
                print("     - Some fragmentation, but not systematic")
            else:
                print("   → WEAK order splitting")
                print("     - Low order fragmentation")
            
            print(f"\n2. LIQUIDITY PERSISTENCE (Lag 5-20):")
            print(f"   ρ₅ = {rho_5:.4f}, ρ₁₀ = {rho_10:.4f}")
            
            persistence_decay = (rho_1 - rho_10) / rho_1 * 100
            print(f"   Decay from lag-1 to lag-10: {persistence_decay:.1f}%")
            
            if rho_10 > 0.6:
                print("   → HIGH liquidity persistence")
                print("     - Order book depth remains stable over 10+ seconds")
                print("     - Market is in stable regime")
            elif rho_10 > 0.3:
                print("   → MODERATE liquidity persistence")
                print("     - Depth gradually decays (ARMA-like behavior)")
                print("     - Supply/demand conditions evolving")
            else:
                print("   → LOW liquidity persistence")
                print("     - Rapid mean reversion of depth")
                print("     - Market actively adjusts to shocks")
            
            print(f"\n3. MEAN REVERSION HORIZON:")
            if result.half_life is not None:
                print(f"   Half-life τ₁/₂ ≈ {result.half_life:.2f} seconds")
                if result.half_life < 5:
                    print("   → FAST mean reversion (high-frequency trading dominant)")
                elif result.half_life < 20:
                    print("   → MODERATE mean reversion (mixed trading horizons)")
                else:
                    print("   → SLOW mean reversion (longer-term position building)")
            
            print(f"\n4. STATISTICAL SIGNIFICANCE:")
            lb_p5 = result.ljung_box_pvalues[4]
            lb_p10 = result.ljung_box_pvalues[9]
            
            print(f"   Ljung-Box (lag 5): p-value = {lb_p5:.6f}")
            print(f"   Ljung-Box (lag 10): p-value = {lb_p10:.6f}")
            
            if lb_p10 < 0.001:
                print("   → EXTREMELY STRONG evidence of serial dependence")
                print("     (p << 0.001, reject H₀ at all conventional levels)")
            elif lb_p10 < 0.05:
                print("   → STRONG evidence of serial dependence")
                print("     (p < 0.05, reject H₀ at α=0.05)")
            else:
                print("   → Weak evidence of serial dependence")
                print("     (p ≥ 0.05, cannot reject H₀)")
        
        print("\n" + "="*70)


def main():
    """Example usage."""
    pass


if __name__ == "__main__":
    main()
