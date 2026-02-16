"""
Garcin-Brouty (2023) Information-Theoretic Dependence Analysis
==============================================================

Method: Symbolic representation + Shannon entropy for detecting
serial dependence in market microstructure

Paper: Garcin M., Brouty R. - A statistical test of behavior

Core Idea:
- Convert continuous log-volumes to binary symbolic sequence
- Measure whether past patterns predict future movements
- Compare entropy under market vs. independence hypothesis
- Information quantity I_L = H^EMH - H reveals predictability

References:
- Garcin & Brouty (2023): Statistical test for symbolic sequences
- Risso (2020): Symbolic representation in market microstructure
"""

import numpy as np
from typing import NamedTuple, Tuple, Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import comb


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class GarcinBroutyResult(NamedTuple):
    """Results from Garcin-Brouty analysis"""
    L_values: np.ndarray  # Lengths of past patterns (1, 2, ..., max_L)
    conditional_probs: Dict[int, np.ndarray]  # π_i^L for each pattern
    pattern_frequencies: Dict[int, np.ndarray]  # p_i^L (frequency of each pattern)
    entropy_market: np.ndarray  # H_{L+1} (real market entropy)
    entropy_emh: np.ndarray  # H_{L+1}^EMH (independence entropy)
    information: np.ndarray  # I_{L+1} = H^EMH - H (predictability)
    max_information: float  # max(I_{L+1}) - strongest dependence
    half_life_L: int  # L where I_L drops to 50% of max
    

@dataclass
class SymbolicSequence:
    """Binary symbolic representation of volume changes"""
    sequence: np.ndarray  # Binary array (0 or 1)
    n: int  # Length of sequence
    changes: np.ndarray  # Underlying differences
    

# ============================================================================
# STEP 1: SYMBOLIC REPRESENTATION
# ============================================================================

class SymbolicRepresentation:
    """Convert continuous log-volume to binary symbolic sequence"""
    
    @staticmethod
    def create_binary_sequence(log_volumes: np.ndarray) -> SymbolicSequence:
        """
        Create binary sequence: Y_t = 1 if X_t - X_{t-1} > 0, else 0
        
        Args:
            log_volumes: Continuous log-volume series X_t
            
        Returns:
            SymbolicSequence with binary representation
        """
        # Compute differences
        changes = np.diff(log_volumes)
        
        # Create binary sequence: 1 if increase, 0 if decrease/no change
        binary = (changes > 0).astype(int)
        
        return SymbolicSequence(
            sequence=binary,
            n=len(binary),
            changes=changes
        )
    
    @staticmethod
    def get_pattern_at_lag(binary_seq: np.ndarray, t: int, L: int) -> Tuple[int, int]:
        """
        Get pattern (Y_t, ..., Y_{t+L-1}) at time t
        
        Pattern is encoded as integer: Y_t * 2^(L-1) + ... + Y_{t+L-1}
        
        Args:
            binary_seq: Binary sequence
            t: Starting time index
            L: Pattern length
            
        Returns:
            (pattern_id, next_bit) where:
            - pattern_id: integer encoding of pattern
            - next_bit: Y_{t+L} (target prediction)
        """
        if t + L >= len(binary_seq):
            return None, None
        
        # Extract pattern (Y_t, ..., Y_{t+L-1})
        pattern = binary_seq[t:t+L]
        
        # Encode as binary number: Y_t is MSB
        pattern_id = int(''.join(pattern.astype(str)), 2)
        
        # Next bit to predict
        next_bit = binary_seq[t + L]
        
        return pattern_id, next_bit


# ============================================================================
# STEP 2: COMPUTE CONDITIONAL PROBABILITIES
# ============================================================================

class ConditionalProbability:
    """Compute conditional probability of next bit given pattern"""
    
    @staticmethod
    def compute_for_length(binary_seq: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        For length L, compute:
        - π_i^L = P(Y_{t+L} = 1 | pattern_i)
        - p_i^L = frequency of pattern_i
        
        Args:
            binary_seq: Binary sequence
            L: Pattern length
            
        Returns:
            (conditional_probs, pattern_freqs) 
            conditional_probs[i] = π_i^L
            pattern_freqs[i] = p_i^L
        """
        n = len(binary_seq)
        n_patterns = 2 ** L
        
        # Arrays to store results
        conditional_probs = np.zeros(n_patterns)
        pattern_freqs = np.zeros(n_patterns)
        
        # Count pattern frequencies and conditional outcomes - vectorized
        pattern_count = np.zeros(n_patterns, dtype=np.int64)
        pattern_next_1 = np.zeros(n_patterns, dtype=np.int64)
        
        # Fast vectorized iteration using array slicing
        for t in range(n - L):
            # Extract pattern and next bit
            pattern_bits = binary_seq[t:t+L]
            pattern_id = int(''.join(pattern_bits.astype(str)), 2)
            next_bit = binary_seq[t + L]
            
            pattern_count[pattern_id] += 1
            if next_bit == 1:
                pattern_next_1[pattern_id] += 1
        
        # Compute frequencies and conditional probabilities - vectorized
        valid_patterns = pattern_count > 0
        pattern_freqs[valid_patterns] = pattern_count[valid_patterns] / (n - L)
        
        # Safe division for conditional probabilities
        with np.errstate(divide='ignore', invalid='ignore'):
            conditional_probs = np.where(
                pattern_count > 0,
                pattern_next_1 / pattern_count,
                0.5
            ).astype(float)
        
        return conditional_probs, pattern_freqs


# ============================================================================
# STEP 3: COMPUTE SHANNON ENTROPY
# ============================================================================

class ShannonEntropy:
    """Compute Shannon entropy for real market and independence hypothesis"""
    
    @staticmethod
    def entropy_with_conditional(conditional_probs: np.ndarray, 
                                 pattern_freqs: np.ndarray) -> float:
        """
        Compute H_{L+1} for real market:
        
        H_{L+1} = -Σ_i p_i^L [π_i^L log₂(p_i^L/π_i^L) + (1-π_i^L) log₂(p_i^L/(1-π_i^L))]
        
        This is the entropy of (Y_{t+L} | past pattern)
        
        Args:
            conditional_probs: π_i^L (probability Y_{t+L}=1 given pattern i)
            pattern_freqs: p_i^L (frequency of pattern i)
            
        Returns:
            Shannon entropy H_{L+1}
        """
        entropy = 0.0
        
        for i in range(len(conditional_probs)):
            pi = pattern_freqs[i]
            
            # Skip if pattern never occurs
            if pi < 1e-10:
                continue
            
            pi_L = conditional_probs[i]
            
            # Avoid log(0)
            if pi_L < 1e-10:
                pi_L = 1e-10
            if pi_L > 1 - 1e-10:
                pi_L = 1 - 1e-10
            
            # Contribution: p_i^L [π_i^L log₂(p_i^L/π_i^L) + (1-π_i^L) log₂(p_i^L/(1-π_i^L))]
            term1 = pi_L * np.log2(pi / pi_L)
            term2 = (1 - pi_L) * np.log2(pi / (1 - pi_L))
            
            entropy += pi * (term1 + term2)
        
        return -entropy  # Return negative for typical entropy definition
    
    @staticmethod
    def entropy_emh(entropy_prev: float) -> float:
        """
        Entropy under EMH (independence) hypothesis:
        
        H_{L+1}^EMH = 1 + H_L
        
        Since if Y_t are independent, Y_{t+L} is independent of pattern,
        so we add 1 bit of entropy (binary outcome)
        
        Args:
            entropy_prev: H_L for L-1 pattern
            
        Returns:
            H_{L+1}^EMH
        """
        return entropy_prev + 1.0


# ============================================================================
# STEP 4: COMPUTE MARKET INFORMATION
# ============================================================================

class MarketInformation:
    """Compute market information quantity"""
    
    @staticmethod
    def compute_information(entropy_emh: float, entropy_market: float) -> float:
        """
        Market Information Quantity:
        
        I_{L+1} = H_{L+1}^EMH - H_{L+1}
        
        Interpretation:
        - I = 0: No additional information (independent, EMH-like)
        - I > 0: Past patterns have predictive power
        - I >> 0: Strong serial dependence
        
        Args:
            entropy_emh: Expected entropy under independence
            entropy_market: Actual entropy in market
            
        Returns:
            Information quantity I
        """
        return max(0, entropy_emh - entropy_market)


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class GarcinBroutyAnalyzer:
    """Complete Garcin-Brouty information-theoretic analysis"""
    
    def __init__(self, log_volumes: np.ndarray, max_L: int = 20):
        """
        Initialize analyzer
        
        Args:
            log_volumes: Series of log-volumes
            max_L: Maximum pattern length to analyze
        """
        self.log_volumes = log_volumes
        self.max_L = max_L
        
        # Create symbolic representation
        self.symbolic = SymbolicRepresentation.create_binary_sequence(log_volumes)
        self.binary_seq = self.symbolic.sequence
    
    def analyze(self) -> GarcinBroutyResult:
        """
        Full Garcin-Brouty analysis for pattern lengths L = 1, 2, ..., max_L
        
        Returns:
            GarcinBroutyResult with all computed metrics
        """
        L_values = np.arange(1, self.max_L + 1)
        
        # Initialize storage
        conditional_probs_dict = {}
        pattern_freqs_dict = {}
        entropy_market_list = []
        entropy_emh_list = []
        information_list = []
        
        # Start with H_0 = 0 (no pattern)
        H_prev = 0.0
        
        # Iterate over pattern lengths
        for L in L_values:
            # Step 1: Compute conditional probabilities
            pi_L, p_L = ConditionalProbability.compute_for_length(self.binary_seq, L)
            conditional_probs_dict[L] = pi_L
            pattern_freqs_dict[L] = p_L
            
            # Step 2: Compute market entropy H_{L+1}
            H_market = ShannonEntropy.entropy_with_conditional(pi_L, p_L)
            entropy_market_list.append(H_market)
            
            # Step 3: Compute EMH entropy
            H_emh = ShannonEntropy.entropy_emh(H_prev)
            entropy_emh_list.append(H_emh)
            
            # Step 4: Compute information quantity
            I_L = MarketInformation.compute_information(H_emh, H_market)
            information_list.append(I_L)
            
            # Update for next iteration
            H_prev = H_market
        
        # Convert to arrays
        entropy_market = np.array(entropy_market_list)
        entropy_emh = np.array(entropy_emh_list)
        information = np.array(information_list)
        
        # Find half-life: where information drops to 50% of maximum
        max_info = np.max(information)
        half_max = 0.5 * max_info
        half_life_idx = np.argmax(information > half_max)
        half_life_L = int(L_values[half_life_idx]) if max_info > 0 else self.max_L
        
        return GarcinBroutyResult(
            L_values=L_values,
            conditional_probs=conditional_probs_dict,
            pattern_frequencies=pattern_freqs_dict,
            entropy_market=entropy_market,
            entropy_emh=entropy_emh,
            information=information,
            max_information=max_info,
            half_life_L=half_life_L
        )
    
    def analyze_by_side(self, bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> Tuple[GarcinBroutyResult, GarcinBroutyResult]:
        """
        Analyze both BID and ASK sides separately
        
        Args:
            bid_volumes: BID side log-volumes
            ask_volumes: ASK side log-volumes
            
        Returns:
            (bid_result, ask_result)
        """
        bid_analyzer = GarcinBroutyAnalyzer(bid_volumes, self.max_L)
        ask_analyzer = GarcinBroutyAnalyzer(ask_volumes, self.max_L)
        
        return bid_analyzer.analyze(), ask_analyzer.analyze()


# ============================================================================
# VISUALIZATION
# ============================================================================

class GarcinBroutyVisualizer:
    """Visualize Garcin-Brouty results"""
    
    @staticmethod
    def plot_information_quantity(bid_result: GarcinBroutyResult, 
                                  ask_result: GarcinBroutyResult,
                                  save_path: str = None):
        """
        Plot information quantity I_{L+1} for both sides
        
        Args:
            bid_result: Analysis result for BID side
            ask_result: Analysis result for ASK side
            save_path: Where to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # BID side
        ax = axes[0]
        ax.bar(bid_result.L_values, bid_result.information, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Pattern Length L', fontsize=11, fontweight='bold')
        ax.set_ylabel('Information Quantity $I_L$ (nats)', fontsize=11, fontweight='bold')
        ax.set_title(f'BID Side: Market Information\nMax I = {bid_result.max_information:.4f}, Half-life = {bid_result.half_life_L}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(bid_result.L_values[::2])
        
        # ASK side
        ax = axes[1]
        ax.bar(ask_result.L_values, ask_result.information, color='#ff7f0e', alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Pattern Length L', fontsize=11, fontweight='bold')
        ax.set_ylabel('Information Quantity $I_L$ (nats)', fontsize=11, fontweight='bold')
        ax.set_title(f'ASK Side: Market Information\nMax I = {ask_result.max_information:.4f}, Half-life = {ask_result.half_life_L}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ask_result.L_values[::2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Information quantity plot saved to {save_path}")
        
        return fig, axes
    
    @staticmethod
    def plot_entropy_comparison(bid_result: GarcinBroutyResult,
                               ask_result: GarcinBroutyResult,
                               save_path: str = None):
        """
        Plot comparison of H_market vs H_EMH
        
        Args:
            bid_result: Analysis result for BID side
            ask_result: Analysis result for ASK side
            save_path: Where to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # BID side
        ax = axes[0]
        ax.plot(bid_result.L_values, bid_result.entropy_market, 'o-', 
               color='#1f77b4', linewidth=2.5, markersize=6, label='Market $H_{L+1}$')
        ax.plot(bid_result.L_values, bid_result.entropy_emh, 's--', 
               color='#d62728', linewidth=2.5, markersize=6, label='EMH $H_{L+1}^{EMH}$')
        ax.fill_between(bid_result.L_values, bid_result.entropy_market, 
                       bid_result.entropy_emh, alpha=0.2, color='green', label='Lost Information')
        ax.set_xlabel('Pattern Length L', fontsize=11, fontweight='bold')
        ax.set_ylabel('Entropy (nats)', fontsize=11, fontweight='bold')
        ax.set_title('BID Side: Entropy Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # ASK side
        ax = axes[1]
        ax.plot(ask_result.L_values, ask_result.entropy_market, 'o-', 
               color='#ff7f0e', linewidth=2.5, markersize=6, label='Market $H_{L+1}$')
        ax.plot(ask_result.L_values, ask_result.entropy_emh, 's--', 
               color='#d62728', linewidth=2.5, markersize=6, label='EMH $H_{L+1}^{EMH}$')
        ax.fill_between(ask_result.L_values, ask_result.entropy_market, 
                       ask_result.entropy_emh, alpha=0.2, color='green', label='Lost Information')
        ax.set_xlabel('Pattern Length L', fontsize=11, fontweight='bold')
        ax.set_ylabel('Entropy (nats)', fontsize=11, fontweight='bold')
        ax.set_title('ASK Side: Entropy Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Entropy comparison plot saved to {save_path}")
        
        return fig, axes
    
    @staticmethod
    def plot_conditional_probabilities(bid_result: GarcinBroutyResult,
                                      ask_result: GarcinBroutyResult,
                                      max_L: int = 5,
                                      save_path: str = None):
        """
        Plot distribution of conditional probabilities π_i^L
        
        Args:
            bid_result: Analysis result for BID side
            ask_result: Analysis result for ASK side
            max_L: Maximum L to show
            save_path: Where to save figure
        """
        fig, axes = plt.subplots(max_L, 2, figsize=(12, 3*max_L))
        
        if max_L == 1:
            axes = axes.reshape(1, -1)
        
        for idx, L in enumerate(range(1, max_L + 1)):
            # BID side
            ax = axes[idx, 0]
            pi_bid = bid_result.conditional_probs[L]
            n_patterns = len(pi_bid)
            ax.hist(pi_bid, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Independence (π=0.5)')
            ax.set_xlabel('Conditional Probability $\pi_i^L$', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'BID: L={L} (n_patterns={n_patterns})', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # ASK side
            ax = axes[idx, 1]
            pi_ask = ask_result.conditional_probs[L]
            n_patterns = len(pi_ask)
            ax.hist(pi_ask, bins=20, color='#ff7f0e', alpha=0.7, edgecolor='black')
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Independence (π=0.5)')
            ax.set_xlabel('Conditional Probability $\pi_i^L$', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'ASK: L={L} (n_patterns={n_patterns})', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Conditional probabilities plot saved to {save_path}")
        
        return fig, axes


# ============================================================================
# INTERPRETATION
# ============================================================================

class GarcinBroutyInterpretation:
    """Interpret Garcin-Brouty results in LOB microstructure context"""
    
    @staticmethod
    def print_interpretation(bid_result: GarcinBroutyResult,
                           ask_result: GarcinBroutyResult,
                           symbol: str = "SANOFI"):
        """Print detailed microstructure interpretation"""
        
        print("\n" + "="*72)
        print("GARCIN-BROUTY INFORMATION-THEORETIC ANALYSIS")
        print(f"Market Microstructure Interpretation: {symbol} LOB")
        print("="*72)
        
        print("\n1. INFORMATION QUANTITY I_L (Predictability from past patterns)")
        print("-" * 72)
        
        print(f"\nBID SIDE:")
        print(f"  Max Information: {bid_result.max_information:.6f} nats")
        print(f"  → {bid_result.max_information:.2%} excess entropy reduction")
        print(f"  Half-life at L = {bid_result.half_life_L}")
        print(f"  → Volume changes are predictable up to lag-{bid_result.half_life_L}")
        
        print(f"\nASK SIDE:")
        print(f"  Max Information: {ask_result.max_information:.6f} nats")
        print(f"  → {ask_result.max_information:.2%} excess entropy reduction")
        print(f"  Half-life at L = {ask_result.half_life_L}")
        print(f"  → Volume changes are predictable up to lag-{ask_result.half_life_L}")
        
        print("\n2. INTERPRETATION ZONES")
        print("-" * 72)
        
        print(f"\nBID SIDE Interpretation:")
        max_info_bid = bid_result.max_information
        
        # Find L with maximum information
        max_idx = np.argmax(bid_result.information)
        L_max = bid_result.L_values[max_idx]
        
        print(f"\n  Zone 1 - Strong Predictability (L=1 to {L_max}):")
        print(f"    • Information increases from {bid_result.information[0]:.4f} to {max_info_bid:.4f} nats")
        print(f"    • Past {L_max}-bit pattern strongly predicts next movement")
        print(f"    • Interpretation:")
        if bid_result.information[0] > ask_result.information[0]:
            print(f"      → BID side shows stronger order splitting")
            print(f"      → Liquidity providers fragment large orders")
        else:
            print(f"      → Moderate order fragmentation")
        
        print(f"\n  Zone 2 - Weak Predictability (L={L_max+1} to {bid_result.L_values[-1]}):")
        remaining_info = bid_result.information[-1]
        print(f"    • Information decays to {remaining_info:.4f} nats")
        print(f"    • Very long patterns have limited predictive power")
        print(f"    • Suggests market reverts toward randomness at longer lags")
        
        # Same for ASK
        print(f"\nASK SIDE Interpretation:")
        max_info_ask = ask_result.max_information
        max_idx_ask = np.argmax(ask_result.information)
        L_max_ask = ask_result.L_values[max_idx_ask]
        
        print(f"\n  Zone 1 - Strong Predictability (L=1 to {L_max_ask}):")
        print(f"    • Information increases from {ask_result.information[0]:.4f} to {max_info_ask:.4f} nats")
        print(f"    • Past {L_max_ask}-bit pattern strongly predicts next movement")
        if ask_result.information[0] > bid_result.information[0]:
            print(f"    • ASK side shows stronger order fragmentation")
        else:
            print(f"    • Comparable to BID side fragmentation")
        
        print(f"\n  Zone 2 - Weak Predictability (L={L_max_ask+1} to {ask_result.L_values[-1]}):")
        remaining_info_ask = ask_result.information[-1]
        print(f"    • Information decays to {remaining_info_ask:.4f} nats")
        
        print("\n3. BID vs ASK COMPARISON")
        print("-" * 72)
        
        if bid_result.max_information > ask_result.max_information:
            ratio = bid_result.max_information / ask_result.max_information
            print(f"\n  BID side is {ratio:.2f}x more predictable than ASK side")
            print(f"  → Stronger order splitting on BID (buy orders fragmented more)")
        else:
            ratio = ask_result.max_information / bid_result.max_information
            print(f"\n  ASK side is {ratio:.2f}x more predictable than BID side")
            print(f"  → Stronger order splitting on ASK (sell orders fragmented more)")
        
        print("\n4. MARKET EFFICIENCY IMPLICATIONS")
        print("-" * 72)
        
        avg_info = (bid_result.max_information + ask_result.max_information) / 2
        
        if avg_info > 0.1:
            print(f"\n  ✗ WEAK FORM EFFICIENCY VIOLATED")
            print(f"    Average max information: {avg_info:.4f} nats")
            print(f"    → Sequential volume patterns contain exploitable information")
            print(f"    → Market microstructure shows non-random behavior")
            print(f"    → Possible applications:")
            print(f"       • Volume prediction models (AR/ARMA)")
            print(f"       • Order execution algorithms")
            print(f"       • Trading strategies based on volume regimes")
        else:
            print(f"\n  ✓ WEAK FORM EFFICIENCY (approximately)")
            print(f"    Max information: {avg_info:.4f} nats (negligible)")
            print(f"    → Volume movements appear random")
        
        print("\n" + "="*72)
