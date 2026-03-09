"""
CIR Estimator Validation Test
Test using Monte Carlo simulation to verify if parameter estimation is biased

Steps:
1. Generate N CIR sample paths with known parameters (κ_true, θ_true, σ_true)
2. For each sample, estimate parameters using the moment estimator
3. Analyze the distribution of estimators (κ_hat, θ_hat, σ_hat)
4. Compare with theoretical properties
5. Check for systematic bias
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / 'estimator_test_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_cir_path(kappa, theta, sigma, T, dt=1.0, D0=None, seed=None):
    """
    Simulate CIR process using Euler scheme:
    dD_t = κ(θ - D_t)dt + σ√(D_t)dW_t
    
    Args:
        kappa: Mean reversion speed
        theta: Long-run mean
        sigma: Volatility
        T: Number of time steps
        dt: Time step size
        D0: Initial depth (default: theta)
        seed: Random seed
    
    Returns:
        D: Array of simulated depth values (length T+1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if D0 is None:
        D0 = theta
    
    D = np.zeros(T + 1)
    D[0] = max(D0, 0.1)  # Ensure positive
    
    for t in range(T):
        dW = np.random.normal(0, np.sqrt(dt))
        sqrt_D = np.sqrt(max(D[t], 0.001))  # Ensure positive for sqrt
        dD = kappa * (theta - D[t]) * dt + sigma * sqrt_D * dW
        D[t+1] = max(D[t] + dD, 0.001)  # Keep positive
    
    return D


def estimate_cir_params_moment_local(D_series, epsilon=1e-6, apply_correction=False):
    """
    Moment-based CIR parameter estimator (same formula used in main script).
    """
    if isinstance(D_series, np.ndarray):
        D = D_series
    else:
        D = D_series.values if hasattr(D_series, 'values') else D_series
    m = len(D)
    if m < 2:
        return None
    theta_hat = np.mean(D)
    num = np.sum(D[:-1] * (D[1:] - D[:-1])) / (m - 1)
    den = np.sum(D * (theta_hat - D)) / m
    kappa_hat = num / den if den != 0 else np.nan
    if apply_correction and np.isfinite(kappa_hat):
        kappa_hat = kappa_hat - 0.5 / m
        kappa_hat = max(kappa_hat, 0.0)
    # sigma using residual formula
    delta = D[1:] - D[:-1]
    eps_hat = delta - kappa_hat * (theta_hat - D[:-1])
    denom = np.sqrt(np.maximum(D[:-1], epsilon))
    z = eps_hat / denom
    sigma_hat = np.sqrt(np.var(z, ddof=1))
    feller = 2 * kappa_hat * theta_hat >= sigma_hat ** 2 if np.isfinite(sigma_hat) else False
    return {
        'kappa_hat': kappa_hat,
        'theta_hat': theta_hat,
        'sigma_hat': sigma_hat,
        'feller': feller,
    }


def monte_carlo_simulation(kappa_true, theta_true, sigma_true, T=100, n_simulations=100, seed=42):
    """
    Run Monte Carlo simulation to estimate distribution of estimators.

    Args:
        kappa_true, theta_true, sigma_true: True CIR parameters
        T: Sample length
        n_simulations: Number of sample paths to generate
        seed: Base random seed
    
    Returns:
        dict with results
    """
    est_func = estimate_cir_params_moment_local
    keys = ['kappa_hat','theta_hat','sigma_hat','feller']

    results = {k: [] for k in keys}

    print(f"Running {n_simulations} simulations with kappa={kappa_true}, theta={theta_true}, sigma={sigma_true} using moment estimator...")
    print(f"Sample length T={T}")

    for i in range(n_simulations):
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{n_simulations} completed")
        # Generate CIR path
        D = simulate_cir_path(kappa_true, theta_true, sigma_true, T, dt=1.0, seed=seed+i)
        # Estimate parameters
        est = est_func(D)
        if est is not None:
            for key in keys:
                if key in est:
                    results[key].append(est[key])
    # convert lists to arrays
    for key in list(results.keys()):
        results[key] = np.array(results[key])
    print(f"✓ Completed {len(results['kappa_hat'])} successful estimations\n")
    return results


def compute_statistics(results, param_name, true_value):
    """Compute statistics for an estimated parameter."""
    estimates = results[param_name]
    
    if len(estimates) == 0:
        return None
    
    mean = np.mean(estimates)
    std = np.std(estimates, ddof=1)
    bias = mean - true_value
    relative_bias = bias / true_value * 100 if true_value != 0 else np.nan
    rmse = np.sqrt(np.mean((estimates - true_value)**2))
    
    # Percentiles
    p05, p25, p50, p75, p95 = np.percentile(estimates, [5, 25, 50, 75, 95])
    
    return {
        'mean': mean,
        'std': std,
        'bias': bias,
        'relative_bias': relative_bias,
        'rmse': rmse,
        'p05': p05,
        'p25': p25,
        'median': p50,
        'p75': p75,
        'p95': p95,
        'true_value': true_value,
        'estimates': estimates,
    }


def plot_histograms(results, param_specs, output_file):
    """
    Create histogram plots for parameter distributions.
    
    Args:
        results: Dict of estimation results
        param_specs: List of (param_name, true_value) tuples
        output_file: Path to save plot
    """
    n_params = len(param_specs)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 4*n_params))
    
    if n_params == 1:
        axes = [axes]
    
    for ax, (param_name, true_value) in zip(axes, param_specs):
        estimates = results[param_name]
        
        if len(estimates) == 0:
            continue
        
        # Histogram
        ax.hist(estimates, bins=20, alpha=0.7, color='blue', edgecolor='black')
        
        # Mark true value
        ax.axvline(true_value, color='red', linestyle='--', linewidth=2, label=f'True: {true_value:.6f}')
        
        # Mark mean
        mean_val = np.mean(estimates)
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Estimated Mean: {mean_val:.6f}')
        
        stats_obj = compute_statistics(results, param_name, true_value)
        if stats_obj:
            title = f'{param_name}\nMean={stats_obj["mean"]:.6f}, Std={stats_obj["std"]:.6f}, Bias={stats_obj["bias"]:.6f} ({stats_obj["relative_bias"]:.2f}%)'
            ax.set_title(title, fontsize=10)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    print(f"Saved: {output_file}")


def print_summary_table(results, param_specs):
    """Print summary table of estimation statistics."""
    print("\n" + "="*90)
    print("MONTE CARLO ESTIMATION RESULTS SUMMARY")
    print("="*90)
    
    # Header
    print(f"{'Parameter':<15} {'True':<12} {'Mean':<12} {'Std Dev':<12} {'Bias':<12} {'Rel Bias %':<12} {'RMSE':<12}")
    print("-"*90)
    
    for param_name, true_value in param_specs:
        stats_obj = compute_statistics(results, param_name, true_value)
        if stats_obj:
            print(f"{param_name:<15} {true_value:<12.6f} {stats_obj['mean']:<12.6f} {stats_obj['std']:<12.6f} "
                  f"{stats_obj['bias']:<12.6f} {stats_obj['relative_bias']:<12.2f} {stats_obj['rmse']:<12.6f}")
    
    print("="*90)
    print("\nPercentile Distribution:")
    print("-"*90)
    print(f"{'Parameter':<15} {'5%':<12} {'25%':<12} {'Median':<12} {'75%':<12} {'95%':<12}")
    print("-"*90)
    
    for param_name, true_value in param_specs:
        stats_obj = compute_statistics(results, param_name, true_value)
        if stats_obj:
            print(f"{param_name:<15} {stats_obj['p05']:<12.6f} {stats_obj['p25']:<12.6f} "
                  f"{stats_obj['median']:<12.6f} {stats_obj['p75']:<12.6f} {stats_obj['p95']:<12.6f}")
    
    print("="*90)


def run_comprehensive_test():
    """Run comprehensive Monte Carlo test."""
    
    print("\n" + "="*90)
    print("CIR PARAMETER ESTIMATOR VALIDATION TEST")
    print("="*90 + "\n")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Scenario 1: Small sample (T=100)',
            'kappa': 0.05,
            'theta': 36.0,
            'sigma': 0.8,
            'T': 100,
            'n_sims': 100,
        },
        {
            'name': 'Scenario 2: Medium sample (T=500)',
            'kappa': 0.05,
            'theta': 36.0,
            'sigma': 0.8,
            'T': 500,
            'n_sims': 100,
        },
        {
            'name': 'Scenario 3: Large sample (T=1000)',
            'kappa': 0.05,
            'theta': 36.0,
            'sigma': 0.8,
            'T': 1000,
            'n_sims': 100,
        },
    ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"\n{'#'*90}")
        print(f"# {scenario['name']} (method=moment)")
        print(f"{'#'*90}")

        # Run simulation
        results = monte_carlo_simulation(
            scenario['kappa'],
            scenario['theta'],
            scenario['sigma'],
            T=scenario['T'],
            n_simulations=scenario['n_sims'],
            seed=42,
        )

        # Parameter specs
        param_specs = [
            ('kappa_hat', scenario['kappa']),
            ('theta_hat', scenario['theta']),
            ('sigma_hat', scenario['sigma']),
        ]

        # Print summary
        print_summary_table(results, param_specs)

        # Generate plots
        plot_name = f"{scenario['name'].replace(' ', '_').replace(':', '').lower()}_moment"
        plot_file = OUT_DIR / f'histogram_{plot_name}.png'
        plot_histograms(results, param_specs, str(plot_file))

        all_results[f"{scenario['name']} (moment)"] = {
            'parameters': scenario,
            'method': 'moment',
            'statistics': {param_name: compute_statistics(results, param_name, true_val)
                          for param_name, true_val in param_specs},
            'feller_condition': {
                'satisfied': np.mean(results['feller']),
                'details': f"{int(np.sum(results['feller']))}/{len(results['feller'])} paths satisfied"
            }
        }
    
    # Save comprehensive results
    summary_file = OUT_DIR / 'comprehensive_test_results.json'
    with open(summary_file, 'w') as f:
        # Convert numpy types for JSON serialization
        output = {}
        for scenario_name, scenario_results in all_results.items():
            output[scenario_name] = {
                'parameters': scenario_results['parameters'],
                'feller_condition': scenario_results['feller_condition'],
                'statistics': {
                    param_name: {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                 for k, v in stats_dict.items() if k != 'estimates'}
                    for param_name, stats_dict in scenario_results['statistics'].items()
                }
            }
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved comprehensive results: {summary_file}\n")
    
    return all_results


if __name__ == '__main__':
    # Run comprehensive Monte Carlo test
    all_results = run_comprehensive_test()

    print("\n✓ All tests completed.")
    print(f"\nOutputs saved to: {OUT_DIR}/")
