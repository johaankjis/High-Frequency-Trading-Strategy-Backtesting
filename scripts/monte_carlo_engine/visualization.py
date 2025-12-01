"""
Visualization utilities for Monte Carlo simulation results.
Generates publication-quality charts for analysis and reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Optional, Tuple
from io import BytesIO
import base64

from config import (
    OptionContract, SimulationConfig, ConstantVolParams, 
    HestonParams, OptionType, ExerciseStyle
)
from sde_models import simulate_gbm, simulate_heston
from pricing_engine import price_option
from analytical_models import black_scholes_price


# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2563eb',
    'secondary': '#059669',
    'accent': '#dc2626',
    'neutral': '#6b7280',
    'light': '#e5e7eb'
}


def plot_price_paths(
    paths: np.ndarray,
    title: str = "Simulated Asset Price Paths",
    num_display: int = 100,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """Plot sample price paths from simulation."""
    fig, ax = plt.subplots(figsize=figsize)
    
    num_paths = min(num_display, paths.shape[0])
    time_steps = np.arange(paths.shape[1])
    
    for i in range(num_paths):
        ax.plot(time_steps, paths[i], alpha=0.3, linewidth=0.5, color=COLORS['primary'])
    
    # Plot mean path
    mean_path = np.mean(paths, axis=0)
    ax.plot(time_steps, mean_path, color=COLORS['accent'], linewidth=2, label='Mean Path')
    
    # Plot confidence bands
    std_path = np.std(paths, axis=0)
    ax.fill_between(
        time_steps,
        mean_path - 2*std_path,
        mean_path + 2*std_path,
        alpha=0.2,
        color=COLORS['secondary'],
        label='95% Confidence Band'
    )
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Asset Price', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_terminal_distribution(
    paths: np.ndarray,
    contract: OptionContract,
    analytical_price: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> Figure:
    """Plot terminal price distribution with payoff analysis."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    terminal_prices = paths[:, -1]
    K = contract.strike_price
    
    # Terminal price distribution
    ax1 = axes[0]
    ax1.hist(terminal_prices, bins=100, density=True, alpha=0.7, color=COLORS['primary'])
    ax1.axvline(K, color=COLORS['accent'], linestyle='--', linewidth=2, label=f'Strike = ${K}')
    ax1.axvline(np.mean(terminal_prices), color=COLORS['secondary'], linestyle='-', 
                linewidth=2, label=f'Mean = ${np.mean(terminal_prices):.2f}')
    
    ax1.set_xlabel('Terminal Asset Price', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Terminal Price Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Payoff distribution
    ax2 = axes[1]
    if contract.option_type == OptionType.CALL:
        payoffs = np.maximum(terminal_prices - K, 0)
        payoff_label = 'Call Payoff'
    else:
        payoffs = np.maximum(K - terminal_prices, 0)
        payoff_label = 'Put Payoff'
    
    # Discount payoffs
    discount = np.exp(-contract.risk_free_rate * contract.time_to_maturity)
    discounted_payoffs = payoffs * discount
    
    ax2.hist(discounted_payoffs, bins=100, density=True, alpha=0.7, color=COLORS['secondary'])
    ax2.axvline(np.mean(discounted_payoffs), color=COLORS['accent'], linestyle='-',
                linewidth=2, label=f'MC Price = ${np.mean(discounted_payoffs):.4f}')
    
    if analytical_price is not None:
        ax2.axvline(analytical_price, color=COLORS['primary'], linestyle='--',
                    linewidth=2, label=f'Analytical = ${analytical_price:.4f}')
    
    ax2.set_xlabel(f'Discounted {payoff_label}', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Payoff Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_convergence(
    convergence_data: Dict,
    figsize: Tuple[int, int] = (12, 5)
) -> Figure:
    """Plot convergence analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    path_counts = np.array(convergence_data['path_counts'])
    rmse = np.array(convergence_data['rmse'])
    theoretical = np.array(convergence_data['theoretical_convergence'])
    
    # RMSE convergence (log-log scale)
    ax1 = axes[0]
    ax1.loglog(path_counts, rmse, 'o-', color=COLORS['primary'], 
               linewidth=2, markersize=8, label='Empirical RMSE')
    
    # Theoretical O(1/sqrt(N)) line
    scale = rmse[0] / theoretical[0]
    ax1.loglog(path_counts, scale * theoretical, '--', color=COLORS['neutral'],
               linewidth=2, label=r'Theoretical $O(1/\sqrt{N})$')
    
    ax1.set_xlabel('Number of Paths', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Convergence Rate Analysis', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Price estimates with error bars
    ax2 = axes[1]
    mean_prices = np.array(convergence_data['mean_prices'])
    std_estimates = np.array(convergence_data['std_of_estimates'])
    
    ax2.errorbar(path_counts, mean_prices, yerr=1.96*std_estimates, 
                 fmt='o-', color=COLORS['primary'], capsize=5, 
                 linewidth=2, markersize=8, label='MC Estimate ± 95% CI')
    
    # Reference line (final estimate)
    ax2.axhline(mean_prices[-1], color=COLORS['neutral'], linestyle='--', 
                alpha=0.7, label=f'Final Estimate: ${mean_prices[-1]:.4f}')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Paths', fontsize=12)
    ax2.set_ylabel('Option Price', fontsize=12)
    ax2.set_title('Price Estimate Convergence', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_heston_variance_paths(
    S_paths: np.ndarray,
    v_paths: np.ndarray,
    num_display: int = 50,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """Plot Heston model price and variance paths."""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    num_paths = min(num_display, S_paths.shape[0])
    time_steps = np.arange(S_paths.shape[1])
    
    # Asset price paths
    ax1 = axes[0]
    for i in range(num_paths):
        ax1.plot(time_steps, S_paths[i], alpha=0.3, linewidth=0.5, color=COLORS['primary'])
    
    mean_S = np.mean(S_paths, axis=0)
    ax1.plot(time_steps, mean_S, color=COLORS['accent'], linewidth=2, label='Mean Price')
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Asset Price', fontsize=12)
    ax1.set_title('Heston Model: Asset Price Paths', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Variance paths
    ax2 = axes[1]
    for i in range(num_paths):
        ax2.plot(time_steps, v_paths[i], alpha=0.3, linewidth=0.5, color=COLORS['secondary'])
    
    mean_v = np.mean(v_paths, axis=0)
    ax2.plot(time_steps, mean_v, color=COLORS['accent'], linewidth=2, label='Mean Variance')
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Heston Model: Stochastic Variance Paths', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_performance_scaling(
    performance_data: List,
    figsize: Tuple[int, int] = (12, 5)
) -> Figure:
    """Plot performance scaling analysis."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    path_counts = [p.num_paths for p in performance_data]
    times = [p.execution_time_ms for p in performance_data]
    throughput = [p.paths_per_second for p in performance_data]
    
    # Execution time scaling
    ax1 = axes[0]
    ax1.plot(path_counts, times, 'o-', color=COLORS['primary'], 
             linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Paths', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('Execution Time Scaling', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Throughput
    ax2 = axes[1]
    ax2.bar(range(len(path_counts)), [t/1e6 for t in throughput], 
            color=COLORS['secondary'], alpha=0.8)
    ax2.set_xticks(range(len(path_counts)))
    ax2.set_xticklabels([f'{p//1000}K' for p in path_counts])
    
    ax2.set_xlabel('Number of Paths', fontsize=12)
    ax2.set_ylabel('Throughput (M paths/sec)', fontsize=12)
    ax2.set_title('Simulation Throughput', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_greeks_surface(
    contract: OptionContract,
    vol_params: ConstantVolParams,
    spot_range: Tuple[float, float] = (80, 120),
    vol_range: Tuple[float, float] = (0.1, 0.5),
    resolution: int = 50,
    figsize: Tuple[int, int] = (14, 10)
) -> Figure:
    """Plot option Greeks as surfaces."""
    from analytical_models import black_scholes_price, black_scholes_greeks
    
    spots = np.linspace(spot_range[0], spot_range[1], resolution)
    vols = np.linspace(vol_range[0], vol_range[1], resolution)
    S, V = np.meshgrid(spots, vols)
    
    # Calculate Greeks for each (S, vol) combination
    prices = np.zeros_like(S)
    deltas = np.zeros_like(S)
    gammas = np.zeros_like(S)
    vegas = np.zeros_like(S)
    
    for i in range(resolution):
        for j in range(resolution):
            temp_contract = OptionContract(
                spot_price=S[i, j],
                strike_price=contract.strike_price,
                time_to_maturity=contract.time_to_maturity,
                risk_free_rate=contract.risk_free_rate,
                dividend_yield=contract.dividend_yield,
                option_type=contract.option_type,
                exercise_style=contract.exercise_style
            )
            temp_vol = ConstantVolParams(sigma=V[i, j])
            
            prices[i, j] = black_scholes_price(temp_contract, temp_vol)
            greeks = black_scholes_greeks(temp_contract, temp_vol)
            deltas[i, j] = greeks['delta']
            gammas[i, j] = greeks['gamma']
            vegas[i, j] = greeks['vega']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw={'projection': '3d'})
    
    surfaces = [
        (axes[0, 0], prices, 'Option Price', 'Price'),
        (axes[0, 1], deltas, 'Delta', 'Delta'),
        (axes[1, 0], gammas, 'Gamma', 'Gamma'),
        (axes[1, 1], vegas, 'Vega', 'Vega')
    ]
    
    for ax, data, title, zlabel in surfaces:
        surf = ax.plot_surface(S, V, data, cmap='viridis', alpha=0.8,
                               linewidth=0, antialiased=True)
        ax.set_xlabel('Spot Price')
        ax.set_ylabel('Volatility')
        ax.set_zlabel(zlabel)
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    return fig


def generate_report_figures(
    contract: OptionContract,
    vol_params,
    config: SimulationConfig
) -> Dict[str, Figure]:
    """Generate all figures for a comprehensive report."""
    figures = {}
    
    # Simulate paths
    if isinstance(vol_params, ConstantVolParams):
        from sde_models import simulate_gbm
        paths = simulate_gbm(contract, vol_params, config)
        figures['paths'] = plot_price_paths(paths, "GBM Price Paths")
        
        analytical = black_scholes_price(contract, vol_params)
        figures['terminal'] = plot_terminal_distribution(paths, contract, analytical)
        
    elif isinstance(vol_params, HestonParams):
        from sde_models import simulate_heston
        S_paths, v_paths = simulate_heston(contract, vol_params, config)
        figures['heston'] = plot_heston_variance_paths(S_paths, v_paths)
        figures['terminal'] = plot_terminal_distribution(S_paths, contract)
    
    return figures


if __name__ == "__main__":
    # Demo visualization
    from config import DEFAULT_OPTION, DEFAULT_SIM_CONFIG, DEFAULT_HESTON
    
    contract = DEFAULT_OPTION
    bs_params = ConstantVolParams(sigma=0.2)
    config = SimulationConfig(num_paths=10000, num_steps=252, seed=42)
    
    # Generate and show sample plots
    paths = simulate_gbm(contract, bs_params, config)
    
    fig1 = plot_price_paths(paths, num_display=100)
    plt.savefig('price_paths.png', dpi=150, bbox_inches='tight')
    
    analytical = black_scholes_price(contract, bs_params)
    fig2 = plot_terminal_distribution(paths, contract, analytical)
    plt.savefig('terminal_dist.png', dpi=150, bbox_inches='tight')
    
    # Heston paths
    S, v = simulate_heston(contract, DEFAULT_HESTON, config)
    fig3 = plot_heston_variance_paths(S, v)
    plt.savefig('heston_paths.png', dpi=150, bbox_inches='tight')
    
    print("Visualization examples saved to current directory.")
    plt.show()
