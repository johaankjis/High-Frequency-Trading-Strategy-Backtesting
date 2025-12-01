"""
Benchmarking suite for comparing Monte Carlo pricing against analytical solutions.
Includes performance profiling, convergence analysis, and accuracy metrics.
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from config import (
    OptionContract, SimulationConfig, OptionType, ExerciseStyle,
    ConstantVolParams, HestonParams, BenchmarkResults,
    DEFAULT_OPTION, DEFAULT_SIM_CONFIG, DEFAULT_HESTON
)
from pricing_engine import MonteCarloEngine, price_option
from analytical_models import (
    black_scholes_price, binomial_tree_price, 
    heston_call_price, heston_put_price
)


@dataclass
class PerformanceMetrics:
    """Container for performance benchmarking results."""
    model_name: str
    num_paths: int
    num_steps: int
    execution_time_ms: float
    paths_per_second: float
    memory_estimate_mb: float


def benchmark_european_bs(
    contract: OptionContract,
    vol_params: ConstantVolParams,
    config: SimulationConfig
) -> BenchmarkResults:
    """
    Benchmark Monte Carlo vs Black-Scholes for European options.
    """
    # Analytical price
    analytical_price = black_scholes_price(contract, vol_params)
    
    # Binomial tree price
    binomial_price = binomial_tree_price(contract, vol_params)
    
    # Monte Carlo price with timing
    start_time = time.perf_counter()
    mc_price, mc_std_err, mc_ci, _ = price_option(contract, vol_params, config)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return BenchmarkResults(
        monte_carlo_price=mc_price,
        analytical_price=analytical_price,
        binomial_price=binomial_price,
        mc_std_error=mc_std_err,
        mc_confidence_interval=mc_ci,
        computation_time_ms=elapsed_ms,
        num_paths=config.num_paths
    )


def benchmark_heston(
    contract: OptionContract,
    heston_params: HestonParams,
    config: SimulationConfig
) -> BenchmarkResults:
    """
    Benchmark Monte Carlo vs semi-analytical Heston pricing.
    """
    # Semi-analytical Heston price
    if contract.option_type == OptionType.CALL:
        analytical_price = heston_call_price(contract, heston_params)
    else:
        analytical_price = heston_put_price(contract, heston_params)
    
    # Monte Carlo price
    start_time = time.perf_counter()
    mc_price, mc_std_err, mc_ci, _ = price_option(contract, heston_params, config)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return BenchmarkResults(
        monte_carlo_price=mc_price,
        analytical_price=analytical_price,
        binomial_price=None,
        mc_std_error=mc_std_err,
        mc_confidence_interval=mc_ci,
        computation_time_ms=elapsed_ms,
        num_paths=config.num_paths
    )


def performance_profile(
    contract: OptionContract,
    vol_params,
    path_counts: List[int] = None,
    step_counts: List[int] = None
) -> List[PerformanceMetrics]:
    """
    Profile execution performance across different simulation sizes.
    """
    if path_counts is None:
        path_counts = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    if step_counts is None:
        step_counts = [252]
    
    results = []
    
    for num_paths in path_counts:
        for num_steps in step_counts:
            config = SimulationConfig(
                num_paths=num_paths,
                num_steps=num_steps,
                seed=42,
                antithetic=True,
                use_jit=True
            )
            
            # Warm-up run (JIT compilation)
            _ = price_option(contract, vol_params, config)
            
            # Timed run
            start_time = time.perf_counter()
            _ = price_option(contract, vol_params, config)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Estimate memory usage (approximate)
            memory_mb = (num_paths * (num_steps + 1) * 8) / (1024 * 1024)
            
            model_name = type(vol_params).__name__.replace("Params", "")
            
            results.append(PerformanceMetrics(
                model_name=model_name,
                num_paths=num_paths,
                num_steps=num_steps,
                execution_time_ms=elapsed_ms,
                paths_per_second=num_paths / (elapsed_ms / 1000),
                memory_estimate_mb=memory_mb
            ))
    
    return results


def convergence_study(
    contract: OptionContract,
    vol_params,
    analytical_price: float,
    path_counts: List[int] = None,
    num_trials: int = 10
) -> Dict:
    """
    Study Monte Carlo convergence behavior with statistical analysis.
    """
    if path_counts is None:
        path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
    
    results = {
        'path_counts': path_counts,
        'mean_prices': [],
        'mean_errors': [],
        'rmse': [],
        'std_of_estimates': [],
        'theoretical_convergence': []
    }
    
    for num_paths in path_counts:
        trial_prices = []
        
        for trial in range(num_trials):
            config = SimulationConfig(
                num_paths=num_paths,
                num_steps=252,
                seed=trial * 1000,  # Different seed each trial
                antithetic=True,
                use_jit=True
            )
            
            price, _, _, _ = price_option(contract, vol_params, config)
            trial_prices.append(price)
        
        trial_prices = np.array(trial_prices)
        errors = trial_prices - analytical_price
        
        results['mean_prices'].append(np.mean(trial_prices))
        results['mean_errors'].append(np.mean(np.abs(errors)))
        results['rmse'].append(np.sqrt(np.mean(errors**2)))
        results['std_of_estimates'].append(np.std(trial_prices))
        
        # Theoretical O(1/sqrt(N)) convergence
        results['theoretical_convergence'].append(1.0 / np.sqrt(num_paths))
    
    return results


def variance_reduction_comparison(
    contract: OptionContract,
    vol_params: ConstantVolParams
) -> Dict:
    """
    Compare variance reduction techniques.
    """
    num_paths = 100_000
    num_trials = 20
    
    techniques = {
        'standard': {'antithetic': False},
        'antithetic': {'antithetic': True}
    }
    
    results = {}
    
    for name, settings in techniques.items():
        trial_prices = []
        trial_std_errs = []
        
        for trial in range(num_trials):
            config = SimulationConfig(
                num_paths=num_paths,
                num_steps=252,
                seed=trial * 100,
                antithetic=settings['antithetic'],
                use_jit=True
            )
            
            price, std_err, _, _ = price_option(contract, vol_params, config)
            trial_prices.append(price)
            trial_std_errs.append(std_err)
        
        results[name] = {
            'mean_price': np.mean(trial_prices),
            'std_of_prices': np.std(trial_prices),
            'mean_std_error': np.mean(trial_std_errs),
            'variance_reduction_ratio': None
        }
    
    # Calculate variance reduction ratio
    if 'standard' in results and 'antithetic' in results:
        std_var = results['standard']['std_of_prices']**2
        anti_var = results['antithetic']['std_of_prices']**2
        results['antithetic']['variance_reduction_ratio'] = std_var / anti_var
    
    return results


def run_full_benchmark_suite() -> Dict:
    """
    Run comprehensive benchmark suite across all models and configurations.
    """
    print("=" * 70)
    print("MONTE CARLO OPTIONS PRICING - BENCHMARK SUITE")
    print("=" * 70)
    
    # Test contracts
    test_contracts = [
        ("ATM Call", OptionContract(
            spot_price=100, strike_price=100, time_to_maturity=1.0,
            risk_free_rate=0.05, dividend_yield=0.02,
            option_type=OptionType.CALL, exercise_style=ExerciseStyle.EUROPEAN
        )),
        ("OTM Put", OptionContract(
            spot_price=100, strike_price=90, time_to_maturity=0.5,
            risk_free_rate=0.05, dividend_yield=0.0,
            option_type=OptionType.PUT, exercise_style=ExerciseStyle.EUROPEAN
        )),
        ("ITM Call", OptionContract(
            spot_price=110, strike_price=100, time_to_maturity=0.25,
            risk_free_rate=0.03, dividend_yield=0.01,
            option_type=OptionType.CALL, exercise_style=ExerciseStyle.EUROPEAN
        )),
        ("American Put", OptionContract(
            spot_price=100, strike_price=100, time_to_maturity=1.0,
            risk_free_rate=0.05, dividend_yield=0.02,
            option_type=OptionType.PUT, exercise_style=ExerciseStyle.AMERICAN
        ))
    ]
    
    bs_params = ConstantVolParams(sigma=0.2)
    heston_params = DEFAULT_HESTON
    config = SimulationConfig(num_paths=100_000, num_steps=252, seed=42)
    
    all_results = {}
    
    # Black-Scholes benchmarks
    print("\n" + "-" * 70)
    print("BLACK-SCHOLES MODEL BENCHMARKS")
    print("-" * 70)
    
    for name, contract in test_contracts:
        if contract.exercise_style == ExerciseStyle.EUROPEAN:
            result = benchmark_european_bs(contract, bs_params, config)
            all_results[f"BS_{name}"] = result
            
            print(f"\n{name}:")
            print(f"  Analytical (BS):  ${result.analytical_price:.6f}")
            print(f"  Binomial Tree:    ${result.binomial_price:.6f}")
            print(f"  Monte Carlo:      ${result.monte_carlo_price:.6f}")
            print(f"  MC Std Error:     ${result.mc_std_error:.6f}")
            print(f"  95% CI:           [${result.mc_confidence_interval[0]:.4f}, ${result.mc_confidence_interval[1]:.4f}]")
            print(f"  Absolute Error:   ${result.absolute_error:.6f}")
            print(f"  Relative Error:   {result.relative_error:.4f}%")
            print(f"  Computation Time: {result.computation_time_ms:.2f} ms")
    
    # Heston model benchmarks
    print("\n" + "-" * 70)
    print("HESTON STOCHASTIC VOLATILITY BENCHMARKS")
    print("-" * 70)
    
    for name, contract in test_contracts[:3]:  # European only
        result = benchmark_heston(contract, heston_params, config)
        all_results[f"Heston_{name}"] = result
        
        print(f"\n{name}:")
        print(f"  Semi-Analytical:  ${result.analytical_price:.6f}")
        print(f"  Monte Carlo:      ${result.monte_carlo_price:.6f}")
        print(f"  MC Std Error:     ${result.mc_std_error:.6f}")
        print(f"  Absolute Error:   ${result.absolute_error:.6f}")
        print(f"  Relative Error:   {result.relative_error:.4f}%")
        print(f"  Computation Time: {result.computation_time_ms:.2f} ms")
    
    # American option (LSM)
    print("\n" + "-" * 70)
    print("AMERICAN OPTION (LONGSTAFF-SCHWARTZ)")
    print("-" * 70)
    
    american_contract = test_contracts[3][1]
    binomial_price = binomial_tree_price(american_contract, bs_params, num_steps=1000)
    
    start_time = time.perf_counter()
    mc_price, mc_std_err, mc_ci, _ = price_option(american_contract, bs_params, config)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    print(f"\nAmerican Put (ATM):")
    print(f"  Binomial (1000 steps): ${binomial_price:.6f}")
    print(f"  Monte Carlo (LSM):     ${mc_price:.6f}")
    print(f"  MC Std Error:          ${mc_std_err:.6f}")
    print(f"  95% CI:                [${mc_ci[0]:.4f}, ${mc_ci[1]:.4f}]")
    print(f"  Computation Time:      {elapsed_ms:.2f} ms")
    
    # Performance profiling
    print("\n" + "-" * 70)
    print("PERFORMANCE PROFILE")
    print("-" * 70)
    
    perf_results = performance_profile(
        DEFAULT_OPTION, bs_params,
        path_counts=[10_000, 50_000, 100_000, 500_000]
    )
    
    print(f"\n{'Paths':>12} | {'Time (ms)':>12} | {'Paths/sec':>15} | {'Memory (MB)':>12}")
    print("-" * 60)
    for p in perf_results:
        print(f"{p.num_paths:>12,} | {p.execution_time_ms:>12.2f} | {p.paths_per_second:>15,.0f} | {p.memory_estimate_mb:>12.2f}")
    
    # Variance reduction comparison
    print("\n" + "-" * 70)
    print("VARIANCE REDUCTION ANALYSIS")
    print("-" * 70)
    
    vr_results = variance_reduction_comparison(DEFAULT_OPTION, bs_params)
    
    for technique, stats in vr_results.items():
        print(f"\n{technique.upper()}:")
        print(f"  Mean Price:       ${stats['mean_price']:.6f}")
        print(f"  Std of Estimates: ${stats['std_of_prices']:.6f}")
        if stats['variance_reduction_ratio']:
            print(f"  Variance Reduction Ratio: {stats['variance_reduction_ratio']:.2f}x")
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = run_full_benchmark_suite()
