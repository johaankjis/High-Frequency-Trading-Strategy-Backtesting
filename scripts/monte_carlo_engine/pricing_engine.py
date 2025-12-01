"""
Monte Carlo pricing engine for European and American options.
Implements variance reduction techniques and Longstaff-Schwartz for American options.
"""

import numpy as np
from numba import jit
from typing import Tuple, Optional, Callable
from scipy.stats import norm
from config import (
    OptionContract, SimulationConfig, OptionType, ExerciseStyle,
    ConstantVolParams, HestonParams, SABRParams, BenchmarkResults
)
from sde_models import simulate_gbm, simulate_heston, simulate_sabr


def european_payoff(paths: np.ndarray, contract: OptionContract) -> np.ndarray:
    """Calculate European option payoff from terminal prices."""
    terminal_prices = paths[:, -1]
    K = contract.strike_price
    
    if contract.option_type == OptionType.CALL:
        return np.maximum(terminal_prices - K, 0)
    else:
        return np.maximum(K - terminal_prices, 0)


def price_european_mc(
    contract: OptionContract,
    paths: np.ndarray,
    config: SimulationConfig
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Price European option using Monte Carlo simulation.
    
    Returns:
        - Option price
        - Standard error
        - 95% confidence interval
    """
    payoffs = european_payoff(paths, contract)
    
    # Discount to present value
    discount = np.exp(-contract.risk_free_rate * contract.time_to_maturity)
    discounted_payoffs = discount * payoffs
    
    # Statistics
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(len(payoffs))
    ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)
    
    return price, std_error, ci_95


@jit(nopython=True)
def _laguerre_polynomials(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """Generate Laguerre polynomial basis functions for regression."""
    n = len(x)
    basis = np.zeros((n, degree + 1))
    
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = 1.0 - x
    if degree >= 2:
        basis[:, 2] = 0.5 * (x**2 - 4*x + 2)
    if degree >= 3:
        basis[:, 3] = (1/6) * (-x**3 + 9*x**2 - 18*x + 6)
    
    return basis


def price_american_lsm(
    contract: OptionContract,
    paths: np.ndarray,
    config: SimulationConfig,
    poly_degree: int = 3
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Price American option using Longstaff-Schwartz Monte Carlo (LSM).
    
    Uses regression to estimate continuation value at each exercise point.
    
    Args:
        contract: Option contract specification
        paths: Simulated price paths (num_paths x num_steps+1)
        config: Simulation configuration
        poly_degree: Degree of polynomial basis for regression
    
    Returns:
        - Option price
        - Standard error  
        - 95% confidence interval
    """
    num_paths, num_steps_plus_1 = paths.shape
    num_steps = num_steps_plus_1 - 1
    dt = contract.time_to_maturity / num_steps
    discount_factor = np.exp(-contract.risk_free_rate * dt)
    
    K = contract.strike_price
    is_call = contract.option_type == OptionType.CALL
    
    # Initialize cash flow matrix
    cash_flows = np.zeros((num_paths, num_steps + 1))
    
    # Terminal payoff
    if is_call:
        cash_flows[:, -1] = np.maximum(paths[:, -1] - K, 0)
    else:
        cash_flows[:, -1] = np.maximum(K - paths[:, -1], 0)
    
    # Backward induction
    for t in range(num_steps - 1, 0, -1):
        # Current stock prices
        S_t = paths[:, t]
        
        # Immediate exercise value
        if is_call:
            exercise_value = np.maximum(S_t - K, 0)
        else:
            exercise_value = np.maximum(K - S_t, 0)
        
        # Only consider in-the-money paths for regression
        itm_mask = exercise_value > 0
        
        if np.sum(itm_mask) < poly_degree + 2:
            continue
        
        # Future discounted cash flows for ITM paths
        future_cf = np.zeros(num_paths)
        for s in range(t + 1, num_steps + 1):
            future_cf += cash_flows[:, s] * (discount_factor ** (s - t))
        
        # Regression for continuation value
        X = S_t[itm_mask]
        Y = future_cf[itm_mask]
        
        # Normalize for numerical stability
        X_mean, X_std = np.mean(X), np.std(X)
        X_norm = (X - X_mean) / (X_std + 1e-10)
        
        # Build polynomial features
        basis = np.column_stack([X_norm**i for i in range(poly_degree + 1)])
        
        # OLS regression
        try:
            coeffs = np.linalg.lstsq(basis, Y, rcond=None)[0]
            continuation_value = basis @ coeffs
        except np.linalg.LinAlgError:
            continuation_value = Y
        
        # Exercise decision: exercise if immediate > continuation
        exercise_mask = np.zeros(num_paths, dtype=bool)
        exercise_indices = np.where(itm_mask)[0]
        for idx, ex_idx in enumerate(exercise_indices):
            if exercise_value[ex_idx] > continuation_value[idx]:
                exercise_mask[ex_idx] = True
        
        # Update cash flows
        for i in range(num_paths):
            if exercise_mask[i]:
                cash_flows[i, t] = exercise_value[i]
                cash_flows[i, t+1:] = 0  # Zero out future cash flows
    
    # Calculate present value
    discount_factors = np.array([discount_factor ** t for t in range(num_steps + 1)])
    pv_cash_flows = np.sum(cash_flows * discount_factors, axis=1)
    
    price = np.mean(pv_cash_flows)
    std_error = np.std(pv_cash_flows) / np.sqrt(num_paths)
    ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)
    
    return price, std_error, ci_95


def price_option(
    contract: OptionContract,
    vol_params,
    config: SimulationConfig
) -> Tuple[float, float, Tuple[float, float], np.ndarray]:
    """
    Main pricing function that routes to appropriate model and pricing method.
    
    Returns:
        - Option price
        - Standard error
        - 95% confidence interval
        - Simulated paths
    """
    # Simulate paths based on volatility model
    if isinstance(vol_params, ConstantVolParams):
        paths = simulate_gbm(contract, vol_params, config)
    elif isinstance(vol_params, HestonParams):
        paths, _ = simulate_heston(contract, vol_params, config)
    elif isinstance(vol_params, SABRParams):
        paths, _ = simulate_sabr(contract, vol_params, config)
    else:
        raise ValueError(f"Unknown volatility model: {type(vol_params)}")
    
    # Price based on exercise style
    if contract.exercise_style == ExerciseStyle.EUROPEAN:
        price, std_err, ci = price_european_mc(contract, paths, config)
    else:
        price, std_err, ci = price_american_lsm(contract, paths, config)
    
    return price, std_err, ci, paths


class MonteCarloEngine:
    """
    High-level Monte Carlo pricing engine with caching and benchmarking.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self._paths_cache = {}
    
    def price(
        self,
        contract: OptionContract,
        vol_params,
        use_cache: bool = True
    ) -> Tuple[float, float, Tuple[float, float]]:
        """Price an option contract."""
        cache_key = (
            id(contract), id(vol_params), 
            self.config.num_paths, self.config.seed
        )
        
        if use_cache and cache_key in self._paths_cache:
            paths = self._paths_cache[cache_key]
            if contract.exercise_style == ExerciseStyle.EUROPEAN:
                return price_european_mc(contract, paths, self.config)
            else:
                return price_american_lsm(contract, paths, self.config)
        
        price, std_err, ci, paths = price_option(
            contract, vol_params, self.config
        )
        
        if use_cache:
            self._paths_cache[cache_key] = paths
        
        return price, std_err, ci
    
    def clear_cache(self):
        """Clear the paths cache."""
        self._paths_cache.clear()
    
    def convergence_analysis(
        self,
        contract: OptionContract,
        vol_params,
        path_counts: list = None
    ) -> dict:
        """
        Analyze convergence behavior across different path counts.
        """
        if path_counts is None:
            path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
        
        results = {
            'path_counts': path_counts,
            'prices': [],
            'std_errors': [],
            'ci_widths': []
        }
        
        original_paths = self.config.num_paths
        
        for n in path_counts:
            self.config.num_paths = n
            price, std_err, ci = self.price(contract, vol_params, use_cache=False)
            
            results['prices'].append(price)
            results['std_errors'].append(std_err)
            results['ci_widths'].append(ci[1] - ci[0])
        
        self.config.num_paths = original_paths
        return results
