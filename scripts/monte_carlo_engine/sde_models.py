"""
Stochastic Differential Equation (SDE) models for asset price simulation.
Implements Geometric Brownian Motion, Heston, and SABR models with optimized vectorization.
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional
from config import (
    SimulationConfig, OptionContract, 
    ConstantVolParams, HestonParams, SABRParams
)


def generate_correlated_normals(
    num_paths: int, 
    num_steps: int, 
    rho: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated standard normal random variables."""
    if seed is not None:
        np.random.seed(seed)
    
    z1 = np.random.standard_normal((num_paths, num_steps))
    z2 = np.random.standard_normal((num_paths, num_steps))
    
    # Cholesky decomposition for correlation
    w1 = z1
    w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    
    return w1, w2


def simulate_gbm(
    contract: OptionContract,
    vol_params: ConstantVolParams,
    config: SimulationConfig
) -> np.ndarray:
    """
    Simulate asset paths using Geometric Brownian Motion (GBM).
    
    dS = (r - q) * S * dt + sigma * S * dW
    
    Uses exact solution for efficiency:
    S(t) = S(0) * exp((r - q - 0.5*sigma^2)*t + sigma*W(t))
    """
    if config.seed is not None:
        np.random.seed(config.seed)
    
    S0 = contract.spot_price
    r = contract.risk_free_rate
    q = contract.dividend_yield
    T = contract.time_to_maturity
    sigma = vol_params.sigma
    
    dt = T / config.num_steps
    num_paths = config.num_paths
    
    if config.antithetic:
        half_paths = num_paths // 2
        z = np.random.standard_normal((half_paths, config.num_steps))
        z = np.vstack([z, -z])  # Antithetic pairs
    else:
        z = np.random.standard_normal((num_paths, config.num_steps))
    
    # Drift and diffusion
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    
    # Cumulative sum for path construction
    log_returns = drift + diffusion
    log_paths = np.cumsum(log_returns, axis=1)
    
    # Prepend initial price
    paths = S0 * np.exp(np.hstack([np.zeros((num_paths, 1)), log_paths]))
    
    return paths


@jit(nopython=True, parallel=True)
def _heston_euler_kernel(
    S0: float, v0: float, r: float, q: float, T: float,
    kappa: float, theta: float, xi: float, rho: float,
    num_paths: int, num_steps: int,
    z1: np.ndarray, z2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled Heston model simulation using Euler-Maruyama scheme.
    
    dS = (r - q) * S * dt + sqrt(v) * S * dW1
    dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
    
    where dW1 and dW2 are correlated with correlation rho.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    
    S = np.zeros((num_paths, num_steps + 1))
    v = np.zeros((num_paths, num_steps + 1))
    
    for i in prange(num_paths):
        S[i, 0] = S0
        v[i, 0] = v0
        
        for j in range(num_steps):
            # Ensure variance stays positive (full truncation scheme)
            v_curr = max(v[i, j], 0.0)
            sqrt_v = np.sqrt(v_curr)
            
            # Asset price update
            S[i, j + 1] = S[i, j] * np.exp(
                (r - q - 0.5 * v_curr) * dt + sqrt_v * sqrt_dt * z1[i, j]
            )
            
            # Variance update with reflection scheme
            v_drift = kappa * (theta - v_curr) * dt
            v_diffusion = xi * sqrt_v * sqrt_dt * z2[i, j]
            v[i, j + 1] = abs(v[i, j] + v_drift + v_diffusion)
    
    return S, v


def simulate_heston(
    contract: OptionContract,
    heston_params: HestonParams,
    config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate asset paths using Heston stochastic volatility model.
    Returns both price paths and variance paths.
    """
    # Generate correlated random numbers
    w1, w2 = generate_correlated_normals(
        config.num_paths, 
        config.num_steps, 
        heston_params.rho,
        config.seed
    )
    
    if config.antithetic:
        half = config.num_paths // 2
        w1 = np.vstack([w1[:half], -w1[:half]])
        w2 = np.vstack([w2[:half], -w2[:half]])
    
    if config.use_jit:
        S, v = _heston_euler_kernel(
            contract.spot_price, heston_params.v0,
            contract.risk_free_rate, contract.dividend_yield,
            contract.time_to_maturity,
            heston_params.kappa, heston_params.theta,
            heston_params.xi, heston_params.rho,
            config.num_paths, config.num_steps,
            w1, w2
        )
    else:
        S, v = _heston_euler_python(
            contract, heston_params, config, w1, w2
        )
    
    return S, v


def _heston_euler_python(
    contract: OptionContract,
    heston_params: HestonParams,
    config: SimulationConfig,
    w1: np.ndarray,
    w2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python/NumPy implementation of Heston (fallback)."""
    dt = contract.time_to_maturity / config.num_steps
    sqrt_dt = np.sqrt(dt)
    
    S = np.zeros((config.num_paths, config.num_steps + 1))
    v = np.zeros((config.num_paths, config.num_steps + 1))
    
    S[:, 0] = contract.spot_price
    v[:, 0] = heston_params.v0
    
    r, q = contract.risk_free_rate, contract.dividend_yield
    kappa, theta = heston_params.kappa, heston_params.theta
    xi = heston_params.xi
    
    for j in range(config.num_steps):
        v_curr = np.maximum(v[:, j], 0)
        sqrt_v = np.sqrt(v_curr)
        
        S[:, j + 1] = S[:, j] * np.exp(
            (r - q - 0.5 * v_curr) * dt + sqrt_v * sqrt_dt * w1[:, j]
        )
        
        v_drift = kappa * (theta - v_curr) * dt
        v_diffusion = xi * sqrt_v * sqrt_dt * w2[:, j]
        v[:, j + 1] = np.abs(v[:, j] + v_drift + v_diffusion)
    
    return S, v


@jit(nopython=True, parallel=True)
def _sabr_kernel(
    F0: float, alpha0: float, beta: float, rho: float, nu: float,
    T: float, num_paths: int, num_steps: int,
    z1: np.ndarray, z2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled SABR model simulation.
    
    dF = alpha * F^beta * dW1
    d(alpha) = nu * alpha * dW2
    
    where dW1 and dW2 are correlated with correlation rho.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    
    F = np.zeros((num_paths, num_steps + 1))
    alpha = np.zeros((num_paths, num_steps + 1))
    
    for i in prange(num_paths):
        F[i, 0] = F0
        alpha[i, 0] = alpha0
        
        for j in range(num_steps):
            F_curr = max(F[i, j], 1e-10)
            alpha_curr = max(alpha[i, j], 1e-10)
            
            # Forward rate update
            F_beta = F_curr ** beta
            F[i, j + 1] = F[i, j] + alpha_curr * F_beta * sqrt_dt * z1[i, j]
            F[i, j + 1] = max(F[i, j + 1], 1e-10)
            
            # Volatility update (log-normal dynamics)
            alpha[i, j + 1] = alpha[i, j] * np.exp(
                -0.5 * nu**2 * dt + nu * sqrt_dt * z2[i, j]
            )
    
    return F, alpha


def simulate_sabr(
    contract: OptionContract,
    sabr_params: SABRParams,
    config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate forward rate paths using SABR model.
    Returns both forward paths and volatility paths.
    """
    w1, w2 = generate_correlated_normals(
        config.num_paths,
        config.num_steps,
        sabr_params.rho,
        config.seed
    )
    
    if config.antithetic:
        half = config.num_paths // 2
        w1 = np.vstack([w1[:half], -w1[:half]])
        w2 = np.vstack([w2[:half], -w2[:half]])
    
    F, alpha = _sabr_kernel(
        contract.spot_price, sabr_params.alpha,
        sabr_params.beta, sabr_params.rho, sabr_params.nu,
        contract.time_to_maturity,
        config.num_paths, config.num_steps,
        w1, w2
    )
    
    return F, alpha
