"""
Analytical pricing models for benchmarking.
Implements Black-Scholes, Binomial Tree, and Heston semi-analytical solutions.
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq
from typing import Tuple, Optional
from config import OptionContract, OptionType, ConstantVolParams, HestonParams


def black_scholes_price(
    contract: OptionContract,
    vol_params: ConstantVolParams
) -> float:
    """
    Calculate Black-Scholes option price.
    
    Call: S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
    Put:  K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)
    """
    S = contract.spot_price
    K = contract.strike_price
    T = contract.time_to_maturity
    r = contract.risk_free_rate
    q = contract.dividend_yield
    sigma = vol_params.sigma
    
    if T <= 0:
        if contract.option_type == OptionType.CALL:
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if contract.option_type == OptionType.CALL:
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price


def black_scholes_greeks(
    contract: OptionContract,
    vol_params: ConstantVolParams
) -> dict:
    """Calculate Black-Scholes Greeks."""
    S = contract.spot_price
    K = contract.strike_price
    T = contract.time_to_maturity
    r = contract.risk_free_rate
    q = contract.dividend_yield
    sigma = vol_params.sigma
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Standard normal PDF at d1
    pdf_d1 = norm.pdf(d1)
    
    is_call = contract.option_type == OptionType.CALL
    
    if is_call:
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * pdf_d1 * sigma * np.exp(-q * T) / (2 * sqrt_T)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (
            -S * pdf_d1 * sigma * np.exp(-q * T) / (2 * sqrt_T)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
    
    # Gamma (same for calls and puts)
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * sqrt_T)
    
    # Vega (same for calls and puts)
    vega = S * np.exp(-q * T) * sqrt_T * pdf_d1
    
    # Rho
    if is_call:
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,  # Daily theta
        'vega': vega / 100,    # Per 1% vol change
        'rho': rho / 100       # Per 1% rate change
    }


def implied_volatility(
    contract: OptionContract,
    market_price: float,
    initial_guess: float = 0.2,
    tol: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Brent's method.
    """
    def objective(sigma):
        vol_params = ConstantVolParams(sigma=sigma)
        return black_scholes_price(contract, vol_params) - market_price
    
    try:
        iv = brentq(objective, 0.001, 5.0, xtol=tol)
        return iv
    except ValueError:
        return np.nan


def binomial_tree_price(
    contract: OptionContract,
    vol_params: ConstantVolParams,
    num_steps: int = 500
) -> float:
    """
    Price option using Cox-Ross-Rubinstein binomial tree.
    Supports both European and American exercise.
    """
    S = contract.spot_price
    K = contract.strike_price
    T = contract.time_to_maturity
    r = contract.risk_free_rate
    q = contract.dividend_yield
    sigma = vol_params.sigma
    
    dt = T / num_steps
    
    # CRR parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    discount = np.exp(-r * dt)
    
    # Terminal stock prices
    stock_prices = S * (u ** np.arange(num_steps, -1, -1)) * (d ** np.arange(0, num_steps + 1))
    
    # Terminal option values
    if contract.option_type == OptionType.CALL:
        option_values = np.maximum(stock_prices - K, 0)
    else:
        option_values = np.maximum(K - stock_prices, 0)
    
    # Backward induction
    is_american = contract.exercise_style.value == "american"
    
    for step in range(num_steps - 1, -1, -1):
        stock_prices = S * (u ** np.arange(step, -1, -1)) * (d ** np.arange(0, step + 1))
        
        # Continuation value
        option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])
        
        # Early exercise for American options
        if is_american:
            if contract.option_type == OptionType.CALL:
                exercise_values = np.maximum(stock_prices - K, 0)
            else:
                exercise_values = np.maximum(K - stock_prices, 0)
            option_values = np.maximum(option_values, exercise_values)
    
    return option_values[0]


def heston_characteristic_function(
    u: complex,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float
) -> complex:
    """
    Heston model characteristic function for pricing.
    Used in semi-analytical pricing via Fourier inversion.
    """
    i = 1j
    
    d = np.sqrt((rho * xi * i * u - kappa)**2 + xi**2 * (i * u + u**2))
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)
    
    C = (r - q) * i * u * T + (kappa * theta / xi**2) * (
        (kappa - rho * xi * i * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    
    D = ((kappa - rho * xi * i * u - d) / xi**2) * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )
    
    return np.exp(C + D * v0)


def heston_call_price(
    contract: OptionContract,
    heston_params: HestonParams,
    integration_limit: float = 100
) -> float:
    """
    Price European call option under Heston model using Fourier inversion.
    Uses the Lewis (2000) formulation for numerical stability.
    """
    S = contract.spot_price
    K = contract.strike_price
    T = contract.time_to_maturity
    r = contract.risk_free_rate
    q = contract.dividend_yield
    
    v0 = heston_params.v0
    kappa = heston_params.kappa
    theta = heston_params.theta
    xi = heston_params.xi
    rho = heston_params.rho
    
    log_moneyness = np.log(S / K) + (r - q) * T
    
    def integrand(u):
        cf = heston_characteristic_function(
            u - 0.5j, T, r, q, v0, kappa, theta, xi, rho
        )
        return np.real(np.exp(-1j * u * log_moneyness) * cf / (u**2 + 0.25))
    
    integral, _ = quad(integrand, 0, integration_limit)
    
    call_price = S * np.exp(-q * T) - (np.sqrt(S * K) / np.pi) * np.exp(
        -0.5 * (r + q) * T
    ) * integral
    
    return max(call_price, 0)


def heston_put_price(
    contract: OptionContract,
    heston_params: HestonParams
) -> float:
    """Price European put using put-call parity."""
    # Temporarily change to call for pricing
    call_contract = OptionContract(
        spot_price=contract.spot_price,
        strike_price=contract.strike_price,
        time_to_maturity=contract.time_to_maturity,
        risk_free_rate=contract.risk_free_rate,
        dividend_yield=contract.dividend_yield,
        option_type=OptionType.CALL,
        exercise_style=contract.exercise_style
    )
    
    call_price = heston_call_price(call_contract, heston_params)
    
    # Put-call parity
    S = contract.spot_price
    K = contract.strike_price
    T = contract.time_to_maturity
    r = contract.risk_free_rate
    q = contract.dividend_yield
    
    put_price = call_price - S * np.exp(-q * T) + K * np.exp(-r * T)
    
    return max(put_price, 0)
