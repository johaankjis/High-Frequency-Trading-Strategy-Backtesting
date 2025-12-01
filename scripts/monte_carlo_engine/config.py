"""
Configuration settings for Monte Carlo simulation engine.
Centralized parameters for reproducibility and easy tuning.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class VolatilityModel(Enum):
    CONSTANT = "constant"
    HESTON = "heston"
    SABR = "sabr"


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation parameters."""
    num_paths: int = 100_000
    num_steps: int = 252  # Trading days in a year
    seed: Optional[int] = 42
    antithetic: bool = True  # Variance reduction
    use_jit: bool = True  # Enable Numba JIT compilation


@dataclass
class OptionContract:
    """Specification for an option contract."""
    spot_price: float  # Current asset price (S0)
    strike_price: float  # Strike price (K)
    time_to_maturity: float  # Time to expiry in years (T)
    risk_free_rate: float  # Risk-free interest rate (r)
    dividend_yield: float = 0.0  # Continuous dividend yield (q)
    option_type: OptionType = OptionType.CALL
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN


@dataclass
class ConstantVolParams:
    """Parameters for constant volatility (Black-Scholes) model."""
    sigma: float = 0.2  # Constant volatility


@dataclass
class HestonParams:
    """Parameters for Heston stochastic volatility model."""
    v0: float = 0.04  # Initial variance
    kappa: float = 2.0  # Mean reversion speed
    theta: float = 0.04  # Long-term variance
    xi: float = 0.3  # Volatility of volatility
    rho: float = -0.7  # Correlation between asset and variance
    
    def feller_condition(self) -> bool:
        """Check if Feller condition is satisfied (2*kappa*theta > xi^2)."""
        return 2 * self.kappa * self.theta > self.xi ** 2


@dataclass
class SABRParams:
    """Parameters for SABR stochastic volatility model."""
    alpha: float = 0.3  # Initial volatility
    beta: float = 0.5  # CEV exponent (0 to 1)
    rho: float = -0.3  # Correlation
    nu: float = 0.4  # Volatility of volatility


@dataclass
class BenchmarkResults:
    """Container for benchmark comparison results."""
    monte_carlo_price: float
    analytical_price: Optional[float]
    binomial_price: Optional[float]
    mc_std_error: float
    mc_confidence_interval: tuple
    absolute_error: Optional[float] = None
    relative_error: Optional[float] = None
    computation_time_ms: float = 0.0
    num_paths: int = 0
    
    def __post_init__(self):
        if self.analytical_price is not None:
            self.absolute_error = abs(self.monte_carlo_price - self.analytical_price)
            self.relative_error = self.absolute_error / self.analytical_price * 100


# Default configurations for quick testing
DEFAULT_OPTION = OptionContract(
    spot_price=100.0,
    strike_price=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    dividend_yield=0.02,
    option_type=OptionType.CALL,
    exercise_style=ExerciseStyle.EUROPEAN
)

DEFAULT_SIM_CONFIG = SimulationConfig(
    num_paths=100_000,
    num_steps=252,
    seed=42,
    antithetic=True,
    use_jit=True
)

DEFAULT_HESTON = HestonParams(
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    xi=0.3,
    rho=-0.7
)
