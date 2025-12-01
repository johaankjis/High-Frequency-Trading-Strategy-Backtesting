# Monte Carlo Options Pricing Engine

A high-performance, full-stack application for pricing European and American options using Monte Carlo simulation with support for multiple volatility models including Black-Scholes (constant volatility), Heston, and SABR stochastic volatility models.

## Features

### Pricing Models
- **Black-Scholes Model** - Analytical pricing with closed-form solutions for European options
- **Binomial Tree** - Cox-Ross-Rubinstein model for American and European options
- **Monte Carlo Simulation** - Path-based pricing with variance reduction techniques
- **Heston Stochastic Volatility** - Semi-analytical and Monte Carlo pricing with mean-reverting variance
- **SABR Model** - Stochastic Alpha Beta Rho volatility model

### Greeks Calculation
Full Greeks computation including:
- **Delta (Δ)** - Price sensitivity to underlying
- **Gamma (Γ)** - Delta sensitivity to underlying
- **Theta (Θ)** - Time decay
- **Vega (ν)** - Volatility sensitivity
- **Rho (ρ)** - Interest rate sensitivity

### Variance Reduction
- **Antithetic Variates** - Reduces Monte Carlo variance by ~2x
- **JIT Compilation** - Numba-accelerated simulations for >1M paths/second throughput

### Interactive Dashboard
- Real-time option pricing with adjustable parameters
- Price path visualization
- Greeks surface plots
- Convergence analysis
- Benchmark comparisons

## Tech Stack

### Frontend
- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS 4** - Utility-first CSS framework
- **Radix UI** - Accessible component primitives
- **Recharts** - Charting library for visualizations

### Backend (Python Engine)
- **NumPy** - Numerical computing
- **Numba** - JIT compilation for performance
- **SciPy** - Scientific computing and statistical functions

## Project Structure

```
├── app/                          # Next.js App Router
│   ├── globals.css               # Global styles
│   ├── layout.tsx                # Root layout
│   └── page.tsx                  # Main page
├── components/                   # React components
│   ├── benchmark-results.tsx     # Benchmark comparison display
│   ├── convergence-chart.tsx     # Monte Carlo convergence visualization
│   ├── greeks-surface.tsx        # 3D Greeks surface plots
│   ├── header.tsx                # Application header
│   ├── option-pricer.tsx         # Main pricing interface
│   ├── price-paths-chart.tsx     # Simulated price paths
│   ├── pricing-dashboard.tsx     # Dashboard container
│   ├── theme-provider.tsx        # Theme management
│   └── ui/                       # Shadcn UI components
├── lib/                          # TypeScript utilities
│   ├── benchmark.ts              # Benchmarking utilities
│   ├── convergence.ts            # Convergence study functions
│   ├── greeks.ts                 # Greeks surface calculations
│   ├── pricing-models.ts         # Option pricing implementations
│   ├── simulation.ts             # Path simulation utilities
│   └── utils.ts                  # Helper functions
├── scripts/                      # Python Monte Carlo engine
│   └── monte_carlo_engine/
│       ├── analytical_models.py  # Black-Scholes, Binomial, Heston analytical
│       ├── benchmarking.py       # Performance benchmarks
│       ├── config.py             # Configuration and data classes
│       ├── main.py               # Entry point and demos
│       ├── pricing_engine.py     # Monte Carlo pricing engine
│       ├── sde_models.py         # Stochastic differential equations
│       └── visualization.py      # Plotting utilities
├── styles/                       # Additional styles
├── hooks/                        # Custom React hooks
├── public/                       # Static assets
├── package.json                  # Node.js dependencies
├── tsconfig.json                 # TypeScript configuration
├── next.config.mjs               # Next.js configuration
├── postcss.config.mjs            # PostCSS configuration
└── components.json               # Shadcn UI configuration
```

## Installation

### Prerequisites
- Node.js 18+ and pnpm
- Python 3.9+ (for Monte Carlo engine)

### Frontend Setup

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start
```

### Python Engine Setup

```bash
cd scripts/monte_carlo_engine

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy numba scipy

# Run the engine demo
python main.py
```

## Usage

### Web Interface

1. Start the development server: `pnpm dev`
2. Open http://localhost:3000
3. Use the interactive dashboard to:
   - Configure option contract parameters (spot, strike, maturity, rates)
   - Select volatility model (constant or Heston)
   - Run Monte Carlo simulations
   - View pricing results and Greeks
   - Analyze convergence behavior

### Python API

```python
from config import OptionContract, SimulationConfig, OptionType, ExerciseStyle, ConstantVolParams, HestonParams
from pricing_engine import price_option
from analytical_models import black_scholes_price, black_scholes_greeks

# Define an option contract
contract = OptionContract(
    spot_price=100.0,
    strike_price=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    dividend_yield=0.02,
    option_type=OptionType.CALL,
    exercise_style=ExerciseStyle.EUROPEAN
)

# Constant volatility (Black-Scholes)
vol_params = ConstantVolParams(sigma=0.20)

# Analytical Black-Scholes price
bs_price = black_scholes_price(contract, vol_params)
greeks = black_scholes_greeks(contract, vol_params)

# Monte Carlo simulation
config = SimulationConfig(num_paths=100_000, num_steps=252, seed=42)
mc_price, std_error, confidence_interval, paths = price_option(contract, vol_params, config)

# Heston stochastic volatility
heston_params = HestonParams(
    v0=0.04,      # Initial variance
    kappa=2.0,    # Mean reversion speed
    theta=0.04,   # Long-term variance
    xi=0.3,       # Vol of vol
    rho=-0.7      # Correlation
)
heston_price, heston_std, heston_ci, _ = price_option(contract, heston_params, config)
```

### TypeScript API

```typescript
import { blackScholesPrice, blackScholesGreeks, monteCarloPrice } from '@/lib/pricing-models';

const contract = {
  spotPrice: 100,
  strikePrice: 100,
  timeToMaturity: 1,
  riskFreeRate: 0.05,
  dividendYield: 0.02,
  volatility: 0.2,
  optionType: 'call' as const,
};

// Analytical pricing
const bsPrice = blackScholesPrice(contract);
const greeks = blackScholesGreeks(contract);

// Monte Carlo pricing
const volParams = {
  model: 'constant' as const,
  v0: 0.04,
  kappa: 2,
  theta: 0.04,
  xi: 0.3,
  rho: -0.7,
};

const mcResult = monteCarloPrice(contract, volParams, 100000);
console.log(`Price: $${mcResult.price.toFixed(4)}`);
console.log(`95% CI: [$${mcResult.ci[0].toFixed(4)}, $${mcResult.ci[1].toFixed(4)}]`);
```

## Mathematical Models

### Black-Scholes Formula

For a European call option:
```
C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)
```

Where:
- `d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)`
- `d₂ = d₁ - σ√T`
- `N(x)` is the cumulative standard normal distribution

### Heston Model

The Heston model describes asset dynamics with stochastic variance:
```
dS = (r - q)S dt + √v S dW₁
dv = κ(θ - v) dt + ξ√v dW₂
```

Where:
- `κ` - Mean reversion speed
- `θ` - Long-term variance
- `ξ` - Volatility of volatility
- `ρ` - Correlation between `dW₁` and `dW₂`

### Monte Carlo Convergence

The standard error decreases at rate O(1/√N) where N is the number of paths:
```
SE = σ / √N
```

Antithetic variates reduce variance by using negatively correlated pairs.

## Performance

Typical performance on modern hardware:
- **Black-Scholes analytical**: < 1ms
- **Binomial tree (500 steps)**: ~10ms
- **Monte Carlo (100K paths)**: ~50ms (TypeScript), ~20ms (Python with Numba)
- **Heston Monte Carlo (100K paths)**: ~100ms

## License

This project is provided for educational and research purposes.
