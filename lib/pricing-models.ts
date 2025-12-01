/**
 * Pricing models for options - TypeScript implementation
 */

function normCDF(x: number): number {
  const a1 = 0.254829592
  const a2 = -0.284496736
  const a3 = 1.421413741
  const a4 = -1.453152027
  const a5 = 1.061405429
  const p = 0.3275911

  const sign = x < 0 ? -1 : 1
  x = Math.abs(x)

  const t = 1.0 / (1.0 + p * x)
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp((-x * x) / 2)

  return 0.5 * (1.0 + sign * y)
}

function normPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
}

interface ContractParams {
  spotPrice: number
  strikePrice: number
  timeToMaturity: number
  riskFreeRate: number
  dividendYield: number
  volatility: number
  optionType: "call" | "put"
  exerciseStyle?: "european" | "american"
}

interface VolParams {
  model: "constant" | "heston"
  v0: number
  kappa: number
  theta: number
  xi: number
  rho: number
}

export function blackScholesPrice(contract: ContractParams): number {
  const {
    spotPrice: S,
    strikePrice: K,
    timeToMaturity: T,
    riskFreeRate: r,
    dividendYield: q,
    volatility: sigma,
    optionType,
  } = contract

  if (T <= 0) {
    return optionType === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0)
  }

  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T))
  const d2 = d1 - sigma * Math.sqrt(T)

  if (optionType === "call") {
    return S * Math.exp(-q * T) * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2)
  } else {
    return K * Math.exp(-r * T) * normCDF(-d2) - S * Math.exp(-q * T) * normCDF(-d1)
  }
}

export function blackScholesGreeks(contract: ContractParams): {
  delta: number
  gamma: number
  theta: number
  vega: number
  rho: number
} {
  const {
    spotPrice: S,
    strikePrice: K,
    timeToMaturity: T,
    riskFreeRate: r,
    dividendYield: q,
    volatility: sigma,
    optionType,
  } = contract

  const sqrtT = Math.sqrt(T)
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
  const d2 = d1 - sigma * sqrtT

  const pdfD1 = normPDF(d1)
  const isCall = optionType === "call"

  let delta: number
  if (isCall) {
    delta = Math.exp(-q * T) * normCDF(d1)
  } else {
    delta = Math.exp(-q * T) * (normCDF(d1) - 1)
  }

  const gamma = (Math.exp(-q * T) * pdfD1) / (S * sigma * sqrtT)

  let theta: number
  if (isCall) {
    theta =
      (-S * pdfD1 * sigma * Math.exp(-q * T)) / (2 * sqrtT) -
      r * K * Math.exp(-r * T) * normCDF(d2) +
      q * S * Math.exp(-q * T) * normCDF(d1)
  } else {
    theta =
      (-S * pdfD1 * sigma * Math.exp(-q * T)) / (2 * sqrtT) +
      r * K * Math.exp(-r * T) * normCDF(-d2) -
      q * S * Math.exp(-q * T) * normCDF(-d1)
  }
  theta = theta / 365

  const vega = (S * Math.exp(-q * T) * sqrtT * pdfD1) / 100

  let rho: number
  if (isCall) {
    rho = (K * T * Math.exp(-r * T) * normCDF(d2)) / 100
  } else {
    rho = (-K * T * Math.exp(-r * T) * normCDF(-d2)) / 100
  }

  return { delta, gamma, theta, vega, rho }
}

export function binomialTreePrice(contract: ContractParams, numSteps = 500): number {
  const {
    spotPrice: S,
    strikePrice: K,
    timeToMaturity: T,
    riskFreeRate: r,
    dividendYield: q,
    volatility: sigma,
    optionType,
    exerciseStyle,
  } = contract

  const dt = T / numSteps
  const u = Math.exp(sigma * Math.sqrt(dt))
  const d = 1 / u
  const p = (Math.exp((r - q) * dt) - d) / (u - d)
  const discount = Math.exp(-r * dt)

  const values: number[] = []

  for (let i = 0; i <= numSteps; i++) {
    const price = S * Math.pow(u, numSteps - i) * Math.pow(d, i)
    if (optionType === "call") {
      values.push(Math.max(price - K, 0))
    } else {
      values.push(Math.max(K - price, 0))
    }
  }

  const isAmerican = exerciseStyle === "american"

  for (let step = numSteps - 1; step >= 0; step--) {
    for (let i = 0; i <= step; i++) {
      const price = S * Math.pow(u, step - i) * Math.pow(d, i)
      const continuationValue = discount * (p * values[i] + (1 - p) * values[i + 1])

      if (isAmerican) {
        const exerciseValue = optionType === "call" ? Math.max(price - K, 0) : Math.max(K - price, 0)
        values[i] = Math.max(continuationValue, exerciseValue)
      } else {
        values[i] = continuationValue
      }
    }
  }

  return values[0]
}

function seededRandom(seed: number) {
  let state = seed
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff
    return state / 0x7fffffff
  }
}

function boxMuller(rand: () => number): number {
  const u1 = rand()
  const u2 = rand()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

export function monteCarloPrice(
  contract: ContractParams,
  volParams: VolParams,
  numPaths = 100000,
): { price: number; stdError: number; ci: [number, number] } {
  const {
    spotPrice: S,
    strikePrice: K,
    timeToMaturity: T,
    riskFreeRate: r,
    dividendYield: q,
    volatility: sigma,
    optionType,
  } = contract

  const rand = seededRandom(42)
  const numSteps = 252
  const dt = T / numSteps

  const payoffs: number[] = []
  const halfPaths = Math.floor(numPaths / 2)

  for (let p = 0; p < halfPaths; p++) {
    let S1 = S
    let S2 = S

    if (volParams.model === "constant") {
      const drift = (r - q - 0.5 * sigma * sigma) * dt
      const diffusion = sigma * Math.sqrt(dt)

      for (let step = 0; step < numSteps; step++) {
        const z = boxMuller(rand)
        S1 *= Math.exp(drift + diffusion * z)
        S2 *= Math.exp(drift - diffusion * z)
      }
    } else {
      let v1 = volParams.v0
      let v2 = volParams.v0

      for (let step = 0; step < numSteps; step++) {
        const z1 = boxMuller(rand)
        const z2 = boxMuller(rand)
        const w1 = z1
        const w2 = volParams.rho * z1 + Math.sqrt(1 - volParams.rho * volParams.rho) * z2

        const sqrtV1 = Math.sqrt(Math.max(v1, 0))
        S1 *= Math.exp((r - q - 0.5 * v1) * dt + sqrtV1 * Math.sqrt(dt) * w1)
        v1 = Math.abs(v1 + volParams.kappa * (volParams.theta - v1) * dt + volParams.xi * sqrtV1 * Math.sqrt(dt) * w2)

        const sqrtV2 = Math.sqrt(Math.max(v2, 0))
        S2 *= Math.exp((r - q - 0.5 * v2) * dt + sqrtV2 * Math.sqrt(dt) * -w1)
        v2 = Math.abs(v2 + volParams.kappa * (volParams.theta - v2) * dt + volParams.xi * sqrtV2 * Math.sqrt(dt) * -w2)
      }
    }

    const payoff1 = optionType === "call" ? Math.max(S1 - K, 0) : Math.max(K - S1, 0)
    const payoff2 = optionType === "call" ? Math.max(S2 - K, 0) : Math.max(K - S2, 0)

    payoffs.push(payoff1, payoff2)
  }

  const discount = Math.exp(-r * T)
  const discountedPayoffs = payoffs.map((p) => p * discount)

  const mean = discountedPayoffs.reduce((a, b) => a + b, 0) / discountedPayoffs.length
  const variance = discountedPayoffs.reduce((sum, p) => sum + (p - mean) ** 2, 0) / (discountedPayoffs.length - 1)
  const stdError = Math.sqrt(variance / discountedPayoffs.length)

  return {
    price: mean,
    stdError,
    ci: [mean - 1.96 * stdError, mean + 1.96 * stdError],
  }
}

export function hestonPrice(contract: ContractParams, volParams: VolParams): number {
  const mc = monteCarloPrice(contract, volParams, 50000)
  return mc.price
}
