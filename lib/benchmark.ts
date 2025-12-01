/**
 * Benchmarking utilities
 */

import { blackScholesPrice, binomialTreePrice, monteCarloPrice } from "./pricing-models"

export interface BenchmarkResult {
  name: string
  bsPrice: number
  binomialPrice: number
  mcPrice: number
  stdError: number
  ci: [number, number]
  absoluteError: number
  relativeError: number
  timeMs: number
}

interface BenchmarkParams {
  name: string
  spot: number
  strike: number
  type: "call" | "put"
}

export function runBenchmarkSuite(params: BenchmarkParams): BenchmarkResult {
  const contract = {
    spotPrice: params.spot,
    strikePrice: params.strike,
    timeToMaturity: 1,
    riskFreeRate: 0.05,
    dividendYield: 0.02,
    volatility: 0.2,
    optionType: params.type,
  }

  const volParams = {
    model: "constant" as const,
    v0: 0.04,
    kappa: 2,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
  }

  const bsPrice = blackScholesPrice(contract)
  const binomialPrice = binomialTreePrice(contract, 500)

  const start = performance.now()
  const mcResult = monteCarloPrice(contract, volParams, 100000)
  const timeMs = performance.now() - start

  return {
    name: params.name,
    bsPrice,
    binomialPrice,
    mcPrice: mcResult.price,
    stdError: mcResult.stdError,
    ci: mcResult.ci,
    absoluteError: Math.abs(mcResult.price - bsPrice),
    relativeError: (Math.abs(mcResult.price - bsPrice) / bsPrice) * 100,
    timeMs,
  }
}
