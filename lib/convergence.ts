/**
 * Convergence study utilities
 */

import { blackScholesPrice, monteCarloPrice } from "./pricing-models"

interface ConvergenceParams {
  spotPrice: number
  strikePrice: number
  timeToMaturity: number
  riskFreeRate: number
  volatility: number
  pathCounts: number[]
  numTrials: number
}

export function runConvergenceStudy(params: ConvergenceParams) {
  const { spotPrice, strikePrice, timeToMaturity, riskFreeRate, volatility, pathCounts, numTrials } = params

  const contract = {
    spotPrice,
    strikePrice,
    timeToMaturity,
    riskFreeRate,
    dividendYield: 0.02,
    volatility,
    optionType: "call" as const,
  }

  const volParams = {
    model: "constant" as const,
    v0: volatility ** 2,
    kappa: 2,
    theta: volatility ** 2,
    xi: 0.3,
    rho: -0.7,
  }

  const analyticalPrice = blackScholesPrice(contract)

  const meanPrices: number[] = []
  const rmse: number[] = []
  const stdErrors: number[] = []
  const theoreticalConvergence: number[] = []

  for (const numPaths of pathCounts) {
    const trialPrices: number[] = []
    const trialStdErrors: number[] = []

    for (let trial = 0; trial < numTrials; trial++) {
      const result = monteCarloPrice(contract, volParams, numPaths)
      trialPrices.push(result.price)
      trialStdErrors.push(result.stdError)
    }

    const meanPrice = trialPrices.reduce((a, b) => a + b, 0) / numTrials
    const errors = trialPrices.map((p) => (p - analyticalPrice) ** 2)
    const rmsError = Math.sqrt(errors.reduce((a, b) => a + b, 0) / numTrials)
    const avgStdError = trialStdErrors.reduce((a, b) => a + b, 0) / numTrials

    meanPrices.push(meanPrice)
    rmse.push(rmsError)
    stdErrors.push(avgStdError)
    theoreticalConvergence.push(1 / Math.sqrt(numPaths))
  }

  const scale = rmse[0] / theoreticalConvergence[0]
  const scaledTheoretical = theoreticalConvergence.map((t) => t * scale)

  return {
    pathCounts,
    meanPrices,
    rmse,
    stdErrors,
    theoreticalConvergence: scaledTheoretical,
    analyticalPrice,
    numTrials,
  }
}
