/**
 * Greeks surface calculation utilities
 */

import { blackScholesPrice, blackScholesGreeks } from "./pricing-models"

interface GreeksSurfaceParams {
  strikePrice: number
  timeToMaturity: number
  riskFreeRate: number
  dividendYield: number
  optionType: "call" | "put"
  spotRange: [number, number]
  volRange: [number, number]
  resolution: number
}

export function calculateGreeksSurface(params: GreeksSurfaceParams) {
  const { strikePrice, timeToMaturity, riskFreeRate, dividendYield, optionType, spotRange, volRange, resolution } =
    params

  const spots: number[] = []
  const vols: number[] = []

  for (let i = 0; i < resolution; i++) {
    spots.push(spotRange[0] + (spotRange[1] - spotRange[0]) * (i / (resolution - 1)))
    vols.push(volRange[0] + (volRange[1] - volRange[0]) * (i / (resolution - 1)))
  }

  const price: number[][] = []
  const delta: number[][] = []
  const gamma: number[][] = []
  const theta: number[][] = []
  const vega: number[][] = []

  for (let vi = 0; vi < resolution; vi++) {
    const priceRow: number[] = []
    const deltaRow: number[] = []
    const gammaRow: number[] = []
    const thetaRow: number[] = []
    const vegaRow: number[] = []

    for (let si = 0; si < resolution; si++) {
      const contract = {
        spotPrice: spots[si],
        strikePrice,
        timeToMaturity,
        riskFreeRate,
        dividendYield,
        volatility: vols[vi],
        optionType,
      }

      const p = blackScholesPrice(contract)
      const g = blackScholesGreeks(contract)

      priceRow.push(p)
      deltaRow.push(g.delta)
      gammaRow.push(g.gamma)
      thetaRow.push(g.theta)
      vegaRow.push(g.vega)
    }

    price.push(priceRow)
    delta.push(deltaRow)
    gamma.push(gammaRow)
    theta.push(thetaRow)
    vega.push(vegaRow)
  }

  return { spots, vols, price, delta, gamma, theta, vega }
}
