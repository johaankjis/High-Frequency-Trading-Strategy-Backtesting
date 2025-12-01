/**
 * Path simulation utilities
 */

interface GBMParams {
  spotPrice: number
  riskFreeRate: number
  dividendYield: number
  volatility: number
  timeToMaturity: number
  numPaths: number
  numSteps: number
  seed: number
}

interface HestonSimParams {
  spotPrice: number
  riskFreeRate: number
  dividendYield: number
  v0: number
  kappa: number
  theta: number
  xi: number
  rho: number
  timeToMaturity: number
  numPaths: number
  numSteps: number
  seed: number
}

function seededRandom(seed: number) {
  let state = seed
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff
    return state / 0x7fffffff
  }
}

function boxMuller(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-10)
  const u2 = rand()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

export function simulateGBM(params: GBMParams) {
  const {
    spotPrice: S0,
    riskFreeRate: r,
    dividendYield: q,
    volatility: sigma,
    timeToMaturity: T,
    numPaths,
    numSteps,
    seed,
  } = params

  const rand = seededRandom(seed)
  const dt = T / numSteps
  const drift = (r - q - 0.5 * sigma * sigma) * dt
  const diffusion = sigma * Math.sqrt(dt)

  const paths: number[][] = []
  const halfPaths = Math.floor(numPaths / 2)

  for (let p = 0; p < halfPaths; p++) {
    const path1: number[] = [S0]
    const path2: number[] = [S0]
    let S1 = S0
    let S2 = S0

    for (let step = 0; step < numSteps; step++) {
      const z = boxMuller(rand)
      S1 *= Math.exp(drift + diffusion * z)
      S2 *= Math.exp(drift - diffusion * z)
      path1.push(S1)
      path2.push(S2)
    }

    paths.push(path1, path2)
  }

  const meanPath: number[] = []
  const upperBand: number[] = []
  const lowerBand: number[] = []

  for (let step = 0; step <= numSteps; step++) {
    const values = paths.map((p) => p[step])
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const std = Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length)

    meanPath.push(mean)
    upperBand.push(mean + 2 * std)
    lowerBand.push(mean - 2 * std)
  }

  return { paths, meanPath, upperBand, lowerBand, variancePaths: null }
}

export function simulateHeston(params: HestonSimParams) {
  const {
    spotPrice: S0,
    riskFreeRate: r,
    dividendYield: q,
    v0,
    kappa,
    theta,
    xi,
    rho,
    timeToMaturity: T,
    numPaths,
    numSteps,
    seed,
  } = params

  const rand = seededRandom(seed)
  const dt = T / numSteps
  const sqrtDt = Math.sqrt(dt)

  const paths: number[][] = []
  const variancePaths: number[][] = []

  for (let p = 0; p < numPaths; p++) {
    const path: number[] = [S0]
    const varPath: number[] = [v0]
    let S = S0
    let v = v0

    for (let step = 0; step < numSteps; step++) {
      const z1 = boxMuller(rand)
      const z2 = boxMuller(rand)
      const w1 = z1
      const w2 = rho * z1 + Math.sqrt(1 - rho * rho) * z2

      const sqrtV = Math.sqrt(Math.max(v, 0))

      S *= Math.exp((r - q - 0.5 * v) * dt + sqrtV * sqrtDt * w1)
      v = Math.abs(v + kappa * (theta - v) * dt + xi * sqrtV * sqrtDt * w2)

      path.push(S)
      varPath.push(v)
    }

    paths.push(path)
    variancePaths.push(varPath)
  }

  const meanPath: number[] = []
  const upperBand: number[] = []
  const lowerBand: number[] = []

  for (let step = 0; step <= numSteps; step++) {
    const values = paths.map((p) => p[step])
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const std = Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length)

    meanPath.push(mean)
    upperBand.push(mean + 2 * std)
    lowerBand.push(mean - 2 * std)
  }

  return { paths, meanPath, upperBand, lowerBand, variancePaths }
}
