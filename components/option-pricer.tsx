"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  blackScholesPrice,
  blackScholesGreeks,
  binomialTreePrice,
  monteCarloPrice,
  hestonPrice,
} from "@/lib/pricing-models"

interface ContractParams {
  spotPrice: number
  strikePrice: number
  timeToMaturity: number
  riskFreeRate: number
  dividendYield: number
  volatility: number
  optionType: "call" | "put"
  exerciseStyle: "european" | "american"
}

interface VolParams {
  model: "constant" | "heston"
  v0: number
  kappa: number
  theta: number
  xi: number
  rho: number
}

export function OptionPricer() {
  const [contract, setContract] = useState<ContractParams>({
    spotPrice: 100,
    strikePrice: 100,
    timeToMaturity: 1,
    riskFreeRate: 0.05,
    dividendYield: 0.02,
    volatility: 0.2,
    optionType: "call",
    exerciseStyle: "european",
  })

  const [volParams, setVolParams] = useState<VolParams>({
    model: "constant",
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
  })

  const [numPaths, setNumPaths] = useState(100000)
  const [isCalculating, setIsCalculating] = useState(false)
  const [mcResult, setMcResult] = useState<{ price: number; stdError: number; ci: [number, number] } | null>(null)

  const analyticalResults = useMemo(() => {
    const bsPrice = blackScholesPrice(contract)
    const greeks = blackScholesGreeks(contract)
    const binomial = binomialTreePrice(contract, 500)
    const heston = volParams.model === "heston" ? hestonPrice(contract, volParams) : null

    return { bsPrice, greeks, binomial, heston }
  }, [contract, volParams])

  const handleCalculate = async () => {
    setIsCalculating(true)
    await new Promise((resolve) => setTimeout(resolve, 500))
    const result = monteCarloPrice(contract, volParams, numPaths)
    setMcResult(result)
    setIsCalculating(false)
  }

  const moneyness = contract.spotPrice / contract.strikePrice
  const moneynessLabel = moneyness > 1.02 ? "ITM" : moneyness < 0.98 ? "OTM" : "ATM"

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <Card className="lg:col-span-1">
        <CardHeader>
          <CardTitle>Contract Parameters</CardTitle>
          <CardDescription>Define the option contract specifications</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="spot">Spot Price (S)</Label>
              <Input
                id="spot"
                type="number"
                value={contract.spotPrice}
                onChange={(e) => setContract({ ...contract, spotPrice: Number(e.target.value) })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="strike">Strike Price (K)</Label>
              <Input
                id="strike"
                type="number"
                value={contract.strikePrice}
                onChange={(e) => setContract({ ...contract, strikePrice: Number(e.target.value) })}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label>Time to Maturity (years): {contract.timeToMaturity.toFixed(2)}</Label>
            <Slider
              value={[contract.timeToMaturity]}
              onValueChange={([v]) => setContract({ ...contract, timeToMaturity: v })}
              min={0.05}
              max={3}
              step={0.05}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Risk-Free Rate: {(contract.riskFreeRate * 100).toFixed(1)}%</Label>
              <Slider
                value={[contract.riskFreeRate]}
                onValueChange={([v]) => setContract({ ...contract, riskFreeRate: v })}
                min={0}
                max={0.15}
                step={0.005}
              />
            </div>
            <div className="space-y-2">
              <Label>Dividend Yield: {(contract.dividendYield * 100).toFixed(1)}%</Label>
              <Slider
                value={[contract.dividendYield]}
                onValueChange={([v]) => setContract({ ...contract, dividendYield: v })}
                min={0}
                max={0.1}
                step={0.005}
              />
            </div>
          </div>

          <Separator />

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Option Type</Label>
              <Select
                value={contract.optionType}
                onValueChange={(v: "call" | "put") => setContract({ ...contract, optionType: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="call">Call</SelectItem>
                  <SelectItem value="put">Put</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Exercise Style</Label>
              <Select
                value={contract.exerciseStyle}
                onValueChange={(v: "european" | "american") => setContract({ ...contract, exerciseStyle: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="european">European</SelectItem>
                  <SelectItem value="american">American</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center gap-2 pt-2">
            <Badge variant={moneynessLabel === "ITM" ? "default" : moneynessLabel === "OTM" ? "secondary" : "outline"}>
              {moneynessLabel}
            </Badge>
            <span className="text-sm text-muted-foreground">Moneyness: {(moneyness * 100).toFixed(1)}%</span>
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-1">
        <CardHeader>
          <CardTitle>Volatility Model</CardTitle>
          <CardDescription>Configure stochastic volatility parameters</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Model</Label>
            <Select
              value={volParams.model}
              onValueChange={(v: "constant" | "heston") => setVolParams({ ...volParams, model: v })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="constant">Constant (Black-Scholes)</SelectItem>
                <SelectItem value="heston">Heston Stochastic Vol</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {volParams.model === "constant" ? (
            <div className="space-y-2">
              <Label>Volatility (σ): {(contract.volatility * 100).toFixed(1)}%</Label>
              <Slider
                value={[contract.volatility]}
                onValueChange={([v]) => setContract({ ...contract, volatility: v })}
                min={0.05}
                max={0.8}
                step={0.01}
              />
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>
                  Initial Variance (v₀): {(volParams.v0 * 100).toFixed(1)}% (σ ={" "}
                  {(Math.sqrt(volParams.v0) * 100).toFixed(1)}%)
                </Label>
                <Slider
                  value={[volParams.v0]}
                  onValueChange={([v]) => setVolParams({ ...volParams, v0: v })}
                  min={0.01}
                  max={0.16}
                  step={0.005}
                />
              </div>
              <div className="space-y-2">
                <Label>Mean Reversion (κ): {volParams.kappa.toFixed(2)}</Label>
                <Slider
                  value={[volParams.kappa]}
                  onValueChange={([v]) => setVolParams({ ...volParams, kappa: v })}
                  min={0.1}
                  max={5}
                  step={0.1}
                />
              </div>
              <div className="space-y-2">
                <Label>Long-term Variance (θ): {(volParams.theta * 100).toFixed(1)}%</Label>
                <Slider
                  value={[volParams.theta]}
                  onValueChange={([v]) => setVolParams({ ...volParams, theta: v })}
                  min={0.01}
                  max={0.16}
                  step={0.005}
                />
              </div>
              <div className="space-y-2">
                <Label>Vol of Vol (ξ): {volParams.xi.toFixed(2)}</Label>
                <Slider
                  value={[volParams.xi]}
                  onValueChange={([v]) => setVolParams({ ...volParams, xi: v })}
                  min={0.1}
                  max={1}
                  step={0.05}
                />
              </div>
              <div className="space-y-2">
                <Label>Correlation (ρ): {volParams.rho.toFixed(2)}</Label>
                <Slider
                  value={[volParams.rho]}
                  onValueChange={([v]) => setVolParams({ ...volParams, rho: v })}
                  min={-0.95}
                  max={0.95}
                  step={0.05}
                />
              </div>
              <div className="text-xs text-muted-foreground">
                Feller Condition (2κθ {">"} ξ²):{" "}
                <span
                  className={
                    2 * volParams.kappa * volParams.theta > volParams.xi ** 2 ? "text-green-600" : "text-red-600"
                  }
                >
                  {2 * volParams.kappa * volParams.theta > volParams.xi ** 2 ? "Satisfied" : "Not Satisfied"}
                </span>
              </div>
            </div>
          )}

          <Separator />

          <div className="space-y-2">
            <Label>Monte Carlo Paths: {numPaths.toLocaleString()}</Label>
            <Slider
              value={[Math.log10(numPaths)]}
              onValueChange={([v]) => setNumPaths(Math.round(Math.pow(10, v)))}
              min={3}
              max={6}
              step={0.1}
            />
          </div>

          <Button onClick={handleCalculate} className="w-full" disabled={isCalculating}>
            {isCalculating ? "Calculating..." : "Run Monte Carlo"}
          </Button>
        </CardContent>
      </Card>

      <Card className="lg:col-span-1">
        <CardHeader>
          <CardTitle>Pricing Results</CardTitle>
          <CardDescription>Comparison of pricing methods</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-lg border p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Black-Scholes</span>
              <span className="font-mono text-lg font-bold">${analyticalResults.bsPrice.toFixed(4)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Binomial Tree</span>
              <span className="font-mono text-lg">${analyticalResults.binomial.toFixed(4)}</span>
            </div>
            {analyticalResults.heston && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Heston (Semi-Analytical)</span>
                <span className="font-mono text-lg">${analyticalResults.heston.toFixed(4)}</span>
              </div>
            )}
            {mcResult && (
              <>
                <Separator />
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Monte Carlo</span>
                  <span className="font-mono text-lg font-bold text-primary">${mcResult.price.toFixed(4)}</span>
                </div>
                <div className="text-xs text-muted-foreground space-y-1">
                  <div>Std Error: ${mcResult.stdError.toFixed(6)}</div>
                  <div>
                    95% CI: [${mcResult.ci[0].toFixed(4)}, ${mcResult.ci[1].toFixed(4)}]
                  </div>
                  <div>
                    Error vs BS: ${Math.abs(mcResult.price - analyticalResults.bsPrice).toFixed(6)} (
                    {((Math.abs(mcResult.price - analyticalResults.bsPrice) / analyticalResults.bsPrice) * 100).toFixed(
                      3,
                    )}
                    %)
                  </div>
                </div>
              </>
            )}
          </div>

          <Separator />

          <div>
            <h4 className="text-sm font-medium mb-3">Greeks (Black-Scholes)</h4>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Delta (Δ)</span>
                <span className="font-mono">{analyticalResults.greeks.delta.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Gamma (Γ)</span>
                <span className="font-mono">{analyticalResults.greeks.gamma.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Theta (Θ)</span>
                <span className="font-mono">{analyticalResults.greeks.theta.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Vega (ν)</span>
                <span className="font-mono">{analyticalResults.greeks.vega.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Rho (ρ)</span>
                <span className="font-mono">{analyticalResults.greeks.rho.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
