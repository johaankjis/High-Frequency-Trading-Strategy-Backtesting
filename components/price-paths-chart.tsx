"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from "recharts"
import { simulateGBM, simulateHeston } from "@/lib/simulation"

export function PricePathsChart() {
  const [numPaths, setNumPaths] = useState(50)
  const [numSteps, setNumSteps] = useState(252)
  const [model, setModel] = useState<"gbm" | "heston">("gbm")
  const [spotPrice, setSpotPrice] = useState(100)
  const [volatility, setVolatility] = useState(0.2)
  const [seed, setSeed] = useState(42)

  const { paths, meanPath, upperBand, lowerBand, variancePaths } = useMemo(() => {
    if (model === "gbm") {
      return simulateGBM({
        spotPrice,
        riskFreeRate: 0.05,
        dividendYield: 0.02,
        volatility,
        timeToMaturity: 1,
        numPaths,
        numSteps,
        seed,
      })
    } else {
      return simulateHeston({
        spotPrice,
        riskFreeRate: 0.05,
        dividendYield: 0.02,
        v0: volatility ** 2,
        kappa: 2,
        theta: volatility ** 2,
        xi: 0.3,
        rho: -0.7,
        timeToMaturity: 1,
        numPaths,
        numSteps,
        seed,
      })
    }
  }, [model, spotPrice, volatility, numPaths, numSteps, seed])

  const chartData = useMemo(() => {
    const data = []
    for (let i = 0; i <= numSteps; i++) {
      const point: Record<string, number> = {
        step: i,
        time: i / numSteps,
        mean: meanPath[i],
        upper: upperBand[i],
        lower: lowerBand[i],
      }
      for (let p = 0; p < Math.min(numPaths, 20); p++) {
        point[`path${p}`] = paths[p][i]
      }
      data.push(point)
    }
    return data
  }, [paths, meanPath, upperBand, lowerBand, numSteps, numPaths])

  const varianceData = useMemo(() => {
    if (!variancePaths) return null
    const data = []
    for (let i = 0; i <= numSteps; i++) {
      const point: Record<string, number> = {
        step: i,
        time: i / numSteps,
      }
      for (let p = 0; p < Math.min(numPaths, 20); p++) {
        point[`var${p}`] = variancePaths[p][i]
      }
      let sum = 0
      for (let p = 0; p < numPaths; p++) {
        sum += variancePaths[p][i]
      }
      point.meanVar = sum / numPaths
      data.push(point)
    }
    return data
  }, [variancePaths, numSteps, numPaths])

  const pathColors = [
    "hsl(var(--chart-1))",
    "hsl(var(--chart-2))",
    "hsl(var(--chart-3))",
    "hsl(var(--chart-4))",
    "hsl(var(--chart-5))",
  ]

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Simulation Parameters</CardTitle>
          <CardDescription>Configure the path simulation settings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-5">
            <div className="space-y-2">
              <Label>Model</Label>
              <Select value={model} onValueChange={(v: "gbm" | "heston") => setModel(v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gbm">Geometric Brownian Motion</SelectItem>
                  <SelectItem value="heston">Heston Stochastic Vol</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Paths: {numPaths}</Label>
              <Slider value={[numPaths]} onValueChange={([v]) => setNumPaths(v)} min={10} max={200} step={10} />
            </div>
            <div className="space-y-2">
              <Label>Steps: {numSteps}</Label>
              <Slider value={[numSteps]} onValueChange={([v]) => setNumSteps(v)} min={50} max={504} step={1} />
            </div>
            <div className="space-y-2">
              <Label>Volatility: {(volatility * 100).toFixed(0)}%</Label>
              <Slider value={[volatility]} onValueChange={([v]) => setVolatility(v)} min={0.1} max={0.6} step={0.01} />
            </div>
            <div className="flex items-end">
              <Button onClick={() => setSeed(Math.floor(Math.random() * 10000))} variant="outline" className="w-full">
                Resimulate
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Asset Price Paths</CardTitle>
          <CardDescription>
            {model === "gbm" ? "Geometric Brownian Motion" : "Heston Model"} simulation with {numPaths} paths
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="time"
                  tickFormatter={(v) => `${(v * 12).toFixed(0)}mo`}
                  label={{ value: "Time", position: "bottom", offset: 0 }}
                  className="text-muted-foreground"
                />
                <YAxis
                  domain={["auto", "auto"]}
                  label={{ value: "Price ($)", angle: -90, position: "insideLeft" }}
                  className="text-muted-foreground"
                />
                <Tooltip
                  formatter={(value: number) => [`$${value.toFixed(2)}`, ""]}
                  labelFormatter={(v) => `Time: ${(v * 12).toFixed(1)} months`}
                  contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))" }}
                />
                <Area dataKey="upper" stroke="none" fill="hsl(var(--chart-2))" fillOpacity={0.1} />
                <Area dataKey="lower" stroke="none" fill="hsl(var(--chart-2))" fillOpacity={0.1} />
                {Array.from({ length: Math.min(numPaths, 20) }, (_, i) => (
                  <Line
                    key={`path${i}`}
                    dataKey={`path${i}`}
                    stroke={pathColors[i % pathColors.length]}
                    strokeWidth={0.5}
                    dot={false}
                    opacity={0.4}
                  />
                ))}
                <Line dataKey="mean" stroke="hsl(var(--chart-5))" strokeWidth={2.5} dot={false} name="Mean Path" />
                <ReferenceLine y={spotPrice} stroke="hsl(var(--muted-foreground))" strokeDasharray="5 5" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {model === "heston" && varianceData && (
        <Card>
          <CardHeader>
            <CardTitle>Stochastic Variance Paths</CardTitle>
            <CardDescription>Heston model variance evolution</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={varianceData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="time" tickFormatter={(v) => `${(v * 12).toFixed(0)}mo`} />
                  <YAxis
                    domain={["auto", "auto"]}
                    tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                    label={{ value: "Variance", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip
                    formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, ""]}
                    contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))" }}
                  />
                  {Array.from({ length: Math.min(numPaths, 20) }, (_, i) => (
                    <Line
                      key={`var${i}`}
                      dataKey={`var${i}`}
                      stroke="hsl(var(--chart-2))"
                      strokeWidth={0.5}
                      dot={false}
                      opacity={0.4}
                    />
                  ))}
                  <Line
                    dataKey="meanVar"
                    stroke="hsl(var(--chart-5))"
                    strokeWidth={2}
                    dot={false}
                    name="Mean Variance"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Terminal Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mean</span>
                <span className="font-mono">${meanPath[numSteps].toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Std Dev</span>
                <span className="font-mono">${((upperBand[numSteps] - lowerBand[numSteps]) / 4).toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">95% Range</span>
                <span className="font-mono">
                  [${lowerBand[numSteps].toFixed(0)}, ${upperBand[numSteps].toFixed(0)}]
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Drift Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Expected Return</span>
                <span className="font-mono">{((meanPath[numSteps] / spotPrice - 1) * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Theoretical (r-q)</span>
                <span className="font-mono">{((0.05 - 0.02) * 100).toFixed(2)}%</span>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Simulation Info</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model</span>
                <span className="font-mono">{model.toUpperCase()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Random Seed</span>
                <span className="font-mono">{seed}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">dt</span>
                <span className="font-mono">{(1 / numSteps).toFixed(5)}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
