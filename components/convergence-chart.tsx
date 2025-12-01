"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  ComposedChart,
} from "recharts"
import { runConvergenceStudy } from "@/lib/convergence"

export function ConvergenceChart() {
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<ReturnType<typeof runConvergenceStudy> | null>(null)

  const handleRunStudy = async () => {
    setIsRunning(true)
    await new Promise((resolve) => setTimeout(resolve, 100))
    const study = runConvergenceStudy({
      spotPrice: 100,
      strikePrice: 100,
      timeToMaturity: 1,
      riskFreeRate: 0.05,
      volatility: 0.2,
      pathCounts: [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000],
      numTrials: 10,
    })
    setResults(study)
    setIsRunning(false)
  }

  const convergenceData = useMemo(() => {
    if (!results) return []
    return results.pathCounts.map((n, i) => ({
      paths: n,
      logPaths: Math.log10(n),
      price: results.meanPrices[i],
      rmse: results.rmse[i],
      logRmse: Math.log10(results.rmse[i]),
      stdError: results.stdErrors[i],
      theoretical: results.theoreticalConvergence[i],
      logTheoretical: Math.log10(results.theoreticalConvergence[i]),
    }))
  }, [results])

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Convergence Analysis</CardTitle>
              <CardDescription>Study Monte Carlo convergence behavior across different path counts</CardDescription>
            </div>
            <Button onClick={handleRunStudy} disabled={isRunning}>
              {isRunning ? "Running..." : "Run Study"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {!results ? (
            <div className="h-[400px] flex items-center justify-center text-muted-foreground">
              Click &ldquo;Run Study&rdquo; to generate convergence analysis
            </div>
          ) : (
            <div className="space-y-6">
              <div className="grid gap-4 md:grid-cols-4">
                <div className="rounded-lg border p-4 text-center">
                  <div className="text-2xl font-bold font-mono">${results.analyticalPrice.toFixed(4)}</div>
                  <div className="text-sm text-muted-foreground">Analytical (BS)</div>
                </div>
                <div className="rounded-lg border p-4 text-center">
                  <div className="text-2xl font-bold font-mono">
                    ${results.meanPrices[results.meanPrices.length - 1].toFixed(4)}
                  </div>
                  <div className="text-sm text-muted-foreground">Best MC Estimate</div>
                </div>
                <div className="rounded-lg border p-4 text-center">
                  <div className="text-2xl font-bold font-mono">{results.pathCounts.length * results.numTrials}</div>
                  <div className="text-sm text-muted-foreground">Total Simulations</div>
                </div>
                <div className="rounded-lg border p-4 text-center">
                  <Badge variant="outline" className="text-lg">
                    O(1/√N)
                  </Badge>
                  <div className="text-sm text-muted-foreground mt-1">Convergence Rate</div>
                </div>
              </div>

              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={convergenceData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis
                      dataKey="logPaths"
                      tickFormatter={(v) => `${Math.pow(10, v).toLocaleString()}`}
                      label={{ value: "Number of Paths", position: "bottom", offset: 20 }}
                      type="number"
                      domain={["dataMin", "dataMax"]}
                    />
                    <YAxis
                      yAxisId="left"
                      dataKey="logRmse"
                      tickFormatter={(v) => `$${Math.pow(10, v).toFixed(4)}`}
                      label={{ value: "RMSE (log scale)", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip
                      formatter={(value: number, name: string) => {
                        if (name === "RMSE") return [`$${Math.pow(10, value).toFixed(6)}`, name]
                        if (name === "Theoretical") return [`$${Math.pow(10, value).toFixed(6)}`, name]
                        return [value, name]
                      }}
                      labelFormatter={(v) => `Paths: ${Math.pow(10, v).toLocaleString()}`}
                      contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))" }}
                    />
                    <Legend />
                    <Line
                      yAxisId="left"
                      dataKey="logRmse"
                      stroke="hsl(var(--chart-1))"
                      strokeWidth={2}
                      dot={{ fill: "hsl(var(--chart-1))", r: 4 }}
                      name="RMSE"
                    />
                    <Line
                      yAxisId="left"
                      dataKey="logTheoretical"
                      stroke="hsl(var(--muted-foreground))"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                      name="Theoretical"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {results && (
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Price Estimates with Confidence Intervals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={convergenceData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis
                      dataKey="logPaths"
                      tickFormatter={(v) => `${(Math.pow(10, v) / 1000).toFixed(0)}K`}
                      type="number"
                    />
                    <YAxis domain={["auto", "auto"]} tickFormatter={(v) => `$${v.toFixed(2)}`} />
                    <Tooltip
                      formatter={(value: number) => [`$${value.toFixed(4)}`, "Price"]}
                      contentStyle={{ backgroundColor: "hsl(var(--card))", borderColor: "hsl(var(--border))" }}
                    />
                    <ReferenceLine
                      y={results.analyticalPrice}
                      stroke="hsl(var(--chart-2))"
                      strokeDasharray="5 5"
                      label={{ value: "BS", position: "right" }}
                    />
                    <Line
                      dataKey="price"
                      stroke="hsl(var(--chart-1))"
                      strokeWidth={2}
                      dot={{ fill: "hsl(var(--chart-1))", r: 4 }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Convergence Data</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 font-medium">Paths</th>
                      <th className="text-right py-2 font-medium">Price</th>
                      <th className="text-right py-2 font-medium">RMSE</th>
                      <th className="text-right py-2 font-medium">Error %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.pathCounts.map((n, i) => (
                      <tr key={n} className="border-b border-border/50">
                        <td className="py-2 font-mono">{n.toLocaleString()}</td>
                        <td className="py-2 text-right font-mono">${results.meanPrices[i].toFixed(4)}</td>
                        <td className="py-2 text-right font-mono">${results.rmse[i].toFixed(6)}</td>
                        <td className="py-2 text-right font-mono">
                          {(
                            (Math.abs(results.meanPrices[i] - results.analyticalPrice) / results.analyticalPrice) *
                            100
                          ).toFixed(3)}
                          %
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
