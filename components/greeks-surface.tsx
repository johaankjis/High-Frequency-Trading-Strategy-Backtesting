"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { calculateGreeksSurface } from "@/lib/greeks"

export function GreeksSurface() {
  const [strikePrice, setStrikePrice] = useState(100)
  const [timeToMaturity, setTimeToMaturity] = useState(1)
  const [riskFreeRate, setRiskFreeRate] = useState(0.05)
  const [optionType, setOptionType] = useState<"call" | "put">("call")
  const [selectedGreek, setSelectedGreek] = useState<"delta" | "gamma" | "theta" | "vega" | "price">("delta")

  const surface = useMemo(() => {
    return calculateGreeksSurface({
      strikePrice,
      timeToMaturity,
      riskFreeRate,
      dividendYield: 0.02,
      optionType,
      spotRange: [70, 130],
      volRange: [0.1, 0.6],
      resolution: 25,
    })
  }, [strikePrice, timeToMaturity, riskFreeRate, optionType])

  const getColor = (value: number, min: number, max: number) => {
    const normalized = (value - min) / (max - min)
    const hue = (1 - normalized) * 240
    return `hsl(${hue}, 70%, 50%)`
  }

  const currentData = surface[selectedGreek]
  const minVal = Math.min(...currentData.flat())
  const maxVal = Math.max(...currentData.flat())

  const greekLabels = {
    delta: { name: "Delta (Δ)", desc: "Rate of change of option price with respect to underlying" },
    gamma: { name: "Gamma (Γ)", desc: "Rate of change of delta with respect to underlying" },
    theta: { name: "Theta (Θ)", desc: "Rate of time decay (daily)" },
    vega: { name: "Vega (ν)", desc: "Sensitivity to volatility changes" },
    price: { name: "Option Price", desc: "Black-Scholes option value" },
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-4">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Parameters</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Option Type</Label>
              <Select value={optionType} onValueChange={(v: "call" | "put") => setOptionType(v)}>
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
              <Label>Strike: ${strikePrice}</Label>
              <Slider value={[strikePrice]} onValueChange={([v]) => setStrikePrice(v)} min={80} max={120} step={1} />
            </div>

            <div className="space-y-2">
              <Label>Time to Maturity: {timeToMaturity.toFixed(2)}y</Label>
              <Slider
                value={[timeToMaturity]}
                onValueChange={([v]) => setTimeToMaturity(v)}
                min={0.1}
                max={2}
                step={0.05}
              />
            </div>

            <div className="space-y-2">
              <Label>Risk-Free Rate: {(riskFreeRate * 100).toFixed(1)}%</Label>
              <Slider
                value={[riskFreeRate]}
                onValueChange={([v]) => setRiskFreeRate(v)}
                min={0}
                max={0.1}
                step={0.005}
              />
            </div>

            <div className="pt-4 border-t space-y-2">
              <Label>Display Greek</Label>
              <Select value={selectedGreek} onValueChange={(v: typeof selectedGreek) => setSelectedGreek(v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="price">Price</SelectItem>
                  <SelectItem value="delta">Delta</SelectItem>
                  <SelectItem value="gamma">Gamma</SelectItem>
                  <SelectItem value="theta">Theta</SelectItem>
                  <SelectItem value="vega">Vega</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle>{greekLabels[selectedGreek].name} Surface</CardTitle>
            <CardDescription>{greekLabels[selectedGreek].desc}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative">
              <div className="flex gap-4">
                <div className="flex-1">
                  <div className="relative aspect-square max-w-[500px] mx-auto">
                    <div className="absolute -left-10 top-1/2 -translate-y-1/2 -rotate-90 text-sm text-muted-foreground whitespace-nowrap">
                      Volatility (%)
                    </div>

                    <div
                      className="grid gap-0.5"
                      style={{ gridTemplateColumns: `repeat(${surface.spots.length}, 1fr)` }}
                    >
                      {surface.vols
                        .slice()
                        .reverse()
                        .map((vol, vi) =>
                          surface.spots.map((spot, si) => {
                            const actualVi = surface.vols.length - 1 - vi
                            const value = currentData[actualVi][si]
                            return (
                              <div
                                key={`${vi}-${si}`}
                                className="aspect-square relative group"
                                style={{ backgroundColor: getColor(value, minVal, maxVal) }}
                              >
                                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-black/70 text-white text-xs font-mono z-10 transition-opacity">
                                  {value.toFixed(3)}
                                </div>
                              </div>
                            )
                          }),
                        )}
                    </div>

                    <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                      <span>${surface.spots[0]}</span>
                      <span>${surface.spots[Math.floor(surface.spots.length / 2)]}</span>
                      <span>${surface.spots[surface.spots.length - 1]}</span>
                    </div>
                    <div className="text-center text-sm text-muted-foreground mt-1">Spot Price ($)</div>

                    <div className="absolute left-0 top-0 h-full flex flex-col justify-between -translate-x-6 text-xs text-muted-foreground">
                      <span>{(surface.vols[surface.vols.length - 1] * 100).toFixed(0)}%</span>
                      <span>{(surface.vols[Math.floor(surface.vols.length / 2)] * 100).toFixed(0)}%</span>
                      <span>{(surface.vols[0] * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>

                <div className="w-20 flex flex-col items-center">
                  <div className="text-xs text-muted-foreground mb-1">{maxVal.toFixed(2)}</div>
                  <div
                    className="w-4 flex-1 rounded"
                    style={{
                      background: `linear-gradient(to bottom, hsl(0, 70%, 50%), hsl(60, 70%, 50%), hsl(120, 70%, 50%), hsl(180, 70%, 50%), hsl(240, 70%, 50%))`,
                    }}
                  />
                  <div className="text-xs text-muted-foreground mt-1">{minVal.toFixed(2)}</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{greekLabels[selectedGreek].name} vs Spot Price</CardTitle>
            <CardDescription>Cross-section at different volatility levels</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[250px] flex items-end gap-1">
              {surface.spots.map((spot, i) => {
                const value = currentData[12][i]
                const height = ((value - minVal) / (maxVal - minVal)) * 100
                return (
                  <div
                    key={i}
                    className="flex-1 bg-primary/70 hover:bg-primary transition-colors rounded-t"
                    style={{ height: `${Math.max(height, 2)}%` }}
                    title={`Spot: $${spot}, Value: ${value.toFixed(4)}`}
                  />
                )
              })}
            </div>
            <div className="flex justify-between mt-2 text-xs text-muted-foreground">
              <span>${surface.spots[0]}</span>
              <span>Spot Price</span>
              <span>${surface.spots[surface.spots.length - 1]}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{greekLabels[selectedGreek].name} vs Volatility</CardTitle>
            <CardDescription>Cross-section at ATM strike</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[250px] flex items-end gap-1">
              {surface.vols.map((vol, i) => {
                const spotIdx = Math.floor(surface.spots.length / 2)
                const value = currentData[i][spotIdx]
                const height = ((value - minVal) / (maxVal - minVal)) * 100
                return (
                  <div
                    key={i}
                    className="flex-1 bg-secondary hover:bg-secondary/80 transition-colors rounded-t"
                    style={{ height: `${Math.max(height, 2)}%` }}
                    title={`Vol: ${(vol * 100).toFixed(0)}%, Value: ${value.toFixed(4)}`}
                  />
                )
              })}
            </div>
            <div className="flex justify-between mt-2 text-xs text-muted-foreground">
              <span>{(surface.vols[0] * 100).toFixed(0)}%</span>
              <span>Volatility</span>
              <span>{(surface.vols[surface.vols.length - 1] * 100).toFixed(0)}%</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
