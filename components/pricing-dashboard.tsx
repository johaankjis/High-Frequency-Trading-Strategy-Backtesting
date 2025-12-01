"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { OptionPricer } from "@/components/option-pricer"
import { BenchmarkResults } from "@/components/benchmark-results"
import { ConvergenceChart } from "@/components/convergence-chart"
import { PricePathsChart } from "@/components/price-paths-chart"
import { GreeksSurface } from "@/components/greeks-surface"

export function PricingDashboard() {
  const [activeTab, setActiveTab] = useState("pricer")

  return (
    <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
      <TabsList className="grid w-full grid-cols-5 max-w-2xl">
        <TabsTrigger value="pricer">Pricer</TabsTrigger>
        <TabsTrigger value="paths">Price Paths</TabsTrigger>
        <TabsTrigger value="greeks">Greeks</TabsTrigger>
        <TabsTrigger value="convergence">Convergence</TabsTrigger>
        <TabsTrigger value="benchmark">Benchmark</TabsTrigger>
      </TabsList>

      <TabsContent value="pricer" className="space-y-6">
        <OptionPricer />
      </TabsContent>

      <TabsContent value="paths" className="space-y-6">
        <PricePathsChart />
      </TabsContent>

      <TabsContent value="greeks" className="space-y-6">
        <GreeksSurface />
      </TabsContent>

      <TabsContent value="convergence" className="space-y-6">
        <ConvergenceChart />
      </TabsContent>

      <TabsContent value="benchmark" className="space-y-6">
        <BenchmarkResults />
      </TabsContent>
    </Tabs>
  )
}
