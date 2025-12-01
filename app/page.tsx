import { Suspense } from "react"
import { PricingDashboard } from "@/components/pricing-dashboard"
import { Header } from "@/components/header"

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<div className="animate-pulse h-96 bg-muted rounded-lg" />}>
          <PricingDashboard />
        </Suspense>
      </div>
    </main>
  )
}
