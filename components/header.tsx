export function Header() {
  return (
    <header className="border-b border-border bg-card">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Monte Carlo Options Pricing</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Stochastic Volatility Framework for Derivatives Valuation
            </p>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs font-mono bg-muted px-2 py-1 rounded">v1.0.0</span>
          </div>
        </div>
      </div>
    </header>
  )
}
