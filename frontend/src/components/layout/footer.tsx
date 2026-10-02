import Link from "next/link";
import { Sparkles } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-surface-border bg-surface/40">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="flex flex-col items-start justify-between gap-8 md:flex-row">
          <div>
            <div className="flex items-center gap-2 font-semibold">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent-2">
                <Sparkles className="h-3.5 w-3.5 text-white" />
              </span>
              DataPilot
            </div>
            <p className="mt-3 max-w-xs text-sm text-muted">
              An autonomous, deterministic-by-default data-science platform — from raw CSV to a
              recommended model, fully traced.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-10 text-sm sm:grid-cols-3">
            <div className="space-y-3">
              <p className="font-medium text-foreground">Product</p>
              <Link href="#features" className="block text-muted hover:text-foreground">
                Features
              </Link>
              <Link href="#pipeline" className="block text-muted hover:text-foreground">
                Pipeline
              </Link>
              <Link href="/login" className="block text-muted hover:text-foreground">
                Dashboard
              </Link>
            </div>
            <div className="space-y-3">
              <p className="font-medium text-foreground">Platform</p>
              <span className="block text-muted">13 engine phases</span>
              <span className="block text-muted">REST API</span>
              <span className="block text-muted">Autonomous agent</span>
            </div>
          </div>
        </div>

        <div className="mt-10 flex flex-col items-center justify-between gap-4 border-t border-surface-border pt-6 text-xs text-muted sm:flex-row">
          <p>&copy; {new Date().getFullYear()} DataPilot. Built for deterministic, explainable ML.</p>
          <p>Phase 14 — Frontend</p>
        </div>
      </div>
    </footer>
  );
}
