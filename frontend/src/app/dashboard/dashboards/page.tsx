"use client";

import { useMemo, useState } from "react";
import { toast } from "sonner";
import { LayoutGrid, Sparkles, Rows3, Columns3, AlertTriangle, Wand2 } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { StatCard } from "@/components/ui/stat-card";
import { Badge } from "@/components/ui/badge";
import { NumericHistogramCard, CategoricalBarCard } from "@/components/ui/eda-charts";
import { cn } from "@/lib/utils";
import { analyzeEda, ApiError, EdaReport } from "@/lib/api";

export default function DashboardsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<EdaReport | null>(null);

  const [selected, setSelected] = useState<string[]>([]);
  const [tileCount, setTileCount] = useState(6);
  const [built, setBuilt] = useState(false);

  const chartableColumns = useMemo(() => {
    if (!report) return [];
    const numeric = report.univariate.numeric.map((c) => ({ column: c.column, kind: "numeric" as const }));
    const categorical = report.univariate.categorical.map((c) => ({
      column: c.column,
      kind: "categorical" as const,
    }));
    return [...numeric, ...categorical];
  }, [report]);

  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setBuilt(false);
    try {
      const data = await analyzeEda(file);
      setReport(data);
      const defaultSelection = [...data.univariate.numeric, ...data.univariate.categorical]
        .slice(0, 8)
        .map((c) => c.column);
      setSelected(defaultSelection);
      setTileCount(Math.min(6, defaultSelection.length) || 1);
      toast.success("Dataset analyzed — choose your categories below");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  function toggleColumn(column: string) {
    setSelected((prev) =>
      prev.includes(column) ? prev.filter((c) => c !== column) : [...prev, column],
    );
  }

  const tiles = useMemo(() => {
    if (!report) return [];
    return selected
      .slice(0, tileCount)
      .map((column) => {
        const numeric = report.univariate.numeric.find((c) => c.column === column);
        if (numeric) {
          const dist = report.distribution.columns.find((d) => d.column === column);
          return { kind: "numeric" as const, column, numeric, dist };
        }
        const categorical = report.univariate.categorical.find((c) => c.column === column);
        if (categorical) return { kind: "categorical" as const, column, categorical };
        return null;
      })
      .filter((t): t is NonNullable<typeof t> => t !== null);
  }, [report, selected, tileCount]);

  return (
    <div>
      <PageHeader
        title="Dashboard builder"
        description="Upload a dataset, pick the categories that matter, and get a Power BI-style grid of charts in one click."
      />

      <Card>
        <CardHeader>
          <CardTitle>1. Upload &amp; analyze</CardTitle>
          <CardDescription>We run the deterministic EDA engine once and chart everything from it.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={file} onFileSelect={setFile} />
          <Button onClick={handleAnalyze} disabled={!file} loading={loading}>
            <Sparkles className="h-4 w-4" /> Analyze dataset
          </Button>
        </CardContent>
      </Card>

      {report && (
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>2. Choose categories &amp; dashboard size</CardTitle>
            <CardDescription>
              Pick which columns to visualize, then tell us how many tiles the dashboard should show.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex flex-wrap gap-2">
              {chartableColumns.map(({ column, kind }) => {
                const active = selected.includes(column);
                return (
                  <button
                    key={column}
                    type="button"
                    onClick={() => toggleColumn(column)}
                    className={cn(
                      "rounded-full border px-3.5 py-1.5 font-mono text-sm transition-colors",
                      active
                        ? "border-primary/50 bg-primary/15 text-primary-2"
                        : "border-surface-border bg-surface-2/40 text-muted hover:text-foreground",
                    )}
                  >
                    {column}
                    <span className="ml-1.5 text-xs opacity-60">{kind === "numeric" ? "#" : "abc"}</span>
                  </button>
                );
              })}
            </div>

            <div className="flex flex-wrap items-end gap-4">
              <div>
                <Label htmlFor="tile-count">How many dashboard tiles?</Label>
                <Input
                  id="tile-count"
                  type="number"
                  min={1}
                  max={Math.max(1, selected.length)}
                  value={tileCount}
                  onChange={(e) => setTileCount(Number(e.target.value) || 1)}
                  className="w-32"
                />
              </div>
              <p className="pb-3 text-sm text-muted">
                {selected.length} categor{selected.length === 1 ? "y" : "ies"} selected — showing the first{" "}
                {Math.min(tileCount, selected.length)}.
              </p>
              <Button
                onClick={() => setBuilt(true)}
                disabled={selected.length === 0}
                className="ml-auto"
              >
                <Wand2 className="h-4 w-4" /> Generate dashboard
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {built && report && (
        <div className="mt-8 space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard icon={<Rows3 className="h-5 w-5" />} label="Rows" value={report.n_rows.toLocaleString()} />
            <StatCard
              icon={<Columns3 className="h-5 w-5" />}
              label="Columns"
              value={report.n_columns.toLocaleString()}
            />
            <StatCard
              icon={<AlertTriangle className="h-5 w-5" />}
              label="Missing cells"
              value={`${report.univariate.missingness.missing_percentage.toFixed(2)}%`}
            />
            <StatCard
              icon={<LayoutGrid className="h-5 w-5" />}
              label="Dashboard tiles"
              value={tiles.length}
            />
          </div>

          <div className="flex items-center gap-2">
            <h2 className="text-xl font-semibold tracking-tight">Your dashboard</h2>
            <Badge variant="primary">{tiles.length} tiles</Badge>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {tiles.map((tile, i) =>
              tile.kind === "numeric" ? (
                <NumericHistogramCard
                  key={tile.column}
                  dist={
                    tile.dist ?? {
                      column: tile.column,
                      status: "unavailable",
                      reason: "No distribution computed for this column",
                      count: tile.numeric.count,
                      minimum: tile.numeric.minimum,
                      maximum: tile.numeric.maximum,
                      mean: tile.numeric.mean,
                      median: tile.numeric.median,
                      std: tile.numeric.std,
                      histogram: {
                        status: "unavailable",
                        n_bins: null,
                        bin_edges: [],
                        bins: [],
                        total_count: null,
                      },
                    }
                  }
                />
              ) : (
                <CategoricalBarCard key={tile.column} cat={tile.categorical} variant={i % 2 === 0 ? "bar" : "pie"} />
              ),
            )}
          </div>
        </div>
      )}
    </div>
  );
}
