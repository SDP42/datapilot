"use client";

import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  LayoutGrid,
  Sparkles,
  Rows3,
  Columns3,
  AlertTriangle,
  Wand2,
  FileDown,
  Printer,
  LayoutTemplate,
} from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { StatCard } from "@/components/ui/stat-card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { NumericHistogramCard, CategoricalBarCard } from "@/components/ui/eda-charts";
import { cn } from "@/lib/utils";
import { downloadDashboardHtml, printDashboardPdf } from "@/lib/export";
import { groupColumnsIntoDashboards, suggestDashboardCount } from "@/lib/dashboard-grouping";
import { analyzeEda, ApiError, EdaReport } from "@/lib/api";

type Tile =
  | { kind: "numeric"; column: string; numeric: EdaReport["univariate"]["numeric"][number] }
  | { kind: "categorical"; column: string; categorical: EdaReport["univariate"]["categorical"][number] };

function buildTile(report: EdaReport, column: string): Tile | null {
  const numeric = report.univariate.numeric.find((c) => c.column === column);
  if (numeric) return { kind: "numeric", column, numeric };
  const categorical = report.univariate.categorical.find((c) => c.column === column);
  if (categorical) return { kind: "categorical", column, categorical };
  return null;
}

function TileGrid({
  report,
  columns,
  innerRef,
}: {
  report: EdaReport;
  columns: string[];
  innerRef?: React.Ref<HTMLDivElement>;
}) {
  const tiles = columns.map((c) => buildTile(report, c)).filter((t): t is Tile => t !== null);
  return (
    <div ref={innerRef} className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {tiles.map((tile, i) =>
        tile.kind === "numeric" ? (
          <NumericHistogramCard
            key={tile.column}
            dist={
              report.distribution.columns.find((d) => d.column === tile.column) ?? {
                column: tile.column,
                status: "unavailable",
                reason: "No distribution computed for this column",
                count: tile.numeric.count,
                minimum: tile.numeric.minimum,
                maximum: tile.numeric.maximum,
                mean: tile.numeric.mean,
                median: tile.numeric.median,
                std: tile.numeric.std,
                histogram: { status: "unavailable", n_bins: null, bin_edges: [], bins: [], total_count: null },
              }
            }
          />
        ) : (
          <CategoricalBarCard key={tile.column} cat={tile.categorical} variant={i % 2 === 0 ? "bar" : "pie"} />
        ),
      )}
    </div>
  );
}

export default function DashboardsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<EdaReport | null>(null);
  const [fileLabel, setFileLabel] = useState("dataset");

  const [selected, setSelected] = useState<string[]>([]);
  const [dashboardCount, setDashboardCount] = useState(1);
  const [built, setBuilt] = useState(false);
  const [activeTab, setActiveTab] = useState("0");

  const dashboardRefs = useRef<Record<string, HTMLDivElement | null>>({});

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
      const allColumns = [...data.univariate.numeric, ...data.univariate.categorical].map(
        (c) => c.column,
      );
      setSelected(allColumns);
      setDashboardCount(suggestDashboardCount(allColumns));
      toast.success(`Dataset analyzed — all ${allColumns.length} chartable columns selected`);
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

  function selectAll() {
    const all = chartableColumns.map((c) => c.column);
    setSelected(all);
    setDashboardCount(suggestDashboardCount(all));
  }

  function clearSelection() {
    setSelected([]);
    setDashboardCount(1);
  }

  const groups = useMemo(() => {
    if (!built || selected.length === 0) return [];
    return groupColumnsIntoDashboards(selected, dashboardCount);
  }, [built, selected, dashboardCount]);

  function handleGenerate() {
    setBuilt(true);
    setActiveTab("0");
  }

  return (
    <div>
      <PageHeader
        title="Dashboard builder"
        description="Upload a dataset and get multiple domain-themed dashboards — like separate HR, supply chain, or sales reports — not one flat pile of charts."
      />

      <Card>
        <CardHeader>
          <CardTitle>1. Upload &amp; analyze</CardTitle>
          <CardDescription>We run the deterministic EDA engine once and chart everything from it.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone
            file={file}
            onFileSelect={(f) => {
              setFile(f);
              if (f) setFileLabel(f.name.replace(/\.csv$/i, ""));
            }}
          />
          <Button onClick={handleAnalyze} disabled={!file} loading={loading}>
            <Sparkles className="h-4 w-4" /> Analyze dataset
          </Button>
        </CardContent>
      </Card>

      {report && (
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>2. Choose categories &amp; how many dashboards</CardTitle>
            <CardDescription>
              Pick which columns to visualize, then tell us how many separate dashboards to split
              them into — each one grouped by detected theme (e.g. workforce, compensation,
              logistics, sales).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="secondary" size="sm" onClick={selectAll} type="button">
                Select all ({chartableColumns.length})
              </Button>
              <Button variant="ghost" size="sm" onClick={clearSelection} type="button">
                Clear
              </Button>
            </div>
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
                <Label htmlFor="dashboard-count">How many dashboards?</Label>
                <Input
                  id="dashboard-count"
                  type="number"
                  min={1}
                  max={Math.max(1, selected.length)}
                  value={dashboardCount}
                  onChange={(e) => setDashboardCount(Math.max(1, Number(e.target.value) || 1))}
                  className="w-32"
                />
              </div>
              <p className="pb-3 text-sm text-muted">
                {selected.length} of {chartableColumns.length} categor
                {chartableColumns.length === 1 ? "y" : "ies"} selected — will be split into{" "}
                {Math.min(dashboardCount, Math.max(1, selected.length))} dashboard
                {Math.min(dashboardCount, Math.max(1, selected.length)) === 1 ? "" : "s"}.
              </p>
              <Button onClick={handleGenerate} disabled={selected.length === 0} className="ml-auto">
                <Wand2 className="h-4 w-4" /> Generate dashboards
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {built && report && groups.length > 0 && (
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
              icon={<LayoutTemplate className="h-5 w-5" />}
              label="Dashboards generated"
              value={groups.length}
            />
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <div className="flex flex-wrap items-center gap-3">
              <TabsList className="flex-wrap">
                {groups.map((g, i) => (
                  <TabsTrigger key={i} value={String(i)}>
                    {g.title}
                    <span className="ml-1.5 opacity-70">({g.columns.length})</span>
                  </TabsTrigger>
                ))}
              </TabsList>
              <div className="ml-auto flex gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    const el = dashboardRefs.current[activeTab];
                    const group = groups[Number(activeTab)];
                    if (el && group) downloadDashboardHtml(el, `${fileLabel}_${group.title}`);
                  }}
                >
                  <FileDown className="h-3.5 w-3.5" /> Download HTML
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    const el = dashboardRefs.current[activeTab];
                    const group = groups[Number(activeTab)];
                    if (el && group) printDashboardPdf(el, `${fileLabel}_${group.title}`);
                  }}
                >
                  <Printer className="h-3.5 w-3.5" /> Download PDF
                </Button>
              </div>
            </div>

            {groups.map((g, i) => (
              <TabsContent key={i} value={String(i)}>
                <div className="mb-4 flex items-center gap-2">
                  <LayoutGrid className="h-4 w-4 text-primary-2" />
                  <h2 className="text-lg font-semibold tracking-tight">{g.title}</h2>
                  <Badge variant="primary">{g.columns.length} tiles</Badge>
                </div>
                <TileGrid
                  report={report}
                  columns={g.columns}
                  innerRef={(el) => {
                    dashboardRefs.current[String(i)] = el;
                  }}
                />
              </TabsContent>
            ))}
          </Tabs>
        </div>
      )}
    </div>
  );
}
