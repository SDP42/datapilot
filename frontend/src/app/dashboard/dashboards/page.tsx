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
  SlidersHorizontal,
  TrendingUp,
  Lightbulb,
  X,
} from "lucide-react";
import {
  Line,
  LineChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

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
import { groupColumnsIntoDashboards, suggestDashboardCount, type DashboardGroup } from "@/lib/dashboard-grouping";
import { parseCsvFile } from "@/lib/csv-parse";
import {
  applyFilters,
  computeCategoricalAnalysis,
  computeNumericDistribution,
  computeTrend,
  distinctValues,
  strongestCorrelation,
  type Row,
} from "@/lib/bi-stats";
import { analyzeEda, ApiError, EdaReport } from "@/lib/api";

const tooltipStyle = {
  background: "var(--surface-2)",
  border: "1px solid var(--surface-border)",
  borderRadius: 8,
  fontSize: 12,
};

function BiDashboardPanel({
  report,
  group,
  rawRows,
  filters,
  onToggleFilter,
  onClearFilters,
  dateColumn,
  innerRef,
}: {
  report: EdaReport;
  group: DashboardGroup;
  rawRows: Row[];
  filters: Record<string, string[]>;
  onToggleFilter: (column: string, value: string) => void;
  onClearFilters: () => void;
  dateColumn: string | null;
  innerRef?: React.Ref<HTMLDivElement>;
}) {
  const groupNumeric = group.columns.filter(
    (c) => report.univariate.numeric.some((n) => n.column === c),
  );
  const groupCategorical = group.columns.filter(
    (c) => report.univariate.categorical.some((n) => n.column === c),
  );

  const filteredRows = useMemo(() => applyFilters(rawRows, filters), [rawRows, filters]);
  const activeFilterCount = Object.values(filters).reduce((a, v) => a + v.length, 0);

  const kpiNumeric = groupNumeric.slice(0, 2).map((col) => {
    const values = filteredRows.map((r) => parseFloat(r[col])).filter(Number.isFinite);
    const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    return { col, avg };
  });
  const topCategory = groupCategorical.length
    ? computeCategoricalAnalysis(filteredRows, groupCategorical[0]).top_values[0]
    : null;

  const trendValueCol = groupNumeric[0] ?? null;
  const trendData = dateColumn ? computeTrend(filteredRows, dateColumn, trendValueCol) : [];

  const insight = strongestCorrelation(filteredRows, groupNumeric);

  return (
    <div ref={innerRef} className="space-y-6">
      {/* Row 1 — KPIs */}
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
          Key metrics
        </h3>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            icon={<Rows3 className="h-5 w-5" />}
            label={activeFilterCount > 0 ? "Rows (filtered)" : "Rows"}
            value={filteredRows.length.toLocaleString()}
          />
          {kpiNumeric.map(({ col, avg }) => (
            <StatCard
              key={col}
              icon={<TrendingUp className="h-5 w-5" />}
              label={`Avg ${col}`}
              value={avg !== null ? avg.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}
            />
          ))}
          {topCategory && (
            <StatCard
              icon={<LayoutGrid className="h-5 w-5" />}
              label={`Top ${groupCategorical[0]}`}
              value={topCategory.value}
            />
          )}
        </div>
      </div>

      {/* Row 2 — Slicers */}
      {groupCategorical.length > 0 && (
        <div>
          <div className="mb-2 flex items-center gap-2">
            <SlidersHorizontal className="h-3.5 w-3.5 text-primary-2" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Slicers — filter this report
            </h3>
            {activeFilterCount > 0 && (
              <button
                onClick={onClearFilters}
                className="ml-auto flex items-center gap-1 text-xs text-muted hover:text-foreground"
              >
                <X className="h-3 w-3" /> Clear filters
              </button>
            )}
          </div>
          <div className="space-y-2 rounded-xl border border-surface-border bg-surface-2/30 p-4">
            {groupCategorical.map((col) => (
              <div key={col} className="flex flex-wrap items-center gap-2">
                <span className="w-28 shrink-0 font-mono text-xs text-muted">{col}</span>
                {distinctValues(rawRows, col)
                  .slice(0, 12)
                  .map((v) => {
                    const active = (filters[col] ?? []).includes(v);
                    return (
                      <button
                        key={v}
                        type="button"
                        onClick={() => onToggleFilter(col, v)}
                        className={cn(
                          "rounded-full border px-2.5 py-1 text-xs transition-colors",
                          active
                            ? "border-primary/50 bg-primary/15 text-primary-2"
                            : "border-surface-border bg-surface/60 text-muted hover:text-foreground",
                        )}
                      >
                        {v}
                      </button>
                    );
                  })}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Row 3 — Trends */}
      {dateColumn && trendData.length > 1 && (
        <div>
          <div className="mb-2 flex items-center gap-2">
            <TrendingUp className="h-3.5 w-3.5 text-primary-2" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Trend{trendValueCol ? ` — ${trendValueCol} over time` : " — volume over time"}
            </h3>
          </div>
          <Card>
            <CardContent className="h-56 pt-6">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
                  <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted)" }} />
                  <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="var(--primary-2)"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Row 4 — Charts */}
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
          Breakdown
        </h3>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {group.columns.map((col, i) =>
            groupNumeric.includes(col) ? (
              <NumericHistogramCard key={col} dist={computeNumericDistribution(filteredRows, col)} />
            ) : (
              <CategoricalBarCard
                key={col}
                cat={computeCategoricalAnalysis(filteredRows, col)}
                variant={i % 2 === 0 ? "bar" : "pie"}
              />
            ),
          )}
        </div>
      </div>

      {/* Row 5 — Conclusion */}
      <div>
        <div className="mb-2 flex items-center gap-2">
          <Lightbulb className="h-3.5 w-3.5 text-warning" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">Conclusion</h3>
        </div>
        <Card className="border-warning/20 bg-gradient-to-br from-warning/5 to-transparent">
          <CardContent className="pt-6">
            {insight ? (
              <>
                <p className="text-sm font-semibold">
                  {insight.columnA} <span className="text-muted">vs</span> {insight.columnB}
                </p>
                <p className="mt-1.5 text-sm text-muted">{insight.text}</p>
              </>
            ) : (
              <p className="text-sm text-muted">
                Not enough numeric columns in this dashboard (or rows after filtering) to compute a
                meaningful relationship.
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function DashboardsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<EdaReport | null>(null);
  const [rawRows, setRawRows] = useState<Row[]>([]);
  const [fileLabel, setFileLabel] = useState("dataset");

  const [selected, setSelected] = useState<string[]>([]);
  const [dashboardCount, setDashboardCount] = useState(1);
  const [built, setBuilt] = useState(false);
  const [activeTab, setActiveTab] = useState("0");
  const [filters, setFilters] = useState<Record<string, string[]>>({});

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

  const dateColumn = useMemo(() => {
    if (!report) return null;
    const entry = Object.entries(report.column_kinds).find(([, kind]) => kind === "datetime");
    return entry ? entry[0] : null;
  }, [report]);

  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setBuilt(false);
    try {
      const [data, parsed] = await Promise.all([analyzeEda(file), parseCsvFile(file)]);
      setReport(data);
      setRawRows(parsed.rows);
      setFilters({});
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

  function toggleFilter(column: string, value: string) {
    setFilters((prev) => {
      const current = prev[column] ?? [];
      const next = current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value];
      return { ...prev, [column]: next };
    });
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
        description="Upload a dataset and get multiple domain-themed dashboards — KPIs, slicers, trends, and an auto-generated conclusion, like a real BI report."
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
                <BiDashboardPanel
                  report={report}
                  group={g}
                  rawRows={rawRows}
                  filters={filters}
                  onToggleFilter={toggleFilter}
                  onClearFilters={() => setFilters({})}
                  dateColumn={dateColumn}
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
