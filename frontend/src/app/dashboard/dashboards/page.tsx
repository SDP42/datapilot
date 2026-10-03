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
  Search,
  CalendarRange,
  ListFilter,
  Hash,
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
  applyAllFilters,
  computeCategoricalAnalysis,
  computeNumericDistribution,
  computeTrend,
  distinctValues,
  numericBounds,
  strongestCorrelation,
  type NumericRange,
  type Row,
} from "@/lib/bi-stats";
import { analyzeEda, ApiError, EdaReport } from "@/lib/api";

const tooltipStyle = {
  background: "var(--surface-2)",
  border: "1px solid var(--surface-border)",
  borderRadius: 8,
  fontSize: 12,
};

const SINGLE_SELECT_THRESHOLD = 8;

interface FilterState {
  categorical: Record<string, string[]>;
  numericRanges: Record<string, NumericRange>;
  dateStart: string | null;
  dateEnd: string | null;
  search: string;
  topN: number;
}

const EMPTY_FILTERS: FilterState = {
  categorical: {},
  numericRanges: {},
  dateStart: null,
  dateEnd: null,
  search: "",
  topN: 8,
};

function SlicerSection({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted">
        <Icon className="h-3 w-3" /> {title}
      </div>
      {children}
    </div>
  );
}

function BiDashboardPanel({
  report,
  group,
  rawRows,
  filters,
  onFiltersChange,
  dateColumn,
  innerRef,
}: {
  report: EdaReport;
  group: DashboardGroup;
  rawRows: Row[];
  filters: FilterState;
  onFiltersChange: (next: FilterState) => void;
  dateColumn: string | null;
  innerRef?: React.Ref<HTMLDivElement>;
}) {
  const groupNumeric = useMemo(
    () => group.columns.filter((c) => report.univariate.numeric.some((n) => n.column === c)),
    [group.columns, report.univariate.numeric],
  );
  const groupCategorical = useMemo(
    () => group.columns.filter((c) => report.univariate.categorical.some((n) => n.column === c)),
    [group.columns, report.univariate.categorical],
  );
  const lowCardinality = useMemo(
    () => groupCategorical.filter((c) => distinctValues(rawRows, c).length <= SINGLE_SELECT_THRESHOLD),
    [groupCategorical, rawRows],
  );
  const highCardinality = useMemo(
    () => groupCategorical.filter((c) => distinctValues(rawRows, c).length > SINGLE_SELECT_THRESHOLD),
    [groupCategorical, rawRows],
  );

  const filteredRows = useMemo(
    () =>
      applyAllFilters(rawRows, {
        categorical: filters.categorical,
        numericRanges: filters.numericRanges,
        dateRange: dateColumn
          ? { column: dateColumn, start: filters.dateStart, end: filters.dateEnd }
          : null,
        search: groupCategorical.length
          ? { columns: groupCategorical, query: filters.search }
          : null,
      }),
    [rawRows, filters, dateColumn, groupCategorical],
  );

  const activeFilterCount =
    Object.values(filters.categorical).reduce((a, v) => a + v.length, 0) +
    Object.values(filters.numericRanges).filter((r) => r.min !== null || r.max !== null).length +
    (filters.dateStart || filters.dateEnd ? 1 : 0) +
    (filters.search.trim() ? 1 : 0);

  function toggleCategorical(column: string, value: string) {
    const current = filters.categorical[column] ?? [];
    const next = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value];
    onFiltersChange({ ...filters, categorical: { ...filters.categorical, [column]: next } });
  }

  function setSingleSelect(column: string, value: string) {
    onFiltersChange({
      ...filters,
      categorical: { ...filters.categorical, [column]: value ? [value] : [] },
    });
  }

  function setRange(column: string, patch: Partial<NumericRange>) {
    const current = filters.numericRanges[column] ?? { min: null, max: null };
    onFiltersChange({
      ...filters,
      numericRanges: { ...filters.numericRanges, [column]: { ...current, ...patch } },
    });
  }

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

      {/* Row 2 — Slicers (6 types) */}
      {(groupCategorical.length > 0 || groupNumeric.length > 0 || dateColumn) && (
        <div>
          <div className="mb-2 flex items-center gap-2">
            <SlidersHorizontal className="h-3.5 w-3.5 text-primary-2" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Slicers — filter this report
            </h3>
            {activeFilterCount > 0 && (
              <button
                onClick={() => onFiltersChange(EMPTY_FILTERS)}
                className="ml-auto flex items-center gap-1 text-xs text-muted hover:text-foreground"
              >
                <X className="h-3 w-3" /> Clear all filters
              </button>
            )}
          </div>
          <div className="grid gap-4 rounded-xl border border-surface-border bg-surface-2/30 p-4 sm:grid-cols-2">
            {groupCategorical.length > 0 && (
              <SlicerSection title="Search (across categories)" icon={Search}>
                <div className="relative">
                  <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted" />
                  <Input
                    value={filters.search}
                    onChange={(e) => onFiltersChange({ ...filters, search: e.target.value })}
                    placeholder="Type to filter rows…"
                    className="h-9 pl-8 text-sm"
                  />
                </div>
              </SlicerSection>
            )}

            {dateColumn && (
              <SlicerSection title={`Date range — ${dateColumn}`} icon={CalendarRange}>
                <div className="flex items-center gap-2">
                  <input
                    type="date"
                    value={filters.dateStart ?? ""}
                    onChange={(e) =>
                      onFiltersChange({ ...filters, dateStart: e.target.value || null })
                    }
                    className="h-9 w-full rounded-lg border border-surface-border bg-surface-2/60 px-2 text-xs text-foreground outline-none focus:border-primary"
                  />
                  <span className="text-xs text-muted">to</span>
                  <input
                    type="date"
                    value={filters.dateEnd ?? ""}
                    onChange={(e) =>
                      onFiltersChange({ ...filters, dateEnd: e.target.value || null })
                    }
                    className="h-9 w-full rounded-lg border border-surface-border bg-surface-2/60 px-2 text-xs text-foreground outline-none focus:border-primary"
                  />
                </div>
              </SlicerSection>
            )}

            {lowCardinality.length > 0 && (
              <SlicerSection title="Category filters (multi-select)" icon={ListFilter}>
                <div className="space-y-2">
                  {lowCardinality.map((col) => (
                    <div key={col} className="flex flex-wrap items-center gap-1.5">
                      <span className="w-24 shrink-0 truncate font-mono text-[11px] text-muted">
                        {col}
                      </span>
                      {distinctValues(rawRows, col).map((v) => {
                        const active = (filters.categorical[col] ?? []).includes(v);
                        return (
                          <button
                            key={v}
                            type="button"
                            onClick={() => toggleCategorical(col, v)}
                            className={cn(
                              "rounded-full border px-2 py-0.5 text-[11px] transition-colors",
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
              </SlicerSection>
            )}

            {highCardinality.length > 0 && (
              <SlicerSection title="Quick pick (single-select)" icon={ListFilter}>
                <div className="space-y-2">
                  {highCardinality.map((col) => (
                    <div key={col} className="flex items-center gap-2">
                      <span className="w-24 shrink-0 truncate font-mono text-[11px] text-muted">
                        {col}
                      </span>
                      <select
                        value={filters.categorical[col]?.[0] ?? ""}
                        onChange={(e) => setSingleSelect(col, e.target.value)}
                        className="h-8 w-full rounded-lg border border-surface-border bg-surface-2/60 px-2 text-xs text-foreground outline-none focus:border-primary"
                      >
                        <option value="">All</option>
                        {distinctValues(rawRows, col).map((v) => (
                          <option key={v} value={v}>
                            {v}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
              </SlicerSection>
            )}

            {groupNumeric.length > 0 && (
              <SlicerSection title="Numeric range" icon={SlidersHorizontal}>
                <div className="space-y-2">
                  {groupNumeric.slice(0, 2).map((col) => {
                    const bounds = numericBounds(rawRows, col);
                    const current = filters.numericRanges[col] ?? { min: null, max: null };
                    return (
                      <div key={col} className="flex items-center gap-2">
                        <span className="w-24 shrink-0 truncate font-mono text-[11px] text-muted">
                          {col}
                        </span>
                        <input
                          type="number"
                          placeholder={bounds ? bounds.min.toFixed(0) : "min"}
                          value={current.min ?? ""}
                          onChange={(e) =>
                            setRange(col, { min: e.target.value ? Number(e.target.value) : null })
                          }
                          className="h-8 w-full rounded-lg border border-surface-border bg-surface-2/60 px-2 text-xs text-foreground outline-none focus:border-primary"
                        />
                        <span className="text-xs text-muted">–</span>
                        <input
                          type="number"
                          placeholder={bounds ? bounds.max.toFixed(0) : "max"}
                          value={current.max ?? ""}
                          onChange={(e) =>
                            setRange(col, { max: e.target.value ? Number(e.target.value) : null })
                          }
                          className="h-8 w-full rounded-lg border border-surface-border bg-surface-2/60 px-2 text-xs text-foreground outline-none focus:border-primary"
                        />
                      </div>
                    );
                  })}
                </div>
              </SlicerSection>
            )}

            {groupCategorical.length > 0 && (
              <SlicerSection title="Top N (chart display limit)" icon={Hash}>
                <select
                  value={filters.topN}
                  onChange={(e) => onFiltersChange({ ...filters, topN: Number(e.target.value) })}
                  className="h-9 w-full rounded-lg border border-surface-border bg-surface-2/60 px-2 text-xs text-foreground outline-none focus:border-primary"
                >
                  {[3, 5, 8, 10, 15, 20].map((n) => (
                    <option key={n} value={n}>
                      Top {n} categories
                    </option>
                  ))}
                </select>
              </SlicerSection>
            )}
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
                cat={computeCategoricalAnalysis(filteredRows, col, filters.topN)}
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
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);

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
      setFilters(EMPTY_FILTERS);
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
        description="Upload a dataset and get multiple domain-themed dashboards — KPIs, six kinds of slicers, trends, and an auto-generated conclusion, like a real BI report."
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
                  onFiltersChange={setFilters}
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
