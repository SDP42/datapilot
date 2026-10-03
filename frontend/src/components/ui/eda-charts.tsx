"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./card";
import { Badge } from "./badge";
import type {
  CategoricalColumnAnalysis,
  CategoricalContingency,
  CategoricalNumericSummary,
  NumericDistribution,
  NumericPairCorrelation,
} from "@/lib/api";

const CHART_COLORS = [
  "var(--primary)",
  "var(--accent)",
  "var(--accent-2)",
  "var(--primary-2)",
  "var(--success)",
  "var(--warning)",
];

const tooltipStyle = {
  background: "var(--surface-2)",
  border: "1px solid var(--surface-border)",
  borderRadius: 8,
  fontSize: 12,
};

function formatEdge(n: number): string {
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function NumericHistogramCard({ dist }: { dist: NumericDistribution }) {
  const unavailable = dist.status !== "completed" || dist.histogram.status !== "completed";
  const data = dist.histogram.bins.map((bin) => ({
    name: `${formatEdge(bin.left_edge)}–${formatEdge(bin.right_edge)}`,
    count: bin.count,
  }));

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle className="text-base">
            Distribution of <span className="font-mono">{dist.column}</span>
          </CardTitle>
          <CardDescription>
            how values of {dist.column} are spread · mean {dist.mean?.toFixed(2) ?? "—"} · median{" "}
            {dist.median?.toFixed(2) ?? "—"}
          </CardDescription>
        </div>
        <Badge variant="primary">numeric</Badge>
      </CardHeader>
      <CardContent className="h-56">
        {unavailable ? (
          <div className="flex h-full items-center justify-center text-sm text-muted">
            {dist.reason ?? dist.histogram.reason ?? "No distribution available"}
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--muted)" }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} allowDecimals={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

export function CategoricalBarCard({
  cat,
  variant = "bar",
}: {
  cat: CategoricalColumnAnalysis;
  variant?: "bar" | "pie";
}) {
  const data = cat.top_values.map((tv) => ({ name: tv.value, count: tv.count }));

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle className="text-base">
            <span className="font-mono">{cat.column}</span> breakdown
          </CardTitle>
          <CardDescription>
            count of rows per {cat.column} value · {cat.unique_count} unique values
          </CardDescription>
        </div>
        <Badge variant="accent">categorical</Badge>
      </CardHeader>
      <CardContent className="h-56">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted">No values</div>
        ) : variant === "pie" ? (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} dataKey="count" nameKey="name" innerRadius="45%" outerRadius="80%" paddingAngle={2}>
                {data.map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
              <XAxis type="number" tick={{ fontSize: 11, fill: "var(--muted)" }} allowDecimals={false} />
              <YAxis
                type="category"
                dataKey="name"
                width={90}
                tick={{ fontSize: 11, fill: "var(--muted)" }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {data.map((_, i) => (
                  <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

/** Ranks every computed numeric-pair correlation by |r| and charts the
 * strongest ones, each bar labeled "A vs B" — the "what against what"
 * a correlation number alone never tells you. */
export function CorrelationRankingCard({
  correlations,
  topN = 8,
}: {
  correlations: NumericPairCorrelation[];
  topN?: number;
}) {
  const ranked = [...correlations]
    .filter((c) => c.correlation !== null)
    .sort((a, b) => Math.abs(b.correlation!) - Math.abs(a.correlation!))
    .slice(0, topN)
    .map((c) => ({
      name: `${c.column_a} vs ${c.column_b}`,
      correlation: c.correlation!,
    }));

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle className="text-base">Strongest numeric relationships</CardTitle>
        <CardDescription>
          Pearson correlation between every pair of numeric columns — ranked by strength,
          positive (blue) vs. negative (red)
        </CardDescription>
      </CardHeader>
      <CardContent className="h-64">
        {ranked.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted">
            Not enough numeric column pairs to compute a correlation.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ranked} layout="vertical" margin={{ left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
              <XAxis
                type="number"
                domain={[-1, 1]}
                tick={{ fontSize: 11, fill: "var(--muted)" }}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={140}
                tick={{ fontSize: 10, fill: "var(--muted)" }}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)}
              />
              <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
                {ranked.map((r, i) => (
                  <Cell key={i} fill={r.correlation >= 0 ? "var(--primary-2)" : "var(--danger)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

/** One categorical-vs-numeric summary — "Average {numeric} by {categorical}" —
 * the grouped-mean chart that a correlation can't show for a non-numeric column. */
export function GroupedMeanBarCard({ summary }: { summary: CategoricalNumericSummary }) {
  const data = summary.groups
    .filter((g) => g.mean !== null)
    .map((g) => ({ name: g.category, mean: g.mean! }));

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle className="text-base">
          Average <span className="font-mono">{summary.numeric_column}</span> by{" "}
          <span className="font-mono">{summary.categorical_column}</span>
        </CardTitle>
        <CardDescription>
          mean {summary.numeric_column} for each {summary.categorical_column} value
        </CardDescription>
      </CardHeader>
      <CardContent className="h-56">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted">
            No groups with a computable mean.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--muted)" }} />
              <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="mean" fill="var(--accent)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

/** One categorical-vs-categorical contingency — a stacked bar of counts,
 * titled "{A} vs {B}" so the relationship being shown is always explicit. */
export function ContingencyBarCard({ contingency }: { contingency: CategoricalContingency }) {
  const bByA = new Map<string, Record<string, number>>();
  const bValues = new Set<string>();
  for (const row of contingency.rows) {
    bValues.add(row.category_b);
    const entry = bByA.get(row.category_a) ?? {};
    entry[row.category_b] = row.count;
    bByA.set(row.category_a, entry);
  }
  const bList = [...bValues].slice(0, 6);
  const data = [...bByA.entries()].slice(0, 10).map(([a, counts]) => ({
    name: a,
    ...Object.fromEntries(bList.map((b) => [b, counts[b] ?? 0])),
  }));

  return (
    <Card className="overflow-hidden sm:col-span-2">
      <CardHeader>
        <CardTitle className="text-base">
          <span className="font-mono">{contingency.column_a}</span> vs{" "}
          <span className="font-mono">{contingency.column_b}</span>
        </CardTitle>
        <CardDescription>
          row counts for every {contingency.column_a} / {contingency.column_b} combination
        </CardDescription>
      </CardHeader>
      <CardContent className="h-64">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted">
            No combinations to show.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--muted)" }} />
              <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} allowDecimals={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {bList.map((b, i) => (
                <Bar key={b} dataKey={b} stackId="a" fill={CHART_COLORS[i % CHART_COLORS.length]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
