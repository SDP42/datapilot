"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./card";
import { Badge } from "./badge";
import type { CategoricalColumnAnalysis, NumericDistribution } from "@/lib/api";

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
          <CardTitle className="font-mono text-base">{dist.column}</CardTitle>
          <CardDescription>
            numeric · mean {dist.mean?.toFixed(2) ?? "—"} · median {dist.median?.toFixed(2) ?? "—"}
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
          <CardTitle className="font-mono text-base">{cat.column}</CardTitle>
          <CardDescription>categorical · {cat.unique_count} unique values</CardDescription>
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
