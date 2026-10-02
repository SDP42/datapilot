"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Database, Play } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Alert } from "@/components/ui/alert";
import { DataTable } from "@/components/ui/data-table";
import { queryAnalytics, ApiError } from "@/lib/api";

const DEFAULT_SQL = "select source, status, count(*) as n from experiments group by 1, 2";

export default function AnalyticsPage() {
  const [experimentIds, setExperimentIds] = useState("");
  const [sql, setSql] = useState(DEFAULT_SQL);
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    try {
      const ids = experimentIds
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const data = await queryAnalytics(ids, sql);
      setRows(data.rows);
      toast.success(`${data.rows.length} row(s) returned`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Query failed");
      setRows(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Experiment analytics"
        description="Phase 13.4 — ad-hoc, read-only DuckDB SQL over already-recorded experiments."
      />

      <Card>
        <CardHeader>
          <CardTitle>Run a query</CardTitle>
          <CardDescription>
            The table is named <code className="font-mono text-primary-2">experiments</code>.
            Requires the optional DuckDB extra on the backend.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="exp-ids">Experiment IDs (comma-separated)</Label>
            <Input
              id="exp-ids"
              placeholder="e.g. 9fffac2d-..., 7a14dd5e-..."
              value={experimentIds}
              onChange={(e) => setExperimentIds(e.target.value)}
            />
          </div>
          <div>
            <Label htmlFor="sql">SQL</Label>
            <textarea
              id="sql"
              value={sql}
              onChange={(e) => setSql(e.target.value)}
              rows={4}
              className="w-full rounded-xl border border-surface-border bg-surface-2/60 px-4 py-3 font-mono text-xs outline-none focus:border-primary focus:ring-2 focus:ring-primary/25"
            />
          </div>
          <Button onClick={handleRun} loading={loading}>
            <Play className="h-4 w-4" /> Run query
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="warning" title="Query unavailable" className="mt-6">
          {error}
        </Alert>
      )}

      {rows && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-4 w-4" /> Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <DataTable
              columns={rows[0] ? Object.keys(rows[0]) : []}
              rows={rows}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
