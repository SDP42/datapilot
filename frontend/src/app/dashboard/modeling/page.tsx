"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Cpu, Trophy } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { StatusBadge } from "@/components/ui/status-badge";
import { Badge } from "@/components/ui/badge";
import { Alert } from "@/components/ui/alert";
import { runModeling, ApiError, ModelingSpec } from "@/lib/api";

export default function ModelingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [objective, setObjective] = useState("");
  const [loading, setLoading] = useState(false);
  const [spec, setSpec] = useState<ModelingSpec | null>(null);

  async function handleRun() {
    if (!file || !objective) return;
    setLoading(true);
    try {
      const data = await runModeling(file, objective);
      setSpec(data);
      toast.success(`Modeling ${data.status}`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Modeling run failed");
    } finally {
      setLoading(false);
    }
  }

  const chartData =
    spec?.selection.ranking
      .filter((r) => r.score !== null && r.score !== undefined)
      .map((r) => ({
        name: r.estimator_name ?? r.family,
        score: r.score as number,
      })) ?? [];

  return (
    <div>
      <PageHeader
        title="Model development"
        description="Phase 7 — readiness, split planning, candidate generation, baseline training, and selection."
      />

      <Card>
        <CardHeader>
          <CardTitle>Run the pipeline</CardTitle>
          <CardDescription>Fits one conservative baseline per candidate family.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={file} onFileSelect={setFile} />
          <div>
            <Label htmlFor="objective">Objective</Label>
            <Input
              id="objective"
              placeholder="e.g. predict churn"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
            />
          </div>
          <Button onClick={handleRun} disabled={!file || !objective} loading={loading}>
            <Cpu className="h-4 w-4" /> Run modeling pipeline
          </Button>
        </CardContent>
      </Card>

      {spec && (
        <div className="mt-8 space-y-6">
          {spec.status !== "completed" ? (
            <Alert variant="warning" title={`Status: ${spec.status}`}>
              {spec.reason ?? "The pipeline could not complete for this dataset."}
            </Alert>
          ) : (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Trophy className="h-4 w-4 text-warning" /> Selected model
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-wrap items-center gap-3">
                  <Badge variant="primary">{spec.selection.selected_family}</Badge>
                  <span className="font-mono text-sm text-muted">
                    {spec.selection.selected_estimator}
                  </span>
                  <Badge variant="success">
                    {spec.selection.selection_metric}: {spec.selection.selected_score?.toFixed(4)}
                  </Badge>
                </CardContent>
              </Card>

              {chartData.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Candidate ranking</CardTitle>
                    <CardDescription>By {spec.selection.selection_metric}</CardDescription>
                  </CardHeader>
                  <CardContent className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
                        <XAxis dataKey="name" tick={{ fontSize: 11, fill: "var(--muted)" }} />
                        <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} />
                        <Tooltip
                          contentStyle={{
                            background: "var(--surface-2)",
                            border: "1px solid var(--surface-border)",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                        />
                        <Bar dataKey="score" fill="var(--primary)" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardHeader>
                  <CardTitle>Training runs</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {spec.training.runs.map((run, i) => (
                    <div
                      key={i}
                      className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-surface-border px-4 py-3"
                    >
                      <span className="text-sm font-medium">{run.family}</span>
                      <div className="flex items-center gap-3">
                        {Object.entries(run.metrics).map(([k, v]) => (
                          <span key={k} className="font-mono text-xs text-muted">
                            {k}={v.toFixed(3)}
                          </span>
                        ))}
                        <StatusBadge status={run.status} />
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </>
          )}
        </div>
      )}
    </div>
  );
}
