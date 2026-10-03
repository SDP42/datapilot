"use client";

import { useMemo, useState } from "react";
import { toast } from "sonner";
import { Boxes, Trophy, Layers, Timer, Sparkles } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { StatusBadge } from "@/components/ui/status-badge";
import { Badge } from "@/components/ui/badge";
import { Alert } from "@/components/ui/alert";
import { StatCard } from "@/components/ui/stat-card";
import { runClusterSearch, ApiError, ExpandedSearchResult } from "@/lib/api";

const CLUSTERING_METRIC_ORDER = ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"];

export default function ClusteringPage() {
  const [file, setFile] = useState<File | null>(null);
  const [objective, setObjective] = useState("cluster rows into segments");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ExpandedSearchResult | null>(null);

  async function handleRun() {
    if (!file || !objective) return;
    setRunning(true);
    try {
      const data = await runClusterSearch(file, objective);
      setResult(data);
      if (data.status === "completed") {
        toast.success(`${data.candidate_count} clustering candidates ranked`);
      } else {
        toast.warning(data.reason ?? "Clustering search could not complete");
      }
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Clustering search failed");
    } finally {
      setRunning(false);
    }
  }

  const metricColumns = useMemo(() => {
    if (!result || result.candidates.length === 0) return [];
    const present = new Set<string>();
    for (const c of result.candidates) {
      for (const k of Object.keys(c.metrics)) present.add(k);
    }
    const ordered = CLUSTERING_METRIC_ORDER.filter((k) => present.has(k));
    const rest = [...present].filter((k) => !ordered.includes(k));
    return [...ordered, ...rest];
  }, [result]);

  const best = result?.candidates.find((c) => c.rank === 1) ?? null;

  return (
    <div>
      <PageHeader
        title="Clustering"
        description="Unsupervised segmentation — KMeans, Agglomerative, DBSCAN, and Gaussian Mixture, swept across a fixed hyperparameter grid and ranked by silhouette score."
      />

      <Card>
        <CardHeader>
          <CardTitle>Run the clustering search</CardTitle>
          <CardDescription>
            No target column needed — every row is grouped by similarity across its numeric and
            categorical features.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={file} onFileSelect={setFile} />
          <div>
            <Label htmlFor="objective">Objective</Label>
            <Input
              id="objective"
              placeholder="e.g. cluster customers into segments"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
            />
            <p className="mt-1.5 text-xs text-muted">
              Describe it as grouping/segmentation — a prediction-shaped objective (e.g. &quot;predict
              churn&quot;) will be inferred as a different task and this search will report
              unavailable.
            </p>
          </div>
          <Button onClick={handleRun} disabled={!file || !objective} loading={running}>
            <Boxes className="h-4 w-4" /> Run clustering search (50+ candidates)
          </Button>
        </CardContent>
      </Card>

      {result && result.status === "completed" && (
        <div className="mt-8 space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              icon={<Layers className="h-5 w-5" />}
              label="Candidates trained"
              value={result.candidate_count}
            />
            <StatCard
              icon={<Timer className="h-5 w-5" />}
              label="Total fit time"
              value={`${result.total_fit_seconds.toFixed(2)}s`}
            />
            <StatCard
              icon={<Sparkles className="h-5 w-5" />}
              label="Ranked by"
              value={result.selection_metric ?? "—"}
            />
            <StatCard
              icon={<Boxes className="h-5 w-5" />}
              label="Best cluster count"
              value={
                typeof best?.hyperparameters.n_clusters === "number"
                  ? best.hyperparameters.n_clusters
                  : "—"
              }
            />
          </div>

          {best && (
            <Card className="border-primary/30 bg-gradient-to-br from-primary/10 to-transparent">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="h-4 w-4 text-warning" /> Best clustering candidate
                </CardTitle>
                <CardDescription>
                  Scored best on silhouette score — how well-separated and internally cohesive
                  the clusters are, from -1 (overlapping) to 1 (perfectly separated).
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-wrap items-center gap-3">
                <Badge variant="primary">{best.family}</Badge>
                <span className="font-mono text-sm">{best.estimator_name}</span>
                <span className="font-mono text-xs text-muted">
                  {Object.entries(best.hyperparameters)
                    .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(",")}]` : v}`)
                    .join(", ") || "default hyperparameters"}
                </span>
                <Badge variant="success">
                  silhouette: {best.metrics.silhouette_score?.toFixed(4) ?? "—"}
                </Badge>
                <span className="ml-auto text-xs text-muted">
                  fit in {best.fit_seconds.toFixed(3)}s
                </span>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>All {result.candidate_count} candidates, ranked</CardTitle>
              <CardDescription>
                Every clustering metric for every candidate, not just silhouette score.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="max-h-[32rem] overflow-y-auto rounded-xl border border-surface-border">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-surface-2/90 backdrop-blur">
                    <tr className="border-b border-surface-border">
                      <th className="px-3 py-2.5 text-left font-medium text-muted">#</th>
                      <th className="px-3 py-2.5 text-left font-medium text-muted">Family</th>
                      <th className="px-3 py-2.5 text-left font-medium text-muted">Estimator</th>
                      <th className="px-3 py-2.5 text-left font-medium text-muted">
                        Hyperparameters
                      </th>
                      {metricColumns.map((m) => (
                        <th
                          key={m}
                          className="px-3 py-2.5 text-right font-medium uppercase text-muted"
                        >
                          {m}
                        </th>
                      ))}
                      <th className="px-3 py-2.5 text-right font-medium text-muted">Fit time</th>
                      <th className="px-3 py-2.5 text-left font-medium text-muted">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.candidates.map((c) => (
                      <tr
                        key={`${c.rank}-${c.estimator_name}-${JSON.stringify(c.hyperparameters)}`}
                        className={
                          "border-b border-surface-border/60 last:border-0 " +
                          (c.rank === 1 ? "bg-primary/10" : "hover:bg-white/[0.02]")
                        }
                      >
                        <td className="px-3 py-2.5 font-mono text-xs">
                          {c.rank === 1 ? (
                            <Badge variant="primary">
                              <Trophy className="h-3 w-3" /> 1
                            </Badge>
                          ) : (
                            c.rank
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-xs text-muted">{c.family}</td>
                        <td className="px-3 py-2.5 font-mono text-xs">{c.estimator_name}</td>
                        <td className="px-3 py-2.5 font-mono text-[11px] text-muted">
                          {Object.entries(c.hyperparameters)
                            .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(",")}]` : v}`)
                            .join(", ") || "—"}
                        </td>
                        {metricColumns.map((m) => (
                          <td key={m} className="px-3 py-2.5 text-right font-mono text-xs">
                            {c.metrics[m] !== undefined ? c.metrics[m].toFixed(4) : "—"}
                          </td>
                        ))}
                        <td className="px-3 py-2.5 text-right font-mono text-xs text-muted">
                          {c.fit_seconds.toFixed(3)}s
                        </td>
                        <td className="px-3 py-2.5">
                          <StatusBadge status={c.status === "completed" ? "completed" : "failed"} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {result && result.status !== "completed" && (
        <Alert variant="warning" title={`Clustering status: ${result.status}`} className="mt-8">
          {result.reason ?? "The clustering search could not complete for this dataset."}
        </Alert>
      )}
    </div>
  );
}
