"use client";

import { useMemo, useState } from "react";
import { toast } from "sonner";
import { Cpu, Trophy, Layers, Timer, Sparkles } from "lucide-react";
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
import { StatCard } from "@/components/ui/stat-card";
import {
  runModeling,
  runModelSearch,
  ApiError,
  ModelingSpec,
  ExpandedSearchResult,
} from "@/lib/api";

const REGRESSION_METRIC_ORDER = ["mse", "rmse", "mae", "r2"];
const CLASSIFICATION_METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc"];

export default function ModelingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [objective, setObjective] = useState("");
  const [loading, setLoading] = useState(false);
  const [spec, setSpec] = useState<ModelingSpec | null>(null);

  const [searching, setSearching] = useState(false);
  const [search, setSearch] = useState<ExpandedSearchResult | null>(null);
  const [crossValidate, setCrossValidate] = useState(false);

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

  async function handleSearch() {
    if (!file || !objective) return;
    setSearching(true);
    try {
      const data = await runModelSearch(file, objective, crossValidate);
      setSearch(data);
      if (data.status === "completed") {
        toast.success(
          data.cross_validation_enabled
            ? `${data.candidate_count} candidates trained and cross-validated`
            : `${data.candidate_count} candidates trained and ranked`,
        );
      } else {
        toast.warning(data.reason ?? "Model search could not complete");
      }
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Model search failed");
    } finally {
      setSearching(false);
    }
  }

  const metricColumns = useMemo(() => {
    if (!search || search.candidates.length === 0) return [];
    const present = new Set<string>();
    for (const c of search.candidates) {
      for (const k of Object.keys(c.metrics)) present.add(k);
    }
    const order = present.has("rmse") ? REGRESSION_METRIC_ORDER : CLASSIFICATION_METRIC_ORDER;
    const cvOrder = [...present].filter((k) => k.startsWith("cv_")).sort();
    const ordered = order.filter((k) => present.has(k));
    const rest = [...present].filter((k) => !ordered.includes(k) && !cvOrder.includes(k));
    return [...ordered, ...cvOrder, ...rest];
  }, [search]);

  const bestCandidate = search?.candidates.find((c) => c.rank === 1) ?? null;

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
          <div className="flex flex-wrap gap-3">
            <Button onClick={handleRun} disabled={!file || !objective} loading={loading}>
              <Cpu className="h-4 w-4" /> Run modeling pipeline
            </Button>
            <Button
              variant="secondary"
              onClick={handleSearch}
              disabled={!file || !objective}
              loading={searching}
            >
              <Layers className="h-4 w-4" /> Run full model search (100+ candidates)
            </Button>
          </div>
          <label className="flex items-center gap-2 text-sm text-muted">
            <input
              type="checkbox"
              checked={crossValidate}
              onChange={(e) => setCrossValidate(e.target.checked)}
              className="h-4 w-4 rounded border-surface-border accent-[var(--primary)]"
            />
            5-fold cross-validation (more reliable accuracy estimate, ~5x slower)
          </label>
          <p className="text-xs text-muted">
            The pipeline above fits one baseline per family. Full search fits every
            (estimator, hyperparameter) combination in the Phase 7.7 catalog — 100+ candidates,
            each with real fit timing and every metric shown, not just the selection metric — and
            ranks all of them by a fixed hyperparameter grid per family. A single train/test split
            can be an unreliable estimate of real-world accuracy; enabling cross-validation scores
            every candidate across 5 folds and ranks by that averaged result instead.
          </p>
        </CardContent>
      </Card>

      {search && search.status === "completed" && (
        <div className="mt-8 space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              icon={<Layers className="h-5 w-5" />}
              label="Candidates trained"
              value={search.candidate_count}
            />
            <StatCard
              icon={<Timer className="h-5 w-5" />}
              label="Total fit time"
              value={`${search.total_fit_seconds.toFixed(2)}s`}
            />
            <StatCard
              icon={<Sparkles className="h-5 w-5" />}
              label="Ranked by"
              value={
                search.cross_validation_enabled
                  ? `cv_${search.selection_metric}_mean`
                  : (search.selection_metric ?? "—")
              }
            />
            <StatCard
              icon={<Trophy className="h-5 w-5" />}
              label="Task type"
              value={search.task_type ?? "—"}
            />
          </div>

          {bestCandidate && (
            <Card className="border-primary/30 bg-gradient-to-br from-primary/10 to-transparent">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="h-4 w-4 text-warning" /> Best candidate &amp; its exact
                  hyperparameters
                </CardTitle>
                <CardDescription>
                  This is the hyperparameter-tuning result — every candidate below was fit with a
                  different, fixed configuration; this one scored best on{" "}
                  {search.cross_validation_enabled
                    ? `cv_${search.selection_metric}_mean (5-fold cross-validated)`
                    : search.selection_metric}
                  .
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-wrap items-center gap-3">
                <Badge variant="primary">{bestCandidate.family}</Badge>
                <span className="font-mono text-sm">{bestCandidate.estimator_name}</span>
                <span className="font-mono text-xs text-muted">
                  {Object.entries(bestCandidate.hyperparameters)
                    .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(",")}]` : v}`)
                    .join(", ") || "default hyperparameters"}
                </span>
                {search.cross_validation_enabled &&
                bestCandidate.metrics[`cv_${search.selection_metric}_mean`] !== undefined ? (
                  <>
                    <Badge variant="success">
                      cv_{search.selection_metric}_mean:{" "}
                      {bestCandidate.metrics[`cv_${search.selection_metric}_mean`]?.toFixed(4)}
                    </Badge>
                    <Badge variant="default">
                      ±{bestCandidate.metrics[`cv_${search.selection_metric}_std`]?.toFixed(4) ?? "—"}
                    </Badge>
                  </>
                ) : (
                  <Badge variant="success">
                    {search.selection_metric}:{" "}
                    {bestCandidate.metrics[search.selection_metric ?? ""]?.toFixed(4) ?? "—"}
                  </Badge>
                )}
                <span className="ml-auto text-xs text-muted">
                  fit in {bestCandidate.fit_seconds.toFixed(3)}s
                </span>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>All {search.candidate_count} candidates, ranked</CardTitle>
              <CardDescription>
                Every metric for every candidate — not just the selection metric — so you can
                compare families yourself rather than trust the ranking alone.
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
                    {search.candidates.map((c) => (
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

      {search && search.status !== "completed" && (
        <Alert variant="warning" title={`Search status: ${search.status}`} className="mt-8">
          {search.reason ?? "The search could not complete for this dataset."}
        </Alert>
      )}

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
