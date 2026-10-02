"use client";

import { useState } from "react";
import { toast } from "sonner";
import { LineChart, Rows3, Columns3, AlertTriangle, Code2 } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Accordion } from "@/components/ui/accordion";
import { StatCard } from "@/components/ui/stat-card";
import { NumericHistogramCard, CategoricalBarCard } from "@/components/ui/eda-charts";
import { analyzeEda, ApiError, EdaReport } from "@/lib/api";

export default function EdaPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<EdaReport | null>(null);

  async function handleRun() {
    if (!file) return;
    setLoading(true);
    try {
      const data = await analyzeEda(file);
      setReport(data);
      toast.success("EDA report generated");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "EDA failed");
    } finally {
      setLoading(false);
    }
  }

  const rawSections = report
    ? Object.entries(report).filter(([key, value]) => key !== "dataset_id" && value !== null)
    : [];

  return (
    <div>
      <PageHeader
        title="Exploratory data analysis"
        description="Phase 4 — univariate / bivariate statistics, hypothesis tests, effect sizes, and real charts, not just numbers."
      />

      <Card>
        <CardHeader>
          <CardTitle>Run EDA</CardTitle>
          <CardDescription>Deterministic and analysis-only.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={file} onFileSelect={setFile} />
          <Button onClick={handleRun} disabled={!file} loading={loading}>
            <LineChart className="h-4 w-4" /> Run EDA
          </Button>
        </CardContent>
      </Card>

      {report && (
        <div className="mt-8 space-y-8">
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
              icon={<LineChart className="h-5 w-5" />}
              label="Charted columns"
              value={report.univariate.numeric.length + report.univariate.categorical.length}
            />
          </div>

          {report.univariate.numeric.length > 0 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold tracking-tight">Numeric distributions</h2>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {report.univariate.numeric.map((col) => {
                  const dist = report.distribution.columns.find((d) => d.column === col.column);
                  return dist ? (
                    <NumericHistogramCard key={col.column} dist={dist} />
                  ) : (
                    <NumericHistogramCard
                      key={col.column}
                      dist={{
                        column: col.column,
                        status: "unavailable",
                        reason: "No distribution computed for this column",
                        count: col.count,
                        minimum: col.minimum,
                        maximum: col.maximum,
                        mean: col.mean,
                        median: col.median,
                        std: col.std,
                        histogram: { status: "unavailable", n_bins: null, bin_edges: [], bins: [], total_count: null },
                      }}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {report.univariate.categorical.length > 0 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold tracking-tight">Categorical breakdowns</h2>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {report.univariate.categorical.map((col) => (
                  <CategoricalBarCard key={col.column} cat={col} />
                ))}
              </div>
            </div>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code2 className="h-4 w-4 text-muted" /> Raw report sections
              </CardTitle>
              <CardDescription>
                Full statistical detail — hypothesis tests, effect sizes, bivariate analysis.
                dataset_id: {report.dataset_id}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Accordion
                items={rawSections.map(([key, value]) => ({
                  title: key,
                  content: (
                    <pre className="max-h-80 overflow-auto whitespace-pre-wrap font-mono text-xs">
                      {JSON.stringify(value, null, 2)}
                    </pre>
                  ),
                }))}
              />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
