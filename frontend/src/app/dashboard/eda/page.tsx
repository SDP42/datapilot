"use client";

import { useState } from "react";
import { toast } from "sonner";
import { LineChart } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Accordion } from "@/components/ui/accordion";
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

  const sections = report
    ? Object.entries(report).filter(([key, value]) => key !== "dataset_id" && value !== null)
    : [];

  return (
    <div>
      <PageHeader
        title="Exploratory data analysis"
        description="Phase 4 — univariate / bivariate statistics, hypothesis tests, effect sizes, and visualization recommendations."
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
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Report sections</CardTitle>
            <CardDescription>dataset_id: {report.dataset_id}</CardDescription>
          </CardHeader>
          <CardContent>
            <Accordion
              items={sections.map(([key, value]) => ({
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
      )}
    </div>
  );
}
