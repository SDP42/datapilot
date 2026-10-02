"use client";

import { useState } from "react";
import { toast } from "sonner";
import { ShieldCheck } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert } from "@/components/ui/alert";
import { analyzeQuality, ApiError, QualityReport } from "@/lib/api";

const SEVERITY_VARIANT: Record<string, "danger" | "warning" | "default"> = {
  high: "danger",
  medium: "warning",
  low: "default",
};

export default function QualityPage() {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<QualityReport | null>(null);

  async function handleRun() {
    if (!file) return;
    setLoading(true);
    try {
      const data = await analyzeQuality(file, target || undefined);
      setReport(data);
      toast.success(`${data.findings.length} finding(s) detected`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Quality analysis failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Data quality analysis"
        description="Phase 2 — deterministic, read-only detection of missing values, duplicates, outliers, skew, and imbalance."
      />

      <Card>
        <CardHeader>
          <CardTitle>Run analysis</CardTitle>
          <CardDescription>Detection only — nothing is modified.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={file} onFileSelect={setFile} />
          <div>
            <Label htmlFor="target">Target column (optional)</Label>
            <Input
              id="target"
              placeholder="e.g. churn"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            />
          </div>
          <Button onClick={handleRun} disabled={!file} loading={loading}>
            <ShieldCheck className="h-4 w-4" /> Analyze quality
          </Button>
        </CardContent>
      </Card>

      {report && (
        <div className="mt-8 space-y-4">
          {report.findings.length === 0 ? (
            <Alert variant="success" title="No findings">
              This dataset passed every deterministic quality check.
            </Alert>
          ) : (
            report.findings.map((finding, i) => (
              <Card key={i}>
                <CardContent className="flex items-start justify-between gap-4 p-5">
                  <div>
                    <div className="flex items-center gap-2">
                      <Badge variant={SEVERITY_VARIANT[finding.severity] ?? "default"}>
                        {finding.severity}
                      </Badge>
                      <span className="text-sm font-medium">{finding.finding_type}</span>
                      {finding.column && (
                        <span className="font-mono text-xs text-muted">· {finding.column}</span>
                      )}
                    </div>
                    <p className="mt-2 text-sm text-muted">{finding.message}</p>
                    {finding.suggested_action && (
                      <p className="mt-1 text-xs text-primary-2">
                        Suggested: {finding.suggested_action}
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      )}
    </div>
  );
}
