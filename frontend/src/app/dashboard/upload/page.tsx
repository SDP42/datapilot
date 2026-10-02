"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Database } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/ui/stat-card";
import { Chip } from "@/components/ui/chip";
import { ingestDataset, ApiError, DatasetProfile, DatasetReference } from "@/lib/api";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ reference: DatasetReference; profile: DatasetProfile } | null>(null);

  async function handleIngest() {
    if (!file) return;
    setLoading(true);
    try {
      const data = await ingestDataset(file);
      setResult(data);
      toast.success("Dataset ingested and profiled");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Ingestion failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Ingest a dataset"
        description="Phase 1 — immutable raw storage and deterministic profiling."
      />

      <Card>
        <CardHeader>
          <CardTitle>Upload CSV</CardTitle>
          <CardDescription>The file is stored read-only and never modified in place.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={file} onFileSelect={setFile} />
          <Button onClick={handleIngest} disabled={!file} loading={loading}>
            <Database className="h-4 w-4" /> Ingest dataset
          </Button>
        </CardContent>
      </Card>

      {result && (
        <div className="mt-8 space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Rows" value={result.profile.n_rows.toLocaleString()} />
            <StatCard label="Columns" value={result.profile.n_columns} />
            <StatCard label="Duplicate rows" value={result.profile.duplicate_row_count} />
            <StatCard label="Numeric columns" value={result.profile.numeric_columns.length} />
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Dataset reference</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              <Chip>id: {result.reference.dataset_id}</Chip>
              <Chip>sha256: {result.reference.sha256.slice(0, 16)}…</Chip>
              <Chip>{result.reference.original_filename}</Chip>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Columns</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {result.profile.column_names.map((col) => (
                <Chip key={col}>{col}</Chip>
              ))}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
