"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { ListChecks, RefreshCw } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { StatusBadge } from "@/components/ui/status-badge";
import { Chip } from "@/components/ui/chip";
import { submitModelingJob, getJob, ApiError, JobRecord } from "@/lib/api";
import { formatDate } from "@/lib/utils";

export default function JobsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [objective, setObjective] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    pollRef.current = setInterval(async () => {
      setJobs((current) => {
        current
          .filter((j) => j.status === "pending" || j.status === "running")
          .forEach(async (j) => {
            try {
              const updated = await getJob(j.job_id);
              setJobs((prev) => prev.map((p) => (p.job_id === updated.job_id ? updated : p)));
            } catch {
              // transient poll failure, next tick retries
            }
          });
        return current;
      });
    }, 2500);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function handleSubmit() {
    if (!file || !objective) return;
    setSubmitting(true);
    try {
      const job = await submitModelingJob(file, objective);
      setJobs((prev) => [job, ...prev]);
      toast.success("Job submitted");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not submit job");
    } finally {
      setSubmitting(false);
    }
  }

  async function refreshJob(jobId: string) {
    const updated = await getJob(jobId);
    setJobs((prev) => prev.map((p) => (p.job_id === updated.job_id ? updated : p)));
  }

  return (
    <div>
      <PageHeader
        title="Background jobs"
        description="Phase 13.3 — submit a modeling run asynchronously and poll it to completion."
      />

      <Card>
        <CardHeader>
          <CardTitle>Submit a job</CardTitle>
          <CardDescription>Returns a job id immediately; runs in the background.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={file} onFileSelect={setFile} />
          <div>
            <Label htmlFor="job-objective">Objective</Label>
            <Input
              id="job-objective"
              placeholder="e.g. predict churn"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
            />
          </div>
          <Button onClick={handleSubmit} disabled={!file || !objective} loading={submitting}>
            <ListChecks className="h-4 w-4" /> Submit job
          </Button>
        </CardContent>
      </Card>

      {jobs.length > 0 && (
        <div className="mt-8 space-y-3">
          {jobs.map((job) => (
            <Card key={job.job_id}>
              <CardContent className="flex flex-wrap items-center justify-between gap-3 p-5">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <Chip>{job.job_id.slice(0, 8)}</Chip>
                    <span className="text-sm font-medium">{job.kind}</span>
                  </div>
                  <p className="mt-1 text-xs text-muted">
                    Submitted {formatDate(job.created_at)} · updated {formatDate(job.updated_at)}
                  </p>
                  {job.error && <p className="mt-1 text-xs text-danger">{job.error}</p>}
                  {job.result && (
                    <p className="mt-1 text-xs text-success">
                      Selected: {job.result.selection.selected_family ?? "n/a"}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <StatusBadge status={job.status} />
                  <button
                    onClick={() => refreshJob(job.job_id)}
                    className="rounded-lg p-2 text-muted hover:bg-white/5 hover:text-foreground"
                  >
                    <RefreshCw className="h-4 w-4" />
                  </button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
