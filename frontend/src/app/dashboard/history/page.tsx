"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { History as HistoryIcon, RefreshCw } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ActivityRecord, ApiError, getHistory } from "@/lib/api";

const KIND_VARIANT: Record<string, "primary" | "accent" | "success" | "warning" | "default"> = {
  ingest: "default",
  quality: "warning",
  eda: "accent",
  modeling: "primary",
  train: "success",
  predict: "success",
};

export default function HistoryPage() {
  const [records, setRecords] = useState<ActivityRecord[]>([]);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    try {
      setRecords(await getHistory(100));
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not load history");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function initialLoad() {
      setLoading(true);
      try {
        const data = await getHistory(100);
        if (!cancelled) setRecords(data);
      } catch (err) {
        if (!cancelled) {
          toast.error(err instanceof ApiError ? err.message : "Could not load history");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void initialLoad();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div>
      <PageHeader
        title="Run history"
        description="Every ingest, quality check, EDA, modeling run, training, and prediction — persisted to the database, not just this session."
        action={
          <Button variant="secondary" size="sm" onClick={load} loading={loading}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        }
      />

      <Card>
        <CardContent className="p-0">
          {records.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-16 text-center">
              <HistoryIcon className="h-8 w-8 text-muted" />
              <p className="text-sm text-muted">
                Nothing yet — run an ingest, quality check, EDA, modeling, or prediction and it
                will show up here.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-surface-border">
              {records.map((r) => (
                <div
                  key={r.activity_id}
                  className="flex flex-wrap items-center gap-3 px-6 py-4 hover:bg-white/[0.02]"
                >
                  <Badge variant={KIND_VARIANT[r.kind] ?? "default"}>{r.kind}</Badge>
                  <span className="font-mono text-sm">
                    {r.dataset_filename ?? r.dataset_id}
                  </span>
                  <span className="text-sm text-muted">{r.summary}</span>
                  <span className="ml-auto whitespace-nowrap text-xs text-muted">
                    {new Date(r.created_at).toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
