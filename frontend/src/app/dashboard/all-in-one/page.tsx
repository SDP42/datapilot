"use client";

import { useState, type ComponentType } from "react";
import { toast } from "sonner";
import {
  Zap,
  CheckCircle2,
  XCircle,
  Loader2,
  Circle,
  Database,
  ShieldCheck,
  LineChart,
  Cpu,
  Save,
} from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/ui/stat-card";
import {
  ingestDataset,
  analyzeQuality,
  analyzeEda,
  runModeling,
  trainAndSaveModel,
  ApiError,
  DatasetProfile,
  QualityReport,
  EdaReport,
  ModelingSpec,
  TrainAndSaveResponse,
} from "@/lib/api";

type StepStatus = "pending" | "running" | "completed" | "failed" | "skipped";

interface StepState {
  key: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  status: StepStatus;
  detail?: string;
}

const INITIAL_STEPS: StepState[] = [
  { key: "ingest", label: "Ingest & profile", icon: Database, status: "pending" },
  { key: "quality", label: "Quality analysis", icon: ShieldCheck, status: "pending" },
  { key: "eda", label: "Exploratory analysis", icon: LineChart, status: "pending" },
  { key: "modeling", label: "Modeling pipeline", icon: Cpu, status: "pending" },
  { key: "train", label: "Train & save model", icon: Save, status: "pending" },
];

function StepBadge({ status }: { status: StepStatus }) {
  if (status === "completed")
    return (
      <Badge variant="success">
        <CheckCircle2 className="h-3 w-3" /> Done
      </Badge>
    );
  if (status === "running")
    return (
      <Badge variant="primary">
        <Loader2 className="h-3 w-3 animate-spin" /> Running
      </Badge>
    );
  if (status === "failed")
    return (
      <Badge variant="danger">
        <XCircle className="h-3 w-3" /> Failed
      </Badge>
    );
  if (status === "skipped") return <Badge variant="default">Skipped</Badge>;
  return (
    <Badge variant="default">
      <Circle className="h-3 w-3" /> Pending
    </Badge>
  );
}

export default function AllInOnePage() {
  const [file, setFile] = useState<File | null>(null);
  const [objective, setObjective] = useState("");
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState<StepState[]>(INITIAL_STEPS);

  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [quality, setQuality] = useState<QualityReport | null>(null);
  const [eda, setEda] = useState<EdaReport | null>(null);
  const [modeling, setModeling] = useState<ModelingSpec | null>(null);
  const [trained, setTrained] = useState<TrainAndSaveResponse | null>(null);

  function updateStep(key: string, patch: Partial<StepState>) {
    setSteps((prev) => prev.map((s) => (s.key === key ? { ...s, ...patch } : s)));
  }

  async function runAll() {
    if (!file) return;
    setRunning(true);
    setSteps(INITIAL_STEPS);
    setProfile(null);
    setQuality(null);
    setEda(null);
    setModeling(null);
    setTrained(null);

    updateStep("ingest", { status: "running" });
    try {
      const ingestResult = await ingestDataset(file);
      setProfile(ingestResult.profile);
      updateStep("ingest", {
        status: "completed",
        detail: `${ingestResult.profile.n_rows} rows x ${ingestResult.profile.n_columns} columns`,
      });
    } catch (err) {
      updateStep("ingest", {
        status: "failed",
        detail: err instanceof ApiError ? err.message : "Ingestion failed",
      });
      toast.error("Ingestion failed — stopping");
      setRunning(false);
      return;
    }

    updateStep("quality", { status: "running" });
    try {
      const q = await analyzeQuality(file);
      setQuality(q);
      updateStep("quality", { status: "completed", detail: `${q.findings.length} finding(s)` });
    } catch (err) {
      updateStep("quality", {
        status: "failed",
        detail: err instanceof ApiError ? err.message : "Quality analysis failed",
      });
    }

    updateStep("eda", { status: "running" });
    try {
      const e = await analyzeEda(file);
      setEda(e);
      updateStep("eda", {
        status: "completed",
        detail: `${e.univariate.numeric.length} numeric, ${e.univariate.categorical.length} categorical`,
      });
    } catch (err) {
      updateStep("eda", {
        status: "failed",
        detail: err instanceof ApiError ? err.message : "EDA failed",
      });
    }

    if (!objective.trim()) {
      updateStep("modeling", { status: "skipped", detail: "no objective given" });
      updateStep("train", { status: "skipped", detail: "no objective given" });
      toast.success("Ingest, quality, and EDA complete — add an objective to also model it");
      setRunning(false);
      return;
    }

    updateStep("modeling", { status: "running" });
    try {
      const m = await runModeling(file, objective);
      setModeling(m);
      updateStep("modeling", {
        status: m.status === "completed" ? "completed" : "failed",
        detail:
          m.status === "completed"
            ? `${m.selection.selected_estimator} (${m.selection.selection_metric}=${m.selection.selected_score?.toFixed(3)})`
            : m.reason ?? "could not complete",
      });
    } catch (err) {
      updateStep("modeling", {
        status: "failed",
        detail: err instanceof ApiError ? err.message : "Modeling failed",
      });
    }

    updateStep("train", { status: "running" });
    try {
      const t = await trainAndSaveModel(file, objective);
      setTrained(t);
      updateStep("train", {
        status: t.model ? "completed" : "skipped",
        detail: t.model ? `saved as ${t.model.estimator_name}` : "nothing to persist",
      });
    } catch (err) {
      updateStep("train", {
        status: "failed",
        detail: err instanceof ApiError ? err.message : "Training failed",
      });
    }

    toast.success("Full pipeline complete");
    setRunning(false);
  }

  const anyResult = profile || quality || eda || modeling;

  return (
    <div>
      <PageHeader
        title="All in one go"
        description="One upload, the whole pipeline — ingest, quality, EDA, modeling, and a saved model — run back to back."
      />

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-primary-2" /> Run the full pipeline
          </CardTitle>
          <CardDescription>
            Objective is optional — without one, only ingest/quality/EDA run (modeling needs a
            target to predict).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={file} onFileSelect={setFile} />
          <div>
            <Label htmlFor="objective">Objective (optional, enables modeling)</Label>
            <Input
              id="objective"
              placeholder="e.g. predict churn"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
            />
          </div>
          <Button onClick={runAll} disabled={!file} loading={running}>
            <Zap className="h-4 w-4" /> Run everything
          </Button>
        </CardContent>
      </Card>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Pipeline progress</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {steps.map((s) => (
            <div
              key={s.key}
              className="flex flex-wrap items-center gap-3 rounded-xl border border-surface-border px-4 py-3"
            >
              <s.icon className="h-4 w-4 text-primary-2" />
              <span className="text-sm font-medium">{s.label}</span>
              {s.detail && <span className="text-xs text-muted">{s.detail}</span>}
              <div className="ml-auto">
                <StepBadge status={s.status} />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {anyResult && (
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {profile && (
            <StatCard
              icon={<Database className="h-5 w-5" />}
              label="Rows x Columns"
              value={`${profile.n_rows} x ${profile.n_columns}`}
            />
          )}
          {quality && (
            <StatCard
              icon={<ShieldCheck className="h-5 w-5" />}
              label="Quality findings"
              value={quality.findings.length}
            />
          )}
          {eda && (
            <StatCard
              icon={<LineChart className="h-5 w-5" />}
              label="Missing cells"
              value={`${eda.univariate.missingness.missing_percentage.toFixed(2)}%`}
            />
          )}
          {modeling && modeling.status === "completed" && (
            <StatCard
              icon={<Cpu className="h-5 w-5" />}
              label="Best model"
              value={modeling.selection.selected_estimator ?? "—"}
            />
          )}
          {trained?.model && (
            <StatCard
              icon={<Save className="h-5 w-5" />}
              label="Saved model"
              value={trained.model.model_id.replace("model-", "").slice(0, 8)}
            />
          )}
        </div>
      )}
    </div>
  );
}
