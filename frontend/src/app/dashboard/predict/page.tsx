"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Brain, Download, RefreshCw, Sparkles, Target } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert } from "@/components/ui/alert";
import { DataTable } from "@/components/ui/data-table";
import { cn } from "@/lib/utils";
import {
  ApiError,
  PersistedModelMetadata,
  PredictionResult,
  listTrainedModels,
  predictWithModel,
  trainAndSaveModel,
} from "@/lib/api";

export default function PredictPage() {
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [objective, setObjective] = useState("");
  const [training, setTraining] = useState(false);
  const [trainStatus, setTrainStatus] = useState<string | null>(null);

  const [models, setModels] = useState<PersistedModelMetadata[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  async function refreshModels() {
    setLoadingModels(true);
    try {
      const data = await listTrainedModels();
      setModels(data);
      if (!selectedModel && data.length > 0) setSelectedModel(data[0].model_id);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not load saved models");
    } finally {
      setLoadingModels(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function initialLoad() {
      setLoadingModels(true);
      try {
        const data = await listTrainedModels();
        if (cancelled) return;
        setModels(data);
        if (data.length > 0) setSelectedModel(data[0].model_id);
      } catch (err) {
        if (!cancelled) {
          toast.error(err instanceof ApiError ? err.message : "Could not load saved models");
        }
      } finally {
        if (!cancelled) setLoadingModels(false);
      }
    }
    void initialLoad();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleTrain() {
    if (!trainFile || !objective) return;
    setTraining(true);
    setTrainStatus(null);
    try {
      const response = await trainAndSaveModel(trainFile, objective);
      setTrainStatus(response.status);
      if (response.model) {
        toast.success(`Model saved: ${response.model.estimator_name}`);
        setSelectedModel(response.model.model_id);
        await refreshModels();
      } else {
        toast.warning(response.reason ?? "Training completed without a model to save");
      }
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Training failed");
    } finally {
      setTraining(false);
    }
  }

  async function handlePredict() {
    if (!predictFile || !selectedModel) return;
    setPredicting(true);
    setResult(null);
    try {
      const data = await predictWithModel(selectedModel, predictFile);
      setResult(data);
      if (data.missing_columns.length === 0) toast.success(`${data.row_count} prediction(s) generated`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  }

  const activeModel = models.find((m) => m.model_id === selectedModel) ?? null;

  const predictionRows =
    result && result.predictions.length > 0
      ? result.predictions.map((p, i) => ({
          row: i + 1,
          prediction: p,
          ...(result.probabilities ? { probability: result.probabilities[i] } : {}),
        }))
      : [];

  function downloadPredictionsCsv() {
    if (!result || result.predictions.length === 0) return;
    const hasProba = Boolean(result.probabilities);
    const header = hasProba ? "row,prediction,probability" : "row,prediction";
    const lines = result.predictions.map((p, i) =>
      hasProba ? `${i + 1},${p},${result.probabilities![i]}` : `${i + 1},${p}`,
    );
    const csv = [header, ...lines].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `predictions_${result.model_id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      <PageHeader
        title="Train & predict"
        description="Phase 7.6 — persist the model Phase 7 selects, then run it against new, unseen rows."
      />

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary-2" /> 1. Train & save a model
          </CardTitle>
          <CardDescription>
            Runs the full modeling pipeline, then keeps the winning estimator so it can serve
            predictions later — not just a one-off metrics report.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={trainFile} onFileSelect={setTrainFile} />
          <div>
            <Label htmlFor="objective">Objective</Label>
            <Input
              id="objective"
              placeholder="e.g. predict churn"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
            />
          </div>
          <Button onClick={handleTrain} disabled={!trainFile || !objective} loading={training}>
            <Sparkles className="h-4 w-4" /> Train & save model
          </Button>
          {trainStatus && trainStatus !== "completed" && (
            <Alert variant="warning" title={`Status: ${trainStatus}`}>
              The pipeline could not select a model to persist for this dataset/objective.
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>2. Saved models</span>
            <Button variant="ghost" size="sm" onClick={refreshModels} loading={loadingModels}>
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </Button>
          </CardTitle>
          <CardDescription>Pick which persisted model to run predictions with.</CardDescription>
        </CardHeader>
        <CardContent>
          {models.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted">
              No saved models yet — train one above.
            </p>
          ) : (
            <div className="space-y-2">
              {models.map((m) => (
                <button
                  key={m.model_id}
                  type="button"
                  onClick={() => setSelectedModel(m.model_id)}
                  className={cn(
                    "flex w-full flex-wrap items-center gap-3 rounded-xl border px-4 py-3 text-left transition-colors",
                    m.model_id === selectedModel
                      ? "border-primary/50 bg-primary/10"
                      : "border-surface-border bg-surface-2/30 hover:bg-white/5",
                  )}
                >
                  <Badge variant={m.category === "classification" ? "accent" : "primary"}>
                    {m.category}
                  </Badge>
                  <span className="font-mono text-sm">{m.estimator_name}</span>
                  <span className="text-xs text-muted">target: {m.target_column ?? "—"}</span>
                  {m.selected_score !== null && m.selected_score !== undefined && (
                    <span className="text-xs text-muted">
                      {m.selection_metric}: {m.selected_score.toFixed(4)}
                    </span>
                  )}
                  <span className="ml-auto text-xs text-muted">
                    {new Date(m.created_at).toLocaleString()}
                  </span>
                </button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-4 w-4 text-primary-2" /> 3. Predict on new data
          </CardTitle>
          <CardDescription>
            {activeModel
              ? `Requires columns: ${activeModel.feature_cols.join(", ")}`
              : "Select a saved model above first."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDropzone file={predictFile} onFileSelect={setPredictFile} />
          <Button
            onClick={handlePredict}
            disabled={!predictFile || !selectedModel}
            loading={predicting}
          >
            <Target className="h-4 w-4" /> Run prediction
          </Button>

          {result && result.missing_columns.length > 0 && (
            <Alert variant="danger" title="Missing required columns">
              {result.notes.join(" ")}
            </Alert>
          )}

          {result && predictionRows.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted">{result.row_count} row(s) predicted</p>
                <Button variant="secondary" size="sm" onClick={downloadPredictionsCsv}>
                  <Download className="h-3.5 w-3.5" /> Download CSV
                </Button>
              </div>
              <DataTable
                columns={Object.keys(predictionRows[0])}
                rows={predictionRows as unknown as Record<string, unknown>[]}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
