"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Brain, Plus, X, Timer, Activity, Cpu, Trophy } from "lucide-react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileDropzone } from "@/components/ui/file-dropzone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert } from "@/components/ui/alert";
import { StatCard } from "@/components/ui/stat-card";
import { NetworkDiagram } from "@/components/dl/network-diagram";
import { trainDeepLearningModel, ApiError, DLModelingResult } from "@/lib/api";

export default function DeepLearningPage() {
  const [file, setFile] = useState<File | null>(null);
  const [objective, setObjective] = useState("");
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([64, 32]);
  const [epochs, setEpochs] = useState(100);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<DLModelingResult | null>(null);

  function updateLayer(index: number, value: number) {
    setHiddenLayers((layers) => layers.map((l, i) => (i === index ? value : l)));
  }

  function addLayer() {
    setHiddenLayers((layers) => [...layers, 16]);
  }

  function removeLayer(index: number) {
    setHiddenLayers((layers) => layers.filter((_, i) => i !== index));
  }

  async function handleTrain() {
    if (!file || !objective || hiddenLayers.length === 0) return;
    setRunning(true);
    try {
      const data = await trainDeepLearningModel(file, objective, hiddenLayers, epochs);
      setResult(data);
      if (data.status === "completed") {
        toast.success("MLP trained and evaluated");
      } else {
        toast.warning(data.reason ?? "Training could not complete");
      }
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Training failed");
    } finally {
      setRunning(false);
    }
  }

  const lossChartData = result?.training?.loss_history.map((loss, i) => ({ epoch: i + 1, loss })) ?? [];

  return (
    <div>
      <PageHeader
        title="Deep learning"
        description="A real PyTorch MLP — build, train, and evaluate a feed-forward neural network, architecture visualized."
      />

      <Card>
        <CardHeader>
          <CardTitle>Configure & train</CardTitle>
          <CardDescription>
            Regression and classification only — PyTorch is required on the backend; a missing
            install reports a clear unavailable status rather than crashing.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
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

          <div>
            <Label>Hidden layers</Label>
            <div className="mt-2 space-y-2">
              {hiddenLayers.map((size, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="w-20 shrink-0 text-xs text-muted">Layer {i + 1}</span>
                  <Input
                    type="number"
                    min={1}
                    value={size}
                    onChange={(e) => updateLayer(i, Math.max(1, Number(e.target.value)))}
                    className="w-28"
                  />
                  <span className="text-xs text-muted">units</span>
                  <button
                    onClick={() => removeLayer(i)}
                    disabled={hiddenLayers.length <= 1}
                    className="ml-auto rounded-lg p-1.5 text-muted hover:bg-white/5 hover:text-danger disabled:opacity-30"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
              <Button variant="secondary" size="sm" onClick={addLayer}>
                <Plus className="h-3.5 w-3.5" /> Add hidden layer
              </Button>
            </div>
          </div>

          <NetworkDiagram hiddenLayerSizes={hiddenLayers} />

          <div className="max-w-xs">
            <Label htmlFor="epochs">Epochs</Label>
            <Input
              id="epochs"
              type="number"
              min={1}
              max={1000}
              value={epochs}
              onChange={(e) => setEpochs(Math.max(1, Number(e.target.value)))}
            />
          </div>

          <Button
            onClick={handleTrain}
            disabled={!file || !objective || hiddenLayers.length === 0}
            loading={running}
          >
            <Brain className="h-4 w-4" /> Train the MLP
          </Button>
        </CardContent>
      </Card>

      {result && result.status === "completed" && (
        <div className="mt-8 space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              icon={<Cpu className="h-5 w-5" />}
              label="Device"
              value={result.training?.device_used ?? "—"}
            />
            <StatCard
              icon={<Activity className="h-5 w-5" />}
              label="Epochs completed"
              value={result.training?.epochs_completed ?? 0}
            />
            <StatCard
              icon={<Timer className="h-5 w-5" />}
              label="Final training loss"
              value={result.training?.final_loss?.toFixed(4) ?? "—"}
            />
            <StatCard
              icon={<Trophy className="h-5 w-5" />}
              label={result.evaluation?.primary_metric ?? "metric"}
              value={
                result.evaluation?.primary_metric
                  ? (result.evaluation.metrics[result.evaluation.primary_metric]?.toFixed(4) ?? "—")
                  : "—"
              }
            />
          </div>

          {lossChartData.length > 1 && (
            <Card>
              <CardHeader>
                <CardTitle>Training loss curve</CardTitle>
                <CardDescription>Mean loss per epoch — real values, not a placeholder.</CardDescription>
              </CardHeader>
              <CardContent className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: "var(--muted)" }} />
                    <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} />
                    <Tooltip
                      contentStyle={{
                        background: "var(--surface-2)",
                        border: "1px solid var(--surface-border)",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Line type="monotone" dataKey="loss" stroke="var(--primary)" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Evaluation metrics</CardTitle>
              <CardDescription>
                On held-out evaluation rows the model never trained on.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {result.evaluation &&
                Object.entries(result.evaluation.metrics).map(([k, v]) => (
                  <Badge key={k} variant="success">
                    {k}: {v.toFixed(4)}
                  </Badge>
                ))}
            </CardContent>
          </Card>
        </div>
      )}

      {result && result.status !== "completed" && (
        <Alert variant="warning" title={`Status: ${result.status}`} className="mt-8">
          {result.reason ?? "Training could not complete for this dataset."}
        </Alert>
      )}
    </div>
  );
}
