import {
  UploadCloud,
  ShieldCheck,
  LineChart,
  Cpu,
  Layers,
  Target,
  Database,
  SlidersHorizontal,
  Boxes,
  type LucideIcon,
} from "lucide-react";
import type { ActivityRecord } from "@/lib/api";

export const ACTIVITY_META: Record<
  ActivityRecord["kind"],
  { label: string; icon: LucideIcon; color: string }
> = {
  ingest: { label: "Ingested a dataset", icon: UploadCloud, color: "text-primary-2" },
  quality: { label: "Ran quality analysis", icon: ShieldCheck, color: "text-warning" },
  eda: { label: "Ran EDA", icon: LineChart, color: "text-accent" },
  modeling: { label: "Ran the modeling pipeline", icon: Cpu, color: "text-primary-2" },
  search: { label: "Ran a full model search", icon: Layers, color: "text-accent-2" },
  tune: { label: "Deep-tuned a model", icon: SlidersHorizontal, color: "text-warning" },
  train: { label: "Trained & saved a model", icon: Target, color: "text-success" },
  predict: { label: "Ran a prediction", icon: Database, color: "text-primary-2" },
  cluster: { label: "Ran clustering", icon: Boxes, color: "text-accent-2" },
};
