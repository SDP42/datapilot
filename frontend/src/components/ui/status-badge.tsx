import type { ComponentType } from "react";
import { Badge } from "./badge";
import { CheckCircle2, Clock, Loader2, XCircle, HelpCircle } from "lucide-react";

type Variant = "success" | "warning" | "danger" | "primary" | "default";
type IconComponent = ComponentType<{ className?: string }>;

const MAP: Record<string, { variant: Variant; icon: IconComponent; label: string }> = {
  completed: { variant: "success", icon: CheckCircle2, label: "Completed" },
  running: { variant: "primary", icon: Loader2, label: "Running" },
  pending: { variant: "warning", icon: Clock, label: "Pending" },
  failed: { variant: "danger", icon: XCircle, label: "Failed" },
  unavailable: { variant: "default", icon: HelpCircle, label: "Unavailable" },
};

export function StatusBadge({ status }: { status: string }) {
  const entry: { variant: Variant; icon: IconComponent; label: string } = MAP[status] ?? {
    variant: "default",
    icon: HelpCircle,
    label: status,
  };
  const Icon = entry.icon;
  return (
    <Badge variant={entry.variant}>
      <Icon className={`h-3 w-3 ${status === "running" ? "animate-spin" : ""}`} />
      {entry.label}
    </Badge>
  );
}
