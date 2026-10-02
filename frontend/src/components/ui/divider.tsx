import { cn } from "@/lib/utils";

export function Divider({ className, label }: { className?: string; label?: string }) {
  if (!label) {
    return <div className={cn("h-px w-full bg-surface-border", className)} />;
  }
  return (
    <div className={cn("flex items-center gap-3", className)}>
      <div className="h-px flex-1 bg-surface-border" />
      <span className="text-xs text-muted">{label}</span>
      <div className="h-px flex-1 bg-surface-border" />
    </div>
  );
}
