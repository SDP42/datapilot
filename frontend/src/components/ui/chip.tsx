import { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Chip({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-lg border border-surface-border bg-surface-2 px-2 py-1 font-mono text-xs text-muted",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
