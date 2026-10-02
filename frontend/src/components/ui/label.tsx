import { LabelHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Label({ className, ...props }: LabelHTMLAttributes<HTMLLabelElement>) {
  return (
    <label
      className={cn("mb-2 block text-xs font-medium uppercase tracking-wide text-muted", className)}
      {...props}
    />
  );
}
