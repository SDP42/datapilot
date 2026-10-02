"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function ProgressBar({
  value,
  className,
  label,
}: {
  value: number;
  className?: string;
  label?: string;
}) {
  const clamped = Math.min(100, Math.max(0, value));
  return (
    <div className={cn("space-y-1.5", className)}>
      {label && (
        <div className="flex justify-between text-xs text-muted">
          <span>{label}</span>
          <span>{clamped.toFixed(0)}%</span>
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-primary via-primary-2 to-accent"
          initial={{ width: 0 }}
          animate={{ width: `${clamped}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}
