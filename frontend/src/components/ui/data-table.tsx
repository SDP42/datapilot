"use client";

import { cn } from "@/lib/utils";

export function DataTable({
  columns,
  rows,
  className,
}: {
  columns: string[];
  rows: Record<string, unknown>[];
  className?: string;
}) {
  if (rows.length === 0) {
    return <p className="py-8 text-center text-sm text-muted">No rows to display.</p>;
  }

  return (
    <div className={cn("overflow-x-auto rounded-xl border border-surface-border", className)}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-surface-border bg-surface-2/60">
            {columns.map((col) => (
              <th key={col} className="whitespace-nowrap px-4 py-3 text-left font-medium text-muted">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-surface-border/60 last:border-0 hover:bg-white/[0.02]">
              {columns.map((col) => (
                <td key={col} className="whitespace-nowrap px-4 py-3 font-mono text-xs text-foreground">
                  {formatCell(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
