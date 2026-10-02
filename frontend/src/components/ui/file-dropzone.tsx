"use client";

import { useCallback, useState, DragEvent } from "react";
import { UploadCloud, FileSpreadsheet, X } from "lucide-react";
import { cn, formatBytes } from "@/lib/utils";

export function FileDropzone({
  onFileSelect,
  file,
  accept = ".csv",
  className,
}: {
  onFileSelect: (file: File | null) => void;
  file: File | null;
  accept?: string;
  className?: string;
}) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const dropped = e.dataTransfer.files?.[0];
      if (dropped) onFileSelect(dropped);
    },
    [onFileSelect],
  );

  if (file) {
    return (
      <div
        className={cn(
          "flex items-center justify-between rounded-xl border border-surface-border bg-surface-2/60 px-4 py-3.5",
          className,
        )}
      >
        <div className="flex items-center gap-3 overflow-hidden">
          <FileSpreadsheet className="h-5 w-5 shrink-0 text-primary-2" />
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">{file.name}</p>
            <p className="text-xs text-muted">{formatBytes(file.size)}</p>
          </div>
        </div>
        <button
          onClick={() => onFileSelect(null)}
          className="shrink-0 rounded-lg p-1.5 text-muted hover:bg-white/5 hover:text-danger"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    );
  }

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors",
        isDragging ? "border-primary bg-primary/5" : "border-surface-border bg-surface-2/30 hover:border-primary/50",
        className,
      )}
    >
      <UploadCloud className={cn("h-8 w-8", isDragging ? "text-primary" : "text-muted")} />
      <p className="text-sm font-medium">Drop a CSV file here, or click to browse</p>
      <p className="text-xs text-muted">Comma-separated values, up to 50&nbsp;MB</p>
      <input
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => onFileSelect(e.target.files?.[0] ?? null)}
      />
    </label>
  );
}
